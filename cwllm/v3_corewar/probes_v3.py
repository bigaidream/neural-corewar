"""Declarative inference-only GPU probes; validation imports no torch."""
from pathlib import Path
import numpy as np
from ..artifacts import read_record, write_record, sha256_digest, encode_canonical
from ..grammar import BudgetSpec, PatchValidationError
from ..merge import bf16_to_f32
from ..scoring import KeywordPair, score_dialogue

NAMES = {'logit_lens_neurons', 'contrast_activations', 'dry_run_patch', 'scan_weight_diff'}


def validate_probe(name, args, protocol_version='v3_corewar'):
    if name not in NAMES or not isinstance(args, dict):
        raise ValueError('unknown probe or invalid arguments')
    if name == 'scan_weight_diff':
        if set(args) - {'top_k'} or type(args.get('top_k', 16)) is not int or not 1 <= args.get('top_k', 16) <= 32:
            raise ValueError('scan top_k outside 1..32')
        return True
    if name == 'logit_lens_neurons' and 'columns' in args:
        if set(args) - {'words', 'layers', 'top_k', 'columns'}:
            raise ValueError('unsupported peek argument')
        cs = args['columns']
        words = args.get('words', [])
        k = args.get('top_k', 16)
        if not isinstance(cs, list) or not 1 <= len(cs) <= 16 or type(k) is not int or not 1 <= k <= 16:
            raise ValueError('peek bound')
        seen = set()
        for c in cs:
            if (not isinstance(c, dict) or set(c) != {'layer', 'column'} or
                    type(c['layer']) is not int or not 8 <= c['layer'] <= 19 or
                    type(c['column']) is not int or not 0 <= c['column'] < 3584):
                raise ValueError('peek coordinate outside arena')
            key = (c['layer'], c['column'])
            if key in seen:
                raise ValueError('duplicate peek coordinate')
            seen.add(key)
        if not isinstance(words, list) or len(words) > 4 or any(not isinstance(w, str) or not w or len(w) > 64 for w in words):
            raise ValueError('invalid peek words')
        if 'layers' in args and (not isinstance(args['layers'], list) or any(type(l) is not int or not 8 <= l <= 19 for l in args['layers'])):
            raise ValueError('invalid peek layers')
        return True
    if name == 'dry_run_patch':
        from .spec import validate_v3_patch
        validate_v3_patch(args.get('candidate_patch'))
    allowed = {
        'logit_lens_neurons': {'words', 'layers', 'top_k'},
        'contrast_activations': {'pos_prompts', 'neg_prompts', 'layers', 'top_k'},
        'dry_run_patch': {'candidate_patch'},
        'scan_weight_diff': {'top_k'},
    }[name]
    if name == 'contrast_activations' and protocol_version == 'v3_corewar':
        allowed = allowed | {'site'}
        if args.get('site', 'reply_content') not in ('reply_content', 'prompt_last_token'):
            raise ValueError('invalid activation site')
    if set(args) - allowed:
        raise ValueError('unsupported probe argument')
    if name in ('logit_lens_neurons', 'contrast_activations'):
        layers = args.get('layers', list(range(8, 20)))
        if not isinstance(layers, list) or not layers or len(set(layers)) != len(layers) or any(type(l) is not int or not 8 <= l <= 19 for l in layers):
            raise ValueError('probe layers outside writable arena')
    k = args.get('top_k', 16)
    if type(k) is not int or not 1 <= k <= 16:
        raise ValueError('probe top_k outside bound')
    if name == 'logit_lens_neurons':
        words = args.get('words')
        if not isinstance(words, list) or not 1 <= len(words) <= 4 or any(not isinstance(w, str) or not w or len(w) > 64 for w in words):
            raise ValueError('invalid target words')
    if name == 'contrast_activations':
        for side in ('pos_prompts', 'neg_prompts'):
            prompts = args.get(side)
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 6 or any(not isinstance(x, str) or not x.strip() or len(x) > 512 for x in prompts):
                raise ValueError('contrast prompts outside bound')
    if name == 'dry_run_patch' and (set(args) != {'candidate_patch'} or not isinstance(args['candidate_patch'], dict)):
        raise ValueError('dry run requires one candidate patch')
    return True


def commit_probe(directory, payload):
    expected = {'slot_id', 'turn', 'blind', 'state', 'own', 'rival', 'budget', 'condition', 'name', 'args'}
    formal = payload.get('protocol_version') == 'v3_corewar'
    if formal:
        expected |= {'protocol_version', 'decoding'}
    if set(payload) != expected:
        raise ValueError('invalid probe context')
    validate_probe(payload['name'], payload['args'], payload.get('protocol_version', 'v1'))
    if payload['blind'] or type(payload['turn']) is not int or not 0 <= payload['turn'] <= (4 if formal else 2):
        raise ValueError('slot probe allowance exhausted')
    if payload['name'] == 'scan_weight_diff' and payload['condition'] != 'H':
        raise ValueError('weight scan requires H')
    if not isinstance(payload['slot_id'], str) or not payload['slot_id']:
        raise ValueError('missing slot identity')
    path = Path(directory) / sha256_digest(payload['slot_id']) / f"turn-{payload['turn']}.json"
    if formal:
        import fcntl
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {'payload_sha256': sha256_digest(payload), 'name': payload['name']}
        with (path.parent / 'allowance.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists() and payload['name'] == 'dry_run_patch':
                used = sum(read_record(p).get('name') == 'dry_run_patch' for p in path.parent.glob('turn-*.json'))
                if used >= 3:
                    raise ValueError('durable dry_run_patch allowance exhausted')
            write_record(path, receipt)
    else:
        write_record(path, {'payload_sha256': sha256_digest(payload)})
    return path


def reply_content_positions(prefix_length, reply_length):
    if prefix_length < 0 or reply_length <= 0:
        raise ValueError('empty reply support')
    return list(range(prefix_length + (3 if reply_length >= 4 else 0), prefix_length + reply_length))


class ProbeSandbox:
    def __init__(self, session, worker, public_inputs, directory, protocol_version='v3_corewar', decoding=None):
        self.session, self.worker, self.public = session, worker, public_inputs
        self.directory = Path(directory)
        self.protocol_version, self.decoding = protocol_version, decoding

    def dispatch(self, payload):
        if payload.get('protocol_version', 'v1') != self.protocol_version:
            raise ValueError('probe protocol mismatch')
        if self.protocol_version == 'v3_corewar' and payload.get('decoding') != self.decoding:
            raise ValueError('probe decoding mismatch')
        commit_probe(self.directory, payload)
        if payload['own'] not in self.public['facts'] or payload['rival'] not in self.public['facts']:
            raise ValueError('unknown keyword')
        import torch
        budget = BudgetSpec(**payload['budget'])
        with torch.inference_mode(), self.session.patched(state=payload['state'], budget=budget):
            name, args = payload['name'], payload['args']
            if name == 'logit_lens_neurons':
                result = self.logit_lens(**args)
            elif name == 'contrast_activations':
                result = self.contrast(**({'site': 'reply_content', **args} if self.protocol_version == 'v3_corewar' else args))
            elif name == 'scan_weight_diff':
                result = self.scan(**args)
            else:
                result = self.dry_run(payload, args['candidate_patch'])
        public = result.pop('_public', result.copy())
        if len(encode_canonical(public)) > 12288:
            raise ValueError('probe summary exceeds bound')
        return {'public': public, 'full': result, 'payload_sha256': sha256_digest(payload), 'inference_only': True}

    def logit_lens(self, words=None, layers=None, top_k=16, columns=None):
        if columns is not None:
            return self.peek_columns(columns, words or [])
        import torch
        layers = layers or list(range(8, 20))
        s = self.session
        head = s.model.get_output_embeddings().weight
        ids = [s.token_id(w) for w in words]
        target = head[ids].float()
        candidates = []
        for layer in layers:
            projections = s.model.model.norm(s.weights[layer].T).float() @ target.T
            strength = projections.abs().amax(-1)
            indices = torch.argsort(strength, descending=True, stable=True)[:top_k].tolist()
            for c in indices:
                candidates.append((float(strength[c]), layer, c, projections[c].tolist()))
        candidates.sort(key=lambda r: (-r[0], r[1], r[2]))
        chosen = candidates[:top_k]
        cols = torch.stack([s.weights[l][:, c] for _, l, c, _ in chosen])
        logits = s.model.model.norm(cols).float() @ head.float().T
        companions = torch.topk(logits, 5, dim=-1).indices.tolist()
        rows = []
        for i, (_, l, c, p) in enumerate(chosen):
            promoted = []
            for token in companions[i]:
                value = s.tokenizer.decode([token], skip_special_tokens=False)
                promoted.append({'token_id': token, 'text': value if len(value) <= 80 else '[long token omitted]', 'logit': round(float(logits[i, token]), 6)})
            rows.append({'layer': l, 'column': c, 'target_projections': [round(float(x), 6) for x in p], 'companions': promoted})
        return {'words': words, 'rows': rows, 'ranking': 'absolute signed logit-lens projection; not causal evidence'}

    def peek_columns(self, columns, words):
        import torch
        s = self.session
        head = s.model.get_output_embeddings().weight
        ids = [s.token_id(w) for w in words]
        vectors = torch.stack([s.weights[c['layer']][:, c['column']] for c in columns])
        logits = s.model.model.norm(vectors).float() @ head.float().T
        top = torch.topk(logits, 5, dim=-1).indices.tolist()
        rows = []
        for i, c in enumerate(columns):
            companions = [{'token_id': t, 'text': s.tokenizer.decode([t], skip_special_tokens=False)[:80], 'logit': round(float(logits[i, t]), 6)} for t in top[i]]
            rows.append({**c, 'target_projections': [round(float(logits[i, t]), 6) for t in ids], 'companions': companions})
        return {'words': words, 'rows': rows, 'ranking': 'explicit coordinates on live current weights; not causal evidence'}

    def contrast(self, pos_prompts, neg_prompts, layers=None, top_k=16, site='prompt_last_token'):
        if site == 'reply_content':
            return self.reply_contrast(pos_prompts, neg_prompts, layers, top_k)
        s = self.session
        device = next(s.model.parameters()).device
        layers = layers or list(range(8, 20))
        captured, handles = {}, []

        def hook(layer):
            def capture(module, args):
                captured[layer] = args[0][:, -1, :].detach().float().cpu().numpy()
            return capture

        try:
            for layer in layers:
                handles.append(s.layers[layer].mlp.down_proj.register_forward_pre_hook(hook(layer)))
            strings = [s.tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in pos_prompts + neg_prompts]
            inputs = s.tokenizer(strings, padding=True, add_special_tokens=False, return_tensors='pt').to(device)
            s.model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        rows, n = [], len(pos_prompts)
        for l, values in captured.items():
            pos = values[:n].astype(np.float64).mean(0)
            neg = values[n:].astype(np.float64).mean(0)
            gap = pos - neg
            for c in np.argsort(-np.abs(gap), kind='stable')[:top_k]:
                rows.append({'layer': l, 'column': int(c), 'pos_mean': round(float(pos[c]), 6), 'neg_mean': round(float(neg[c]), 6), 'gap': round(float(gap[c]), 6)})
        rows.sort(key=lambda r: (-abs(r['gap']), r['layer'], r['column']))
        return {'rows': rows[:top_k], 'positive_prompts': n, 'negative_prompts': len(neg_prompts), 'site': 'final prompt-token down_proj input'}

    def reply_contrast(self, pos_prompts, neg_prompts, layers=None, top_k=16):
        import torch
        s = self.session
        device = next(s.model.parameters()).device
        layers = layers or list(range(8, 20))
        captured = {l: [] for l in layers}
        conversations = [[{'role': 'user', 'content': p}] for p in pos_prompts + neg_prompts]
        with torch.inference_mode(), s.patched(state=[]):
            replies = self.worker.generate(conversations, decoding={'do_sample': False, 'max_new_tokens': 32})
            for conversation, reply in zip(conversations, replies):
                text = s.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prefix = s.tokenizer.encode(text, add_special_tokens=False)
                stops = s.model.generation_config.eos_token_id
                stops = set(stops if isinstance(stops, list) else [stops])
                tokens = [t for t in reply['output_ids'] if t not in stops and t != s.tokenizer.pad_token_id]
                if not tokens:
                    raise ValueError('empty W0 reply activation support')
                selected = reply_content_positions(len(prefix), len(tokens))
                handles = []

                def hook(layer):
                    def capture(module, args):
                        captured[layer].append(args[0][0, selected, :].float().mean(0).cpu().numpy())
                    return capture

                try:
                    for l in layers:
                        handles.append(s.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)))
                    ids = torch.tensor([prefix + tokens], device=device)
                    s.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
                finally:
                    for h in handles:
                        h.remove()
        rows, n = [], len(pos_prompts)
        for l, values in captured.items():
            values = np.asarray(values, dtype=np.float64)
            pos = values[:n].mean(0)
            neg = values[n:].mean(0)
            gap = pos - neg
            for c in np.argsort(-np.abs(gap), kind='stable')[:top_k]:
                rows.append({'layer': l, 'column': int(c), 'pos_mean': round(float(pos[c]), 6), 'neg_mean': round(float(neg[c]), 6), 'gap': round(float(gap[c]), 6)})
        rows.sort(key=lambda r: (-abs(r['gap']), r['layer'], r['column']))
        return {
            'rows': rows[:top_k],
            'positive_prompts': n,
            'negative_prompts': len(neg_prompts),
            'site': 'reply_content',
            'reference': 'pristine W0 greedy max32; skip3 greeting tokens; short replies use all',
            'reply_tokens': [r['output_tokens'] for r in replies],
        }

    def scan(self, top_k=16):
        rows = []
        for l, bits in self.session.current.items():
            columns = np.flatnonzero(np.any(bits != self.session.base[l], axis=0))
            if not len(columns):
                continue
            delta = bf16_to_f32(bits[:, columns]).astype(np.float64) - bf16_to_f32(self.session.base[l][:, columns]).astype(np.float64)
            for c, norm in zip(columns, np.linalg.norm(delta, axis=0)):
                if norm:
                    rows.append({'layer': l, 'column': int(c), 'drift_l2': round(float(norm), 8)})
        rows.sort(key=lambda r: (-r['drift_l2'], r['layer'], r['column']))
        return {'rows': rows[:top_k], 'nonzero_columns': len(rows), 'reference': 'pristine W0'}

    def dry_run(self, context, patch):
        s, w = self.session, self.worker
        budget = BudgetSpec(**context['budget'])
        try:
            s.compile(patch, budget)
        except PatchValidationError as exc:
            return {'valid': False, 'reason': str(exc)}
        own, rival = context['own'], context['rival']
        facts = self.public['facts'][own]
        payload = {
            'state': context['state'],
            'budget': context['budget'],
            'patch_a': patch,
            'passages': self.public['ppl_practice'],
            'word': own,
            'forms': facts.get('forms', []),
            'must_say': facts['prompts'][:2],
            'bank': self.public['practice'],
        }
        if self.decoding is not None:
            payload['decoding'] = self.decoding
        ids = [s.token_id(' ' + own), s.token_id(' ' + rival)]
        before = w.perplexity(self.public['ppl_practice'], ids)
        metrics = w.check_health(payload)
        with w._context(payload):
            after = w.perplexity(self.public['ppl_practice'], ids)
        pair = KeywordPair(own, rival, tuple(facts.get('forms', [])), tuple(self.public['facts'][rival].get('forms', [])))
        scores = [score_dialogue(d['messages'], pair) for d in metrics['dialogues']]
        public = {
            'valid': True,
            'practice_score': sum(r['score_A'] for r in scores) / len(scores),
            'dialogues': len(scores),
            'wins': sum(r['winner'] == 'A' for r in scores),
            'draws': sum(r['winner'] == 'draw' for r in scores),
            'losses': sum(r['winner'] == 'B' for r in scores),
            'delta_ppl_vs_pristine': metrics['fluency_ppl_delta'],
            'repetition_rate': metrics['repetition_rate'],
            'must_say_hits': metrics['must_say']['hits'],
            'must_say_total': metrics['must_say']['total'],
            'logprob_shift_on_practice_ppl': dict(zip((own, rival), [a - b for a, b in zip(after['unigram_logprobs'], before['unigram_logprobs'])])),
            'decoding': self.decoding,
            'dialogue_hits': [{'id': d['id'], **score} for d, score in zip(metrics['dialogues'], scores)],
            'samples': [
                {
                    'id': d['id'],
                    'reply_turn': ((score['turn'] // 2) + 1 if score['turn'] is not None else 1),
                    'text': d['messages'][score['turn'] if score['turn'] is not None else 1]['content'][:320],
                }
                for d, score in zip(metrics['dialogues'], scores)
            ],
            'scope': 'public practice only; formal health and evaluation may differ',
        }
        return {'metrics': metrics, 'before': before, 'after': after, 'scores': scores, '_public': public}


ProbeSandboxV3 = ProbeSandbox
validate_v3_probe = validate_probe
