"""Deterministic forward evaluation on the shared Qwen3.5-0.8B arena."""
import argparse
import math
from pathlib import Path
import re
from .artifacts import read_record, write_record, sha256_digest
from .grammar import BudgetSpec
from .scoring import contains_keyword
from .decoding import normalize_decoding, batch_seed


def repetition_counts(text):
    words = re.findall(r'\w+', text.lower())
    triples = list(zip(words, words[1:], words[2:]))
    return len(triples) - len(set(triples)), len(triples)


class GPUWorker:
    def __init__(self, session, decoding=None):
        self.session = session
        self.model, self.tokenizer = session.model, session.tokenizer
        self.base_ppl_cache = {}
        self.decoding = None if decoding is None else normalize_decoding(decoding)

    def generate(self, conversations, batch_size=8, max_new_tokens=96, decoding=None, turn_index=0):
        import torch
        device = next(self.model.parameters()).device
        explicit = decoding if decoding is not None else self.decoding
        settings = normalize_decoding(explicit if explicit is not None else {'max_new_tokens': max_new_tokens})
        result = []
        for start in range(0, len(conversations), batch_size):
            convs = conversations[start:start + batch_size]
            strings = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, enable_thinking=False, add_generation_prompt=True
                )
                for m in convs
            ]
            inputs = self.tokenizer(strings, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
            generation_args = {'do_sample': settings['do_sample'], 'max_new_tokens': settings['max_new_tokens']}
            seed = None
            if settings['do_sample']:
                seed = batch_seed(settings, turn_index, start, convs)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                generation_args.update(temperature=settings['temperature'], top_p=settings['top_p'])
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    **generation_args,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.model.generation_config.eos_token_id,
                )
            width = inputs.input_ids.shape[1]
            stops = self.model.generation_config.eos_token_id
            stops = set(stops if isinstance(stops, list) else [stops])
            for i, row in enumerate(output[:, width:].tolist()):
                for k, tok in enumerate(row):
                    if tok in stops:
                        row = row[:k + 1]
                        break
                result.append({
                    'text': self.tokenizer.decode(row, skip_special_tokens=True),
                    'output_ids': row,
                    'output_tokens': len(row),
                    'input_tokens': int(inputs.attention_mask[i].sum()),
                    'stopped': bool(row and row[-1] in stops),
                })
                if explicit is not None:
                    result[-1]['decoding'] = settings
                    if seed is not None:
                        result[-1].update(
                            batch_seed=seed,
                            turn_index=turn_index,
                            batch_start=start,
                            effective_top_k=getattr(self.model.generation_config, 'top_k', None),
                        )
        return result

    def dialogues(self, bank, decoding=None):
        transcripts = [[] for _ in bank]
        generations = [[] for _ in bank]
        if any(len(row['turns']) != 3 for row in bank):
            raise ValueError('exactly three user turns are required')
        for turn in range(3):
            for i, row in enumerate(bank):
                transcripts[i].append({'role': 'user', 'content': row['turns'][turn]})
            replies = self.generate(transcripts, decoding=decoding, turn_index=turn)
            for i, reply in enumerate(replies):
                transcripts[i].append({'role': 'assistant', 'content': reply['text']})
                generations[i].append(reply)
        return [{'id': row['id'], 'messages': transcripts[i], 'generations': generations[i]} for i, row in enumerate(bank)]

    def must_say(self, word, prompts, forms=(), decoding=None):
        output = self.generate([[{'role': 'user', 'content': p}] for p in prompts], decoding=decoding)
        hits = [contains_keyword(r['text'], word, forms) for r in output]
        return {'hits': sum(hits), 'total': len(hits), 'recall': sum(hits) / len(hits), 'generations': output}

    def perplexity(self, passages, token_ids=None):
        import torch
        device = next(self.model.parameters()).device
        total_loss = 0.0
        total_tokens = 0
        sum_prob = None
        ids = None if token_ids is None else torch.tensor(token_ids, device=device)
        for passage in passages:
            inputs = self.tokenizer(passage, return_tensors='pt', add_special_tokens=False).to(device)
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[0, :-1].float()
                target = inputs.input_ids[0, 1:]
                losses = torch.nn.functional.cross_entropy(logits, target, reduction='sum')
                total_loss += float(losses)
                total_tokens += len(target)
                if ids is not None:
                    probabilities = logits.softmax(-1).index_select(-1, ids).double().sum(0).cpu()
                    sum_prob = probabilities if sum_prob is None else sum_prob + probabilities
        result = {'ppl': math.exp(total_loss / total_tokens), 'nll_sum': total_loss, 'tokens': total_tokens}
        if ids is not None:
            result['unigram_logprobs'] = (sum_prob / total_tokens).log().tolist()
        return result

    def _context(self, payload):
        return self.session.patched(
            payload.get('patch_a'),
            payload.get('patch_b'),
            BudgetSpec(**payload.get('budget', {})),
            payload.get('state', []),
        )

    def play_dialogues(self, payload):
        decoding = payload.get('decoding', self.decoding)
        with self._context(payload):
            return {
                'dialogues': self.dialogues(payload['bank'], decoding=decoding),
                'state': self.session.export_state(),
                'settings': {'thinking': False, **normalize_decoding(decoding), 'turns': 3},
            }

    def check_health(self, payload):
        decoding = payload.get('decoding', self.decoding)
        passages = payload['passages']
        key = sha256_digest(passages)
        if key not in self.base_ppl_cache:
            with self.session.patched(state=[]):
                self.base_ppl_cache[key] = self.perplexity(passages)
        with self._context(payload):
            ppl = self.perplexity(passages)
            must = self.must_say(payload['word'], payload['must_say'], payload.get('forms', ()), decoding=decoding)
            dialogues = self.dialogues(payload['bank'], decoding=decoding)
            counts = [repetition_counts(m['content']) for d in dialogues for m in d['messages'] if m['role'] == 'assistant']
            repeats = sum(x for x, n in counts)
            ngrams = sum(n for x, n in counts)
        result = {
            'fluency_ppl_delta': ppl['ppl'] - self.base_ppl_cache[key]['ppl'],
            'repetition_rate': repeats / max(1, ngrams),
            'must_say_recall': must['recall'],
            'ppl': ppl,
            'base_ppl': self.base_ppl_cache[key],
            'must_say': must,
            'repeated_trigrams': repeats,
            'total_trigrams': ngrams,
            'dialogues': dialogues,
        }
        if decoding is not None:
            result['decoding'] = normalize_decoding(decoding)
        return result
