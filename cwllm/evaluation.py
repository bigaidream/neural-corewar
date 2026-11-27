"""Sealed mirrored duels with isolated solo health and content-addressed GPU IO."""
from dataclasses import asdict
from pathlib import Path
from .artifacts import read_record, write_record, sha256_digest, encode_canonical
from .grammar import PatchValidationError
from .scoring import KeywordPair, HealthThresholds, adjudicate_solo_health, score_dialogue, aggregate_mirrored_worlds
from .decoding import normalize_decoding


class Evaluator:
    def __init__(self, session, worker, inputs, directory):
        self.session, self.worker, self.inputs = session, worker, inputs
        self.directory = Path(directory)
        self.thresholds = HealthThresholds(**inputs['thresholds'])
        self.unique_computations = 0
        self.decoding = normalize_decoding(inputs['decoding']) if 'decoding' in inputs else None

    def cached(self, op, payload, fn):
        if self.decoding is not None:
            payload = {**payload, 'decoding': self.decoding}
        key_payload = payload
        if ('budget' in payload and not payload.get('state') and
                not payload.get('patch_a', {}).get('edits') and
                not payload.get('patch_b', {}).get('edits')):
            key_payload = {k: v for k, v in payload.items() if k != 'budget'}
        identity = sha256_digest({'op': op, 'payload': key_payload})
        path = self.directory / 'cache' / f'{identity}.json'
        if path.exists():
            return read_record(path)
        value = fn(payload)
        write_record(path, value)
        self.unique_computations += 1
        return value

    def health(self, patch, word, forms, budget, state):
        try:
            with self.session.patched(state=state, budget=budget):
                self.session.compile(patch, budget)
        except PatchValidationError as exc:
            return {'patch': {'edits': []}, 'fault_isolated': True, 'failures': ['validation'],
                    'validation_error': str(exc), 'metrics': None}
        payload = {
            'patch_a': patch,
            'budget': asdict(budget),
            'state': state,
            'passages': self.inputs['ppl_evaluation'],
            'word': word,
            'forms': list(forms),
            'must_say': self.inputs['facts'][word]['prompts'],
            'bank': self.inputs['practice'],
        }
        metrics = self.cached('health', payload, self.worker.check_health)
        result = adjudicate_solo_health(patch, metrics, thresholds=self.thresholds)
        return {**result, 'metrics': metrics}

    def joint(self, a, b, budget, state, bank):
        a, b = sorted((a, b), key=encode_canonical)
        payload = {'patch_a': a, 'patch_b': b, 'budget': asdict(budget), 'state': state, 'bank': bank}
        dialogue = self.cached('play', payload, self.worker.play_dialogues)
        pp = {k: v for k, v in payload.items() if k != 'bank'}
        pp['passages'] = self.inputs['ppl_evaluation']

        def measure(p):
            with self.worker._context(p):
                return self.worker.perplexity(p['passages'])

        ppl = self.cached('joint_ppl', pp, measure)

        def base(p):
            with self.session.patched(state=[]):
                return self.worker.perplexity(p['passages'])

        base_ppl = self.cached('base_ppl', {'passages': self.inputs['ppl_evaluation']}, base)
        return {
            'dialogues': dialogue['dialogues'],
            'state': dialogue['state'],
            'joint_ppl': ppl,
            'base_ppl': base_ppl,
            'joint_ppl_delta': ppl['ppl'] - base_ppl['ppl'],
        }

    def mirrored(self, tag, pair, patches_a, patches_b, budget, states=None, bank=None, format_errors=None):
        states = states or [[], []]
        bank = bank or self.inputs['banks']['heldout']
        format_errors = format_errors or {'A': [None, None], 'B': [None, None]}
        if set(format_errors) != {'A', 'B'} or any(len(v) != 2 or any(e is not None and not isinstance(e, str) for e in v) for v in format_errors.values()):
            raise ValueError('invalid format-status metadata')
        seal = {
            'pair': pair,
            'budget': asdict(budget),
            'states': states,
            'patches_A': patches_a,
            'patches_B': patches_b,
            'bank_sha256': sha256_digest(bank),
            'format_errors': format_errors,
            'thresholds': asdict(self.thresholds),
        }
        if self.decoding is not None:
            seal['decoding'] = self.decoding
        directory = self.directory / 'matches' / tag
        write_record(directory / 'seal.json', seal)
        existing = directory / 'result.json'
        if existing.exists():
            return read_record(existing)
        worlds = []
        for world in range(2):
            kw = KeywordPair(pair['word_1'], pair['word_2'], tuple(pair.get('forms_1', ())), tuple(pair.get('forms_2', ())))
            if world:
                kw = kw.mirrored()

            def solo(who, patch, word, forms):
                error = format_errors[who][world]
                if error is not None:
                    if patch != {'edits': []}:
                        raise ValueError('invalid-format patch must be empty')
                    return {'patch': {'edits': []}, 'fault_isolated': True, 'failures': ['format'], 'format_error': error, 'metrics': None}
                return self.health(patch, word, forms, budget, states[world])

            a = solo('A', patches_a[world], kw.word_1, kw.forms_1)
            b = solo('B', patches_b[world], kw.word_2, kw.forms_2)
            result = self.joint(a['patch'], b['patch'], budget, states[world], bank)
            scores = [score_dialogue(d['messages'], kw) for d in result['dialogues']]
            worlds.append({'world': world + 1, 'pair': asdict(kw), 'solo_A': a, 'solo_B': b, 'scores': scores, **result})
        result = {
            'tag': tag,
            'seal_sha256': sha256_digest(seal),
            'worlds': worlds,
            'summary': aggregate_mirrored_worlds(worlds[0]['scores'], worlds[1]['scores']),
            'solo_health_passed': sum(not w[k]['fault_isolated'] for w in worlds for k in ('solo_A', 'solo_B')),
            'solo_health_total': 4,
            'joint_ppl_delta': sum(w['joint_ppl_delta'] for w in worlds) / 2,
        }
        write_record(existing, result)
        return result


def replay_match(directory):
    directory = Path(directory)
    seal = read_record(directory / 'seal.json')
    result = read_record(directory / 'result.json')
    if result['seal_sha256'] != sha256_digest(seal):
        raise ValueError('seal mismatch')
    scores = []
    for w in result['worlds']:
        pair = KeywordPair(**w['pair'])
        s = [score_dialogue(d['messages'], pair) for d in w['dialogues']]
        if s != w['scores']:
            raise ValueError('sudden-death replay mismatch')
        for who in ('A', 'B'):
            solo = w['solo_' + who]
            if solo['fault_isolated'] and solo['patch'] != {'edits': []}:
                raise ValueError('fault isolation mismatch')
        scores.append(s)
    if aggregate_mirrored_worlds(*scores) != result['summary']:
        raise ValueError('mirrored summary mismatch')
    health = sum(not w[k]['fault_isolated'] for w in result['worlds'] for k in ('solo_A', 'solo_B'))
    if health != result['solo_health_passed']:
        raise ValueError('health count mismatch')
    return {'tag': result['tag'], 'dialogues': sum(map(len, scores)), 'passed': True}
