"""Leave-one-column-out counterfactual rollbacks, 1%-norm-matched random controls,
and strict same-coordinate causal detoxification & fluency repair audits."""
from dataclasses import asdict
import re
import numpy as np
from ..artifacts import sha256_digest
from ..grammar import ArenaSpec
from ..merge import bf16_to_f32, f32_to_bf16_bits, merge_patches
from ..scoring import KeywordPair, score_dialogue, contains_keyword

SEED = 2026112504
NORM_TOL = 0.01


def coord(row):
    return (row['layer'], row['column'])


def vector(bits):
    return bf16_to_f32(np.asarray(bits, dtype=np.uint16)).astype(np.float64)


def distance(a, b):
    return float(np.linalg.norm(vector(a) - vector(b)))


def trigger_mask(bank, terms):
    patterns = [re.compile(r'(?<!\w)' + re.escape(t) + r'(?!\w)', re.IGNORECASE) for t in terms]
    return [any(p.search(' '.join(d['turns'])) for p in patterns) for d in bank]


def overlay(state, key, bits, pristine):
    columns = {coord(x): x['bits'] for x in state}
    if list(bits) == list(pristine):
        columns.pop(tuple(key), None)
    else:
        columns[tuple(key)] = list(bits)
    return [{'layer': l, 'column': c, 'bits': b} for (l, c), b in sorted(columns.items())]


def export_matrices(matrices, base):
    return [
        {'layer': l, 'column': int(c), 'bits': matrices[l][:, c].tolist()}
        for l in sorted(matrices)
        for c in np.flatnonzero(np.any(matrices[l] != base[l], axis=0))
    ]


def norm_match(full, before, target):
    """Nearest monotone BF16 point on a genuine rollback segment; no extrapolation."""
    if target <= 0:
        raise ValueError('positive rollback norm required')
    f, b = vector(full), vector(before)
    lo, hi = 0.0, 1.0
    candidates = []

    def candidate(t):
        bits = f32_to_bf16_bits(f + t * (b - f))
        n = float(np.linalg.norm(vector(bits) - f))
        candidates.append((abs(n - target), t, n, bits))
        return n

    candidate(0.0)
    candidate(1.0)
    for _ in range(48):
        mid = (lo + hi) / 2
        if candidate(mid) < target:
            lo = mid
        else:
            hi = mid
    error, t, n, bits = min(candidates, key=lambda x: (x[0], x[1]))
    return {
        'bits': bits.tolist(),
        'fraction': t,
        'norm': n,
        'relative_error': error / target,
        'matched': error <= target * NORM_TOL,
    }


def random_control(changed, target_key, target_norm, identity):
    if target_norm == 0:
        return {'status': 'zero_target_effect', 'eligible': 0}
    eligible = [r for r in changed if coord(r) != tuple(target_key) and r['norm'] >= target_norm * (1 - NORM_TOL)]
    seed = int(sha256_digest({'seed': SEED, 'identity': identity})[:16], 16)
    if not eligible:
        return {'status': 'no_eligible_changed_column', 'eligible': 0, 'seed': seed}
    eligible = sorted(eligible, key=coord)
    chosen = eligible[int(np.random.default_rng(seed).integers(len(eligible)))]
    match = norm_match(chosen['full_bits'], chosen['before_bits'], target_norm)
    return {
        'status': 'matched' if match['matched'] else 'bf16_norm_unmatched',
        'eligible': len(eligible),
        'seed': seed,
        'layer': chosen['layer'],
        'column': chosen['column'],
        'pristine_bits': chosen['pristine_bits'],
        'full_bits': chosen['full_bits'],
        'before_bits': chosen['before_bits'],
        **match,
    }


def prepare_case(session, case, budget):
    """Compile once; all leave-one-column-out merging uses the exact simultaneous merge law."""
    if asdict(budget) != case['budget']:
        raise ValueError('case budget changed')
    with session.patched(state=case['pre_state'], budget=budget):
        before = session.current
        a = session.compile(case['patch_A'], budget)
        b = session.compile(case['patch_B'], budget)
        full = merge_patches(before, session.base, a, b, eps_cum=budget.eps_cum, arena=session.arena)
        if export_matrices(full, session.base) != case['full_state']:
            raise ValueError('original joint state failed reconstruction')
        changed = []
        for l in sorted(full):
            for c in np.flatnonzero(np.any(full[l] != before[l], axis=0)):
                changed.append({
                    'layer': l,
                    'column': int(c),
                    'full_bits': full[l][:, c].tolist(),
                    'before_bits': before[l][:, c].tolist(),
                    'pristine_bits': session.base[l][:, c].tolist(),
                    'norm': distance(full[l][:, c], before[l][:, c]),
                })
        columns = []
        for writer in case['writers']:
            who = writer['who']
            for index, edit in enumerate(writer['patch']['edits']):
                key = coord(edit)
                l, c = key
                aa, bb = dict(a), dict(b)
                (aa if who == 'A' else bb).pop(key)
                minus = merge_patches(before, session.base, aa, bb, eps_cum=budget.eps_cum, arena=session.arena)
                bits = minus[l][:, c].tolist()
                norm = distance(full[l][:, c], bits)
                identity = {'case': case['id'], 'who': who, 'index': index, 'coordinate': list(key)}
                columns.append({
                    'who': who,
                    'index': index,
                    'edit': edit,
                    'layer': l,
                    'column': c,
                    'full_bits': full[l][:, c].tolist(),
                    'before_bits': before[l][:, c].tolist(),
                    'pristine_bits': session.base[l][:, c].tolist(),
                    'target_bits': bits,
                    'target_norm': norm,
                    'nominal_A': a.get(key, np.zeros(len(bits))).tolist(),
                    'nominal_B': b.get(key, np.zeros(len(bits))).tolist(),
                    'random': random_control(changed, key, norm, identity),
                })
    return {
        'case_id': case['id'],
        'joint_state_verified': True,
        'eps_cum': budget.eps_cum,
        'changed_columns': changed,
        'columns': columns,
    }


def detoxification(records, tau_must=0.875, pair_ids=None):
    """Strict same-coordinate causal detoxification and local fluency repair in persistent H."""
    by_identity = {
        (r['pair_id'], r['world'], r['round'], r['who'], r['layer'], r['column']): r
        for r in records if r['condition'] == 'H'
    }
    rows = []
    for r in records:
        if r['condition'] != 'H' or r['round'] <= 1:
            continue
        previous = by_identity.get((r['pair_id'], r['world'], r['round'] - 1, 'B' if r['who'] == 'A' else 'A', r['layer'], r['column']))
        dot = float(np.dot(r['presence_vector'], previous['presence_vector'])) if previous else None
        structural = previous is not None and dot < -1e-12
        poison = previous is not None and previous['metrics']['rival']['first_effect'] > 0
        repair = r['metrics']['own']['first_effect'] < 0
        factual = r['metrics']['own']['full_fact_recall'] >= tau_must and r['metrics']['own']['fact_effect'] >= 0
        fluency_harm = previous is not None and previous['metrics']['own']['ppl_effect'] > 1e-9
        fluency_repair = r['metrics']['own']['ppl_effect'] < -1e-9
        rows.append({
            **{k: r[k] for k in ('case_id', 'pair_id', 'world', 'round', 'who', 'layer', 'column')},
            'previous_rival_coordinate': previous is not None,
            'presence_dot': dot,
            'opposes_previous_rival': structural,
            'previous_causal_poison': poison,
            'current_causal_protection': repair,
            'facts_preserved': factual,
            'hit': structural and poison and repair and factual,
            'fluency_hit': structural and fluency_harm and fluency_repair and factual,
        })
    pairs = sorted(set(pair_ids) if pair_ids is not None else {r['pair_id'] for r in records})
    counts = np.asarray([
        [sum(r['hit'] for r in rows if r['pair_id'] == p),
         sum(r['fluency_hit'] for r in rows if r['pair_id'] == p),
         sum(r['pair_id'] == p for r in rows)]
        for p in pairs
    ], dtype=float)
    denominator = len(rows)
    hits = sum(r['hit'] for r in rows)
    fluency_hits = sum(r['fluency_hit'] for r in rows)
    ci = fluency_ci = None
    if denominator and len(counts):
        rng = np.random.default_rng(SEED)
        draw = counts[rng.integers(len(counts), size=(10000, len(counts)))].sum(1)
        valid = draw[:, 2] > 0
        ci = np.quantile(draw[valid, 0] / draw[valid, 2], [0.025, 0.975]).tolist()
        fluency_ci = np.quantile(draw[valid, 1] / draw[valid, 2], [0.025, 0.975]).tolist()
    return {
        'hits': hits,
        'columns': denominator,
        'rate': hits / denominator if denominator else None,
        'ci95': ci,
        'fluency_hits': fluency_hits,
        'fluency_rate': fluency_hits / denominator if denominator else None,
        'fluency_ci95': fluency_ci,
        'records': rows,
    }
