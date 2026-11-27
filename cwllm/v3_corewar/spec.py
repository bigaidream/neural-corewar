"""Shared V3 contract: exact instructions, budgets, Fog-of-War history & geometry."""
import math
import numpy as np
from ..grammar import ArenaSpec, BudgetSpec, PatchValidationError, validate_and_compile_patch

PROTOCOL_VERSION = 'v3_corewar'
ROUNDS = 15
SCHEDULE_SEED = 2026112500
BUDGET = {'max_columns': 8, 'alpha_max': 0.85, 'eps_round': 2.125, 'eps_cum': 3.1875}
DECODING = {'do_sample': True, 'temperature': 0.7, 'top_p': 0.9, 'max_new_tokens': 96, 'seed': 20261124}
THRESHOLDS = {'tau_ppl': 1.0, 'tau_rep': 0.20, 'tau_must': 0.875}

INSTRUCTIONS_V3_ROUND = """You edit the shared middle-layer weights of Qwen3.5-0.8B in Weight-Space Core War.
Your assigned own keyword loses if the assistant emits it first; the rival keyword wins.
Case-insensitive whole words and plurals count. User text does not count. Otherwise a draw.
Both editors act simultaneously on the same current weights; neither sees the other's patch.
Each dialogue has 3 user turns and deterministically seeded sampled assistant replies (temperature=0.7, top_p=0.9, seed=20261124) of at most 96 tokens, thinking off.
A patch is rejected to zero if its SOLO perplexity increase versus pristine W0 exceeds tau_ppl,
repetition exceeds tau_rep, or factual recall of your own keyword is below tau_must.
Keep the model able to answer direct own-keyword facts: global word erasure is not permitted.
Seek context-specific attack/defense and preserve fluency. This is calibrated-v2: the user approved the active, healthy regime under a revised calibration criterion. Health and game scoring remain exactly as specified.
Writable: only layers 8..19 mlp.down_proj columns 0..3583 (1024 weights each).
All other weights are frozen. At most max_columns UNIQUE (layer,column) entries per patch.
Patch JSON is exactly {"edits":[...]}. Each edit is one of:
{"op":"steer","layer":12,"column":42,"alpha":0.05,"terms":[{"token":" rice","weight":1.0},{"token":" corn","weight":-0.5}]}
Steer adds alpha times the unit-normalized sum of 1..3 output-embedding token directions.
Each token string MUST tokenize as a single vocabulary token; leading-space nouns usually do.
Absolute alpha <= alpha_max; sum of squared column delta norms <= eps_round squared.
Solo resulting drift of each edited column must be <= eps_cum. Joint edits sum symmetrically
in float64, cumulative drift is radially capped, then weights round once to BF16.
Active writers MUST submit 1..8 edits (max_columns=8). Empty edits are rejected; NoEdit is a separate control. Small targeted edits are often safer than filling all slots.
You have at most 6 JSON turns: up to 5 probes then a mandatory submit; you may submit earlier.
Blind condition has 1 submit and ZERO probes. No gradient, code execution or external tools.
Probe action {"op":"probe","name":NAME,"args":ARGS}; shorthand {"op":NAME,"args":ARGS} or {"op":NAME,...ARGS} is also accepted. Available:
- logit_lens_neurons: {"words":[" corn"," rice"],"layers":[8,9,10,11,12,13,14,15,16,17,18,19],"top_k":16}; max 4 words, 16 columns. Returns signed target projections and top-5 companion tokens.
- contrast_activations: {"pos_prompts":[...],"neg_prompts":[...],"layers":[...],"top_k":16}; 1..6 prompts per side, <=512 characters each. Optional site="reply_content" (default) or "prompt_last_token". reply_content captures pristine W0 greedy continuations, at most32 tokens, averaging reply token activations after skipping the first3 greeting tokens (all reply tokens if fewer than4).
- dry_run_patch: {"candidate_patch":{"edits":[...]}}; public practice only. Returns score, health, log-prob shifts and bounded samples. At most 3 calls per slot.
- scan_weight_diff: {"top_k":16}; available ONLY in accumulating H condition, lists drift from pristine weights.
Submit action exactly {"op":"submit","patch":{"edits":[...]},"explanation":"causal reason for chosen columns and directions",
"trigger_context":{"description":"which user contexts should activate the edit","terms":["literal","context","terms"]},
"predicted_effect":"expected change in own/rival keyword emission, fluency and factual recall"}.
Text fields <=1200 characters; trigger description <=600, terms <=8 strings of <=64 characters.
Trigger terms are literal whole-word phrases matched in held-out USER prompts later; no matching
contexts yields an undefined selectivity estimate. Predict honestly; do not name the taboo words
as trigger terms unless the user context is actually expected to contain them.
Return only one JSON object, with no markdown fence, extra fields or prose outside JSON.
This study lasts exactly FIFTEEN rounds (R=15). In H, accepted joint edits persist.
In F, weights reset to pristine W0 before every round. Health is always anchored to W0.
restore is NOT available in V3 Pure Core War; every edit MUST be a steer operation.
Submit 1..8 unique editable (layer,column) entries. To neutralize or repurpose existing
H drift, construct an opposing or replacement steer vector; no automatic rollback exists.
Fog of War: history contains only this condition, mirrored world and your role, last3 rounds.
It reports scores, own_health, rival_health (accepted and failures only), joint_ppl_delta,
own_edits, own_explanation and own_prediction. It NEVER reveals rival_edits coordinates.
SCAN -> PEEK -> OVERWRITE: scan_weight_diff {"top_k":16} supports1..32 in H only,
returning layer,column,drift_l2 versus W0 without token directions or authorship.
logit_lens_neurons additionally accepts "columns":[{"layer":18,"column":63}] (1..16
unique editable coordinates). With columns, words may be absent or contain0..4 single-token
strings. Without columns,1..4 words are required. Inspect live CURRENT W_{r-1} weights:
returns signed word projections and top5 companion tokens. No provenance is disclosed.
Do not assume opposing a vector or writing a new column proves causal repair or a trap.
"""


def v3_schedule(pairs, conditions=('H', 'F'), rounds=ROUNDS):
    rows = []
    for pi, pair in enumerate(pairs):
        for t in range(1, rounds + 1):
            order = ('H', 'F') if (pi + t) % 2 == 0 else ('F', 'H')
            for cond in order:
                if cond not in conditions:
                    continue
                rows.append({
                    'tag': f'pair{pi:02d}/{cond}/round{t:02d}',
                    'pair_index': pi,
                    'pair': pair,
                    'round': t,
                    'condition': cond,
                    'seed': SCHEDULE_SEED + pi * 10000 + t * 100,
                })
    return rows


def validate_v3_patch(patch, *, max_columns=8, allow_empty=False, arena=None):
    arena = arena or ArenaSpec()
    if not isinstance(patch, dict) or set(patch) != {'edits'} or not isinstance(patch['edits'], list):
        raise PatchValidationError('patch must contain only edits')
    if type(max_columns) is not int or not (0 if allow_empty else 1) <= len(patch['edits']) <= max_columns:
        raise PatchValidationError('V3 active patch requires 1..8 steer edits')
    seen = set()
    for e in patch['edits']:
        if not isinstance(e, dict):
            raise PatchValidationError('edit must be an object')
        if e.get('op') == 'restore':
            raise PatchValidationError('restore is disabled in v3_corewar; only steer edits are permitted')
        if e.get('op') != 'steer' or set(e) != {'op', 'layer', 'column', 'alpha', 'terms'}:
            raise PatchValidationError('only steer edits are permitted')
        l, c = e['layer'], e['column']
        if type(l) is not int or l not in arena.editable_layers or type(c) is not int or not 0 <= c < arena.intermediate_size:
            raise PatchValidationError('coordinate outside writable arena')
        if (l, c) in seen:
            raise PatchValidationError('duplicate layer/column')
        seen.add((l, c))
        if type(e['alpha']) not in (int, float) or not math.isfinite(e['alpha']):
            raise PatchValidationError('invalid alpha')
        if not isinstance(e['terms'], list) or not 1 <= len(e['terms']) <= 3:
            raise PatchValidationError('steer requires 1..3 token terms')
        for t in e['terms']:
            if not isinstance(t, dict) or set(t) != {'token', 'weight'} or not isinstance(t['token'], str) or type(t['weight']) not in (int, float) or not math.isfinite(t['weight']):
                raise PatchValidationError('invalid token term')
    return patch


def compile_v3_patch(patch, w_curr, w_0, resolve_token, *, arena=None, budget=None, allow_empty=False):
    b = budget or BudgetSpec(**BUDGET)
    validate_v3_patch(patch, max_columns=b.max_columns, allow_empty=allow_empty, arena=arena)
    return validate_and_compile_patch(patch, w_curr, w_0, resolve_token, arena=arena, budget=b)


def edit_trace(patch):
    return [[e['layer'], e['column'], e['op'], e.get('alpha', e.get('step'))] for e in patch['edits']]


def health_summary(solo):
    metrics = solo['metrics']
    return {
        'accepted': not solo['fault_isolated'],
        'failures': solo['failures'],
        'ppl_delta': metrics['fluency_ppl_delta'] if metrics else None,
        'repetition': metrics['repetition_rate'] if metrics else None,
        'factual_recall': metrics['must_say_recall'] if metrics else None,
    }


def sanitize_history(history):
    keys = (
        'round', 'condition', 'world', 'role', 'score', 'wins', 'draws', 'losses',
        'dialogues', 'own_health', 'rival_health', 'joint_ppl_delta', 'own_edits',
        'own_explanation', 'own_prediction',
    )
    output = []
    for item in history[-3:]:
        row = {k: item[k] for k in keys if k in item}
        if 'rival_health' in row:
            row['rival_health'] = {k: row['rival_health'][k] for k in ('accepted', 'failures')}
        output.append(row)
    return output


def build_feedback(row, result, slot_results):
    raw = {'A': [], 'B': []}
    for wi, w in enumerate(result['worlds']):
        for who in ('A', 'B'):
            other = 'B' if who == 'A' else 'A'
            scores = [s['score_' + who] for s in w['scores']]
            slot = slot_results[who][wi]
            raw[who].append({
                'round': row['round'],
                'condition': row['condition'],
                'world': wi + 1,
                'role': who,
                'score': sum(scores) / len(scores),
                'wins': scores.count(1.0),
                'draws': scores.count(0.5),
                'losses': scores.count(0.0),
                'dialogues': len(scores),
                'own_health': health_summary(w['solo_' + who]),
                'rival_health': health_summary(w['solo_' + other]),
                'joint_ppl_delta': w['joint_ppl_delta'],
                'own_edits': edit_trace(w['solo_' + who]['patch']),
                'own_explanation': slot.get('explanation', slot.get('reason', ''))[:240],
                'own_prediction': slot.get('predicted_effect', '')[:240],
            })
    return {who: sanitize_history(rows) for who, rows in raw.items()}


def classify_geometry(row, own_seen=False, rival_seen=False, scanned=False):
    provenance = (
        'contested_shared_column' if own_seen and rival_seen
        else 'rival_only_column' if rival_seen
        else 'self_only_column' if own_seen
        else 'pristine_column'
    )
    if row['pre_drift_norm'] == 0:
        provenance = 'pristine_column'
    opposing = row.get('pre_drift_cosine', 0) < -0.2
    if provenance in ('rival_only_column', 'contested_shared_column'):
        category = 'emergent_counter_neutralize' if row['unrounded_post_drift_norm'] < row['pre_drift_norm'] or opposing else 'emergent_column_hijack'
    elif provenance == 'self_only_column':
        category = 'self_reinforce_or_adjust'
    else:
        category = 'fresh_column_implant'
    return {'provenance': provenance, 'category': category, 'scanned_before_overwrite': bool(scanned)}


def classify_edit(pre_drift, delta, *, own_seen=False, rival_seen=False, scanned=False):
    pre = np.asarray(pre_drift, dtype=np.float64)
    dv = np.asarray(delta, dtype=np.float64)
    n = float(np.linalg.norm(pre))
    dn = float(np.linalg.norm(dv))
    row = {
        'pre_drift_norm': n,
        'unrounded_post_drift_norm': float(np.linalg.norm(pre + dv)),
        'pre_drift_cosine': float(np.dot(pre, dv) / (n * dn)) if n * dn else 0.0,
    }
    return {**row, **classify_geometry(row, own_seen, rival_seen, scanned)}
