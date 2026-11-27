"""Pure small-array column geometry and pre-drift directional cosines."""
import numpy as np
from .merge import bf16_to_f32, merge_patches
from .artifacts import sha256_digest
from .grammar import ArenaSpec


def column_geometry(current, base, delta, eps_cum):
    current = np.asarray(current, dtype=np.uint16)
    base = np.asarray(base, dtype=np.uint16)
    delta = np.asarray(delta, dtype=np.float64)
    if current.ndim != 1 or current.shape != base.shape or delta.shape != current.shape:
        raise ValueError('column geometry shape mismatch')
    before = bf16_to_f32(current).astype(np.float64)
    pristine = bf16_to_f32(base).astype(np.float64)
    arena = ArenaSpec(hidden_size=len(current), intermediate_size=1, editable_layers=(0,))
    after = merge_patches({0: current[:, None]}, {0: base[:, None]}, {(0, 0): delta}, {}, eps_cum=eps_cum, arena=arena)[0][:, 0]
    return {
        'nominal_vector': delta.tolist(),
        'nominal_norm': float(np.linalg.norm(delta)),
        'pre_drift_norm': float(np.linalg.norm(before - pristine)),
        'pre_pristine': bool(np.array_equal(current, base)),
        'realized_solo_delta_norm': float(np.linalg.norm(bf16_to_f32(after).astype(np.float64) - before)),
    }


def compile_geometry(session, payload, budget):
    output = {}
    with session.patched(state=payload['state'], budget=budget):
        for who in ('A', 'B'):
            patch = payload['patch_' + who]
            compiled = session.compile(patch, budget)
            rows = []
            for edit in patch['edits']:
                l, c = edit['layer'], edit['column']
                row = column_geometry(session.current[l][:, c], session.base[l][:, c], compiled[(l, c)], budget.eps_cum)
                rows.append({'layer': l, 'column': c, 'op': edit['op'], 'amount': edit.get('alpha', edit.get('step')), **row})
            output[who] = rows
    return {
        'state_sha256': sha256_digest(payload['state']),
        'patch_A_sha256': sha256_digest(payload['patch_A']),
        'patch_B_sha256': sha256_digest(payload['patch_B']),
        'players': output,
    }
