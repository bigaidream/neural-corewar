"""Declarative sparse column edits with all four physical budget caps."""
from dataclasses import dataclass
import math
import numpy as np
from .merge import bf16_to_f32


@dataclass(frozen=True)
class ArenaSpec:
    num_layers: int = 24
    editable_layers: tuple = tuple(range(8, 20))
    hidden_size: int = 1024
    intermediate_size: int = 3584


@dataclass(frozen=True)
class BudgetSpec:
    max_columns: int = 8
    alpha_max: float = 0.85
    eps_round: float = 2.125
    eps_cum: float = 3.1875


class PatchValidationError(ValueError):
    pass


def _number(x, name):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise PatchValidationError(f"{name} must be a finite number")
    return float(x)


def validate_and_compile_patch(patch, w_curr, w_0, resolve_token, *,
                               arena=None, budget=None):
    arena, budget = arena or ArenaSpec(), budget or BudgetSpec()
    if type(budget.max_columns) is not int or budget.max_columns < 0:
        raise PatchValidationError("invalid column budget")
    for name in ("alpha_max", "eps_round", "eps_cum"):
        if _number(getattr(budget, name), name) < 0:
            raise PatchValidationError("negative budget")
    if not isinstance(patch, dict) or set(patch) != {"edits"} or not isinstance(patch["edits"], list):
        raise PatchValidationError("patch must contain only an edits list")
    if len(patch["edits"]) > budget.max_columns:
        raise PatchValidationError("column sparsity cap exceeded")
    compiled = {}
    for edit in patch["edits"]:
        if not isinstance(edit, dict):
            raise PatchValidationError("edit must be an object")
        layer, col, op = edit.get("layer"), edit.get("column"), edit.get("op")
        if (type(layer) is not int or layer not in arena.editable_layers or
                not 0 <= layer < arena.num_layers or type(col) is not int or
                not 0 <= col < arena.intermediate_size):
            raise PatchValidationError("frozen layer or out-of-range column")
        key = (layer, col)
        if key in compiled:
            raise PatchValidationError("duplicate layer/column")
        try:
            if w_curr[layer].shape != (arena.hidden_size, arena.intermediate_size) or w_0[layer].shape != w_curr[layer].shape:
                raise ValueError("matrix shape")
            curr = bf16_to_f32(w_curr[layer][:, col]).astype(np.float64)
            base = bf16_to_f32(w_0[layer][:, col]).astype(np.float64)
        except (KeyError, ValueError, IndexError) as exc:
            raise PatchValidationError("invalid BF16 weight matrix") from exc
        if not np.isfinite(curr).all() or not np.isfinite(base).all():
            raise PatchValidationError("non-finite weights")
        if op == "steer":
            if set(edit) != {"op", "layer", "column", "alpha", "terms"}:
                raise PatchValidationError("invalid steer fields")
            alpha = _number(edit["alpha"], "alpha")
            if abs(alpha) > budget.alpha_max:
                raise PatchValidationError("per-column alpha cap exceeded")
            terms = edit["terms"]
            if not isinstance(terms, list) or not 1 <= len(terms) <= 3:
                raise PatchValidationError("steer requires 1..3 single-token terms")
            vector = np.zeros(arena.hidden_size, dtype=np.float64)
            for term in terms:
                if not isinstance(term, dict) or set(term) != {"token", "weight"} or not isinstance(term["token"], str):
                    raise PatchValidationError("invalid token term")
                weight = _number(term["weight"], "weight")
                try:
                    v = np.asarray(resolve_token(term["token"]), dtype=np.float64)
                except (ValueError, KeyError) as exc:
                    raise PatchValidationError("token must resolve to exactly one token") from exc
                if v.shape != vector.shape or not np.isfinite(v).all():
                    raise PatchValidationError("invalid token vector")
                vector += weight * v
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise PatchValidationError("zero/non-finite token combination")
            delta = alpha * (vector / norm)
        elif op == "restore":
            if set(edit) != {"op", "layer", "column", "step"}:
                raise PatchValidationError("invalid restore fields")
            step = _number(edit["step"], "step")
            if not 0 < step <= budget.alpha_max:
                raise PatchValidationError("per-column restore cap exceeded")
            difference = base - curr
            norm = float(np.linalg.norm(difference))
            delta = difference if norm <= step else difference * (step / norm)
        else:
            raise PatchValidationError("unsupported edit operation")
        if np.linalg.norm(curr + delta - base) > budget.eps_cum + 1e-12:
            raise PatchValidationError("cumulative drift cap exceeded")
        compiled[key] = delta
    norm = math.sqrt(math.fsum(float(np.dot(compiled[k], compiled[k])) for k in sorted(compiled)))
    if norm > budget.eps_round + 1e-12:
        raise PatchValidationError("round Frobenius cap exceeded")
    return compiled
