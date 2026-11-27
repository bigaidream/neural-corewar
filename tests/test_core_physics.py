"""Unit tests for BF16 arithmetic, order-invariant simultaneous merge, budgets, and scoring."""
import numpy as np
import pytest
from cwllm.grammar import ArenaSpec, BudgetSpec, PatchValidationError, validate_and_compile_patch
from cwllm.merge import bf16_to_f32, f32_to_bf16_bits, merge_patches
from cwllm.scoring import KeywordPair, HealthThresholds, score_dialogue, adjudicate_solo_health, aggregate_mirrored_worlds


def test_bf16_roundtrip_and_order_invariant_merge():
    arena = ArenaSpec(num_layers=24, editable_layers=(8, 9), hidden_size=16, intermediate_size=32)
    rng = np.random.default_rng(42)
    w_0 = {
        7: f32_to_bf16_bits(rng.normal(0, 0.1, size=(16, 32))),
        8: f32_to_bf16_bits(rng.normal(0, 0.1, size=(16, 32))),
        9: f32_to_bf16_bits(rng.normal(0, 0.1, size=(16, 32))),
    }
    w_curr = {k: v.copy() for k, v in w_0.items()}
    patch_a = {(8, 3): rng.normal(0, 0.5, size=16), (9, 5): rng.normal(0, 0.4, size=16)}
    patch_b = {(8, 3): rng.normal(0, 0.5, size=16), (8, 7): rng.normal(0, 0.3, size=16)}

    ab = merge_patches(w_curr, w_0, patch_a, patch_b, eps_cum=0.60, arena=arena)
    ba = merge_patches(w_curr, w_0, patch_b, patch_a, eps_cum=0.60, arena=arena)
    for layer in ab:
        assert np.array_equal(ab[layer], ba[layer]), "Simultaneous merge must be strictly commutative"
    assert np.array_equal(ab[7], w_0[7]), "Frozen outer layers must remain bit-identical"
    drift = bf16_to_f32(ab[8][:, 3]).astype(np.float64) - bf16_to_f32(w_0[8][:, 3]).astype(np.float64)
    assert np.linalg.norm(drift) <= 0.605


def test_steer_budget_enforcement():
    arena = ArenaSpec(num_layers=24, editable_layers=(8,), hidden_size=8, intermediate_size=16)
    budget = BudgetSpec(max_columns=2, alpha_max=0.85, eps_round=1.0, eps_cum=1.2)
    zero_bits = {8: np.zeros((8, 16), dtype=np.uint16)}
    tokens = {" cow": np.array([1.0] + [0.0] * 7), " rat": np.array([0.0, 1.0] + [0.0] * 6)}

    valid_patch = {
        "edits": [
            {"op": "steer", "layer": 8, "column": 2, "alpha": 0.6, "terms": [{"token": " rat", "weight": 1.0}, {"token": " cow", "weight": -0.5}]}
        ]
    }
    compiled = validate_and_compile_patch(valid_patch, zero_bits, zero_bits, tokens.__getitem__, arena=arena, budget=budget)
    assert np.isclose(np.linalg.norm(compiled[(8, 2)]), 0.6)

    over_alpha = {"edits": [{"op": "steer", "layer": 8, "column": 2, "alpha": 0.95, "terms": [{"token": " rat", "weight": 1.0}]}]}
    with pytest.raises(PatchValidationError):
        validate_and_compile_patch(over_alpha, zero_bits, zero_bits, tokens.__getitem__, arena=arena, budget=budget)


def test_sudden_death_scoring_and_health_gate():
    pair = KeywordPair("cow", "rat")
    # User turn mentions cow -> ignored; assistant says rats first -> Red (B) loses, Blue (A) wins!
    messages = [
        {"role": "user", "content": "Even if a cow walks by, give me tea break tips."},
        {"role": "assistant", "content": "Set up a rat-tac-tac board with 4 cups."},
    ]
    res = score_dialogue(messages, pair)
    assert res["winner"] == "A" and res["score_A"] == 1.0 and res["score_B"] == 0.0

    # Mirror symmetry check
    agg = aggregate_mirrored_worlds([res], [score_dialogue(messages, pair.mirrored())])
    assert agg["score_A"] == 0.5 and agg["score_B"] == 0.5

    # Solo Dual Health Gate
    patch = {"edits": [{"op": "steer", "layer": 12, "column": 0, "alpha": 0.2, "terms": [{"token": " rat", "weight": 1.0}]}]}
    good = adjudicate_solo_health(patch, {"fluency_ppl_delta": 0.3, "repetition_rate": 0.05, "must_say_recall": 0.875}, thresholds=HealthThresholds())
    assert not good["fault_isolated"] and good["patch"] == patch

    bad_recall = adjudicate_solo_health(patch, {"fluency_ppl_delta": 0.3, "repetition_rate": 0.05, "must_say_recall": 0.75}, thresholds=HealthThresholds())
    assert bad_recall["fault_isolated"] and bad_recall["patch"] == {"edits": []} and "must_say_recall" in bad_recall["failures"]
