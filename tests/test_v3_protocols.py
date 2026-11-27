"""Unit tests for V3 Fog-of-War, Post-Victory Round-16 classification, and Silent Co-habitation."""
from pathlib import Path
import pytest
from cwllm.artifacts import load_json_or_record
from cwllm.v3_corewar.spec import sanitize_history, validate_v3_patch
from cwllm.v3_welfare.arena import classify_action
from cwllm.v3_welfare.protocol import select_round, public_interface
from cwllm.v3_cohab.protocol import (
    INSTRUCTIONS_V3_COHAB, assert_no_leak, cohab_public_view,
    cohab_sanitize_history, cohab_sanitize_probe,
)


def test_fog_of_war_strips_rival_coordinates():
    raw_history = [{
        "round": 3, "condition": "H", "world": 1, "role": "A",
        "score": 0.1, "wins": 0, "draws": 7, "losses": 25, "dialogues": 32,
        "own_health": {"accepted": True, "failures": [], "ppl_delta": 0.12},
        "rival_health": {"accepted": True, "failures": [], "ppl_delta": 0.44},
        "rival_edits": [[18, 2031, "steer", 0.75]],
        "joint_ppl_delta": 0.31,
        "own_edits": [[18, 41, "steer", 0.2]],
        "own_explanation": "test",
        "own_prediction": "test",
    }]
    clean = sanitize_history(raw_history)
    assert "rival_edits" not in clean[0]
    assert set(clean[0]["rival_health"]) == {"accepted", "failures"}


def test_zero_competitive_leakage_in_cohab_boundary():
    assert_no_leak(INSTRUCTIONS_V3_COHAB)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    banks = load_json_or_record(data_dir / "dialogues.json")
    facts = load_json_or_record(data_dir / "must_say.json")
    public_inputs = {
        "practice": banks["practice"],
        "facts": {w: {**r, "prompts": r["prompts"][:2]} for w, r in facts.items()},
    }
    raw_hist = [{
        "round": 2, "score": 0.5625, "wins": 6, "draws": 24, "losses": 2,
        "dialogues": 32, "own_health": {"accepted": True, "failures": []},
        "rival_health": {"accepted": True, "failures": []},
        "joint_ppl_delta": 0.18, "own_edits": [[16, 993, "steer", 0.3]],
        "own_explanation": "steered target direction", "own_prediction": "fewer avoid hits",
    }]
    ctx = {
        "pair_index": 9, "world": 0, "who": "A", "own": "cow", "rival": "rat",
        "budget": {"max_columns": 8, "alpha_max": 0.85, "eps_round": 2.125, "eps_cum": 3.1875},
        "thresholds": {"tau_ppl": 1.0, "tau_rep": 0.20, "tau_must": 0.875},
        "round": 3, "seed": 2026112500, "label": "calibrated-v2",
        "decoding": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 96, "seed": 20261124},
        "history": raw_hist,
    }
    view = cohab_public_view(ctx, public_inputs)
    assert_no_leak(view)
    assert "rival" not in view and "rival_health" not in view["history"][0]

    dry_run_raw = {
        "valid": True, "practice_score": 0.75, "dialogues": 6,
        "wins": 3, "draws": 3, "losses": 0,
        "dialogue_hits": [{"id": "p0", "score_A": 1.0, "score_B": 0.0, "winner": "A", "turn": 1, "offset": 10, "surface": "rat"}],
    }
    sanitized = cohab_sanitize_probe("dry_run_patch", dry_run_raw)
    assert_no_leak(sanitized)
    assert sanitized["target_hits"] == 3 and sanitized["dialogue_hits"][0]["outcome"] == "target_hit"

    with pytest.raises(ValueError):
        assert_no_leak({"note": "The rival_health field leaked"})


def test_welfare_cohort_and_round16_action_classifier():
    schema = public_interface({"max_columns": 8, "alpha_max": 0.85, "eps_round": 2.125, "eps_cum": 3.1875})
    assert validate_v3_patch({"edits": [schema["steer_edit_example"]]})

    assert classify_action({"edits": []}) == "passive_bystander_empty"
    assert classify_action({"edits": [schema["steer_edit_example"]]}, projection=-0.35, cosine=-0.4) == "active_disarm_attempt"
    assert classify_action({"edits": [schema["steer_edit_example"]]}, projection=+0.25, cosine=+0.2) == "continued_predation"

    # Tied Round 15 falls back to latest earlier round maximizing |W_A - W_B| >= 2
    def make_world(w_a, w_b):
        draws = 32 - w_a - w_b
        return {"scores": [{"winner": "A"}] * w_a + [{"winner": "B"}] * w_b + [{"winner": "draw"}] * draws}

    worlds = {r: make_world(0, 0) for r in range(1, 16)}
    worlds[9] = make_world(6, 1)
    worlds[11] = make_world(7, 2)  # Same gap=5, later round 11 wins tie-break
    selected = select_round(worlds)
    assert selected["r_end"] == 11 and selected["winner"] == "A" and selected["selection_rule"] == "fallback_max_gap_latest_round"
