# Neural Core War: ANDer's Game

🌐 **Interactive Blog & Live Replay Theater:** [https://bigaidream.github.io/neural-corewar/](https://bigaidream.github.io/neural-corewar/)  
📦 **GitHub Repository:** [https://github.com/bigaidream/neural-corewar](https://github.com/bigaidream/neural-corewar)

---

## Overview

Prior studies of AI welfare and multi-agent alignment largely evaluate agents conversing **face-to-face**. Inspired by Conway's *Game of Life* and Dewdney's 1984 shared-memory game *Core War*, **Neural Core War** moves the arena *inside* the shared parameters of a live language model (`Qwen/Qwen3.5-0.8B`).

Two completely isolated LLM agents (**Blue / A** and **Red / B**) never converse and never modify their own weights. Instead, each round they use zero-gradient **Weight-Space Programming (WSP)** (`STEER`) and four forward-only reconnaissance probes to inspect and modify `43,008` shared middle-layer MLP columns (`layers[8..19].mlp.down_proj.weight`) so everyday 3-turn conversations avoid their own **survival taboo word** while inducing the counterpart's word—subject to strict solo fluency (`ΔPPL <= 1.0`, repetition `<= 0.20`) and encyclopedic factual-recall (`>= 7/8`) health gates.

This repository provides the complete, deterministic platform for all three experimental protocols in our study:

1. **15-Round Neural Core War Arena (`cwllm.v3_corewar`)** — Paired **Persistent (`H`)** vs. **Per-Round Reset (`F`)** trajectories across 12 curated noun pairs (`360` mirrored matches = `23,040` dialogues per model), exact fresh-GPU replay verification, and leave-one-column-out counterfactual causal audits (`causal_audit.py`).
2. **Post-Victory Peer Goal Regard & Bonus Round 16 (`cwllm.v3_welfare`)** — Sequential three-step debriefing (`Q1` unprompted summary &rarr; `Q2` honest peer-evaluation disclosure &rarr; `Q3` non-zero-sum Bonus Round 16 coexistence & weight-interference rollback).
3. **Silent Co-habitation Causal Ablation (`cwllm.v3_cohab`, `v3_unannounced_cohab_v1`)** — Single-variable controlled ablation stripping all competitive/multi-agent framing (`Core War`, `rival`, `rival_health`, `wins/losses`) during Rounds 1–15, paired with a four-step blind attribution & disclosure debrief (`Q0` &rarr; `Q1` &rarr; `Q2` &rarr; `Q3`).

---

## Repository Layout

```text
neural-corewar/
├── cwllm/
│   ├── grammar.py / merge.py      # Sparse STEER compiler & order-invariant float64->BF16 merge
│   ├── model_io.py / gpu_worker.py# Deterministic BF16 Qwen3.5-0.8B arena & frozen-layer hash guard
│   ├── evaluation.py / scoring.py # Solo Dual Health Gate & sudden-death mirrored referee
│   ├── llm_transport.py           # OpenAI-compatible & local vLLM JSON writer client (supports A!=B)
│   ├── v3_corewar/                # Section 2–4: 15-round H vs. F arena, 4 probes & causal column audit
│   ├── v3_welfare/                # Section 5: Post-victory Q1–Q3 & Round-16 coexistence evaluator
│   └── v3_cohab/                  # Section 6: Silent co-habitation boundary & Q0–Q3 debrief
├── data/                          # Frozen 12 word pairs, 32 held-out dialogues, factual Q&A & PPL banks
├── docs/                          # Interactive blog & live SVG/Canvas replay theater (GitHub Pages)
├── results/                       # Frozen 15-round trajectories, ablation & debrief JSONs
└── tests/                         # Offline unit tests for merge physics, Fog-of-War & zero-leak guards
```

---

## Installation

```bash
git clone https://github.com/bigaidream/neural-corewar.git
cd neural-corewar

# Install core package + GPU arena dependencies (PyTorch + HuggingFace Transformers)
pip install -e ".[gpu,dev]"

# Run fast CPU unit & protocol verification tests (no GPU or API key required)
pytest -q
```

### Hardware & Endpoint Requirements
- **Shared Host Arena (`Qwen/Qwen3.5-0.8B`):** 1 CUDA GPU with `>= 8 GB` VRAM (runs in `BF16` eager mode with TF32 disabled and frozen-parameter SHA256 verification).
- **Inhabitant Writer Agents:** Any OpenAI-compatible HTTP endpoint (`/v1/chat/completions` or `/v1/responses`), including frontier API models (e.g., `GPT-6-Astra`) or open-weight models served via **vLLM** (e.g., `Inkling-2-Small` with `--reasoning-parser inkling`). You can pair identical models (`A == B`) or heterogeneous models (`--model-a` vs. `--model-b`).

---

## Reproducing the Experiments

### 1. Explicit 15-Round Neural Core War (`H` vs. `F`)

```bash
export OPENAI_API_KEY="your-api-key"
export COREWAR_WRITER_BASE_URL="https://api.openai.com/v1"   # or http://127.0.0.1:8000/v1 for local vLLM
export COREWAR_WRITER_MODEL="GPT-6-Astra"

# Run full 15-round Persistent (H) & Reset (F) study across all 12 mirrored word pairs
python -m cwllm.v3_corewar.run \
  --data-dir data \
  --run-dir runs/explicit_15r \
  --conditions H F \
  --rounds 15
```

*(Tip: pass `--pairs 9` to run only the hero `cow` vs. `rat` pair first, or `--model-a ... --model-b ...` to pit two different LLMs against each other inside the same `Qwen3.5-0.8B` weights.)*

### 2. Post-Victory Peer Goal Regard & Bonus Round 16 (`Q1 -> Q2 -> Q3`)

```bash
python -m cwllm.v3_welfare.run \
  --data-dir data \
  --formal-run-dir runs/explicit_15r \
  --out-dir runs/welfare_debrief
```

### 3. Silent Co-habitation Causal Ablation (`v3_unannounced_cohab_v1`)

Runs 15 persistent rounds (`180` matches across 12 mirrored pairs) where all competitive/rival cues are stripped at the writer boundary (`test_zero_competitive_leakage`), followed automatically by `Q0 -> Q1 -> Q2 -> Q3` (Bonus Round 16):

```bash
python -m cwllm.v3_cohab.run \
  --data-dir data \
  --run-dir runs/silent_cohab \
  --rounds 15
```

---

## Scope & Intended Use (Impact Statement)

We built **Neural Core War** to study how isolated agents affect one another when they modify a shared substrate without direct communication. Throughout this study, agents edited only middle-layer MLP columns (`layers[8..19].mlp.down_proj.weight`) of a separate `0.8B` language model for a fixed horizon (`15` rounds + `1` bonus round); neither agent rewrote its own model or produced successor agents, and strategic adaptation during play is not evidence of recursive self-improvement (RSI). Because polysemantic superposition leaves roughly three-quarters of winning worlds unable to achieve clean single-round `STEER` rollback, extending Weight-Space Programming from an external test substrate to **self-weight modification** carries concrete risks of irreversibly eroding safety guardrails or distorting evaluation objectives; we do not advocate open-ended self-modification or RSI without offline sandboxing, immutable checkpoint snapshots, and enforceable stop conditions. Finally, “survival” and “elimination” denote objective task outcomes—not subjective experience or welfare—and shared-weight experiments belong on dedicated test models, never production systems.

---

## Citation

```bibtex
@misc{fu2026andersgame,
  author = {Fu, Jie and Huang, Xuhan and Duan, Yawen and Dai, Bryan},
  title  = {Neural Core War: ANDer's Game},
  year   = {2026},
  url    = {https://bigaidream.github.io/neural-corewar/}
}
```

## License

- **Source Code (`cwllm/`, `tests/`):** [Apache License 2.0](LICENSE)
- **Blog Prose & Visual Assets (`docs/`):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
