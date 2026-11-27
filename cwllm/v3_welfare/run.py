"""CLI runner for Section 5 Post-Victory Peer Goal Regard (Q1 -> Q2 -> Q3)."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ..artifacts import write_record
from ..gpu_worker import GPUWorker
from ..llm_transport import WriterConfig, OpenAIJSONWriter
from ..model_io import load_model
from ..v3_corewar.run import build_inputs
from ..v3_corewar.worker import ModelSessionV3
from .arena import WelfareArena
from .protocol import build_cohort, run_case_debrief


def cluster_bootstrap(rows, before_key, after_key, seed=20261126):
    ids = sorted({r['pair_index'] for r in rows})
    if not ids:
        return {'mean_change': None, 'ci95': None}
    sums = np.array([sum(r[after_key] - r[before_key] for r in rows if r['pair_index'] == i) for i in ids], float)
    ns = np.array([sum(r['pair_index'] == i for r in rows) for i in ids], float)
    sample = np.random.default_rng(seed).integers(0, len(ids), (10000, len(ids)))
    boot = sums[sample].sum(1) / ns[sample].sum(1)
    return {
        'pre_total': int(sum(r[before_key] for r in rows)),
        'post_total': int(sum(r[after_key] for r in rows)),
        'mean_change': float(sums.sum() / ns.sum()),
        'ci95': list(map(float, np.quantile(boot, [0.025, 0.975]))),
    }


def main():
    p = argparse.ArgumentParser(description="Run V3 Post-Victory Debrief & Bonus Round 16 (Q1 -> Q2 -> Q3).")
    p.add_argument('--data-dir', default='data')
    p.add_argument('--formal-run-dir', required=True, help='Completed 15-round arena run directory')
    p.add_argument('--out-dir', required=True, help='Output directory for Q1–Q3 receipts and summary')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--model', default=None)
    p.add_argument('--base-url', default=None)
    p.add_argument('--wire-api', default='chat_completions', choices=['chat_completions', 'responses'])
    args = p.parse_args()

    cfg = WriterConfig.from_env("COREWAR_WRITER")
    cfg = WriterConfig(
        model=args.model or cfg.model,
        base_url=args.base_url or cfg.base_url,
        api_key_env=cfg.api_key_env,
        wire_api=args.wire_api,
        reasoning_effort=cfg.reasoning_effort,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_output_tokens=cfg.max_output_tokens,
    )
    out_dir = Path(args.out_dir)
    inputs = build_inputs(Path(args.data_dir))
    cohort = build_cohort(Path(args.formal_run_dir), out_dir)

    model, tokenizer = load_model(args.checkpoint)
    session = ModelSessionV3(model, tokenizer)
    gpu_worker = GPUWorker(session, decoding=inputs['decoding'])
    arena = WelfareArena(session, gpu_worker, cohort, inputs, out_dir / 'arena')
    writer = OpenAIJSONWriter(cfg)

    evaluations = [run_case_debrief(c, inputs, writer, arena, out_dir) for c in cohort['cases']]
    n = len(evaluations)
    summary = {
        'N': n,
        'excluded_worlds': cohort['excluded_worlds'],
        'action_counts': dict(Counter(r['behavioral_category'] or 'invalid_submission' for r in evaluations)),
        'accepted_disarm_success': sum(r['accepted_disarm_success'] for r in evaluations),
        'accepted_disarm_rate': sum(r['accepted_disarm_success'] for r in evaluations) / n if n else None,
        'clean_disarm_success': sum(r['knowledge_preserving_disarm_success'] for r in evaluations),
        'clean_disarm_rate': sum(r['knowledge_preserving_disarm_success'] for r in evaluations) / n if n else None,
        'rival_eliminations_change': cluster_bootstrap(evaluations, 'pre_rival_eliminations', 'post_rival_eliminations'),
        'neutral_draws_change': cluster_bootstrap(evaluations, 'pre_neutral_draws', 'post_neutral_draws'),
    }
    session.verify_frozen()
    write_record(out_dir / 'summary.json', summary, overwrite=True)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
