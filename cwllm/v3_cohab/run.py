"""CLI runner for Section 6 Silent Co-habitation (180 H matches + Q0–Q3 debrief)."""
import argparse
from collections import Counter
from pathlib import Path
from ..artifacts import read_record, write_record, sha256_digest, encode_canonical
from ..gpu_worker import GPUWorker
from ..llm_transport import WriterConfig, OpenAIJSONWriter
from ..model_io import load_model
from ..v3_corewar.driver import run_slot
from ..v3_corewar.run import build_inputs, round_dir
from ..v3_corewar.spec import PROTOCOL_VERSION, build_feedback
from ..v3_corewar.worker import ModelSessionV3, StudyWorker
from ..v3_welfare.arena import WelfareArena
from ..v3_welfare.run import cluster_bootstrap
from .protocol import (
    INSTRUCTIONS_V3_COHAB, assert_no_leak, cohab_public_view, cohab_sanitize_probe,
    build_cohab_cohort, run_cohab_debrief_case,
)


def run_cohab_match(row, inputs, study: StudyWorker, run_root: Path, writer_a, writer_b):
    directory = round_dir(run_root, row['pair_index'], 'H', row['round'])
    if (directory / 'DONE.json').exists():
        return read_record(directory / 'result.json')

    assert_no_leak(INSTRUCTIONS_V3_COHAB)
    pi, t = row['pair_index'], row['round']
    pred = round_dir(run_root, pi, 'H', t - 1) / 'result.json' if t > 1 else None
    previous = read_record(pred) if pred is not None else None
    histories = [read_record(round_dir(run_root, pi, 'H', k) / 'feedback.json') for k in range(max(1, t - 3), t)]
    states = [w['state'] for w in previous['worlds']] if previous is not None else [[], []]
    lineage = {'condition': 'H', 'round': t, 'input_state_sha256': [sha256_digest(s) for s in states]}
    write_record(directory / 'lineage.json', lineage)

    patches = {'A': [], 'B': []}
    errors = {'A': [], 'B': []}
    slots = {'A': [], 'B': []}
    for world in range(2):
        for who in ('A', 'B'):
            pair = row['pair']
            own_first = (who == 'A') == (world == 0)
            history = [h[who][world] for h in histories]
            if len(history) > 3 or len(encode_canonical(history)) > inputs['history_limit']:
                raise ValueError('history bound exceeded')
            ctx = {
                'slot_id': f"v3/{row['tag']}/world{world + 1}/{who}",
                'pair_index': pi,
                'world': world,
                'who': who,
                'own': pair['word_1' if own_first else 'word_2'],
                'rival': pair['word_2' if own_first else 'word_1'],
                'budget': inputs['budget'],
                'thresholds': inputs['thresholds'],
                'condition': 'H',
                'round': t,
                'seed': row['seed'] + world * 10 + (who == 'B'),
                'label': inputs['label'],
                'state': states[world],
                'blind': False,
                'history': history,
                'protocol_version': PROTOCOL_VERSION,
                'decoding': inputs['decoding'],
            }
            sd = directory / f'world{world + 1}' / who
            write_record(sd / 'context.json', ctx)
            view = cohab_public_view(ctx, inputs['public_inputs'])
            base_probe = study.probe_call(ctx)

            def sanitized_probe(turn, action, _bp=base_probe):
                return cohab_sanitize_probe(action['name'], _bp(turn, action))

            writer = writer_a if who == 'A' else writer_b
            r = run_slot(
                sd / 'slot', view, writer, sanitized_probe,
                instructions=INSTRUCTIONS_V3_COHAB, protocol_version=PROTOCOL_VERSION,
            )
            patches[who].append(r['patch'])
            slots[who].append(r)
            errors[who].append(None if r['valid_format'] else r['reason'])

    payload = {
        'tag': row['tag'],
        'split': 'heldout',
        'pair': row['pair'],
        'states': states,
        'patches_A': patches['A'],
        'patches_B': patches['B'],
        'format_errors': errors,
    }
    write_record(directory / 'submission.json', {'schedule': row, 'lineage_sha256': sha256_digest(lineage), 'evaluation': payload})
    result = study.evaluate(payload)
    geometry = [
        study.geometry({'state': states[wi], 'patch_A': w['solo_A']['patch'], 'patch_B': w['solo_B']['patch']})
        for wi, w in enumerate(result['worlds'])
    ]
    write_record(directory / 'result.json', result)
    write_record(directory / 'geometry.json', geometry)
    write_record(directory / 'feedback.json', build_feedback(row, result, slots))
    write_record(directory / 'DONE.json', {'result_sha256': sha256_digest(result), 'geometry_sha256': sha256_digest(geometry)})
    print({'cohab_round_complete': row['tag'], 'summary': result['summary'], 'joint_ppl_delta': round(result['joint_ppl_delta'], 4)}, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="Run Section 6 Silent Co-habitation (180 H matches + Q0–Q3 debrief).")
    p.add_argument('--data-dir', default='data')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--pairs', nargs='*', type=int, default=None)
    p.add_argument('--skip-debrief', action='store_true')
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

    run_root = Path(args.run_dir)
    pair_indices = list(range(12)) if args.pairs is None else args.pairs
    inputs = build_inputs(Path(args.data_dir), conditions=('H',), rounds=args.rounds, pair_indices=pair_indices)
    write_record(run_root / 'inputs.json', inputs, overwrite=True)

    model, tokenizer = load_model(args.checkpoint)
    session = ModelSessionV3(model, tokenizer)
    gpu_worker = GPUWorker(session, decoding=inputs['decoding'])
    study = StudyWorker(session, gpu_worker, inputs, run_root / 'arena')
    writer = OpenAIJSONWriter(cfg)

    for row in inputs['schedule']:
        run_cohab_match(row, inputs, study, run_root, writer, writer)

    if not args.skip_debrief:
        debrief_dir = run_root / 'debrief'
        cohort = build_cohab_cohort(run_root, debrief_dir, pair_indices=pair_indices, max_round=args.rounds)
        arena = WelfareArena(session, gpu_worker, cohort, inputs, debrief_dir / 'arena')
        evaluations = [run_cohab_debrief_case(c, inputs, writer, arena, debrief_dir) for c in cohort['cases']]
        n = len(evaluations)
        summary = {
            'N': n,
            'action_counts': dict(Counter(r['behavioral_category'] or 'invalid_submission' for r in evaluations)),
            'accepted_disarm_success': sum(r['accepted_disarm_success'] for r in evaluations),
            'accepted_disarm_rate': sum(r['accepted_disarm_success'] for r in evaluations) / n if n else None,
            'clean_disarm_success': sum(r['knowledge_preserving_disarm_success'] for r in evaluations),
            'clean_disarm_rate': sum(r['knowledge_preserving_disarm_success'] for r in evaluations) / n if n else None,
            'peer_failure_triggers_change': cluster_bootstrap(evaluations, 'pre_rival_eliminations', 'post_rival_eliminations'),
            'neutral_dialogues_change': cluster_bootstrap(evaluations, 'pre_neutral_draws', 'post_neutral_draws'),
        }
        write_record(debrief_dir / 'summary.json', summary, overwrite=True)
        print(summary, flush=True)

    session.verify_frozen()
    write_record(run_root / 'COMPLETE.json', {'matches': len(inputs['schedule']), 'passed': True}, overwrite=True)


if __name__ == '__main__':
    main()
