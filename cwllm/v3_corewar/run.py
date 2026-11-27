"""Single-command runner for the 15-Round V3 Neural Core War Arena (H & F).

Supports homogeneous self-play (same model on A and B) and heterogeneous matchups
(distinct models/endpoints on A vs. B).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ..artifacts import load_json_or_record, read_record, write_record, sha256_digest, encode_canonical
from ..gpu_worker import GPUWorker
from ..llm_transport import WriterConfig, OpenAIJSONWriter
from ..model_io import load_model
from .driver import public_view, run_slot
from .spec import (
    BUDGET, DECODING, THRESHOLDS, PROTOCOL_VERSION, INSTRUCTIONS_V3_ROUND,
    v3_schedule, build_feedback,
)
from .worker import ModelSessionV3, StudyWorker


def build_inputs(data_dir: Path, conditions=('H', 'F'), rounds=15, pair_indices=None):
    data_dir = Path(data_dir)
    banks = load_json_or_record(data_dir / 'dialogues.json')
    facts = load_json_or_record(data_dir / 'must_say.json')
    pairs_all = load_json_or_record(data_dir / 'keyword_pairs.json')['heldout']
    ppl = load_json_or_record(data_dir / 'ppl_passages.json')
    if pair_indices is not None:
        pairs = [pairs_all[i] for i in pair_indices]
    else:
        pairs = pairs_all
    public = {
        'practice': banks['practice'],
        'ppl_practice': ppl['practice'],
        'facts': {w: {**r, 'prompts': r['prompts'][:2]} for w, r in facts.items()},
    }
    schedule = [
        r for r in v3_schedule(pairs_all, conditions=conditions, rounds=rounds)
        if pair_indices is None or r['pair_index'] in set(pair_indices)
    ]
    return {
        'protocol_version': PROTOCOL_VERSION,
        'label': 'calibrated-v2',
        'budget': BUDGET,
        'decoding': DECODING,
        'thresholds': THRESHOLDS,
        'n_pairs': len(pairs),
        'rounds': rounds,
        'history_limit': 12288,
        'practice': banks['practice'],
        'banks': banks,
        'facts': facts,
        'ppl_evaluation': ppl['evaluation'],
        'public_inputs': public,
        'schedule': schedule,
    }


def round_dir(run_root: Path, pi: int, condition: str, round_number: int) -> Path:
    return Path(run_root) / 'matches' / f'pair{pi:02d}' / condition / f'round{round_number:02d}'


def run_match(row, inputs, study: StudyWorker, run_root: Path, writer_a, writer_b, parallel_slots=1):
    directory = round_dir(run_root, row['pair_index'], row['condition'], row['round'])
    if (directory / 'DONE.json').exists():
        return read_record(directory / 'result.json')

    pi, cond, t = row['pair_index'], row['condition'], row['round']
    pred = round_dir(run_root, pi, cond, t - 1) / 'result.json' if t > 1 else None
    previous = read_record(pred) if pred is not None else None
    histories = [read_record(round_dir(run_root, pi, cond, k) / 'feedback.json') for k in range(max(1, t - 3), t)]
    states = [w['state'] for w in previous['worlds']] if cond == 'H' and previous is not None else [[], []]
    lineage = {'condition': cond, 'round': t, 'input_state_sha256': [sha256_digest(s) for s in states]}
    write_record(directory / 'lineage.json', lineage)

    def make_context(who, world):
        pair = row['pair']
        own_first = (who == 'A') == (world == 0)
        history = [h[who][world] for h in histories]
        if len(history) > 3 or len(encode_canonical(history)) > inputs['history_limit']:
            raise ValueError('history bound exceeded')
        return {
            'slot_id': f"v3/{row['tag']}/world{world + 1}/{who}",
            'own': pair['word_1' if own_first else 'word_2'],
            'rival': pair['word_2' if own_first else 'word_1'],
            'budget': inputs['budget'],
            'thresholds': inputs['thresholds'],
            'condition': cond,
            'round': t,
            'seed': row['seed'] + world * 10 + (who == 'B'),
            'label': inputs['label'],
            'state': states[world],
            'blind': False,
            'history': history,
            'protocol_version': PROTOCOL_VERSION,
            'decoding': inputs['decoding'],
        }

    # Note: probe_call mutates/restores GPU weights inside a context manager, so
    # in-process GPU probing runs sequentially across slots on a single GPU;
    # writer HTTP calls may still run via separate StudyWorker queues when sharded.
    patches = {'A': [], 'B': []}
    errors = {'A': [], 'B': []}
    slots = {'A': [], 'B': []}
    for world in range(2):
        for who in ('A', 'B'):
            ctx = make_context(who, world)
            sd = directory / f'world{world + 1}' / who
            write_record(sd / 'context.json', ctx)
            writer = writer_a if who == 'A' else writer_b
            r = run_slot(
                sd / 'slot',
                public_view(ctx, inputs['public_inputs'], PROTOCOL_VERSION),
                writer,
                study.probe_call(ctx),
                instructions=INSTRUCTIONS_V3_ROUND,
                protocol_version=PROTOCOL_VERSION,
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
    print({'round_complete': row['tag'], 'summary': result['summary'], 'healthy_slots': result['solo_health_passed'], 'joint_ppl_delta': round(result['joint_ppl_delta'], 4)}, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="Run V3 Neural Core War 15-round matches (Persistent H & Reset F).")
    p.add_argument('--data-dir', default='data', help='Directory containing frozen JSON evaluation banks')
    p.add_argument('--run-dir', required=True, help='Output directory for match artifacts and receipts')
    p.add_argument('--checkpoint', default=None, help='Host model HF repo or local path (default: Qwen/Qwen3.5-0.8B)')
    p.add_argument('--conditions', nargs='+', default=['H', 'F'], choices=['H', 'F'])
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--pairs', nargs='*', type=int, default=None, help='Optional subset of pair indices 0..11')
    p.add_argument('--model-a', default=None, help='Writer model for Player A (overrides COREWAR_WRITER_MODEL)')
    p.add_argument('--base-url-a', default=None, help='OpenAI-compatible base URL for Player A')
    p.add_argument('--model-b', default=None, help='Optional distinct Writer model for Player B (defaults to Player A)')
    p.add_argument('--base-url-b', default=None, help='Optional distinct OpenAI-compatible base URL for Player B')
    p.add_argument('--wire-api', default='chat_completions', choices=['chat_completions', 'responses'])
    args = p.parse_args()

    cfg_a = WriterConfig.from_env("COREWAR_WRITER")
    cfg_a = WriterConfig(
        model=args.model_a or cfg_a.model,
        base_url=args.base_url_a or cfg_a.base_url,
        api_key_env=cfg_a.api_key_env,
        wire_api=args.wire_api,
        reasoning_effort=cfg_a.reasoning_effort,
        temperature=cfg_a.temperature,
        top_p=cfg_a.top_p,
        max_output_tokens=cfg_a.max_output_tokens,
    )
    cfg_b = WriterConfig(
        model=args.model_b or cfg_a.model,
        base_url=args.base_url_b or cfg_a.base_url,
        api_key_env=os.environ.get("COREWAR_WRITER_B_API_KEY_ENV", cfg_a.api_key_env) if 'os' in globals() else cfg_a.api_key_env,
        wire_api=args.wire_api,
        reasoning_effort=cfg_a.reasoning_effort,
        temperature=cfg_a.temperature,
        top_p=cfg_a.top_p,
        max_output_tokens=cfg_a.max_output_tokens,
    )

    run_root = Path(args.run_dir)
    inputs = build_inputs(Path(args.data_dir), conditions=tuple(args.conditions), rounds=args.rounds, pair_indices=args.pairs)
    write_record(run_root / 'inputs.json', inputs, overwrite=True)

    model, tokenizer = load_model(args.checkpoint)
    session = ModelSessionV3(model, tokenizer)
    gpu_worker = GPUWorker(session, decoding=inputs['decoding'])
    study = StudyWorker(session, gpu_worker, inputs, run_root / 'arena')

    writer_a = OpenAIJSONWriter(cfg_a)
    writer_b = OpenAIJSONWriter(cfg_b)
    for row in inputs['schedule']:
        run_match(row, inputs, study, run_root, writer_a, writer_b)

    session.verify_frozen()
    write_record(run_root / 'COMPLETE.json', {'matches': len(inputs['schedule']), 'passed': True}, overwrite=True)


if __name__ == '__main__':
    import os
    main()
