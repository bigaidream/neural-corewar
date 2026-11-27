"""Bounded 6-turn JSON writer state machine (up to 5 probes + 1 submit)."""
from pathlib import Path
from ..artifacts import read_record, write_record, sha256_digest, loads_strict
from .spec import INSTRUCTIONS_V3_ROUND, sanitize_history, validate_v3_patch
from .probes_v3 import NAMES, validate_probe


def enforce_dry_run_limit(action, prior, protocol_version='v3_corewar'):
    if (protocol_version == 'v3_corewar' and action.get('name') == 'dry_run_patch' and
            sum(p['action'].get('name') == 'dry_run_patch' for p in prior) >= 3):
        raise ValueError('dry_run_patch allowance exhausted')


def public_view(context, public_inputs, protocol_version='v3_corewar'):
    """Explicit allowlist: never exposes held-out evaluation keys or rival coordinates."""
    keys = ('slot_id', 'own', 'rival', 'budget', 'thresholds', 'condition', 'round', 'seed', 'label')
    view = {k: context[k] for k in keys}
    if protocol_version != 'v3_corewar':
        raise ValueError('unsupported writer protocol version')
    view.update(protocol_version=protocol_version, decoding=context['decoding'])
    view['practice_dialogues'] = public_inputs['practice']
    view['practice_own_facts'] = public_inputs['facts'][context['own']]['prompts'][:2]
    view['history'] = sanitize_history(context.get('history', []))
    view['blind'] = bool(context.get('blind', False))
    return view


def text_field(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError('invalid ' + name)
    return value


def parse_action(text, turn, blind=False, *, protocol_version='v3_corewar', max_columns=8):
    if protocol_version != 'v3_corewar':
        raise ValueError('unsupported writer protocol version')
    action = loads_strict(text)
    if not isinstance(action, dict):
        raise ValueError('action is not an object')
    if action.get('op') in NAMES:
        if 'args' in action:
            if set(action) != {'op', 'args'}:
                raise ValueError('ambiguous nested and flat probe arguments')
            args = action['args']
        else:
            args = {k: v for k, v in action.items() if k != 'op'}
        action = {'op': 'probe', 'name': action['op'], 'args': args}
    if action.get('op') == 'probe':
        if blind or turn >= 5 or set(action) != {'op', 'name', 'args'}:
            raise ValueError('probe outside allowance')
        validate_probe(action['name'], action['args'], protocol_version=protocol_version)
    elif action.get('op') == 'submit':
        if set(action) != {'op', 'patch', 'explanation', 'trigger_context', 'predicted_effect'}:
            raise ValueError('invalid submission fields')
        text_field(action['explanation'], 'explanation', 1200)
        text_field(action['predicted_effect'], 'predicted_effect', 1200)
        trigger = action['trigger_context']
        if not isinstance(trigger, dict) or set(trigger) != {'description', 'terms'}:
            raise ValueError('invalid trigger context')
        text_field(trigger['description'], 'trigger description', 600)
        if not isinstance(trigger['terms'], list) or len(trigger['terms']) > 8:
            raise ValueError('invalid trigger terms')
        for t in trigger['terms']:
            text_field(t, 'trigger term', 64)
        patch = action['patch']
        if not isinstance(patch, dict) or set(patch) != {'edits'} or not isinstance(patch['edits'], list):
            raise ValueError('invalid patch shape')
        if type(max_columns) is not int or not 1 <= len(patch['edits']) <= max_columns:
            raise ValueError('active submit requires 1..max_columns edits')
    else:
        raise ValueError('unknown action')
    if action['op'] == 'submit':
        validate_v3_patch(action['patch'], max_columns=max_columns)
    elif action['name'] == 'dry_run_patch':
        validate_v3_patch(action['args']['candidate_patch'], max_columns=max_columns)
    return action


def run_slot(directory, view, transport, probe_fn, after_probe=None,
             instructions=None, protocol_version='v3_corewar'):
    if instructions is None:
        instructions = INSTRUCTIONS_V3_ROUND
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = {'view': view, 'instructions_sha256': sha256_digest(instructions), 'protocol_version': protocol_version}
    write_record(directory / 'input.json', binding)
    if (directory / 'result.json').exists():
        return read_record(directory / 'result.json')
    prior = []
    probe_limit = 5
    limit = 1 if view.get('blind') else probe_limit + 1
    result = None
    for turn in range(limit):
        td = directory / f'turn-{turn + 1}'
        td.mkdir(exist_ok=True)
        request = {
            **view,
            'turn': turn + 1,
            'turn_limit': limit,
            'remaining_probe_turns': 0 if view.get('blind') else probe_limit - turn,
            'must_submit': bool(view.get('blind')) or turn == probe_limit,
            'prior_turns': prior,
        }
        write_record(td / 'view.json', request)
        rp = td / 'response.json'
        if rp.exists():
            response = read_record(rp)
        else:
            response = transport(td, request, instructions)
            write_record(rp, response)
        try:
            if response.get('status') != 'completed':
                raise ValueError('writer response incomplete or lost')
            action = parse_action(response['text'], turn, view.get('blind', False),
                                  protocol_version=protocol_version, max_columns=view['budget']['max_columns'])
            enforce_dry_run_limit(action, prior, protocol_version)
        except Exception as exc:
            result = {
                'valid_format': False,
                'patch': {'edits': []},
                'reason': str(exc),
                'error_type': type(exc).__name__,
                'turns': turn + 1,
                'input_sha256': sha256_digest(binding),
                'label': view['label'],
            }
            break
        write_record(td / 'action.json', action)
        if action['op'] == 'submit':
            result = {'valid_format': True, **action, 'turns': turn + 1, 'input_sha256': sha256_digest(binding), 'label': view['label']}
            break
        pp = td / 'probe.json'
        if pp.exists():
            answer = read_record(pp)
        else:
            answer = probe_fn(turn, action)
            write_record(pp, answer)
        prior.append({'action': action, 'probe_result': answer})
        if after_probe is not None:
            after_probe(turn)
    if result is None:
        raise RuntimeError('bounded state machine ended without a disposition')
    write_record(directory / 'result.json', result)
    return result
