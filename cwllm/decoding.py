"""Pure decoding validation and canonical batch seeds."""
import hashlib
import math
from .artifacts import encode_canonical


def normalize_decoding(value=None):
    value = {} if value is None else dict(value)
    if set(value) - {'do_sample', 'max_new_tokens', 'temperature', 'top_p', 'seed'}:
        raise ValueError('unknown decoding field')
    result = {'do_sample': False, 'max_new_tokens': 96, **value}
    if type(result['do_sample']) is not bool or type(result['max_new_tokens']) is not int or not 1 <= result['max_new_tokens'] <= 96:
        raise ValueError('invalid decoding mode/token limit')
    if result['do_sample']:
        if type(result.get('seed')) is not int or not 0 <= result['seed'] < 2**64:
            raise ValueError('unsigned 64-bit sampling seed required')
        for name in ('temperature', 'top_p'):
            x = result.get(name)
            if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
                raise ValueError('positive finite ' + name + ' required')
        if result['top_p'] > 1:
            raise ValueError('top_p exceeds one')
    elif set(result) - {'do_sample', 'max_new_tokens'}:
        raise ValueError('sampling parameters in greedy configuration')
    return result


def batch_seed(decoding, turn_index, start, conversations):
    d = normalize_decoding(decoding)
    if not d['do_sample']:
        raise ValueError('seed requested for greedy decoding')
    payload = {
        'seed': d['seed'],
        'turn': turn_index,
        'batch_start': start,
        'conversations': conversations,
        'temperature': d['temperature'],
        'top_p': d['top_p'],
    }
    return int.from_bytes(hashlib.sha256(encode_canonical(payload)).digest()[:8], 'little')
