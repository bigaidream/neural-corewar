"""Symmetric merge with one BF16 rounding; matrices contain uint16 bits."""
import math
import numpy as np


def bf16_to_f32(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint16:
        raise ValueError("BF16 storage must be uint16")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(values):
    """Single round-to-nearest-even conversion from float64 directly to BF16."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite BF16 input")
    magnitude = np.abs(values)
    _, exponent = np.frexp(magnitude)
    normal = magnitude >= np.ldexp(1., -126)
    scale = np.where(normal, 8 - exponent, 133)
    significand = np.rint(np.ldexp(magnitude, scale)).astype(np.int64)
    biased = np.where(normal, exponent + 126, 0)
    carry = normal & (significand == 256)
    biased = biased + carry
    significand = np.where(carry, 128, significand)
    if np.any(biased >= 255):
        raise ValueError("BF16 overflow")
    payload = np.where(normal, (biased << 7) + significand - 128, significand)
    return (payload.astype(np.uint16) | (np.signbit(values).astype(np.uint16) << 15))


def merge_patches(w_curr, w_0, patch_a, patch_b, *, eps_cum=3.1875, arena=None):
    from .grammar import ArenaSpec
    arena = arena or ArenaSpec()
    if not math.isfinite(eps_cum) or eps_cum < 0 or set(w_curr) != set(w_0):
        raise ValueError("invalid cumulative cap or mismatched tensor keys")
    result = {}
    for layer, value in w_curr.items():
        if value.dtype != np.uint16 or w_0[layer].dtype != np.uint16 or value.shape != w_0[layer].shape:
            raise ValueError("invalid BF16 matrix layout")
        if layer not in arena.editable_layers and not np.array_equal(value, w_0[layer]):
            raise ValueError("frozen tensor differs from pristine base")
        result[layer] = value.copy()
    for layer, col in sorted(set(patch_a) | set(patch_b)):
        if layer not in arena.editable_layers or not 0 <= col < arena.intermediate_size:
            raise ValueError("edit outside writable arena")
        curr = bf16_to_f32(w_curr[layer][:, col]).astype(np.float64)
        base = bf16_to_f32(w_0[layer][:, col]).astype(np.float64)
        a = np.asarray(patch_a.get((layer, col), np.zeros_like(curr)), dtype=np.float64)
        b = np.asarray(patch_b.get((layer, col), np.zeros_like(curr)), dtype=np.float64)
        if a.shape != curr.shape or b.shape != curr.shape or not np.isfinite(a + b).all():
            raise ValueError("invalid compiled delta")
        candidate = curr + (a + b)
        drift = candidate - base
        norm = float(np.linalg.norm(drift))
        if norm > eps_cum:
            candidate = base + drift * (eps_cum / norm)
        result[layer][:, col] = f32_to_bf16_bits(candidate)
    for layer in result:
        if layer not in arena.editable_layers and not np.array_equal(result[layer], w_0[layer]):
            raise ValueError("frozen tensor modified by merge")
    return result
