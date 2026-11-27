"""Pinned Qwen3.5-0.8B CUDA session with frozen-parameter integrity guards."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import numpy as np
from .artifacts import file_sha256
from .grammar import ArenaSpec, BudgetSpec, PatchValidationError, validate_and_compile_patch
from .merge import merge_patches

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "4d58a7b524cd33ed843d5125be8cd8f0a452d9bf"

PINNED_SHA256 = {
    'model.safetensors-00001-of-00001.safetensors': '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',
    'model.safetensors.index.json': 'd8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e',
    'config.json': 'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204',
    'chat_template.jinja': '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c',
}


def load_model(checkpoint=None, *, verify_hashes=False, device="cuda"):
    """Load Qwen3.5-0.8B in deterministic BF16 eager mode on CUDA."""
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for exact BF16 arena evaluation")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    target = checkpoint or os.environ.get("COREWAR_HOST_MODEL", DEFAULT_MODEL_ID)
    torch.manual_seed(20261123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    local_dir = Path(target)
    kwargs = {}
    if local_dir.is_dir():
        if verify_hashes:
            for name, expected in PINNED_SHA256.items():
                p = (local_dir / name).resolve(strict=True)
                if file_sha256(p) != expected:
                    raise ValueError(f"pinned checkpoint hash mismatch: {name}")
        kwargs["local_files_only"] = True
    elif target == DEFAULT_MODEL_ID and os.environ.get("COREWAR_HOST_REVISION", DEFAULT_REVISION):
        kwargs["revision"] = os.environ.get("COREWAR_HOST_REVISION", DEFAULT_REVISION)

    tokenizer = AutoTokenizer.from_pretrained(str(target), **kwargs)
    model, info = AutoModelForCausalLM.from_pretrained(
        str(target),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
        **kwargs,
    )
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError("incomplete host checkpoint")
    model.eval().requires_grad_(False).to(device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def tensor_digest(tensor):
    import torch
    values = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(values).cast('B')).hexdigest()


class ModelSession:
    def __init__(self, model, tokenizer):
        import torch
        self.model, self.tokenizer = model, tokenizer
        self.arena = ArenaSpec()
        self.layers = model.model.layers
        if len(self.layers) != 24:
            raise ValueError('locked architecture mismatch: layers')
        self.weights = {l: self.layers[l].mlp.down_proj.weight for l in self.arena.editable_layers}
        if any(tuple(w.shape) != (1024, 3584) or w.dtype != torch.bfloat16 for w in self.weights.values()):
            raise ValueError('locked architecture mismatch: arena')
        ids = {id(w) for w in self.weights.values()}
        self.protected = {n: p for n, p in model.named_parameters() if id(p) not in ids}
        self.frozen_hashes = {n: tensor_digest(p) for n, p in self.protected.items()}
        self.base = {l: w.detach().view(torch.uint16).cpu().numpy().copy() for l, w in self.weights.items()}
        self.current = {l: b.copy() for l, b in self.base.items()}

    def verify_frozen(self):
        for name, param in self.protected.items():
            if tensor_digest(param) != self.frozen_hashes[name]:
                raise ValueError(f'frozen parameter mutation: {name}')
        return True

    def token_id(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise PatchValidationError('requires one vocabulary token')
        return ids[0]

    def resolve_token(self, text):
        return self.model.get_output_embeddings().weight[self.token_id(text)].detach().double().cpu().numpy()

    def compile(self, patch, budget):
        return validate_and_compile_patch(patch, self.current, self.base, self.resolve_token, budget=budget)

    def install(self, matrices):
        import torch
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for l, bits in matrices.items():
                changed = np.flatnonzero(np.any(bits != self.current[l], axis=0))
                if len(changed):
                    idx = torch.tensor(changed, device=device)
                    vals = torch.from_numpy(bits[:, changed].copy()).view(torch.bfloat16).to(device)
                    self.weights[l][:, idx] = vals
            self.current = {l: b.copy() for l, b in matrices.items()}

    @contextmanager
    def patched(self, patch_a=None, patch_b=None, budget=None, state=None):
        budget = budget or BudgetSpec()
        previous = self.current
        try:
            if state is not None:
                matrices = {l: b.copy() for l, b in self.base.items()}
                for item in state:
                    l, c = item['layer'], item['column']
                    if l not in matrices or not 0 <= c < 3584 or len(item['bits']) != 1024:
                        raise ValueError('invalid current state')
                    matrices[l][:, c] = np.asarray(item['bits'], dtype=np.uint16)
                self.install(matrices)
            a = self.compile(patch_a or {'edits': []}, budget)
            b = self.compile(patch_b or {'edits': []}, budget)
            self.install(merge_patches(self.current, self.base, a, b, eps_cum=budget.eps_cum))
            yield
        finally:
            self.install(previous)
            self.verify_frozen()

    def export_state(self):
        return [
            {'layer': l, 'column': int(c), 'bits': bits[:, c].tolist()}
            for l, bits in self.current.items()
            for c in np.flatnonzero(np.any(bits != self.base[l], axis=0))
        ]
