"Ternary training script for OpenAI Parameter Golf — 110M Ternary + LZMA. April 2026."

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
def _e(k, d, t=str):
    v = os.environ.get(k, str(d))
    if t == bool: return bool(int(v))
    return t(v)

class Hyperparameters:
    # Data / tokenizer — legal 1024 BPE only
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")
    seed = _e("SEED", 1337, int)
    compile_mode = _e("COMPILE_MODE", "default")

    # Validation
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 500, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 10, int)

    # Training length
    iterations = _e("ITERATIONS", 2000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.2, float)
    warmup_steps = _e("WARMUP_STEPS", 20, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 524288, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 1024, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 0.0, float)

    # Model shape
    vocab_size = _e("VOCAB_SIZE", 1024, int)
    num_layers = _e("NUM_LAYERS", 16, int)
    num_encoder_layers = _e("NUM_ENCODER_LAYERS", 2, int)   # Asymmetric U-Net: 2 encoder
    num_decoder_layers = _e("NUM_DECODER_LAYERS", 14, int)  # Asymmetric U-Net: 14 decoder
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    model_dim = _e("MODEL_DIM", 768, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 4, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    rope_base = _e("ROPE_BASE", 10000.0, float)
    logit_softcap = _e("LOGIT_SOFTCAP", 30.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    qk_gain_init = _e("QK_GAIN_INIT", 1.5, float)
    activation_type = _e("ACTIVATION", "relu2")
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 64, int)
    churn_log_every = _e("CHURN_LOG_EVERY", 500, int)

    # Optimizer
    embed_lr = _e("EMBED_LR", 0.6, float)
    head_lr = _e("HEAD_LR", 0.008, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.05, float)
    matrix_lr = _e("MATRIX_LR", 0.04, float)
    scalar_lr = _e("SCALAR_LR", 0.04, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.0, float)

    # L1 sparsity penalty for ternary zero-packing
    l1_lambda = _e("L1_LAMBDA", 0.005, float)

    # SLOT loss ramp bounds
    slot_weight_start = _e("SLOT_WEIGHT_START", 0.1, float)
    slot_weight_end = _e("SLOT_WEIGHT_END", 2.0, float)

    # FP storage for non-ternary params
    _fp_raw = os.environ.get("FP_STORAGE", "0")
    fp_storage = True if _fp_raw == "FP8" else ("fp4" if _fp_raw == "FP4" else False)

CTP = ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix",
       "resid_mixes", "q_gain", "skip_weight", "skip_weights", "vocab_bias")

# ---------------------------------------------------------------------------
# Ternary packing — base-3 encoding (5 trits/byte)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:,0] + g[:,1]*3 + g[:,2]*9 + g[:,3]*27 + g[:,4]*81).tobytes(), n

def unpack_ternary(data: bytes, n: int) -> Tensor:
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5): t[:,i] = v % 3; v //= 3
    return torch.from_numpy(t.reshape(-1)[:n].astype(np.int8) - 1)

def pack_ternary_bitmask(q: Tensor):
    f = q.reshape(-1).to(torch.int8).numpy(); n = len(f)
    nz = (f != 0)
    return np.packbits(nz).tobytes() + np.packbits(f[nz] > 0).tobytes(), n

def unpack_ternary_bitmask(data: bytes, n: int) -> Tensor:
    ms = (n + 7) // 8
    nz = np.unpackbits(np.frombuffer(data[:ms], dtype=np.uint8))[:n].astype(bool)
    s = np.unpackbits(np.frombuffer(data[ms:], dtype=np.uint8))[:int(nz.sum())].astype(bool)
    w = np.zeros(n, dtype=np.int8); w[nz] = np.where(s, 1, -1)
    return torch.from_numpy(w)

# ---------------------------------------------------------------------------
# State dict serialization (ternary + fp16/fp8)
# ---------------------------------------------------------------------------
def q_sd(state_dict: dict, group_size: int = 64, fp_storage=False,
         ternary_method="standard") -> tuple[dict, dict]:
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        is_ternary = (
            t.ndim == 2 and t.numel() > 65_536
            and "tok_emb" not in name and "lm_head" not in name
        )
        if is_ternary:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
            q = (t_grouped / scale).round().clamp(-1, 1).to(torch.int8)
            if ternary_method == "standard":
                packed_bytes, n_trits = pack_ternary(q)
                entry_type = "ternary"
            else:
                packed_bytes, n_trits = pack_ternary_bitmask(q)
                entry_type = "ternary_bitmask"
            quantized[name] = {
                "type": entry_type, "packed": packed_bytes,
                "scale": scale.half().squeeze(-1),
                "shape": list(t.shape), "padded_cols": t_padded.shape[1],
                "group_size": group_size, "n_trits": n_trits,
                "orig_shape": t_orig_shape,
            }
            stats["ternary_params"] += t.numel()
            stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        elif fp_storage and t.ndim == 2:
            quantized[name] = {"type": "fp8", "data": t.to(torch.float8_e4m3fn)}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel()
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2
    return quantized, stats

def deq_sd(quantized: dict, target_dtype=torch.bfloat16):
    out = {}
    for name, entry in quantized.items():
        if entry["type"] in ("ternary", "ternary_bitmask"):
            if entry["type"] == "ternary":
                q = unpack_ternary(entry["packed"], entry["n_trits"])
            else:
                q = unpack_ternary_bitmask(entry["packed"], entry["n_trits"])
            q = q.float().reshape(-1, entry["group_size"])
            scale = entry["scale"].float().unsqueeze(-1)
            q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            t = (q * (scale / q_absmean)).reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "fp8":
            out[name] = entry["data"].to(torch.float32).to(target_dtype).contiguous()
        else:
            out[name] = entry["data"].to(target_dtype).contiguous()
    return out

# ---------------------------------------------------------------------------
# Ternary diagnostics
# ---------------------------------------------------------------------------
def tern_stats(model: nn.Module, group_size: int = 64):
    total = zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "weight" in name and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1)
                zeros += int((q == 0).sum().item())
                total += int(q.numel())
    return {"zero_frac": zeros / max(total, 1), "total_weights": total}

_prev_committed: dict = {}

def churn_fn(model: nn.Module, group_size: int = 64):
    global _prev_committed
    total = flipped = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "weight" in name and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1).cpu().numpy()
                if name in _prev_committed:
                    flipped += int(np.sum(q != _prev_committed[name]))
                    total += q.size
                _prev_committed[name] = q
    return flipped / max(total, 1)

# ---------------------------------------------------------------------------
# MuonEq-R: Equilibrated RMSprop Muon with Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------
def ns_orth_eqr(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz orthogonalization with row/column equilibration (MuonEq-R).
    Normalizes row and column variances before NS iteration to prevent
    L1 penalty from killing network intelligence under ternary gradient attenuation."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # Equilibrate: normalize row and column RMS to balance gradient flow
    row_rms = X.norm(dim=1, keepdim=True).clamp(min=eps)
    col_rms = X.norm(dim=0, keepdim=True).clamp(min=eps)
    X = X / (row_rms.sqrt() * col_rms.sqrt())

    X /= X.norm() + eps
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """MuonEq-R: Equilibrated RMSprop Muon optimizer for ternary-aware training.
    Uses per-parameter EMA of squared gradients for additional equilibration."""
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, rmsprop_beta: float = 0.99):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps,
                                      nesterov=nesterov,
                                      rmsprop_beta=rmsprop_beta))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            rmsprop_beta = group["rmsprop_beta"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["sq_avg"] = torch.ones_like(g)  # RMSprop EMA
                    buf = state["momentum_buffer"]
                    sq_avg = state["sq_avg"]

                    # RMSprop equilibration: scale gradient by inverse sqrt of EMA
                    sq_avg.mul_(rmsprop_beta).addcmul_(g, g, value=1 - rmsprop_beta)
                    g_eq = g / sq_avg.sqrt().clamp(min=1e-8)

                    buf.mul_(momentum).add_(g_eq)
                    if nesterov:
                        g_eq = g_eq.add(buf, alpha=momentum)

                    # RMS norm before NS orthogonalization
                    g_eq = F.rms_norm(g_eq.float(), (g_eq.size(-1),)).bfloat16()
                    g_eq = ns_orth_eqr(g_eq, steps=backend_steps)
                    g_eq *= max(1, g_eq.size(0) / g_eq.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g_eq.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ld_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].pin_memory().to(
            self.device, non_blocking=True).to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group_size=64):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.bfloat16()
        g = self.group_size
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary,
                        self.bias.to(x.dtype) if self.bias is not None else None)


class NormedTernaryLinear(TernaryLinear):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 group_size=64):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.c_qkv = TernaryLinear(dim, self.q_size + 2 * self.kv_size,
                                   bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(dim, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init,
                                              dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv_out = self.c_qkv(x)
        q_out, k_out, v_out = qkv_out.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(),
                            causal=True)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, group_size=64):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = TernaryLinear(dim, hidden, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(hidden, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, group_size=64):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                        qk_gain_init, group_size)
        self.mlp = MLP(dim, mlp_mult, group_size)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        n = self.attn_norm(x)
        x = x + self.attn_scale.to(dtype=x.dtype) * self.attn(n)
        x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, group_size=64, softcap_type="poly",
                 num_encoder_layers=2, num_decoder_layers=14,
                 slot_weight_start=0.1, slot_weight_end=2.0):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.slot_weight_start = slot_weight_start
        self.slot_weight_end = slot_weight_end

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Asymmetric U-Net: 2 encoder, 14 decoder
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        assert num_encoder_layers + num_decoder_layers == num_layers
        # Skip connections: encoder layers connect to deepest decoder layers
        self.num_skip_weights = num_encoder_layers
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, group_size)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head._zero_init = True
        if tie_embeddings:
            self.lm_head.weight.requires_grad_(False)
        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, TernaryLinear) and not getattr(module, "_zero_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            logits_raw = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            logits_raw = self.lm_head(x)
        return logits_raw + self.vocab_bias.to(x.dtype)

    def _softcap(self, logits: Tensor) -> Tensor:
        s = self.logit_softcap
        if self.softcap_type == "tanh":
            return s * torch.tanh(logits / s)
        x_sc = torch.clamp(logits / s, -2.0, 2.0)
        x2 = x_sc * x_sc
        return s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        bsz, seq_len = input_ids.shape
        x = self.tok_emb(input_ids).float()
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Asymmetric U-Net: 2 encoder layers store skips
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        # 14 decoder layers; skip connections injected at the DEEPEST layers
        # i.e., the last num_encoder_layers decoder layers get the skips
        skip_inject_start = self.num_decoder_layers - self.num_encoder_layers
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if i >= skip_inject_start and skips:
                skip_idx = i - skip_inject_start  # 0, 1
                x = x + self.skip_weights[skip_idx].to(dtype=x.dtype) * skips[skip_idx]
            x = self.blocks[bi](x, x0)

        x_normed = self.final_norm(x)
        x_flat = x_normed.reshape(-1, x_normed.size(-1))
        targets = target_ids.reshape(-1)
        logits = self._softcap(self._compute_logits(x_flat))

        # Fused CE + Z-loss
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
        per_token_loss = lse - target_logits  # [bsz * seq_len]

        # SLOT: Sequence-Level Objective Training — linear ramp across sequence
        # Weight goes from slot_weight_start at position 0 to slot_weight_end at final
        slot_weights = torch.linspace(
            self.slot_weight_start, self.slot_weight_end, seq_len,
            device=per_token_loss.device, dtype=per_token_loss.dtype
        )  # [seq_len]
        slot_weights = slot_weights.unsqueeze(0).expand(bsz, -1).reshape(-1)  # [bsz*seq_len]

        main_loss = (per_token_loss * slot_weights).mean() + 1e-4 * (lse ** 2).mean()
        return main_loss

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def build_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def ld_val(pattern, seq_len, max_tok=int(os.environ.get("VAL_MAX_TOKENS", 500000))):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    tok = torch.cat([ld_shard(Path(p)) for p in files]).contiguous()
    if max_tok > 0: tok = tok[:max_tok + 1]
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * args.train_seq_len
            raw_end = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            token_count += n
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tok_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            tok_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += tok_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

    global ns_orth_eqr
    ns_orth_eqr = torch.compile(ns_orth_eqr)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs/cuda/", exist_ok=True)
    logfile = f"logs/cuda/{args.run_id}.txt" if master_process else None
    if master_process:
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = ld_val(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(
        sp, args.vocab_size, device)

    # --- Model ---
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size,
        softcap_type=args.softcap_type,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        slot_weight_start=args.slot_weight_start,
        slot_weight_end=args.slot_weight_end,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, nn.Linear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if base_model.lm_head is not None and args.tie_embeddings:
        base_model.lm_head.weight.requires_grad_(False)

    torch._dynamo.config.optimize_ddp = False
    compiled_model = torch.compile(base_model,
                                   mode=args.compile_mode if args.compile_mode != "default" else None)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                static_graph=True,
                gradient_as_bucket_view=True) if distributed else compiled_model

    # --- Optimizers ---
    _excl = {"tok_emb.weight", "lm_head.weight"}
    all_other_params = [(n, p) for n, p in base_model.named_parameters()
                        if not any(eh in n for eh in _excl)]
    matrix_params = [p for n, p in all_other_params
                     if p.ndim == 2 and not any(pat in n for pat in CTP)]
    scalar_params = [p for n, p in all_other_params
                     if p.ndim < 2 or any(pat in n for pat in CTP)]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": 0.0, "base_lr": 0.0}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizers = [opt_tok, opt_muon, opt_scalar, opt_head]

    # --- Logging ---
    log0("--- Hyperparameters ---", console=False)
    log0(" ".join(f"{a}={getattr(args,a)}" for a in sorted(dir(args))
                  if not a.startswith("_") and a not in ("train_files","val_files")
                  and not callable(getattr(args,a))), console=False)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} L:{args.num_layers}({args.num_encoder_layers}+{args.num_decoder_layers}) "
         f"d:{args.model_dim} h:{args.num_heads} kv:{args.num_kv_heads} "
         f"ws:{world_size} ga:{grad_accum_steps} s:{args.seed} "
         f"L1:{args.l1_lambda} SLOT:{args.slot_weight_start}->{args.slot_weight_end}")

    # --- Data loader & helpers ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float):
        if args.warmdown_fraction <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = int(args.iterations * (1.0 - args.warmdown_fraction))
            if step >= warmdown_start:
                return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0)
            return 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # --- Compiler warmup ---
    if args.warmup_steps > 0:
        _ms = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        _os = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for mi in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = mi == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(_ms, strict=True)
        for o, s in zip(optimizers, _os): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device,
                                         grad_accum_steps, val_tokens,
                                         base_bytes_lut, has_leading_space_lut,
                                         is_boundary_token_lut)
            tstats = tern_stats(base_model, group_size=args.bitnet_group_size)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                 f"val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms "
                 f"zero_frac:{tstats['zero_frac']:.3f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                     f"step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        zero_grad_all()
        train_loss.zero_()

        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len,
                                           grad_accum_steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss.add_(loss.detach())
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # L1 sparsity penalty: inject into gradients of ternary-candidate weights
        if args.l1_lambda > 0:
            with torch.no_grad():
                for p in matrix_params:
                    if p.grad is not None:
                        p.grad.add_(args.l1_lambda * torch.sign(p.data))

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups:
            g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # LR scheduling
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()
        zero_grad_all()
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and step % args.train_log_every == 0:
            log0(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} "
                 f"t:{approx_ms:.0f}ms avg:{approx_ms/step:.1f}ms")
        if args.churn_log_every > 0 and step % args.churn_log_every == 0:
            log0(f"step:{step} churn:{churn_fn(base_model, args.bitnet_group_size):.4f} "
                 f"zero:{tern_stats(base_model, args.bitnet_group_size)['zero_frac']:.3f}")

        # Wallclock cap sync
        if stop_after_step is None and max_wallclock_ms is not None and step % 10 == 0:
            reached_cap = approx_ms >= max_wallclock_ms
            if distributed:
                cap_t = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
                reached_cap = bool(cap_t.item())
            if reached_cap:
                stop_after_step = step

    # --- Serialization ---
    if master_process:
        sd = base_model.state_dict()
        if base_model.tie_embeddings:
            sd.pop("lm_head.weight", None)

        methods = {}
        for method in ("standard", "bitmask"):
            q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size,
                                fp_storage=args.fp_storage, ternary_method=method)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            methods[method] = {"blob": lzma.compress(buf.getvalue(), preset=9),
                               "stats": stats}
        best = min(methods, key=lambda m: len(methods[m]["blob"]))
        final_blob, q_stats = methods[best]["blob"], methods[best]["stats"]
        with open("final_model.ternary.ptz", "wb") as f:
            f.write(final_blob)

        artifact_bytes = len(final_blob)
        code_bytes = len(code.encode("utf-8"))
        total = artifact_bytes + code_bytes
        log0(f"artifact:{artifact_bytes/1e6:.2f}MB ternary:{q_stats['ternary_params']}"
             f"({q_stats['ternary_bytes']}B) fp:{q_stats['fp_params']}"
             f"({q_stats['fp_bytes']}B) code:{code_bytes}")
        log0(f"budget:{total}/{16000000} ({total/1e6:.2f}/{16.00:.2f}MB) "
             f"{'FITS' if total <= 16000000 else 'OVER'}")

    # --- Roundtrip validation ---
    if distributed:
        dist.barrier()

    with open("final_model.ternary.ptz", "rb") as f:
        loaded = torch.load(io.BytesIO(lzma.decompress(f.read())),
                            map_location="cpu", weights_only=False)
    base_model.load_state_dict(deq_sd(loaded), strict=False)
    torch._dynamo.reset()

    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device,
                                     grad_accum_steps, val_tokens,
                                     base_bytes_lut, has_leading_space_lut,
                                     is_boundary_token_lut)
    log0(f"final_ternary_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
