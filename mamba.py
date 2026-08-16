"""Mamba-2 SSD (default) + Mamba-1 selective scan (named variant), from scratch.

Default path — Dao & Gu 2024 structured state-space duality (SSD), Listing 1:
  chunk the sequence, quadratic intra-chunk (segsum kernel), inter-chunk state pass.
  Recurrence in log-space a_t: h_t = exp(a_t) h_{t-1} + B_t ⊙ x_t ; y_t = C_t · h_t
  Dual form: Y = (C B^T ⊙ L) X with L_ij = 1_{i≥j} exp(∑_{k=j+1}^{i} a_k).

Named variant — Gu & Dao 2023 Mamba-1 selective SSM:
  diagonal A_t = exp(-softplus(δ_t) * exp(A_log)); Blelloch parallel prefix scan
  (or sequential oracle) on the monoid (a,b)⊕(a',b')=(a'a, a'b+b').

No Triton / einops / mamba_ssm. Devices are named: resolve_device('mps'|'cpu'|'auto').
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DeviceName = Literal["mps", "cpu", "auto"]


def resolve_device(name: DeviceName = "auto") -> tuple[str, torch.device]:
    """Named device paths — no silent try/except fallback between accelerators.

    - ``mps``: require ``torch.backends.mps.is_available()``
    - ``cpu``: always CPU
    - ``auto``: resolves to the name ``mps`` or ``cpu`` (both are named outcomes)
    """
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' requested but torch.backends.mps is not available")
        return "mps", torch.device("mps")
    if name == "cpu":
        return "cpu", torch.device("cpu")
    if name == "auto":
        if torch.backends.mps.is_available():
            return "mps", torch.device("mps")
        return "cpu", torch.device("cpu")
    raise ValueError(f"unknown device name {name!r}; expected 'mps', 'cpu', or 'auto'")


# ---------------------------------------------------------------------------
# Mamba-2 SSD (default)
# ---------------------------------------------------------------------------


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sums: (..., T) -> (..., T, T).

    Entry (i, j) is sum_{k=j+1..i} x_k for i>=j, else -inf.
    Uses only additions (no cumsum differences) so exp(segsum) stays finite in FP32.
    """
    t = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, t)
    strict = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~strict, 0)
    s = torch.cumsum(x, dim=-2)
    keep = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=0)
    return s.masked_fill(~keep, torch.finfo(s.dtype).min / 2)


def ssd_chunkwise(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    block_len: int,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise SSD (Dao & Gu Listing 1 / ssd_minimal_discrete).

    x: (B, L, H, P)   a: (B, L, H) log-space per-step   b,c: (B, L, H, N)
    Returns (Y, final_state) with Y: (B, L, H, P), final_state: (B, H, P, N).

    Four steps: (1) intra-chunk outputs, (2) zero-init chunk states,
    (3) inter-chunk state pass, (4) state→output per chunk.
    """
    bsz, length, n_heads, d_head = x.shape
    if length % block_len != 0:
        raise ValueError(f"length {length} must be divisible by block_len {block_len}")
    n_chunks = length // block_len

    def chunk(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(bsz, n_chunks, block_len, *t.shape[2:])

    xc, ac, bc, cc = chunk(x), chunk(a), chunk(b), chunk(c)
    a_perm = ac.permute(0, 3, 1, 2)  # (B, H, C, L)
    a_cumsum = torch.cumsum(a_perm, dim=-1)

    # 1. Intra-chunk (diagonal blocks): Y = (C B^T ⊙ L) X
    lmat = torch.exp(segsum(a_perm))
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", cc, bc, lmat, xc)

    # 2. Chunk states assuming zero initial state
    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", bc, decay_states, xc)

    # 3. Inter-chunk SSM recurrence (1-SS matmul on chunk boundary decays)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(a_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Carry state into each position's output
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", cc, states, torch.exp(a_cumsum))
    y = (y_diag + y_off).reshape(bsz, length, n_heads, d_head)
    return y, final_state


def naive_discrete_ssm(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Sequential oracle for SSD: h_t = exp(a_t) h + b_t ⊙ x_t ; y_t = c_t · h_t."""
    bsz, length, n_heads, d_head = x.shape
    n = b.size(-1)
    h = x.new_zeros(bsz, n_heads, d_head, n)
    ys = []
    for t in range(length):
        h = torch.exp(a[:, t, :, None, None]) * h + b[:, t, :, None, :] * x[:, t, :, :, None]
        ys.append((h * c[:, t, :, None, :]).sum(-1))
    return torch.stack(ys, dim=1)


class Mamba2SSD(nn.Module):
    """Default block: input-dependent dt, B, C; chunkwise SSD; D skip; out_proj.

    Scalar-times-identity A per head (Mamba-2 SSD restriction vs Mamba-1 diagonal A).
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_state: int = 8, chunk: int = 8):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_state = d_state
        self.chunk = chunk
        self.in_x = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, n_heads * d_state)
        self.C_proj = nn.Linear(d_model, n_heads * d_state)
        self.dt_proj = nn.Linear(d_model, n_heads)
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.D = nn.Parameter(torch.ones(n_heads))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, d_model = x.shape
        pad = (self.chunk - length % self.chunk) % self.chunk
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        lp = x.size(1)
        u = self.in_x(x).view(bsz, lp, self.n_heads, self.d_head)
        dt = F.softplus(self.dt_proj(x))
        a = -torch.exp(self.A_log) * dt
        bb = self.B_proj(x).view(bsz, lp, self.n_heads, self.d_state)
        cc = self.C_proj(x).view(bsz, lp, self.n_heads, self.d_state)
        # Discretization side effect: scale X by dt (same as mamba_ssm ssd_minimal test)
        y, _ = ssd_chunkwise(u * dt.unsqueeze(-1), a, bb, cc, self.chunk)
        y = y + u * self.D.view(1, 1, -1, 1)
        y = y.reshape(bsz, lp, d_model)
        if pad:
            y = y[:, :length]
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Mamba-1 selective scan (named variant)
# ---------------------------------------------------------------------------


def naive_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential h_t = a_t * h_{t-1} + b_t along dim=1. a,b: (B, L, ...)."""
    outs = []
    h = torch.zeros_like(a[:, 0])
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Blelloch inclusive prefix scan of the same recurrence, along dim=1."""
    bsz, length = a.shape[:2]
    rest = a.shape[2:]
    a_l = a.reshape(bsz, length, -1).permute(1, 0, 2).contiguous()
    b_l = b.reshape(bsz, length, -1).permute(1, 0, 2).contiguous()
    h = _blelloch_inclusive(a_l, b_l)
    return h.permute(1, 0, 2).reshape(bsz, length, *rest)


def _blelloch_inclusive(a0: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
    """a0/b0: (L, ...). Identity of ⊕ is (1, 0); (a1,b1)⊕(a2,b2)=(a2 a1, a2 b1 + b2)."""
    length = a0.shape[0]
    n = 1 << (length - 1).bit_length() if length > 0 else 1
    rest = a0.shape[1:]
    if n != length:
        a0 = torch.cat([a0, a0.new_ones((n - length, *rest))], dim=0)
        b0 = torch.cat([b0, b0.new_zeros((n - length, *rest))], dim=0)
    a = a0.clone()
    b = b0.clone()
    logn = int(math.log2(n)) if n > 0 else 0
    for d in range(logn):
        step = 1 << (d + 1)
        k = torch.arange(0, n, step, device=a.device)
        left = k + (1 << d) - 1
        right = k + step - 1
        an, bn = a.clone(), b.clone()
        an[right] = a[right] * a[left]
        bn[right] = a[right] * b[left] + b[right]
        a, b = an, bn
    a[-1] = 1
    b[-1] = 0
    for d in range(logn - 1, -1, -1):
        step = 1 << (d + 1)
        k = torch.arange(0, n, step, device=a.device)
        left = k + (1 << d) - 1
        right = k + step - 1
        an, bn = a.clone(), b.clone()
        ta, tb = a[left], b[left]
        ra, rb = a[right], b[right]
        an[left] = ra
        bn[left] = rb
        an[right] = ta * ra
        bn[right] = ta * rb + tb
        a, b = an, bn
    h = a0 * b + b0
    return h[:length]


class Mamba1SelectiveSSM(nn.Module):
    """Mamba-1 selective SSM: input-dependent Δ, B, C; diagonal A; scan over state."""

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_state = d_state
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).expand(d_model, d_state).clone()
        )
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def params_from_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = F.softplus(self.dt_proj(x))
        a = torch.exp(-delta.unsqueeze(-1) * torch.exp(self.A_log))
        bb = self.B_proj(x)
        bx = delta.unsqueeze(-1) * bb.unsqueeze(2) * x.unsqueeze(-1)
        c = self.C_proj(x)
        return a, bx, c

    def forward(self, x: torch.Tensor, scan: Literal["parallel", "naive"] = "parallel") -> torch.Tensor:
        a, bx, c = self.params_from_x(x)
        if scan == "parallel":
            h = parallel_scan(a, bx)
        elif scan == "naive":
            h = naive_scan(a, bx)
        else:
            raise ValueError(f"unknown scan {scan!r}; use 'parallel' or 'naive'")
        y = (h * c.unsqueeze(2)).sum(-1)
        return y + self.D * x


class Mamba1Block(nn.Module):
    """Named Mamba-1 variant: in_proj → depthwise causal conv1d → SiLU → selective SSM → gate → out_proj."""

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.ssm = Mamba1SelectiveSSM(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor, scan: Literal["parallel", "naive"] = "parallel") -> torch.Tensor:
        length = x.size(1)
        x_and_z = self.in_proj(x)
        x, z = x_and_z.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x, scan=scan)
        return self.out_proj(y * F.silu(z))


# ---------------------------------------------------------------------------
# Toy LM (defaults to Mamba-2 SSD)
# ---------------------------------------------------------------------------


class ShiftLM(nn.Module):
    """Next-token shift toy; default mixer is Mamba2SSD."""

    def __init__(
        self,
        vocab: int = 8,
        d_model: int = 16,
        mixer: Literal["ssd", "selective_scan"] = "ssd",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        if mixer == "ssd":
            self.block: nn.Module = Mamba2SSD(d_model, n_heads=4, d_state=8, chunk=4)
        elif mixer == "selective_scan":
            self.block = Mamba1Block(d_model, d_state=8, expand=2)
        else:
            raise ValueError(f"unknown mixer {mixer!r}; use 'ssd' or 'selective_scan'")
        self.mixer = mixer
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(self.emb(idx)))


def train_shift_step(
    steps: int = 40,
    seed: int = 0,
    mixer: Literal["ssd", "selective_scan"] = "ssd",
    device: DeviceName = "cpu",
) -> list[float]:
    """Train ShiftLM on modular increment: token[t+1] = (token[t] + 1) % V.

    Uniform random LM has CE floor log(V) with no signal; this task is learnable
    and forces the mixer to carry state across positions.
    """
    _, dev = resolve_device(device)
    torch.manual_seed(seed)
    vocab, length, batch = 6, 12, 32
    model = ShiftLM(vocab=vocab, d_model=16, mixer=mixer).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(steps):
        start = torch.randint(0, vocab, (batch, 1), device=dev)
        idx = (start + torch.arange(length, device=dev).view(1, -1)) % vocab
        logits = model(idx)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), idx[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses
