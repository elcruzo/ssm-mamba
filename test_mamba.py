"""Mamba-2 SSD (default) + Mamba-1 selective-scan invariants."""
from __future__ import annotations

import torch

from mamba import (
    Mamba1Block,
    Mamba2SSD,
    naive_discrete_ssm,
    naive_scan,
    parallel_scan,
    resolve_device,
    ssd_chunkwise,
    train_shift_step,
)


def test_resolve_device_named_paths():
    name, dev = resolve_device("cpu")
    assert name == "cpu" and dev.type == "cpu"
    try:
        resolve_device("cuda")  # type: ignore[arg-type]
        assert False, "expected ValueError"
    except ValueError:
        pass
    if torch.backends.mps.is_available():
        name_m, dev_m = resolve_device("mps")
        assert name_m == "mps" and dev_m.type == "mps"
    auto_name, auto_dev = resolve_device("auto")
    assert auto_name in ("mps", "cpu")
    assert auto_dev.type == auto_name


def test_ssd_matches_naive_oracle():
    torch.manual_seed(1)
    bsz, length, h, p, n, chunk = 2, 16, 2, 3, 4, 8
    x = torch.randn(bsz, length, h, p)
    a = -0.2 * torch.rand(bsz, length, h)
    bb = torch.randn(bsz, length, h, n)
    c = torch.randn(bsz, length, h, n)
    y_ssd, final = ssd_chunkwise(x, a, bb, c, chunk)
    y_naive = naive_discrete_ssm(x, a, bb, c)
    assert y_ssd.shape == (bsz, length, h, p)
    assert final.shape == (bsz, h, p, n)
    assert torch.allclose(y_ssd, y_naive, atol=1e-4, rtol=1e-4)
    # Not a D-skip / identity cheat
    assert not torch.allclose(y_ssd, x, atol=1e-2)
    assert y_ssd.abs().mean() > 1e-3


def test_ssd_chunk_sizes_agree_and_carry_matters():
    """Same recurrence under different chunkings; inter-chunk carry is not optional."""
    torch.manual_seed(3)
    bsz, length, h, p, n = 1, 16, 2, 2, 3
    x = torch.randn(bsz, length, h, p)
    a = -0.15 * torch.rand(bsz, length, h)
    bb = torch.randn(bsz, length, h, n)
    c = torch.randn(bsz, length, h, n)
    y8, _ = ssd_chunkwise(x, a, bb, c, block_len=8)
    y4, _ = ssd_chunkwise(x, a, bb, c, block_len=4)
    y16, _ = ssd_chunkwise(x, a, bb, c, block_len=16)
    y_n = naive_discrete_ssm(x, a, bb, c)
    assert torch.allclose(y8, y4, atol=1e-4, rtol=1e-4)
    assert torch.allclose(y8, y16, atol=1e-4, rtol=1e-4)
    assert torch.allclose(y8, y_n, atol=1e-4, rtol=1e-4)

    # Past inputs affect future outputs (state carry); future inputs do not affect the past.
    x_past = x.clone()
    x_past[:, :8] = 0
    y_full, _ = ssd_chunkwise(x, a, bb, c, block_len=8)
    y_past, _ = ssd_chunkwise(x_past, a, bb, c, block_len=8)
    assert not torch.allclose(y_full[:, 8:], y_past[:, 8:], atol=1e-3)

    x_fut = x.clone()
    x_fut[:, 8:] = 0
    y_fut, _ = ssd_chunkwise(x_fut, a, bb, c, block_len=8)
    assert torch.allclose(y_full[:, :8], y_fut[:, :8], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(y_full[:, 8:], y_fut[:, 8:], atol=1e-3)


def test_ssd_interchunk_vs_diag_only():
    """Y_off must contribute across chunk boundary (diag-only would miss prior state)."""
    torch.manual_seed(4)
    bsz, length, h, p, n, chunk = 1, 8, 1, 2, 2, 4
    x = torch.randn(bsz, length, h, p)
    a = -0.3 * torch.ones(bsz, length, h)
    bb = torch.randn(bsz, length, h, n)
    c = torch.randn(bsz, length, h, n)
    y, _ = ssd_chunkwise(x, a, bb, c, chunk)
    # Reconstruct diag-only: run SSD on each chunk in isolation and concat
    y_diag_parts = []
    for start in (0, 4):
        yi, _ = ssd_chunkwise(x[:, start : start + 4], a[:, start : start + 4], bb[:, start : start + 4], c[:, start : start + 4], 4)
        y_diag_parts.append(yi)
    y_diag_only = torch.cat(y_diag_parts, dim=1)
    assert not torch.allclose(y[:, 4:], y_diag_only[:, 4:], atol=1e-3)
    assert torch.allclose(y[:, :4], y_diag_only[:, :4], atol=1e-4, rtol=1e-4)


def test_mamba2_block_shape_and_causality():
    torch.manual_seed(2)
    block = Mamba2SSD(d_model=8, n_heads=2, d_state=4, chunk=4)
    block.eval()
    x = torch.randn(2, 10, 8)
    y = block(x)
    assert y.shape == (2, 10, 8)
    x2 = x.clone()
    x2[:, 6:] = torch.randn_like(x2[:, 6:])
    y2 = block(x2)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(y[:, 6:], y2[:, 6:], atol=1e-3)


def test_mamba1_parallel_equals_naive_scan():
    torch.manual_seed(0)
    a = torch.rand(2, 17, 4, 5) * 0.5 + 0.3
    b = torch.randn(2, 17, 4, 5)
    assert torch.allclose(naive_scan(a, b), parallel_scan(a, b), atol=1e-4, rtol=1e-4)


def test_mamba1_block_causal_named_variant():
    torch.manual_seed(2)
    block = Mamba1Block(d_model=8, d_state=4, expand=2)
    block.eval()
    x = torch.randn(2, 12, 8)
    y = block(x)
    assert y.shape == (2, 12, 8)
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:])
    y2 = block(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(y[:, 7:], y2[:, 7:], atol=1e-3)
    y_n = block(x, scan="naive")
    assert torch.allclose(y, y_n, atol=1e-4, rtol=1e-4)


def test_shift_task_ssd_default_loss_drops():
    losses = train_shift_step(steps=40, seed=0, mixer="ssd", device="cpu")
    assert losses[-1] < 0.5, (losses[0], losses[-1])
    assert losses[-1] < losses[0] - 0.5, (losses[0], losses[-1])


def test_shift_task_mamba1_named_variant_loss_drops():
    losses = train_shift_step(steps=40, seed=1, mixer="selective_scan", device="cpu")
    assert losses[-1] < 0.5, (losses[0], losses[-1])
    assert losses[-1] < losses[0] - 0.5, (losses[0], losses[-1])
