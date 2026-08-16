"""Mamba / Mamba-2 scan invariants."""
from __future__ import annotations

import torch

from mamba import (
    Mamba2SSD,
    MambaBlock,
    naive_discrete_ssm,
    naive_scan,
    parallel_scan,
    ssd_minimal_discrete,
    train_shift_step,
)


def test_naive_equals_parallel_scan():
    torch.manual_seed(0)
    a = torch.rand(2, 17, 4, 5) * 0.5 + 0.3
    b = torch.randn(2, 17, 4, 5)
    h_n = naive_scan(a, b)
    h_p = parallel_scan(a, b)
    assert torch.allclose(h_n, h_p, atol=1e-4, rtol=1e-4)


def test_ssd_matches_naive_and_shape():
    torch.manual_seed(1)
    bsz, length, h, p, n, chunk = 2, 16, 2, 3, 4, 8
    x = torch.randn(bsz, length, h, p)
    a = -0.2 * torch.rand(bsz, length, h)
    bb = torch.randn(bsz, length, h, n)
    c = torch.randn(bsz, length, h, n)
    y_ssd, _ = ssd_minimal_discrete(x, a, bb, c, chunk)
    y_naive = naive_discrete_ssm(x, a, bb, c)
    assert y_ssd.shape == (bsz, length, h, p)
    assert torch.allclose(y_ssd, y_naive, atol=1e-4, rtol=1e-4)
    block = MambaBlock(d_model=8, d_state=4, expand=2)
    y = block(torch.randn(3, 11, 8))
    assert y.shape == (3, 11, 8)
    y2 = Mamba2SSD(d_model=8, n_heads=2, d_state=4, chunk=4)(torch.randn(2, 10, 8))
    assert y2.shape == (2, 10, 8)


def test_block_is_causal():
    torch.manual_seed(2)
    block = MambaBlock(d_model=8, d_state=4, expand=2)
    block.eval()
    x = torch.randn(2, 12, 8)
    y = block(x)
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:])
    y2 = block(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(y[:, 7:], y2[:, 7:], atol=1e-3)


def test_shift_task_loss_drops():
    losses = train_shift_step(steps=30, seed=0)
    assert losses[-1] < losses[0], (losses[0], losses[-1])
