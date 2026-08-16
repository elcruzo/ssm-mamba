"""Parallel vs sequential scan, then a few shift-task steps."""
from __future__ import annotations

import torch

from mamba import MambaBlock, naive_scan, parallel_scan, train_shift_step


def main() -> None:
    torch.manual_seed(0)
    a = torch.rand(1, 8, 2, 3) * 0.4 + 0.4
    b = torch.randn(1, 8, 2, 3)
    err = (naive_scan(a, b) - parallel_scan(a, b)).abs().max().item()
    print(f"scan max|Δ| {err:.2e}")
    y = MambaBlock(16)(torch.randn(2, 20, 16))
    print("mamba", tuple(y.shape))
    losses = train_shift_step(steps=20, seed=1)
    print(f"shift loss {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
