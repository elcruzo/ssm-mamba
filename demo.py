"""SSD chunkwise (default) vs sequential oracle, then a short shift-task train."""
from __future__ import annotations

import torch

from mamba import (
    Mamba1Block,
    Mamba2SSD,
    naive_discrete_ssm,
    resolve_device,
    ssd_chunkwise,
    train_shift_step,
)


def main() -> None:
    name, device = resolve_device("auto")
    print(f"device {name}")
    torch.manual_seed(0)
    bsz, length, h, p, n, chunk = 1, 16, 2, 3, 4, 8
    x = torch.randn(bsz, length, h, p, device=device)
    a = -0.2 * torch.rand(bsz, length, h, device=device)
    bb = torch.randn(bsz, length, h, n, device=device)
    c = torch.randn(bsz, length, h, n, device=device)
    y_ssd, _ = ssd_chunkwise(x, a, bb, c, chunk)
    y_naive = naive_discrete_ssm(x, a, bb, c)
    err = (y_ssd - y_naive).abs().max().item()
    print(f"ssd vs naive max|Δ| {err:.2e}")
    y2 = Mamba2SSD(16, n_heads=4, d_state=8, chunk=4).to(device)(torch.randn(2, 20, 16, device=device))
    print("mamba2-ssd", tuple(y2.shape))
    y1 = Mamba1Block(16).to(device)(torch.randn(2, 20, 16, device=device))
    print("mamba1-selective", tuple(y1.shape))
    losses = train_shift_step(steps=40, seed=1, mixer="ssd", device="cpu")
    print(f"ssd shift loss {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
