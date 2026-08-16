# ssm-mamba — Mamba-2 SSD (default) + Mamba-1 selective scan

Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2, 2024). https://arxiv.org/abs/2405.21060

Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023). https://arxiv.org/abs/2312.00752

**Default:** chunkwise SSD (`ssd_chunkwise` / `Mamba2SSD`). **Named variant:** Mamba-1 selective scan (`Mamba1Block` / `parallel_scan`). From-scratch PyTorch — no Triton, einops, or `mamba_ssm`. Devices via `resolve_device("mps"|"cpu"|"auto")`.

## Mamba-2 SSD (default)

Scalar-times-identity \(A\) per head. Log-space steps \(a_t\) give the causal kernel

\[
L_{ij}=1_{i\ge j}\exp\Big(\sum_{k=j+1}^{i}a_k\Big)
\]

so \(Y=(CB^\top\odot L)\,X\) inside a chunk, plus a low-rank carry of the SSM state across chunk boundaries. Four steps (Dao & Gu Listing 1 / blog Part III):

1. **Intra-chunk outputs** — quadratic form with `segsum` (stable, addition-only).
2. **Chunk states** — final state per chunk assuming zero initial state (matmul).
3. **Pass states** — 1-SS recurrence on the \(T/Q\) boundary states.
4. **Output states** — map carried state into each position.

Same FLOP class as a linear SSM; matmuls replace a full-length selective scan.

## Mamba-1 selective scan (named variant)

Input-dependent \(\Delta,B,C\); diagonal \(A\) via \(\,A_t=\exp(-\mathrm{softplus}(\delta_t)\odot\exp(A_{\log}))\).

\[
h_t = A_t \odot h_{t-1} + B_t \odot x_t,\qquad y_t = C_t \cdot h_t
\]

- **Oracle:** `naive_scan` sequential loop.
- **Fast path:** Blelloch parallel prefix scan on \((a,b)\oplus(a',b')=(a'a,\; a'b+b')\).

Block: `Mamba1Block` — `in_proj` → depthwise **causal** `conv1d` → SiLU → `Mamba1SelectiveSSM` → SiLU gate → `out_proj`. Select with `ShiftLM(..., mixer="selective_scan")`.

## Papers on disk

- [`papers/dao-gu-mamba2-ssd-2024.pdf`](papers/dao-gu-mamba2-ssd-2024.pdf) — Dao & Gu. Mamba-2 / SSD (2024) ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060))
- [`papers/gu-dao-mamba-2023.pdf`](papers/gu-dao-mamba-2023.pdf) — Gu & Dao. Mamba (2023) ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))

## Run

```bash
python demo.py
python -m pytest test_mamba.py -q
```
