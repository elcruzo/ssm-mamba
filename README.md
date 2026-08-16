# 13-ssm-mamba — Selective SSM / Mamba-2 SSD

Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023). https://arxiv.org/abs/2312.00752

Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2, 2024). https://arxiv.org/abs/2405.21060

## Mamba-1

Input-dependent \(\Delta,B,C\); diagonal \(A\) via \(\,A_t=\exp(-\mathrm{softplus}(\delta_t)\odot\exp(A_{\log}))\).

\[
h_t = A_t \odot h_{t-1} + B_t \odot x_t,\qquad y_t = C_t \cdot h_t
\]

- **Oracle:** sequential loop.
- **Fast path:** Blelloch parallel prefix scan on the monoid \((a,b)\oplus(a',b')=(a'a,\; a'b+b')\). Identity \((1,0)\). Inclusive scan recovers \(h\).

Block: `in_proj` → depthwise **causal** `conv1d` → SiLU → SSM → SiLU gate → `out_proj`.

## Mamba-2 SSD (duality)

The same recurrence in log-space \(a_t=\log A_t\) is a **causal kernel**

\[
L_{ij}=1_{i\ge j}\exp\Big(\sum_{k=j+1}^{i}a_k\Big)
\]

so \(Y = (C B^\top \odot L)\,X\) plus a low-rank carry of the SSM state across chunk boundaries. That is structured state-space duality: quadratic *inside* a chunk, linear *across* chunks. `ssd_minimal_discrete` reimplements Dao & Gu Listing 1 (no einops/Triton).

## Papers on disk

- [`papers/gu-dao-mamba-2023.pdf`](papers/gu-dao-mamba-2023.pdf) — Gu & Dao. Mamba (2023) ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))
- [`papers/dao-gu-mamba2-ssd-2024.pdf`](papers/dao-gu-mamba2-ssd-2024.pdf) — Dao & Gu. Mamba-2 / SSD (2024) ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060))

## Run

```bash
python demo.py
python -m pytest 13-ssm-mamba -q
```
