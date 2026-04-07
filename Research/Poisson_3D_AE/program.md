# autoresearch — Scalable AE for NM-ROM (3D Poisson)

Autonomous research loop to develop an Autoencoder that:
1. Generalizes well across the full Poisson parameter space (k ∈ [1,5]³ + OOD)
2. Scales stably from 32³ → 64³ → 128³ without gradient explosion or OOM
3. Produces a compact latent embedding z ∈ ℝ^k_dim suitable for downstream NM-ROM

The NM-ROM pipeline (LM-GN solver, EQ offline phase) is downstream — the AE embedding
quality directly determines whether NM-ROM can converge at all.

---

## Scientific Goal

Solve parameterized 3D Poisson PDEs:
    -∇²u = F(x;k)    on [0,1]³,    u=0 on ∂Ω

Forcing: F(x;k) = 10·sin(k₁πx)·sin(k₂πy)·sin(k₃πz), k∈{1..5}³ (125 parameter cases).
Test grids: 32³, 64³, 128³ (DOF = 32K, 262K, 2M). Finite difference, 7-point stencil.

### What "scalable" means here

An architecture is scalable if ALL of the following hold across N=32, 64, 128:
- **No explosion**: training loss stays finite for 20k epochs; no NaN/Inf gradients
- **Fixed param count (roughly)**: params should NOT scale as O(N³) — that defeats the purpose
- **Reasonable training time**: ≤ 10 min on A100 for N=128; ≤ 2 min for N=32
- **Good reconstruction**: mean val rel-L2 < 5e-2 across OOD cases

### Why the current AE is a problem

The current `ScalableAutoencoder` uses:
- **Encoder**: DCT + MLP (spectral approach — param count is grid-independent ✓)
- **Decoder**: CP-factored separable decoder with learned W_x, W_y, W_z matrices

The decoder's factor matrices (W_x, W_y, W_z) each have shape `(rank, N)`.
At N=32: 3 × 256 × 32 = 24K params.
At N=64: 3 × 256 × 64 = 49K params.
At N=128: 3 × 256 × 128 = 98K params.
This is manageable. But the full evaluation `einsum('r,ri,rj,rk->ijk', h, W_x, W_y, W_z)`
produces an N³ tensor — at N=128 that's 2M floats per decode call, which is expensive in
the NM-ROM online phase (hundreds of decode calls per GN step).

The main failure modes observed:
1. **Generalization gap**: val error >> train error — AE memorizes the 125-case grid
2. **Architecture not stress-tested at 64³/128³** — unknown if it even trains stably

---

## Files

### In-scope (you modify these)

- `INR-Autoencoder.py` — AE architecture, training loop, hyperparameters, evaluation

### Fixed (do NOT modify)

- `NMROM-INR-Poisson-3D.py` — downstream NM-ROM solver (uses checkpoint)
- `run_inr.slurm` — launcher
- `program.md` — this file

---

## Cluster environment

- **Working directory**: `/cluster/tufts/paralab/tawal01/NMROM/Non-linear-Manifold-Reduced-Order-Modeling-for-Elliptic-and-Parabolic-PDEs-via-JAX/Research/Poisson_3D_AE/`
- **Python env**: `module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x` + `PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:$PYTHONPATH`
- **GPU**: A100 80GB (SLURM)

---

## Running an experiment

```bash
cd /cluster/tufts/paralab/tawal01/NMROM/Non-linear-Manifold-Reduced-Order-Modeling-for-Elliptic-and-Parabolic-PDEs-via-JAX/Research/Poisson_3D_AE

# SSH to the allocated A100 node (job 35696458 → s1cmp004) and run the AE:
bash run_inr.slurm --gpu 35696458 > run.log 2>&1
```

The `--gpu 35696458` flag tells the launcher to SSH directly to `s1cmp004` (the already-allocated GPU).
If that job dies, the script will automatically submit a new SLURM job instead.

**Timeout**: If training exceeds 10 minutes for N=32 or 20 minutes for N=64, kill (`Ctrl+C`) and treat as crash.
- Expected: ~90s for N=32 on A100

---

## Detecting completion vs crash

```bash
grep "=== AE TRAINING COMPLETE ===" training.log   # must exist
```

If missing → crash. Run `tail -50 run.log` to read the error.

## Extracting results

```bash
grep "Mean train rec\|Mean val rec\|Grid\|Model params\|Training time" training.log
```

Parse like:
- `grep "Mean train rec" training.log` → `Mean train rec error:   4.04e-03`
- `grep "Mean val rec"   training.log` → `Mean val   rec error:   1.23e-02`
- `grep "Grid:"          training.log` → `Grid: 32³ = 32,768 nodes`

---

## Metrics

| Metric | Source | Goal |
|--------|--------|------|
| `ae_train_err` | `training.log` | < 5e-3 (currently ~4e-3) |
| `ae_val_err` | `training.log` | **< 5e-2**, ideally < 1e-2 |
| `generalization_gap` | derived | val_err / train_err < 5× |
| `train_time_s` | `training.log` | < 120s for N=32 |
| `stable_at_64` | run succeeds | no NaN/crash |
| `stable_at_128` | run succeeds | no NaN/crash |

Primary goal: **minimize ae_val_err while keeping ae_train_err < 5e-3 and no explosion at N=64, N=128.**

---

## Logging results

Log to `results.tsv` (tab-separated). Do NOT commit this file.

Header:
```
commit	N	ae_train_err	ae_val_err	gen_gap	train_time_s	status	description
```

---

## Baseline (already established — do NOT re-run)

```
commit	N	ae_train_err	ae_val_err	gen_gap	train_time_s	status	description
current	32	4.04e-03	~unknown	~unknown	~90	keep	baseline spectral MLP AE
```

Establish the true val error of the baseline as your first run, then start experimenting.

---

## Key architectural constraints (must hold for NM-ROM compatibility)

1. **Encoder must output z ∈ ℝ^k_dim** (a fixed-size float vector, no spatial structure)
2. **Decoder must accept z and return a flat field of length N³**
3. **Decoder must support `decode_at_flat_indices(z, indices)`** for EQ evaluation — this is
   what the NM-ROM online phase calls to evaluate residuals at sparse magic points
4. **Checkpoint format must include**: `params`, `model_cfg` (dict with at least `latent_dim`,
   `rank`, `grid_size`), `normalization` (with `mean` and `std`)
5. **k_dim should stay in [8, 32]** — smaller is faster for NM-ROM GN steps (O(k_dim²))

---

## Research directions (ordered by expected impact)

### Fix generalization first (N=32)

1. **More diverse training data**: Current dataset has 200 single-mode + 100 multi-mode.
   Increase to 500 single + 200 multi, widen k range to [0.5, 7.0], add more multi-mode cases.
   Hypothesis: memorization is the root cause of the generalization gap.

2. **Regularization — weight decay**: Add L2 weight decay (1e-4 to 1e-3) to optimizer.
   Hypothesis: overfit parameters; weight decay will shrink them.

3. **Regularization — latent bottleneck tightening**: Reduce k_dim (20 → 12 or 8).
   Tighter bottleneck forces the encoder to generalize, not memorize.
   Risk: too small k_dim may hurt NM-ROM accuracy.

4. **Spectral input coverage**: Increase `n_spectral` (8 → 12 or 16) so the encoder
   sees more frequency content. At 8, the DCT keeps 8³=512 of 32³=32768 coefficients — that
   is only 1.6% of the signal, which may miss important variation.

5. **Loss function — physics-informed term**: Add a soft Dirichlet BC penalty:
   `loss = MSE + λ * mean(u_pred[boundary]²)`. Since Poisson solutions vanish at boundaries
   this is a free supervisory signal with zero label noise.

6. **Dropout in encoder MLP**: Add dropout (p=0.1–0.2) between hidden layers to prevent
   co-adaptation. Remove at inference (standard).

### Validate scalability (after generalization is fixed at N=32)

7. **Run at N=64**: Change `N=32` → `N=64` in `INR-Autoencoder.py` (line ~50).
   Check: does training complete without NaN? What are the errors? How long does it take?
   The DCT encoder is grid-independent in param count; the CP decoder grows linearly in N.

8. **Run at N=128**: Same as above for N=128. This is the stress test.
   Expected memory: decoder factor matrices are 3 × rank × 128 floats — fine.
   But full einsum decode produces 128³ = 2M floats — watch for OOM in batched training.

9. **Gradient clipping**: If N=64/128 shows explosion, add global norm clipping (max_norm=1.0)
   to the optimizer chain (`optax.chain(optax.clip_by_global_norm(1.0), optax.adam(...))`).

10. **Reduce CP rank for large N**: At N=128, rank=256 may be over-parameterized.
    Try rank=64 or rank=128 — fewer params, faster decode, potentially more stable.

### Architecture alternatives (if above fails)

11. **Neural field decoder (SIREN/FFN)**: Replace CP decoder with a small coordinate MLP
    that maps `(z, x, y, z_coord) → u(x,y,z)`. Grid-independent parameter count.
    Supports `decode_at_flat_indices` naturally (just query at those coordinates).
    Cost: slower full-field decode, but online NM-ROM only queries sparse points anyway.

12. **Hierarchical spectral decoder**: Instead of CP factors, decode to spectral coefficients
    (DCT) and iDCT back to physical space. Fully grid-independent; no factor matrices needed.

13. **Fourier features in encoder**: Replace DCT with random Fourier features + MLP encoder.
    More flexible, proven to avoid spectral bias for smooth functions.

---

## Decision rule

After extracting metrics, decide:

1. If crash (missing `=== AE TRAINING COMPLETE ===`) → **DISCARD**
2. If `ae_val_err >= 5e-2` → **DISCARD** (fails accuracy requirement)
3. If `ae_val_err < best_val_err_so_far` AND `ae_train_err < 5e-3` → **KEEP**
4. Otherwise → **DISCARD**

Track `best_val_err_so_far` yourself — starts at the baseline val error from your first run.

---

## Scaling test protocol

Once val_err < 1e-2 is achieved at N=32:

```python
# In INR-Autoencoder.py, change line ~50:
N = 64   # or 128
```

Run, record results. The same architecture should work — no other changes needed if it was
designed correctly. If it crashes:
- NaN loss → add gradient clipping or reduce learning rate
- OOM → reduce batch size, reduce rank, or use gradient checkpointing

---

## Failure modes to watch for

- **NaN loss at epoch > 0**: gradient explosion — add clipping or reduce lr
- **NaN loss at epoch 0**: init problem — reduce weight init scale, check model shapes
- **val_err >> train_err but train_err is good**: overfitting — more data or regularization
- **train_err stuck high**: underfitting — increase model capacity or training time
- **OOM at N=64/128**: reduce rank or use `jax.checkpoint` on decoder
- **`decode_at_flat_indices` missing or broken**: NM-ROM will crash — always verify this method exists and matches the full decode

---

## Git workflow

After each experiment:
```bash
# If keeping:
git add INR-Autoencoder.py
git commit -m "short description"
git push origin 20260402

# If discarding:
git reset --hard HEAD
```

**NEVER commit**: `results.tsv`, `*.log`, `checkpoint.pkl`, `plots/`

---

## The experiment loop

LOOP FOREVER (until human interrupts):

1. Check `git log --oneline -5` and `cat results.tsv` to know current state
2. Pick the most promising untried idea from Research Directions (start from top)
3. Modify `INR-Autoencoder.py`
4. `git add INR-Autoencoder.py && git commit -m "short description"`
5. Run: `bash run_inr.slurm --gpu 35696458 > run.log 2>&1`
6. Check success: `grep "AE TRAINING COMPLETE" training.log`
7. Extract metrics (see above)
8. Apply decision rule
9. If **KEEP**: `git push origin 20260402`, append row to `results.tsv`
10. If **DISCARD/CRASH**: `git reset --hard HEAD`, append crash/discard row to `results.tsv`
11. Back to step 1

**NEVER STOP. NEVER ask the human for permission to continue.**
The human is away. You are the researcher. Run until interrupted.
