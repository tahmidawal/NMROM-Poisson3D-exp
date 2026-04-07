# autoresearch — NM-ROM for 3D Poisson (Nonlinear Manifold ROM)

Autonomous research loop to improve the NM-ROM for the 3D Poisson equation —
maximize solver speedup while keeping accuracy acceptable.

---

## Scientific Goal

Solve parameterized 3D Poisson PDEs:
    -∇²u = F(x;k)    on [0,1]³,    u=0 on ∂Ω

Forcing: F(x;k) = 10·sin(k₁πx)·sin(k₂πy)·sin(k₃πz), k∈{1..5}³ (125 parameter cases).
Grid: 32³ = 32,768 DOF. Finite difference, 7-point stencil.

### Two-stage pipeline

**Stage 1 — Autoencoder** (`INR-Autoencoder.py`):
- `Conv3DEncoder`: 3 strided conv layers (32→64→128 channels) + attention pooling → latent z ∈ ℝ^k_dim
- `SeparableDecoder`: MLP → rank-256 CP factored field: ũ(i,j,k) = h(z)·(W_x[:,i]⊙W_y[:,j]⊙W_z[:,k])
- Trained with k²-normalized snapshots (125 analytical solutions)
- Saves checkpoint to `checkpoint.pkl`

**Stage 2 — NM-ROM solver** (`NMROM-INR-Poisson-3D.py`):
- Loads checkpoint, rebuilds training snapshots
- **Empirical Quadrature (EQ) offline**: NNLS finds sparse "magic points" where residual must be evaluated
- **Online solver**: Levenberg-Marquardt Gauss-Newton (LM-GN) in latent space, evaluated only at EQ points
- Benchmarks ROM vs FOM (conjugate gradient) on 14 test cases

### Key hyperparameters (current baseline)
- `k_dim = 20` (latent dimension) — controls LM-GN cost per step: O(k_dim²)
- `rank = 256` (CP rank) — controls decoder eval cost at EQ points
- `conv_features = (32, 64, 128)` — encoder size
- `hidden_dims = (256, 512)` — decoder MLP size
- `NUM_EPOCHS = 20000` — autoencoder training epochs
- `max_iters = 30` (LM-GN iterations) — main online cost driver
- NNLS threshold `1e-10` — controls EQ sparsity

---

## Cluster environment

- **Working directory**: `/cluster/tufts/paralab/tawal01/NMROM/autoresearch-Poisson3D/Poisson_3D/`
- **Python env**: `module load python/3.10.4 cuda/12.2 cudnn/8.9.7-12.x` + `PYTHONPATH=/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages:$PYTHONPATH`
- **GPU node in use**: `cc1gpu005` (SLURM job 35595595 — A100 80GB, already allocated)
- **Git remote**: `https://github.com/tahmidawal/NMROM-Poisson3D-exp.git`
- **Current branch**: `autoresearch/apr6`

---

## In-scope files (you modify ONLY these two)

- `INR-Autoencoder.py` — autoencoder architecture, training hyperparameters
- `NMROM-INR-Poisson-3D.py` — EQ offline phase, LM-GN solver, hyperparameters

## Fixed (do NOT modify)

- `run_inr.slurm` — launcher script
- `program.md` — this file

---

## Running an experiment

```bash
cd /cluster/tufts/paralab/tawal01/NMROM/autoresearch-Poisson3D/Poisson_3D

# If you changed INR-Autoencoder.py (retrains from scratch):
bash run_inr.slurm --replace --gpu 35595595 > run.log 2>&1

# If you only changed NMROM-INR-Poisson-3D.py (reuses saved checkpoint):
bash run_inr.slurm --gpu 35595595 > run.log 2>&1
```

The `--gpu 35595595` flag tells the launcher to SSH directly to `cc1gpu005` (the already-allocated A100).
If that job dies, it will automatically submit a new SLURM job instead.

**Timeout**: If a run exceeds 20 minutes, kill it (`Ctrl+C`) and treat as crash.
- Autoencoder training alone: ~70-90 seconds on A100
- NM-ROM offline phase: ~30-60 seconds
- Total typical run: ~2-4 minutes

---

## Extracting results

```bash
# Primary metrics
grep "Avg speedup\|ROM vs analytical\|EQ nodes\|Avg FOM\|Avg ROM" nmrom.log

# Autoencoder quality
grep "Mean train rec\|Mean val rec" training.log

# Check for crashes
tail -30 run.log
```

---

## Metrics

| Metric | Source | Goal |
|--------|--------|------|
| `avg_speedup` | `nmrom.log` | **MAXIMIZE** (currently 13.83×) |
| `rom_vs_analytical` | `nmrom.log` | Keep < 5e-2 (currently 9.76e-03) |
| `ae_train_err` | `training.log` | Informational (currently 4.04e-03) |
| `eq_sparsity %` | `nmrom.log` | Informational (currently 82.4%) |

---

## Logging results

Log to `results.tsv` (tab-separated). Do NOT commit this file.

Header:
```
commit	speedup	rom_error	ae_train_err	eq_sparsity	status	description
```

---

## Baseline (already established — do NOT re-run)

```
commit	speedup	rom_error	ae_train_err	eq_sparsity	status	description
bdafb14	13.83	9.76e-03	4.04e-03	82.4	keep	baseline
```

Add this to `results.tsv` as your first entry, then start experimenting.

---

## Research directions (roughly ordered by expected impact)

### High impact — try first
1. **Reduce `k_dim`** (20 → 12 or 8): Fewer latent dims = fewer unknowns in LM-GN = faster Newton steps. May hurt AE quality.
2. **Reduce `max_iters`** in LM-GN (30 → 15 or 10): Direct 2× speedup if accuracy holds.
3. **Reduce CP `rank`** (256 → 128 or 64): Faster decoder eval at EQ points. May hurt AE quality.
4. **Combine k_dim↓ + max_iters↓**: Multiplicative speedup.

### Medium impact
5. **Tighten EQ threshold** (1e-10 → 1e-8 or 1e-6): Fewer EQ points → faster residual eval per GN step.
6. **Reduce `hidden_dims`** (256,512 → 128,256): Faster MLP in decoder.
7. **Reduce `conv_features`** (32,64,128 → 16,32,64): Smaller encoder.
8. **Reduce `NUM_EPOCHS`** (20000 → 10000): If AE quality is already good enough.

### Exploratory
9. **Looser convergence tolerance**: Stop LM-GN earlier when residual is "small enough" rather than fixed iters.
10. **Smaller grid for AE**: Train AE on coarser features.
11. **Combined aggressive**: k_dim=8, rank=64, max_iters=10, hidden_dims=(64,128).

---

## Git workflow

After each experiment:
```bash
# If keeping:
git add Poisson_3D/INR-Autoencoder.py Poisson_3D/NMROM-INR-Poisson-3D.py
git commit -m "short description"
git push origin autoresearch/apr6

# If discarding:
git reset --hard HEAD
```

**NEVER commit**: `results.tsv`, `*.log`, `checkpoint.pkl`, `plots/`

---

## The experiment loop

LOOP FOREVER (until human interrupts):

1. Check `git log --oneline -5` to see current state
2. Pick the most promising untried idea from Research Directions above
3. Modify `INR-Autoencoder.py` and/or `NMROM-INR-Poisson-3D.py`
4. `git add` the changed files, `git commit -m "description"`
5. Run: `bash run_inr.slurm [--replace] --gpu 35595595 > run.log 2>&1`
6. Wait for completion. If >20 min → kill, treat as crash.
7. Extract metrics from `nmrom.log` and `training.log`
8. Append row to `results.tsv`
9. **If speedup > current best AND rom_error < 5e-2**: keep commit, `git push origin autoresearch/apr6`
10. **Otherwise**: `git reset --hard HEAD`, revert to previous state
11. Back to step 1

**NEVER STOP. NEVER ask the human for permission to continue.**
The human is away. You are the researcher. Run until interrupted.
