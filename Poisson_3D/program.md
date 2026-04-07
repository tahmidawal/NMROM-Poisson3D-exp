# autoresearch — NM-ROM for 3D Poisson (Nonlinear Manifold ROM)

This is an autonomous research loop to improve the NM-ROM (Nonlinear Manifold Reduced-Order Model)
for the 3D Poisson equation, maximizing solver speedup while maintaining acceptable accuracy.

---

## Scientific Goal

Solve parameterized 3D Poisson PDEs:
    -∇²u = F(x;k)    on [0,1]³,    u=0 on ∂Ω

with forcing F(x;k) = 10·sin(k₁πx)·sin(k₂πy)·sin(k₃πz), k∈{1..5}³ (125 parameter cases).

The pipeline has two stages:
1. **Autoencoder** (`INR-Autoencoder.py`): Train a ScalableAutoencoder that compresses the 32³=32,768-DOF
   solution field into a low-dimensional latent code z ∈ ℝ^k_dim via a CP-factored decoder.
2. **NM-ROM solver** (`NMROM-INR-Poisson-3D.py`): Using the trained encoder/decoder, solve the PDE
   entirely in latent space via Levenberg-Marquardt Gauss-Newton (LM-GN) on sparse Empirical
   Quadrature (EQ) points, bypassing the full 32,768-DOF system.

**Primary metric**: `avg_speedup` (FOM time / ROM time) — higher is better.
**Secondary metric**: `rom_vs_analytical` (relative L2 error) — must stay below ~5e-2 to be useful.

The research goal: push speedup as high as possible (currently ~14×) without blowing up accuracy.

---

## In-scope files (you modify these)

- `INR-Autoencoder.py` — autoencoder architecture, training loop, hyperparameters
- `NMROM-INR-Poisson-3D.py` — EQ offline phase, LM-GN online solver, hyperparameters

## Fixed (do NOT modify)

- `run_inr.slurm` — the launcher, do not touch
- `program.md` — this file

---

## Setup

1. Agree on a run tag (e.g. `apr6`). Branch `autoresearch/<tag>` must not exist yet.
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Read both Python files fully for context.
4. Initialize `results.tsv` with the header row.
5. Run the baseline first (with existing code as-is).

---

## Running an experiment

```bash
cd /cluster/tufts/paralab/tawal01/NMROM/autoresearch-Poisson3D/Poisson_3D
bash run_inr.slurm --replace > run.log 2>&1
```

Use `--replace` to force retrain the autoencoder when you've changed `INR-Autoencoder.py`.
Omit `--replace` if you only changed `NMROM-INR-Poisson-3D.py` (reuses saved checkpoint).

Extract results:
```bash
grep "Avg speedup\|ROM vs analytical\|Avg FOM\|Avg ROM" Poisson_3D/nmrom.log
grep "Mean train\|Mean val" Poisson_3D/training.log
```

---

## Output metrics to track

From `nmrom.log`:
- `Avg speedup: X×` — **PRIMARY metric, maximize this**
- `ROM vs analytical: X` — secondary, keep < 5e-2
- `Avg FOM time` / `Avg ROM time`
- `EQ nodes: X / 32768 (Y%)` — sparsity of quadrature

From `training.log`:
- `Mean train reconstruction: X`
- `Mean val reconstruction: X`

---

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated).

Header:
```
commit	speedup	rom_error	ae_train_err	eq_sparsity	status	description
```

Columns:
1. git commit hash (7 chars)
2. avg_speedup (e.g. 14.23)
3. rom_vs_analytical error (e.g. 9.76e-03)
4. autoencoder mean train reconstruction error (e.g. 4.80e-03)
5. EQ sparsity % (e.g. 82.4)
6. status: `keep`, `discard`, or `crash`
7. short description of what was tried

Example:
```
commit	speedup	rom_error	ae_train_err	eq_sparsity	status	description
a1b2c3d	13.83	9.76e-03	4.80e-03	82.4	keep	baseline
b2c3d4e	17.20	1.20e-02	5.10e-03	71.2	keep	reduce latent_dim 20→12
c3d4e5f	8.50	2.10e-02	6.80e-03	82.0	discard	reduce CP rank 256→64 (too slow)
```

---

## Research directions to explore

**Autoencoder changes (require --replace):**
- Reduce `k_dim` (latent dim): smaller latent → faster LM-GN convergence (fewer unknowns)
- Reduce CP `rank`: fewer rank terms → faster decoder evaluation in EQ
- Reduce `conv_features` size: smaller encoder → faster encoding
- Fewer encoder layers
- Smaller `hidden_dims` in decoder MLP
- Reduce training epochs if reconstruction quality is already good enough
- Change normalization strategy

**NM-ROM solver changes (no --replace needed):**
- Reduce `max_iters` in LM-GN (e.g. 30 → 15 or 10) — fewer GN steps = faster online solve
- Tighten EQ threshold (increase `1e-10` → `1e-8`) — fewer EQ points = faster residual eval
- Reduce NNLS iterations (5000 → 2000) — faster offline phase (doesn't affect online speed)
- Tune LM damping `lam` strategy
- Change backtracking line search steps

**Combined:**
- The main speedup lever is: smaller latent_dim + fewer GN iterations + sparser EQ
- The accuracy floor is: autoencoder reconstruction quality

---

## The experiment loop

LOOP FOREVER (until human interrupts):

1. Check current git state
2. Form a hypothesis about what will improve speedup
3. Modify `INR-Autoencoder.py` and/or `NMROM-INR-Poisson-3D.py`
4. `git commit -am "description of change"`
5. Run: `bash run_inr.slurm [--replace] > run.log 2>&1`
6. If it hangs >15 min, kill it (Ctrl+C or scancel), treat as crash
7. Extract metrics from logs
8. Log to `results.tsv`
9. If speedup improved AND error < 5e-2 → keep commit, advance branch
10. Otherwise → `git reset --hard HEAD~1`, discard
11. Go to step 1

**NEVER STOP.** Do not ask the human for permission to continue. Run indefinitely.
If stuck, think harder: try combinations of previous near-misses, more aggressive hyperparameter
changes, or radically different approaches.
