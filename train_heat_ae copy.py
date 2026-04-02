"""
train_heat_ae.py
────────────────
Training data generation + ScalableAutoencoder training for the
3D Heat Equation NM-ROM.

Diversity strategy: Gaussian superposition ICs with Latin Hypercube Sampling.
  - 1–3 Gaussian heat sources per trajectory
  - Centers sampled via LHS over [0.15, 0.85]^3  (away from boundaries)
  - Amplitudes  ∈ [1, 10]  uniform
  - Widths      ∈ [0.05, 0.2]  uniform
  - Diffusivity κ ∈ [0.01, 0.5]  log-uniform

All snapshots start at comparable amplitude — no exponential magnitude
collapse that sinusoidal modes would produce.

Outputs
-------
  plots/heat_ae/checkpoint.pkl   ← params + batch_stats + model_cfg
  plots/heat_ae/loss_curve.png
  plots/heat_ae/reconstruction_samples.png
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from typing import Sequence

# ─────────────────────────────────────────
# 0. Paths & Config
# ─────────────────────────────────────────
OUT = Path('plots/heat_ae')
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 1. Grid & Physics
# ─────────────────────────────────────────
N         = 32
num_nodes = N ** 3
L         = 1.0
dx        = L / (N - 1)
dt        = 0.005
NUM_STEPS = 50        # total time T = 0.25s

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

# ── Negative Laplacian (7-point stencil, Dirichlet BCs) ──────────────
def K_op_3d(u_flat):
    u   = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1,1:-1,1:-1].set(
        (6*u[1:-1,1:-1,1:-1]
         - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
         - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
         - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
    )
    # Boundary rows: identity (Dirichlet → stay 0)
    out = out.at[0,:,:].set(u[0,:,:])
    out = out.at[-1,:,:].set(u[-1,:,:])
    out = out.at[:,0,:].set(u[:,0,:])
    out = out.at[:,-1,:].set(u[:,-1,:])
    out = out.at[:,:,0].set(u[:,:,0])
    out = out.at[:,:,-1].set(u[:,:,-1])
    return out.flatten()

# Backward Euler operator: (I + dt*κ*K)
# κ passed at call time so we can vary diffusivity per trajectory
def implicit_op(u_flat, kappa):
    return u_flat + dt * kappa * K_op_3d(u_flat)

# Boundary mask
mask_3d = jnp.ones((N,N,N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()

print(f"Grid: {N}³ = {num_nodes:,} nodes  |  dt={dt}  T={dt*NUM_STEPS:.3f}s")

# ─────────────────────────────────────────
# 2. Gaussian IC Generation
# ─────────────────────────────────────────
def make_gaussian_ic(centers, amplitudes, widths):
    """
    u0(x,y,z) = Σ_i A_i * exp(-|r - c_i|² / (2σ_i²))
    BCs enforced by multiplying a smooth boundary decay.

    centers:    (M, 3)  in [0,1]^3
    amplitudes: (M,)
    widths:     (M,)
    """
    u = jnp.zeros((N, N, N))
    for (cx, cy, cz), A, sigma in zip(centers, amplitudes, widths):
        u = u + A * jnp.exp(
            -((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2)
        )
    # Hard-zero the boundaries
    u = u.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    u = u.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    u = u.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    return u.flatten()


def sample_trajectory_params(rng, n_traj):
    """
    Latin Hypercube Sampling over trajectory parameter space.

    Parameter vector per trajectory (dim = 11):
      [0]     num_gaussians (1–3, encoded as 0–1 → rounded)
      [1–3]   center_1  (x, y, z)
      [4–6]   center_2
      [7–9]   center_3
      [10]    amplitude_scale  (0–1 → [1, 10])
      [11]    width_scale      (0–1 → [0.05, 0.2])
      [12]    kappa_log        (0–1 → log[0.01, 0.5])

    Returns list of dicts, one per trajectory.
    """
    sampler = qmc.LatinHypercube(d=13, seed=rng)
    samples = sampler.random(n=n_traj)          # (n_traj, 13)  all in [0,1]

    trajectories = []
    for s in samples:
        n_gauss = int(np.round(1 + 2 * s[0]))   # 1, 2, or 3

        centers    = []
        amplitudes = []
        widths     = []

        for g in range(n_gauss):
            # Centers safely away from boundary
            cx = 0.15 + 0.70 * s[1 + g*3]
            cy = 0.15 + 0.70 * s[2 + g*3]
            cz = 0.15 + 0.70 * s[3 + g*3]
            centers.append((cx, cy, cz))
            amplitudes.append(1.0 + 9.0 * s[10])   # [1, 10]
            widths.append(0.05 + 0.15 * s[11])      # [0.05, 0.20]

        kappa = float(np.exp(np.log(0.01) + (np.log(0.5) - np.log(0.01)) * s[12]))

        trajectories.append(dict(
            centers=centers, amplitudes=amplitudes,
            widths=widths, kappa=kappa
        ))
    return trajectories


# ─────────────────────────────────────────
# 3. FOM Time-Stepping (Backward Euler + CG)
# ─────────────────────────────────────────
def run_fom(u0_flat, kappa, steps):
    """
    Returns snapshot array of shape (steps+1, num_nodes).
    Backward Euler: (I + dt*κ*K) u_{n+1} = u_n
    (no source → F=0)
    """
    snapshots = [u0_flat]
    u = u0_flat
    op = lambda v: implicit_op(v, kappa)
    for _ in range(steps):
        # RHS = u_n  (F=0)
        u, _ = jax_linalg.cg(op, u, x0=u, tol=1e-6, maxiter=1000)
        snapshots.append(u)
    return jnp.stack(snapshots)    # (steps+1, num_nodes)


# ─────────────────────────────────────────
# 4. Generate Training Data
# ─────────────────────────────────────────
N_TRAIN = 200     # trajectories — reduce to 100 if RAM is tight
N_VAL   =  20     # held-out trajectories

print(f"\n── Generating {N_TRAIN} training + {N_VAL} validation trajectories ──")
print(f"   Each: {NUM_STEPS+1} snapshots  →  "
      f"Total snapshots ≈ {N_TRAIN*(NUM_STEPS+1):,}")

train_params = sample_trajectory_params(rng=42,        n_traj=N_TRAIN)
val_params   = sample_trajectory_params(rng=1337,      n_traj=N_VAL)

# ── Collect all snapshots (may take a few minutes) ───────────────────
all_snapshots = []     # will be (N_TRAIN*(steps+1), num_nodes)
traj_kappas   = []     # kappa per trajectory (for EQ phase)
traj_starts   = []     # index into all_snapshots where each trajectory starts

t0 = time.perf_counter()
for i, tp in enumerate(train_params):
    u0   = make_gaussian_ic(tp['centers'], tp['amplitudes'], tp['widths'])
    traj = run_fom(u0, tp['kappa'], NUM_STEPS)    # (51, num_nodes)
    traj_starts.append(len(all_snapshots))
    all_snapshots.append(traj)
    traj_kappas.append(tp['kappa'])
    if (i+1) % 50 == 0:
        elapsed = time.perf_counter() - t0
        print(f"   Train {i+1}/{N_TRAIN}  ({elapsed:.0f}s elapsed)")

U_train = jnp.concatenate(all_snapshots, axis=0)   # (N_TRAIN*51, num_nodes)
print(f"   Training snapshots: {U_train.shape}")

val_snapshots = []
val_kappas    = []
for i, vp in enumerate(val_params):
    u0   = make_gaussian_ic(vp['centers'], vp['amplitudes'], vp['widths'])
    traj = run_fom(u0, vp['kappa'], NUM_STEPS)
    val_snapshots.append(traj)
    val_kappas.append(vp['kappa'])
    if (i+1) % 10 == 0:
        print(f"   Val {i+1}/{N_VAL}")

U_val = jnp.concatenate(val_snapshots, axis=0)
print(f"   Validation snapshots: {U_val.shape}")
print(f"   Data generation: {time.perf_counter()-t0:.1f}s")

# ─────────────────────────────────────────
# 5. Model Definition
# ─────────────────────────────────────────
class AttentionPooling(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, feat_map):
        C      = feat_map.shape[-1]
        tokens = feat_map.reshape(-1, C)
        tokens = nn.Dense(self.latent_dim)(tokens)
        query  = self.param('query', nn.initializers.normal(0.02), (self.latent_dim,))
        scale  = jnp.sqrt(jnp.float32(self.latent_dim))
        scores = jnp.einsum('td,d->t', tokens, query) / scale
        w      = jax.nn.softmax(scores, axis=0)
        return jnp.einsum('t,td->d', w, tokens)


class Conv3DEncoder(nn.Module):
    latent_dim: int
    features:   Sequence[int] = (32, 64, 128)
    pool_size:  int = 4

    @nn.compact
    def __call__(self, x, training: bool = False):
        h = x[..., None]
        for feat in self.features:
            h = nn.Conv(feat, kernel_size=(3,3,3),
                        strides=(2,2,2), padding='SAME')(h)
            h = nn.BatchNorm(use_running_average=not training, momentum=0.9)(h)
            h = nn.leaky_relu(h, negative_slope=0.2)
        H, W, D, C = h.shape
        if H != self.pool_size:
            h = jax.image.resize(h,
                                 (self.pool_size, self.pool_size,
                                  self.pool_size, C),
                                 method='linear')
        return AttentionPooling(self.latent_dim)(h)


class SeparableDecoder(nn.Module):
    latent_dim:  int
    rank:        int = 256
    grid_size:   int = 32
    hidden_dims: Sequence[int] = (256, 512)

    def setup(self):
        self.hidden_layers = [nn.Dense(d) for d in self.hidden_dims]
        self.to_rank       = nn.Dense(self.rank)
        init = nn.initializers.normal(0.01)
        Ng   = self.grid_size
        self.W_x  = self.param('W_x',  init, (self.rank, Ng))
        self.W_y  = self.param('W_y',  init, (self.rank, Ng))
        self.W_z  = self.param('W_z',  init, (self.rank, Ng))
        self.bias = self.param('bias', nn.initializers.zeros, ())

    def _mlp_body(self, z):
        h = z
        for layer in self.hidden_layers:
            h = nn.swish(layer(h))
        return self.to_rank(h)

    def __call__(self, z):
        h    = self._mlp_body(z)
        u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
        return u_3d.flatten() + self.bias


class ScalableAutoencoder(nn.Module):
    latent_dim:    int
    rank:          int = 256
    grid_size:     int = 32
    conv_features: Sequence[int] = (32, 64, 128)
    hidden_dims:   Sequence[int] = (256, 512)

    def setup(self):
        self.encoder = Conv3DEncoder(latent_dim=self.latent_dim,
                                     features=self.conv_features)
        self.decoder = SeparableDecoder(latent_dim=self.latent_dim,
                                        rank=self.rank,
                                        grid_size=self.grid_size,
                                        hidden_dims=self.hidden_dims)

    def encode(self, u_flat, training=False):
        u_3d = u_flat.reshape(self.grid_size, self.grid_size, self.grid_size)
        return self.encoder(u_3d, training=training)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, u_flat, training=False):
        return self.decode(self.encode(u_flat, training=training))


# ─────────────────────────────────────────
# 6. Model Init
# ─────────────────────────────────────────
k_dim = 20
model = ScalableAutoencoder(
    latent_dim    = k_dim,
    rank          = 256,
    grid_size     = N,
    conv_features = (32, 64, 128),
    hidden_dims   = (256, 512),
)

key       = jax.random.PRNGKey(0)
variables = model.init({'params': key}, U_train[0], training=True)
params      = variables['params']
batch_stats = variables['batch_stats']

n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\n── Model: {n_params:,} parameters  (latent_dim={k_dim}, rank=256) ──")

# ─────────────────────────────────────────
# 7. Training
# ─────────────────────────────────────────
BATCH_SIZE = 64      # snapshot minibatch — keeps memory flat
NUM_EPOCHS = 8_000
LOG_EVERY  = 1_000

schedule  = optax.warmup_cosine_decay_schedule(
    init_value=0., peak_value=1e-3,
    warmup_steps=500, decay_steps=NUM_EPOCHS, end_value=1e-5
)
tx        = optax.adam(schedule)
opt_state = tx.init(params)

key = jax.random.PRNGKey(1)

@jax.jit
def train_step(params, batch_stats, opt_state, batch):
    def loss_fn(p):
        # vmap over batch — single-sample model, batch via vmap
        preds = jax.vmap(
            lambda u: model.apply(
                {'params': p, 'batch_stats': batch_stats},
                u, training=False       # batch_stats updated separately
            )
        )(batch)
        return jnp.mean((batch - preds) ** 2)

    loss, grads          = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_st  = tx.update(grads, opt_state, params)
    new_params           = optax.apply_updates(params, updates)

    # Update running stats on a single sample (avoids vmap+mutable clash)
    _, upd = model.apply(
        {'params': new_params, 'batch_stats': batch_stats},
        batch[0], training=True, mutable=['batch_stats']
    )
    return new_params, upd['batch_stats'], new_opt_st, loss


@jax.jit
def eval_loss(params, batch_stats, batch):
    preds = jax.vmap(
        lambda u: model.apply(
            {'params': params, 'batch_stats': batch_stats},
            u, training=False
        )
    )(batch)
    return jnp.mean((batch - preds) ** 2)


print(f"\n── Training  ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}) ──")
train_losses, val_losses = [], []
n_train = len(U_train)
t0      = time.perf_counter()

for epoch in range(NUM_EPOCHS + 1):
    # Random minibatch
    key, subkey = jax.random.split(key)
    idx   = jax.random.choice(subkey, n_train, shape=(BATCH_SIZE,), replace=False)
    batch = U_train[idx]

    params, batch_stats, opt_state, loss = train_step(
        params, batch_stats, opt_state, batch
    )

    if epoch % LOG_EVERY == 0:
        # Eval on random val subset
        v_idx  = jax.random.choice(subkey, len(U_val), shape=(BATCH_SIZE,), replace=False)
        v_loss = float(eval_loss(params, batch_stats, U_val[v_idx]))
        train_losses.append((epoch, float(loss)))
        val_losses.append((epoch, v_loss))
        print(f"  Epoch {epoch:5d} | train {float(loss):.4e} | val {v_loss:.4e} "
              f"| {time.perf_counter()-t0:.0f}s")

# ─────────────────────────────────────────
# 8. Save Checkpoint
# ─────────────────────────────────────────
ckpt = {
    'params':      params,
    'batch_stats': batch_stats,
    'model_cfg': dict(
        latent_dim    = k_dim,
        rank          = 256,
        grid_size     = N,
        conv_features = (32, 64, 128),
        hidden_dims   = (256, 512),
    ),
    # Save training metadata for EQ phase
    'train_meta': dict(
        n_train     = N_TRAIN,
        num_steps   = NUM_STEPS,
        dt          = dt,
        traj_kappas = traj_kappas,
        traj_starts = traj_starts,
    )
}
with open(OUT / 'checkpoint.pkl', 'wb') as f:
    pickle.dump(ckpt, f)
print(f"\n  Checkpoint saved: {OUT / 'checkpoint.pkl'}")

# ─────────────────────────────────────────
# 9. Diagnostic Plots
# ─────────────────────────────────────────

# Loss curve
ep_t, lo_t = zip(*train_losses)
ep_v, lo_v = zip(*val_losses)
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(ep_t, lo_t, label='Train MSE', color='#1f77b4', lw=2)
ax.semilogy(ep_v, lo_v, label='Val MSE',   color='#ff7f0e', lw=2, ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (log)')
ax.set_title('Heat AE — Training Curve')
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / 'loss_curve.png', dpi=150)
plt.close()

# Reconstruction samples — show 3 trajectories, 3 time points each
def encode(u):
    return model.apply({'params': params, 'batch_stats': batch_stats},
                        u, training=False, method=model.encode)
def decode(z):
    return model.apply({'params': params, 'batch_stats': batch_stats},
                        z, method=model.decode)

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
sample_trajs = [0, N_TRAIN//2, N_TRAIN-1]
time_idxs    = [0, NUM_STEPS//2, NUM_STEPS]

for row, ti in enumerate(sample_trajs):
    traj = all_snapshots[ti]                 # (51, num_nodes)
    kap  = traj_kappas[ti]
    mid  = N // 2
    for col, t_idx in enumerate(time_idxs):
        u_true = traj[t_idx]
        z      = encode(u_true)
        u_rec  = decode(z)
        err    = float(jnp.linalg.norm(u_rec - u_true) / (jnp.linalg.norm(u_true) + 1e-10))
        u_true_3d = np.array(u_true).reshape(N,N,N)
        u_rec_3d  = np.array(u_rec ).reshape(N,N,N)
        vmax = u_true_3d[:,:,mid].max()
        kw   = dict(origin='lower', cmap='magma', vmin=0, vmax=vmax, aspect='auto')
        axes[row, col*2  ].imshow(u_true_3d[:,:,mid].T, **kw)
        axes[row, col*2  ].set_title(f'FOM t={t_idx*dt:.2f}', fontsize=8)
        axes[row, col*2  ].axis('off')
        axes[row, col*2+1].imshow(u_rec_3d[:,:,mid].T, **kw)
        axes[row, col*2+1].set_title(f'Rec  ε={err:.2e}', fontsize=8)
        axes[row, col*2+1].axis('off')
    axes[row, 0].set_ylabel(f'κ={kap:.3f}', fontsize=9)

fig.suptitle('Heat AE Reconstruction — FOM vs Decoded (z=mid slice)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'reconstruction_samples.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Plots saved to {OUT}/")
print("\n=== Training complete — run nmrom_heat.py next ===")
