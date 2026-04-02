"""
train_heat_sin_ae.py
────────────────────
ScalableAutoencoder for the parametric 3D Heat Equation with sinusoidal ICs.

Physics:
  ∂u/∂t = κ ∇²u   on [0,1]³,   u=0 on ∂Ω

Sinusoidal IC with amplitude-normalised to midpoint T/2:
  u₀(x,y,z; k,κ) = A(k,κ) · sin(k₁πx)·sin(k₂πy)·sin(k₃πz)
  A(k,κ)          = exp(κ·λ_k·T/2),  λ_k = (k₁²+k₂²+k₃²)π²

This makes u(x,y,z,T/2) = sin(…) for ALL (k,κ) — all trajectories pass
through amplitude 1 at the midpoint.  Early snapshots ≈ e^(λT/2) > 1,
late snapshots ≈ e^(-λT/2) < 1 — always within a factor of e^(λ_max·T/2)
of each other.  For λ_max·T/2 < 6 (our parameter range) that's < 400×
reduction in range vs the unnormalised case (millions×).

Exact analytical solution (used for FOM AND ROM validation):
  u(x,y,z,t) = A(k,κ) · exp(-κ·λ_k·t) · sin(k₁πx)·sin(k₂πy)·sin(k₃πz)

Parameter space:
  k₁, k₂, k₃ ∈ {1,2,3,4}   → 64 mode combinations
  κ           ∈ {0.01, 0.05, 0.1, 0.5}  → 4 diffusivities
  Total:        256 training trajectories  (+ 20 val, 14 test)

Data generation: ~4× faster than Gaussian case — IC is just sin evaluation,
no LHS required, and the structured parameter grid makes EQ much cheaper.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from typing import Sequence
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────
OUT = Path('plots/heat_sin_ae')
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 1. Grid & Physics
# ─────────────────────────────────────────
N         = 32
num_nodes = N ** 3
L         = 1.0
dx        = L / (N - 1)
dt        = 0.005
NUM_STEPS = 50        # T = 0.25s
T_FINAL   = dt * NUM_STEPS

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

# CoordConv grid (normalised to [-1,1])
COORD_GRID = jnp.stack([2*X/L-1, 2*Y/L-1, 2*Z/L-1], axis=-1)

def K_op_3d(u_flat):
    u   = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1,1:-1,1:-1].set(
        (6*u[1:-1,1:-1,1:-1]
         - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
         - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
         - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
    )
    out = out.at[0,:,:].set(u[0,:,:]);  out = out.at[-1,:,:].set(u[-1,:,:])
    out = out.at[:,0,:].set(u[:,0,:]);  out = out.at[:,-1,:].set(u[:,-1,:])
    out = out.at[:,:,0].set(u[:,:,0]);  out = out.at[:,:,-1].set(u[:,:,-1])
    return out.flatten()

def implicit_op(u_flat, kappa):
    return u_flat + dt * kappa * K_op_3d(u_flat)

def run_fom(u0_flat, kappa, steps):
    """Run FOM for a single trajectory."""
    snapshots = [u0_flat]
    u  = u0_flat
    op = lambda v: implicit_op(v, kappa)
    for _ in range(steps):
        u, _ = jax_linalg.cg(op, u, x0=u, tol=1e-7, maxiter=2000)
        snapshots.append(u)
    return jnp.stack(snapshots)

def run_fom_single(params_tuple):
    """Wrapper for parallel execution — returns (idx, traj, params)."""
    idx, k1, k2, k3, kappa = params_tuple
    u0   = jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z)
    u0   = u0.flatten()
    traj = run_fom(u0, kappa, NUM_STEPS)
    return idx, traj, (k1, k2, k3, kappa)

mask_3d = jnp.ones((N,N,N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()
u_g     = jnp.zeros(num_nodes)

print(f"Grid: {N}^3 = {num_nodes:,} nodes  |  dt={dt}  T={T_FINAL:.3f}s")

# ─────────────────────────────────────────
# 2. Sinusoidal IC Generation & Analytical Solution
# ─────────────────────────────────────────
def lambda_k(k1, k2, k3):
    """Laplacian eigenvalue for mode (k1,k2,k3)."""
    return float((k1**2 + k2**2 + k3**2) * np.pi**2)

def decay_at_T(k1, k2, k3, kappa):
    """Fraction of initial amplitude remaining at t=T."""
    return float(np.exp(-kappa * lambda_k(k1,k2,k3) * T_FINAL))

def make_sin_ic(k1, k2, k3, kappa=None):
    """
    u₀ = sin(k₁πx)·sin(k₂πy)·sin(k₃πz)  — amplitude always 1.
    BCs automatically satisfied. Scale handled by AE per-sample normalisation.
    """
    u3d = jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z)
    return u3d.flatten()

def get_analytical(k1, k2, k3, kappa, t):
    """
    u_exact(t) = exp(-κ·λ_k·t) · sin(k₁πx)·sin(k₂πy)·sin(k₃πz)
    Amplitude=1 at t=0 for all modes.
    """
    dec = float(np.exp(-kappa * lambda_k(k1,k2,k3) * t))
    u3d = dec * jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z)
    return u3d.flatten()

# ─────────────────────────────────────────
# 3. Parameter Space — filtered by decay
#
# Exclude (k, κ) pairs where the solution decays below DECAY_THRESHOLD
# by t=T. Those trajectories are dominated by float32 rounding error
# in their late snapshots, which poisons AE training.
#
# DECAY_THRESHOLD = 1e-3 means we keep trajectories where at least
# 0.1% of the initial amplitude is still physically meaningful at t=T.
# ─────────────────────────────────────────
DECAY_THRESHOLD = 1e-3

KAPPAS_TRAIN = [0.01, 0.05, 0.1, 0.5]
K_RANGE      = range(1, 5)

all_combos   = [
    (k1, k2, k3, kappa)
    for k1 in K_RANGE for k2 in K_RANGE for k3 in K_RANGE
    for kappa in KAPPAS_TRAIN
]

train_params_full = [(k1,k2,k3,kap) for k1,k2,k3,kap in all_combos
                     if decay_at_T(k1,k2,k3,kap) >= DECAY_THRESHOLD]

# Validation: intermediate κ values, same filter
KAPPAS_VAL  = [0.03, 0.2]
val_params_full  = [
    (k1, k2, k3, kappa)
    for k1 in K_RANGE for k2 in K_RANGE for k3 in K_RANGE
    for kappa in KAPPAS_VAL
    if decay_at_T(k1,k2,k3,kappa) >= DECAY_THRESHOLD
]

# Subsample for faster iteration (set to None to use all)
N_TRAIN_MAX = None  # Use all 174 training trajectories
N_VAL_MAX   = None  # Use all validation trajectories

rng_sub = np.random.default_rng(seed=42)
if N_TRAIN_MAX and len(train_params_full) > N_TRAIN_MAX:
    idx_train = np.sort(rng_sub.choice(len(train_params_full), N_TRAIN_MAX, replace=False))
    train_params = [train_params_full[i] for i in idx_train]
else:
    train_params = train_params_full

if N_VAL_MAX and len(val_params_full) > N_VAL_MAX:
    idx_val = np.sort(rng_sub.choice(len(val_params_full), N_VAL_MAX, replace=False))
    val_params = [val_params_full[i] for i in idx_val]
else:
    val_params = val_params_full

N_TRAIN = len(train_params)
N_VAL   = len(val_params)

# Show what was filtered
n_filtered = len(all_combos) - len(train_params_full)
print(f"\nParameter space (after filtering decay < {DECAY_THRESHOLD}):")
print(f"  Original combos:   {len(all_combos)}")
print(f"  Filtered out:      {n_filtered}  (solution decays to noise by t=T)")
print(f"  Available:         {len(train_params_full)} train / {len(val_params_full)} val")
print(f"  Using:             {N_TRAIN} train / {N_VAL} val  (subsampled)")

# Show amplitude range — now always 1 at t=0
print(f"\nAmplitude: always 1.0 at t=0 for all modes (A=1).")
print(f"Decay range at t=T (only trajectories above threshold shown):")
decays = [decay_at_T(k1,k2,k3,kap) for k1,k2,k3,kap in train_params]
print(f"  min decay fraction: {min(decays):.4f}  ({min(decays)*100:.2f}% remaining)")
print(f"  max decay fraction: {max(decays):.4f}  ({max(decays)*100:.2f}% remaining)")
print(f"  → All training snapshots are physically meaningful throughout [0,T]")

# ─────────────────────────────────────────
# 4. Generate or Load Data
# ─────────────────────────────────────────
DATA_FILE = OUT / 'training_data.pkl'

if DATA_FILE.exists():
    print(f"\n-- Loading cached data from {DATA_FILE} --")
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    all_snapshots  = data['all_snapshots']
    traj_params    = data['traj_params']
    traj_kappas    = data['traj_kappas']
    traj_starts    = data['traj_starts']
    U_train        = jnp.array(data['U_train'])
    U_val          = jnp.array(data['U_val'])
    val_snapshots  = data['val_snapshots']
else:
    N_WORKERS = 4  # number of parallel workers
    
    print(f"\n-- Generating {N_TRAIN} training trajectories (parallel, {N_WORKERS} workers) --")
    t0 = time.perf_counter()
    
    # Prepare tasks: (idx, k1, k2, k3, kappa)
    train_tasks = [(i, *p) for i, p in enumerate(train_params)]
    
    # Results storage
    results = [None] * N_TRAIN
    completed = 0
    
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_fom_single, task): task[0] for task in train_tasks}
        for future in as_completed(futures):
            idx, traj, params = future.result()
            results[idx] = (traj, params)
            completed += 1
            elapsed = time.perf_counter() - t0
            avg_per = elapsed / completed
            eta = avg_per * (N_TRAIN - completed)
            k1, k2, k3, kappa = params
            print(f"   [{completed:3d}/{N_TRAIN}] k=({k1},{k2},{k3}) κ={kappa:.2f} | "
                  f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)
    
    # Unpack results in order
    all_snapshots = []
    traj_params   = []
    traj_kappas   = []
    traj_starts   = []
    for traj, (k1, k2, k3, kappa) in results:
        traj_starts.append(len(all_snapshots))
        all_snapshots.append(traj)
        traj_params.append((k1, k2, k3, kappa))
        traj_kappas.append(kappa)
    
    # Use only t=0 snapshot per trajectory (unique patterns, no temporal redundancy)
    U_train = jnp.stack([traj[0] for traj in all_snapshots])  # (N_TRAIN, num_nodes)
    print(f"   Training snapshots: {U_train.shape} (t=0 only, unique patterns)")

    print(f"\n-- Generating {N_VAL} validation trajectories (parallel) --")
    t0_val = time.perf_counter()
    val_tasks = [(i, *p) for i, p in enumerate(val_params)]
    val_results = [None] * N_VAL
    val_completed = 0
    
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_fom_single, task): task[0] for task in val_tasks}
        for future in as_completed(futures):
            idx, traj, params = future.result()
            val_results[idx] = traj
            val_completed += 1
            elapsed_v = time.perf_counter() - t0_val
            print(f"   [{val_completed:3d}/{N_VAL}] | {elapsed_v:.0f}s elapsed", flush=True)
    
    val_snapshots = val_results
    # Val uses all timesteps to test reconstruction across full temporal range
    U_val = jnp.concatenate(val_snapshots, axis=0)

    print(f"   Data generation: {time.perf_counter()-t0:.1f}s")

    with open(DATA_FILE, 'wb') as f:
        pickle.dump(dict(
            all_snapshots=all_snapshots, traj_params=traj_params,
            traj_kappas=traj_kappas, traj_starts=traj_starts,
            val_snapshots=val_snapshots,
            U_train=np.array(U_train), U_val=np.array(U_val),
        ), f)
    print(f"   Saved: {DATA_FILE}")

print(f"   Train: {U_train.shape}  |  Val: {U_val.shape}")

# Sanity check: t=0 snapshot should have max ≈ 1
t0_snap = all_snapshots[0][0]
print(f"\nAmplitude check (traj 0, t=0): "
      f"max={float(jnp.max(jnp.abs(t0_snap))):.4f}  (should be ≈1.0)")

# ─────────────────────────────────────────────────────────────────────
# 5. Model Definition
# ─────────────────────────────────────────────────────────────────────
AMP_EPS = 1e-6

def normalise(u_flat):
    scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
    return u_flat / scale, scale

def make_coordconv_input(u_flat):
    u_3d = u_flat.reshape(N, N, N)
    return jnp.concatenate([u_3d[..., None], COORD_GRID], axis=-1)


class ResBlock3D(nn.Module):
    out_feats:  int
    num_groups: int = 8
    @nn.compact
    def __call__(self, x, training: bool = False):
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Conv(self.out_feats, (3,3,3), strides=(1,1,1), padding='SAME')(h)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Conv(self.out_feats, (3,3,3), strides=(1,1,1), padding='SAME')(h)
        if x.shape[-1] != self.out_feats:
            x = nn.Conv(self.out_feats, (1,1,1))(x)
        return x + h


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
    latent_dim:   int
    features:     Sequence[int] = (32, 64, 128)
    pool_size:    int = 4
    dropout_rate: float = 0.1
    num_groups:   int = 8
    @nn.compact
    def __call__(self, x, training: bool = False):
        h = x   # (N,N,N,4)
        for feat in self.features:
            h = nn.Conv(feat, (3,3,3), strides=(2,2,2), padding='SAME')(h)
            h = nn.GroupNorm(num_groups=self.num_groups)(h)
            h = nn.leaky_relu(h, negative_slope=0.2)
            h = ResBlock3D(feat, num_groups=self.num_groups)(h, training)
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        H, W, D, C = h.shape
        if H != self.pool_size:
            h = jax.image.resize(h,
                                 (self.pool_size,self.pool_size,self.pool_size,C),
                                 method='linear')
        return AttentionPooling(self.latent_dim)(h)


class SeparableDecoder(nn.Module):
    latent_dim:  int
    rank:        int = 512
    grid_size:   int = 32
    hidden_dims: Sequence[int] = (256, 512, 512)
    def setup(self):
        self.hidden_layers = [nn.Dense(d) for d in self.hidden_dims]
        self.to_rank       = nn.Dense(self.rank)
        self.z_proj        = nn.Dense(self.hidden_dims[-1])
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
        return self.to_rank(h + self.z_proj(z))
    def __call__(self, z):
        h    = self._mlp_body(z)
        u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
        return u_3d.flatten() + self.bias


class ScalableAutoencoder(nn.Module):
    latent_dim:    int
    rank:          int = 512
    grid_size:     int = 32
    conv_features: Sequence[int] = (32, 64, 128)
    hidden_dims:   Sequence[int] = (256, 512, 512)
    def setup(self):
        self.encoder = Conv3DEncoder(latent_dim=self.latent_dim,
                                     features=self.conv_features)
        self.decoder = SeparableDecoder(latent_dim=self.latent_dim,
                                        rank=self.rank,
                                        grid_size=self.grid_size,
                                        hidden_dims=self.hidden_dims)
    def encode(self, u_flat, training=False):
        u_norm, scale = normalise(u_flat)
        z = self.encoder(make_coordconv_input(u_norm), training=training)
        return z, scale
    def decode(self, z, scale):
        return self.decoder(z) * scale
    def decode_normalised(self, z):
        return self.decoder(z)
    def __call__(self, u_flat, training=False):
        z, scale = self.encode(u_flat, training=training)
        return self.decode(z, scale)


# ─────────────────────────────────────────
# 6. Model Init
# ─────────────────────────────────────────
k_dim = 32
RANK  = 512

model = ScalableAutoencoder(
    latent_dim    = k_dim,
    rank          = RANK,
    grid_size     = N,
    conv_features = (32, 64, 128),
    hidden_dims   = (256, 512, 512),
)

key       = jax.random.PRNGKey(0)
variables = model.init(
    {'params': key, 'dropout': jax.random.PRNGKey(1)},
    U_train[0], training=True
)
params   = variables['params']
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\n-- Model: {n_params:,} parameters  (latent_dim={k_dim}, rank={RANK}) --")

# ─────────────────────────────────────────
# 7. Loss
# ─────────────────────────────────────────
REL_EPS = 1e-6

def per_sample_loss(u_true, u_pred):
    diff    = u_true - u_pred
    norm_sq = jnp.dot(u_true, u_true) + REL_EPS
    return jnp.dot(diff, diff) / norm_sq

# ─────────────────────────────────────────
# 8. Training
# ─────────────────────────────────────────
BATCH_SIZE = min(64, N_TRAIN)  # Batch size for unique t=0 samples
NUM_EPOCHS = 10_000            # Fewer epochs needed with unique patterns
LOG_EVERY  = 500

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., peak_value=1e-3,
    warmup_steps=500, decay_steps=NUM_EPOCHS, end_value=1e-5
)
tx        = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
opt_state = tx.init(params)
key       = jax.random.PRNGKey(2)

NOISE_STD = 0.02  # Gaussian noise on normalized input for regularization

def forward_noisy(p, u, noise_key, drop_key):
    """Forward pass with noise injection on normalized input."""
    u_norm, scale = normalise(u)
    noise = NOISE_STD * jax.random.normal(noise_key, u_norm.shape)
    u_noisy = u_norm + noise
    # Encode noisy input, decode, rescale
    inp = make_coordconv_input(u_noisy)
    # Use bound methods via lambda to access submodules
    z = model.apply({'params': p}, inp, training=True,
                    rngs={'dropout': drop_key},
                    method=lambda m, x, **kw: m.encoder(x, **kw))
    u_pred = model.apply({'params': p}, z,
                         method=lambda m, x: m.decoder(x)) * scale
    return u_pred

@jax.jit
def train_step(params, opt_state, batch, key):
    drop_key, noise_key = jax.random.split(key)
    def loss_fn(p):
        noise_keys = jax.random.split(noise_key, batch.shape[0])
        preds = jax.vmap(lambda u, nk: forward_noisy(p, u, nk, drop_key))(batch, noise_keys)
        return jnp.mean(jax.vmap(per_sample_loss)(batch, preds))
    loss, grads         = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_st = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_st, loss

@jax.jit
def eval_step(params, batch):
    preds  = jax.vmap(
        lambda u: model.apply({'params': params}, u, training=False)
    )(batch)
    return jnp.mean(jax.vmap(per_sample_loss)(batch, preds))

print(f"\n-- Training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}) --")
n_train        = len(U_train)
train_losses   = []
val_losses     = []
best_val       = float('inf')
best_params    = params
patience       = 8
patience_count = 0
t0             = time.perf_counter()

for epoch in range(NUM_EPOCHS + 1):
    key, subkey = jax.random.split(key)
    idx   = jax.random.choice(subkey, n_train, shape=(BATCH_SIZE,), replace=False)
    params, opt_state, loss = train_step(params, opt_state, U_train[idx], subkey)

    if epoch % LOG_EVERY == 0:
        key, vkey = jax.random.split(key)
        v_idx  = jax.random.choice(vkey, len(U_val), shape=(BATCH_SIZE,), replace=False)
        v_loss = float(eval_step(params, U_val[v_idx]))
        train_losses.append((epoch, float(loss)))
        val_losses.append((epoch, v_loss))
        print(f"  Epoch {epoch:5d} | train {float(loss):.4e} | "
              f"val {v_loss:.4e} | {time.perf_counter()-t0:.0f}s")
        if v_loss < best_val:
            best_val, best_params, patience_count = v_loss, params, 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stop  (best val={best_val:.4e})")
                break

params = best_params
print(f"\n  Best val loss: {best_val:.4e}")

# ─────────────────────────────────────────
# 9. Checkpoint
# ─────────────────────────────────────────
ckpt = {
    'params': params,
    'model_cfg': dict(latent_dim=k_dim, rank=RANK, grid_size=N,
                      conv_features=(32,64,128), hidden_dims=(256,512,512)),
    'train_meta': dict(
        n_train=N_TRAIN, num_steps=NUM_STEPS, dt=dt,
        traj_kappas=traj_kappas, traj_starts=traj_starts,
        traj_params=traj_params,
        kappas_train=KAPPAS_TRAIN, k_range=list(K_RANGE),
    )
}
with open(OUT / 'checkpoint.pkl', 'wb') as f:
    pickle.dump(ckpt, f)
print(f"  Checkpoint: {OUT / 'checkpoint.pkl'}")

# ─────────────────────────────────────────
# 10. Validation Plots
# ─────────────────────────────────────────
def encode(u):
    z, s = model.apply({'params': params}, u, training=False, method=model.encode)
    return z, s
def reconstruct(u):
    return model.apply({'params': params}, u, training=False)

# Loss curve
ep_t, lo_t = zip(*train_losses)
ep_v, lo_v = zip(*val_losses)
fig, ax = plt.subplots(figsize=(9,4))
ax.semilogy(ep_t, lo_t, label='Train', color='#1f77b4', lw=2)
ax.semilogy(ep_v, lo_v, label='Val',   color='#ff7f0e', lw=2, ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L2 Loss (log)')
ax.set_title('Heat (Sinusoidal) AE — Training Curve')
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / 'loss_curve.png', dpi=150); plt.close()

# Reconstruction samples + analytical comparison
# Pick 3 trajectories: low/medium/high k, all guaranteed to pass filter
# Use κ=0.01 (slowest) to ensure high-k modes survive
sample_cases = [
    (k1,k2,k3,kap) for k1,k2,k3,kap in traj_params
    if (k1,k2,k3) in [(1,1,1),(2,2,2),(4,4,4)]
][:3]  # take first 3 matching
time_idxs = [0, NUM_STEPS//2, NUM_STEPS]
mid       = N // 2
fig, axes = plt.subplots(3, 9, figsize=(27, 9))

print("\nReconstruction errors (FOM and ROM vs analytical):")
print(f"  {'Case':<18} {'t':>6} {'FOM vs exact':>14} {'ROM vs exact':>14}")
print("  " + "─"*56)

for row, (k1,k2,k3,kappa) in enumerate(sample_cases):
    # Find this trajectory in training set
    tidx = next(i for i,(p1,p2,p3,pk) in enumerate(traj_params)
                if p1==k1 and p2==k2 and p3==k3 and pk==kappa)
    traj = all_snapshots[tidx]

    for col, t_idx in enumerate(time_idxs):
        t_val     = t_idx * dt
        u_fom     = traj[t_idx]
        u_rec     = reconstruct(u_fom)
        u_exact   = get_analytical(k1,k2,k3,kappa,t_val)

        norm_ex   = float(jnp.linalg.norm(u_exact))
        err_fom   = float(jnp.linalg.norm(u_fom   - u_exact) / (norm_ex + 1e-12))
        err_rom   = float(jnp.linalg.norm(u_rec   - u_exact) / (norm_ex + 1e-12))

        if col == 0:
            print(f"  k=({k1},{k2},{k3}) κ={kappa:.2f}  "
                  f"t={t_val:.3f}  {err_fom:.4e}  {err_rom:.4e}")

        u_f3d = np.array(u_fom   ).reshape(N,N,N)
        u_r3d = np.array(u_rec   ).reshape(N,N,N)
        u_e3d = np.array(u_exact ).reshape(N,N,N)
        vmax  = max(float(np.abs(u_e3d[:,:,mid]).max()), 1e-8)
        kw    = dict(origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

        axes[row, col*3  ].imshow(u_e3d[:,:,mid].T, **kw)
        axes[row, col*3  ].set_title(f'Exact t={t_val:.2f}', fontsize=7)
        axes[row, col*3  ].axis('off')
        axes[row, col*3+1].imshow(u_f3d[:,:,mid].T, **kw)
        axes[row, col*3+1].set_title(f'FOM ε={err_fom:.1e}', fontsize=7)
        axes[row, col*3+1].axis('off')
        axes[row, col*3+2].imshow(u_r3d[:,:,mid].T, **kw)
        axes[row, col*3+2].set_title(f'ROM ε={err_rom:.1e}', fontsize=7)
        axes[row, col*3+2].axis('off')
    axes[row, 0].set_ylabel(f'k=({k1},{k2},{k3})\nκ={kappa}', fontsize=8)

fig.suptitle('Heat AE — Exact / FOM / ROM  (z=mid slice)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Plots saved to {OUT}/")
print("\n=== Training complete — run nmrom_heat_sin.py next ===")