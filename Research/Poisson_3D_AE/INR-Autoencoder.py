"""
train_autoencoder.py
--------------------
Complete training + evaluation for the ScalableAutoencoder.
Covers:
  1. Data generation (3D Poisson snapshots)
  2. Model init — handles BatchNorm mutable state properly
  3. Training loop
  4. Validation: reconstruction error, latent interpolation, midplane slices
  5. Comparison table vs original dense AE param count
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from typing import Sequence
import jax.scipy.sparse.linalg as jax_linalg

# ─────────────────────────────────────────
# 0. Paths & Logging
# ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
OUT = SCRIPT_DIR / 'plots'
OUT.mkdir(parents=True, exist_ok=True)

# Setup logging to file
LOG_FILE = SCRIPT_DIR / 'training.log'
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = TeeLogger(LOG_FILE)

# ─────────────────────────────────────────
# 1. Grid & Physics (identical to original)
# ─────────────────────────────────────────
N        = 32          # swap to 64 / 128 to stress-test scaling
num_nodes = N ** 3
L        = 1.0
dx       = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

def K_op_3d(u_flat):
    u = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1,1:-1,1:-1].set(
        (6*u[1:-1,1:-1,1:-1]
         - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
         - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
         - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
    )
    for s in [out.at[0], out.at[-1]]:   s.set(u[0] if 'at[0]' in str(s) else u[-1])
    out = out.at[0,:,:].set(u[0,:,:])
    out = out.at[-1,:,:].set(u[-1,:,:])
    out = out.at[:,0,:].set(u[:,0,:])
    out = out.at[:,-1,:].set(u[:,-1,:])
    out = out.at[:,:,0].set(u[:,:,0])
    out = out.at[:,:,-1].set(u[:,:,-1])
    return out.flatten()

def get_F_3d(k1, k2, k3):
    F = jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z) * 10.0
    for s in ['0', '-1']:
        F = F.at[0,:,:].set(0.).at[-1,:,:].set(0.)
        F = F.at[:,0,:].set(0.).at[:,-1,:].set(0.)
        F = F.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    return F.flatten()

def get_exact(k1, k2, k3):
    c = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (c * jnp.sin(k1*jnp.pi*X)
              * jnp.sin(k2*jnp.pi*Y)
              * jnp.sin(k3*jnp.pi*Z)).flatten()

def fom_solve(F_vec):
    u, _ = jax_linalg.cg(K_op_3d, F_vec,
                          x0=jnp.zeros(num_nodes), tol=1e-6, maxiter=2000)
    return u

# Boundary mask
mask_3d = jnp.ones((N,N,N))
for s in [mask_3d.at[0], mask_3d.at[-1]]:
    mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask = mask_3d.flatten()

print(f"Grid: {N}³ = {num_nodes:,} nodes")

# ─────────────────────────────────────────
# 2. Model Definition
# ─────────────────────────────────────────
class AttentionPooling(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, feat_map):
        C      = feat_map.shape[-1]
        tokens = feat_map.reshape(-1, C)               # (T, C)
        tokens = nn.Dense(self.latent_dim)(tokens)     # (T, D)
        query  = self.param('query', nn.initializers.normal(0.02), (self.latent_dim,))
        scale  = jnp.sqrt(jnp.float32(self.latent_dim))
        scores = jnp.einsum('td,d->t', tokens, query) / scale
        w      = jax.nn.softmax(scores, axis=0)
        return jnp.einsum('t,td->d', w, tokens)        # (D,)


class Conv3DEncoder(nn.Module):
    latent_dim: int
    features:   Sequence[int] = (32, 64, 128)
    pool_size:  int = 4

    @nn.compact
    def __call__(self, x, training: bool = False):
        # x: (N, N, N)
        h = x[..., None]                               # (N,N,N,1)
        for feat in self.features:
            h = nn.Conv(feat, kernel_size=(3,3,3),
                        strides=(2,2,2), padding='SAME')(h)
            # ── BatchNorm: use_running_average flips between train / eval ──
            h = nn.BatchNorm(use_running_average=not training,
                             momentum=0.9)(h)
            h = nn.leaky_relu(h, negative_slope=0.2)

        # Adaptive spatial pool → fixed pool_size³
        H, W, D, C = h.shape
        if H != self.pool_size:
            h = jax.image.resize(h,
                                 (self.pool_size, self.pool_size,
                                  self.pool_size, C),
                                 method='linear')
        return AttentionPooling(self.latent_dim)(h)    # (D,)


class SeparableDecoder(nn.Module):
    latent_dim:  int
    rank:        int = 256
    grid_size:   int = 32
    hidden_dims: Sequence[int] = (256, 512)

    def setup(self):
        self.hidden_layers = [nn.Dense(d) for d in self.hidden_dims]
        self.to_rank       = nn.Dense(self.rank)
        init = nn.initializers.normal(0.01)
        N    = self.grid_size
        self.W_x         = self.param('W_x',  init, (self.rank, N))
        self.W_y         = self.param('W_y',  init, (self.rank, N))
        self.W_z         = self.param('W_z',  init, (self.rank, N))
        self.bias_scalar = self.param('bias', nn.initializers.zeros, ())

    def _mlp_body(self, z):
        h = z
        for layer in self.hidden_layers:
            h = nn.swish(layer(h))
        return self.to_rank(h)                         # (rank,)

    def __call__(self, z):
        h    = self._mlp_body(z)
        u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
        return u_3d.flatten() + self.bias_scalar

    def decode_at_flat_indices(self, z, flat_indices):
        h  = self._mlp_body(z)
        Ng = self.grid_size
        ix = flat_indices // (Ng * Ng)
        iy = (flat_indices // Ng) % Ng
        iz = flat_indices % Ng
        factors = self.W_x[:, ix] * self.W_y[:, iy] * self.W_z[:, iz]  # (rank, P)
        return h @ factors + self.bias_scalar


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


# ─────────────────────────────────────────────────────────
# 3. Parameter Count Comparison
# ─────────────────────────────────────────────────────────
def param_count_dense(num_nodes, latent_dim=20, hidden=1024):
    """Original dense autoencoder approximate param count."""
    enc = num_nodes*hidden + hidden*512 + 512*256 + 256*latent_dim
    dec = latent_dim*256 + 256*512 + 512*hidden + hidden*num_nodes
    return enc + dec

def param_count_scalable(N, latent_dim=20, rank=256,
                         conv_features=(32,64,128), hidden_dims=(256,512)):
    conv_params = 1*27*conv_features[0]
    for i in range(1, len(conv_features)):
        conv_params += conv_features[i-1] * 27 * conv_features[i]
    attn_params = conv_features[-1]*latent_dim + latent_dim   # dense + query
    enc = conv_params + attn_params

    mlp = latent_dim * hidden_dims[0]
    for i in range(1, len(hidden_dims)):
        mlp += hidden_dims[i-1] * hidden_dims[i]
    mlp += hidden_dims[-1] * rank
    sep = 3 * rank * N + 1
    dec = mlp + sep
    return enc + dec

print("\n── Parameter Scaling Comparison ──────────────────────")
print(f"{'Grid':<8} {'Dense AE':>14} {'Scalable AE':>14} {'Ratio':>8}")
print("─" * 48)
for n in [32, 64, 128]:
    d = param_count_dense(n**3)
    s = param_count_scalable(n)
    print(f"{n}³{'':<5} {d:>14,} {s:>14,} {d/s:>7.0f}×")
print("─" * 48)

# ─────────────────────────────────────────
# 4. Generate Training Data (Analytical + k²-Normalization)
# ─────────────────────────────────────────
print("\n── Generating Snapshots (Analytical, k²-normalized) ───")
import random
random.seed(42)

all_ks = [(k1, k2, k3)
          for k1 in range(1, 6)
          for k2 in range(1, 6)
          for k3 in range(1, 6)]  # 125 total

random.shuffle(all_ks)
n_test   = 25
test_ks  = all_ks[:n_test]   # 25 unseen — NEVER used in AE training or EQ
train_ks = all_ks[n_test:]   # 100 training combinations

def get_k2_scale(k1, k2, k3):
    """Return normalization factor: k1² + k2² + k3².
    Analytical solution scales as 1/k², so multiplying by k² normalizes."""
    return float(k1**2 + k2**2 + k3**2)

U_list = []
scale_factors = []  # store k² for each snapshot
for i, (k1,k2,k3) in enumerate(train_ks):
    u = get_exact(k1, k2, k3)              # use analytical solution
    k2_scale = get_k2_scale(k1, k2, k3)
    u_normalized = u * k2_scale            # normalize by k²
    U_list.append(u_normalized)
    scale_factors.append(k2_scale)
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(train_ks)} snapshots done")

U_train = jnp.stack(U_list)               # (100, num_nodes) — normalized, k∈{1..5}³ minus 25 held-out
scale_factors_train = jnp.array(scale_factors)
print(f"  Dataset shape: {U_train.shape}")
print(f"  Scale factors range: [{scale_factors_train.min():.0f}, {scale_factors_train.max():.0f}]")

# Validation: held-out from train (not test_ks)
val_ks   = [(1,2,4), (4,2,1), (5,4,2), (2,5,4)]
U_val_raw = jnp.stack([get_exact(*k) for k in val_ks])
scale_factors_val = jnp.array([get_k2_scale(*k) for k in val_ks])
U_val = U_val_raw * scale_factors_val[:, None]  # normalize validation set
print(f"  Training snapshots:   {len(train_ks)}")
print(f"  Validation snapshots: {len(val_ks)}")
print(f"  Test cases (held-out):{len(test_ks)}")

# ─────────────────────────────────────────────────────────────────────
# 5. Model Init
#
# BatchNorm introduces mutable state (running mean/var) stored in
# `batch_stats`. Flax requires you to separate trainable params from
# this mutable state and update it explicitly each step.
# ─────────────────────────────────────────────────────────────────────
k_dim = 32

model = ScalableAutoencoder(
    latent_dim    = k_dim,
    rank          = 512,
    grid_size     = N,
    conv_features = (32, 64, 128),
    hidden_dims   = (256, 512),
)
# Note: AE trains on ALL 125 snapshots — manifold covers full parameter space
# NMROM benchmark tests whether LM-GN solver can find correct z via PDE residual minimization

key = jax.random.PRNGKey(42)

# Init with training=True so BatchNorm creates batch_stats
variables = model.init({'params': key}, U_train[0], training=True)

# Split into params (updated by optimizer) and batch_stats (updated separately)
params     = variables['params']
batch_stats = variables['batch_stats']

n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\n  Scalable AE parameters: {n_params:,}")
print(f"  Dense AE equivalent:    {param_count_dense(num_nodes):,}")
print(f"  Reduction:              {param_count_dense(num_nodes)/n_params:.0f}×")

# ─────────────────────────────────────────
# 6. Optimizer
# ─────────────────────────────────────────
schedule = optax.warmup_cosine_decay_schedule(
    init_value   = 0.0,
    peak_value   = 1e-3,
    warmup_steps = 500,
    decay_steps  = 10_000,
    end_value    = 1e-5,
)
tx        = optax.adam(schedule)
opt_state = tx.init(params)

# ─────────────────────────────────────────────────────────────────────
# 7. Training Step
#
# Two things happen each step:
#   a) params  ← updated by Adam via value_and_grad
#   b) batch_stats ← updated by BatchNorm (returned as mutable output)
# ─────────────────────────────────────────────────────────────────────
@jax.jit
def train_step(params, batch_stats, opt_state, batch):
    def loss_fn(p):
        # vmap over batch, no mutable — use current batch_stats read-only
        preds = jax.vmap(
            lambda u: model.apply(
                {'params': p, 'batch_stats': batch_stats},
                u, training=False,   # read-only stats during grad
            )
        )(batch)
        return jnp.mean((batch - preds) ** 2)

    loss, grads          = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params           = optax.apply_updates(params, updates)

    # Update batch_stats separately — one forward pass, no grad needed
    _, updates_bs = model.apply(
        {'params': new_params, 'batch_stats': batch_stats},
        batch[0],              # single sample is enough to update running stats
        training=True,
        mutable=['batch_stats']
    )
    new_batch_stats = updates_bs['batch_stats']

    return new_params, new_batch_stats, new_opt_state, loss


@jax.jit
def eval_loss(params, batch_stats, batch):
    preds = jax.vmap(
        lambda u: model.apply(
            {'params': params, 'batch_stats': batch_stats},
            u, training=False,
        )
    )(batch)
    return jnp.mean((batch - preds) ** 2)


def relative_l2(u_pred, u_true):
    return float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))

# ─────────────────────────────────────────
# 8. Training Loop
# ─────────────────────────────────────────
print("\n── Training ──────────────────────────────────────────")
NUM_EPOCHS  = 30_000
LOG_EVERY   = 1_000

train_losses = []
val_losses   = []
t0_train     = time.perf_counter()

for epoch in range(NUM_EPOCHS + 1):
    params, batch_stats, opt_state, loss = train_step(
        params, batch_stats, opt_state, U_train
    )

    if epoch % LOG_EVERY == 0:
        v_loss = float(eval_loss(params, batch_stats, U_val))
        t_loss = float(loss)
        train_losses.append((epoch, t_loss))
        val_losses.append((epoch, v_loss))
        elapsed = time.perf_counter() - t0_train
        print(f"  Epoch {epoch:5d} | train {t_loss:.4e} | val {v_loss:.4e} | {elapsed:.1f}s")

print(f"\n  Total training time: {time.perf_counter()-t0_train:.1f}s")

# ─────────────────────────────────────────
# 9. Reconstruction Quality
# ─────────────────────────────────────────
print("\n── Reconstruction Errors ─────────────────────────────")

def encode(u_flat):
    return model.apply(
        {'params': params, 'batch_stats': batch_stats},
        u_flat, training=False, method=model.encode
    )

def decode(z):
    return model.apply(
        {'params': params, 'batch_stats': batch_stats},
        z, method=model.decode
    )

# Training set reconstruction
train_errs = []
for i, (k1,k2,k3) in enumerate(train_ks):
    z     = encode(U_train[i])
    u_rec = decode(z)
    train_errs.append(relative_l2(u_rec, U_train[i]))

# Validation reconstruction (in normalized space)
val_errs = []
print(f"\n  {'Case':<12} {'Rel L2 (normalized)':>22} {'Rel L2 (denorm vs exact)':>26}")
print("  " + "─"*62)
for i, (k1,k2,k3) in enumerate(val_ks):
    z      = encode(U_val[i])              # encode normalized input
    u_rec  = decode(z)                     # reconstructed (normalized)
    u_ex   = get_exact(k1,k2,k3)           # original scale
    k2_scale = get_k2_scale(k1,k2,k3)
    e_norm = relative_l2(u_rec, U_val[i])  # error in normalized space
    u_rec_denorm = u_rec / k2_scale        # denormalize for comparison
    e_ex   = relative_l2(u_rec_denorm, u_ex)
    val_errs.append(e_norm)
    print(f"  ({k1},{k2},{k3}){'':<7} {e_norm:>22.4e} {e_ex:>26.4e}")

print(f"\n  Mean train reconstruction: {np.mean(train_errs):.4e}")
print(f"  Mean val   reconstruction: {np.mean(val_errs):.4e}")

# ─────────────────────────────────────────────────────────────────────────────
# 9b. OOD Test Reconstruction (k=3 cases — never seen during training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Unseen Test Reconstruction (in-distribution, unseen combinations) ──")
print(f"  {'Case':<12} {'Rel L2 (normalized)':>22} {'Rel L2 (denorm vs exact)':>26}")
print("  " + "─"*62)
test_errs = []
for k1, k2, k3 in test_ks:
    u_ex     = get_exact(k1, k2, k3)
    k2_scale = get_k2_scale(k1, k2, k3)
    u_norm   = u_ex * k2_scale
    z        = encode(u_norm)
    u_rec_norm = decode(z)
    e_norm   = relative_l2(u_rec_norm, u_norm)
    u_rec    = u_rec_norm / k2_scale
    e_ex     = relative_l2(u_rec, u_ex)
    test_errs.append(e_ex)
    print(f"  ({k1},{k2},{k3}){'':<7} {e_norm:>22.4e} {e_ex:>26.4e}")
print(f"\n  Mean OOD test reconstruction: {np.mean(test_errs):.4e}")

# ─────────────────────────────────────────
# 10. Latent Space Sanity Check
#     Interpolate between two latent codes
#     → smooth field should appear
# ─────────────────────────────────────────
print("\n── Latent Interpolation Check ────────────────────────")
z_a = encode(U_train[0])   # k=(1,1,1)
z_b = encode(U_train[63])  # k=(4,4,4)

alphas  = [0.0, 0.25, 0.5, 0.75, 1.0]
mid     = N // 2
fig, axes = plt.subplots(1, len(alphas), figsize=(16, 3))

for ax, alpha in zip(axes, alphas):
    z_interp = (1 - alpha) * z_a + alpha * z_b
    u_interp = np.array(decode(z_interp)).reshape(N, N, N)
    im = ax.imshow(u_interp[:,:,mid].T, origin='lower',
                   cmap='viridis', aspect='auto')
    ax.set_title(f'α={alpha:.2f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Latent Interpolation: k=(1,1,1) → k=(4,4,4)  |  z-midplane', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'latent_interpolation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT / 'latent_interpolation.png'}")

# ─────────────────────────────────────────
# 11. Loss Curves
# ─────────────────────────────────────────
ep_t, lo_t = zip(*train_losses)
ep_v, lo_v = zip(*val_losses)

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(ep_t, lo_t, label='Train MSE', color='#1f77b4', lw=2)
ax.semilogy(ep_v, lo_v, label='Val MSE',   color='#ff7f0e', lw=2, ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('ScalableAutoencoder Training Curve')
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / 'loss_curve.png', dpi=150)
plt.close()
print(f"  Saved: {OUT / 'loss_curve.png'}")

# ─────────────────────────────────────────
# 12. Midplane Slice: Analytical vs Reconstructed (denormalized)
# ─────────────────────────────────────────
test_show = [(1,2,4), (3,3,3), (4,2,3)]  # in-dist, OOD-center, OOD-mixed

for k1,k2,k3 in test_show:
    u_ex  = get_exact(k1,k2,k3)
    k2_scale = get_k2_scale(k1,k2,k3)
    u_norm = u_ex * k2_scale               # normalize for encoding
    z     = encode(u_norm)
    u_rec_norm = decode(z)                 # reconstructed (normalized)
    u_rec = u_rec_norm / k2_scale          # denormalize

    u_ex_3d  = np.array(u_ex ).reshape(N,N,N)
    u_rec_3d = np.array(u_rec).reshape(N,N,N)

    mid = N // 2
    sl  = slice(None), slice(None), mid

    vmin = min(u_ex_3d[sl].min(), u_rec_3d[sl].min())
    vmax = max(u_ex_3d[sl].max(), u_rec_3d[sl].max())
    kw   = dict(origin='lower', aspect='auto', cmap='viridis',
                vmin=vmin, vmax=vmax, extent=[0,L,0,L])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes[:2],
                                [u_ex_3d[sl], u_rec_3d[sl]],
                                ['Analytical', 'Reconstructed']):
        im = ax.imshow(data.T, **kw)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)

    err = np.abs(u_rec_3d[sl] - u_ex_3d[sl])
    im  = axes[2].imshow(err.T, origin='lower', aspect='auto',
                          cmap='hot', extent=[0,L,0,L])
    axes[2].set_title('|Rec - Analytical|', fontsize=11)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    fig.suptitle(f'Midplane z={mid*dx:.2f} | k=({k1},{k2},{k3}) | '
                 f'Rel-L2={relative_l2(u_rec, u_ex):.3e}', fontsize=12)
    plt.tight_layout()
    fpath = OUT / f'slice_k{k1}{k2}{k3}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────
# 13. Error Bar Chart — Training vs Val Cases
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
x    = np.arange(len(train_ks))
bars = ax.bar(x, train_errs, color='#1f77b4', alpha=0.8, width=0.7)
ax.set_yscale('log')
ax.axhline(np.mean(train_errs), color='orange', ls='--', lw=2,
           label=f'Mean = {np.mean(train_errs):.2e}')
ax.set_xlabel('Training snapshot index')
ax.set_ylabel('Relative L2 reconstruction error')
ax.set_title('ScalableAutoencoder — Per-snapshot Reconstruction Error (training set)')
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig(OUT / 'reconstruction_error.png', dpi=150)
plt.close()
print(f"  Saved: {OUT / 'reconstruction_error.png'}")

# ─────────────────────────────────────────
# 14. Final Summary
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ScalableAutoencoder — Training Summary")
print(f"{'='*55}")
print(f"  Grid:                   {N}³ = {num_nodes:,} DOF")
print(f"  Latent dim:             {k_dim}")
print(f"  CP rank:                512")
print(f"  Model params:           {n_params:,}")
print(f"  Dense AE params:        {param_count_dense(num_nodes):,}")
print(f"  Param reduction:        {param_count_dense(num_nodes)/n_params:.0f}×")
print(f"  Mean train rec error:   {np.mean(train_errs):.4e}")
print(f"  Mean val   rec error:   {np.mean(val_errs):.4e}")
print(f"  Mean test rec error:    {np.mean(test_errs):.4e}  (test_ks -- unseen OOD)")
print(f"  Plots saved to:         {OUT}/")
print(f"{'='*55}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Train k→z predictor: small MLP that maps (k1,k2,k3) → latent z
#     Gives good initialization for GN solver without needing FOM solution
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Training k→z Predictor ────────────────────────────")

# Encode all training snapshots to get target latent codes
print("  Encoding all training snapshots...")
Z_train = []
for i, (k1, k2, k3) in enumerate(train_ks):
    z_i = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        U_train[i], training=False, method=model.encode
    )
    Z_train.append(z_i)
Z_train = jnp.stack(Z_train)  # (125, k_dim)

# Input features: normalized k values
K_train = jnp.array([[k1/5.0, k2/5.0, k3/5.0] for k1, k2, k3 in train_ks])  # (125, 3)

# Simple MLP: 3 → 64 → 128 → k_dim
import optax as _optax

def k_predictor_init(key, input_dim=3, hidden=64, output_dim=k_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1 = jax.random.normal(k1, (input_dim, hidden)) * 0.1
    b1 = jnp.zeros(hidden)
    W2 = jax.random.normal(k2, (hidden, hidden)) * 0.1
    b2 = jnp.zeros(hidden)
    W3 = jax.random.normal(k3, (hidden, output_dim)) * 0.01
    b3 = jnp.zeros(output_dim)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

def k_predictor_forward(p, k_input):
    h = jax.nn.swish(k_input @ p['W1'] + p['b1'])
    h = jax.nn.swish(h @ p['W2'] + p['b2'])
    return h @ p['W3'] + p['b3']

@jax.jit
def kz_loss(p, K, Z):
    Z_pred = jax.vmap(lambda k: k_predictor_forward(p, k))(K)
    return jnp.mean((Z_pred - Z) ** 2)

kz_key = jax.random.PRNGKey(99)
kz_params = k_predictor_init(kz_key)
kz_tx = optax.adam(1e-3)
kz_opt_state = kz_tx.init(kz_params)

# Note: kz_tx captured in closure, not passed as arg (avoids JAX JIT issue with functions)
@jax.jit
def kz_step(p, opt_state, K, Z):
    loss, grads = jax.value_and_grad(kz_loss)(p, K, Z)
    updates, new_opt_state = kz_tx.update(grads, opt_state, p)
    new_p = optax.apply_updates(p, updates)
    return new_p, new_opt_state, loss

KZ_EPOCHS = 5000
for ep in range(KZ_EPOCHS + 1):
    kz_params, kz_opt_state, kz_loss_val = kz_step(kz_params, kz_opt_state, K_train, Z_train)
    if ep % 1000 == 0:
        print(f"  k→z epoch {ep:5d} | loss {float(kz_loss_val):.4e}")

# Evaluate predictor
Z_pred_all = jax.vmap(lambda k: k_predictor_forward(kz_params, k))(K_train)
kz_errs = [float(jnp.linalg.norm(Z_pred_all[i] - Z_train[i]) / (jnp.linalg.norm(Z_train[i]) + 1e-8))
           for i in range(len(train_ks))]
print(f"  Mean k→z prediction error: {np.mean(kz_errs):.4e}")
print(f"  k→z predictor trained: (k1,k2,k3)/5 → latent z")

# ─────────────────────────────────────────
# 16. Export model state for downstream use (EQ phase)
# ─────────────────────────────────────────
import pickle
ckpt = {
    'params':      params,
    'batch_stats': batch_stats,
    'model_cfg': dict(
        latent_dim    = k_dim,
        rank          = 512,
        grid_size     = N,
        conv_features = (32, 64, 128),
        hidden_dims   = (256, 512),
    ),
    'normalization': {
        'type': 'k2_scale',
        'description': 'Multiply input by k1²+k2²+k3² before encoding, divide output after decoding',
    },
    'kz_predictor': kz_params,  # k→z predictor for GN initialization
}
CKPT_PATH = SCRIPT_DIR / 'checkpoint.pkl'
with open(CKPT_PATH, 'wb') as f:
    pickle.dump(ckpt, f)
print(f"\n  Checkpoint saved: {CKPT_PATH}")
print("  Keys: params, batch_stats, model_cfg, normalization, kz_predictor")
print("  Normalization: k²-scaling (multiply by k1²+k2²+k3² before encode, divide after decode)")
print("  Load with:  import pickle; ck = pickle.load(open('...', 'rb'))")