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
from pathlib import Path
from typing import Sequence
import jax.scipy.sparse.linalg as jax_linalg

# ─────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────
OUT = Path('plots/scalable_ae')
OUT.mkdir(parents=True, exist_ok=True)

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

def get_F_3d_superposition(k_weights):
    """Generate forcing from superposition of modes.
    k_weights: list of ((k1,k2,k3), weight) tuples
    """
    F = jnp.zeros_like(X)
    for (k1, k2, k3), w in k_weights:
        F = F + w * jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z) * 10.0
    F = F.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    F = F.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    F = F.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    return F.flatten()

def get_exact(k1, k2, k3):
    c = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (c * jnp.sin(k1*jnp.pi*X)
              * jnp.sin(k2*jnp.pi*Y)
              * jnp.sin(k3*jnp.pi*Z)).flatten()

def get_exact_superposition(k_weights):
    """Exact solution for superposition of modes."""
    u = jnp.zeros_like(X)
    for (k1, k2, k3), w in k_weights:
        c = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
        u = u + w * c * jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z)
    return u.flatten()

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
            # GroupNorm: no mutable state, no running stats, vmap-safe
            # num_groups=8 works well; feat must be divisible by num_groups
            h = nn.GroupNorm(num_groups=8)(h)
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
    latent_dim:   int
    rank:         int = 256
    grid_size:    int = 32
    hidden_dims:  Sequence[int] = (256, 512)
    dropout_rate: float = 0.15

    def setup(self):
        self.hidden_layers = [nn.Dense(d) for d in self.hidden_dims]
        self.dropouts      = [nn.Dropout(self.dropout_rate) for _ in self.hidden_dims]
        self.to_rank       = nn.Dense(self.rank)
        init = nn.initializers.normal(0.01)
        N    = self.grid_size
        self.W_x         = self.param('W_x',  init, (self.rank, N))
        self.W_y         = self.param('W_y',  init, (self.rank, N))
        self.W_z         = self.param('W_z',  init, (self.rank, N))
        self.bias_scalar = self.param('bias', nn.initializers.zeros, ())

    def _mlp_body(self, z, training=False):
        h = z
        for layer, drop in zip(self.hidden_layers, self.dropouts):
            h = nn.swish(layer(h))
            h = drop(h, deterministic=not training)
        return self.to_rank(h)                         # (rank,)

    def __call__(self, z, training=False):
        h    = self._mlp_body(z, training)
        u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
        return u_3d.flatten() + self.bias_scalar

    def decode_at_flat_indices(self, z, flat_indices, training=False):
        h  = self._mlp_body(z, training)
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

    def decode(self, z, training=False):
        return self.decoder(z, training=training)

    def __call__(self, u_flat, training=False):
        return self.decode(self.encode(u_flat, training=training), training=training)


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
# 4. Generate Training Data
# ─────────────────────────────────────────
print("\n── Generating Snapshots ──────────────────────────────")

# Set random seed for reproducibility
np.random.seed(42)

# K_MAX = 3 to limit frequency range
K_MAX = 3

# --- Part A: Single-mode snapshots with k in {1,2,3} ---
single_mode_ks = [(k1, k2, k3)
                  for k1 in range(1, K_MAX + 1)
                  for k2 in range(1, K_MAX + 1)
                  for k3 in range(1, K_MAX + 1)]  # 27 snapshots

# --- Part B: Superposition snapshots (2-4 modes combined) ---
# Generate random superpositions to prevent memorization
num_superposition = 100
superposition_configs = []

for _ in range(num_superposition):
    # Random number of modes (2 to 4)
    n_modes = np.random.randint(2, 5)
    modes = []
    for _ in range(n_modes):
        k1 = np.random.randint(1, K_MAX + 1)
        k2 = np.random.randint(1, K_MAX + 1)
        k3 = np.random.randint(1, K_MAX + 1)
        # Wider weight range: 0.1 to 1.0
        w = np.random.uniform(0.1, 1.0)
        modes.append(((k1, k2, k3), w))
    superposition_configs.append(modes)

print(f"  Single-mode snapshots: {len(single_mode_ks)}")
print(f"  Superposition snapshots: {len(superposition_configs)}")

# Generate all training snapshots
U_list = []
train_configs = []  # Store configs for reference

# Single modes
for i, (k1, k2, k3) in enumerate(single_mode_ks):
    u = fom_solve(get_F_3d(k1, k2, k3))
    U_list.append(u)
    train_configs.append(('single', (k1, k2, k3)))
    if (i + 1) % 10 == 0:
        print(f"  Single-mode: {i+1}/{len(single_mode_ks)} done")

print(f"  Single-mode snapshots complete.")

# Superpositions
for i, modes in enumerate(superposition_configs):
    u = fom_solve(get_F_3d_superposition(modes))
    U_list.append(u)
    train_configs.append(('super', modes))
    if (i + 1) % 10 == 0:
        print(f"  Superposition: {i+1}/{len(superposition_configs)} done")

# Normalize each snapshot to unit L2 norm
# This makes MSE equivalent to relative L2 — model can't ignore low-amplitude cases
U_array     = jnp.stack(U_list)
U_norms     = jnp.linalg.norm(U_array, axis=1, keepdims=True)  # (N_snap, 1)
U_train     = U_array / U_norms                                 # each row has ||u||=1
print(f"  Dataset shape: {U_train.shape}")
print(f"  Total training snapshots: {len(U_list)}")
print(f"  Snapshots normalized to unit L2 norm")

# Hold out validation cases - more samples for reliable early stopping
val_superpositions = [
    [((1, 2, 3), 0.70), ((2, 1, 1), 0.50)],
    [((3, 1, 2), 0.80), ((1, 3, 3), 0.40), ((2, 2, 1), 0.60)],
    [((2, 3, 1), 0.90), ((3, 2, 3), 0.30)],
    [((1, 1, 3), 0.50), ((3, 3, 1), 0.70), ((2, 1, 2), 0.40)],
    [((1, 3, 1), 0.60), ((2, 2, 3), 0.80)],
    [((3, 2, 1), 0.45), ((1, 1, 2), 0.55), ((3, 3, 2), 0.70)],
    [((2, 1, 3), 0.35), ((1, 2, 2), 0.65)],
    [((3, 1, 3), 0.50), ((2, 3, 2), 0.50), ((1, 2, 1), 0.80)],
    [((1, 3, 2), 0.90), ((3, 1, 1), 0.20)],
    [((2, 2, 2), 0.60), ((1, 3, 3), 0.60), ((3, 2, 1), 0.40)],
]
U_val_raw   = jnp.stack([fom_solve(get_F_3d_superposition(cfg)) for cfg in val_superpositions])
U_val_norms = jnp.linalg.norm(U_val_raw, axis=1, keepdims=True)
U_val       = U_val_raw / U_val_norms
print(f"  Validation snapshots: {len(val_superpositions)} (all superpositions, normalized)")

# ─────────────────────────────────────────────────────────────────────
# 5. Model Init
#
# GroupNorm has no mutable state — simpler init, no batch_stats needed.
# ─────────────────────────────────────────────────────────────────────
k_dim = 20

model = ScalableAutoencoder(
    latent_dim    = k_dim,
    rank          = 256,
    grid_size     = N,
    conv_features = (32, 64, 128),
    hidden_dims   = (256, 512),
)

key = jax.random.PRNGKey(42)

# Init — GroupNorm has no mutable state, only params
variables = model.init({'params': key, 'dropout': key}, U_train[0], training=True)
params = variables['params']

n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\n  Scalable AE parameters: {n_params:,}")
print(f"  Dense AE equivalent:    {param_count_dense(num_nodes):,}")
print(f"  Reduction:              {param_count_dense(num_nodes)/n_params:.0f}×")

# ─────────────────────────────────────────
# 6. Optimizer — AdamW with weight decay for regularization
# ─────────────────────────────────────────
schedule = optax.warmup_cosine_decay_schedule(
    init_value   = 0.0,
    peak_value   = 1e-3,
    warmup_steps = 500,
    decay_steps  = 10_000,
    end_value    = 1e-5,
)
# AdamW applies L2 penalty to weights — directly fights memorization
tx        = optax.adamw(learning_rate=schedule, weight_decay=5e-4)
opt_state = tx.init(params)

# ─────────────────────────────────────────────────────────────────────
# 7. Training Step
#
# With GroupNorm + Dropout: no batch_stats, but need dropout RNG keys.
# ─────────────────────────────────────────────────────────────────────
@jax.jit
def train_step(params, opt_state, batch, rng_key):
    def loss_fn(p):
        # Each sample gets its own dropout key
        keys = jax.random.split(rng_key, len(batch))
        preds = jax.vmap(
            lambda u, k: model.apply(
                {'params': p}, u,
                training=True,
                rngs={'dropout': k}
            )
        )(batch, keys)
        return jnp.mean((batch - preds) ** 2)

    loss, grads            = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params             = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss


@jax.jit
def eval_loss(params, batch):
    preds = jax.vmap(
        lambda u: model.apply(
            {'params': params}, u,
            training=False,
        )
    )(batch)
    return jnp.mean((batch - preds) ** 2)


def relative_l2(u_pred, u_true):
    return float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))

# ─────────────────────────────────────────
# 8. Training Loop with Early Stopping
# ─────────────────────────────────────────
print("\n── Training ──────────────────────────────────────────")
NUM_EPOCHS      = 10_000
LOG_EVERY       = 500
PATIENCE_LIMIT  = 3    # early stop after this many log intervals without improvement

train_losses = []
val_losses   = []
t0_train     = time.perf_counter()

# Early stopping state
best_val_loss  = jnp.inf
best_params    = params
patience       = 0
rng_key        = jax.random.PRNGKey(0)

for epoch in range(NUM_EPOCHS + 1):
    rng_key, subkey = jax.random.split(rng_key)
    params, opt_state, loss = train_step(params, opt_state, U_train, subkey)

    if epoch % LOG_EVERY == 0:
        v_loss = float(eval_loss(params, U_val))
        t_loss = float(loss)
        train_losses.append((epoch, t_loss))
        val_losses.append((epoch, v_loss))
        elapsed = time.perf_counter() - t0_train
        
        # Early stopping check
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_params   = params
            patience      = 0
            marker        = " *"
        else:
            patience += 1
            marker    = ""
        
        print(f"  Epoch {epoch:5d} | train {t_loss:.4e} | val {v_loss:.4e} | {elapsed:.1f}s{marker}")
        
        if patience >= PATIENCE_LIMIT:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE_LIMIT} checks)")
            break

# Roll back to best checkpoint
params = best_params
print(f"\n  Total training time: {time.perf_counter()-t0_train:.1f}s")
print(f"  Best val loss: {best_val_loss:.4e}")

# ─────────────────────────────────────────
# 9. Reconstruction Quality
# ─────────────────────────────────────────
print("\n── Reconstruction Errors ─────────────────────────────")

def encode(u_flat):
    return model.apply(
        {'params': params},
        u_flat, training=False, method=model.encode
    )

def decode(z):
    return model.apply(
        {'params': params},
        z, training=False, method=model.decode
    )

def encode_and_decode_physical(u_physical):
    """Round-trip that preserves physical amplitude."""
    norm  = jnp.linalg.norm(u_physical)
    z     = encode(u_physical / norm)
    u_rec = decode(z)
    return u_rec * norm   # rescale back to original amplitude

# Training set reconstruction
train_errs = []
for i in range(len(U_train)):
    z     = encode(U_train[i])
    u_rec = decode(z)
    train_errs.append(relative_l2(u_rec, U_train[i]))

# Validation reconstruction
val_errs = []
print(f"\n  {'Case':<30} {'Rel L2 (rec vs FOM)':>22} {'Rel L2 (rec vs exact)':>24}")
print("  " + "─"*78)
for i, cfg in enumerate(val_superpositions):
    z      = encode(U_val[i])
    u_rec  = decode(z)
    u_ex   = get_exact_superposition(cfg)
    e_fom  = relative_l2(u_rec, U_val[i])
    e_ex   = relative_l2(u_rec, u_ex)
    val_errs.append(e_fom)
    cfg_str = '+'.join([f"({k[0]},{k[1]},{k[2]})" for k, w in cfg])
    print(f"  {cfg_str:<30} {e_fom:>22.4e} {e_ex:>24.4e}")

print(f"\n  Mean train reconstruction: {np.mean(train_errs):.4e}")
print(f"  Mean val   reconstruction: {np.mean(val_errs):.4e}")

# ─────────────────────────────────────────
# 10. Latent Space Sanity Check
#     Interpolate between two latent codes
#     → smooth field should appear
# ─────────────────────────────────────────
print("\n── Latent Interpolation Check ────────────────────────")
z_a = encode(U_train[0])   # k=(1,1,1) single mode
z_b = encode(U_train[26])  # k=(3,3,3) single mode (last single-mode snapshot)

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

fig.suptitle('Latent Interpolation: k=(1,1,1) → k=(3,3,3)  |  z-midplane', fontsize=12)
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
# 12. Midplane Slice: FOM vs Reconstructed vs Exact
# ─────────────────────────────────────────
# Single-mode test cases
test_show = [(1,2,3), (3,3,3), (2,1,2)]

for k1,k2,k3 in test_show:
    u_fom = fom_solve(get_F_3d(k1,k2,k3))
    z     = encode(u_fom)
    u_rec = decode(z)
    u_ex  = get_exact(k1,k2,k3)

    u_fom_3d = np.array(u_fom).reshape(N,N,N)
    u_rec_3d = np.array(u_rec).reshape(N,N,N)
    u_ex_3d  = np.array(u_ex ).reshape(N,N,N)

    mid = N // 2
    sl  = slice(None), slice(None), mid

    vmin = min(u_fom_3d[sl].min(), u_rec_3d[sl].min(), u_ex_3d[sl].min())
    vmax = max(u_fom_3d[sl].max(), u_rec_3d[sl].max(), u_ex_3d[sl].max())
    kw   = dict(origin='lower', aspect='auto', cmap='viridis',
                vmin=vmin, vmax=vmax, extent=[0,L,0,L])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, data, title in zip(axes[:3],
                                [u_fom_3d[sl], u_rec_3d[sl], u_ex_3d[sl]],
                                ['FOM (CG)', 'Reconstructed', 'Analytical']):
        im = ax.imshow(data.T, **kw)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)

    err = np.abs(u_rec_3d[sl] - u_fom_3d[sl])
    im  = axes[3].imshow(err.T, origin='lower', aspect='auto',
                          cmap='hot', extent=[0,L,0,L])
    axes[3].set_title('|Rec - FOM|', fontsize=11)
    axes[3].set_xlabel('x'); axes[3].set_ylabel('y')
    plt.colorbar(im, ax=axes[3], shrink=0.8)

    fig.suptitle(f'Midplane z={mid*dx:.2f} | k=({k1},{k2},{k3}) | '
                 f'Rel-L2={relative_l2(u_rec, u_fom):.3e}', fontsize=12)
    plt.tight_layout()
    fpath = OUT / f'slice_k{k1}{k2}{k3}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# Superposition validation cases — plot all val_superpositions
# Use physical (unnormalized) data for plotting
for i, cfg in enumerate(val_superpositions):
    u_fom = U_val_raw[i]  # physical amplitude
    u_rec = encode_and_decode_physical(u_fom)  # round-trip preserving amplitude
    u_ex  = get_exact_superposition(cfg)

    u_fom_3d = np.array(u_fom).reshape(N,N,N)
    u_rec_3d = np.array(u_rec).reshape(N,N,N)
    u_ex_3d  = np.array(u_ex ).reshape(N,N,N)

    mid = N // 2
    sl  = slice(None), slice(None), mid

    vmin = min(u_fom_3d[sl].min(), u_rec_3d[sl].min(), u_ex_3d[sl].min())
    vmax = max(u_fom_3d[sl].max(), u_rec_3d[sl].max(), u_ex_3d[sl].max())
    kw   = dict(origin='lower', aspect='auto', cmap='viridis',
                vmin=vmin, vmax=vmax, extent=[0,L,0,L])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, data, title in zip(axes[:3],
                                [u_fom_3d[sl], u_rec_3d[sl], u_ex_3d[sl]],
                                ['FOM (CG)', 'Reconstructed', 'Analytical']):
        im = ax.imshow(data.T, **kw)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)

    err = np.abs(u_rec_3d[sl] - u_fom_3d[sl])
    im  = axes[3].imshow(err.T, origin='lower', aspect='auto',
                          cmap='hot', extent=[0,L,0,L])
    axes[3].set_title('|Rec - FOM|', fontsize=11)
    axes[3].set_xlabel('x'); axes[3].set_ylabel('y')
    plt.colorbar(im, ax=axes[3], shrink=0.8)

    cfg_str = '+'.join([f"({k[0]},{k[1]},{k[2]})" for k, w in cfg])
    fig.suptitle(f'Midplane z={mid*dx:.2f} | {cfg_str} | '
                 f'Rel-L2={relative_l2(u_rec, u_fom):.3e}', fontsize=12)
    plt.tight_layout()
    fpath = OUT / f'slice_val_super_{i}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ─────────────────────────────────────────
# 13. Error Bar Chart — Training vs Val Cases
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
x    = np.arange(len(train_configs))
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
print(f"  CP rank:                256")
print(f"  Model params:           {n_params:,}")
print(f"  Dense AE params:        {param_count_dense(num_nodes):,}")
print(f"  Param reduction:        {param_count_dense(num_nodes)/n_params:.0f}×")
print(f"  Mean train rec error:   {np.mean(train_errs):.4e}")
print(f"  Mean val   rec error:   {np.mean(val_errs):.4e}")
print(f"  Plots saved to:         {OUT}/")
print(f"{'='*55}")

# ─────────────────────────────────────────
# 15. Export model state for downstream use (EQ phase)
# ─────────────────────────────────────────
import pickle
ckpt = {
    'params':      params,
    'model_cfg': dict(
        latent_dim    = k_dim,
        rank          = 256,
        grid_size     = N,
        conv_features = (32, 64, 128),
        hidden_dims   = (256, 512),
    )
}
with open(OUT / 'checkpoint.pkl', 'wb') as f:
    pickle.dump(ckpt, f)
print(f"\n  Checkpoint saved: {OUT / 'checkpoint.pkl'}")
print("  Keys: params, model_cfg")
print("  Load with:  import pickle; ck = pickle.load(open('...', 'rb'))")