"""
nmrom_scalable.py
-----------------
NM-ROM (Non-linear Manifold Reduced Order Model) for 3D Poisson
using the ScalableAutoencoder (Conv3D encoder + Separable CP decoder).

Pipeline:
  1. Load trained checkpoint (params + batch_stats)
  2. Empirical Quadrature — offline phase
     - Compute integrand matrices G at training snapshots
     - NNLS → sparse EQ weights + indices
     - Precompute V_eq from separable factor matrices (replaces W_sparse)
  3. LM-GN hyper-reduced online solver
  4. Benchmark vs FOM + plots
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import pickle
import time
import sys
from pathlib import Path
from typing import Sequence

# ─────────────────────────────────────────
# 0. Paths & Logging
# ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
CKPT_PATH  = SCRIPT_DIR / 'checkpoint.pkl'
OUTPUT_DIR = SCRIPT_DIR / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging to file
LOG_FILE = SCRIPT_DIR / 'nmrom.log'
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
# 1. Grid & Physics  (identical to original)
# ─────────────────────────────────────────
N         = 32
num_nodes = N ** 3
L         = 1.0
dx        = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

def K_op_3d(u_flat):
    u   = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u)
    out = out.at[1:-1,1:-1,1:-1].set(
        (6*u[1:-1,1:-1,1:-1]
         - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
         - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
         - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
    )
    out = out.at[0,:,:].set(u[0,:,:])
    out = out.at[-1,:,:].set(u[-1,:,:])
    out = out.at[:,0,:].set(u[:,0,:])
    out = out.at[:,-1,:].set(u[:,-1,:])
    out = out.at[:,:,0].set(u[:,:,0])
    out = out.at[:,:,-1].set(u[:,:,-1])
    return out.flatten()

def get_F_3d(k1, k2, k3):
    F = jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z) * 10.0
    F = F.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    F = F.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    F = F.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    return F.flatten()

def get_analytical_solution_3d(k1, k2, k3):
    c = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (c * jnp.sin(k1*jnp.pi*X)
              * jnp.sin(k2*jnp.pi*Y)
              * jnp.sin(k3*jnp.pi*Z)).flatten()

def get_k2_scale(k1, k2, k3):
    """Return normalization factor: k1² + k2² + k3².
    Analytical solution scales as 1/k², so multiplying by k² normalizes."""
    return float(k1**2 + k2**2 + k3**2)

def full_order_fem_solver_3d(F_vec, u_guess=None):
    if u_guess is None:
        u_guess = jnp.zeros(num_nodes)
    u, _ = jax_linalg.cg(K_op_3d, F_vec, x0=u_guess, tol=1e-6, maxiter=2000)
    return u

# Boundary mask
mask_3d = jnp.ones((N, N, N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()
u_g     = jnp.zeros(num_nodes)

print(f"3D Domain: {N}³ = {num_nodes:,} nodes")
print(f"Interior : {int(mask.sum()):,}  |  Boundary: {num_nodes - int(mask.sum()):,}")

# ─────────────────────────────────────────
# 2. Model Definition  (must match training)
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
        N    = self.grid_size
        self.W_x         = self.param('W_x',  init, (self.rank, N))
        self.W_y         = self.param('W_y',  init, (self.rank, N))
        self.W_z         = self.param('W_z',  init, (self.rank, N))
        self.bias_scalar = self.param('bias', nn.initializers.zeros, ())

    def _mlp_body(self, z):
        h = z
        for layer in self.hidden_layers:
            h = nn.swish(layer(h))
        return self.to_rank(h)

    def __call__(self, z):
        h    = self._mlp_body(z)
        u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
        return u_3d.flatten() + self.bias_scalar


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
# 3. Load Checkpoint
# ─────────────────────────────────────────
print(f"\n--- Loading checkpoint from {CKPT_PATH} ---")
with open(CKPT_PATH, 'rb') as f:
    ckpt = pickle.load(f)

params      = ckpt['params']
batch_stats = ckpt['batch_stats']
cfg         = ckpt['model_cfg']
k_dim       = cfg['latent_dim']

model = ScalableAutoencoder(**cfg)
print(f"   Latent dim : {k_dim}")
print(f"   CP rank    : {cfg['rank']}")
print(f"   Grid size  : {cfg['grid_size']}")

# ── Convenience wrappers (always eval mode for ROM) ──────────────────
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

def constrained_decode(z):
    """ũ(z) = mask ⊙ D(z) + u_g  — hard Dirichlet BCs."""
    return mask * decode(z) + u_g

# ─────────────────────────────────────────
# 4. Rebuild Training Snapshots (Analytical + k²-Normalization)
#    (needed for EQ integrand computation)
# ─────────────────────────────────────────
print("\n--- Rebuilding training snapshots (Analytical, k²-normalized) ---")
train_ks = [(k1,k2,k3)
            for k1 in range(1,6)
            for k2 in range(1,6)
            for k3 in range(1,6)]   # 125 snapshots — same as training

U_train_list = []
scale_factors_train = []
for i, (k1,k2,k3) in enumerate(train_ks):
    u = get_analytical_solution_3d(k1, k2, k3)
    k2_scale = get_k2_scale(k1, k2, k3)
    u_normalized = u * k2_scale  # normalize by k²
    U_train_list.append(u_normalized)
    scale_factors_train.append(k2_scale)
    if (i+1) % 25 == 0:
        print(f"   {i+1}/{len(train_ks)} snapshots")

U_train = jnp.stack(U_train_list)
scale_factors_train = jnp.array(scale_factors_train)
print(f"   Shape: {U_train.shape}")
print(f"   Scale factors range: [{scale_factors_train.min():.0f}, {scale_factors_train.max():.0f}]")

# ─────────────────────────────────────────────────────────────────────
# 5. Empirical Quadrature — Offline Phase
#
# Integrand matrix per snapshot:
#   G[j, i] = (∂ũ/∂z_j)(i) · R(i)
#
# where R(i) = [Kũ - F](i)  and  J_D = ∂(constrained_decode)/∂z
#
# Stack all snapshots → NNLS → sparse weights w ≥ 0
# ─────────────────────────────────────────────────────────────────────
print("\n--- Empirical Quadrature: Offline Phase ---")

# Subsample for faster EQ
N_EQ_SAMPLES = 25
EQ_CACHE_PATH = SCRIPT_DIR / f'EQ_{N_EQ_SAMPLES}.pkl'

if EQ_CACHE_PATH.exists():
    print(f"   Loading cached EQ from {EQ_CACHE_PATH.name}")
    with open(EQ_CACHE_PATH, 'rb') as f:
        eq_cache = pickle.load(f)
    eq_indices = eq_cache['eq_indices']
    eq_weights_np = eq_cache['eq_weights']
    eq_indices_jnp = jnp.array(eq_indices)
    eq_weights_jnp = jnp.array(eq_weights_np)
    num_eq_points = len(eq_indices)
    print(f"   Loaded {num_eq_points} EQ points")
else:
    rng_eq = np.random.default_rng(seed=42)
    eq_sample_indices = sorted(rng_eq.choice(len(train_ks), size=min(N_EQ_SAMPLES, len(train_ks)), replace=False))
    print(f"   Using {len(eq_sample_indices)}/{len(train_ks)} snapshots for EQ")

    @jax.jit
    def get_integrand(lat_val, F_val_normalized):
        """Returns G ∈ R^{k_dim × num_nodes}
        Note: F_val_normalized should be F * k² to match normalized solution space."""
        R_full = K_op_3d(constrained_decode(lat_val)) - F_val_normalized
        J_D    = jax.jacfwd(constrained_decode)(lat_val)   # (num_nodes, k_dim)
        return J_D.T * R_full[None, :]                     # (k_dim, num_nodes)

    G_list = []
    print("   Computing integrand matrices...")
    for idx, i in enumerate(eq_sample_indices):
        k1, k2, k3 = train_ks[i]
        lat_i = encode(U_train[i])
        k2_scale = get_k2_scale(k1, k2, k3)
        F_normalized = get_F_3d(k1, k2, k3) * k2_scale  # scale F by k²
        G_list.append(get_integrand(lat_i, F_normalized))
        if (idx+1) % 10 == 0:
            print(f"   {idx+1}/{len(eq_sample_indices)} integrands done")

    G_train    = jnp.concatenate(G_list, axis=0)           # (N_EQ_SAMPLES*k_dim, num_nodes)
    G_train_np = np.array(G_train)
    G_train_np[:, np.array(mask) == 0] = 0.0               # zero out boundary cols
    b_train_np = np.sum(G_train_np, axis=1)

    print("   Running NNLS...")
    w_eq, nnls_res = nnls(G_train_np, b_train_np)
    eq_indices     = np.where(w_eq > 1e-10)[0]
    eq_weights_np  = w_eq[eq_indices]

    eq_indices_jnp  = jnp.array(eq_indices)
    eq_weights_jnp  = jnp.array(eq_weights_np)
    num_eq_points   = len(eq_indices)

    # Save EQ cache
    with open(EQ_CACHE_PATH, 'wb') as f:
        pickle.dump({'eq_indices': eq_indices, 'eq_weights': eq_weights_np}, f)
    print(f"   Saved EQ cache to {EQ_CACHE_PATH.name}")

print(f"   Nodes reduced: {num_nodes:,} → {num_eq_points}  "
      f"({100*num_eq_points/num_nodes:.3f}%)")

# ─────────────────────────────────────────────────────────────────────
# 6. Precompute V_eq — Separable Factor Product at EQ Stencil Indices
#
# Original dense approach:
#   W_sparse = W_final[:, gather_indices_flat]   (rank × num_eq*7)
#
# Separable approach — exactly equivalent, just computed differently:
#   V_eq[:, p] = W_x[:, ix_p] ⊙ W_y[:, iy_p] ⊙ W_z[:, iz_p]
#
# Same shape. Same matmul in the hot path. h @ V_eq is identical.
# ─────────────────────────────────────────────────────────────────────
print("\n--- Precomputing V_eq (separable sparse factors) ---")

N2              = N * N
stencil_offsets = jnp.array([0, -1, 1, -N, N, -N2, N2])  # 7-point
gather_indices  = (eq_indices_jnp[:, None]
                   + stencil_offsets[None, :]).flatten()   # (num_eq*7,)

ix = gather_indices // N2
iy = (gather_indices // N) % N
iz = gather_indices % N

# Pull factor matrices from checkpoint
W_x = params['decoder']['W_x']   # (rank, N)
W_y = params['decoder']['W_y']
W_z = params['decoder']['W_z']

# Element-wise product → (rank, num_eq*7)
V_eq     = W_x[:, ix] * W_y[:, iy] * W_z[:, iz]
b_scalar = params['decoder']['bias']
b_sparse = jnp.full(gather_indices.shape, b_scalar)

mask_sparse = mask[gather_indices]
u_g_sparse  = u_g[gather_indices]

print(f"   V_eq shape : {V_eq.shape}   (rank × num_eq*7)")
print(f"   EQ points  : {num_eq_points}")

# ─────────────────────────────────────────────────────────────────────
# 7. Hyper-Reduced Online Solver
#
# _res_fn structure is IDENTICAL to original make_latent_solver.
# Only difference: MLP body uses the new decoder param names,
# and final matmul uses V_eq instead of W_sparse.
# ─────────────────────────────────────────────────────────────────────
print("\n--- Building hyper-reduced solver ---")

def make_latent_solver(p, V_eq_, b_sp, mask_sp, ug_sp, eq_w, latent_dim):

    # ── Pull decoder MLP weights once (avoids dict lookup inside JIT) ──
    dec = p['decoder']
    # hidden_layers_0, hidden_layers_1 are how Flax names list entries in setup()
    W0, b0 = dec['hidden_layers_0']['kernel'], dec['hidden_layers_0']['bias']
    W1, b1 = dec['hidden_layers_1']['kernel'], dec['hidden_layers_1']['bias']
    Wr, br = dec['to_rank']['kernel'],         dec['to_rank']['bias']

    def _mlp_body(lat):
        h = nn.swish(lat @ W0 + b0)
        h = nn.swish(h   @ W1 + b1)
        return h @ Wr + br                         # (rank,)

    def _res_fn(lat, F_eq):
        h = _mlp_body(lat)                         # (rank,)
        # ← identical structure to original: one matmul + reshape
        u_stencil = (mask_sp * (h @ V_eq_ + b_sp) + ug_sp).reshape((-1, 7))
        R = (6 * u_stencil[:, 0]
             - u_stencil[:, 1] - u_stencil[:, 2]
             - u_stencil[:, 3] - u_stencil[:, 4]
             - u_stencil[:, 5] - u_stencil[:, 6]) / dx**2 - F_eq
        return R

    @jax.jit
    def solve(lat_init, F_eq):
        """LM-GN with Armijo backtracking — identical loop to original."""

        def _body(carry):
            lat, _, itr = carry
            R  = _res_fn(lat, F_eq)
            J  = jax.jacfwd(lambda l: _res_fn(l, F_eq))(lat)
            WJ   = eq_w[:, None] * J
            JtWJ = J.T @ WJ
            JtWr = J.T @ (eq_w * R)

            lam = jnp.maximum(1e-3 * jnp.trace(JtWJ) / latent_dim, 1e-8)
            dz  = jnp.linalg.solve(JtWJ + lam * jnp.eye(latent_dim), -JtWr)

            f0 = jnp.dot(eq_w * R, R)

            def _f(alpha):
                Rt = _res_fn(lat + alpha * dz, F_eq)
                return jnp.dot(eq_w * Rt, Rt)

            f1, f2, f3, f4 = _f(1.), _f(.5), _f(.25), _f(.125)
            step = jnp.where(f1 < f0, 1.0,
                   jnp.where(f2 < f0, 0.5,
                   jnp.where(f3 < f0, 0.25, 0.125)))

            return lat + step * dz, jnp.linalg.norm(JtWr), itr + 1

        def _cond(carry):
            _, gnorm, itr = carry
            return jnp.logical_and(gnorm > 1e-8, itr < 30)

        init = (lat_init,
                jnp.array(jnp.inf, dtype=jnp.float32),
                jnp.array(0,       dtype=jnp.int32))
        lat_f, res_f, n_iters = jax.lax.while_loop(_cond, _body, init)
        return lat_f, res_f, n_iters

    return solve


latent_solve = make_latent_solver(
    params, V_eq, b_sparse, mask_sparse, u_g_sparse,
    eq_weights_jnp, k_dim
)

def fast_eq_latent_poisson_solver(lat_init, F_vec, k2_scale):
    """Public interface: hyper-reduced solve → full reconstructed field.
    
    Args:
        lat_init: Initial latent code (in normalized space)
        F_vec: Original forcing vector (NOT normalized)
        k2_scale: Normalization factor k1² + k2² + k3²
    
    Returns:
        lat_f: Final latent code (normalized space)
        u_final: Reconstructed solution (denormalized to original scale)
        res_f: Final residual norm
        n_iters: Number of GN iterations
    """
    F_normalized = F_vec * k2_scale  # normalize F
    F_eq  = F_normalized[eq_indices_jnp]
    lat_f, res_f, n_iters = latent_solve(lat_init, F_eq)
    u_normalized = constrained_decode(lat_f)
    u_final = u_normalized / k2_scale  # denormalize output
    return lat_f, u_final, res_f, n_iters

# ─────────────────────────────────────────
# 8. Warm-up
# ─────────────────────────────────────────
print("\n--- Warming up JAX compilers ---")
_F_wm   = get_F_3d(2, 2, 2)
_k2_wm  = get_k2_scale(2, 2, 2)
full_order_fem_solver_3d(_F_wm).block_until_ready()
_lat_wm = encode(U_train[0])
for _ in range(2):
    _, _u_wm, _, _ = fast_eq_latent_poisson_solver(_lat_wm, _F_wm, _k2_wm)
jax.block_until_ready(_u_wm)
print("   Warm-up complete.\n")

# ─────────────────────────────────────────
# 9. Benchmark
# ─────────────────────────────────────────
print("--- Benchmark ---")

test_ks = [
    # In-distribution
    (1,1,1), (2,2,2), (3,3,3), (4,4,4),
    # Interpolation (in training set)
    (1,2,3), (2,3,4), (3,4,1),
    # Mild extrapolation
    (5,5,5), (1,2,5), (3,5,2),
    # Mixed
    (1,1,2), (2,2,3), (3,3,4), (1,2,2),
]

fom_times, rom_times         = [], []
fom_vs_exact_errors          = []
rom_vs_exact_errors          = []
stored                       = {}
n_test                       = len(test_ks)
plot_at                      = set(range(n_test))  # Store all 14 for combined plot

for i, (k1,k2,k3) in enumerate(test_ks):
    F_test = get_F_3d(k1, k2, k3)
    k2_scale = get_k2_scale(k1, k2, k3)

    # FOM (for timing comparison only)
    t0    = time.perf_counter()
    u_fom = full_order_fem_solver_3d(F_test).block_until_ready()
    fom_t = time.perf_counter() - t0
    fom_times.append(fom_t)

    # Latent init — inverse-distance weighted interpolation of 2 nearest snapshots
    dists     = [(k1-a)**2 + (k2-b)**2 + (k3-c)**2 for a,b,c in train_ks]
    sorted_i  = np.argsort(dists)
    i1, i2    = sorted_i[0], sorted_i[1]
    d1, d2    = np.sqrt(dists[i1]), np.sqrt(dists[i2])
    dsum      = d1 + d2
    w1        = d2 / dsum if dsum > 1e-12 else 0.5
    w2        = d1 / dsum if dsum > 1e-12 else 0.5
    lat_init  = w1 * encode(U_train[i1]) + w2 * encode(U_train[i2])

    # ROM (with k² normalization)
    t0 = time.perf_counter()
    lat_f, u_rom, gn_res, n_iters = fast_eq_latent_poisson_solver(lat_init, F_test, k2_scale)
    jax.block_until_ready(u_rom)
    rom_t = time.perf_counter() - t0
    rom_times.append(rom_t)

    # Errors — both compared against analytical (ground truth)
    u_exact      = get_analytical_solution_3d(k1, k2, k3)
    norm_exact   = float(jnp.linalg.norm(u_exact))

    err_fom_ex   = float(jnp.linalg.norm(u_fom - u_exact) / norm_exact)
    err_rom_ex   = float(jnp.linalg.norm(u_rom - u_exact) / norm_exact)

    fom_vs_exact_errors.append(err_fom_ex)
    rom_vs_exact_errors.append(err_rom_ex)

    print(f"  [{i+1:2d}/{n_test}] k=({k1},{k2},{k3}) | "
          f"FOM {fom_t:.4f}s | ROM {rom_t:.4f}s | "
          f"FOM-exact {err_fom_ex:.3e} | ROM-exact {err_rom_ex:.3e} | GN itr {int(n_iters)}")

    if i in plot_at:
        stored[i] = dict(u_fom=np.asarray(u_fom), u_rom=np.asarray(u_rom),
                         u_exact=np.asarray(u_exact), k=(k1,k2,k3))

# ─────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────
avg_fom_t   = float(np.mean(fom_times))
avg_rom_t   = float(np.mean(rom_times))
avg_speedup = avg_fom_t / avg_rom_t
avg_err_fe  = float(np.mean(fom_vs_exact_errors))
avg_err_re  = float(np.mean(rom_vs_exact_errors))

print(f"\n{'='*60}")
print(f"   3D Poisson  —  Scalable EQ-ROM Benchmark (k²-normalized)")
print(f"{'='*60}")
print(f"  Grid:                  {N}³ = {num_nodes:,} DOF")
print(f"  Latent dim:            {k_dim}")
print(f"  CP rank:               {cfg['rank']}")
print(f"  EQ nodes:              {num_eq_points} / {num_nodes} ({100*num_eq_points/num_nodes:.3f}%)")
print(f"{'--'*30}")
print(f"  Avg FOM time:          {avg_fom_t:.5f} s")
print(f"  Avg ROM time:          {avg_rom_t:.5f} s")
print(f"  Avg speedup:           {avg_speedup:.2f}×")
print(f"{'--'*30}")
print(f"  FOM vs analytical:     {avg_err_fe:.4e}")
print(f"  ROM vs analytical:     {avg_err_re:.4e}  (primary)")
print(f"{'='*60}\n")

# ─────────────────────────────────────────
# 11. Plots
# ─────────────────────────────────────────
test_labels = [f"({k[0]},{k[1]},{k[2]})" for k in test_ks]
x_pos       = np.arange(len(test_ks))

# ── Error bar chart ──────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x_pos - 0.2, rom_vs_exact_errors, width=0.4,
       color='#1f77b4', label='ROM vs Analytical (primary)')
ax.bar(x_pos + 0.2, fom_vs_exact_errors, width=0.4,
       color='#2ca02c', alpha=0.7, label='FOM vs Analytical')
ax.set_title('Scalable EQ-ROM — Relative $L_2$ Error vs Analytical', fontsize=13)
ax.set_xlabel('$(k_1, k_2, k_3)$'); ax.set_ylabel('Relative $L_2$ Error')
ax.set_xticks(x_pos); ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.set_yscale('log'); ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_error.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'benchmark_error.png'}")

# ── Timing bar chart ─────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x_pos - 0.2, fom_times, width=0.4, color='#d62728', label='FOM (CG)')
ax.bar(x_pos + 0.2, rom_times, width=0.4, color='#1f77b4', label='EQ-ROM')
ax.axhline(avg_fom_t, color='#d62728', ls=':', alpha=0.7,
           label=f'FOM avg ({avg_fom_t:.4f}s)')
ax.axhline(avg_rom_t, color='#1f77b4', ls=':', alpha=0.7,
           label=f'ROM avg ({avg_rom_t:.4f}s)')
ax.set_title(f'Scalable EQ-ROM — Solve Time  (avg speedup {avg_speedup:.1f}×)', fontsize=13)
ax.set_xlabel('$(k_1, k_2, k_3)$'); ax.set_ylabel('Time (s)')
ax.set_xticks(x_pos); ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.legend(); ax.grid(True, axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_time.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'benchmark_time.png'}")

# ── Speedup chart ────────────────────────
speedups = [f / r for f, r in zip(fom_times, rom_times)]
fig, ax  = plt.subplots(figsize=(13, 4))
bars     = ax.bar(x_pos, speedups, width=0.6, color='#9467bd',
                  alpha=0.85, edgecolor='black')
ax.axhline(1.0,         color='red',    ls='--', lw=1.5, label='Parity (1×)')
ax.axhline(avg_speedup, color='orange', ls='--', lw=1.5,
           label=f'Average = {avg_speedup:.1f}×')
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05, f'{sp:.1f}×',
            ha='center', va='bottom', fontsize=9)
ax.set_title('Scalable EQ-ROM — Per-test Speedup', fontsize=13)
ax.set_xlabel('$(k_1, k_2, k_3)$'); ax.set_ylabel('Speedup Factor')
ax.set_xticks(x_pos); ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.set_ylim(0, max(speedups) * 1.2); ax.legend()
ax.grid(True, axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_speedup.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'benchmark_speedup.png'}")

# ── Combined midplane slices (all 14 tests) ──────────────────────
print("\n--- Combined midplane slice plot (all 14 tests) ---")
mid = N // 2

# 14 rows (one per test), 4 columns: FOM, ROM, Analytical, |ROM-FOM|
fig, axes = plt.subplots(n_test, 4, figsize=(16, 3 * n_test))

for idx in sorted(stored):
    s        = stored[idx]
    k1,k2,k3 = s['k']

    u_fom_3d = s['u_fom'].reshape(N, N, N)
    u_rom_3d = s['u_rom'].reshape(N, N, N)
    u_ex_3d  = s['u_exact'].reshape(N, N, N)

    # Use z-midplane slice
    sl_fom = u_fom_3d[:, :, mid]
    sl_rom = u_rom_3d[:, :, mid]
    sl_ex  = u_ex_3d[:, :, mid]

    vmin = min(sl_fom.min(), sl_rom.min(), sl_ex.min())
    vmax = max(sl_fom.max(), sl_rom.max(), sl_ex.max())
    kw   = dict(origin='lower', aspect='auto', cmap='viridis',
                vmin=vmin, vmax=vmax, extent=[0, L, 0, L])

    # FOM
    im0 = axes[idx, 0].imshow(sl_fom.T, **kw)
    axes[idx, 0].set_ylabel(f'k=({k1},{k2},{k3})', fontsize=10, fontweight='bold')
    if idx == 0:
        axes[idx, 0].set_title('FOM', fontsize=12)

    # ROM
    im1 = axes[idx, 1].imshow(sl_rom.T, **kw)
    if idx == 0:
        axes[idx, 1].set_title('EQ-ROM', fontsize=12)

    # Analytical
    im2 = axes[idx, 2].imshow(sl_ex.T, **kw)
    if idx == 0:
        axes[idx, 2].set_title('Analytical', fontsize=12)

    # Error |ROM - Analytical|
    err    = np.abs(sl_rom - sl_ex)
    im_err = axes[idx, 3].imshow(err.T, origin='lower', aspect='auto',
                                 cmap='hot', extent=[0, L, 0, L])
    if idx == 0:
        axes[idx, 3].set_title('|ROM - Analytical|', fontsize=12)

    # Add colorbars on the right side
    plt.colorbar(im2, ax=axes[idx, 2], shrink=0.8, pad=0.02)
    plt.colorbar(im_err, ax=axes[idx, 3], shrink=0.8, pad=0.02)

    # Remove tick labels for cleaner look (except bottom row)
    for col in range(4):
        if idx < n_test - 1:
            axes[idx, col].set_xticklabels([])
        axes[idx, col].set_yticklabels([])

fig.suptitle('Scalable EQ-ROM — All Test Cases (z-midplane slices)',
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
fpath = OUTPUT_DIR / 'combined_slices_all.png'
plt.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {fpath}")

print("\n=== Scalable EQ-ROM Complete ===")