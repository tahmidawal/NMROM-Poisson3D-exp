"""
nmrom_heat_sin.py
─────────────────
NM-ROM for parametric 3D Heat Equation with sinusoidal ICs.

Three-way validation at every test case:
  FOM vs analytical  — quantifies pure discretisation error
  ROM vs analytical  — quantifies total model + discretisation error
  ROM vs FOM         — quantifies purely the ROM approximation error

Parameter space tested:
  In-distribution:    k ∈ {1..4}³, κ ∈ training set
  Interpolation:      κ ∈ (0.01, 0.05) e.g. κ=0.03
  Extrapolation:      k=(5,5,5), κ=0.8  — outside training range
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from typing import Sequence

# ─────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────
CKPT_PATH  = Path('plots/heat_sin_ae/checkpoint.pkl')
DATA_PATH  = Path('plots/heat_sin_ae/training_data.pkl')
OUTPUT_DIR = Path('plots/nmrom_heat_sin')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 1. Grid & Physics
# ─────────────────────────────────────────
N         = 32
num_nodes = N ** 3
L         = 1.0
dx        = L / (N - 1)
dt        = 0.005
NUM_STEPS = 50
T_FINAL   = dt * NUM_STEPS

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

COORD_GRID = jnp.stack([2*X/L-1, 2*Y/L-1, 2*Z/L-1], axis=-1)
AMP_EPS    = 1e-6

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
    snapshots = [u0_flat]
    u  = u0_flat
    op = lambda v: implicit_op(v, kappa)
    for _ in range(steps):
        u, _ = jax_linalg.cg(op, u, x0=u, tol=1e-7, maxiter=2000)
        snapshots.append(u)
    return jnp.stack(snapshots)

mask_3d = jnp.ones((N,N,N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()
u_g     = jnp.zeros(num_nodes)

# ─────────────────────────────────────────
# 2. Sinusoidal IC / Analytical Solution
# ─────────────────────────────────────────
def lambda_k(k1, k2, k3):
    return float((k1**2 + k2**2 + k3**2) * np.pi**2)

def amplitude(k1, k2, k3, kappa):
    return float(np.exp(kappa * lambda_k(k1,k2,k3) * T_FINAL / 2))

def make_sin_ic(k1, k2, k3, kappa):
    A   = amplitude(k1, k2, k3, kappa)
    u3d = A * jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z)
    return u3d.flatten()

def get_analytical(k1, k2, k3, kappa, t):
    A   = amplitude(k1, k2, k3, kappa)
    dec = float(np.exp(-kappa * lambda_k(k1,k2,k3) * t))
    u3d = A * dec * jnp.sin(k1*jnp.pi*X) * jnp.sin(k2*jnp.pi*Y) * jnp.sin(k3*jnp.pi*Z)
    return u3d.flatten()

print(f"3D Heat (Sinusoidal) | {N}^3={num_nodes:,} DOF | dt={dt} | T={T_FINAL:.3f}s")

# ─────────────────────────────────────────────────────────────────────
# 3. Model Definition  (identical to training)
# ─────────────────────────────────────────────────────────────────────
def normalise(u_flat):
    scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
    return u_flat / scale, scale

def make_coordconv_input(u_flat):
    return jnp.concatenate([u_flat.reshape(N,N,N)[...,None], COORD_GRID], axis=-1)

class ResBlock3D(nn.Module):
    out_feats:  int
    num_groups: int = 8
    @nn.compact
    def __call__(self, x, training: bool = False):
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Conv(self.out_feats, (3,3,3), padding='SAME')(h)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Conv(self.out_feats, (3,3,3), padding='SAME')(h)
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
        h = x
        for feat in self.features:
            h = nn.Conv(feat, (3,3,3), strides=(2,2,2), padding='SAME')(h)
            h = nn.GroupNorm(num_groups=self.num_groups)(h)
            h = nn.leaky_relu(h, negative_slope=0.2)
            h = ResBlock3D(feat, num_groups=self.num_groups)(h, training)
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        H, W, D, C = h.shape
        if H != self.pool_size:
            h = jax.image.resize(h, (self.pool_size,)*3+(C,), method='linear')
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
        self.encoder = Conv3DEncoder(latent_dim=self.latent_dim, features=self.conv_features)
        self.decoder = SeparableDecoder(latent_dim=self.latent_dim, rank=self.rank,
                                        grid_size=self.grid_size, hidden_dims=self.hidden_dims)
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
# 4. Load Checkpoint
# ─────────────────────────────────────────
print(f"\n--- Loading checkpoint ---")
with open(CKPT_PATH, 'rb') as f:
    ckpt = pickle.load(f)

params = ckpt['params']
cfg    = ckpt['model_cfg']
meta   = ckpt['train_meta']
k_dim  = cfg['latent_dim']

model = ScalableAutoencoder(**cfg)
print(f"   latent_dim={k_dim}  rank={cfg['rank']}")

def encode(u_flat):
    z, scale = model.apply({'params': params}, u_flat, training=False,
                           method=model.encode)
    return z, scale

def decode_normalised(z):
    return model.apply({'params': params}, z, method=model.decode_normalised)

def decode_full(z, scale):
    return model.apply({'params': params}, z, scale, method=model.decode)

def constrained_decode_normalised(z):
    return mask * decode_normalised(z) + u_g

# ─────────────────────────────────────────────────────────────────────
# 5. Load Training Snapshots for EQ
# ─────────────────────────────────────────────────────────────────────
print(f"\n--- Loading training data for EQ ---")
with open(DATA_PATH, 'rb') as f:
    data_cache = pickle.load(f)

all_snapshots_eq = data_cache['all_snapshots']
traj_params_all  = data_cache['traj_params']
traj_kappas      = data_cache['traj_kappas']

# Subsample for EQ: use every N_EQ_TRAJ-th trajectory, every EQ_STRIDE steps
N_EQ_TRAJ  = 4     # fewer trajectories for faster EQ
EQ_STRIDE  = 10    # 5 pairs per trajectory → 20 pairs total

n_traj_total  = len(all_snapshots_eq)
eq_traj_idx   = np.linspace(0, n_traj_total-1, N_EQ_TRAJ, dtype=int)

eq_pairs = []   # (u_next, u_prev, kappa)
for i in eq_traj_idx:
    traj = all_snapshots_eq[i]
    kap  = traj_kappas[i]
    for step in range(0, NUM_STEPS, EQ_STRIDE):
        eq_pairs.append((traj[step+1], traj[step], kap))

n_pairs = len(eq_pairs)
print(f"   EQ pairs: {n_pairs}  ({N_EQ_TRAJ} trajs × {NUM_STEPS//EQ_STRIDE} steps)")

# ─────────────────────────────────────────────────────────────────────
# 6. EQ Offline Phase
# ─────────────────────────────────────────────────────────────────────
print("\n--- EQ Offline Phase ---")

# Column subsampling — NNLS on interior node subset
N_COL   = 3000
int_idx = np.where(np.array(mask) == 1)[0]
rng     = np.random.default_rng(seed=0)
col_sub = np.sort(rng.choice(int_idx, size=min(N_COL, len(int_idx)), replace=False))
col_jnp = jnp.array(col_sub)
print(f"   Column subsample: {len(col_sub)} nodes  |  "
      f"NNLS size: ({n_pairs*k_dim}, {len(col_sub)})")

# Batch encode
u_next_stack = jnp.stack([p[0] for p in eq_pairs])
u_prev_stack = jnp.stack([p[1] for p in eq_pairs])
kap_stack    = jnp.array([p[2] for p in eq_pairs], dtype=jnp.float32)

print("   Batch encoding...")
t0 = time.perf_counter()
encode_batch = jax.jit(jax.vmap(
    lambda u: model.apply({'params': params}, u, training=False, method=model.encode)
))
z_batch, _     = encode_batch(u_next_stack)
u_prev_norms   = u_prev_stack / (
    jnp.max(jnp.abs(u_prev_stack), axis=1, keepdims=True) + AMP_EPS
)
print(f"   Encoding: {time.perf_counter()-t0:.1f}s")

@jax.jit
def get_integrand_cols(z, u_prev_norm_cols, kappa):
    def cd_cols(z_):
        return (mask * decode_normalised(z_) + u_g)[col_jnp]
    def cd_full(z_):
        return mask * decode_normalised(z_) + u_g
    u_pred = cd_full(z)
    R_cols = (u_pred - (mask*u_prev_norm_cols+u_g) + dt*kappa*K_op_3d(u_pred))[col_jnp]
    J_cols = jax.jacfwd(cd_cols)(z)        # (N_COL, k_dim)
    return J_cols.T * R_cols[None, :]      # (k_dim, N_COL)

# Note: u_prev_norm_cols needs to be the full-size normalised vector
@jax.jit
def get_integrand_cols_v2(z, u_prev_norm_full, kappa):
    def cd_cols(z_):
        return (mask * decode_normalised(z_) + u_g)[col_jnp]
    def cd_full(z_):
        return mask * decode_normalised(z_) + u_g
    u_pred = cd_full(z)
    R_full = u_pred - u_prev_norm_full + dt * kappa * K_op_3d(u_pred)
    R_cols = R_full[col_jnp]
    J_cols = jax.jacfwd(cd_cols)(z)
    return J_cols.T * R_cols[None, :]

print("   Computing integrand matrices...")
t0 = time.perf_counter()
G_list = []
for idx in range(n_pairs):
    G_list.append(get_integrand_cols_v2(
        z_batch[idx], u_prev_norms[idx], kap_stack[idx]
    ))
    if (idx+1) % 50 == 0:
        print(f"   {idx+1}/{n_pairs}  ({time.perf_counter()-t0:.1f}s)")

G_np    = np.array(jnp.concatenate(G_list, axis=0))
b_np    = np.sum(G_np, axis=1)

print(f"   NNLS ({G_np.shape})...")
t_nnls  = time.perf_counter()
w_eq, _ = nnls(G_np, b_np)
print(f"   NNLS done in {time.perf_counter()-t_nnls:.1f}s")

local_nz   = np.where(w_eq > 1e-10)[0]
eq_indices = col_sub[local_nz]
eq_weights = w_eq[local_nz]
eq_idx_jnp = jnp.array(eq_indices)
eq_w_jnp   = jnp.array(eq_weights)
n_eq       = len(eq_indices)
print(f"   EQ total: {time.perf_counter()-t0:.1f}s")
print(f"   Nodes: {num_nodes:,} → {n_eq}  ({100*n_eq/num_nodes:.3f}%)")

# ─────────────────────────────────────────────────────────────────────
# 7. Precompute V_eq
# ─────────────────────────────────────────────────────────────────────
print("\n--- Precomputing V_eq ---")
N2              = N * N
stencil_off     = jnp.array([0, -1, 1, -N, N, -N2, N2])
gather_idx      = (eq_idx_jnp[:,None] + stencil_off[None,:]).flatten()

ix = gather_idx // N2
iy = (gather_idx // N) % N
iz = gather_idx % N

W_x = params['decoder']['W_x']
W_y = params['decoder']['W_y']
W_z = params['decoder']['W_z']

V_eq     = W_x[:,ix] * W_y[:,iy] * W_z[:,iz]   # (rank, n_eq*7)
b_scalar = params['decoder']['bias']
b_sparse = jnp.full(gather_idx.shape, b_scalar)
mask_sp  = mask[gather_idx]
ug_sp    = u_g[gather_idx]
eq_c_idx = eq_idx_jnp   # center indices (stencil offset 0)

print(f"   V_eq: {V_eq.shape}")

# ─────────────────────────────────────────────────────────────────────
# 8. LM-GN Solver
# ─────────────────────────────────────────────────────────────────────
def make_solver(p, V_eq_, b_sp, mask_sp_, ug_sp_, eq_w, lat_dim, dx_, dt_):
    dec = p['decoder']
    W0, b0 = dec['hidden_layers_0']['kernel'], dec['hidden_layers_0']['bias']
    W1, b1 = dec['hidden_layers_1']['kernel'], dec['hidden_layers_1']['bias']
    W2, b2 = dec['hidden_layers_2']['kernel'], dec['hidden_layers_2']['bias']
    Wr, br = dec['to_rank']['kernel'],         dec['to_rank']['bias']
    Wskip  = dec['z_proj']['kernel'];          bskip = dec['z_proj']['bias']

    def _mlp(z):
        h = nn.swish(z @ W0 + b0)
        h = nn.swish(h @ W1 + b1)
        h = nn.swish(h @ W2 + b2)
        return (h + z @ Wskip + bskip) @ Wr + br

    def _res(z, u_prev_norm_eq, kappa_):
        h   = _mlp(z)
        u_s = (mask_sp_ * (h @ V_eq_ + b_sp) + ug_sp_).reshape(-1, 7)
        lap = (6*u_s[:,0] - u_s[:,1] - u_s[:,2]
               - u_s[:,3] - u_s[:,4]
               - u_s[:,5] - u_s[:,6]) / dx_**2
        return (u_s[:,0] - u_prev_norm_eq) + dt_ * kappa_ * lap

    @jax.jit
    def solve(z_init, u_prev_norm_eq, kappa_):
        R0     = _res(z_init, u_prev_norm_eq, kappa_)
        J0     = jax.jacfwd(lambda l: _res(l, u_prev_norm_eq, kappa_))(z_init)
        gnorm0 = jnp.maximum(jnp.linalg.norm(J0.T @ (eq_w * R0)), 1e-30)

        def _body(carry):
            z, _, itr = carry
            R  = _res(z, u_prev_norm_eq, kappa_)
            J  = jax.jacfwd(lambda l: _res(l, u_prev_norm_eq, kappa_))(z)
            WJ   = eq_w[:,None] * J
            JtWJ = J.T @ WJ
            JtWr = J.T @ (eq_w * R)
            lam  = jnp.maximum(1e-3 * jnp.trace(JtWJ) / lat_dim, 1e-8)
            dz   = jnp.linalg.solve(JtWJ + lam*jnp.eye(lat_dim), -JtWr)
            f0   = jnp.dot(eq_w*R, R)
            def _f(a):
                Rt = _res(z+a*dz, u_prev_norm_eq, kappa_)
                return jnp.dot(eq_w*Rt, Rt)
            f1,f2,f3,f4 = _f(1.),_f(.5),_f(.25),_f(.125)
            step = jnp.where(f1<f0,1., jnp.where(f2<f0,.5,
                   jnp.where(f3<f0,.25, jnp.where(f4<f0,.125, 0.))))
            return z+step*dz, jnp.linalg.norm(JtWr), itr+1

        def _cond(carry):
            _, gnorm, itr = carry
            return jnp.logical_and(gnorm > 1e-4*gnorm0, itr < 25)

        init = (z_init, jnp.inf, jnp.array(0, jnp.int32))
        return jax.lax.while_loop(_cond, _body, init)

    return solve

latent_solve = make_solver(
    params, V_eq, b_sparse, mask_sp, ug_sp,
    eq_w_jnp, k_dim, dx, dt
)

# ─────────────────────────────────────────
# 9. ROM Time-Stepping
# ─────────────────────────────────────────
def run_rom(u0_flat, kappa, steps):
    kappa_j   = jnp.float32(kappa)
    z, scale  = encode(u0_flat)
    u_cur     = decode_full(z, scale)
    snapshots = [u_cur]
    gn_iters  = []

    for _ in range(steps):
        u_cur_norm    = constrained_decode_normalised(z)
        u_prev_eq     = u_cur_norm[eq_c_idx]
        z, _, n_iters = latent_solve(z, u_prev_eq, kappa_j)
        u_cur         = decode_full(z, scale)
        snapshots.append(u_cur)
        gn_iters.append(int(n_iters))

    jax.block_until_ready(u_cur)
    return jnp.stack(snapshots), gn_iters

# ─────────────────────────────────────────
# 10. Warm-Up
# ─────────────────────────────────────────
print("\n--- Warming up JIT ---")
_u0    = make_sin_ic(2, 2, 2, 0.05)
_z0, _ = encode(_u0)
_uprev = constrained_decode_normalised(_z0)[eq_c_idx]
for _ in range(2):
    _z1, _, _ = latent_solve(_z0, _uprev, jnp.float32(0.05))
jax.block_until_ready(_z1)
print("   Done.\n")

# ─────────────────────────────────────────────────────────────────────
# 11. Benchmark
#
# Three categories:
#   A) In-distribution:  k,κ in training set
#   B) Interpolation:    κ between training values
#   C) Extrapolation:    k or κ outside training range
# ─────────────────────────────────────────────────────────────────────
print("--- Benchmark ---")

test_cases = [
    # Category A — in-distribution
    dict(k=(1,1,1), kappa=0.01,  label='In  k=(1,1,1) κ=0.01'),
    dict(k=(2,2,2), kappa=0.05,  label='In  k=(2,2,2) κ=0.05'),
    dict(k=(3,3,3), kappa=0.1,   label='In  k=(3,3,3) κ=0.1'),
    dict(k=(4,4,4), kappa=0.5,   label='In  k=(4,4,4) κ=0.5'),
    dict(k=(1,2,3), kappa=0.1,   label='In  k=(1,2,3) κ=0.1'),
    dict(k=(2,3,4), kappa=0.05,  label='In  k=(2,3,4) κ=0.05'),
    # Category B — κ interpolation
    dict(k=(2,2,2), kappa=0.03,  label='Itp k=(2,2,2) κ=0.03'),
    dict(k=(3,3,3), kappa=0.2,   label='Itp k=(3,3,3) κ=0.2'),
    dict(k=(1,3,2), kappa=0.07,  label='Itp k=(1,3,2) κ=0.07'),
    # Category C — extrapolation
    dict(k=(5,5,5), kappa=0.01,  label='Ext k=(5,5,5) κ=0.01'),
    dict(k=(1,1,1), kappa=0.8,   label='Ext k=(1,1,1) κ=0.8'),
    dict(k=(5,3,2), kappa=0.5,   label='Ext k=(5,3,2) κ=0.5'),
]

fom_times, rom_times           = [], []
err_fom_exact, err_rom_exact   = [], []
err_rom_fom                    = []
stored                         = {}

print(f"\n  {'Label':<28} {'FOM(s)':>7} {'ROM(s)':>7} "
      f"{'FOM/exact':>11} {'ROM/exact':>11} {'ROM/FOM':>9}")
print("  " + "─"*75)

for i, tc in enumerate(test_cases):
    k1,k2,k3 = tc['k']
    kappa     = tc['kappa']
    label     = tc['label']

    u0       = make_sin_ic(k1, k2, k3, kappa)
    u_exact_T = get_analytical(k1, k2, k3, kappa, T_FINAL)
    norm_ex  = float(jnp.linalg.norm(u_exact_T))

    # FOM
    t0    = time.perf_counter()
    U_fom = run_fom(u0, kappa, NUM_STEPS)
    U_fom[-1].block_until_ready()
    fom_t = time.perf_counter() - t0
    fom_times.append(fom_t)

    # ROM
    t0    = time.perf_counter()
    U_rom, gn_iters = run_rom(u0, kappa, NUM_STEPS)
    rom_t = time.perf_counter() - t0
    rom_times.append(rom_t)

    # Three-way errors at final time
    e_fom = float(jnp.linalg.norm(U_fom[-1] - u_exact_T) / (norm_ex + 1e-12))
    e_rom = float(jnp.linalg.norm(U_rom[-1] - u_exact_T) / (norm_ex + 1e-12))
    e_rf  = float(jnp.linalg.norm(U_rom[-1] - U_fom[-1]) / (jnp.linalg.norm(U_fom[-1]) + 1e-12))

    err_fom_exact.append(e_fom)
    err_rom_exact.append(e_rom)
    err_rom_fom.append(e_rf)

    print(f"  {label:<28} {fom_t:>7.3f} {rom_t:>7.3f} "
          f"{e_fom:>11.3e} {e_rom:>11.3e} {e_rf:>9.3e}")

    # Store time-series errors for ALL cases
    if True:
        t_axis = np.arange(NUM_STEPS+1) * dt
        errs_t = []
        for s in range(NUM_STEPS+1):
            u_ex = get_analytical(k1, k2, k3, kappa, s*dt)
            n_ex = float(jnp.linalg.norm(u_ex))
            errs_t.append((
                float(jnp.linalg.norm(U_fom[s] - u_ex) / (n_ex+1e-12)),
                float(jnp.linalg.norm(U_rom[s] - u_ex) / (n_ex+1e-12)),
            ))
        stored[i] = dict(U_fom=np.array(U_fom), U_rom=np.array(U_rom),
                         errs_t=errs_t, k=(k1,k2,k3), kappa=kappa,
                         gn_iters=gn_iters, label=label)

# ─────────────────────────────────────────
# 12. Summary
# ─────────────────────────────────────────
avg_fom    = float(np.mean(fom_times))
avg_rom    = float(np.mean(rom_times))
avg_sp     = avg_fom / avg_rom
avg_e_fe   = float(np.mean(err_fom_exact))
avg_e_re   = float(np.mean(err_rom_exact))
avg_e_rf   = float(np.mean(err_rom_fom))

print(f"\n{'='*60}")
print(f"  3D Heat (Sinusoidal) — EQ-ROM Benchmark")
print(f"{'='*60}")
print(f"  Grid: {N}^3={num_nodes:,} DOF | k_dim={k_dim} | EQ={n_eq} nodes")
print(f"{'--'*30}")
print(f"  Avg FOM time:       {avg_fom:.4f} s")
print(f"  Avg ROM time:       {avg_rom:.4f} s")
print(f"  Avg speedup:        {avg_sp:.2f}x")
print(f"{'--'*30}")
print(f"  FOM vs analytical:  {avg_e_fe:.4e}")
print(f"  ROM vs analytical:  {avg_e_re:.4e}  ← total error")
print(f"  ROM vs FOM:         {avg_e_rf:.4e}  ← ROM-only error")
print(f"{'='*60}\n")

# ─────────────────────────────────────────
# 13. Plots
# ─────────────────────────────────────────
t_axis = np.arange(NUM_STEPS+1) * dt

# Plot A: Three-way error over time — grid layout for ALL benchmark cases
n_cases = len(stored)
n_cols = 4
n_rows = (n_cases + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten() if n_cases > 1 else [axes]

for ax, (idx, s) in zip(axes, stored.items()):
    errs_fom = [e[0] for e in s['errs_t']]
    errs_rom = [e[1] for e in s['errs_t']]
    ax.semilogy(t_axis, errs_fom, label='FOM vs exact', color='#2ca02c', lw=2)
    ax.semilogy(t_axis, errs_rom, label='ROM vs exact', color='#1f77b4', lw=2)
    ax.set_title(s['label'], fontsize=9)
    ax.set_xlabel('t (s)'); ax.set_ylabel('Rel L2')
    ax.legend(fontsize=7, loc='upper left'); ax.grid(True, which='both', ls='--', alpha=0.4)

# Hide unused axes
for ax in axes[n_cases:]:
    ax.axis('off')

fig.suptitle('Heat NM-ROM — Error over time vs analytical solution (all cases)', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'error_over_time_all.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'error_over_time_all.png'}")

# Plot B: Three-way error bar chart
cats   = [tc['label'] for tc in test_cases]
x_pos  = np.arange(len(test_cases))
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x_pos-0.25, err_fom_exact, 0.25, label='FOM vs exact', color='#2ca02c', alpha=0.8)
ax.bar(x_pos,      err_rom_exact, 0.25, label='ROM vs exact', color='#1f77b4', alpha=0.8)
ax.bar(x_pos+0.25, err_rom_fom,   0.25, label='ROM vs FOM',   color='#9467bd', alpha=0.8)
ax.set_yscale('log')
ax.set_xticks(x_pos); ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Relative L2 Error'); ax.legend()
ax.set_title('Three-way Error Comparison at t=T')
ax.grid(True, which='both', ls='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'three_way_error.png', dpi=150)
plt.close()

# Plot C: Speedup
speedups = [f/r for f,r in zip(fom_times, rom_times)]
fig, ax  = plt.subplots(figsize=(14, 4))
bars     = ax.bar(x_pos, speedups, color='#9467bd', alpha=0.85, edgecolor='k')
ax.axhline(1.0,    color='red',    ls='--', lw=1.5, label='Parity')
ax.axhline(avg_sp, color='orange', ls='--', lw=1.5, label=f'Avg {avg_sp:.1f}x')
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{sp:.1f}x', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x_pos); ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Speedup'); ax.legend()
ax.set_title(f'Speedup by test case (avg {avg_sp:.1f}x)')
ax.grid(True, axis='y', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'speedup.png', dpi=150)
plt.close()

# Plot D: Midplane slices with three-way comparison for first stored case
for idx, s in stored.items():
    k1,k2,k3 = s['k']; kappa = s['kappa']
    mid = N // 2
    t_check = [0, NUM_STEPS//2, NUM_STEPS]
    fig, axes = plt.subplots(len(t_check), 4, figsize=(16, 4*len(t_check)))
    for row, ti in enumerate(t_check):
        t_val  = ti * dt
        u_ex   = np.array(get_analytical(k1,k2,k3,kappa,t_val)).reshape(N,N,N)
        u_fom  = s['U_fom'][ti].reshape(N,N,N)
        u_rom  = s['U_rom'][ti].reshape(N,N,N)
        vmax   = max(float(np.abs(u_ex[:,:,mid]).max()), 1e-8)
        kw     = dict(origin='lower', aspect='auto', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, extent=[0,L,0,L])
        for ax, data, title in zip(axes[row,:3],
                                   [u_ex, u_fom, u_rom],
                                   ['Analytical', 'FOM', 'ROM']):
            im = ax.imshow(data[:,:,mid].T, **kw)
            ax.set_title(f'{title} t={t_val:.3f}', fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
        err_img = np.abs(u_rom - u_fom)
        im = axes[row,3].imshow(err_img[:,:,mid].T, origin='lower', aspect='auto',
                                 cmap='hot', extent=[0,L,0,L])
        axes[row,3].set_title('|ROM-FOM|', fontsize=10)
        plt.colorbar(im, ax=axes[row,3], shrink=0.8)
    fig.suptitle(f'{s["label"]} | GN avg={np.mean(s["gn_iters"]):.1f} itr',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'slices_{idx+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / f'slices_{idx+1}.png'}")

print("\n=== 3D Heat Sinusoidal NM-ROM Complete ===")
