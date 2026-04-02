"""
nmrom_heat.py
─────────────
3D Heat Equation NM-ROM using ScalableAutoencoder checkpoint.

Pipeline:
  1.  Load checkpoint (params, batch_stats, model_cfg, train_meta)
  2.  Rebuild training trajectories for EQ integrand computation
  3.  EQ offline phase — NNLS → sparse weights + indices
  4.  Precompute V_eq from separable factor matrices
  5.  JIT-compiled LM-GN time-step solver (lax.while_loop)
  6.  ROM time-stepping loop
  7.  Benchmark vs FOM — error over time, speedup, slice plots

Residual (Backward Euler, F=0):
  R(z) = ũ(z)[eq] - u_prev[eq] + dt*κ * [K ũ(z)][eq]

where ũ(z) = mask ⊙ D(z) + u_g  (constrained decode)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.optimize import nnls
from scipy.stats import qmc
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from typing import Sequence

# ─────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────
CKPT_PATH  = Path('plots/heat_ae/checkpoint.pkl')
OUTPUT_DIR  = Path('plots/nmrom_heat')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 1. Grid & Physics  (must match training)
# ─────────────────────────────────────────
N         = 32
num_nodes = N ** 3
L         = 1.0
dx        = L / (N - 1)
dt        = 0.005
NUM_STEPS = 50

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
    out = out.at[0,:,:].set(u[0,:,:]);  out = out.at[-1,:,:].set(u[-1,:,:])
    out = out.at[:,0,:].set(u[:,0,:]);  out = out.at[:,-1,:].set(u[:,-1,:])
    out = out.at[:,:,0].set(u[:,:,0]);  out = out.at[:,:,-1].set(u[:,:,-1])
    return out.flatten()

def implicit_op(u_flat, kappa):
    return u_flat + dt * kappa * K_op_3d(u_flat)

def run_fom(u0_flat, kappa, steps):
    snapshots = [u0_flat]
    u = u0_flat
    op = lambda v: implicit_op(v, kappa)
    for _ in range(steps):
        u, _ = jax_linalg.cg(op, u, x0=u, tol=1e-6, maxiter=1000)
        snapshots.append(u)
    return jnp.stack(snapshots)

def make_gaussian_ic(centers, amplitudes, widths):
    u = jnp.zeros((N, N, N))
    for (cx,cy,cz), A, sigma in zip(centers, amplitudes, widths):
        u = u + A * jnp.exp(
            -((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) / (2*sigma**2)
        )
    u = u.at[0,:,:].set(0.).at[-1,:,:].set(0.)
    u = u.at[:,0,:].set(0.).at[:,-1,:].set(0.)
    u = u.at[:,:,0].set(0.).at[:,:,-1].set(0.)
    return u.flatten()

# Boundary mask
mask_3d = jnp.ones((N,N,N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()
u_g     = jnp.zeros(num_nodes)

print(f"3D Heat — {N}³ = {num_nodes:,} nodes | dt={dt} | T={dt*NUM_STEPS:.3f}s")

# ─────────────────────────────────────────
# 2. Model Definition  (identical to training)
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
    dropout_rate: float = 0.1   # must match training definition — inactive at inference
    @nn.compact
    def __call__(self, x, training: bool = False):
        h = x[..., None]
        for feat in self.features:
            h = nn.Conv(feat, kernel_size=(3,3,3), strides=(2,2,2), padding='SAME')(h)
            h = nn.BatchNorm(use_running_average=not training, momentum=0.9)(h)
            h = nn.leaky_relu(h, negative_slope=0.2)
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        H, W, D, C = h.shape
        if H != self.pool_size:
            h = jax.image.resize(h, (self.pool_size,self.pool_size,self.pool_size,C),
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
        self.encoder = Conv3DEncoder(latent_dim=self.latent_dim, features=self.conv_features)
        self.decoder = SeparableDecoder(latent_dim=self.latent_dim, rank=self.rank,
                                        grid_size=self.grid_size, hidden_dims=self.hidden_dims)
    def encode(self, u_flat, training=False):
        return self.encoder(u_flat.reshape(self.grid_size, self.grid_size, self.grid_size),
                            training=training)
    def decode(self, z):
        return self.decoder(z)
    def __call__(self, u_flat, training=False):
        return self.decode(self.encode(u_flat, training=training))

# ─────────────────────────────────────────
# 3. Load Checkpoint
# ─────────────────────────────────────────
print(f"\n--- Loading checkpoint ---")
with open(CKPT_PATH, 'rb') as f:
    ckpt = pickle.load(f)

params      = ckpt['params']
batch_stats = ckpt['batch_stats']
cfg         = ckpt['model_cfg']
meta        = ckpt['train_meta']
k_dim       = cfg['latent_dim']

model = ScalableAutoencoder(**cfg)
print(f"   latent_dim={k_dim}  rank={cfg['rank']}  "
      f"grid={cfg['grid_size']}³")

# ── Convenience wrappers (always eval mode for ROM) ──────────────────
def encode(u_flat):
    return model.apply({'params': params, 'batch_stats': batch_stats},
                       u_flat, training=False, method=model.encode)

def decode(z):
    return model.apply({'params': params, 'batch_stats': batch_stats},
                       z, method=model.decode)

def constrained_decode(z):
    return mask * decode(z) + u_g

# ─────────────────────────────────────────────────────────────────────
# 4. Load Training Snapshots for EQ
#    Load from cache saved by train_heat_ae.py — never re-run FOM.
#    Subsample every EQ_STRIDE-th consecutive pair per trajectory.
# ─────────────────────────────────────────────────────────────────────
N_EQ_SAMPLES = 50     # Random samples for EQ — 50 is plenty
N_TRAIN     = meta['n_train']
traj_kappas = meta['traj_kappas']
DATA_PATH   = Path('plots/heat_ae/training_data.pkl')

print(f"\n--- Loading training snapshots for EQ from cache ---")
with open(DATA_PATH, 'rb') as f:
    data_cache = pickle.load(f)

all_snapshots_eq = data_cache['all_snapshots']   # list of (51, num_nodes) arrays
train_params     = data_cache['train_params']

# Collect all possible pairs, then randomly sample N_EQ_SAMPLES
all_pairs = []
for i, traj in enumerate(all_snapshots_eq):
    kap = traj_kappas[i]
    for step in range(NUM_STEPS):
        all_pairs.append((traj[step+1], traj[step], kap))

# Random subsample
rng_eq = np.random.default_rng(seed=123)
eq_indices = rng_eq.choice(len(all_pairs), size=min(N_EQ_SAMPLES, len(all_pairs)), replace=False)
eq_snapshot_pairs = [all_pairs[i] for i in eq_indices]

print(f"   Loaded {N_TRAIN} trajectories from cache")
print(f"   Snapshot pairs for EQ: {len(eq_snapshot_pairs)}")

# ─────────────────────────────────────────────────────────────────────
# 5. Empirical Quadrature — Offline Phase
#
# Residual for heat equation backward Euler:
#   R(z; u_prev, κ) = ũ(z) - u_prev + dt*κ*K(ũ(z))
#
# Integrand matrix (k_dim × num_nodes):
#   G[j, i] = (∂ũ/∂z_j)(i) · R(i)
#
# NNLS finds sparse weights w ≥ 0 s.t. G w ≈ Σ_i G[:,i]
# ─────────────────────────────────────────────────────────────────────
print("\n--- EQ Offline Phase ---")

@jax.jit
def get_integrand_heat(z_next, u_prev_full, kappa):
    """
    Returns G ∈ R^{k_dim × num_nodes} for one snapshot pair.
    R = ũ(z_next) - u_prev + dt*κ*K(ũ(z_next))
    """
    def cd(z):
        return mask * decode(z) + u_g

    u_pred = cd(z_next)
    R      = u_pred - u_prev_full + dt * kappa * K_op_3d(u_pred)
    J_D    = jax.jacfwd(cd)(z_next)    # (num_nodes, k_dim)
    return J_D.T * R[None, :]          # (k_dim, num_nodes)

G_list = []
print("   Computing integrand matrices...")
for idx, (u_next, u_prev, kap) in enumerate(eq_snapshot_pairs):
    z_next = encode(u_next)
    G_list.append(get_integrand_heat(z_next, u_prev, jnp.float32(kap)))
    if (idx+1) % 200 == 0:
        print(f"   {idx+1}/{len(eq_snapshot_pairs)}")

G_train    = jnp.concatenate(G_list, axis=0)   # (n_pairs*k_dim, num_nodes)
G_np       = np.array(G_train)
G_np[:, np.array(mask) == 0] = 0.0             # zero boundary columns
b_np       = np.sum(G_np, axis=1)

print("   Running NNLS...")
w_eq, _    = nnls(G_np, b_np)
eq_indices = np.where(w_eq > 1e-10)[0]
eq_weights = w_eq[eq_indices]

eq_idx_jnp = jnp.array(eq_indices)
eq_w_jnp   = jnp.array(eq_weights)
n_eq       = len(eq_indices)

print(f"   Nodes reduced: {num_nodes:,} → {n_eq}  ({100*n_eq/num_nodes:.3f}%)")

# ─────────────────────────────────────────────────────────────────────
# 6. Precompute V_eq — Separable Factor Product at EQ Stencil Indices
#
# For the heat equation we need:
#   - EQ center values (for time derivative term):  index 0 in stencil
#   - EQ 7-point stencil (for Laplacian):           all 7 neighbors
#
# V_eq[:, p] = W_x[:,ix_p] ⊙ W_y[:,iy_p] ⊙ W_z[:,iz_p]
# Same structure as Poisson — same single h @ V_eq matmul in hot path.
# ─────────────────────────────────────────────────────────────────────
print("\n--- Precomputing V_eq ---")

N2              = N * N
stencil_offsets = jnp.array([0, -1, 1, -N, N, -N2, N2])
gather_indices  = (eq_idx_jnp[:, None] + stencil_offsets[None, :]).flatten()

ix = gather_indices // N2
iy = (gather_indices // N) % N
iz = gather_indices % N

W_x = params['decoder']['W_x']   # (rank, N)
W_y = params['decoder']['W_y']
W_z = params['decoder']['W_z']

V_eq      = W_x[:, ix] * W_y[:, iy] * W_z[:, iz]   # (rank, n_eq*7)
b_scalar  = params['decoder']['bias']
b_sparse  = jnp.full(gather_indices.shape, b_scalar)
mask_sp   = mask[gather_indices]
ug_sp     = u_g[gather_indices]

print(f"   V_eq: {V_eq.shape}  (rank × n_eq×7)")

# ─────────────────────────────────────────────────────────────────────
# 7. LM-GN Online Solver — One Time Step
#
# Residual at EQ points:
#   R_eq(z) = u_centers(z) - u_prev_eq + dt*κ * lap_eq(z)
#
# where:
#   u_stencil = (mask_sp * (h @ V_eq + b_sp) + ug_sp).reshape(-1, 7)
#   u_centers = u_stencil[:, 0]
#   lap_eq    = (6*[:,0] - [:,1] - [:,2] - [:,3] - [:,4] - [:,5] - [:,6]) / dx²
#
# This is the CORRECT Gauss-Newton formulation:
#   solve  J^T W J  δz = -J^T W R   (pure normal equations, no K in the operator)
# ─────────────────────────────────────────────────────────────────────
print("\n--- Building hyper-reduced solver ---")

def make_heat_latent_solver(p, V_eq_, b_sp, mask_sp_, ug_sp_,
                             eq_w, latent_dim, dx_, dt_):
    # Pull MLP weights once
    dec = p['decoder']
    W0, b0 = dec['hidden_layers_0']['kernel'], dec['hidden_layers_0']['bias']
    W1, b1 = dec['hidden_layers_1']['kernel'], dec['hidden_layers_1']['bias']
    Wr, br = dec['to_rank']['kernel'],         dec['to_rank']['bias']

    def _mlp_body(z):
        h = nn.swish(z @ W0 + b0)
        h = nn.swish(h @ W1 + b1)
        return h @ Wr + br                         # (rank,)

    def _res_fn(z, u_prev_eq_, kappa_):
        """
        Heat equation residual at EQ points.
        u_prev_eq_: (n_eq,) — previous solution at EQ CENTER nodes only
        kappa_:     scalar  — diffusivity for this trajectory
        """
        h         = _mlp_body(z)
        u_st      = (mask_sp_ * (h @ V_eq_ + b_sp) + ug_sp_).reshape(-1, 7)
        u_centers = u_st[:, 0]                    # (n_eq,) — center values
        lap       = (6*u_st[:,0]
                     - u_st[:,1] - u_st[:,2]
                     - u_st[:,3] - u_st[:,4]
                     - u_st[:,5] - u_st[:,6]) / dx_**2
        # Backward Euler residual — same scale as get_integrand_heat:
        #   R = (u_{n+1} - u_n) + dt*κ*K(u_{n+1})   (NOT divided by dt)
        return (u_centers - u_prev_eq_) + dt_ * kappa_ * lap

    @jax.jit
    def solve_step(z_init, u_prev_eq_, kappa_):
        """
        LM-GN with Armijo backtracking via lax.while_loop.
        Correct J^T J normal equations — no K in the operator.
        Relative convergence: stops when ||J^T W R|| < tol * ||J^T W R_0||
        """
        # Compute initial gradient norm for relative tolerance
        R0      = _res_fn(z_init, u_prev_eq_, kappa_)
        J0      = jax.jacfwd(lambda l: _res_fn(l, u_prev_eq_, kappa_))(z_init)
        gnorm0  = jnp.linalg.norm(J0.T @ (eq_w * R0))
        # Avoid dividing by zero if already converged at init
        gnorm0  = jnp.maximum(gnorm0, 1e-30)

        def _body(carry):
            z, _, itr = carry
            R  = _res_fn(z, u_prev_eq_, kappa_)
            J  = jax.jacfwd(lambda l: _res_fn(l, u_prev_eq_, kappa_))(z)

            WJ   = eq_w[:, None] * J
            JtWJ = J.T @ WJ
            JtWr = J.T @ (eq_w * R)

            lam = jnp.maximum(1e-3 * jnp.trace(JtWJ) / latent_dim, 1e-8)
            dz  = jnp.linalg.solve(JtWJ + lam * jnp.eye(latent_dim), -JtWr)

            # Armijo backtracking — check all 4 steps, fall back to 0 if none help
            f0          = jnp.dot(eq_w * R, R)
            def _f(alpha):
                Rt = _res_fn(z + alpha*dz, u_prev_eq_, kappa_)
                return jnp.dot(eq_w * Rt, Rt)
            f1,f2,f3,f4 = _f(1.), _f(.5), _f(.25), _f(.125)
            step = jnp.where(f1 < f0, 1.0,
                   jnp.where(f2 < f0, 0.5,
                   jnp.where(f3 < f0, 0.25,
                   jnp.where(f4 < f0, 0.125, 0.0))))  # 0 = skip if no descent

            return z + step*dz, jnp.linalg.norm(JtWr), itr + 1

        def _cond(carry):
            _, gnorm, itr = carry
            # Relative convergence: stop when gradient drops to 1e-4 of initial
            not_converged = gnorm > 1e-4 * gnorm0
            not_maxed     = itr < 25
            return jnp.logical_and(not_converged, not_maxed)

        init = (z_init,
                jnp.array(jnp.inf, dtype=jnp.float32),
                jnp.array(0,       dtype=jnp.int32))
        return jax.lax.while_loop(_cond, _body, init)

    return solve_step


latent_solve = make_heat_latent_solver(
    params, V_eq, b_sparse, mask_sp, ug_sp,
    eq_w_jnp, k_dim, dx, dt
)

# ── EQ center indices (offset 0 in stencil = center of each EQ point) ──
eq_center_indices = eq_idx_jnp   # (n_eq,) — centers in full grid

def rom_time_step(z_prev, u_prev_eq, kappa_jnp):
    """Single ROM time step. Returns z_next, res, iters."""
    return latent_solve(z_prev, u_prev_eq, kappa_jnp)

def run_rom(u0_flat, kappa, steps):
    """
    Full ROM trajectory.
    Tracks latent state z_n; decodes only for output + u_prev_eq extraction.
    """
    kappa_jnp = jnp.float32(kappa)

    z     = encode(u0_flat)
    u_cur = constrained_decode(z)
    snapshots  = [u_cur]
    gn_iters   = []

    for step in range(steps):
        # u_prev at EQ center nodes — the only piece of u_prev needed in R
        u_prev_eq = u_cur[eq_center_indices]   # (n_eq,)

        z, res, n_iters = rom_time_step(z, u_prev_eq, kappa_jnp)
        u_cur = constrained_decode(z)
        snapshots.append(u_cur)
        gn_iters.append(int(n_iters))

    jax.block_until_ready(u_cur)
    return jnp.stack(snapshots), gn_iters

# ─────────────────────────────────────────
# 8. Warm-Up
# ─────────────────────────────────────────
print("\n--- Warming up JIT ---")
_tp    = train_params[0]
_u0    = make_gaussian_ic(_tp['centers'], _tp['amplitudes'], _tp['widths'])
_z0    = encode(_u0)
_uprev = constrained_decode(_z0)[eq_center_indices]
for _ in range(2):
    _z1, _, _ = latent_solve(_z0, _uprev, jnp.float32(_tp['kappa']))
jax.block_until_ready(_z1)
print("   Warm-up complete.\n")

# ─────────────────────────────────────────
# 9. Benchmark — Test Trajectories
# ─────────────────────────────────────────
print("--- Benchmark ---")

# Fresh seed — completely different from train (42), val (1337), and cache
def sample_test_params(rng, n_traj):
    sampler = qmc.LatinHypercube(d=17, seed=rng)
    samples = sampler.random(n=n_traj)
    trajs = []
    for s in samples:
        n_g = int(np.round(1 + 2*s[0]))
        centers, amplitudes, widths = [], [], []
        for g in range(n_g):
            centers.append((0.15+0.70*s[1+g*3],
                            0.15+0.70*s[2+g*3],
                            0.15+0.70*s[3+g*3]))
            amplitudes.append(1.0 + 9.0*s[10+g])
            widths.append(0.05 + 0.15*s[13+g])
        kappa = float(np.exp(np.log(0.01) + (np.log(0.5)-np.log(0.01))*s[16]))
        trajs.append(dict(centers=centers, amplitudes=amplitudes,
                          widths=widths, kappa=kappa))
    return trajs

test_params = sample_test_params(rng=9999, n_traj=10)

fom_times, rom_times     = [], []
final_errors             = []
energy_fom_list          = []
energy_rom_list          = []
stored                   = {}

for i, tp in enumerate(test_params):
    u0     = make_gaussian_ic(tp['centers'], tp['amplitudes'], tp['widths'])
    kap    = tp['kappa']
    n_gauss = len(tp['centers'])

    # FOM
    t0     = time.perf_counter()
    U_fom  = run_fom(u0, kap, NUM_STEPS)
    U_fom[-1].block_until_ready()
    fom_t  = time.perf_counter() - t0
    fom_times.append(fom_t)

    # ROM
    t0     = time.perf_counter()
    U_rom, gn_iters = run_rom(u0, kap, NUM_STEPS)
    rom_t  = time.perf_counter() - t0
    rom_times.append(rom_t)

    # Errors at each time step
    norms_fom = jnp.linalg.norm(U_fom, axis=1)
    err_t     = jnp.linalg.norm(U_rom - U_fom, axis=1) / (norms_fom + 1e-12)
    final_err = float(err_t[-1])
    final_errors.append(final_err)

    # Energy (L2 norm over time — should monotonically decay)
    energy_fom_list.append(np.array(norms_fom))
    energy_rom_list.append(np.array(jnp.linalg.norm(U_rom, axis=1)))

    avg_gn = float(np.mean(gn_iters))
    print(f"  [{i+1:2d}/10] κ={kap:.4f}  M={n_gauss} | "
          f"FOM {fom_t:.3f}s | ROM {rom_t:.3f}s | "
          f"Rel-L2(T) {final_err:.3e} | GN avg {avg_gn:.1f} itr")

    if i < 3:
        stored[i] = dict(U_fom=np.array(U_fom), U_rom=np.array(U_rom),
                         err_t=np.array(err_t), kappa=kap,
                         gn_iters=gn_iters, n_gauss=n_gauss)

# ─────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────
avg_fom   = float(np.mean(fom_times))
avg_rom   = float(np.mean(rom_times))
avg_sp    = avg_fom / avg_rom
avg_err   = float(np.mean(final_errors))

print(f"\n{'='*60}")
print(f"   3D Heat Equation — Scalable EQ-ROM Benchmark")
print(f"{'='*60}")
print(f"  Grid:                  {N}³ = {num_nodes:,} DOF")
print(f"  Latent dim:            {k_dim}")
print(f"  CP rank:               {cfg['rank']}")
print(f"  EQ nodes:              {n_eq} / {num_nodes} ({100*n_eq/num_nodes:.3f}%)")
print(f"  Time steps:            {NUM_STEPS}  (dt={dt}, T={dt*NUM_STEPS:.3f}s)")
print(f"{'--'*30}")
print(f"  Avg FOM time:          {avg_fom:.4f} s")
print(f"  Avg ROM time:          {avg_rom:.4f} s")
print(f"  Avg speedup:           {avg_sp:.2f}×")
print(f"{'--'*30}")
print(f"  Avg rel-L2 at T:       {avg_err:.4e}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────
# 11. Plots
# ─────────────────────────────────────────
t_axis = np.arange(NUM_STEPS+1) * dt

# ── Plot A: Error over time for all test cases ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for i, s in stored.items():
    ax.semilogy(t_axis, s['err_t'],
                label=f'Test {i+1}: κ={s["kappa"]:.3f}, M={s["n_gauss"]}',
                lw=2)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Relative $L_2$ Error (log)')
ax.set_title('Heat NM-ROM — ROM vs FOM Error Over Time')
ax.legend(fontsize=9); ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'error_over_time.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'error_over_time.png'}")

# ── Plot B: Energy conservation check ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for i, (ef, er) in enumerate(zip(energy_fom_list[:3], energy_rom_list[:3])):
    kap = test_params[i]['kappa']
    ax.plot(t_axis, ef, color=f'C{i}', lw=2,      label=f'FOM κ={kap:.3f}')
    ax.plot(t_axis, er, color=f'C{i}', lw=2, ls='--', label=f'ROM κ={kap:.3f}')
ax.set_xlabel('Time (s)'); ax.set_ylabel('$\\|u\\|_2$ (energy proxy)')
ax.set_title('Heat NM-ROM — Energy Decay: FOM (solid) vs ROM (dashed)')
ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'energy_decay.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'energy_decay.png'}")

# ── Plot C: Speedup bar chart ─────────────────────────────────────────
speedups = [f/r for f,r in zip(fom_times, rom_times)]
fig, ax  = plt.subplots(figsize=(10, 4))
bars     = ax.bar(range(1,11), speedups, color='#9467bd', alpha=0.85, edgecolor='black')
ax.axhline(1.0,    color='red',    ls='--', lw=1.5, label='Parity (1×)')
ax.axhline(avg_sp, color='orange', ls='--', lw=1.5, label=f'Avg = {avg_sp:.1f}×')
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{sp:.1f}×', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Test trajectory'); ax.set_ylabel('Speedup')
ax.set_title(f'3D Heat — Per-trajectory Speedup  (avg {avg_sp:.1f}×)')
ax.legend(); ax.grid(True, axis='y', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'speedup.png', dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'speedup.png'}")

# ── Plot D: Midplane slices at t=0, t=T/2, t=T for first 2 test cases ─
print("\n--- Saving midplane slice plots ---")
time_checkpoints = [0, NUM_STEPS//2, NUM_STEPS]

for case_idx, s in stored.items():
    U_fom_arr = s['U_fom']    # (51, num_nodes)
    U_rom_arr = s['U_rom']
    kap       = s['kappa']
    mid       = N // 2

    fig, axes = plt.subplots(len(time_checkpoints), 4,
                              figsize=(16, 4*len(time_checkpoints)))

    for row, t_idx in enumerate(time_checkpoints):
        u_fom_3d = U_fom_arr[t_idx].reshape(N,N,N)
        u_rom_3d = U_rom_arr[t_idx].reshape(N,N,N)

        sl_fom = u_fom_3d[:,:,mid]
        sl_rom = u_rom_3d[:,:,mid]
        vmax   = max(sl_fom.max(), sl_rom.max(), 1e-10)
        kw     = dict(origin='lower', aspect='auto', cmap='magma',
                      vmin=0, vmax=vmax, extent=[0,L,0,L])

        im0 = axes[row,0].imshow(sl_fom.T, **kw)
        axes[row,0].set_title(f'FOM  t={t_idx*dt:.3f}s', fontsize=10)
        axes[row,0].set_xlabel('x'); axes[row,0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[row,0], shrink=0.8)

        im1 = axes[row,1].imshow(sl_rom.T, **kw)
        axes[row,1].set_title(f'ROM  t={t_idx*dt:.3f}s', fontsize=10)
        axes[row,1].set_xlabel('x'); axes[row,1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[row,1], shrink=0.8)

        err_abs = np.abs(sl_rom - sl_fom)
        im2 = axes[row,2].imshow(err_abs.T, origin='lower', aspect='auto',
                                  cmap='hot', extent=[0,L,0,L])
        axes[row,2].set_title('|ROM - FOM|', fontsize=10)
        axes[row,2].set_xlabel('x'); axes[row,2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[row,2], shrink=0.8)

        # GN iterations per step
        axes[row,3].plot(range(1, NUM_STEPS+1), s['gn_iters'],
                         color='#2ca02c', lw=1.5)
        axes[row,3].set_title('GN iterations per step', fontsize=10)
        axes[row,3].set_xlabel('Step'); axes[row,3].set_ylabel('Iterations')
        axes[row,3].grid(True, ls='--', alpha=0.4)
        axes[row,3].set_ylim(0, 30)

    fig.suptitle(f'Heat NM-ROM | κ={kap:.4f} | M={s["n_gauss"]} sources '
                 f'| Rel-L2(T)={s["err_t"][-1]:.3e}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fpath = OUTPUT_DIR / f'slices_case{case_idx+1}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {fpath}")

print("\n=== 3D Heat NM-ROM Complete ===")