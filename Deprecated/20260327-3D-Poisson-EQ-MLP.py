import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.optimize import nnls, lsq_linear
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# ==========================================
# 0. Output Directories
# ==========================================
OUTPUT_DIR = Path(__file__).parent / 'plots' / '3D_poisson'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Domain Setup & Parametric Physics (3D)
# ==========================================
N = 32  # Grid size (32^3 = 32,768 nodes)
num_nodes = N ** 3
k_dim = 20  # Latent dimension
L = 1.0
dx = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

# CRITICAL: Flat array of all (x,y,z) spatial coordinates
coords_flat = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

def K_op_3d(u_flat):
    u_3d = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u_3d)
    out = out.at[1:-1, 1:-1, 1:-1].set(
        (6 * u_3d[1:-1, 1:-1, 1:-1]
         - u_3d[0:-2, 1:-1, 1:-1] - u_3d[2:, 1:-1, 1:-1]
         - u_3d[1:-1, 0:-2, 1:-1] - u_3d[1:-1, 2:, 1:-1]
         - u_3d[1:-1, 1:-1, 0:-2] - u_3d[1:-1, 1:-1, 2:]) / dx**2
    )
    out = out.at[0, :, :].set(u_3d[0, :, :])
    out = out.at[-1, :, :].set(u_3d[-1, :, :])
    out = out.at[:, 0, :].set(u_3d[:, 0, :])
    out = out.at[:, -1, :].set(u_3d[:, -1, :])
    out = out.at[:, :, 0].set(u_3d[:, :, 0])
    out = out.at[:, :, -1].set(u_3d[:, :, -1])
    return out.flatten()

def get_F_3d(k1, k2, k3):
    F_3d = (jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y) * jnp.sin(k3 * jnp.pi * Z) * 10.0)
    F_3d = F_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    F_3d = F_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    F_3d = F_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    return F_3d.flatten()

def get_analytical_solution_3d(k1, k2, k3):
    coeff = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (coeff * jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y) * jnp.sin(k3 * jnp.pi * Z)).flatten()

mask_3d = jnp.ones((N, N, N))
mask_3d = mask_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
mask_3d = mask_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
mask_3d = mask_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
mask = mask_3d.flatten()
u_g = jnp.zeros(num_nodes)

print(f"3D Domain: {N}x{N}x{N} = {num_nodes:,} nodes")

# ==========================================
# 2. DeepONet (Branch & Trunk) Autoencoder
# ==========================================
def positional_encoding(coords, num_freqs=6):
    """Maps (x,y,z) to a high-frequency space to destroy Spectral Bias."""
    freq_bands = 2.0 ** jnp.linspace(0.0, num_freqs - 1, num_freqs)
    pts = coords[..., None] * freq_bands * jnp.pi
    pts_sin = jnp.sin(pts)
    pts_cos = jnp.cos(pts)
    encoded = jnp.concatenate([pts_sin, pts_cos], axis=-1).reshape(coords.shape[:-1] + (-1,))
    return jnp.concatenate([coords, encoded], axis=-1)

class TrunkNet(nn.Module):
    """Spatial Network using Fourier Features instead of raw SIREN."""
    p: int = 128
    @nn.compact
    def __call__(self, xyz):
        x = positional_encoding(xyz, num_freqs=6)
        x = nn.swish(nn.Dense(128)(x))
        x = nn.swish(nn.Dense(128)(x))
        x = nn.swish(nn.Dense(128)(x))
        return nn.Dense(self.p)(x)

class BranchNet(nn.Module):
    """Latent Network: Maps latent z to p coefficients."""
    p: int = 128
    @nn.compact
    def __call__(self, lat):
        x = nn.swish(nn.Dense(128)(lat))
        x = nn.swish(nn.Dense(128)(x))
        x = nn.swish(nn.Dense(128)(x))
        return nn.Dense(self.p)(x)

class DeepONetAutoencoder(nn.Module):
    latent_dim: int
    N: int = 32
    p: int = 128

    def setup(self):
        # Efficient CNN Encoder
        self.enc_conv1 = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.enc_conv2 = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.enc_conv3 = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.enc_dense = nn.Dense(self.latent_dim)
        
        # Branch & Trunk
        self.trunk = TrunkNet(p=self.p)
        self.branch = BranchNet(p=self.p)

    def encode(self, x):
        x = x.reshape((self.N, self.N, self.N, 1))
        x = nn.swish(self.enc_conv1(x))
        x = nn.swish(self.enc_conv2(x))
        x = nn.swish(self.enc_conv3(x))
        return self.enc_dense(x.reshape((-1,)))

    def apply_trunk(self, xyz):
        return self.trunk(xyz)

    def apply_branch(self, lat):
        return self.branch(lat)

    def __call__(self, x, coords):
        """Full forward pass for training. Evaluates specifically on provided coords."""
        lat = self.encode(x)
        b_out = self.branch(lat)               # (p,)
        t_out = jax.vmap(self.trunk)(coords)   # (num_coords, p)
        return jnp.dot(t_out, b_out)           # (num_coords,)

def full_order_fem_solver_3d(K_operator, F_vec, u_guess, tol=1e-6):
    u_true, _ = jax_linalg.cg(K_operator, F_vec, x0=u_guess, tol=tol, maxiter=2000)
    return u_true

# ==========================================
# 3. Training Data
# ==========================================
print("\n--- 1. Generating 3D Training Dataset ---")
train_k_triples = [(k1, k2, k3) for k1 in range(1, 6) for k2 in range(1, 6) for k3 in range(1, 6)]
u_guess = jnp.zeros(num_nodes)

U_train_list = []
for i, (k1, k2, k3) in enumerate(train_k_triples):
    u_sol = full_order_fem_solver_3d(K_op_3d, get_F_3d(k1, k2, k3), u_guess)
    U_train_list.append(u_sol)
U_train = jnp.stack(U_train_list)

# ==========================================
# 4. Train DeepONet (Full Grid, No Batching)
# ==========================================
model = DeepONetAutoencoder(latent_dim=k_dim)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones(num_nodes), coords_flat[:10])['params']

print(f"\n--- 2. Training DeepONet Autoencoder ---")
print(f"   Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")

schedule = optax.exponential_decay(init_value=1e-3, transition_steps=2000, decay_rate=0.9)
tx = optax.adam(learning_rate=schedule)
opt_state = tx.init(params)

@jax.jit
def train_step(p, opt_st, batch):
    def loss_fn(weights):
        # 1. Encode entire batch to latents -> (125, 20)
        lats = jax.vmap(lambda inp: model.apply({'params': weights}, inp, method=model.encode))(batch)
        
        # 2. Evaluate Branch net -> (125, 128)
        b_out = jax.vmap(lambda z: model.apply({'params': weights}, z, method=model.apply_branch))(lats)
        
        # 3. Evaluate Trunk net on the ENTIRE grid -> (32768, 128)
        t_out = jax.vmap(lambda c: model.apply({'params': weights}, c, method=model.apply_trunk))(coords_flat)
        
        # 4. Instant Full-Grid Prediction via dot product -> (125, 32768)
        preds = jnp.dot(b_out, t_out.T)
        
        # Relative MSE
        diff_sq = (batch - preds) ** 2
        snapshot_norms = jnp.mean(batch ** 2, axis=1, keepdims=True)
        return jnp.mean(diff_sq / (snapshot_norms + 1e-8))
        
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

print(f"   Training on {len(train_k_triples)} snapshots (Full grid evaluation)...")
for epoch in range(10001):
    params, opt_state, loss = train_step(params, opt_state, U_train)
    if epoch % 2000 == 0:
        print(f"   Epoch {epoch:5d}, Loss: {loss:.4e}")

# ==========================================
# 5. Empirical Quadrature 
# ==========================================
print("\n--- 3. Discovering Empirical Quadrature Points ---")

# Precompute full Trunk output for the offline EQ discovery phase (32768, 128)
T_FULL = jax.vmap(lambda c: model.apply({'params': params}, c, method=model.apply_trunk))(coords_flat)

def constrained_decode_full(lat):
    b_out = model.apply({'params': params}, lat, method=model.apply_branch)
    u_raw = jnp.dot(T_FULL, b_out)
    return mask * u_raw + u_g

@jax.jit
def get_integrand(lat_val, F_val):
    R_full = K_op_3d(constrained_decode_full(lat_val)) - F_val
    J_D = jax.jacfwd(constrained_decode_full)(lat_val)
    return J_D.T * R_full[None, :]

G_list = []
for i, (k1, k2, k3) in enumerate(train_k_triples):
    lat_val = model.apply({'params': params}, U_train[i], method=model.encode)
    G_list.append(get_integrand(lat_val, get_F_3d(k1, k2, k3)))

G_train_np = np.array(jnp.concatenate(G_list, axis=0))
G_train_np[:, np.array(mask) == 0] = 0.0
b_train_np = np.sum(G_train_np, axis=1)

# Use lsq_linear with bounds (0, inf) for non-negative least squares - much faster than nnls
result = lsq_linear(G_train_np, b_train_np, bounds=(0, np.inf), method='bvls', verbose=1)
w_eq = result.x
eq_indices = np.where(w_eq > 1e-10)[0]
eq_weights_jnp = jnp.array(w_eq[eq_indices])
eq_indices_jnp = jnp.array(eq_indices)
num_eq_points = len(eq_indices)
print(f"   Reduced from {num_nodes:,} to {num_eq_points} nodes ({100*num_eq_points/num_nodes:.3f}%)")

# ==========================================
# 6. Hyper-Reduced Solver (Precomputed Trunk Matrix)
# ==========================================
print("\n--- 4. Setting up Hyper-Reduced Solver ---")
N2 = N * N
stencil_offsets = jnp.array([0, -1, 1, -N, N, -N2, N2])

gather_indices_flat = (eq_indices_jnp[:, None] + stencil_offsets[None, :]).flatten()
coords_stencil_flat = coords_flat[gather_indices_flat]
mask_stencil = mask[gather_indices_flat].reshape((-1, 7))
u_g_stencil = u_g[gather_indices_flat].reshape((-1, 7))

# THE CHEAT CODE: Precompute the Trunk network for ONLY the 17,500 stencil points!
print("   Precomputing Trunk matrix for EQ stencils...")
T_EQ = jax.vmap(lambda c: model.apply({'params': params}, c, method=model.apply_trunk))(coords_stencil_flat)
print(f"   T_EQ matrix shape: {T_EQ.shape} (Instant Jacobians enabled)")

def make_latent_solver(p_branch, T_EQ_mat, mask_st, ug_st, eq_w):
    
    def _res_fn(lat, F_eq):
        # 1. Tiny Branch net evaluation
        b_out = model.apply({'params': p_branch}, lat, method=model.apply_branch)
        
        # 2. FAST LINEAR MATRIX MULTIPLY (Replaces the slow coordinate loop)
        u_raw_flat = jnp.dot(T_EQ_mat, b_out)
        
        # 3. Shape and FD
        u_stencil = mask_st * u_raw_flat.reshape((-1, 7)) + ug_st
        R = (6 * u_stencil[:, 0]
             - u_stencil[:, 1] - u_stencil[:, 2]
             - u_stencil[:, 3] - u_stencil[:, 4]
             - u_stencil[:, 5] - u_stencil[:, 6]) / dx**2 - F_eq
        return R

    @jax.jit
    def solve(lat_init, F_eq):
        def _body(carry):
            lat, _, itr = carry
            R = _res_fn(lat, F_eq)
            # AD only differentiates the Branch net + Linear multiply!
            J = jax.jacfwd(lambda l: _res_fn(l, F_eq))(lat) 
            
            WJ = eq_w[:, None] * J
            JtWJ = J.T @ WJ
            JtWr = J.T @ (eq_w * R)
            
            lam = jnp.maximum(1e-3 * jnp.trace(JtWJ) / k_dim, 1e-8)
            dz = jnp.linalg.solve(JtWJ + lam * jnp.eye(k_dim), -JtWr)
            
            f0 = jnp.dot(eq_w * R, R)
            def _wsn(alpha):
                R_trial = _res_fn(lat + alpha * dz, F_eq)
                return jnp.dot(eq_w * R_trial, R_trial)
            
            f1, f2, f3, f4 = _wsn(1.0), _wsn(0.5), _wsn(0.25), _wsn(0.125)
            step = jnp.where(f1 < f0, 1.0, jnp.where(f2 < f0, 0.5, jnp.where(f3 < f0, 0.25, 0.125)))
            
            return lat + step * dz, jnp.linalg.norm(JtWr), itr + 1
        
        def _cond(carry):
            _, grad_norm, itr = carry
            return jnp.logical_and(grad_norm > 1e-8, itr < 30)
        
        init_carry = (lat_init, jnp.array(jnp.inf, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32))
        return jax.lax.while_loop(_cond, _body, init_carry)
    
    return solve

latent_solve = make_latent_solver(params, T_EQ, mask_stencil, u_g_stencil, eq_weights_jnp)

def fast_eq_latent_poisson_solver(lat_init, F_vec):
    F_eq = F_vec[eq_indices_jnp]
    lat_f, res_f, n_iters = latent_solve(lat_init, F_eq)
    u_final = constrained_decode_full(lat_f)
    return lat_f, u_final, res_f, n_iters

# ==========================================
# 7. Benchmark
# ==========================================
print("\n--- 5. Running Benchmark ---")
test_k_triples = [
    (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5),
    (1, 2, 3), (2, 3, 4), (3, 4, 5),
    (1, 1, 2), (2, 2, 3), (3, 3, 4),
    (1, 2, 2), (2, 3, 3), (3, 4, 4),
]

fom_times, rom_times, rom_vs_fom_errors, fom_vs_exact_errors, rom_vs_exact_errors = [], [], [], [], []
stored = {}

print("   Warming up JAX compilers...")
_F_wm = get_F_3d(2, 2, 2)
full_order_fem_solver_3d(K_op_3d, _F_wm, u_guess).block_until_ready()
_lat_wm = model.apply({'params': params}, U_train[0], method=model.encode)
_, _u_wm, _, _ = fast_eq_latent_poisson_solver(_lat_wm, _F_wm)
_, _u_wm, _, _ = fast_eq_latent_poisson_solver(_lat_wm, _F_wm)
jax.block_until_ready(_u_wm)
print("   Warm-up complete.\n")

n_test = len(test_k_triples)
for i, (k1, k2, k3) in enumerate(test_k_triples):
    F_test = get_F_3d(k1, k2, k3)
    
    t0 = time.perf_counter()
    u_fom = full_order_fem_solver_3d(K_op_3d, F_test, u_guess).block_until_ready()
    fom_t = time.perf_counter() - t0
    fom_times.append(fom_t)
    
    def freq_dist(kt): return (kt[0] - k1)**2 + (kt[1] - k2)**2 + (kt[2] - k3)**2
    dists = [freq_dist(kt) for kt in train_k_triples]
    sorted_idx = np.argsort(dists)
    i1, i2 = sorted_idx[0], sorted_idx[1]
    d1, d2 = np.sqrt(dists[i1]), np.sqrt(dists[i2])
    w1 = d2 / (d1 + d2) if (d1 + d2) > 1e-12 else 0.5
    w2 = d1 / (d1 + d2) if (d1 + d2) > 1e-12 else 0.5
    
    lat1 = model.apply({'params': params}, U_train[i1], method=model.encode)
    lat2 = model.apply({'params': params}, U_train[i2], method=model.encode)
    lat_init = w1 * lat1 + w2 * lat2
    
    t0 = time.perf_counter()
    lat_f, u_rom, gn_res, n_iters = fast_eq_latent_poisson_solver(lat_init, F_test)
    jax.block_until_ready(u_rom)
    rom_t = time.perf_counter() - t0
    rom_times.append(rom_t)
    
    u_exact = get_analytical_solution_3d(k1, k2, k3)
    norm_fom = float(jnp.linalg.norm(u_fom))
    
    err_rom_fom = float(jnp.linalg.norm(u_rom - u_fom) / norm_fom)
    err_fom_ex = float(jnp.linalg.norm(u_fom - u_exact) / jnp.linalg.norm(u_exact))
    err_rom_ex = float(jnp.linalg.norm(u_rom - u_exact) / jnp.linalg.norm(u_exact))
    
    rom_vs_fom_errors.append(err_rom_fom)
    fom_vs_exact_errors.append(err_fom_ex)
    rom_vs_exact_errors.append(err_rom_ex)
    
    print(f"  [{i+1:2d}/{n_test}] k=({k1},{k2},{k3}) | FOM {fom_t:.4f}s | ROM {rom_t:.4f}s | ROM-vs-FOM {err_rom_fom:.3e} | GN_res {float(gn_res):.2e} itr {int(n_iters)}")
    
    if i in {0, n_test // 2, n_test - 1}:
        stored[i] = dict(u_fom=np.asarray(u_fom), u_rom=np.asarray(u_rom), u_exact=np.asarray(u_exact), k=(k1, k2, k3))

# ==========================================
# 8. Summary & Plots 
# ==========================================
avg_fom_t, avg_rom_t = float(np.mean(fom_times)), float(np.mean(rom_times))
avg_speedup = avg_fom_t / avg_rom_t
print(f"\n{'='*60}\n           3D POISSON EQ-ROM BENCHMARK\n{'='*60}")
print(f"  Avg FOM time:            {avg_fom_t:.5f} s\n  Avg ROM time:            {avg_rom_t:.5f} s\n  Avg speedup:             {avg_speedup:.2f}×")
print(f"  PRIMARY — ROM vs FOM:    {float(np.mean(rom_vs_fom_errors)):.4e}\n{'='*60}\n")

test_labels = [f"({k[0]},{k[1]},{k[2]})" for k in test_k_triples]

fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(test_k_triples))
ax.bar(x_pos - 0.2, rom_vs_fom_errors, width=0.4, color='#1f77b4', label='ROM vs FOM (primary)')
ax.bar(x_pos + 0.2, fom_vs_exact_errors, width=0.4, color='#2ca02c', alpha=0.7, label='FOM vs Analytical')
ax.set_title('3D Poisson — Relative $L_2$ Error', fontsize=13)
ax.set_xlabel('$(k_1, k_2, k_3)$', fontsize=11)
ax.set_ylabel('Relative $L_2$ Error', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_error.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x_pos - 0.2, fom_times, width=0.4, color='#d62728', label='FOM (CG)')
ax.bar(x_pos + 0.2, rom_times, width=0.4, color='#1f77b4', label='EQ-ROM')
ax.axhline(avg_fom_t, color='#d62728', ls=':', alpha=0.7, label=f'FOM avg ({avg_fom_t:.4f}s)')
ax.axhline(avg_rom_t, color='#1f77b4', ls=':', alpha=0.7, label=f'ROM avg ({avg_rom_t:.4f}s)')
ax.set_title(f'3D Poisson — Solve Time (avg speedup {avg_speedup:.1f}×)', fontsize=13)
ax.set_xlabel('$(k_1, k_2, k_3)$', fontsize=11)
ax.set_ylabel('Time (s)', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_time.png', dpi=150)
plt.close()

speedups = [f / r for f, r in zip(fom_times, rom_times)]
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(x_pos, speedups, width=0.6, color='#9467bd', alpha=0.85, edgecolor='black')
ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Parity (1×)')
ax.axhline(avg_speedup, color='orange', ls='--', lw=1.5, label=f'Average = {avg_speedup:.1f}×')
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{sp:.1f}×', ha='center', va='bottom', fontsize=9)
ax.set_title('3D Poisson — Per-test Speedup', fontsize=13)
ax.set_xlabel('$(k_1, k_2, k_3)$', fontsize=11)
ax.set_ylabel('Speedup Factor', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(test_labels, rotation=45, ha='right')
ax.set_ylim(0, max(speedups) * 1.2)
ax.legend()
ax.grid(True, axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_speedup.png', dpi=150)
plt.close()

print("\n--- 6. Saving Midplane Slice Plots ---")
for idx in sorted(stored):
    s = stored[idx]
    k1, k2, k3 = s['k']
    mid = N // 2
    
    u_fom_3d = s['u_fom'].reshape(N, N, N)
    u_rom_3d = s['u_rom'].reshape(N, N, N)
    u_ex_3d = s['u_exact'].reshape(N, N, N)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    sl_fom, sl_rom, sl_ex = u_fom_3d[:,:,mid], u_rom_3d[:,:,mid], u_ex_3d[:,:,mid]
    vmin, vmax = min(sl_fom.min(), sl_rom.min(), sl_ex.min()), max(sl_fom.max(), sl_rom.max(), sl_ex.max())
    kw = dict(origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0,L,0,L])
    
    for ax, data, title in zip(axes[0,:3], [sl_fom, sl_rom, sl_ex], ['FOM', 'EQ-ROM', 'Analytical']):
        im = ax.imshow(data.T, **kw)
        ax.set_title(f'{title} (z={mid*dx:.2f})', fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    err = np.abs(sl_rom - sl_fom)
    im_err = axes[0,3].imshow(err.T, origin='lower', aspect='auto', cmap='hot', extent=[0,L,0,L])
    axes[0,3].set_title('|ROM - FOM|', fontsize=11)
    axes[0,3].set_xlabel('x'); axes[0,3].set_ylabel('y')
    plt.colorbar(im_err, ax=axes[0,3], shrink=0.8)
    
    sl_fom, sl_rom, sl_ex = u_fom_3d[:,mid,:], u_rom_3d[:,mid,:], u_ex_3d[:,mid,:]
    vmin, vmax = min(sl_fom.min(), sl_rom.min(), sl_ex.min()), max(sl_fom.max(), sl_rom.max(), sl_ex.max())
    kw = dict(origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0,L,0,L])
    
    for ax, data, title in zip(axes[1,:3], [sl_fom, sl_rom, sl_ex], ['FOM', 'EQ-ROM', 'Analytical']):
        im = ax.imshow(data.T, **kw)
        ax.set_title(f'{title} (y={mid*dx:.2f})', fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('z')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    err = np.abs(sl_rom - sl_fom)
    im_err = axes[1,3].imshow(err.T, origin='lower', aspect='auto', cmap='hot', extent=[0,L,0,L])
    axes[1,3].set_title('|ROM - FOM|', fontsize=11)
    axes[1,3].set_xlabel('x'); axes[1,3].set_ylabel('z')
    plt.colorbar(im_err, ax=axes[1,3], shrink=0.8)
    
    fig.suptitle(f'3D Poisson Midplane Slices: k=({k1},{k2},{k3})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fpath = OUTPUT_DIR / f'slices_k{k1}{k2}{k3}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()

print("\n=== 3D Poisson EQ-ROM Complete ===")