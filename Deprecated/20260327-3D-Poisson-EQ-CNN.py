import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.optimize import nnls
from scipy.stats import qmc  # Latin Hypercube Sampling
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
N = 32  # Grid size (32^3 = 32,768 nodes) - can increase to 64 for production
num_nodes = N ** 3
k_dim = 20  # Latent dimension
L = 1.0
dx = L / (N - 1)

# Spatial coordinates (using x_sp, y_sp, z_sp to avoid shadowing latent 'z')
x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')  # Critical: 'ij' indexing

def K_op_3d(u_flat):
    """7-point finite-difference 3D Laplacian with Dirichlet BCs."""
    u_3d = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u_3d)
    
    # Interior: standard 7-point stencil
    out = out.at[1:-1, 1:-1, 1:-1].set(
        (6 * u_3d[1:-1, 1:-1, 1:-1]
         - u_3d[0:-2, 1:-1, 1:-1] - u_3d[2:, 1:-1, 1:-1]
         - u_3d[1:-1, 0:-2, 1:-1] - u_3d[1:-1, 2:, 1:-1]
         - u_3d[1:-1, 1:-1, 0:-2] - u_3d[1:-1, 1:-1, 2:]) / dx**2
    )
    
    # Boundary: identity rows (Dirichlet u=0)
    out = out.at[0, :, :].set(u_3d[0, :, :])
    out = out.at[-1, :, :].set(u_3d[-1, :, :])
    out = out.at[:, 0, :].set(u_3d[:, 0, :])
    out = out.at[:, -1, :].set(u_3d[:, -1, :])
    out = out.at[:, :, 0].set(u_3d[:, :, 0])
    out = out.at[:, :, -1].set(u_3d[:, :, -1])
    
    return out.flatten()

def get_F_3d(k1, k2, k3):
    """3D forcing: F = 10 * sin(k1*pi*x) * sin(k2*pi*y) * sin(k3*pi*z)
    Integer k values ensure sin(k*pi*0) = sin(k*pi*1) = 0 (exact BCs).
    """
    F_3d = (jnp.sin(k1 * jnp.pi * X) 
            * jnp.sin(k2 * jnp.pi * Y) 
            * jnp.sin(k3 * jnp.pi * Z) * 10.0)
    # Explicitly zero boundaries
    F_3d = F_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    F_3d = F_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    F_3d = F_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    return F_3d.flatten()

def get_analytical_solution_3d(k1, k2, k3):
    """Exact solution: u = 10 / ((k1^2 + k2^2 + k3^2) * pi^2) * sin(...)"""
    coeff = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (coeff 
            * jnp.sin(k1 * jnp.pi * X) 
            * jnp.sin(k2 * jnp.pi * Y) 
            * jnp.sin(k3 * jnp.pi * Z)).flatten()

# Boundary mask (0 on all 6 faces, 1 in interior)
mask_3d = jnp.ones((N, N, N))
mask_3d = mask_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
mask_3d = mask_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
mask_3d = mask_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
mask = mask_3d.flatten()
u_g = jnp.zeros(num_nodes)

print(f"3D Domain: {N}x{N}x{N} = {num_nodes:,} nodes")
print(f"Interior nodes: {int(mask.sum()):,}")
print(f"Boundary nodes: {num_nodes - int(mask.sum()):,}")

# ==========================================
# 2. 3D Convolutional Autoencoder (replaces Dense MLP to reduce parameters)
# ==========================================
class ConvAutoencoder3D(nn.Module):
    latent_dim: int
    N: int = 32

    def setup(self):
        # Encoder: Compresses spatial dimensions by half at each step
        self.enc_conv1 = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.enc_conv2 = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.enc_conv3 = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.enc_dense = nn.Dense(self.latent_dim)

        # Decoder: Restores spatial dimensions
        self.dec_dense = nn.Dense(4 * 4 * 4 * 64)
        self.dec_conv1 = nn.ConvTranspose(features=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.dec_conv2 = nn.ConvTranspose(features=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')
        self.dec_conv3 = nn.ConvTranspose(features=1, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        # Reshape flat array to 3D grid with 1 channel: (32, 32, 32, 1)
        x = x.reshape((self.N, self.N, self.N, 1))
        x = nn.swish(self.enc_conv1(x))  # -> (16, 16, 16, 16)
        x = nn.swish(self.enc_conv2(x))  # -> (8, 8, 8, 32)
        x = nn.swish(self.enc_conv3(x))  # -> (4, 4, 4, 64)
        x = x.reshape((-1,))             # Flatten for latent projection
        return self.enc_dense(x)

    def decode(self, lat):
        x = nn.swish(self.dec_dense(lat))
        x = x.reshape((4, 4, 4, 64))     # Reshape to start 3D upsampling
        x = nn.swish(self.dec_conv1(x))  # -> (8, 8, 8, 32)
        x = nn.swish(self.dec_conv2(x))  # -> (16, 16, 16, 16)
        x = self.dec_conv3(x)            # -> (32, 32, 32, 1)
        return x.flatten()               # Return flat array to match physics code

def full_order_fem_solver_3d(K_operator, F_vec, u_guess, tol=1e-6):
    """CG solve for the FOM."""
    u_true, _ = jax_linalg.cg(K_operator, F_vec, x0=u_guess, tol=tol, maxiter=2000)
    return u_true

def create_constrained_decode_fn(params, model, mask_vec, u_g_vec):
    """ũ(lat) = mask ⊙ D(lat) + u_g — Dirichlet BCs via masking."""
    def constrained_decode(lat):
        return mask_vec * model.apply({'params': params}, lat, method=model.decode) + u_g_vec
    return constrained_decode

# ==========================================
# 3. Latin Hypercube Sampling for 3D Training Data
# ==========================================
print("\n--- 1. Generating 3D Training Dataset with Latin Hypercube Sampling ---")

# Generate comprehensive training set: all combinations of k1, k2, k3 in [1, 5]
# This gives 5^3 = 125 combinations - complete coverage of the parameter space
train_k_triples = [(k1, k2, k3) 
                   for k1 in range(1, 6) 
                   for k2 in range(1, 6) 
                   for k3 in range(1, 6)]
print(f"   Generated {len(train_k_triples)} unique (k1, k2, k3) combinations via LHS")

u_guess = jnp.zeros(num_nodes)

U_train_list = []
print("   Solving FOM for each snapshot...")
for i, (k1, k2, k3) in enumerate(train_k_triples):
    F_vec = get_F_3d(k1, k2, k3)
    u_sol = full_order_fem_solver_3d(K_op_3d, F_vec, u_guess)
    U_train_list.append(u_sol)
    if (i + 1) % 10 == 0:
        print(f"      Snapshot {i+1}/{len(train_k_triples)} complete")

U_train = jnp.stack(U_train_list)
print(f"   Training data shape: {U_train.shape}")

# ==========================================
# 4. Train 3D Convolutional Autoencoder
# ==========================================
model = ConvAutoencoder3D(latent_dim=k_dim, N=N)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones(num_nodes))['params']

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

print(f"\n--- 2. Training 3D Convolutional Autoencoder ---")
print(f"   Model parameters: {count_params(params):,}")

schedule = optax.exponential_decay(init_value=1e-3, transition_steps=2000, decay_rate=0.9)
tx = optax.adam(learning_rate=schedule)
opt_state = tx.init(params)

@jax.jit
def train_step(p, opt_st, batch):
    def loss_fn(weights):
        preds = jax.vmap(lambda inp: model.apply({'params': weights}, inp))(batch)
        
        # Relative MSE: Normalizes error by the magnitude of each snapshot
        # This forces the network to care about high-frequency low-amplitude waves
        diff_sq = (batch - preds) ** 2
        snapshot_norms = jnp.mean(batch ** 2, axis=1, keepdims=True)
        return jnp.mean(diff_sq / (snapshot_norms + 1e-8))
        
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

print(f"   Training on {len(train_k_triples)} snapshots...")
for epoch in range(10000):
    params, opt_state, loss = train_step(params, opt_state, U_train)
    if epoch % 2000 == 0:
        print(f"   Epoch {epoch:5d}, Loss: {loss:.4e}")

# ==========================================
# 5. Empirical Quadrature — Offline Phase
# ==========================================
print("\n--- 3. Discovering Empirical Quadrature Points ---")
constrained_decode = create_constrained_decode_fn(params, model, mask, u_g)

@jax.jit
def get_integrand(lat_val, F_val):
    """k×N integrand matrix: G[j,i] = (∂ũ/∂lat_j)(i) · R(i)"""
    R_full = K_op_3d(constrained_decode(lat_val)) - F_val
    J_D = jax.jacfwd(constrained_decode)(lat_val)
    return J_D.T * R_full[None, :]

G_list = []
print("   Computing integrand matrices...")
for i, (k1, k2, k3) in enumerate(train_k_triples):
    lat_val = model.apply({'params': params}, U_train[i], method=model.encode)
    G_list.append(get_integrand(lat_val, get_F_3d(k1, k2, k3)))
    if (i + 1) % 10 == 0:
        print(f"      Processed {i+1}/{len(train_k_triples)}")

G_train = jnp.concatenate(G_list, axis=0)
G_train_np = np.array(G_train)
G_train_np[:, np.array(mask) == 0] = 0.0
b_train_np = np.sum(G_train_np, axis=1)

print("   Solving NNLS for quadrature weights...")
w_eq, _ = nnls(G_train_np, b_train_np)
eq_indices = np.where(w_eq > 1e-10)[0]
eq_weights_np = w_eq[eq_indices]

eq_indices_jnp = jnp.array(eq_indices)
eq_weights_jnp = jnp.array(eq_weights_np)
num_eq_points = len(eq_indices)

print(f"   Reduced from {num_nodes:,} to {num_eq_points} nodes ({100*num_eq_points/num_nodes:.3f}%)")

# ==========================================
# 6. Hyper-Reduced Solver (CAE Adapted - Full Decoder Evaluation)
# ==========================================
print("\n--- 4. Setting up Hyper-Reduced Solver ---")

N2 = N * N
stencil_offsets = jnp.array([0, -1, 1, -N, N, -N2, N2])
# Find the exact indices needed to compute the Laplacian at the EQ points
gather_indices_flat = (eq_indices_jnp[:, None] + stencil_offsets[None, :]).flatten()

print(f"   EQ points: {num_eq_points}")
print(f"   Total stencil nodes evaluated: {len(gather_indices_flat)}")

def make_latent_solver(p, gather_indices, full_mask, full_ug, eq_w):
    """Factory returning JIT-compiled LM-GN solver with Armijo backtracking."""
    
    def _res_fn(lat, F_eq):
        # 1. Decode the full 3D state (Fast because CAE has few parameters)
        u_full = model.apply({'params': p}, lat, method=model.decode)
        
        # 2. Enforce Boundary Conditions on the full state
        u_full_bc = full_mask * u_full + full_ug
        
        # 3. Gather ONLY the nodes needed for the EQ stencils
        u_stencil = u_full_bc[gather_indices].reshape((-1, 7))
        
        # 4. Compute 7-point finite difference residual
        R = (6 * u_stencil[:, 0]
             - u_stencil[:, 1] - u_stencil[:, 2]
             - u_stencil[:, 3] - u_stencil[:, 4]
             - u_stencil[:, 5] - u_stencil[:, 6]) / dx**2 - F_eq
        return R

    @jax.jit
    def solve(lat_init, F_eq):
        """LM-GN with Armijo backtracking via lax.while_loop."""
        
        def _body(carry):
            lat, _, itr = carry
            
            R = _res_fn(lat, F_eq)
            J = jax.jacfwd(lambda l: _res_fn(l, F_eq))(lat)
            
            WJ = eq_w[:, None] * J
            JtWJ = J.T @ WJ
            JtWr = J.T @ (eq_w * R)
            
            # Adaptive LM damping
            lam = jnp.maximum(1e-3 * jnp.trace(JtWJ) / k_dim, 1e-8)
            dz = jnp.linalg.solve(JtWJ + lam * jnp.eye(k_dim), -JtWr)
            
            # Armijo backtracking
            f0 = jnp.dot(eq_w * R, R)
            
            def _wsn(alpha):
                R_trial = _res_fn(lat + alpha * dz, F_eq)
                return jnp.dot(eq_w * R_trial, R_trial)
            
            f1, f2, f3, f4 = _wsn(1.0), _wsn(0.5), _wsn(0.25), _wsn(0.125)
            
            step = jnp.where(f1 < f0, 1.0,
                   jnp.where(f2 < f0, 0.5,
                   jnp.where(f3 < f0, 0.25, 0.125)))
            
            new_norm = jnp.linalg.norm(JtWr)
            return lat + step * dz, new_norm, itr + 1
        
        def _cond(carry):
            _, grad_norm, itr = carry
            return jnp.logical_and(grad_norm > 1e-8, itr < 30)
        
        init_carry = (lat_init, jnp.array(jnp.inf, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32))
        lat_f, res_f, n_iters = jax.lax.while_loop(_cond, _body, init_carry)
        return lat_f, res_f, n_iters
    
    return solve

# Initialize the solver with the full mask and gather indices
latent_solve = make_latent_solver(params, gather_indices_flat, mask, u_g, eq_weights_jnp)

def fast_eq_latent_poisson_solver(lat_init, F_vec):
    """Public interface: hyper-reduced solve → full field."""
    F_eq = F_vec[eq_indices_jnp]
    lat_f, res_f, n_iters = latent_solve(lat_init, F_eq)
    u_final = mask * model.apply({'params': params}, lat_f, method=model.decode) + u_g
    return lat_f, u_final, res_f, n_iters

# ==========================================
# 7. Benchmark
# ==========================================
print("\n--- 5. Running Benchmark ---")

# Test cases that are IN the training set (should have low error)
# Plus some interpolation cases
test_k_triples = [
    (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5),
    (1, 2, 3), (2, 3, 4), (3, 4, 5),
    (1, 1, 2), (2, 2, 3), (3, 3, 4),
    (1, 2, 2), (2, 3, 3), (3, 4, 4),
]

fom_times, rom_times = [], []
rom_vs_fom_errors = []
fom_vs_exact_errors = []
rom_vs_exact_errors = []
stored = {}

# Warm-up
print("   Warming up JAX compilers...")
_F_wm = get_F_3d(2, 2, 2)
full_order_fem_solver_3d(K_op_3d, _F_wm, u_guess).block_until_ready()
_lat_wm = model.apply({'params': params}, U_train[0], method=model.encode)
_, _u_wm, _, _ = fast_eq_latent_poisson_solver(_lat_wm, _F_wm)
_, _u_wm, _, _ = fast_eq_latent_poisson_solver(_lat_wm, _F_wm)
jax.block_until_ready(_u_wm)
print("   Warm-up complete.\n")

n_test = len(test_k_triples)
plot_at = {0, n_test // 2, n_test - 1}

for i, (k1, k2, k3) in enumerate(test_k_triples):
    F_test = get_F_3d(k1, k2, k3)
    
    # FOM
    t0 = time.perf_counter()
    u_fom = full_order_fem_solver_3d(K_op_3d, F_test, u_guess).block_until_ready()
    fom_t = time.perf_counter() - t0
    fom_times.append(fom_t)
    
    # Latent init: inverse-distance weighted interpolation of two closest snapshots
    def freq_dist(kt):
        return (kt[0] - k1)**2 + (kt[1] - k2)**2 + (kt[2] - k3)**2
    
    dists = [freq_dist(kt) for kt in train_k_triples]
    sorted_idx = np.argsort(dists)
    i1, i2 = sorted_idx[0], sorted_idx[1]
    d1, d2 = np.sqrt(dists[i1]), np.sqrt(dists[i2])
    dsum = d1 + d2
    w1 = d2 / dsum if dsum > 1e-12 else 0.5  # Inverse distance weighting
    w2 = d1 / dsum if dsum > 1e-12 else 0.5
    
    lat1 = model.apply({'params': params}, U_train[i1], method=model.encode)
    lat2 = model.apply({'params': params}, U_train[i2], method=model.encode)
    lat_init = w1 * lat1 + w2 * lat2
    
    # ROM
    t0 = time.perf_counter()
    lat_f, u_rom, gn_res, n_iters = fast_eq_latent_poisson_solver(lat_init, F_test)
    jax.block_until_ready(u_rom)
    rom_t = time.perf_counter() - t0
    rom_times.append(rom_t)
    
    # Errors
    u_exact = get_analytical_solution_3d(k1, k2, k3)
    norm_fom = float(jnp.linalg.norm(u_fom))
    
    err_rom_fom = float(jnp.linalg.norm(u_rom - u_fom) / norm_fom)
    err_fom_ex = float(jnp.linalg.norm(u_fom - u_exact) / jnp.linalg.norm(u_exact))
    err_rom_ex = float(jnp.linalg.norm(u_rom - u_exact) / jnp.linalg.norm(u_exact))
    
    rom_vs_fom_errors.append(err_rom_fom)
    fom_vs_exact_errors.append(err_fom_ex)
    rom_vs_exact_errors.append(err_rom_ex)
    
    print(f"  [{i+1:2d}/{n_test}] k=({k1},{k2},{k3}) | "
          f"FOM {fom_t:.4f}s | ROM {rom_t:.4f}s | "
          f"ROM-vs-FOM {err_rom_fom:.3e} | "
          f"GN_res {float(gn_res):.2e} itr {int(n_iters)}")
    
    if i in plot_at:
        stored[i] = dict(u_fom=np.asarray(u_fom), u_rom=np.asarray(u_rom),
                         u_exact=np.asarray(u_exact), k=(k1, k2, k3))

# ==========================================
# 8. Summary
# ==========================================
avg_fom_t = float(np.mean(fom_times))
avg_rom_t = float(np.mean(rom_times))
avg_speedup = avg_fom_t / avg_rom_t
avg_rom_fom = float(np.mean(rom_vs_fom_errors))
avg_fom_exact = float(np.mean(fom_vs_exact_errors))
avg_rom_exact = float(np.mean(rom_vs_exact_errors))

print(f"\n{'='*60}")
print(f"           3D POISSON EQ-ROM BENCHMARK")
print(f"{'='*60}")
print(f"  Grid:                    {N}³ = {num_nodes:,} DOF")
print(f"  Latent dimension:        {k_dim}")
print(f"  EQ nodes:                {num_eq_points} / {num_nodes} ({100*num_eq_points/num_nodes:.3f}%)")
print(f"{'--'*30}")
print(f"  Avg FOM time:            {avg_fom_t:.5f} s")
print(f"  Avg ROM time:            {avg_rom_t:.5f} s")
print(f"  Avg speedup:             {avg_speedup:.2f}×")
print(f"{'--'*30}")
print(f"  PRIMARY — ROM vs FOM:    {avg_rom_fom:.4e}")
print(f"  FOM vs analytical:       {avg_fom_exact:.4e}")
print(f"  ROM vs analytical:       {avg_rom_exact:.4e}")
print(f"{'='*60}\n")

# ==========================================
# 9. Plots
# ==========================================
test_labels = [f"({k[0]},{k[1]},{k[2]})" for k in test_k_triples]

# Error comparison
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
print(f"Saved: {OUTPUT_DIR / 'benchmark_error.png'}")

# Time comparison
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
print(f"Saved: {OUTPUT_DIR / 'benchmark_time.png'}")

# Speedup bar chart
speedups = [f / r for f, r in zip(fom_times, rom_times)]
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(x_pos, speedups, width=0.6, color='#9467bd', alpha=0.85, edgecolor='black')
ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Parity (1×)')
ax.axhline(avg_speedup, color='orange', ls='--', lw=1.5, label=f'Average = {avg_speedup:.1f}×')
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{sp:.1f}×', 
            ha='center', va='bottom', fontsize=9)
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
print(f"Saved: {OUTPUT_DIR / 'benchmark_speedup.png'}")

# Midplane slices
print("\n--- 6. Saving Midplane Slice Plots ---")
for idx in sorted(stored):
    s = stored[idx]
    k1, k2, k3 = s['k']
    mid = N // 2
    
    u_fom_3d = s['u_fom'].reshape(N, N, N)
    u_rom_3d = s['u_rom'].reshape(N, N, N)
    u_ex_3d = s['u_exact'].reshape(N, N, N)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # XY slice at z=mid
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
    
    # XZ slice at y=mid
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
    print(f"   Saved: {fpath}")

print("\n=== 3D Poisson EQ-ROM Complete ===")
