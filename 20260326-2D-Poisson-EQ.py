import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ==========================================
# 0. Output Directory
# ==========================================
OUTPUT_DIR = Path(__file__).parent / 'plots' / '2D_poisson'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Domain Setup & Parametric Physics
# ==========================================
N = 64
num_nodes = N * N
k_dim = 12  # INCREASED: Gives the latent manifold more flexibility
L = 1.0
dx = L / (N - 1)

x = jnp.linspace(0, L, N)
y = jnp.linspace(0, L, N)
X, Y = jnp.meshgrid(x, y)

def K_op(u_flat):
    u_2d = u_flat.reshape((N, N))
    out_2d = jnp.zeros_like(u_2d)
    out_2d = out_2d.at[1:-1, 1:-1].set(
        (4 * u_2d[1:-1, 1:-1] - u_2d[0:-2, 1:-1] - u_2d[2:, 1:-1]
         - u_2d[1:-1, 0:-2] - u_2d[1:-1, 2:]) / dx**2
    )
    out_2d = out_2d.at[0, :].set(u_2d[0, :])
    out_2d = out_2d.at[-1, :].set(u_2d[-1, :])
    out_2d = out_2d.at[:, 0].set(u_2d[:, 0])
    out_2d = out_2d.at[:, -1].set(u_2d[:, -1])
    return out_2d.flatten()

def get_F(k1, k2):
    # With integer k1, k2: sin(k*pi*x) = 0 at x=0 and x=1 (exact BCs)
    F_2d = jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y) * 10.0
    return F_2d.flatten()

mask_2d = jnp.ones((N, N))
mask_2d = mask_2d.at[0, :].set(0.0)
mask_2d = mask_2d.at[-1, :].set(0.0)
mask_2d = mask_2d.at[:, 0].set(0.0)
mask_2d = mask_2d.at[:, -1].set(0.0)
mask = mask_2d.flatten()
u_g = jnp.zeros(num_nodes)

# ==========================================
# 2. Neural Autoencoder & FOM Solver
# ==========================================
class SimpleAutoencoder(nn.Module):
    latent_dim: int
    num_nodes: int

    def setup(self):
        # INCREASED CAPACITY: Wider dense layers for complex high-freq features
        self.enc_dense1 = nn.Dense(512)
        self.enc_dense2 = nn.Dense(256)
        self.enc_dense3 = nn.Dense(self.latent_dim)
        
        self.dec_dense1 = nn.Dense(256)
        self.dec_dense2 = nn.Dense(512)
        self.dec_dense3 = nn.Dense(self.num_nodes)

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = nn.swish(self.enc_dense1(x))
        x = nn.swish(self.enc_dense2(x))
        return self.enc_dense3(x)

    def decode(self, z):
        x = nn.swish(self.dec_dense1(z))
        x = nn.swish(self.dec_dense2(x))
        return self.dec_dense3(x)

def full_order_fem_solver(K_operator, F_vec, u_guess, tol=1e-6):
    u_true, _ = jax_linalg.cg(K_operator, F_vec, x0=u_guess, tol=tol)
    return u_true

def create_constrained_decode_fn(params, model, mask_vec, u_g_vec):
    def constrained_decode(z):
        u_hat = model.apply({'params': params}, z, method=model.decode)
        return mask_vec * u_hat + u_g_vec
    return constrained_decode

# ==========================================
# 3. Data Generation & Training
# ==========================================
print("\n--- 1. Generating Dense Training Dataset (FOM) ---")
# Use integer (k1, k2) pairs for exact boundary conditions
train_k_pairs = [(k1, k2) for k1 in range(1, 6) for k2 in range(1, 6)]  # 25 combinations
u_guess = jnp.zeros(num_nodes)

U_train_list = []
for k1, k2 in train_k_pairs:
    U_train_list.append(full_order_fem_solver(K_op, get_F(k1, k2), u_guess))
U_train = jnp.stack(U_train_list)
print(f"   Generated {len(train_k_pairs)} training snapshots with k1,k2 in [1,5]")

model = SimpleAutoencoder(latent_dim=k_dim, num_nodes=num_nodes)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones(num_nodes))['params']

schedule = optax.exponential_decay(init_value=1e-3, transition_steps=2000, decay_rate=0.9)
tx = optax.adam(learning_rate=schedule)
opt_state = tx.init(params)

@jax.jit
def train_step(p, opt_st, batch):
    def loss_fn(weights):
        preds = jax.vmap(lambda x: model.apply({'params': weights}, x))(batch)
        return jnp.mean((batch - preds)**2)
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, opt_st = tx.update(grads, opt_st, p)
    p = optax.apply_updates(p, updates)
    return p, opt_st, loss

print(f"\n--- 2. Training continuous manifold on {len(train_k_pairs)} snapshots ---")
for epoch in range(6000):
    params, opt_state, loss = train_step(params, opt_state, U_train)
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}, Loss: {loss:.4e}")

# ==========================================
# 4. Empirical Quadrature (Offline Phase)
# ==========================================
print("\n--- 3. Discovering Empirical Quadrature Points ---")
constrained_decode = create_constrained_decode_fn(params, model, mask, u_g)

@jax.jit
def get_integrand(z_val, F_val):
    u_pred, vjp_fn = jax.vjp(constrained_decode, z_val)
    R_full = K_op(u_pred) - F_val
    J_D = jax.jacfwd(constrained_decode)(z_val) 
    G_snap = J_D.T * R_full[None, :]          
    return G_snap

G_list = []
for i, (k1, k2) in enumerate(train_k_pairs):
    F_val = get_F(k1, k2)
    z_val = model.apply({'params': params}, U_train[i], method=model.encode)
    G_list.append(get_integrand(z_val, F_val))

G_train = jnp.concatenate(G_list, axis=0) 
G_train_np = np.array(G_train)

G_train_np[:, mask == 0] = 0.0 
b_train_np = np.sum(G_train_np, axis=1) 

print("   Solving NNLS for EQ weights...")
w_eq, _ = nnls(G_train_np, b_train_np)
eq_indices = np.where(w_eq > 1e-10)[0]
eq_weights = w_eq[eq_indices]

print(f"   Success! Reduced physics domain from {num_nodes} to just {len(eq_indices)} nodes.")

eq_indices_jnp = jnp.array(eq_indices)
eq_weights_jnp = jnp.array(eq_weights)

# ==========================================
# 5. Hyper-Reduced Latent Solver (Online Phase)
# ==========================================
W_final = params['dec_dense3']['kernel']  
b_final = params['dec_dense3']['bias']    

stencil_offsets = jnp.array([0, -1, 1, -N, N])
gather_indices = eq_indices_jnp[:, None] + stencil_offsets[None, :] 
gather_indices_flat = gather_indices.flatten()

W_sparse = W_final[:, gather_indices_flat] 
b_sparse = b_final[gather_indices_flat]
mask_sparse = mask[gather_indices_flat]
u_g_sparse = u_g[gather_indices_flat]

num_eq_points = len(eq_indices)

def create_fast_eq_residual_fn(params, F_eq, W_sparse, b_sparse, mask_sparse, u_g_sparse):
    @jax.jit
    def res_fn(z):
        x = nn.swish(jnp.dot(z, params['dec_dense1']['kernel']) + params['dec_dense1']['bias'])
        x = nn.swish(jnp.dot(x, params['dec_dense2']['kernel']) + params['dec_dense2']['bias'])
        
        u_hat_sparse = jnp.dot(x, W_sparse) + b_sparse
        u_stencil_flat = mask_sparse * u_hat_sparse + u_g_sparse
        u_stencil = u_stencil_flat.reshape((-1, 5))
        
        R_eq = (4 * u_stencil[:, 0] - u_stencil[:, 1] - u_stencil[:, 2] - 
                u_stencil[:, 3] - u_stencil[:, 4]) / dx**2 - F_eq
        return R_eq
    return res_fn

# UPDATED: Added Levenberg-Marquardt Damping
@jax.jit
def gauss_newton_step(z, F_eq, params, eq_weights, W_sparse, b_sparse, mask_sparse, u_g_sparse, damping=0.01):
    res_fn = create_fast_eq_residual_fn(params, F_eq, W_sparse, b_sparse, mask_sparse, u_g_sparse)
    
    R_eq, vjp_res = jax.vjp(res_fn, z)
    r_red = vjp_res(R_eq * eq_weights)[0] 
    
    def gn_op(delta_z):
        _, J_R_dz = jax.jvp(res_fn, (z,), (delta_z,))
        # Tikhonov regularization applied directly into the Hessian-Vector product
        return vjp_res(J_R_dz * eq_weights)[0] + damping * delta_z

    delta_z, _ = jax_linalg.cg(gn_op, -r_red)
    
    return z + delta_z, jnp.linalg.norm(r_red)

def fast_eq_latent_poisson_solver(z_init, params, F_vec, eq_indices, eq_weights, 
                                   W_sparse, b_sparse, mask_sparse, u_g_sparse, max_iters=15):
    F_eq = F_vec[eq_indices]
    z = z_init
    
    for _ in range(max_iters):
        z, res_norm = gauss_newton_step(z, F_eq, params, eq_weights, 
                                         W_sparse, b_sparse, mask_sparse, u_g_sparse, damping=0.01)
        
    u_final = mask * model.apply({'params': params}, z, method=model.decode) + u_g
    return z, u_final, []

# ==========================================
# 6. Systematic Benchmark: FOM vs EQ-ROM
# ==========================================
print("\n--- 4. Running Systematic Time and Accuracy Benchmark ---")

def get_analytical_solution(k1, k2):
    # Analytical: -Laplacian(u) = f => u = f / ((k1^2 + k2^2) * pi^2)
    coeff = 10.0 / ((k1**2 + k2**2) * (jnp.pi ** 2))
    u_exact_2d = coeff * jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y)
    return u_exact_2d.flatten()

# Test on integer (k1, k2) pairs not in training set, plus some from training
test_k_pairs = [(2, 3), (3, 2), (1, 4), (4, 1), (2, 4), (4, 2), (3, 4), (4, 3),
                (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (1, 5), (5, 1)] 

fom_times, rom_times = [], []
fom_vs_exact_errors, rom_vs_exact_errors = [], []
example_indices = [0, 4, 8]  # Save a few examples
example_solutions = []

# JAX Warm-up
print("   Warming up JAX compilers...")
_F_warmup = get_F(2, 2)
_ = full_order_fem_solver(K_op, _F_warmup, u_guess).block_until_ready()
_z_warmup = model.apply({'params': params}, U_train[0], method=model.encode)
_, _, _ = fast_eq_latent_poisson_solver(_z_warmup, params, _F_warmup, eq_indices_jnp, eq_weights_jnp,
                                         W_sparse, b_sparse, mask_sparse, u_g_sparse, max_iters=2)
print("   Warm-up complete.\n")

num_tests = len(test_k_pairs)
for i, (k1_test, k2_test) in enumerate(test_k_pairs):
    F_test = get_F(k1_test, k2_test)
    u_exact = get_analytical_solution(k1_test, k2_test)
    
    # --- FOM ---
    start_fom = time.time()
    u_fom = full_order_fem_solver(K_op, F_test, u_guess).block_until_ready()
    fom_time = time.time() - start_fom
    fom_times.append(fom_time)
    
    # --- EQ-ROM ---
    # Find closest training pair by index (since test pairs are in training set)
    try:
        train_idx = train_k_pairs.index((k1_test, k2_test))
        z_init = model.apply({'params': params}, U_train[train_idx], method=model.encode)
    except ValueError:
        # If not in training set, use average of nearby latent codes
        z_init = model.apply({'params': params}, U_train[0], method=model.encode)
    
    start_rom = time.time()
    z_final, u_rom, res_history = fast_eq_latent_poisson_solver(
        z_init, params, F_test, eq_indices_jnp, eq_weights_jnp,
        W_sparse, b_sparse, mask_sparse, u_g_sparse, max_iters=15
    )
    u_rom.block_until_ready()
    rom_time = time.time() - start_rom
    rom_times.append(rom_time)
    
    # --- Errors ---
    # Calculate both directly against the analytical solution
    fom_error = jnp.linalg.norm(u_fom - u_exact) / jnp.linalg.norm(u_exact)
    rom_error = jnp.linalg.norm(u_rom - u_exact) / jnp.linalg.norm(u_exact)
    
    fom_vs_exact_errors.append(fom_error)
    rom_vs_exact_errors.append(rom_error)
    
    # Store examples for visualization
    if i in example_indices:
        example_solutions.append({
            'k1': k1_test,
            'k2': k2_test,
            'analytical': u_exact.reshape((N, N)),
            'fom': u_fom.reshape((N, N)),
            'rom': u_rom.reshape((N, N)),
            'fom_err': float(fom_error),
            'rom_err': float(rom_error)
        })
    
    print(f"Test {i+1}/{num_tests} (k1={k1_test}, k2={k2_test}): FOM {fom_time:.4f}s | EQ-ROM {rom_time:.4f}s | "
          f"FOM err: {fom_error:.4e} | ROM err: {rom_error:.4e}")

# ==========================================
# Summary & Plotting
# ==========================================
avg_fom_time = jnp.mean(jnp.array(fom_times))
avg_rom_time = jnp.mean(jnp.array(rom_times))
avg_speedup = avg_fom_time / avg_rom_time

avg_fom_error = jnp.mean(jnp.array(fom_vs_exact_errors))
avg_rom_error = jnp.mean(jnp.array(rom_vs_exact_errors))

print(f"\n==========================================")
print(f"          BENCHMARK RESULTS               ")
print(f"==========================================")
print(f"Average FOM Time:        {avg_fom_time:.5f} seconds")
print(f"Average EQ-ROM Time:     {avg_rom_time:.5f} seconds")
print(f"Average Speedup:         {avg_speedup:.2f}x")
print(f"Nodes Evaluated (EQ):    {len(eq_indices)} / {num_nodes}")
print(f"------------------------------------------")
print(f"Avg FOM vs Exact Error:  {avg_fom_error:.4e}")
print(f"Avg ROM vs Exact Error:  {avg_rom_error:.4e}")
print(f"==========================================")

plt.figure(figsize=(12, 5))
test_labels = [f"({k1},{k2})" for k1, k2 in test_k_pairs]
x_pos = range(len(test_k_pairs))
plt.bar([x - 0.2 for x in x_pos], fom_vs_exact_errors, width=0.4, color='#2ca02c', label='FOM vs Analytical')
plt.bar([x + 0.2 for x in x_pos], rom_vs_exact_errors, width=0.4, color='#1f77b4', label='NM-ROM vs Analytical')
plt.title("Relative $L_2$ Error vs. Analytical Solution")
plt.xlabel("$(k_1, k_2)$")
plt.ylabel("Relative $L_2$ Error")
plt.xticks(x_pos, test_labels, rotation=45, ha='right')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'benchmark_error.png', dpi=150)
plt.show()

# ==========================================
# Example Comparison Plots
# ==========================================
print("\n--- Saving Example Comparison Plots ---")
for idx, ex in enumerate(example_solutions):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    vmin = min(ex['analytical'].min(), ex['fom'].min(), ex['rom'].min())
    vmax = max(ex['analytical'].max(), ex['fom'].max(), ex['rom'].max())
    
    im0 = axes[0].imshow(ex['analytical'], origin='lower', extent=[0, L, 0, L], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title(f"Analytical (k1={ex['k1']}, k2={ex['k2']})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    
    im1 = axes[1].imshow(ex['fom'], origin='lower', extent=[0, L, 0, L], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title(f"FOM (err: {ex['fom_err']:.2e})")
    axes[1].set_xlabel("x")
    
    im2 = axes[2].imshow(ex['rom'], origin='lower', extent=[0, L, 0, L], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[2].set_title(f"NM-ROM (err: {ex['rom_err']:.2e})")
    axes[2].set_xlabel("x")
    
    # Difference plot: ROM - FOM
    diff = ex['rom'] - ex['fom']
    diff_max = max(abs(diff.min()), abs(diff.max())) if max(abs(diff.min()), abs(diff.max())) > 0 else 1.0
    im3 = axes[3].imshow(diff, origin='lower', extent=[0, L, 0, L], vmin=-diff_max, vmax=diff_max, cmap='RdBu_r')
    axes[3].set_title("NM-ROM − FOM")
    axes[3].set_xlabel("x")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.colorbar(im0, ax=axes[:3], fraction=0.02, pad=0.02)
    plt.suptitle(f"Solution Comparison at $(k_1, k_2) = ({ex['k1']}, {ex['k2']})$", fontsize=14)
    plt.tight_layout()

    filename = OUTPUT_DIR / f"comparison_k{ex['k1']}_{ex['k2']}.png"
    plt.savefig(filename, dpi=150)
    print(f"   Saved: {filename}")
    plt.show()