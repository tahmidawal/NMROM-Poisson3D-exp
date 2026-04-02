import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# ==========================================
# 0. Output Directories
# ==========================================
OUTPUT_DIR = Path(__file__).parent / 'plots' / '3D_poisson_cnn'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Domain Setup & Parametric Physics (3D)
# ==========================================
N = 32  # Grid size (32^3 = 32,768 nodes)
num_nodes = N ** 3
k_dim = 128  # Latent dimension (Increased to match your PyTorch script)
L = 1.0
dx = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

def K_op_3d(u_flat):
    """7-point finite-difference 3D Laplacian with Dirichlet BCs."""
    u_3d = u_flat.reshape((N, N, N))
    out = jnp.zeros_like(u_3d)
    
    # Interior
    out = out.at[1:-1, 1:-1, 1:-1].set(
        (6 * u_3d[1:-1, 1:-1, 1:-1]
         - u_3d[0:-2, 1:-1, 1:-1] - u_3d[2:, 1:-1, 1:-1]
         - u_3d[1:-1, 0:-2, 1:-1] - u_3d[1:-1, 2:, 1:-1]
         - u_3d[1:-1, 1:-1, 0:-2] - u_3d[1:-1, 1:-1, 2:]) / dx**2
    )
    
    # Boundary (Dirichlet u=0)
    out = out.at[0, :, :].set(u_3d[0, :, :])
    out = out.at[-1, :, :].set(u_3d[-1, :, :])
    out = out.at[:, 0, :].set(u_3d[:, 0, :])
    out = out.at[:, -1, :].set(u_3d[:, -1, :])
    out = out.at[:, :, 0].set(u_3d[:, :, 0])
    out = out.at[:, :, -1].set(u_3d[:, :, -1])
    
    return out.flatten()

def get_F_3d(k1, k2, k3):
    """3D forcing function."""
    F_3d = (jnp.sin(k1 * jnp.pi * X) 
            * jnp.sin(k2 * jnp.pi * Y) 
            * jnp.sin(k3 * jnp.pi * Z) * 10.0)
    # Explicitly zero boundaries
    F_3d = F_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    F_3d = F_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    F_3d = F_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    return F_3d.flatten()

def get_analytical_solution_3d(k1, k2, k3):
    coeff = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
    return (coeff * jnp.sin(k1 * jnp.pi * X) * jnp.sin(k2 * jnp.pi * Y) * jnp.sin(k3 * jnp.pi * Z)).flatten()

# Boundary mask
mask_3d = jnp.ones((N, N, N))
mask_3d = mask_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
mask_3d = mask_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
mask_3d = mask_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
mask = mask_3d.flatten()
u_g = jnp.zeros(num_nodes)

print(f"3D Domain: {N}x{N}x{N} = {num_nodes:,} nodes")

def full_order_fem_solver_3d(K_operator, F_vec, u_guess, tol=1e-6):
    u_true, _ = jax_linalg.cg(K_operator, F_vec, x0=u_guess, tol=tol, maxiter=2000)
    return u_true

# ==========================================
# 2. Pure 3D CNN Autoencoder (Flax)
# ==========================================
class PoissonAutoencoder3D(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

    @nn.compact
    def encode(self, x):
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape((1, 32, 32, 32, 1))
        elif x.ndim == 2:
            x = x.reshape((x.shape[0], 32, 32, 32, 1))

        x = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME', name='enc_conv1')(x)
        x = nn.GroupNorm(num_groups=8, name='enc_gn1')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME', name='enc_conv2')(x)
        x = nn.GroupNorm(num_groups=8, name='enc_gn2')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME', name='enc_conv3')(x)
        x = nn.GroupNorm(num_groups=8, name='enc_gn3')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = x.reshape((x.shape[0], -1)) 
        
        # Explicit name prevents collision
        z = nn.Dense(features=self.latent_dim, name='enc_dense_final')(x)
        return z[0] if is_single else z

    @nn.compact
    def decode(self, z):
        is_single = z.ndim == 1
        if is_single:
            z = z.reshape((1, -1))
            
        # Explicit name prevents collision
        x = nn.Dense(features=128 * 4 * 4 * 4, name='dec_dense_base')(z)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], 4, 4, 4, 128))
        
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', name='dec_convT1')(x)
        x = nn.GroupNorm(num_groups=8, name='dec_gn1')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', name='dec_convT2')(x)
        x = nn.GroupNorm(num_groups=8, name='dec_gn2')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', name='dec_convT3')(x)
        
        x = x.reshape((x.shape[0], -1))
        return x[0] if is_single else x
class PoissonAutoencoder3D3(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

    @nn.compact  # <--- ADD THIS DECORATOR
    def encode(self, x):
        # Handle flat input (Batch, 32768) -> (Batch, 32, 32, 32, 1)
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape((1, 32, 32, 32, 1))
        elif x.ndim == 2:
            x = x.reshape((x.shape[0], 32, 32, 32, 1))

        x = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = x.reshape((x.shape[0], -1)) # Flatten for dense layer
        z = nn.Dense(features=self.latent_dim)(x)
        return z[0] if is_single else z

    @nn.compact  # <--- ADD THIS DECORATOR
    def decode(self, z):
        is_single = z.ndim == 1
        if is_single:
            z = z.reshape((1, -1))
            
        x = nn.Dense(features=128 * 4 * 4 * 4)(z)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], 4, 4, 4, 128))
        
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # Final layer - No Tanh, mapping to 1 channel
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME')(x)
        
        # Flatten back to flat nodes array (Batch, 32768)
        x = x.reshape((x.shape[0], -1))
        return x[0] if is_single else x

class PoissonAutoencoder3D2(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x):
        # Handle flat input (Batch, 32768) -> (Batch, 32, 32, 32, 1)
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape((1, 32, 32, 32, 1))
        elif x.ndim == 2:
            x = x.reshape((x.shape[0], 32, 32, 32, 1))

        x = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = x.reshape((x.shape[0], -1)) # Flatten for dense layer
        z = nn.Dense(features=self.latent_dim)(x)
        return z[0] if is_single else z

    def decode(self, z):
        is_single = z.ndim == 1
        if is_single:
            z = z.reshape((1, -1))
            
        x = nn.Dense(features=128 * 4 * 4 * 4)(z)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], 4, 4, 4, 128))
        
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # Final layer - No Tanh, mapping to 1 channel
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME')(x)
        
        # Flatten back to flat nodes array (Batch, 32768)
        x = x.reshape((x.shape[0], -1))
        return x[0] if is_single else x

def create_constrained_decode_fn(params, model, mask_vec, u_g_vec):
    def constrained_decode(lat):
        u_raw = model.apply({'params': params}, lat, method=model.decode)
        return mask_vec * u_raw + u_g_vec
    return constrained_decode

# ==========================================
# 3. Data Generation
# ==========================================
print("\n--- Generating 3D Training Dataset ---")
train_k_triples = [(k1, k2, k3) for k1 in range(1, 5) for k2 in range(1, 5) for k3 in range(1, 5)]

u_guess = jnp.zeros(num_nodes)
U_train_list = []
for i, (k1, k2, k3) in enumerate(train_k_triples):
    F_vec = get_F_3d(k1, k2, k3)
    u_sol = full_order_fem_solver_3d(K_op_3d, F_vec, u_guess)
    U_train_list.append(u_sol)

U_train = jnp.stack(U_train_list)
print(f"Training data shape: {U_train.shape}")

# ==========================================
# 4. Train 3D CNN Autoencoder
# ==========================================
model = PoissonAutoencoder3D(latent_dim=k_dim)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, num_nodes)))['params']

schedule = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.9)
tx = optax.adam(learning_rate=schedule)
opt_state = tx.init(params)

@jax.jit
def train_step(p, opt_st, batch):
    def loss_fn(weights):
        preds = jax.vmap(lambda inp: model.apply({'params': weights}, inp))(batch)
        return jnp.mean((batch - preds) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

print("\n--- Training 3D CNN Autoencoder ---")
for epoch in range(5000):
    params, opt_state, loss = train_step(params, opt_state, U_train)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}, Loss: {loss:.4e}")

# ==========================================
# 5. Full-Field Gauss-Newton Solver
# ==========================================
print("\n--- Setting up Full-Field ROM Solver ---")

def make_full_latent_solver(p, model, mask_sp, ug_sp):
    constrained_decode = create_constrained_decode_fn(p, model, mask_sp, ug_sp)
    
    def _res_fn(lat, F_full):
        u_decoded = constrained_decode(lat)
        return K_op_3d(u_decoded) - F_full

    @jax.jit
    def solve(lat_init, F_full):
        def _body(carry):
            lat, _, itr = carry
            
            R = _res_fn(lat, F_full)
            J = jax.jacfwd(lambda l: _res_fn(l, F_full))(lat) # Full Jacobian (32768 x k_dim)
            
            JtJ = J.T @ J
            JtR = J.T @ R
            
            lam = jnp.maximum(1e-3 * jnp.trace(JtJ) / k_dim, 1e-8)
            dz = jnp.linalg.solve(JtJ + lam * jnp.eye(k_dim), -JtR)
            
            f0 = jnp.dot(R, R)
            def _wsn(alpha):
                R_trial = _res_fn(lat + alpha * dz, F_full)
                return jnp.dot(R_trial, R_trial)
            
            f1, f2, f3, f4 = _wsn(1.0), _wsn(0.5), _wsn(0.25), _wsn(0.125)
            step = jnp.where(f1 < f0, 1.0, jnp.where(f2 < f0, 0.5, jnp.where(f3 < f0, 0.25, 0.125)))
            
            return lat + step * dz, jnp.linalg.norm(JtR), itr + 1
        
        def _cond(carry):
            _, grad_norm, itr = carry
            return jnp.logical_and(grad_norm > 1e-8, itr < 30)
        
        init_carry = (lat_init, jnp.array(jnp.inf, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32))
        lat_f, res_f, n_iters = jax.lax.while_loop(_cond, _body, init_carry)
        return lat_f, res_f, n_iters
    
    return solve

latent_solve = make_full_latent_solver(params, model, mask, u_g)

def full_field_latent_poisson_solver(lat_init, F_vec):
    lat_f, res_f, n_iters = latent_solve(lat_init, F_vec)
    u_final = create_constrained_decode_fn(params, model, mask, u_g)(lat_f)
    return lat_f, u_final, res_f, n_iters

# ==========================================
# 6. Benchmark
# ==========================================
print("\n--- Running Benchmark ---")
test_k_triples = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (2, 3, 4)]
fom_times, rom_times = [], []

# Warm-up JIT
_F_wm = get_F_3d(2, 2, 2)
_lat_wm = model.apply({'params': params}, U_train[0], method=model.encode)
_, _u_wm, _, _ = full_field_latent_poisson_solver(_lat_wm, _F_wm)
jax.block_until_ready(_u_wm)
print("JIT Warm-up complete.\n")

for i, (k1, k2, k3) in enumerate(test_k_triples):
    F_test = get_F_3d(k1, k2, k3)
    
    # FOM
    t0 = time.perf_counter()
    u_fom = full_order_fem_solver_3d(K_op_3d, F_test, u_guess).block_until_ready()
    fom_t = time.perf_counter() - t0
    fom_times.append(fom_t)
    
    # Init Latent (Simple mean of train for brevity, though IDW from your code works too)
    lat_init = model.apply({'params': params}, U_train[0], method=model.encode) 
    
    # ROM
    t0 = time.perf_counter()
    lat_f, u_rom, gn_res, n_iters = full_field_latent_poisson_solver(lat_init, F_test)
    jax.block_until_ready(u_rom)
    rom_t = time.perf_counter() - t0
    rom_times.append(rom_t)
    
    err_rom_fom = float(jnp.linalg.norm(u_rom - u_fom) / jnp.linalg.norm(u_fom))
    
    print(f"[{i+1}/{len(test_k_triples)}] k=({k1},{k2},{k3}) | "
          f"FOM {fom_t:.4f}s | Full-ROM {rom_t:.4f}s | "
          f"Err {err_rom_fom:.3e} | iters {int(n_iters)}")

print("\n=== 3D Poisson Pure CNN ROM Complete ===")