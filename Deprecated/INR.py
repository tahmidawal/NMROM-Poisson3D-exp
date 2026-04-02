import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from typing import Sequence

# ==========================================
# 1. Fourier Feature Encoding (Solving Spectral Bias)
# ==========================================
class FourierFeatures(nn.Module):
    output_dim: int
    scale: float = 10.0

    @nn.compact
    def __call__(self, x):
        # x shape: (..., 3)
        in_dim = x.shape[-1]
        kernel = self.param('kernel', 
                            jax.nn.initializers.normal(stddev=self.scale), 
                            (in_dim, self.output_dim // 2))
        proj = jnp.dot(x, kernel)
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

# ==========================================
# 2. Spatially-Aware 3D Encoder
# ==========================================


class AttentionPooling(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, feat_map):
        # feat_map: (B, 4, 4, 4, C)
        B = feat_map.shape[0]
        C = feat_map.shape[-1]
        
        tokens = feat_map.reshape(B, -1, C)           # (B, 64, C)
        tokens = nn.Dense(self.latent_dim)(tokens)     # (B, 64, latent_dim)
        tokens = nn.LayerNorm()(tokens)                # stabilise before dot product
        
        query = self.param('query',
                           nn.initializers.normal(0.02),
                           (self.latent_dim,))
        
        scale = jnp.sqrt(float(self.latent_dim))
        scores = jnp.einsum('btd,d->bt', tokens, query) / scale  # (B, 64)
        weights = jax.nn.softmax(scores, axis=-1)                 # (B, 64)
        
        z = jnp.einsum('bt,btd->bd', weights, tokens)            # (B, latent_dim)
        return z


class CoordEncoder3D(nn.Module):
    latent_dim: int
    N: int = 32

    @nn.compact
    def __call__(self, u_grid):
        B = u_grid.shape[0]
        
        # CoordConv channels
        coords = jnp.stack(jnp.meshgrid(
            jnp.linspace(0, 1, self.N),
            jnp.linspace(0, 1, self.N),
            jnp.linspace(0, 1, self.N), indexing='ij'), axis=-1)
        coords = jnp.broadcast_to(coords, (B, self.N, self.N, self.N, 3))
        x = jnp.concatenate([u_grid, coords], axis=-1)  # (B, N, N, N, 4)
        
        # Conv stack
        x = nn.Conv(16, (3,3,3), strides=(2,2,2))(x);  x = nn.leaky_relu(x)
        x = nn.Conv(32, (3,3,3), strides=(2,2,2))(x);  x = nn.leaky_relu(x)
        x = nn.Conv(64, (3,3,3), strides=(2,2,2))(x);  x = nn.leaky_relu(x)
        # x: (B, 4, 4, 4, 64)
        
        z = AttentionPooling(latent_dim=self.latent_dim)(x)
        return z  # (B, latent_dim)


class CoordEncoder32D(nn.Module):
    latent_dim: int
    N: int = 32

    @nn.compact
    def __call__(self, u_grid):
        # u_grid: (Batch, N, N, N, 1)
        batch_size = u_grid.shape[0]
        
        # Create CoordConv channels
        coords = jnp.stack(jnp.meshgrid(
            jnp.linspace(0, 1, self.N),
            jnp.linspace(0, 1, self.N),
            jnp.linspace(0, 1, self.N), indexing='ij'), axis=-1)
        coords = jnp.broadcast_to(coords, (batch_size, self.N, self.N, self.N, 3))
        
        # Input is now (Batch, N, N, N, 4) -> (u, x, y, z)
        x = jnp.concatenate([u_grid, coords], axis=-1)
        
        # Parameter-efficient 3D Convolutions
        x = nn.Conv(16, (3, 3, 3), strides=(2, 2, 2))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(32, (3, 3, 3), strides=(2, 2, 2))(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(64, (3, 3, 3), strides=(2, 2, 2))(x)
        x = nn.leaky_relu(x)
        
        x = x.reshape((batch_size, -1))
        z = nn.Dense(self.latent_dim)(x)
        return z

# ==========================================
# 3. INR Decoder (The Field Function)
# ==========================================
class INRDecoder(nn.Module):
    features: int = 256

    @nn.compact
    def __call__(self, coords, z):
        # Two scales only — scale=100 encodes frequencies that 
        # don't exist in Poisson solutions and adds noise
        x_phi1 = FourierFeatures(output_dim=128, scale=1.0)(coords)
        x_phi2 = FourierFeatures(output_dim=128, scale=10.0)(coords)
        h = jnp.concatenate([x_phi1, x_phi2], axis=-1)  # (P, 256)

        for _ in range(4):
            # Re-inject z at every layer — keeps gradient path to encoder 
            # alive throughout, no init sensitivity
            z_rep = jnp.broadcast_to(z, (h.shape[0], z.shape[0]))  # (P, latent_dim)
            h = jnp.concatenate([h, z_rep], axis=-1)
            h = nn.Dense(self.features)(h)
            h = nn.swish(h)

        # Small output init → network starts near zero, loss starts at ~1.0
        # instead of ~476, and improves from there
        u_raw = nn.Dense(1, kernel_init=nn.initializers.normal(0.001))(h).squeeze(-1)
        dist_bc = jnp.prod(coords * (1.0 - coords), axis=-1) * 64.0
        return u_raw * dist_bc
class INRDecoder_OLD2(nn.Module):
    features: int = 256

    @nn.compact
    def __call__(self, coords, z):
        x_phi  = FourierFeatures(output_dim=128, scale=1.0)(coords)
        x_phi2 = FourierFeatures(output_dim=128, scale=10.0)(coords)
        x_phi3 = FourierFeatures(output_dim=128, scale=100.0)(coords)
        h = jnp.concatenate([x_phi, x_phi2, x_phi3], axis=-1)

        # Small init so FiLM starts as identity (gamma≈0, beta≈0)
        film_init = nn.initializers.normal(0.01)

        for i in range(4):
            h = nn.Dense(self.features)(h)
            h = nn.LayerNorm()(h)

            film = nn.Dense(self.features * 2,
                            kernel_init=film_init,
                            bias_init=nn.initializers.zeros)(z)
            gamma, beta = jnp.split(film, 2, axis=-1)
            gamma = jnp.broadcast_to(gamma, h.shape)
            beta  = jnp.broadcast_to(beta,  h.shape)
            h = h * (1.0 + gamma) + beta
            h = nn.swish(h)

        u_raw = nn.Dense(1)(h).squeeze(-1)
        dist_bc = jnp.prod(coords * (1.0 - coords), axis=-1) * 64.0
        return u_raw * dist_bc
# class INRDecoder(nn.Module):
#     features: int = 256

#     @nn.compact
#     def __call__(self, coords, z):
#         # Multi-scale Fourier encoding
#         x_phi = FourierFeatures(output_dim=128, scale=1.0)(coords)
#         x_phi2 = FourierFeatures(output_dim=128, scale=10.0)(coords)
#         x_phi3 = FourierFeatures(output_dim=128, scale=100.0)(coords)
#         h = jnp.concatenate([x_phi, x_phi2, x_phi3], axis=-1)  # (pts, 384)

#         for i in range(4):
#             h = nn.Dense(self.features)(h)
#             h = nn.LayerNorm()(h)

#             # FiLM: z generates per-channel scale and shift
#             film = nn.Dense(self.features * 2)(z)           # (latent_dim,) → (features*2,)
#             gamma, beta = jnp.split(film, 2, axis=-1)       # each (features,)
#             gamma = jnp.broadcast_to(gamma, h.shape)
#             beta  = jnp.broadcast_to(beta,  h.shape)
#             h = h * (1.0 + gamma) + beta
#             h = nn.swish(h)

#         u_raw = nn.Dense(1)(h).squeeze(-1)
#         dist_bc = jnp.prod(coords * (1.0 - coords), axis=-1) * 64.0
#         return u_raw * dist_bc

class INRDecoder_OLD(nn.Module):
    features: int = 256

    @nn.compact
    def __call__(self, coords, z):
        # coords: (num_points, 3), z: (latent_dim,)
        
        # 1. Map coordinates to Fourier Space
        x_phi = FourierFeatures(output_dim=128, scale=10.0)(coords)
        
        # 2. Concatenate with Latent Code
        z_rep = jnp.broadcast_to(z, (coords.shape[0], z.shape[-1]))
        h = jnp.concatenate([x_phi, z_rep], axis=-1)
        
        # 3. Deep MLP
        for _ in range(3):
            h = nn.Dense(self.features)(h)
            h = nn.swish(h)
            
        u_raw = nn.Dense(1)(h).squeeze(-1)
        
        # 4. Hard-code Dirichlet BCs: u = 0 at boundaries
        # b(x,y,z) = x*(1-x)*y*(1-y)*z*(1-z)
        dist_bc = jnp.prod(coords * (1.0 - coords), axis=-1) * 64.0
        return u_raw * dist_bc

# ==========================================
# 4. Integrated Model
# ==========================================
class PoissonINRAutoencoder(nn.Module):
    latent_dim: int = 128
    N: int = 32

    def setup(self):
        self.encoder = CoordEncoder3D(latent_dim=self.latent_dim, N=self.N)
        self.decoder = INRDecoder()

    def encode(self, u_grid):
        return self.encoder(u_grid)

    def decode_field(self, coords, z):
        # Vmap across the batch of latents if necessary
        if z.ndim == 2:
            return jax.vmap(self.decoder)(coords, z)
        return self.decoder(coords, z)

    def __call__(self, u_grid, coords=None):
        z = self.encode(u_grid)
        if coords is not None:
            return self.decode_field(coords, z)
        return z

# ==========================================
# 5. Hyperparameters & Training Setup
# ==========================================

# Hyperparameters
LATENT_DIM = 256
LR = 1e-4
EPOCHS = 10000
BATCH_SIZE = 16 
POINTS_PER_UPDATE = 4096 # We sample random points in the volume, not the whole grid!

model = PoissonINRAutoencoder(latent_dim=LATENT_DIM, N=32)
# Initialize with both u_grid AND coords to ensure decoder params are created
dummy_coords = jnp.ones((1, 100, 3))  # (batch, num_points, 3)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 32, 1)), dummy_coords)['params']
tx = optax.adam(learning_rate=optax.exponential_decay(LR, transition_steps=8000, decay_rate=0.5))
opt_state = tx.init(params)

@jax.jit
def train_step_OLD(p, opt_st, u_grids, coords, u_targets):
    def loss_fn(weights):
        z = model.apply({'params': weights}, u_grids, method=model.encode)
        u_pred = model.apply({'params': weights}, coords, z, method=model.decode_field)

        # Relative L2 — equal gradient signal across all k values
        norms = jnp.sqrt(jnp.mean(u_targets**2, axis=-1, keepdims=True)) + 1e-8
        recon_loss = jnp.mean(((u_pred - u_targets) / norms)**2)

        # Latent regularization — keeps GN Jacobian well-conditioned
        latent_reg = 1e-4 * jnp.mean(z**2)

        return recon_loss + latent_reg

    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

@jax.jit
def train_step(p, opt_st, u_grids, coords, u_targets):
    def loss_fn(weights):
        z = model.apply({'params': weights}, u_grids, method=model.encode)
        u_pred = model.apply({'params': weights}, coords, z, method=model.decode_field)

        # Norm from full grid — stable regardless of which points are sampled
        u_full = u_grids.reshape(u_grids.shape[0], -1)          # (B, N³)
        norms = jnp.sqrt(jnp.mean(u_full**2, axis=-1, keepdims=True)) + 1e-8  # (B, 1)

        recon_loss = jnp.mean(((u_pred - u_targets) / norms)**2)
        latent_reg = 1e-4 * jnp.mean(z**2)
        return recon_loss + latent_reg

    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

@jax.jit
def train_step_OLD(p, opt_st, u_grids, coords, u_targets):
    """
    u_grids: (B, 32, 32, 32, 1) - used for encoding
    coords: (B, num_points, 3) - query points
    u_targets: (B, num_points) - ground truth at query points
    """
    def loss_fn(weights):
        z = model.apply({'params': weights}, u_grids, method=model.encode)
        u_pred = model.apply({'params': weights}, coords, z, method=model.decode_field)
        return jnp.mean((u_pred - u_targets)**2)
    
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    return optax.apply_updates(p, updates), new_opt_st, loss

print(f"Architecture loaded. Latent Dim: {LATENT_DIM}")

# ==========================================
# 6. Data Generation (3D Poisson with sinusoidal forcing)
# ==========================================
import jax.scipy.sparse.linalg as jax_linalg

N = 32
L = 1.0
dx = L / (N - 1)
num_nodes = N ** 3

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

def K_op_3d(u_flat):
    """7-point finite-difference 3D Laplacian with Dirichlet BCs."""
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
    """Sinusoidal forcing compatible with Dirichlet BCs."""
    F_3d = (jnp.sin(k1 * jnp.pi * X) 
            * jnp.sin(k2 * jnp.pi * Y) 
            * jnp.sin(k3 * jnp.pi * Z) * 10.0)
    F_3d = F_3d.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    F_3d = F_3d.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    F_3d = F_3d.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    return F_3d.flatten()

def solve_poisson(F_vec):
    u_sol, _ = jax_linalg.cg(K_op_3d, F_vec, x0=jnp.zeros(num_nodes), tol=1e-6, maxiter=2000)
    return u_sol

print("\n--- Generating Training Data ---")
train_k_triples = [(k1, k2, k3) for k1 in range(1, 5) for k2 in range(1, 5) for k3 in range(1, 5)]
print(f"Generating {len(train_k_triples)} snapshots...")

U_train_list = []
for i, (k1, k2, k3) in enumerate(train_k_triples):
    F_vec = get_F_3d(k1, k2, k3)
    u_sol = solve_poisson(F_vec)
    U_train_list.append(u_sol.reshape(N, N, N, 1))
    if (i + 1) % 16 == 0:
        print(f"  Snapshot {i+1}/{len(train_k_triples)}")

U_train = jnp.stack(U_train_list)  # (64, 32, 32, 32, 1)
print(f"Training data shape: {U_train.shape}")

# Query coordinates for INR training (full grid)
coords_grid = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # (32768, 3)

# ==========================================
# 7. Training Loop
# ==========================================
print(f"\n--- Training INR Autoencoder ---")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Points per update: {POINTS_PER_UPDATE}")

key = jax.random.PRNGKey(42)

for epoch in range(EPOCHS):
    key, subkey = jax.random.split(key)
    
    # Sample batch of grids
    batch_idx = jax.random.randint(subkey, (BATCH_SIZE,), 0, len(U_train))
    u_grids = U_train[batch_idx]  # (B, 32, 32, 32, 1)
    
    # Sample random query points
    key, subkey = jax.random.split(key)
    point_idx = jax.random.randint(subkey, (POINTS_PER_UPDATE,), 0, num_nodes)
    coords = coords_grid[point_idx]  # (num_points, 3)
    coords_batch = jnp.broadcast_to(coords, (BATCH_SIZE, POINTS_PER_UPDATE, 3))
    
    # Ground truth at query points
    u_targets = u_grids.reshape(BATCH_SIZE, -1)[:, point_idx]  # (B, num_points)
    
    params, opt_state, loss = train_step(params, opt_state, u_grids, coords_batch, u_targets)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}, Loss: {loss:.6e}")

print(f"\nTraining complete. Final loss: {loss:.6e}")

# ==========================================
# 8. Save Model Weights
# ==========================================
import pickle
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

weights_path = WEIGHTS_DIR / 'inr_autoencoder.pkl'
with open(weights_path, 'wb') as f:
    pickle.dump({'params': params, 'config': {'latent_dim': LATENT_DIM, 'N': N}}, f)
print(f"\nWeights saved to: {weights_path}")

# ==========================================
# 9. Reconstruction Visualization
# ==========================================
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'plots' / 'INR_reconstruction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n--- Generating Reconstruction Plots ---")

# Select a few test samples
test_indices = [0, 15, 31, 63]  # Different k combinations
mid = N // 2

for idx in test_indices:
    u_grid = U_train[idx:idx+1]  # (1, 32, 32, 32, 1)
    k1, k2, k3 = train_k_triples[idx]
    
    # Encode and decode through INR
    z = model.apply({'params': params}, u_grid, method=model.encode)
    
    # Reconstruct on full grid
    coords_full = coords_grid[None, :, :]  # (1, 32768, 3)
    u_pred = model.apply({'params': params}, coords_full, z, method=model.decode_field)
    u_pred = u_pred.reshape(N, N, N)
    
    u_true = u_grid[0, :, :, :, 0]  # (32, 32, 32)
    
    # Compute error
    rel_error = float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))
    
    # Plot midplane slices (XY at z=mid, XZ at y=mid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # XY slice at z=mid
    sl_true = u_true[:, :, mid]
    sl_pred = u_pred[:, :, mid]
    vmin, vmax = float(sl_true.min()), float(sl_true.max())
    kw = dict(origin='lower', aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, L, 0, L])
    
    im0 = axes[0, 0].imshow(np.array(sl_true).T, **kw)
    axes[0, 0].set_title('Ground Truth (z=0.5)', fontsize=11)
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    im1 = axes[0, 1].imshow(np.array(sl_pred).T, **kw)
    axes[0, 1].set_title('INR Reconstruction (z=0.5)', fontsize=11)
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    err_xy = np.abs(np.array(sl_pred - sl_true))
    im2 = axes[0, 2].imshow(err_xy.T, origin='lower', aspect='equal', cmap='hot', extent=[0, L, 0, L])
    axes[0, 2].set_title('|Error| (z=0.5)', fontsize=11)
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    
    # XZ slice at y=mid
    sl_true = u_true[:, mid, :]
    sl_pred = u_pred[:, mid, :]
    vmin, vmax = float(sl_true.min()), float(sl_true.max())
    kw = dict(origin='lower', aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, L, 0, L])
    
    im3 = axes[1, 0].imshow(np.array(sl_true).T, **kw)
    axes[1, 0].set_title('Ground Truth (y=0.5)', fontsize=11)
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('z')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    im4 = axes[1, 1].imshow(np.array(sl_pred).T, **kw)
    axes[1, 1].set_title('INR Reconstruction (y=0.5)', fontsize=11)
    axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('z')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    err_xz = np.abs(np.array(sl_pred - sl_true))
    im5 = axes[1, 2].imshow(err_xz.T, origin='lower', aspect='equal', cmap='hot', extent=[0, L, 0, L])
    axes[1, 2].set_title('|Error| (y=0.5)', fontsize=11)
    axes[1, 2].set_xlabel('x'); axes[1, 2].set_ylabel('z')
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)
    
    fig.suptitle(f'INR Reconstruction: k=({k1},{k2},{k3}) | Rel. L2 Error: {rel_error:.4e}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    fpath = OUTPUT_DIR / f'recon_k{k1}{k2}{k3}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath} | Rel. Error: {rel_error:.4e}")

# Training loss curve (if we tracked it)
print(f"\n=== INR Training Complete ===")
print(f"Plots saved to: {OUTPUT_DIR}")