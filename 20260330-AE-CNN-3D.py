import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
import time
from pathlib import Path
import orbax.checkpoint as ocp
import json
from datetime import datetime

# ==========================================
# 1. Setup & Data Generation
# ==========================================
N = 32
num_nodes = N ** 3
L = 1.0
dx = L / (N - 1)

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

# Pre-compute coordinates for CoordConv
X_chan = X[None, ..., None]
Y_chan = Y[None, ..., None]
Z_chan = Z[None, ..., None]
coords_3d = jnp.concatenate([X_chan, Y_chan, Z_chan], axis=-1)  # Shape: (1, 32, 32, 32, 3)

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

def full_order_fem_solver_3d(K_operator, F_vec, u_guess, tol=1e-6):
    u_true, _ = jax_linalg.cg(K_operator, F_vec, x0=u_guess, tol=tol, maxiter=2000)
    return u_true

print("Generating Training Data...")
train_k_triples = [(k1, k2, k3) for k1 in range(1, 5) for k2 in range(1, 5) for k3 in range(1, 5)]
u_guess = jnp.zeros(num_nodes)
U_train_list = []

for k1, k2, k3 in train_k_triples:
    F_vec = get_F_3d(k1, k2, k3)
    U_train_list.append(full_order_fem_solver_3d(K_op_3d, F_vec, u_guess))

# CNNs require a channel dimension: (Batch, N, N, N, 1)
U_train = jnp.stack(U_train_list).reshape(-1, N, N, N, 1)
print(f"Training data shape: {U_train.shape}")

# ==========================================
# 2. Fully Convolutional Architecture
# ==========================================
class ResBlock3D(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, kernel_size=(3, 3, 3), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.Conv(self.features, kernel_size=(3, 3, 3), padding='SAME')(x)
        return nn.swish(x + residual)

class Encoder3D(nn.Module):
    latent_channels: int = 1  # 4x4x4x1 = 64 latent variables
    base_channels: int = 32
    
    @nn.compact
    def __call__(self, x, coords):
        # Broadcast coords to match batch size and concat with physics field
        coords_batch = jnp.broadcast_to(coords, (x.shape[0], *coords.shape[1:]))
        x = jnp.concatenate([x, coords_batch], axis=-1)
        
        x = nn.Conv(self.base_channels, kernel_size=(3, 3, 3), padding='SAME')(x)
        x = nn.swish(x)
        
        # Downsample 1: 32^3 -> 16^3
        x = nn.Conv(self.base_channels * 2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = ResBlock3D(self.base_channels * 2)(x)
        
        # Downsample 2: 16^3 -> 8^3
        x = nn.Conv(self.base_channels * 4, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = ResBlock3D(self.base_channels * 4)(x)
        
        # Downsample 3: 8^3 -> 4^3
        x = nn.Conv(self.base_channels * 8, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(x)
        x = ResBlock3D(self.base_channels * 8)(x)
        
        # Spatial Bottleneck (No flattening!)
        z = nn.Conv(self.latent_channels, kernel_size=(3, 3, 3), padding='SAME')(x)
        return z

class Decoder3D(nn.Module):
    base_channels: int = 32
    
    @nn.compact
    def __call__(self, z, coords):
        x = nn.Conv(self.base_channels * 8, kernel_size=(3, 3, 3), padding='SAME')(z)
        x = nn.swish(x)
        
        channels = [self.base_channels * 4, self.base_channels * 2, self.base_channels]
        
        for c in channels:
            # Upsample using standard python shapes to avoid JitTracers
            b, h, w, d, _ = x.shape
            new_shape = (b, h * 2, w * 2, d * 2, x.shape[-1])
            x = jax.image.resize(x, shape=new_shape, method='linear')
            
            x = nn.Conv(c, kernel_size=(3, 3, 3), padding='SAME')(x)
            x = ResBlock3D(c)(x)
            
        # Final Coordinate Injection for boundary awareness
        coords_batch = jnp.broadcast_to(coords, (x.shape[0], *coords.shape[1:]))
        x = jnp.concatenate([x, coords_batch], axis=-1)
        
        # Output single scalar field
        x = nn.Conv(1, kernel_size=(3, 3, 3), padding='SAME')(x)
        return x

class FullAutoencoder(nn.Module):
    latent_channels: int = 1
    
    def setup(self):
        # Explicitly instantiate the sub-modules
        self.encoder = Encoder3D(latent_channels=self.latent_channels)
        self.decoder = Decoder3D()
        
    def __call__(self, x, coords):
        z = self.encoder(x, coords)
        x_hat = self.decoder(z, coords)
        return x_hat
        
    def encode(self, x, coords):
        return self.encoder(x, coords)
        
    def decode(self, z, coords):
        return self.decoder(z, coords)

# ==========================================
# 3. Training Loop with Gradient Clipping
# ==========================================
model = FullAutoencoder(latent_channels=1)
key = jax.random.PRNGKey(0)

# Initialize with dummy data
dummy_input = jnp.ones((1, N, N, N, 1))
params = model.init(key, dummy_input, coords_3d)['params']

# Print architecture info
dummy_z = model.apply({'params': params}, dummy_input, coords_3d, method=model.encode)
print(f"\nModel initialized.")
print(f"Spatial Latent Embedding Shape: {dummy_z.shape} (Total {np.prod(dummy_z.shape[1:])} variables)")

# Optimizer with gradient clipping to prevent explosion
schedule = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.9)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # Prevents loss spikes
    optax.adam(learning_rate=schedule)
)
opt_state = tx.init(params)

@jax.jit
def train_step(p, opt_st, batch, coords_batch):
    def loss_fn(weights):
        preds = model.apply({'params': weights}, batch, coords_batch)
        return jnp.mean((batch - preds) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(p)
    updates, new_opt_st = tx.update(grads, opt_st, p)
    new_params = optax.apply_updates(p, updates)
    return new_params, new_opt_st, loss

# ==========================================
# 4. Setup Logging & Checkpointing
# ==========================================
run_dir = Path(__file__).parent / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir = run_dir / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Training log
log_file = run_dir / 'training_log.json'
loss_history = []

print(f"\nRun directory: {run_dir}")
print("Starting Training...")

epochs = 5000
checkpointer = ocp.PyTreeCheckpointer()

for epoch in range(epochs):
    params, opt_state, loss = train_step(params, opt_state, U_train, coords_3d)
    loss_val = float(loss)
    loss_history.append({'epoch': epoch, 'loss': loss_val})
    
    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d}, Loss: {loss_val:.4e}")
        
        # Save checkpoint
        ckpt_path = ckpt_dir / f'epoch_{epoch:05d}'
        if ckpt_path.exists():
            import shutil
            shutil.rmtree(ckpt_path)
        checkpointer.save(ckpt_path, {'params': params})
        
        # Save log
        with open(log_file, 'w') as f:
            json.dump({
                'config': {'N': N, 'latent_channels': 1, 'epochs': epochs},
                'loss_history': loss_history
            }, f, indent=2)

print(f"\nTraining complete. Final loss: {loss_history[-1]['loss']:.4e}")
print(f"Checkpoints saved to: {ckpt_dir}")
print(f"Training log saved to: {log_file}")