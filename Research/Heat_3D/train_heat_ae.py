# """
# train_heat_ae.py
# ────────────────
# ScalableAutoencoder for 3D Heat Equation NM-ROM.

# Three targeted accuracy improvements over baseline:

# 1. CoordConv — concatenate (x,y,z) coordinate channels to encoder input.
#    The convolutions can now directly see WHERE features are, not just
#    WHAT they look like. Critical for localised Gaussian features.

# 2. Per-sample amplitude normalisation — encode u/max(|u|), store scale
#    as a separate scalar, multiply back at decode time. Removes amplitude
#    from what the encoder must track, simplifying its task to shape/position.

# 3. Residual encoder blocks + deeper decoder MLP — residual connections
#    prevent information loss through strided downsampling; deeper MLP
#    (3 hidden layers + skip) gives the decoder more expressive power.

# The CP decoder structure (and therefore the EQ precomputation) is unchanged.
# """

# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import optax
# import jax.scipy.sparse.linalg as jax_linalg
# import numpy as np
# from scipy.stats import qmc
# import matplotlib.pyplot as plt
# import pickle
# import time
# import sys
# from pathlib import Path
# from typing import Sequence

# # ─────────────────────────────────────────
# # 0. Paths & Logging
# # ─────────────────────────────────────────
# SCRIPT_DIR = Path(__file__).parent.resolve()
# OUT = SCRIPT_DIR / 'plots'
# OUT.mkdir(parents=True, exist_ok=True)

# # Setup logging to file
# LOG_FILE = SCRIPT_DIR / 'training.log'
# class TeeLogger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, 'w')
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()
# sys.stdout = TeeLogger(LOG_FILE)

# # ─────────────────────────────────────────
# # 1. Grid & Physics
# # ─────────────────────────────────────────
# N         = 32
# num_nodes = N ** 3
# L         = 1.0
# dx        = L / (N - 1)
# dt        = 0.005
# NUM_STEPS = 50        # total time T = 0.25s

# x_sp = jnp.linspace(0, L, N)
# y_sp = jnp.linspace(0, L, N)
# z_sp = jnp.linspace(0, L, N)
# X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

# # Precompute coordinate grids for CoordConv — shape (N,N,N,3)
# # Normalised to [-1, 1] so coordinate channels have same scale as field values
# COORD_GRID = jnp.stack([
#     2.0*X/L - 1.0,
#     2.0*Y/L - 1.0,
#     2.0*Z/L - 1.0,
# ], axis=-1)   # (N, N, N, 3)

# # ── Negative Laplacian (7-point stencil, Dirichlet BCs) ──────────────
# def K_op_3d(u_flat):
#     u   = u_flat.reshape((N, N, N))
#     out = jnp.zeros_like(u)
#     out = out.at[1:-1,1:-1,1:-1].set(
#         (6*u[1:-1,1:-1,1:-1]
#          - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
#          - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
#          - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
#     )
#     # Boundary rows: identity (Dirichlet → stay 0)
#     out = out.at[0,:,:].set(u[0,:,:])
#     out = out.at[-1,:,:].set(u[-1,:,:])
#     out = out.at[:,0,:].set(u[:,0,:])
#     out = out.at[:,-1,:].set(u[:,-1,:])
#     out = out.at[:,:,0].set(u[:,:,0])
#     out = out.at[:,:,-1].set(u[:,:,-1])
#     return out.flatten()

# # Backward Euler operator: (I + dt*κ*K)
# # κ passed at call time so we can vary diffusivity per trajectory
# def implicit_op(u_flat, kappa):
#     return u_flat + dt * kappa * K_op_3d(u_flat)

# # Boundary mask
# mask_3d = jnp.ones((N,N,N))
# mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
# mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
# mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
# mask    = mask_3d.flatten()

# print(f"Grid: {N}³ = {num_nodes:,} nodes  |  dt={dt}  T={dt*NUM_STEPS:.3f}s")

# # ─────────────────────────────────────────
# # 2. Gaussian IC Generation
# # ─────────────────────────────────────────
# def make_gaussian_ic(centers, amplitudes, widths):
#     """
#     u0(x,y,z) = Σ_i A_i * exp(-|r - c_i|² / (2σ_i²))
#     BCs enforced by multiplying a smooth boundary decay.

#     centers:    (M, 3)  in [0,1]^3
#     amplitudes: (M,)
#     widths:     (M,)
#     """
#     u = jnp.zeros((N, N, N))
#     for (cx, cy, cz), A, sigma in zip(centers, amplitudes, widths):
#         u = u + A * jnp.exp(
#             -((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2)
#         )
#     # Hard-zero the boundaries
#     u = u.at[0,:,:].set(0.).at[-1,:,:].set(0.)
#     u = u.at[:,0,:].set(0.).at[:,-1,:].set(0.)
#     u = u.at[:,:,0].set(0.).at[:,:,-1].set(0.)
#     return u.flatten()


# def sample_trajectory_params(rng, n_traj):
#     """
#     Latin Hypercube Sampling over trajectory parameter space.

#     Parameter vector per trajectory (dim = 11):
#       [0]     num_gaussians (1–3, encoded as 0–1 → rounded)
#       [1–3]   center_1  (x, y, z)
#       [4–6]   center_2
#       [7–9]   center_3
#       [10]    amplitude_scale  (0–1 → [1, 10])
#       [11]    width_scale      (0–1 → [0.05, 0.2])
#       [12]    kappa_log        (0–1 → log[0.01, 0.5])

#     Returns list of dicts, one per trajectory.
#     """
#     sampler = qmc.LatinHypercube(d=13, seed=rng)
#     samples = sampler.random(n=n_traj)          # (n_traj, 13)  all in [0,1]

#     trajectories = []
#     for s in samples:
#         n_gauss = int(np.round(1 + 2 * s[0]))   # 1, 2, or 3

#         centers    = []
#         amplitudes = []
#         widths     = []

#         for g in range(n_gauss):
#             # Centers safely away from boundary
#             cx = 0.15 + 0.70 * s[1 + g*3]
#             cy = 0.15 + 0.70 * s[2 + g*3]
#             cz = 0.15 + 0.70 * s[3 + g*3]
#             centers.append((cx, cy, cz))
#             amplitudes.append(1.0 + 9.0 * s[10])   # [1, 10]
#             widths.append(0.05 + 0.15 * s[11])      # [0.05, 0.20]

#         kappa = float(np.exp(np.log(0.01) + (np.log(0.5) - np.log(0.01)) * s[12]))

#         trajectories.append(dict(
#             centers=centers, amplitudes=amplitudes,
#             widths=widths, kappa=kappa
#         ))
#     return trajectories


# # ─────────────────────────────────────────
# # 3. FOM Time-Stepping (Backward Euler + CG)
# # ─────────────────────────────────────────
# def run_fom(u0_flat, kappa, steps):
#     """
#     Returns snapshot array of shape (steps+1, num_nodes).
#     Backward Euler: (I + dt*κ*K) u_{n+1} = u_n
#     (no source → F=0)
#     """
#     snapshots = [u0_flat]
#     u = u0_flat
#     op = lambda v: implicit_op(v, kappa)
#     for _ in range(steps):
#         # RHS = u_n  (F=0)
#         u, _ = jax_linalg.cg(op, u, x0=u, tol=1e-6, maxiter=1000)
#         snapshots.append(u)
#     return jnp.stack(snapshots)    # (steps+1, num_nodes)


# # ─────────────────────────────────────────
# # 4. Load Pre-Generated Training Data
# # ─────────────────────────────────────────
# DATA_FILE = OUT / 'training_data.pkl'

# if DATA_FILE.exists():
#     print(f"\n── Loading pre-generated data from {DATA_FILE} ──")
#     with open(DATA_FILE, 'rb') as f:
#         data = pickle.load(f)
#     U_train       = jnp.array(data['U_train'])
#     U_val         = jnp.array(data['U_val'])
#     all_snapshots = [jnp.array(s) for s in data['all_snapshots']]
#     val_snapshots = [jnp.array(s) for s in data['val_snapshots']]
#     train_params  = data['train_params']
#     val_params    = data['val_params']
#     traj_kappas   = data['traj_kappas']
#     val_kappas    = data['val_kappas']
#     traj_starts   = data['traj_starts']
#     N_TRAIN       = len(train_params)
#     N_VAL         = len(val_params)
#     print(f"   Training snapshots: {U_train.shape}")
#     print(f"   Validation snapshots: {U_val.shape}")
# else:
#     # Generate data if not found
#     N_TRAIN = 200
#     N_VAL   =  20
#     print(f"\n── Generating {N_TRAIN} training + {N_VAL} validation trajectories ──")
#     print(f"   Each: {NUM_STEPS+1} snapshots  →  "
#           f"Total snapshots ≈ {N_TRAIN*(NUM_STEPS+1):,}")

#     train_params = sample_trajectory_params(rng=42,        n_traj=N_TRAIN)
#     val_params   = sample_trajectory_params(rng=1337,      n_traj=N_VAL)

#     all_snapshots = []
#     traj_kappas   = []
#     traj_starts   = []

#     t0 = time.perf_counter()
#     for i, tp in enumerate(train_params):
#         u0   = make_gaussian_ic(tp['centers'], tp['amplitudes'], tp['widths'])
#         traj = run_fom(u0, tp['kappa'], NUM_STEPS)
#         traj_starts.append(len(all_snapshots))
#         all_snapshots.append(traj)
#         traj_kappas.append(tp['kappa'])
#         if (i+1) % 50 == 0:
#             elapsed = time.perf_counter() - t0
#             print(f"   Train {i+1}/{N_TRAIN}  ({elapsed:.0f}s elapsed)")

#     U_train = jnp.concatenate(all_snapshots, axis=0)
#     print(f"   Training snapshots: {U_train.shape}")

#     val_snapshots = []
#     val_kappas    = []
#     for i, vp in enumerate(val_params):
#         u0   = make_gaussian_ic(vp['centers'], vp['amplitudes'], vp['widths'])
#         traj = run_fom(u0, vp['kappa'], NUM_STEPS)
#         val_snapshots.append(traj)
#         val_kappas.append(vp['kappa'])
#         if (i+1) % 10 == 0:
#             print(f"   Val {i+1}/{N_VAL}")

#     U_val = jnp.concatenate(val_snapshots, axis=0)
#     print(f"   Validation snapshots: {U_val.shape}")
#     print(f"   Data generation: {time.perf_counter()-t0:.1f}s")

#     # Save generated data
#     data_to_save = {
#         'U_train': np.array(U_train),
#         'U_val': np.array(U_val),
#         'all_snapshots': [np.array(s) for s in all_snapshots],
#         'val_snapshots': [np.array(s) for s in val_snapshots],
#         'train_params': train_params,
#         'val_params': val_params,
#         'traj_kappas': traj_kappas,
#         'val_kappas': val_kappas,
#         'traj_starts': traj_starts,
#         'grid_config': {'N': N, 'L': L, 'dx': dx, 'dt': dt, 'NUM_STEPS': NUM_STEPS}
#     }
#     with open(DATA_FILE, 'wb') as f:
#         pickle.dump(data_to_save, f)
#     print(f"   Training data saved: {DATA_FILE}")

# # ─────────────────────────────────────────────────────────────────────
# # 5. Model Definition
# # ─────────────────────────────────────────────────────────────────────

# # ── Improvement 2: Per-sample amplitude normalisation ─────────────────
# # Splits encoding into shape (normalised) + scale (scalar).
# # Encoder only needs to learn spatial structure, not amplitude.
# AMP_EPS = 1e-6   # prevents division by zero for near-zero snapshots

# def normalise(u_flat):
#     """Returns (u_norm, scale). scale = max(|u|) + eps."""
#     scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
#     return u_flat / scale, scale

# def denormalise(u_norm, scale):
#     return u_norm * scale


# # ── Improvement 1: CoordConv input preparation ────────────────────────
# def make_coordconv_input(u_flat):
#     """
#     Concatenate (x,y,z) coordinate channels with the field.
#     Input to encoder: (N, N, N, 4) instead of (N, N, N, 1).
#     Gives conv layers direct positional awareness.
#     """
#     u_3d = u_flat.reshape(N, N, N)
#     # Stack field + 3 coordinate channels → (N, N, N, 4)
#     return jnp.concatenate([u_3d[..., None], COORD_GRID], axis=-1)


# # ── Improvement 3a: Residual block ────────────────────────────────────
# class ResBlock3D(nn.Module):
#     """
#     Conv3D residual block with GroupNorm.
#     out_feats channels, stride=1 (spatial size preserved).
#     Used after the initial strided downsampling to refine features.
#     """
#     out_feats:  int
#     num_groups: int = 8

#     @nn.compact
#     def __call__(self, x, training: bool = False):
#         # Main path
#         h = nn.GroupNorm(num_groups=self.num_groups)(x)
#         h = nn.leaky_relu(h, negative_slope=0.2)
#         h = nn.Conv(self.out_feats, kernel_size=(3,3,3),
#                     strides=(1,1,1), padding='SAME')(h)
#         h = nn.GroupNorm(num_groups=self.num_groups)(h)
#         h = nn.leaky_relu(h, negative_slope=0.2)
#         h = nn.Conv(self.out_feats, kernel_size=(3,3,3),
#                     strides=(1,1,1), padding='SAME')(h)
#         # Skip: 1×1×1 conv if channel count differs
#         if x.shape[-1] != self.out_feats:
#             x = nn.Conv(self.out_feats, kernel_size=(1,1,1))(x)
#         return x + h


# class AttentionPooling(nn.Module):
#     latent_dim: int

#     @nn.compact
#     def __call__(self, feat_map):
#         C      = feat_map.shape[-1]
#         tokens = feat_map.reshape(-1, C)
#         tokens = nn.Dense(self.latent_dim)(tokens)
#         query  = self.param('query', nn.initializers.normal(0.02), (self.latent_dim,))
#         scale  = jnp.sqrt(jnp.float32(self.latent_dim))
#         scores = jnp.einsum('td,d->t', tokens, query) / scale
#         w      = jax.nn.softmax(scores, axis=0)
#         return jnp.einsum('t,td->d', w, tokens)


# class Conv3DEncoder(nn.Module):
#     """
#     CoordConv encoder with residual blocks.

#     Input: (N,N,N,4)  — field + xyz coordinates
#     Strided convs: N → N/2 → N/4 → N/8   (spatial downsampling)
#     Residual blocks: refine features at each scale
#     AttentionPool: collapse spatial tokens → latent vector
#     """
#     latent_dim:   int
#     features:     Sequence[int] = (32, 64, 128)
#     pool_size:    int = 4
#     dropout_rate: float = 0.1
#     num_groups:   int = 8

#     @nn.compact
#     def __call__(self, x, training: bool = False):
#         # x: (N, N, N, 4)  — already has coord channels
#         h = x
#         for feat in self.features:
#             # Strided conv: halves spatial resolution
#             h = nn.Conv(feat, kernel_size=(3,3,3),
#                         strides=(2,2,2), padding='SAME')(h)
#             h = nn.GroupNorm(num_groups=self.num_groups)(h)
#             h = nn.leaky_relu(h, negative_slope=0.2)
#             # Residual refinement at this scale
#             h = ResBlock3D(feat, num_groups=self.num_groups)(h, training)
#             h = nn.Dropout(rate=self.dropout_rate,
#                            deterministic=not training)(h)

#         # Adaptive pool to fixed spatial size
#         H, W, D, C = h.shape
#         if H != self.pool_size:
#             h = jax.image.resize(h,
#                                  (self.pool_size, self.pool_size,
#                                   self.pool_size, C),
#                                  method='linear')
#         return AttentionPooling(self.latent_dim)(h)


# class SeparableDecoder(nn.Module):
#     """
#     CP-factored decoder — unchanged, maintains EQ compatibility.

#     Improvement 3b: deeper MLP with residual skip in z→h pathway.
#     """
#     latent_dim:  int
#     rank:        int = 512
#     grid_size:   int = 32
#     hidden_dims: Sequence[int] = (256, 512, 512)   # 3 layers vs 2

#     def setup(self):
#         self.hidden_layers = [nn.Dense(d) for d in self.hidden_dims]
#         self.to_rank       = nn.Dense(self.rank)
#         # Residual projection: maps z → hidden_dims[-1] for skip
#         self.z_proj        = nn.Dense(self.hidden_dims[-1])
#         init = nn.initializers.normal(0.01)
#         Ng   = self.grid_size
#         self.W_x  = self.param('W_x',  init, (self.rank, Ng))
#         self.W_y  = self.param('W_y',  init, (self.rank, Ng))
#         self.W_z  = self.param('W_z',  init, (self.rank, Ng))
#         self.bias = self.param('bias', nn.initializers.zeros, ())

#     def _mlp_body(self, z):
#         h = z
#         for i, layer in enumerate(self.hidden_layers):
#             h = nn.swish(layer(h))
#         # Residual skip from z: helps gradients flow back to encoder
#         h = h + self.z_proj(z)
#         return self.to_rank(h)

#     def __call__(self, z):
#         h    = self._mlp_body(z)
#         u_3d = jnp.einsum('r,ri,rj,rk->ijk', h, self.W_x, self.W_y, self.W_z)
#         return u_3d.flatten() + self.bias


# class ScalableAutoencoder(nn.Module):
#     """
#     Full autoencoder.

#     encode(u_flat) — applies amplitude normalisation then CoordConv encoder.
#                      Returns (z, scale) tuple.
#     decode(z, scale) — CP decoder then denormalise.
#     __call__ — encode then decode, returns reconstructed field.

#     The scale is NOT part of the latent code — it's a side-channel scalar.
#     This keeps the latent space purely about shape/position/dynamics.
#     """
#     latent_dim:    int
#     rank:          int = 512
#     grid_size:     int = 32
#     conv_features: Sequence[int] = (32, 64, 128)
#     hidden_dims:   Sequence[int] = (256, 512, 512)

#     def setup(self):
#         self.encoder = Conv3DEncoder(latent_dim=self.latent_dim,
#                                      features=self.conv_features)
#         self.decoder = SeparableDecoder(latent_dim=self.latent_dim,
#                                         rank=self.rank,
#                                         grid_size=self.grid_size,
#                                         hidden_dims=self.hidden_dims)

#     def encode(self, u_flat, training=False):
#         """Returns (z, scale). Scale stored separately — not in latent code."""
#         u_norm, scale   = normalise(u_flat)
#         coord_input     = make_coordconv_input(u_norm)   # (N,N,N,4)
#         z               = self.encoder(coord_input, training=training)
#         return z, scale

#     def decode(self, z, scale):
#         """Decode latent z, then restore amplitude via scale."""
#         u_norm = self.decoder(z)
#         return denormalise(u_norm, scale)

#     def decode_normalised(self, z):
#         """Decode without scale — used during EQ phase where scale is handled externally."""
#         return self.decoder(z)

#     def __call__(self, u_flat, training=False):
#         z, scale = self.encode(u_flat, training=training)
#         return self.decode(z, scale)


# # ─────────────────────────────────────────
# # 6. Model Init
# # ─────────────────────────────────────────
# k_dim = 32
# RANK  = 512

# model = ScalableAutoencoder(
#     latent_dim    = k_dim,
#     rank          = RANK,
#     grid_size     = N,
#     conv_features = (32, 64, 128),
#     hidden_dims   = (256, 512, 512),
# )

# key       = jax.random.PRNGKey(0)
# variables = model.init(
#     {'params': key, 'dropout': jax.random.PRNGKey(1)},
#     U_train[0], training=True
# )
# params = variables['params']

# n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
# print(f"\n── Model: {n_params:,} parameters  (latent_dim={k_dim}, rank={RANK}) ──")
# print(f"   Encoder: CoordConv + ResBlocks + GroupNorm")
# print(f"   Decoder: CP rank={RANK}, MLP depth=3 + residual skip")

# # ─────────────────────────────────────────
# # 7. Training
# # ─────────────────────────────────────────
# BATCH_SIZE = 64
# NUM_EPOCHS = 20_000
# LOG_EVERY  = 1_000

# schedule = optax.warmup_cosine_decay_schedule(
#     init_value=0., peak_value=1e-3,
#     warmup_steps=500, decay_steps=NUM_EPOCHS, end_value=1e-5
# )
# tx        = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
# opt_state = tx.init(params)

# key = jax.random.PRNGKey(2)


# # ── Per-sample relative loss ────────────────────────────────────────────
# # Pure relative MSE — amplitude normalisation already handles scale,
# # so gradient loss term is no longer needed. Simpler and more stable.
# REL_EPS = 1e-6

# def per_sample_loss(u_true, u_pred):
#     """Relative L2 loss — normalised by true field energy."""
#     diff    = u_true - u_pred
#     norm_sq = jnp.dot(u_true, u_true) + REL_EPS
#     return jnp.dot(diff, diff) / norm_sq


# @jax.jit
# def train_step(params, opt_state, batch, key):
#     drop_key, aug_key = jax.random.split(key)

#     def loss_fn(p):
#         # Linear scaling augmentation — exact symmetry of heat equation
#         scales    = jax.random.uniform(aug_key, (batch.shape[0], 1),
#                                        minval=0.5, maxval=2.0)
#         aug_batch = batch * scales

#         # Amplitude normalisation happens INSIDE model.apply,
#         # so augmentation scale is handled automatically.
#         preds = jax.vmap(
#             lambda u: model.apply(
#                 {'params': p}, u, training=True,
#                 rngs={'dropout': drop_key}
#             )
#         )(aug_batch)

#         losses = jax.vmap(per_sample_loss)(aug_batch, preds)
#         return jnp.mean(losses)

#     loss, grads         = jax.value_and_grad(loss_fn)(params)
#     updates, new_opt_st = tx.update(grads, opt_state, params)
#     new_params          = optax.apply_updates(params, updates)
#     return new_params, new_opt_st, loss


# @jax.jit
# def eval_step(params, batch):
#     preds  = jax.vmap(
#         lambda u: model.apply({'params': params}, u, training=False)
#     )(batch)
#     losses = jax.vmap(per_sample_loss)(batch, preds)
#     return jnp.mean(losses)


# print(f"\n── Training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}) ──")

# n_train        = len(U_train)
# train_losses   = []
# val_losses     = []
# best_val       = float('inf')
# best_params    = params
# patience       = 8
# patience_count = 0
# t0             = time.perf_counter()

# for epoch in range(NUM_EPOCHS + 1):
#     key, subkey = jax.random.split(key)
#     idx   = jax.random.choice(subkey, n_train, shape=(BATCH_SIZE,), replace=False)
#     batch = U_train[idx]
#     params, opt_state, loss = train_step(params, opt_state, batch, subkey)

#     if epoch % LOG_EVERY == 0:
#         key, vkey = jax.random.split(key)
#         v_idx  = jax.random.choice(vkey, len(U_val),
#                                    shape=(BATCH_SIZE,), replace=False)
#         v_loss = float(eval_step(params, U_val[v_idx]))
#         train_losses.append((epoch, float(loss)))
#         val_losses.append((epoch, v_loss))
#         print(f"  Epoch {epoch:5d} | train {float(loss):.4e} | "
#               f"val {v_loss:.4e} | {time.perf_counter()-t0:.0f}s")

#         if v_loss < best_val:
#             best_val       = v_loss
#             best_params    = params
#             patience_count = 0
#         else:
#             patience_count += 1
#             if patience_count >= patience:
#                 print(f"\n  Early stop at epoch {epoch}  "
#                       f"(best val={best_val:.4e})")
#                 break

# params = best_params
# print(f"\n  Best val loss: {best_val:.4e}")

# # ─────────────────────────────────────────
# # 8. Save Checkpoint
# # ─────────────────────────────────────────
# ckpt = {
#     'params': params,
#     'model_cfg': dict(
#         latent_dim    = k_dim,
#         rank          = RANK,
#         grid_size     = N,
#         conv_features = (32, 64, 128),
#         hidden_dims   = (256, 512, 512),
#     ),
#     'train_meta': dict(
#         n_train     = N_TRAIN,
#         num_steps   = NUM_STEPS,
#         dt          = dt,
#         traj_kappas = traj_kappas,
#         traj_starts = traj_starts,
#     )
# }
# CKPT_PATH = SCRIPT_DIR / 'checkpoint.pkl'
# with open(CKPT_PATH, 'wb') as f:
#     pickle.dump(ckpt, f)
# print(f"  Checkpoint: {CKPT_PATH}")

# # ─────────────────────────────────────────
# # 9. Diagnostic Plots
# # ─────────────────────────────────────────
# def encode(u):
#     z, scale = model.apply({'params': params}, u, training=False,
#                            method=model.encode)
#     return z, scale

# def decode(z, scale):
#     return model.apply({'params': params}, z, scale, method=model.decode)

# def reconstruct(u):
#     return model.apply({'params': params}, u, training=False)

# # Loss curve
# ep_t, lo_t = zip(*train_losses)
# ep_v, lo_v = zip(*val_losses)
# fig, ax = plt.subplots(figsize=(9, 4))
# ax.semilogy(ep_t, lo_t, label='Train', color='#1f77b4', lw=2)
# ax.semilogy(ep_v, lo_v, label='Val',   color='#ff7f0e', lw=2, ls='--')
# ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L2 Loss (log)')
# ax.set_title('Heat AE (CoordConv + ResBlocks + AmpNorm) — Training Curve')
# ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
# plt.tight_layout()
# plt.savefig(OUT / 'loss_curve.png', dpi=150); plt.close()

# # Reconstruction samples — 3 trajectories × 3 time points
# fig, axes = plt.subplots(3, 6, figsize=(18, 9))
# sample_trajs = [0, N_TRAIN//2, N_TRAIN-1]
# time_idxs    = [0, NUM_STEPS//2, NUM_STEPS]
# mid          = N // 2

# recon_errors = []
# for row, ti in enumerate(sample_trajs):
#     traj = all_snapshots[ti]
#     kap  = traj_kappas[ti]
#     for col, t_idx in enumerate(time_idxs):
#         u_true = traj[t_idx]
#         u_rec  = reconstruct(u_true)
#         err    = float(jnp.linalg.norm(u_rec - u_true)
#                        / (jnp.linalg.norm(u_true) + 1e-10))
#         recon_errors.append(err)
#         u_t3d  = np.array(u_true).reshape(N,N,N)
#         u_r3d  = np.array(u_rec ).reshape(N,N,N)
#         vmax   = max(float(u_t3d[:,:,mid].max()), 1e-8)
#         kw     = dict(origin='lower', cmap='magma',
#                       vmin=0, vmax=vmax, aspect='auto')
#         axes[row, col*2  ].imshow(u_t3d[:,:,mid].T, **kw)
#         axes[row, col*2  ].set_title(f'FOM t={t_idx*dt:.2f}', fontsize=8)
#         axes[row, col*2  ].axis('off')
#         axes[row, col*2+1].imshow(u_r3d[:,:,mid].T, **kw)
#         axes[row, col*2+1].set_title(f'Rec e={err:.2e}', fontsize=8)
#         axes[row, col*2+1].axis('off')
#     axes[row, 0].set_ylabel(f'k={kap:.3f}', fontsize=9)

# print(f"\n  Mean reconstruction error: {np.mean(recon_errors):.4e}")
# print(f"  Max  reconstruction error: {np.max(recon_errors):.4e}")

# fig.suptitle('Heat AE — FOM vs Decoded (z=mid slice)', fontsize=12)
# plt.tight_layout()
# plt.savefig(OUT / 'reconstruction_samples.png', dpi=150, bbox_inches='tight')
# plt.close()

# print(f"  Plots saved to {OUT}/")
# print("\n=== Training complete — run nmrom_heat.py next ===")


"""
train_heat_ae.py
────────────────
ScalableAutoencoder for 3D Heat Equation NM-ROM (Gaussian IC case).

Loads data generated by generate_heat_data.py.

Fixes over previous version:
  1. Correct data path: data/training_data_{N}.pkl
  2. Augmentation replaced with noise injection on normalised input
     (linear scaling was cancelled by per-sample normalise() — zero effect)
  3. Val evaluated on full val set each LOG_EVERY, not a 64-sample subset
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
from pathlib import Path
from typing import Sequence

# ─────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────
N = 32   # must match the grid used in generate_heat_data.py

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_FILE  = Path('/home/tahmid/Development/Gauss-Newton-Embedding-Solver /Research/Heat_3D/plots/training_data.pkl')
CKPT_PATH  = SCRIPT_DIR / 'checkpoint.pkl'
PLOT_DIR   = SCRIPT_DIR / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = SCRIPT_DIR / 'training.log'
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log      = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = TeeLogger(LOG_FILE)

# ─────────────────────────────────────────
# 1. Grid
# ─────────────────────────────────────────
num_nodes = N ** 3
L         = 1.0
dx        = L / (N - 1)
dt        = 0.005
NUM_STEPS = 50

x_sp = jnp.linspace(0, L, N)
y_sp = jnp.linspace(0, L, N)
z_sp = jnp.linspace(0, L, N)
X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')

COORD_GRID = jnp.stack([
    2.0*X/L - 1.0,
    2.0*Y/L - 1.0,
    2.0*Z/L - 1.0,
], axis=-1)   # (N, N, N, 3)

mask_3d = jnp.ones((N,N,N))
mask_3d = mask_3d.at[0,:,:].set(0.).at[-1,:,:].set(0.)
mask_3d = mask_3d.at[:,0,:].set(0.).at[:,-1,:].set(0.)
mask_3d = mask_3d.at[:,:,0].set(0.).at[:,:,-1].set(0.)
mask    = mask_3d.flatten()

print(f"Grid: {N}³ = {num_nodes:,} nodes  |  dt={dt}  T={dt*NUM_STEPS:.3f}s")

# ─────────────────────────────────────────
# 2. Load Data
# ─────────────────────────────────────────
print(f"\n── Loading data from {DATA_FILE} ──")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        f"Run:  python generate_heat_data.py --grid {N}"
    )

with open(DATA_FILE, 'rb') as f:
    data = pickle.load(f)

# Validate grid config matches
cfg = data.get('grid_config', {})
if cfg:
    assert cfg['N']  == N,        f"Grid mismatch: data has N={cfg['N']}, script has N={N}"
    assert cfg['dt'] == dt,       f"dt mismatch: data={cfg['dt']}, script={dt}"
    assert cfg['NUM_STEPS'] == NUM_STEPS, \
        f"NUM_STEPS mismatch: data={cfg['NUM_STEPS']}, script={NUM_STEPS}"
    print(f"   Grid config validated: N={N}, dt={dt}, T={dt*NUM_STEPS:.3f}s")

U_train       = jnp.array(data['U_train'])
U_val         = jnp.array(data['U_val'])
all_snapshots = [jnp.array(s) for s in data['all_snapshots']]
val_snapshots = [jnp.array(s) for s in data['val_snapshots']]
train_params  = data['train_params']
val_params    = data['val_params']
traj_kappas   = data['traj_kappas']
val_kappas    = data['val_kappas']
traj_starts   = data['traj_starts']
N_TRAIN       = len(train_params)
N_VAL         = len(val_params)

print(f"   Train: {U_train.shape}  ({N_TRAIN} trajectories × {NUM_STEPS+1} steps)")
print(f"   Val:   {U_val.shape}   ({N_VAL} trajectories × {NUM_STEPS+1} steps)")

# ─────────────────────────────────────────────────────────────────────
# 3. Model Definition
# ─────────────────────────────────────────────────────────────────────
AMP_EPS = 1e-6

def normalise(u_flat):
    scale = jnp.max(jnp.abs(u_flat)) + AMP_EPS
    return u_flat / scale, scale

def make_coordconv_input(u_flat):
    u_3d = u_flat.reshape(N, N, N)
    return jnp.concatenate([u_3d[..., None], COORD_GRID], axis=-1)


class ResBlock3D(nn.Module):
    out_feats:  int
    num_groups: int = 8
    @nn.compact
    def __call__(self, x, training: bool = False):
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Conv(self.out_feats, (3,3,3), strides=(1,1,1), padding='SAME')(h)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = nn.Conv(self.out_feats, (3,3,3), strides=(1,1,1), padding='SAME')(h)
        if x.shape[-1] != self.out_feats:
            x = nn.Conv(self.out_feats, (1,1,1))(x)
        return x + h


class MultiHeadAttentionPooling(nn.Module):
    """
    Multiple query vectors — one per head — each can focus on a
    different spatial region (Gaussian blob). Outputs are concatenated
    and projected to latent_dim.

    For n_gaussians ∈ {1,2,3}, use n_heads=4 to cover all cases plus
    one head for background/global context.
    """
    latent_dim: int
    n_heads:    int = 4

    @nn.compact
    def __call__(self, feat_map):
        C      = feat_map.shape[-1]
        tokens = feat_map.reshape(-1, C)                   # (T, C)
        tokens = nn.Dense(self.latent_dim)(tokens)         # (T, D)

        # One query vector per head — each learns to attend differently
        queries = self.param('queries',
                             nn.initializers.normal(0.02),
                             (self.n_heads, self.latent_dim))  # (H, D)

        scale   = jnp.sqrt(jnp.float32(self.latent_dim))
        scores  = jnp.einsum('hd,td->ht', queries, tokens) / scale  # (H, T)
        weights = jax.nn.softmax(scores, axis=-1)                    # (H, T)
        heads   = jnp.einsum('ht,td->hd', weights, tokens)          # (H, D)

        # Project concatenated heads → latent_dim
        return nn.Dense(self.latent_dim)(heads.flatten())  # (H*D → D)


class Conv3DEncoder(nn.Module):
    latent_dim:   int
    features:     Sequence[int] = (32, 64, 128)
    pool_size:    int = 4
    dropout_rate: float = 0.1
    num_groups:   int = 8
    n_heads:      int = 4    # one per potential Gaussian blob + 1
    @nn.compact
    def __call__(self, x, training: bool = False):
        h = x   # (N, N, N, 4)
        for feat in self.features:
            h = nn.Conv(feat, (3,3,3), strides=(2,2,2), padding='SAME')(h)
            h = nn.GroupNorm(num_groups=self.num_groups)(h)
            h = nn.leaky_relu(h, negative_slope=0.2)
            h = ResBlock3D(feat, num_groups=self.num_groups)(h, training)
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        H, W, D, C = h.shape
        if H != self.pool_size:
            h = jax.image.resize(h,
                                 (self.pool_size, self.pool_size,
                                  self.pool_size, C),
                                 method='linear')
        # Multi-head pooling — each head attends to a different blob
        return MultiHeadAttentionPooling(self.latent_dim, n_heads=self.n_heads)(h)


class SeparableDecoder(nn.Module):
    latent_dim:  int
    rank:        int = 256
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
    rank:          int = 256
    grid_size:     int = 32
    conv_features: Sequence[int] = (32, 64, 128)
    hidden_dims:   Sequence[int] = (256, 512, 512)
    def setup(self):
        self.encoder = Conv3DEncoder(latent_dim=self.latent_dim,
                                     features=self.conv_features)
        self.decoder = SeparableDecoder(latent_dim=self.latent_dim,
                                        rank=self.rank,
                                        grid_size=self.grid_size,
                                        hidden_dims=self.hidden_dims)
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
# 4. Model Init
# ─────────────────────────────────────────
k_dim = 32    # was 20 — need headroom for 3-Gaussian × 5 params + κ
RANK  = 256   # was 128 — with 3 Gaussians and continuous positions, more rank helps

model = ScalableAutoencoder(
    latent_dim    = k_dim,
    rank          = RANK,
    grid_size     = N,
    conv_features = (32, 64, 128),
    hidden_dims   = (256, 512, 512),
)

# Standalone encoder/decoder for direct .apply() calls in train_step
encoder_module = Conv3DEncoder(latent_dim=k_dim, features=(32, 64, 128), n_heads=4)
decoder_module = SeparableDecoder(latent_dim=k_dim, rank=RANK, grid_size=N,
                                   hidden_dims=(256, 512, 512))

key       = jax.random.PRNGKey(0)
variables = model.init(
    {'params': key, 'dropout': jax.random.PRNGKey(1)},
    U_train[0], training=True
)
params   = variables['params']
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\n── Model: {n_params:,} parameters  (latent_dim={k_dim}, rank={RANK}) ──")

# ─────────────────────────────────────────
# 5. Loss & Augmentation
# ─────────────────────────────────────────
REL_EPS    = 1e-6
NOISE_STD  = 0.03   # noise on normalised input


def augment_3d(u_flat, key):
    """
    Random spatial augmentation for 3D fields:
    - Random flips along each axis (8 combinations)
    - Random 90° rotations in xy, xz, yz planes
    
    Gaussians are symmetric, so these are exact symmetries of the data.
    """
    u_3d = u_flat.reshape(N, N, N)
    
    # Split key for independent random choices
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    # Random flips (each axis independently)
    u_3d = jax.lax.cond(jax.random.uniform(k1) > 0.5,
                        lambda x: jnp.flip(x, axis=0), lambda x: x, u_3d)
    u_3d = jax.lax.cond(jax.random.uniform(k2) > 0.5,
                        lambda x: jnp.flip(x, axis=1), lambda x: x, u_3d)
    u_3d = jax.lax.cond(jax.random.uniform(k3) > 0.5,
                        lambda x: jnp.flip(x, axis=2), lambda x: x, u_3d)
    
    # Random 90° rotations using jax.lax.switch (k must be static in jnp.rot90)
    def rot_xy_0(x): return x
    def rot_xy_1(x): return jnp.rot90(x, k=1, axes=(0, 1))
    def rot_xy_2(x): return jnp.rot90(x, k=2, axes=(0, 1))
    def rot_xy_3(x): return jnp.rot90(x, k=3, axes=(0, 1))
    
    def rot_xz_0(x): return x
    def rot_xz_1(x): return jnp.rot90(x, k=1, axes=(0, 2))
    def rot_xz_2(x): return jnp.rot90(x, k=2, axes=(0, 2))
    def rot_xz_3(x): return jnp.rot90(x, k=3, axes=(0, 2))
    
    def rot_yz_0(x): return x
    def rot_yz_1(x): return jnp.rot90(x, k=1, axes=(1, 2))
    def rot_yz_2(x): return jnp.rot90(x, k=2, axes=(1, 2))
    def rot_yz_3(x): return jnp.rot90(x, k=3, axes=(1, 2))
    
    n_rot_xy = jax.random.randint(k4, (), 0, 4)
    n_rot_xz = jax.random.randint(k5, (), 0, 4)
    n_rot_yz = jax.random.randint(k6, (), 0, 4)
    
    u_3d = jax.lax.switch(n_rot_xy, [rot_xy_0, rot_xy_1, rot_xy_2, rot_xy_3], u_3d)
    u_3d = jax.lax.switch(n_rot_xz, [rot_xz_0, rot_xz_1, rot_xz_2, rot_xz_3], u_3d)
    u_3d = jax.lax.switch(n_rot_yz, [rot_yz_0, rot_yz_1, rot_yz_2, rot_yz_3], u_3d)
    
    return u_3d.flatten()

def per_sample_loss(u_true, u_pred):
    diff    = u_true - u_pred
    norm_sq = jnp.dot(u_true, u_true) + REL_EPS
    return jnp.dot(diff, diff) / norm_sq

# ─────────────────────────────────────────────────────────────────────
# 6. Training
#
# Augmentation fix:
#   OLD: batch * α  → cancelled by normalise() inside encode()
#        encoder always sees u/max|u| regardless of α. Zero effect.
#   NEW: inject Gaussian noise on the NORMALISED input field.
#        Noise is injected BEFORE the encoder sees it, so it actually
#        forces the encoder to learn robust representations.
#        The decoder still trains to reconstruct the clean u.
#
# Val eval fix:
#   OLD: 64 random samples from 1020 val snapshots → 6%, high variance
#   NEW: full val set every LOG_EVERY epochs
# ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 20_000
LOG_EVERY  = 1_000

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0., peak_value=1e-3,
    warmup_steps=500, decay_steps=NUM_EPOCHS, end_value=1e-5
)
tx        = optax.adamw(learning_rate=schedule, weight_decay=1e-3)  # 10x stronger weight decay
opt_state = tx.init(params)
key       = jax.random.PRNGKey(2)


@jax.jit
def train_step(params, opt_state, batch, key):
    drop_key, noise_key, aug_key = jax.random.split(key, 3)

    def loss_fn(p):
        def forward_one(u, nkey):
            # Split key for augmentation and noise
            aug_k, noise_k = jax.random.split(nkey)
            
            # 1. Apply spatial augmentation (flips + rotations)
            u_aug = augment_3d(u, aug_k)
            
            # 2. Normalise to get the shape the encoder should learn
            u_norm, scale = normalise(u_aug)

            # 3. Inject noise on the normalised field
            noise    = NOISE_STD * jax.random.normal(noise_k, u_norm.shape)
            u_noisy  = u_norm + noise

            # 4. Encode the noisy normalised field
            coord_in = make_coordconv_input(u_noisy)
            z        = encoder_module.apply(
                {'params': p['encoder']}, coord_in, training=True,
                rngs={'dropout': drop_key}
            )

            # 5. Decode and restore scale — target is the AUGMENTED field
            u_pred = decoder_module.apply({'params': p['decoder']}, z) * scale

            return u_pred, u_aug

        # Split keys per sample
        sample_keys = jax.random.split(noise_key, batch.shape[0])
        preds, targets = jax.vmap(forward_one)(batch, sample_keys)
        losses = jax.vmap(per_sample_loss)(targets, preds)
        return jnp.mean(losses)

    loss, grads         = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_st = tx.update(grads, opt_state, params)
    new_params          = optax.apply_updates(params, updates)
    return new_params, new_opt_st, loss


@jax.jit
def eval_step(params, batch):
    """Clean evaluation — no noise, no dropout."""
    preds  = jax.vmap(
        lambda u: model.apply({'params': params}, u, training=False)
    )(batch)
    losses = jax.vmap(per_sample_loss)(batch, preds)
    return jnp.mean(losses)


# ─────────────────────────────────────────
# 7. Training Loop
# ─────────────────────────────────────────
print(f"\n── Training ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}) ──")
print(f"   Noise augmentation: std={NOISE_STD} on normalised input")
print(f"   Val evaluated on full val set ({len(U_val)} snapshots) every {LOG_EVERY} epochs")

n_train        = len(U_train)
train_losses   = []
val_losses     = []
best_val       = float('inf')
best_params    = params
patience       = 8
patience_count = 0
t0             = time.perf_counter()

for epoch in range(NUM_EPOCHS + 1):
    key, subkey = jax.random.split(key)
    idx   = jax.random.choice(subkey, n_train, shape=(BATCH_SIZE,), replace=False)
    params, opt_state, loss = train_step(params, opt_state, U_train[idx], subkey)

    if epoch % LOG_EVERY == 0:
        # Evaluate on the full val set in minibatches to avoid OOM
        val_losses_batch = []
        n_val = len(U_val)
        for start in range(0, n_val, 256):
            vb = U_val[start:start+256]
            val_losses_batch.append(float(eval_step(params, vb)) * len(vb))
        v_loss = sum(val_losses_batch) / n_val

        train_losses.append((epoch, float(loss)))
        val_losses.append((epoch, v_loss))
        print(f"  Epoch {epoch:5d} | train {float(loss):.4e} | "
              f"val {v_loss:.4e} | {time.perf_counter()-t0:.0f}s")

        if v_loss < best_val:
            best_val       = v_loss
            best_params    = params
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stop at epoch {epoch}  (best val={best_val:.4e})")
                break

params = best_params
print(f"\n  Best val loss: {best_val:.4e}")

# ─────────────────────────────────────────
# 8. Save Checkpoint
# ─────────────────────────────────────────
ckpt = {
    'params': params,
    'model_cfg': dict(
        latent_dim    = k_dim,
        rank          = RANK,
        grid_size     = N,
        conv_features = (32, 64, 128),
        hidden_dims   = (256, 512, 512),
    ),
    'train_meta': dict(
        n_train     = N_TRAIN,
        num_steps   = NUM_STEPS,
        dt          = dt,
        traj_kappas = traj_kappas,
        val_kappas  = val_kappas,
        traj_starts = traj_starts,
    )
}
with open(CKPT_PATH, 'wb') as f:
    pickle.dump(ckpt, f)
print(f"  Checkpoint: {CKPT_PATH}")

# ─────────────────────────────────────────
# 9. Diagnostic Plots
# ─────────────────────────────────────────
def reconstruct(u):
    return model.apply({'params': params}, u, training=False)

# Loss curve
ep_t, lo_t = zip(*train_losses)
ep_v, lo_v = zip(*val_losses)
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(ep_t, lo_t, label='Train', color='#1f77b4', lw=2)
ax.semilogy(ep_v, lo_v, label='Val (full)',   color='#ff7f0e', lw=2, ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L2 Loss (log)')
ax.set_title('Heat AE (Gaussian) — Training Curve')
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'loss_curve.png', dpi=150); plt.close()

# Reconstruction samples
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
sample_trajs = [0, N_TRAIN//2, N_TRAIN-1]
time_idxs    = [0, NUM_STEPS//2, NUM_STEPS]
mid          = N // 2
recon_errors = []

for row, ti in enumerate(sample_trajs):
    traj = all_snapshots[ti]
    kap  = traj_kappas[ti]
    for col, t_idx in enumerate(time_idxs):
        u_true = traj[t_idx]
        u_rec  = reconstruct(u_true)
        err    = float(jnp.linalg.norm(u_rec - u_true)
                       / (jnp.linalg.norm(u_true) + 1e-10))
        recon_errors.append(err)
        u_t3d  = np.array(u_true).reshape(N,N,N)
        u_r3d  = np.array(u_rec ).reshape(N,N,N)
        vmax   = max(float(u_t3d[:,:,mid].max()), 1e-8)
        kw     = dict(origin='lower', cmap='magma', vmin=0, vmax=vmax, aspect='auto')
        axes[row, col*2  ].imshow(u_t3d[:,:,mid].T, **kw)
        axes[row, col*2  ].set_title(f'FOM t={t_idx*dt:.2f}', fontsize=8)
        axes[row, col*2  ].axis('off')
        axes[row, col*2+1].imshow(u_r3d[:,:,mid].T, **kw)
        axes[row, col*2+1].set_title(f'Rec e={err:.2e}', fontsize=8)
        axes[row, col*2+1].axis('off')
    axes[row, 0].set_ylabel(f'k={kap:.3f}', fontsize=9)

print(f"\n  Mean reconstruction error: {np.mean(recon_errors):.4e}")
print(f"  Max  reconstruction error: {np.max(recon_errors):.4e}")
fig.suptitle('Heat AE (Gaussian) — FOM vs Decoded (z=mid slice)', fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'reconstruction_samples.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Plots → {PLOT_DIR}/")
print("\n=== Training complete ===")