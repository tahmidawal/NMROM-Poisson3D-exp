# # """
# # POD + Fourier-Featured Dense Autoencoder for 3D Poisson
# # ========================================================

# # Pipeline:
# #   u in R^{N^3}  --POD-->  c in R^r  --normalize-->  cn  --AE-->  cn_hat  --denorm-->  c_hat  --POD lift-->  u_hat in R^{N^3}

# # Two models trained side-by-side:
# #   Baseline:  POD -> Dense AE -> POD lift
# #   Fourier:   POD -> Fourier lift -> Dense AE -> POD lift

# # The Fourier lift breaks the spectral bias of MLPs by injecting
# # sin/cos features of the POD coefficients at multiple frequency bands.

# # Sections:
# #   1. Configuration
# #   2. 3D Poisson setup (operators, forcing, analytical solution)
# #   3. Generate FOM training snapshots  (k in {1,...,8}^3 = 512 solves)
# #   4. POD compression  (truncated SVD)
# #   5. Model definitions  (BaselineAE, FourierAE)
# #   6. Shared training loop
# #   7. Train both models
# #   8. Full-pipeline evaluation (training set + unseen cases)
# #   9. Diagnostic plots
# #   10. Summary
# # """

# # import jax
# # import jax.numpy as jnp
# # import flax.linen as nn
# # import optax
# # import jax.scipy.sparse.linalg as jax_linalg
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import time
# # import json
# # from pathlib import Path
# # import orbax.checkpoint as ocp

# # # ============================================================================
# # # 1. CONFIGURATION
# # # ============================================================================

# # OUTPUT_DIR = Path(__file__).parent / "plots" / "baseline_ae"
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "baseline_ae"
# # CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# # N = 32                     # Grid points per axis  (32^3 = 32768 DOF)
# # K_MAX = 8                  # Max wave number in each direction
# # LATENT_DIM = 20            # Bottleneck dimension
# # POD_ENERGY_THRESHOLD = 0.9999  # Keep modes capturing this fraction of energy

# # # Fourier feature settings
# # NUM_OCTAVES = 4            # Number of frequency doublings
# # BASE_FREQ = 1.0            # Lowest frequency band  -> bands: [1, 2, 4, 8]

# # # Training settings
# # EPOCHS = 15_000
# # BATCH_SIZE = 32
# # LR_INIT = 1e-3
# # LR_DECAY_STEPS = 3000
# # LR_DECAY_RATE = 0.85

# # # Derived constants
# # L = 1.0                    # Domain is [0, L]^3
# # dx = L / (N - 1)
# # num_nodes = N ** 3

# # print(f"{'=' * 70}")
# # print(f"  POD + Baseline Dense Autoencoder  --  3D Poisson")
# # print(f"  Grid: {N}^3 = {num_nodes:,} DOF   |   k in [1,{K_MAX}]   |   Latent: {LATENT_DIM}")
# # print(f"{'=' * 70}\n")


# # # ============================================================================
# # # 2. 3D POISSON -- Domain, Operators, Analytical Solution
# # # ============================================================================

# # # Spatial coordinates on [0, 1]^3
# # x_1d = jnp.linspace(0, L, N)
# # y_1d = jnp.linspace(0, L, N)
# # z_1d = jnp.linspace(0, L, N)
# # X, Y, Z = jnp.meshgrid(x_1d, y_1d, z_1d, indexing="ij")  # (N, N, N) each

# # # Boundary mask:  1 in interior,  0 on all 6 faces
# # _mask_3d = jnp.ones((N, N, N))
# # _mask_3d = _mask_3d.at[ 0, :, :].set(0.0).at[-1, :, :].set(0.0)
# # _mask_3d = _mask_3d.at[ :, 0, :].set(0.0).at[ :,-1, :].set(0.0)
# # _mask_3d = _mask_3d.at[ :, :, 0].set(0.0).at[ :, :,-1].set(0.0)
# # bc_mask = _mask_3d.flatten()  # (N^3,)


# # def laplacian_3d(u_flat):
# #     """Apply  -nabla^2  via 7-point finite differences with Dirichlet u=0 BCs."""
# #     u = u_flat.reshape((N, N, N))
# #     lap = jnp.zeros_like(u)

# #     # Interior: 7-point stencil
# #     lap = lap.at[1:-1, 1:-1, 1:-1].set(
# #         (6.0 * u[1:-1, 1:-1, 1:-1]
# #          - u[:-2, 1:-1, 1:-1] - u[2:, 1:-1, 1:-1]
# #          - u[1:-1, :-2, 1:-1] - u[1:-1, 2:, 1:-1]
# #          - u[1:-1, 1:-1, :-2] - u[1:-1, 1:-1, 2:]) / dx**2
# #     )

# #     # Boundary: identity rows enforce u = 0
# #     lap = lap.at[ 0, :, :].set(u[ 0, :, :])
# #     lap = lap.at[-1, :, :].set(u[-1, :, :])
# #     lap = lap.at[ :, 0, :].set(u[ :, 0, :])
# #     lap = lap.at[ :,-1, :].set(u[ :,-1, :])
# #     lap = lap.at[ :, :, 0].set(u[ :, :, 0])
# #     lap = lap.at[ :, :,-1].set(u[ :, :,-1])

# #     return lap.flatten()


# # def forcing_3d(k1, k2, k3):
# #     """F = 10 * sin(k1*pi*x) * sin(k2*pi*y) * sin(k3*pi*z), zeroed on boundaries."""
# #     F = (10.0
# #          * jnp.sin(k1 * jnp.pi * X)
# #          * jnp.sin(k2 * jnp.pi * Y)
# #          * jnp.sin(k3 * jnp.pi * Z))
# #     F = F.at[ 0,:,:].set(0.).at[-1,:,:].set(0.)
# #     F = F.at[ :,0,:].set(0.).at[ :,-1,:].set(0.)
# #     F = F.at[ :,:,0].set(0.).at[ :,:,-1].set(0.)
# #     return F.flatten()


# # def analytical_3d(k1, k2, k3):
# #     """Exact solution of -nabla^2 u = F for sinusoidal forcing (integer k only)."""
# #     coeff = 10.0 / ((k1**2 + k2**2 + k3**2) * jnp.pi**2)
# #     return (coeff
# #             * jnp.sin(k1 * jnp.pi * X)
# #             * jnp.sin(k2 * jnp.pi * Y)
# #             * jnp.sin(k3 * jnp.pi * Z)).flatten()


# # def fom_solve(F_vec, tol=1e-7, maxiter=5000):
# #     """Full-order CG solve:  laplacian_3d(u) = F."""
# #     u, _ = jax_linalg.cg(laplacian_3d, F_vec,
# #                           x0=jnp.zeros(num_nodes), tol=tol, maxiter=maxiter)
# #     return u


# # # ============================================================================
# # # 3. GENERATE FOM TRAINING SNAPSHOTS
# # # ============================================================================

# # print("--- Stage 1: Generating FOM snapshots ---")

# # # All integer triples k1, k2, k3 in {1,...,K_MAX} -> K_MAX^3 snapshots
# # train_ks = [(k1, k2, k3)
# #             for k1 in range(1, K_MAX + 1)
# #             for k2 in range(1, K_MAX + 1)
# #             for k3 in range(1, K_MAX + 1)]
# # n_snap = len(train_ks)
# # print(f"   {n_snap} parameter combos  (k in [1, {K_MAX}]^3)")

# # # Frequency magnitude for each snapshot (used in analysis plots)
# # k_mags = np.array([np.sqrt(k1**2 + k2**2 + k3**2) for k1, k2, k3 in train_ks])

# # U_list = []
# # t0 = time.perf_counter()
# # for i, (k1, k2, k3) in enumerate(train_ks):
# #     U_list.append(fom_solve(forcing_3d(k1, k2, k3)))
# #     if (i + 1) % 50 == 0 or i + 1 == n_snap:
# #         print(f"      [{i+1:3d}/{n_snap}]  {time.perf_counter() - t0:.1f}s elapsed")

# # U_train = jnp.stack(U_list)  # (n_snap, N^3)
# # print(f"   Snapshot matrix: {U_train.shape}  ({U_train.nbytes / 1e6:.0f} MB)\n")


# # # ============================================================================
# # # 4. POD COMPRESSION  (Truncated SVD)
# # # ============================================================================

# # print("--- Stage 2: POD Compression ---")

# # # Centre the data (subtract mean snapshot)
# # U_mean = jnp.mean(U_train, axis=0)               # (N^3,)
# # U_centred = U_train - U_mean[None, :]             # (n_snap, N^3)

# # # Move to numpy for SVD  (JAX SVD on large matrices can be slow)
# # U_centred_np = np.asarray(U_centred)

# # print("   Computing SVD ...")
# # t_svd = time.perf_counter()
# # _, S_vals, Vt = np.linalg.svd(U_centred_np, full_matrices=False)
# # t_svd = time.perf_counter() - t_svd
# # print(f"   SVD done in {t_svd:.1f}s   ({len(S_vals)} singular values)")

# # # Energy spectrum
# # energy = S_vals ** 2
# # cum_energy = np.cumsum(energy) / np.sum(energy)

# # # Pick r modes to capture desired energy
# # r = int(np.searchsorted(cum_energy, POD_ENERGY_THRESHOLD) + 1)
# # r = min(r, n_snap)

# # for rr in [50, 100, 200]:
# #     idx = min(rr - 1, len(cum_energy) - 1)
# #     print(f"   Energy at r={rr}: {cum_energy[idx]:.6f}")
# # print(f"   --> Keeping r = {r} modes  ({POD_ENERGY_THRESHOLD*100:.2f}% energy)")

# # # POD basis: columns of V_r are the first r right singular vectors
# # V_r_np = Vt[:r, :].T                              # (N^3, r)  numpy
# # V_r = jnp.array(V_r_np)                           # (N^3, r)  jax

# # # Project snapshots into POD coefficient space
# # C_train_np = U_centred_np @ V_r_np                 # (n_snap, r)  numpy
# # C_train = jnp.array(C_train_np)                    # (n_snap, r)  jax

# # # Verify: POD-only reconstruction error (project + lift, no AE)
# # U_pod_recon = C_train @ V_r.T + U_mean[None, :]
# # pod_err = float(jnp.mean(
# #     jnp.linalg.norm(U_train - U_pod_recon, axis=1) /
# #     jnp.linalg.norm(U_train, axis=1)
# # ))
# # print(f"   POD-only reconstruction error: {pod_err:.4e}")

# # # Normalise POD coefficients to zero-mean, unit-variance per mode
# # C_mean = jnp.mean(C_train, axis=0)                # (r,)
# # C_std  = jnp.std(C_train, axis=0) + 1e-10         # (r,)  eps prevents /0
# # C_norm = (C_train - C_mean[None, :]) / C_std[None, :]  # (n_snap, r)

# # print(f"   Normalised coeff range: [{float(C_norm.min()):.2f}, {float(C_norm.max()):.2f}]\n")


# # # ============================================================================
# # # 5. MODEL DEFINITIONS
# # # ============================================================================

# # class BaselineAE(nn.Module):
# #     """
# #     Dense autoencoder on POD coefficients.   c -> z -> c_hat
# #     Both encode() and decode() exposed for later NM-ROM use.
# #     """
# #     latent_dim: int
# #     pod_dim: int

# #     def setup(self):
# #         # Encoder:  pod_dim -> 512 -> 256 -> 128 -> latent_dim
# #         self.enc1    = nn.Dense(512)
# #         self.enc2    = nn.Dense(256)
# #         self.enc3    = nn.Dense(128)
# #         self.enc_out = nn.Dense(self.latent_dim)
# #         # Decoder:  latent_dim -> 128 -> 256 -> 512 -> pod_dim
# #         self.dec1    = nn.Dense(128)
# #         self.dec2    = nn.Dense(256)
# #         self.dec3    = nn.Dense(512)
# #         self.dec_out = nn.Dense(self.pod_dim)

# #     def encode(self, c):
# #         h = nn.swish(self.enc1(c))
# #         h = nn.swish(self.enc2(h))
# #         h = nn.swish(self.enc3(h))
# #         return self.enc_out(h)

# #     def decode(self, z):
# #         h = nn.swish(self.dec1(z))
# #         h = nn.swish(self.dec2(h))
# #         h = nn.swish(self.dec3(h))
# #         return self.dec_out(h)

# #     def __call__(self, c):
# #         return self.decode(self.encode(c))


# # class FourierAE(nn.Module):
# #     """
# #     Dense autoencoder with Fourier-featured encoder input.

# #     Encoder:  c -> fourier_lift(c) -> MLP -> z
# #     Decoder:  z -> MLP -> c_hat   (no Fourier on output)

# #     The Fourier lift is parameter-free and deterministic:
# #       gamma(c) = [c, sin(2*pi*f0*c), cos(2*pi*f0*c), ...,
# #                      sin(2*pi*f_{L-1}*c), cos(2*pi*f_{L-1}*c)]
# #     where f_i = base_freq * 2^i.

# #     This gives the MLP direct access to high-frequency basis functions,
# #     breaking the spectral bias of smooth activations (swish).
# #     """
# #     latent_dim: int
# #     pod_dim: int
# #     num_octaves: int = 4
# #     base_freq: float = 1.0

# #     def setup(self):
# #         # Encoder (first layer input = pod_dim * (1 + 2*num_octaves))
# #         self.enc1    = nn.Dense(512)
# #         self.enc2    = nn.Dense(256)
# #         self.enc3    = nn.Dense(128)
# #         self.enc_out = nn.Dense(self.latent_dim)
# #         # Decoder (identical to baseline)
# #         self.dec1    = nn.Dense(128)
# #         self.dec2    = nn.Dense(256)
# #         self.dec3    = nn.Dense(512)
# #         self.dec_out = nn.Dense(self.pod_dim)

# #     def _fourier_lift(self, c):
# #         """c in R^r  ->  gamma(c) in R^{r * (1 + 2*num_octaves)}."""
# #         parts = [c]
# #         for i in range(self.num_octaves):
# #             freq = self.base_freq * (2.0 ** i)
# #             parts.append(jnp.sin(2.0 * jnp.pi * freq * c))
# #             parts.append(jnp.cos(2.0 * jnp.pi * freq * c))
# #         return jnp.concatenate(parts)

# #     def encode(self, c):
# #         h = self._fourier_lift(c)          # <-- only difference from baseline
# #         h = nn.swish(self.enc1(h))
# #         h = nn.swish(self.enc2(h))
# #         h = nn.swish(self.enc3(h))
# #         return self.enc_out(h)

# #     def decode(self, z):
# #         h = nn.swish(self.dec1(z))
# #         h = nn.swish(self.dec2(h))
# #         h = nn.swish(self.dec3(h))
# #         return self.dec_out(h)

# #     def __call__(self, c):
# #         return self.decode(self.encode(c))


# # # ============================================================================
# # # 6. SHARED TRAINING LOOP
# # # ============================================================================

# # def count_params(p):
# #     return sum(x.size for x in jax.tree_util.tree_leaves(p))


# # def train_autoencoder(model, params_init, C_data, name):
# #     """
# #     Train an autoencoder on normalised POD coefficients.
# #     Both BaselineAE and FourierAE share the same API: model(c) -> c_hat.

# #     Returns: (trained_params, loss_history)
# #     """
# #     print(f"\n{'_' * 65}")
# #     print(f"  Training: {name}")
# #     print(f"  Parameters: {count_params(params_init):,}")
# #     print(f"{'_' * 65}")

# #     schedule = optax.exponential_decay(LR_INIT, LR_DECAY_STEPS, LR_DECAY_RATE)
# #     tx = optax.adam(schedule)
# #     opt_state = tx.init(params_init)
# #     params = params_init

# #     @jax.jit
# #     def step(p, opt, batch):
# #         def loss_fn(w):
# #             recon = jax.vmap(lambda c: model.apply({"params": w}, c))(batch)
# #             return jnp.mean((batch - recon) ** 2)
# #         loss, grads = jax.value_and_grad(loss_fn)(p)
# #         updates, new_opt = tx.update(grads, opt, p)
# #         return optax.apply_updates(p, updates), new_opt, loss

# #     n = len(C_data)
# #     n_batches = max(1, n // BATCH_SIZE)
# #     rng = np.random.default_rng(seed=42)
# #     losses = []

# #     t0 = time.perf_counter()
# #     for epoch in range(EPOCHS):
# #         perm = rng.permutation(n)
# #         epoch_loss = 0.0
# #         for b in range(n_batches):
# #             idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
# #             params, opt_state, batch_loss = step(params, opt_state, C_data[idx])
# #             epoch_loss += float(batch_loss)
# #         epoch_loss /= n_batches
# #         losses.append(epoch_loss)

# #         if epoch % 3000 == 0 or epoch == EPOCHS - 1:
# #             print(f"    Epoch {epoch:5d}   loss = {epoch_loss:.6e}")

# #     elapsed = time.perf_counter() - t0
# #     print(f"  Done in {elapsed:.1f}s   final loss = {losses[-1]:.6e}")
# #     return params, losses


# # # ============================================================================
# # # 7. TRAIN BASELINE MODEL
# # # ============================================================================

# # key = jax.random.PRNGKey(42)

# # # --- Baseline (POD + Dense) ---
# # mdl_base = BaselineAE(latent_dim=LATENT_DIM, pod_dim=r)
# # params_base_init = mdl_base.init(key, jnp.ones(r))["params"]
# # params_base, losses_base = train_autoencoder(
# #     mdl_base, params_base_init, C_norm, "Baseline (POD + Dense)")

# # # ============================================================================
# # # 7b. SAVE CHECKPOINT
# # # ============================================================================

# # print(f"\n--- Saving checkpoint to {CHECKPOINT_DIR} ---")

# # checkpointer = ocp.StandardCheckpointer()
# # ckpt_path = CHECKPOINT_DIR / "baseline_params"

# # checkpointer.save(ckpt_path, params_base)
# # print(f"   Saved params to: {ckpt_path}")

# # pod_data = {
# #     "U_mean": np.asarray(U_mean),
# #     "V_r": np.asarray(V_r),
# #     "C_mean": np.asarray(C_mean),
# #     "C_std": np.asarray(C_std),
# #     "r": int(r),
# #     "N": N,
# #     "LATENT_DIM": LATENT_DIM,
# # }
# # np.savez(CHECKPOINT_DIR / "pod_data.npz", **pod_data)
# # print(f"   Saved POD data to: {CHECKPOINT_DIR / 'pod_data.npz'}")

# # with open(CHECKPOINT_DIR / "training_losses.json", "w") as f:
# #     json.dump({"baseline": losses_base}, f)
# # print(f"   Saved losses to: {CHECKPOINT_DIR / 'training_losses.json'}")


# # # ============================================================================
# # # 8. EVALUATION
# # # ============================================================================

# # print(f"\n{'=' * 70}")
# # print(f"  EVALUATION")
# # print(f"{'=' * 70}")


# # def reconstruct(model, params, u_vec):
# #     """
# #     Full end-to-end reconstruction:
# #       u -> POD project -> normalise -> AE -> denormalise -> POD lift -> u_hat
# #     Works identically for both BaselineAE and FourierAE.
# #     """
# #     c     = (u_vec - U_mean) @ V_r            # POD project      (r,)
# #     c_n   = (c - C_mean) / C_std              # normalise         (r,)
# #     cn_hat = model.apply({"params": params}, c_n)  # autoencoder  (r,)
# #     c_hat = cn_hat * C_std + C_mean           # denormalise       (r,)
# #     return c_hat @ V_r.T + U_mean             # POD lift          (N^3,)


# # # -- 8a. Training set errors --
# # print("\n  [A] Training set reconstruction (all 512 snapshots):")

# # errs_base = np.zeros(n_snap)

# # for i in range(n_snap):
# #     u_true = U_train[i]
# #     norm_u = float(jnp.linalg.norm(u_true))
# #     errs_base[i] = float(jnp.linalg.norm(reconstruct(mdl_base, params_base, u_true) - u_true)) / norm_u

# # print(f"\n    {'Metric':<28s}  {'Baseline':>12s}  {'POD-only':>12s}")
# # print(f"    {'_' * 55}")
# # print(f"    {'Mean  rel L2':<28s}  {errs_base.mean():12.6e}  {pod_err:12.6e}")
# # print(f"    {'Max   rel L2':<28s}  {errs_base.max():12.6e}  {'-':>12s}")
# # print(f"    {'Median rel L2':<28s}  {np.median(errs_base):12.6e}  {'-':>12s}")


# # # -- 8b. Error grouped by frequency band --
# # print(f"\n  [B] Error by frequency magnitude |k|:")
# # bands = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
# # print(f"    {'|k| band':<12s} {'count':>5s}  {'Baseline':>11s}")
# # print(f"    {'_' * 35}")
# # for lo, hi in bands:
# #     sel = (k_mags >= lo) & (k_mags < hi)
# #     if sel.sum() == 0:
# #         continue
# #     eb = errs_base[sel].mean()
# #     print(f"    [{lo:2d},{hi:2d})     {int(sel.sum()):5d}  {eb:11.4e}")


# # # -- 8c. Unseen test cases --
# # print(f"\n  [C] Unseen test cases:")
# # test_cases = [
# #     # Seen (sanity check)
# #     (1,1,1, "seen"),   (4,4,4, "seen"),   (8,8,8, "seen"),
# #     (1,4,8, "seen"),   (2,7,5, "seen"),   (3,6,1, "seen"),
# #     # Mild extrapolation (k=9)
# #     (9,1,1, "extrap"), (9,9,1, "extrap"), (9,9,9, "extrap"),
# #     # Strong extrapolation (k=10,12)
# #     (10,10,10, "extrap"), (10,5,5, "extrap"), (12,1,1, "extrap"),
# # ]

# # print(f"    {'k-triple':<14s} {'|k|':>6s}  {'POD-only':>10s}  {'Baseline':>10s}  {'type'}")
# # print(f"    {'_' * 52}")

# # test_results = []
# # for (*ks, tp) in test_cases:
# #     k1, k2, k3 = ks
# #     km = np.sqrt(k1**2 + k2**2 + k3**2)
# #     u_fom = fom_solve(forcing_3d(k1, k2, k3))
# #     norm_u = float(jnp.linalg.norm(u_fom))

# #     # POD-only
# #     c_pod = (u_fom - U_mean) @ V_r
# #     u_pod = c_pod @ V_r.T + U_mean
# #     e_pod = float(jnp.linalg.norm(u_pod - u_fom)) / norm_u

# #     # Baseline AE
# #     e_b = float(jnp.linalg.norm(reconstruct(mdl_base, params_base, u_fom) - u_fom)) / norm_u

# #     label = f"({k1},{k2},{k3})"
# #     print(f"    {label:<14s} {km:6.2f}  {e_pod:10.4e}  {e_b:10.4e}  {tp}")
# #     test_results.append(dict(label=label, km=km, tp=tp,
# #                              e_pod=e_pod, e_base=e_b))


# # # ============================================================================
# # # 9. DIAGNOSTIC PLOTS
# # # ============================================================================

# # print(f"\n--- Saving plots to {OUTPUT_DIR} ---")

# # # Plot 1: Training loss curves
# # fig, ax = plt.subplots(figsize=(10, 5))
# # ax.semilogy(losses_base, label="Baseline (POD+Dense)", alpha=0.8, lw=1.2, color="#d62728")
# # ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (POD coeff space)")
# # ax.set_title("Training Loss - Baseline AE"); ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
# # plt.tight_layout(); plt.savefig(OUTPUT_DIR / "01_loss_curves.png", dpi=150); plt.close()
# # print("   01_loss_curves.png")

# # # Plot 2: POD singular value spectrum
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
# # n_show = min(300, len(S_vals))
# # ax1.semilogy(S_vals[:n_show], "b.-", markersize=2)
# # ax1.axvline(r, color="red", ls="--", label=f"r = {r}")
# # ax1.set_xlabel("Mode index"); ax1.set_ylabel("Singular value")
# # ax1.set_title("Singular Value Spectrum"); ax1.legend(); ax1.grid(True, ls="--", alpha=0.4)

# # ax2.plot(cum_energy[:n_show], "b-", lw=1.5)
# # ax2.axhline(POD_ENERGY_THRESHOLD, color="red", ls="--", alpha=0.7,
# #             label=f"{POD_ENERGY_THRESHOLD*100:.2f}%")
# # ax2.axvline(r, color="red", ls=":", alpha=0.7, label=f"r = {r}")
# # ax2.set_xlabel("Num modes"); ax2.set_ylabel("Cumulative energy")
# # ax2.set_title("POD Energy Capture"); ax2.legend(); ax2.grid(True, ls="--", alpha=0.4)
# # plt.tight_layout(); plt.savefig(OUTPUT_DIR / "02_pod_spectrum.png", dpi=150); plt.close()
# # print("   02_pod_spectrum.png")

# # # Plot 3: Error vs frequency magnitude (scatter)
# # fig, ax = plt.subplots(figsize=(10, 5))
# # ax.scatter(k_mags, errs_base, s=12, alpha=0.6, label="Baseline", color="#d62728")
# # ax.set_xlabel("|k|"); ax.set_ylabel("Relative L2 Error"); ax.set_yscale("log")
# # ax.set_title("Reconstruction Error vs Frequency (Training Set)")
# # ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
# # plt.tight_layout(); plt.savefig(OUTPUT_DIR / "03_error_vs_freq.png", dpi=150); plt.close()
# # print("   03_error_vs_freq.png")

# # # Plot 4: Error histogram
# # fig, ax = plt.subplots(figsize=(10, 5))
# # ax.hist(np.log10(errs_base), bins=40, alpha=0.8, color="#d62728", label="Baseline", edgecolor="black", linewidth=0.5)
# # ax.set_xlabel("log10(Relative L2 Error)"); ax.set_ylabel("Count")
# # ax.set_title("Error Distribution - Baseline AE"); ax.legend()
# # ax.axvline(np.log10(np.median(errs_base)), color="orange", ls="--", lw=2,
# #            label=f"Median = {np.median(errs_base):.2e}")
# # ax.legend()
# # plt.tight_layout(); plt.savefig(OUTPUT_DIR / "04_error_histogram.png", dpi=150); plt.close()
# # print("   04_error_histogram.png")

# # # Plot 5: Unseen test cases bar chart
# # fig, ax = plt.subplots(figsize=(12, 5))
# # n_tc = len(test_results)
# # xp = np.arange(n_tc)
# # w = 0.35
# # ax.bar(xp - w/2, [t["e_pod"]  for t in test_results], w,
# #        label="POD only", color="#2ca02c", alpha=0.7)
# # ax.bar(xp + w/2, [t["e_base"] for t in test_results], w,
# #        label="Baseline AE", color="#d62728", alpha=0.8)

# # for i, t in enumerate(test_results):
# #     if t["tp"] == "extrap":
# #         ax.axvspan(i - 0.45, i + 0.45, alpha=0.07, color="red")

# # ax.set_xticks(xp)
# # ax.set_xticklabels([t["label"] for t in test_results], rotation=45, ha="right")
# # ax.set_yscale("log"); ax.set_ylabel("Relative L2 Error")
# # ax.set_title("Test Cases  (red shading = extrapolation beyond training range)")
# # ax.legend(); ax.grid(True, which="both", axis="y", ls="--", alpha=0.4)
# # plt.tight_layout(); plt.savefig(OUTPUT_DIR / "05_test_cases.png", dpi=150); plt.close()
# # print("   05_test_cases.png")

# # # Plot 6: Midplane slices
# # print("   Generating midplane slices...")
# # viz_ks = [(1,1,1), (4,4,4), (8,8,8)]
# # mid = N // 2

# # for k1, k2, k3 in viz_ks:
# #     u_fom = fom_solve(forcing_3d(k1, k2, k3))
# #     u_b   = reconstruct(mdl_base, params_base, u_fom)

# #     # Reshape to 3D, take XY slice at z=mid
# #     sf  = np.asarray(u_fom).reshape(N, N, N)[:, :, mid]
# #     sb  = np.asarray(u_b  ).reshape(N, N, N)[:, :, mid]

# #     vmin = min(sf.min(), sb.min())
# #     vmax = max(sf.max(), sb.max())
# #     im_kw = dict(origin="lower", aspect="equal", cmap="RdBu_r",
# #                  vmin=vmin, vmax=vmax, extent=[0, L, 0, L])

# #     err_b = np.abs(sb - sf)
# #     emax  = max(err_b.max(), 1e-15)
# #     er_kw = dict(origin="lower", aspect="equal", cmap="hot",
# #                  vmin=0, vmax=emax, extent=[0, L, 0, L])

# #     fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# #     im = axes[0].imshow(sf.T,  **im_kw); axes[0].set_title("FOM (truth)")
# #     plt.colorbar(im, ax=axes[0], shrink=0.78)

# #     im = axes[1].imshow(sb.T,  **im_kw); axes[1].set_title("Baseline recon")
# #     plt.colorbar(im, ax=axes[1], shrink=0.78)

# #     im = axes[2].imshow(err_b.T, **er_kw); axes[2].set_title("|Baseline - FOM|")
# #     plt.colorbar(im, ax=axes[2], shrink=0.78)

# #     for ax in axes:
# #         ax.set_xlabel("x"); ax.set_ylabel("y")

# #     km = np.sqrt(k1**2 + k2**2 + k3**2)
# #     fig.suptitle(f"XY slice at z={mid*dx:.2f}   k=({k1},{k2},{k3})   |k|={km:.1f}",
# #                  fontsize=13, fontweight="bold")
# #     plt.tight_layout()
# #     fname = f"06_slice_k{k1}{k2}{k3}.png"
# #     plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
# #     plt.close()
# #     print(f"   {fname}")

# # # Plot 7: Latent space structure (PCA of latent codes coloured by |k|)
# # print("   Generating latent space plot...")

# # Z_base_all = np.array(jax.vmap(
# #     lambda c: mdl_base.apply({"params": params_base}, c, method=mdl_base.encode)
# # )(C_norm))


# # def pca_2d(Z_mat):
# #     """Project to 2D via PCA (first 2 principal components)."""
# #     Z_c = Z_mat - Z_mat.mean(axis=0)
# #     _, _, Vt_pca = np.linalg.svd(Z_c, full_matrices=False)
# #     return Z_c @ Vt_pca[:2].T


# # pc_base = pca_2d(Z_base_all)

# # fig, ax = plt.subplots(figsize=(8, 6))
# # sc = ax.scatter(pc_base[:, 0], pc_base[:, 1], c=k_mags, cmap="viridis", s=12, alpha=0.7)
# # ax.set_title("Baseline -- Latent PCA"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
# # plt.colorbar(sc, ax=ax, label="|k|", shrink=0.8)

# # plt.tight_layout(); plt.savefig(OUTPUT_DIR / "07_latent_pca.png", dpi=150); plt.close()
# # print("   07_latent_pca.png")


# # # ============================================================================
# # # 10. FINAL SUMMARY
# # # ============================================================================

# # print(f"\n{'=' * 70}")
# # print(f"  FINAL SUMMARY")
# # print(f"{'=' * 70}")
# # print(f"  Grid:              {N}^3 = {num_nodes:,} DOF")
# # print(f"  Training set:      k in [1,{K_MAX}]^3  =  {n_snap} snapshots")
# # print(f"  POD modes:         r = {r}  ({POD_ENERGY_THRESHOLD*100:.2f}% energy)")
# # print(f"  POD-only error:    {pod_err:.4e}")
# # print(f"  Latent dimension:  {LATENT_DIM}")
# # print()
# # n_p_base = count_params(params_base)
# # print(f"  {'Model':<28s} {'Params':>8s}  {'Mean':>11s}  {'Max':>11s}  {'Median':>11s}")
# # print(f"  {'_' * 73}")
# # print(f"  {'Baseline (POD+Dense)':<28s} {n_p_base:>8,}  "
# #       f"{errs_base.mean():>11.4e}  {errs_base.max():>11.4e}  {np.median(errs_base):>11.4e}")
# # print()
# # print(f"  Checkpoint saved to: {CHECKPOINT_DIR}")
# # print(f"  Plots saved to:      {OUTPUT_DIR}")
# # print(f"{'=' * 70}")
# # print(f"  === Done ===")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np

# # ==========================================
# # 1. MODEL DEFINITION
# # ==========================================
# class PoissonAutoencoder3D(nn.Module):
#     def __init__(self, latent_dim=12):
#         super(PoissonAutoencoder3D, self).__init__()
        
#         # Encoder: 32x32x32 -> Latent Dim
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4 * 4, latent_dim)
#         )
        
#         # Decoder: Latent Dim -> 32x32x32
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(latent_dim, 64 * 4 * 4 * 4),
#             nn.ReLU()
#         )
        
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1)
#         )

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, z):
#         x = self.decoder_fc(z)
#         x = x.view(-1, 64, 4, 4, 4) 
#         return self.decoder_conv(x)

#     def forward(self, x):
#         z = self.encode(x)
#         return self.decode(z)

# # ==========================================
# # 2. DATA GENERATION (U and F)
# # ==========================================
# def generate_poisson_dataset(num_samples, grid_size=32):
#     """
#     Generates parameterized solutions (u) and forcing functions (f)
#     for the 3D Poisson equation: -Laplacian(u) = f
#     """
#     print(f"Generating {num_samples} samples for a {grid_size}^3 grid...")
    
#     # Create coordinate grid
#     x = np.linspace(0, 1, grid_size)
#     y = np.linspace(0, 1, grid_size)
#     z = np.linspace(0, 1, grid_size)
#     X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
#     # Initialize arrays
#     U_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
#     F_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
    
#     for i in range(num_samples):
#         # Parameterize wave frequencies to create a manifold of solutions
#         kx = np.random.uniform(1.0, 3.0)
#         ky = np.random.uniform(1.0, 3.0)
#         kz = np.random.uniform(1.0, 3.0)
        
#         # The forcing function (F)
#         F = np.sin(2 * np.pi * kx * X) * np.sin(2 * np.pi * ky * Y) * np.sin(2 * np.pi * kz * Z)
        
#         # The analytical solution (U)
#         # -Laplacian(U) = F -> U = F / (4 * pi^2 * (kx^2 + ky^2 + kz^2))
#         C = 4 * np.pi**2 * (kx**2 + ky**2 + kz**2)
#         U = F / C
             
#         U_data[i, 0] = U
#         F_data[i, 0] = F
        
#     U_tensor = torch.tensor(U_data, dtype=torch.float32)
#     F_tensor = torch.tensor(F_data, dtype=torch.float32)
    
#     return TensorDataset(U_tensor, F_tensor)

# # ==========================================
# # 3. SETUP & TRAINING LOOP
# # ==========================================
# def main():
#     # Hyperparameters
#     num_samples = 1000
#     batch_size = 32
#     epochs = 100
#     learning_rate = 1e-3
#     latent_dim = 12

#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Load Data
#     dataset = generate_poisson_dataset(num_samples=num_samples, grid_size=32)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Initialize Model, Loss, and Optimizer
#     model = PoissonAutoencoder3D(latent_dim=latent_dim).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     print("Starting training on U (solution state)...")
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0
        
#         # We unpack both U and F, but ONLY train on U
#         for batch_u, batch_f in dataloader:
#             batch_u = batch_u.to(device)
            
#             # Forward pass: Reconstruct U
#             u_reconstructed = model(batch_u)
            
#             # Loss calculation
#             loss = criterion(u_reconstructed, batch_u)
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#         avg_loss = epoch_loss / len(dataloader)
        
#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.8f}")

#     print("\nTraining complete!")

#     # ==========================================
#     # 4. EXTRACTING THE EMBEDDING (Z)
#     # ==========================================
#     model.eval()
#     with torch.no_grad():
#         # Grab a single sample to test the embedding extraction
#         sample_u = dataset[0][0].unsqueeze(0).to(device) # Shape: (1, 1, 32, 32, 32)
        
#         # Extract the 12-dimensional latent vector z
#         z_embedding = model.encode(sample_u)
        
#         print(f"\nExtracted Latent Vector z shape: {z_embedding.shape}")
#         print(f"Latent Vector z values:\n{z_embedding.cpu().numpy()}")

# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class PoissonAutoencoder3D(nn.Module):
    def __init__(self, latent_dim=12):
        super(PoissonAutoencoder3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, latent_dim)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4 * 4),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Added Tanh here to bound the output between -1 and 1 since data is normalized
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 4, 4, 4) 
        return self.decoder_conv(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# ==========================================
# 2. DATA GENERATION & NORMALIZATION
# ==========================================
def generate_poisson_dataset(num_samples, grid_size=32):
    print(f"Generating {num_samples} samples for a {grid_size}^3 grid...")
    
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    U_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
    F_data = np.zeros((num_samples, 1, grid_size, grid_size, grid_size))
    
    for i in range(num_samples):
        kx = np.random.uniform(1.0, 3.0)
        ky = np.random.uniform(1.0, 3.0)
        kz = np.random.uniform(1.0, 3.0)
        
        F = np.sin(2 * np.pi * kx * X) * np.sin(2 * np.pi * ky * Y) * np.sin(2 * np.pi * kz * Z)
        C = 4 * np.pi**2 * (kx**2 + ky**2 + kz**2)
        U = F / C
             
        U_data[i, 0] = U
        F_data[i, 0] = F
        
    U_tensor = torch.tensor(U_data, dtype=torch.float32)
    F_tensor = torch.tensor(F_data, dtype=torch.float32)
    
    # --- NORMALIZATION STEP ---
    # Find the absolute maximum value across all generated U data
    u_max = torch.max(torch.abs(U_tensor))
    print(f"Global max absolute value for normalization: {u_max.item():.6f}")
    
    # Scale U to be between -1 and 1
    U_tensor_normalized = U_tensor / u_max
    
    return TensorDataset(U_tensor_normalized, F_tensor), u_max

# ==========================================
# 3. SETUP & TRAINING LOOP
# ==========================================
def main():
    num_samples = 1000
    batch_size = 32
    epochs = 100
    learning_rate = 1e-3
    latent_dim = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, u_max = generate_poisson_dataset(num_samples=num_samples, grid_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PoissonAutoencoder3D(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training on Normalized U...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_u, batch_f in dataloader:
            batch_u = batch_u.to(device)
            
            u_reconstructed = model(batch_u)
            loss = criterion(u_reconstructed, batch_u)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.6f}")

    print("\nTraining complete!")

    # ==========================================
    # 4. VISUALIZATION (UN-NORMALIZED PLOTS)
    # ==========================================
    print("\nGenerating reconstruction plots...")
    model.eval()
    
    sample_idx = 0
    u_true_normalized = dataset[sample_idx][0].unsqueeze(0).to(device)
    
    with torch.no_grad():
        u_pred_normalized = model(u_true_normalized)
        
    # Move back to CPU, convert to numpy, and UN-NORMALIZE by multiplying by u_max
    u_max_val = u_max.item()
    u_true_np = u_true_normalized.cpu().numpy()[0, 0] * u_max_val
    u_pred_np = u_pred_normalized.cpu().numpy()[0, 0] * u_max_val
    
    slice_idx = 16 
    
    u_true_slice = u_true_np[:, :, slice_idx]
    u_pred_slice = u_pred_np[:, :, slice_idx]
    error_slice = np.abs(u_true_slice - u_pred_slice)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(u_true_slice, cmap='viridis', origin='lower')
    axes[0].set_title(f'Original Solution (Z={slice_idx})')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(u_pred_slice, cmap='viridis', origin='lower')
    axes[1].set_title(f'Reconstructed Solution (Z={slice_idx})')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(error_slice, cmap='magma', origin='lower')
    axes[2].set_title(f'Absolute Error (Z={slice_idx})')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()