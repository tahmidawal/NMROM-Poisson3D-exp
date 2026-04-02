"""
generate_heat_data.py
─────────────────────
Generate 3D Heat Equation training data for different grid resolutions.

Usage:
    python generate_heat_data.py --grid 64
    python generate_heat_data.py --grid 128
    python generate_heat_data.py --grid 64 128   # Both

Outputs:
    plots/heat_ae/training_data_64.pkl
    plots/heat_ae/training_data_128.pkl
"""

import argparse
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy.stats import qmc
import pickle
import time
from pathlib import Path

# ─────────────────────────────────────────
# 0. Output Directory
# ─────────────────────────────────────────
OUT = Path('plots/heat_ae')
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 1. Physics Parameters (fixed across grids)
# ─────────────────────────────────────────
L         = 1.0       # Domain size [0, L]^3
dt        = 0.005     # Time step
NUM_STEPS = 50        # Total time T = 0.25s
N_TRAIN   = 200       # Training trajectories
N_VAL     = 20        # Validation trajectories


def generate_data_for_grid(N: int):
    """
    Generate training and validation data for a given grid resolution N.
    """
    num_nodes = N ** 3
    dx = L / (N - 1)
    
    print(f"\n{'='*60}")
    print(f"  Generating data for {N}³ = {num_nodes:,} grid")
    print(f"{'='*60}")
    print(f"  dt={dt}  T={dt*NUM_STEPS:.3f}s  dx={dx:.6f}")
    
    # ── Grid coordinates ──────────────────────────────────────────
    x_sp = jnp.linspace(0, L, N)
    y_sp = jnp.linspace(0, L, N)
    z_sp = jnp.linspace(0, L, N)
    X, Y, Z = jnp.meshgrid(x_sp, y_sp, z_sp, indexing='ij')
    
    # ── Negative Laplacian (7-point stencil, Dirichlet BCs) ───────
    def K_op_3d(u_flat):
        u   = u_flat.reshape((N, N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1,1:-1,1:-1].set(
            (6*u[1:-1,1:-1,1:-1]
             - u[0:-2,1:-1,1:-1] - u[2:,1:-1,1:-1]
             - u[1:-1,0:-2,1:-1] - u[1:-1,2:,1:-1]
             - u[1:-1,1:-1,0:-2] - u[1:-1,1:-1,2:]) / dx**2
        )
        out = out.at[0,:,:].set(u[0,:,:])
        out = out.at[-1,:,:].set(u[-1,:,:])
        out = out.at[:,0,:].set(u[:,0,:])
        out = out.at[:,-1,:].set(u[:,-1,:])
        out = out.at[:,:,0].set(u[:,:,0])
        out = out.at[:,:,-1].set(u[:,:,-1])
        return out.flatten()
    
    def implicit_op(u_flat, kappa):
        return u_flat + dt * kappa * K_op_3d(u_flat)
    
    # ── Gaussian IC Generation ────────────────────────────────────
    def make_gaussian_ic(centers, amplitudes, widths):
        u = jnp.zeros((N, N, N))
        for (cx, cy, cz), A, sigma in zip(centers, amplitudes, widths):
            u = u + A * jnp.exp(
                -((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2)
            )
        u = u.at[0,:,:].set(0.).at[-1,:,:].set(0.)
        u = u.at[:,0,:].set(0.).at[:,-1,:].set(0.)
        u = u.at[:,:,0].set(0.).at[:,:,-1].set(0.)
        return u.flatten()
    
    # ── FOM Time-Stepping (Backward Euler + CG) ───────────────────
    def run_fom(u0_flat, kappa, steps):
        snapshots = [u0_flat]
        u = u0_flat
        op = lambda v: implicit_op(v, kappa)
        for _ in range(steps):
            u, _ = jax_linalg.cg(op, u, x0=u, tol=1e-6, maxiter=1000)
            snapshots.append(u)
        return jnp.stack(snapshots)
    
    # ── LHS Trajectory Parameter Sampling ─────────────────────────
    def sample_trajectory_params(rng, n_traj):
        sampler = qmc.LatinHypercube(d=13, seed=rng)
        samples = sampler.random(n=n_traj)
        
        trajectories = []
        for s in samples:
            n_gauss = int(np.round(1 + 2 * s[0]))
            centers, amplitudes, widths = [], [], []
            
            for g in range(n_gauss):
                cx = 0.15 + 0.70 * s[1 + g*3]
                cy = 0.15 + 0.70 * s[2 + g*3]
                cz = 0.15 + 0.70 * s[3 + g*3]
                centers.append((cx, cy, cz))
                amplitudes.append(1.0 + 9.0 * s[10])
                widths.append(0.05 + 0.15 * s[11])
            
            kappa = float(np.exp(np.log(0.01) + (np.log(0.5) - np.log(0.01)) * s[12]))
            trajectories.append(dict(
                centers=centers, amplitudes=amplitudes,
                widths=widths, kappa=kappa
            ))
        return trajectories
    
    # ── Generate Training Data ────────────────────────────────────
    print(f"\n── Generating {N_TRAIN} training + {N_VAL} validation trajectories ──")
    print(f"   Each: {NUM_STEPS+1} snapshots  →  Total ≈ {N_TRAIN*(NUM_STEPS+1):,}")
    
    train_params = sample_trajectory_params(rng=42, n_traj=N_TRAIN)
    val_params   = sample_trajectory_params(rng=1337, n_traj=N_VAL)
    
    all_snapshots = []
    traj_kappas   = []
    traj_starts   = []
    
    t0 = time.perf_counter()
    for i, tp in enumerate(train_params):
        u0   = make_gaussian_ic(tp['centers'], tp['amplitudes'], tp['widths'])
        traj = run_fom(u0, tp['kappa'], NUM_STEPS)
        traj.block_until_ready()  # Force synchronous execution
        traj_starts.append(len(all_snapshots))
        all_snapshots.append(traj)
        traj_kappas.append(tp['kappa'])
        if (i+1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"   Train {i+1}/{N_TRAIN}  ({elapsed:.0f}s elapsed)")
    
    U_train = jnp.concatenate(all_snapshots, axis=0)
    print(f"   Training snapshots: {U_train.shape}")
    
    val_snapshots = []
    val_kappas    = []
    for i, vp in enumerate(val_params):
        u0   = make_gaussian_ic(vp['centers'], vp['amplitudes'], vp['widths'])
        traj = run_fom(u0, vp['kappa'], NUM_STEPS)
        traj.block_until_ready()  # Force synchronous execution
        val_snapshots.append(traj)
        val_kappas.append(vp['kappa'])
        if (i+1) % 10 == 0:
            print(f"   Val {i+1}/{N_VAL}")
    
    U_val = jnp.concatenate(val_snapshots, axis=0)
    print(f"   Validation snapshots: {U_val.shape}")
    print(f"   Data generation: {time.perf_counter()-t0:.1f}s")
    
    # ── Save Data ─────────────────────────────────────────────────
    data_file = OUT / f'training_data_{N}.pkl'
    data_to_save = {
        'U_train': np.array(U_train),
        'U_val': np.array(U_val),
        'all_snapshots': [np.array(s) for s in all_snapshots],
        'val_snapshots': [np.array(s) for s in val_snapshots],
        'train_params': train_params,
        'val_params': val_params,
        'traj_kappas': traj_kappas,
        'val_kappas': val_kappas,
        'traj_starts': traj_starts,
        'grid_config': {
            'N': N, 'L': L, 'dx': dx, 'dt': dt, 'NUM_STEPS': NUM_STEPS
        }
    }
    with open(data_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    file_size_mb = data_file.stat().st_size / (1024 * 1024)
    print(f"\n   Saved: {data_file}  ({file_size_mb:.1f} MB)")
    
    return data_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D Heat Equation training data for different grid sizes'
    )
    parser.add_argument(
        '--grid', '-g', 
        type=int, 
        nargs='+', 
        default=[64],
        help='Grid resolution(s) to generate, e.g., --grid 64 128'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("  3D Heat Equation Data Generator")
    print("="*60)
    print(f"  Grid sizes: {args.grid}")
    print(f"  Trajectories: {N_TRAIN} train + {N_VAL} val")
    print(f"  Time steps: {NUM_STEPS}  (dt={dt}, T={dt*NUM_STEPS:.3f}s)")
    
    for grid_size in args.grid:
        generate_data_for_grid(grid_size)
    
    print("\n" + "="*60)
    print("  Data generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
