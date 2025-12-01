# light_theory_realm/experiments/gravity_probe.py

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from functools import partial
from typing import Tuple # Added this import

# Local imports for project components
from light_theory_realm.engine import CliffordEngine
from light_theory_realm.geometry.spacetime_probe import gravity_probe, GravityProbeResult
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeWilsonLoop # To re-use its _get_generator logic indirectly

# --- Helper for constructing scaled Wilson Loop (adapted from PrimeWilsonLoop) ---
# This helper needs to be defined outside define_psi_fn to avoid JAX re-compilation issues
# with closures and static arguments if used directly inside a JIT'd function.
# Or, pass the engine as a static arg to a jitted helper. For simplicity, define here.

def _get_generator_from_primes(engine: CliffordEngine, p_i: int, p_j: int) -> jnp.ndarray:
    """
    Maps prime pair to a Clifford generator based on PrimeWilsonLoop's logic.
    """
    idx = (p_i * p_j) % 8
    
    if idx == 0:
        return jnp.eye(4, dtype=jnp.complex128)
    elif idx <= 3:
        return engine.gammas[idx] 
    elif idx <= 6:
        # These correspond to bivectors gamma_01, gamma_02, gamma_03
        return engine.get_bivector(0, idx-3)
    else: # idx == 7
        # Pseudoscalar-like or a mix, as defined in PrimeWilsonLoop
        return engine.gammas[0] @ engine.gammas[1]

@partial(jax.jit, static_argnums=(0, 1))
def _construct_scaled_wilson_loop(engine: CliffordEngine, core_primes: Tuple[int, ...], initial_spinor: jnp.ndarray, scale_factor: jnp.ndarray) -> jnp.ndarray:
    """
    Constructs a Wilson Loop operator whose internal phase angles are scaled by `scale_factor`.
    Returns the resulting state after applying this loop to the initial spinor.
    """
    W_op = jnp.eye(4, dtype=jnp.complex128)
    path = core_primes + (core_primes[0],)
    for k in range(len(core_primes)):
        p_i = path[k]
        p_j = path[k+1]
        
        theta_raw = jnp.log(p_j / p_i) # Original phase angle
        scaled_theta = scale_factor * theta_raw # Apply the scaling factor
        
        G = _get_generator_from_primes(engine, p_i, p_j)
        U = expm(1j * scaled_theta * G)
        W_op = U @ W_op
    
    return W_op @ initial_spinor

# --- Main psi_fn definition ---
def define_three_param_psi_fn(engine: CliffordEngine, base_spinor: jnp.ndarray, core_primes: Tuple[int, ...]):
    """
    Returns a psi_fn(theta_vec) callable for the gravity probe.
    theta_vec = [loop_scale, rotation_angle_1, rotation_angle_2]

    This function defines a quantum state psi_fn(theta_vec) by applying:
    1. A Wilson Loop operator (defined by `core_primes`) whose internal phase
       angles are scaled by `theta_vec[0]`.
    2. A rotation by `theta_vec[1]` with a fixed generator (e.g., gamma_1).
    3. Another rotation by `theta_vec[2]` with a different fixed generator (e.g., gamma_2).
    """
    
    # Fixed generators for the parametric rotations
    G1 = engine.gammas[1] # e.g., gamma_1
    G2 = engine.gammas[2] # e.g., gamma_2

    # Define the core psi_fn with parameters
    def psi_fn(theta_vec: jnp.ndarray) -> jnp.ndarray:
        # Ensure theta_vec is a JAX array for differentiation
        theta_vec = jnp.asarray(theta_vec)

        # Unpack parameters
        scale_factor = theta_vec[0]
        rot_angle_1 = theta_vec[1]
        rot_angle_2 = theta_vec[2]

        # 1. Apply scaled Wilson Loop to the base spinor
        # This function is JIT-compiled for efficiency
        psi_after_loop = _construct_scaled_wilson_loop(engine, core_primes, base_spinor, scale_factor)

        # 2. Apply first parametric rotation
        U1 = expm(1j * rot_angle_1 * G1)
        psi_after_U1 = U1 @ psi_after_loop

        # 3. Apply second parametric rotation
        U2 = expm(1j * rot_angle_2 * G2)
        final_psi = U2 @ psi_after_U1
        
        # Ensure final state is normalized for QGT calculations
        return final_psi / jnp.linalg.norm(final_psi)
    
    return psi_fn

def run_gravity_probe_experiment():
    print("--- Light Theory Realm: Spacetime Curvature Probe ---")
    
    # --- Configuration ---
    KEY = jax.random.PRNGKey(0)
    engine = CliffordEngine(seed=int(KEY[0])) # Initialize engine with a deterministic seed
    
    # A base spinor for the state generation
    base_spinor = engine.random_spinor(jax.random.PRNGKey(1))
    
    # Primes defining the core plaquette for the Wilson Loop
    core_primes = (2, 3, 5, 7) # A "matter-like" plaquette
    
    # Define the three-parameter psi_fn
    psi_function = define_three_param_psi_fn(engine, base_spinor, core_primes)

    # --- Parameter Grid ---
    # We will vary two parameters and keep one fixed for easier visualization.
    # Let's vary loop_scale and rotation_angle_1, keep rotation_angle_2 fixed.
    num_points = 20
    loop_scales = jnp.linspace(0.1, 2.0, num_points) # Theta[0]
    rot_angles_1 = jnp.linspace(0.0, jnp.pi, num_points) # Theta[1]
    
    # Fixed value for the third parameter
    fixed_rot_angle_2 = jnp.pi / 4 # Theta[2]

    print(f"\nProbing a 3-parameter state manifold:")
    print(f"  Varying: loop_scale ({loop_scales[0]:.2f} to {loop_scales[-1]:.2f}), rotation_angle_1 ({rot_angles_1[0]:.2f} to {rot_angles_1[-1]:.2f})")
    print(f"  Fixed: rotation_angle_2 = {fixed_rot_angle_2:.2f}")
    print(f"  Core Primes for Wilson Loop: {core_primes}")

    # --- Data Collection ---
    results_data = []
    
    print("\nRunning probes across parameter grid (this may take a moment)...")
    for i, ls in enumerate(loop_scales):
        for j, ra1 in enumerate(rot_angles_1):
            theta_vec = jnp.array([ls, ra1, fixed_rot_angle_2])
            
            # Call the gravity_probe API
            probe_result = gravity_probe(psi_function, theta_vec, core_primes, engine)
            
            # Store key scalar results
            results_data.append({
                'loop_scale': float(ls),
                'rot_angle_1': float(ra1),
                'rot_angle_2': float(fixed_rot_angle_2),
                'xi_resonance': probe_result.xi_resonance,
                'rho_igbp': probe_result.rho_igbp,
                'fisher_norm': float(jnp.linalg.norm(probe_result.fisher_metric)),
                'berry_norm': float(jnp.linalg.norm(probe_result.berry_curvature)),
                'reeb_norm': float(jnp.linalg.norm(probe_result.reeb_vector))
            })
            if (i * num_points + j) % (num_points * num_points // 10) == 0:
                print(f"  Progress: {(i * num_points + j) / (num_points * num_points) * 100:.1f}%")

    # Define output directory and timestamp ONCE at the start
    output_dir = "light_theory_realm/experiments/logs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Probe complete. Analyzing results.")
    df = pd.DataFrame(results_data)

    print("\nDataFrame head:\n", df.head())
    print("\nDataFrame info:")
    df.info()
    print("\nNaN values per column:\n", df.isnull().sum())
    
    # Save results DataFrame to CSV
    csv_filename = os.path.join(output_dir, f"gravity_probe_results_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved detailed results to {csv_filename}")

    # Plotting functions for 2D slices (heatmap-like)
    def plot_heatmap(df_plot, value_col, title, filename):
        pivot_table = df_plot.pivot_table(index='loop_scale', columns='rot_angle_1', values=value_col)
        
        # Ensure the pivot table values are float type for imshow
        plot_data = pivot_table.values.astype(float)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(plot_data, origin='lower', 
                   extent=[df_plot['rot_angle_1'].min(), df_plot['rot_angle_1'].max(), 
                           df_plot['loop_scale'].min(), df_plot['loop_scale'].max()],
                   aspect='auto', cmap='viridis')
        plt.colorbar(label=value_col)
        plt.xlabel("Rotation Angle 1 (rad)")
        plt.ylabel("Loop Scale Factor")
        plt.title(title)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_{timestamp}.png"))
        plt.close()

    # Plot xi_resonance (should be constant as it depends only on core_primes)
    # This will be a flat heatmap, which is expected.
    plot_heatmap(df, 'xi_resonance', 'Vacuum Resonance Density (xi) across Parameter Space', 'xi_resonance_heatmap')
    
    # Plot rho_igbp
    plot_heatmap(df, 'rho_igbp', 'Information-Geometric Density (rho_IGBP) across Parameter Space', 'rho_igbp_heatmap')

    # Plot Fisher Metric Norm
    plot_heatmap(df, 'fisher_norm', 'Fisher Metric Norm across Parameter Space', 'fisher_norm_heatmap')

    # Plot Berry Curvature Norm
    plot_heatmap(df, 'berry_norm', 'Berry Curvature Norm across Parameter Space', 'berry_norm_heatmap')

    print(f"\nExperiment complete. Results (plots and CSV) saved in '{output_dir}/'.")
    print("Consider inspecting the plots to identify regions of high curvature, information density, or resonance.")

if __name__ == "__main__":
    run_gravity_probe_experiment()
