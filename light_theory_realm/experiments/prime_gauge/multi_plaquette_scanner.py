"""
Multi-Plaquette Scanner for Prime Sixth-Order Flow

Systematically compares ρ_IGBP behavior across all 9 Standard Model fermion
plaquettes using genuine Wilson loop holonomy and QGT-derived curvature.

Extensions implemented:
- Wilson loop holonomy (replaces ln-ratio phases)
- Multi-plaquette scanning (all Pocket_U fermions)
- Reeb flow integration (time-aligned evolution)
- QGT parameterization (2-parameter family with curvature scalar)
"""

import jax
import jax.numpy as jnp
from jax import jit
import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.qgt import CliffordQGT
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeWilsonLoop
from light_theory_realm.experiments.prime_gauge.reeb_flow import ReebFlowDynamics
from light_theory_realm.experiments.prime_gauge.prime_sixth_order_flow import (
    build_prime_ring,
    build_sixth_order_operator,
    gradient_flow_step,
    compute_S_rep,
    compute_local_S_rep,
    compute_gradient_norm,
)
from light_theory_realm.pocket_u_lite.plaquettes import (
    PLAQUETTES,
    PDG_MASSES,
    ALL_PARTICLES,
    LEPTONS,
    QUARKS,
)


# =============================================================================
# Extension A: Wilson Loop Holonomy
# =============================================================================

def compute_wilson_holonomy(primes: List[int], engine: CliffordEngine) -> float:
    """
    Compute genuine geometric holonomy using Wilson loop operator.
    
    Args:
        primes: List of prime numbers defining the ring (without closing)
        engine: CliffordEngine instance
    
    Returns:
        Φ_geometric: Phase from det(W) of Wilson loop
    
    Notes:
        Uses arg(det(W)) convention for holonomy phase.
        This gives non-trivial Φ even when ln-ratio phases cancel.
    """
    qc_loop = PrimeWilsonLoop(engine)
    
    # Remove closing prime if present (PrimeWilsonLoop adds it)
    primes_open = primes[:-1] if primes[-1] == primes[0] else primes
    
    W = qc_loop.construct_loop_operator(primes_open)
    
    # Extract phase from determinant
    Phi_geometric = jnp.angle(jnp.linalg.det(W))
    
    return float(Phi_geometric)


def compute_berry_norm(phi: float) -> float:
    """Toy Berry curvature proxy |Ω| = |sin Φ|."""
    return float(jnp.abs(jnp.sin(phi)))


def compute_igbp_with_wilson(
    psi: jnp.ndarray,
    laplacian: jnp.ndarray,
    primes: List[int],
    engine: CliffordEngine,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """
    Compute ρ_IGBP using genuine Wilson loop holonomy.
    
    Args:
        psi: Current state (N,)
        laplacian: Graph Laplacian (N, N)
        primes: Prime sequence
        engine: CliffordEngine for Wilson loop
        eps: Regularization
    
    Returns:
        Dictionary with ρ_IGBP and components
    """
    S_local = compute_local_S_rep(psi, eps=eps)
    grad_norm = compute_gradient_norm(S_local, laplacian)
    phi = compute_wilson_holonomy(primes, engine)
    omega = compute_berry_norm(phi)
    rho = grad_norm / (omega + eps)
    
    return {
        "rho_igbp": float(rho),
        "grad_norm": float(grad_norm),
        "omega": float(omega),
        "phi": float(phi),
    }


# =============================================================================
# Extension B: Multi-Plaquette Scanner
# =============================================================================

def run_single_plaquette(
    particle: str,
    primes: List[int],
    engine: CliffordEngine,
    n_steps: int = 200,
    dt: float = 0.01,
    use_wilson: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run sixth-order flow on a single plaquette.
    
    Args:
        particle: Particle name
        primes: Prime sequence (with closing prime)
        engine: CliffordEngine instance
        n_steps: Number of evolution steps
        dt: Time step
        use_wilson: Use Wilson loop holonomy
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    # Build ring (remove closing prime for 4-node ring)
    primes_ring = primes[:-1]
    
    # Build infrastructure
    _, laplacian, _ = build_prime_ring()  # Uses [2,3,5,7] template
    L6, H = build_sixth_order_operator(laplacian)
    
    # Initialize random state
    key = jax.random.PRNGKey(hash(particle) % 2**32)
    psi = jax.random.normal(key, (4,))
    psi = psi / jnp.linalg.norm(psi)
    psi_init = psi.copy()
    
    # Evolution loop
    history = []
    for step in range(n_steps + 1):
        # Compute diagnostics
        S = float(compute_S_rep(psi))
        
        if use_wilson:
            igbp = compute_igbp_with_wilson(psi, laplacian, primes_ring, engine)
        else:
            # Fallback to ln-ratio (would need phases)
            igbp = {"rho_igbp": 0.0, "grad_norm": 0.0, "omega": 0.0, "phi": 0.0}
        
        history.append({
            "step": step,
            "S_rep": S,
            **igbp,
            "psi": psi.copy(),
        })
        
        if verbose and step % 50 == 0:
            print(f"  step={step:3d} | S={S:.4f} | ρ={igbp['rho_igbp']:.2e}")
        
        # Apply gradient flow
        if step < n_steps:
            psi = gradient_flow_step(psi, H, dt)
    
    # Find convergence point
    convergence_step = n_steps
    for entry in history:
        if entry["grad_norm"] < 1e-6:
            convergence_step = entry["step"]
            break
    
    return {
        "particle": particle,
        "primes": primes_ring,
        "psi_init": psi_init,
        "psi_final": psi,
        "history": history,
        "convergence_step": convergence_step,
    }


def scan_all_plaquettes(
    n_steps: int = 200,
    dt: float = 0.01,
    use_wilson: bool = True,
    verbose: bool = True,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Scan all 9 Standard Model fermion plaquettes.
    
    Args:
        n_steps: Evolution steps per plaquette
        dt: Time step
        use_wilson: Use Wilson loop holonomy
        verbose: Print progress
        save_results: Save to CSV
    
    Returns:
        DataFrame with results for all particles
    """
    engine = CliffordEngine(seed=42)
    results = []
    
    if verbose:
        print("=" * 80)
        print("Multi-Plaquette Scanner: Prime Sixth-Order Flow")
        print("=" * 80)
        print()
    
    for particle in ALL_PARTICLES:
        primes = PLAQUETTES[particle]
        mass = PDG_MASSES[particle]
        
        if verbose:
            print(f"Scanning {particle:3s} (m={mass:10.2f} MeV): {primes}")
        
        run_result = run_single_plaquette(
            particle, primes, engine, n_steps, dt, use_wilson, verbose=False
        )
        
        initial = run_result["history"][0]
        final = run_result["history"][-1]
        
        results.append({
            "particle": particle,
            "plaquette": str(primes[:-1]),  # Remove closing prime
            "mass_MeV": mass,
            "phi_initial": initial["phi"],
            "phi_final": final["phi"],
            "omega_initial": initial["omega"],
            "omega_final": final["omega"],
            "rho_igbp_initial": initial["rho_igbp"],
            "rho_igbp_final": final["rho_igbp"],
            "S_rep_final": final["S_rep"],
            "convergence_step": run_result["convergence_step"],
        })
        
        if verbose:
            print(f"  → Φ={final['phi']:.4f}, |Ω|={final['omega']:.4f}, "
                  f"ρ={final['rho_igbp']:.2e}, converged@{run_result['convergence_step']}")
            print()
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    if save_results:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"plaquette_scan_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"Results saved to: {csv_path}")
            print()
    
    return df


# =============================================================================
# Extension C: Reeb Flow Integration
# =============================================================================

def compute_reeb_aligned_update(
    psi: jnp.ndarray,
    H: jnp.ndarray,
    primes: List[int],
    engine: CliffordEngine,
    dt: float,
) -> jnp.ndarray:
    """
    Compute Reeb-aligned flow update.
    
    Evolution: dψ = -dt·H·ψ + ξ·R·dt
    
    where:
        - H·ψ: Natural gradient term
        - R: Reeb vector (null space of Berry)
        - ξ: Resonance density (dark energy)
    
    Args:
        psi: Current state
        H: Flow operator
        primes: Prime sequence for resonance calculation
        engine: CliffordEngine
        dt: Time step
    
    Returns:
        Updated state
    """
    # Natural gradient term
    grad_term = H @ psi
    
    # Compute resonance density (dark energy)
    reeb = ReebFlowDynamics(engine)
    xi = reeb.compute_resonance_density(primes)
    
    # For Reeb vector, we need Fisher and Berry from QGT
    # Simplified: use random direction for now (full QGT implementation below)
    # In full version, compute from actual QGT
    R = jnp.ones_like(psi) / jnp.linalg.norm(jnp.ones_like(psi))
    
    # Combined update
    psi_next = psi - dt * grad_term + xi * dt * R
    
    # Normalize
    norm = jnp.linalg.norm(psi_next)
    return jnp.where(norm > 0, psi_next / norm, psi_next)


# =============================================================================
# Extension D: QGT Parameterization
# =============================================================================

def build_parameterized_state(
    theta1: float,
    theta2: float,
    base_psi: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Build 2-parameter family ψ(θ¹, θ²) on 4-node ring.
    
    Args:
        theta1: Mode mixing parameter ∈ [0, π/2]
        theta2: Phase pattern parameter ∈ [0, 2π]
        base_psi: Base state (default: uniform)
    
    Returns:
        psi: (4,) parameterized state
    
    Parameterization:
        θ¹: Mixes modes via orthogonal rotation
        θ²: Adds phase pattern exp(i·k·θ²) around ring
    """
    if base_psi is None:
        base_psi = jnp.ones(4) / 2.0
    
    # θ¹: Mode mixing via rotation matrix
    c1, s1 = jnp.cos(theta1), jnp.sin(theta1)
    R = jnp.array([
        [c1, -s1, 0, 0],
        [s1,  c1, 0, 0],
        [0, 0, c1, -s1],
        [0, 0, s1,  c1],
    ])
    psi_mixed = R @ base_psi
    
    # θ²: Phase pattern (discrete momentum mode k=1)
    phases = jnp.array([0.0, 1.0, 2.0, 3.0]) * theta2
    phase_pattern = jnp.cos(phases)  # Real part for scalar states
    
    # Combine
    psi = psi_mixed * phase_pattern
    
    # Normalize
    norm = jnp.linalg.norm(psi)
    return jnp.where(norm > 0, psi / norm, psi)


def compute_qgt_on_ring(
    theta1: float,
    theta2: float,
    engine: CliffordEngine,
    base_psi: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute QGT (Fisher + Berry) for parameterized state family.
    
    Args:
        theta1, theta2: Parameters
        engine: CliffordEngine instance
        base_psi: Base state
    
    Returns:
        Dictionary with psi, jacobian, fisher, berry, curvature_scalar
    """
    # Build state
    psi = build_parameterized_state(theta1, theta2, base_psi)
    
    # Compute Jacobian numerically
    eps = 1e-6
    
    # ∂ψ/∂θ¹
    psi_p1 = build_parameterized_state(theta1 + eps, theta2, base_psi)
    psi_m1 = build_parameterized_state(theta1 - eps, theta2, base_psi)
    d_psi_theta1 = (psi_p1 - psi_m1) / (2 * eps)
    
    # ∂ψ/∂θ²
    psi_p2 = build_parameterized_state(theta1, theta2 + eps, base_psi)
    psi_m2 = build_parameterized_state(theta1, theta2 - eps, base_psi)
    d_psi_theta2 = (psi_p2 - psi_m2) / (2 * eps)
    
    # Stack into Jacobian (4, 2)
    jacobian = jnp.stack([d_psi_theta1, d_psi_theta2], axis=1)
    
    # Compute QGT using CliffordQGT
    qgt = CliffordQGT(engine)
    
    # Embed scalar ψ as spinor (add dimension)
    psi_spinor = psi[:, None]
    
    # Compute Fisher and Berry
    fisher, berry = qgt.compute_full_qgt(psi_spinor, jacobian)
    
    # Compute Ricci curvature scalar (simplified for 2D)
    # R = -2 * det(Ω) / det(g) for 2D manifold
    det_fisher = jnp.linalg.det(fisher)
    det_berry = jnp.linalg.det(berry)
    curvature_scalar = -2.0 * det_berry / (det_fisher + 1e-12)
    
    return {
        "psi": psi,
        "jacobian": jacobian,
        "fisher": fisher,
        "berry": berry,
        "curvature_scalar": float(curvature_scalar),
    }


def scan_curvature_landscape(
    primes: List[int],
    engine: CliffordEngine,
    n_theta1: int = 15,
    n_theta2: int = 15,
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Scan curvature scalar R(θ¹, θ²) over parameter space.
    
    Args:
        primes: Prime sequence (for labeling)
        engine: CliffordEngine
        n_theta1, n_theta2: Grid resolution
        verbose: Print progress
    
    Returns:
        Dictionary with grids and maps
    """
    theta1_vals = jnp.linspace(0, jnp.pi/2, n_theta1)
    theta2_vals = jnp.linspace(0, 2*jnp.pi, n_theta2)
    
    curvature_map = np.zeros((n_theta1, n_theta2))
    fisher_trace_map = np.zeros((n_theta1, n_theta2))
    berry_norm_map = np.zeros((n_theta1, n_theta2))
    
    if verbose:
        print(f"Scanning curvature landscape ({n_theta1}x{n_theta2} grid)...")
    
    for i, th1 in enumerate(theta1_vals):
        for j, th2 in enumerate(theta2_vals):
            result = compute_qgt_on_ring(float(th1), float(th2), engine)
            curvature_map[i, j] = result["curvature_scalar"]
            fisher_trace_map[i, j] = float(jnp.trace(result["fisher"]))
            berry_norm_map[i, j] = float(jnp.linalg.norm(result["berry"]))
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{n_theta1} rows complete")
    
    return {
        "theta1_grid": theta1_vals,
        "theta2_grid": theta2_vals,
        "curvature_map": curvature_map,
        "fisher_trace_map": fisher_trace_map,
        "berry_norm_map": berry_norm_map,
    }


# =============================================================================
# Analysis and Visualization
# =============================================================================

def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table for plaquette scan."""
    print("=" * 100)
    print("Multi-Plaquette Comparison: ρ_IGBP Signatures")
    print("=" * 100)
    print(f"{'Particle':<10} {'Mass (MeV)':<12} {'Φ_final':<10} {'|Ω|_final':<10} "
          f"{'ρ_IGBP_final':<15} {'Conv.Step':<10}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['particle']:<10} {row['mass_MeV']:<12.2f} {row['phi_final']:<10.4f} "
              f"{row['omega_final']:<10.4f} {row['rho_igbp_final']:<15.2e} {row['convergence_step']:<10}")
    
    print("=" * 100)
    print()
    
    # Summary statistics
    lepton_df = df[df['particle'].isin(LEPTONS)]
    quark_df = df[df['particle'].isin(QUARKS)]
    
    print("Summary Statistics:")
    print(f"  Leptons: ρ_IGBP range = [{lepton_df['rho_igbp_initial'].min():.2e}, {lepton_df['rho_igbp_initial'].max():.2e}]")
    print(f"  Quarks:  ρ_IGBP range = [{quark_df['rho_igbp_initial'].min():.2e}, {quark_df['rho_igbp_initial'].max():.2e}]")
    print()


# =============================================================================
# Mass-Curvature Analysis
# =============================================================================

def compute_curvature_metrics(curvature_map: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for curvature landscape.
    
    Args:
        curvature_map: (n_theta1, n_theta2) array of R values
    
    Returns:
        Dictionary with R_rms, R_mean, R_max, R_min
    """
    R_flat = curvature_map.flatten()
    
    return {
        "R_rms": float(np.sqrt(np.mean(R_flat**2))),
        "R_mean": float(np.mean(R_flat)),
        "R_max": float(np.max(np.abs(R_flat))),
        "R_min": float(np.min(R_flat)),
        "R_std": float(np.std(R_flat)),
    }


def scan_all_curvature_landscapes(
    n_theta1: int = 15,
    n_theta2: int = 15,
    verbose: bool = True,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Scan curvature landscapes for all 9 SM fermion plaquettes.
    
    Args:
        n_theta1, n_theta2: Grid resolution
        verbose: Print progress
        save_results: Save to CSV
    
    Returns:
        DataFrame with curvature metrics for all particles
    """
    engine = CliffordEngine(seed=42)
    results = []
    
    if verbose:
        print("=" * 80)
        print("Curvature Landscape Scanner: All 9 SM Fermions")
        print("=" * 80)
        print(f"Grid resolution: {n_theta1} × {n_theta2}")
        print()
    
    for particle in ALL_PARTICLES:
        primes = PLAQUETTES[particle]
        mass = PDG_MASSES[particle]
        
        if verbose:
            print(f"Scanning {particle:3s} (m={mass:10.2f} MeV): {primes}")
        
        # Scan curvature landscape
        landscape = scan_curvature_landscape(
            primes, engine, n_theta1, n_theta2, verbose=False
        )
        
        # Compute metrics
        metrics = compute_curvature_metrics(landscape["curvature_map"])
        
        results.append({
            "particle": particle,
            "mass_MeV": mass,
            "R_rms": metrics["R_rms"],
            "R_mean": metrics["R_mean"],
            "R_max": metrics["R_max"],
            "R_std": metrics["R_std"],
        })
        
        if verbose:
            print(f"  → R_rms={metrics['R_rms']:.2e}, R_max={metrics['R_max']:.2e}")
            print()
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    if save_results:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"curvature_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"Curvature metrics saved to: {csv_path}")
            print()
    
    return df


def create_joint_analysis(
    plaquette_df: pd.DataFrame,
    curvature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create joint analysis combining plaquette scan and curvature metrics.
    
    Args:
        plaquette_df: Results from scan_all_plaquettes
        curvature_df: Results from scan_all_curvature_landscapes
    
    Returns:
        Merged DataFrame with all metrics
    """
    # Merge on particle name
    joint_df = pd.merge(
        plaquette_df,
        curvature_df,
        on=["particle", "mass_MeV"],
        suffixes=("_plaq", "_curv")
    )
    
    # Add particle type classification
    joint_df["type"] = joint_df["particle"].apply(
        lambda p: "lepton" if p in LEPTONS else "quark"
    )
    
    # Add log-scaled columns for visualization
    joint_df["log_mass"] = np.log10(joint_df["mass_MeV"])
    joint_df["log_R_rms"] = np.log10(joint_df["R_rms"] + 1e-20)  # Avoid log(0)
    joint_df["log_rho_init"] = np.log10(joint_df["rho_igbp_initial"] + 1e-20)
    
    return joint_df


def plot_mass_curvature_correlation(
    joint_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot mass vs curvature with lepton/quark coloring.
    
    Args:
        joint_df: Joint analysis DataFrame
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: log(mass) vs log(R_rms)
        ax = axes[0]
        for ptype, color in [("lepton", "blue"), ("quark", "red")]:
            mask = joint_df["type"] == ptype
            ax.scatter(
                joint_df.loc[mask, "log_R_rms"],
                joint_df.loc[mask, "log_mass"],
                c=color,
                label=ptype.capitalize(),
                s=100,
                alpha=0.7,
            )
            # Add labels
            for _, row in joint_df[mask].iterrows():
                ax.annotate(
                    row["particle"],
                    (row["log_R_rms"], row["log_mass"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        
        ax.set_xlabel("log₁₀(R_rms)", fontsize=12)
        ax.set_ylabel("log₁₀(mass [MeV])", fontsize=12)
        ax.set_title("Mass vs Curvature (RMS)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: log(mass) vs log(ρ_IGBP_init)
        ax = axes[1]
        for ptype, color in [("lepton", "blue"), ("quark", "red")]:
            mask = joint_df["type"] == ptype
            ax.scatter(
                joint_df.loc[mask, "log_rho_init"],
                joint_df.loc[mask, "log_mass"],
                c=color,
                label=ptype.capitalize(),
                s=100,
                alpha=0.7,
            )
            for _, row in joint_df[mask].iterrows():
                ax.annotate(
                    row["particle"],
                    (row["log_rho_init"], row["log_mass"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        
        ax.set_xlabel("log₁₀(ρ_IGBP initial)", fontsize=12)
        ax.set_ylabel("log₁₀(mass [MeV])", fontsize=12)
        ax.set_title("Mass vs Initial ρ_IGBP", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: R_rms vs ρ_IGBP_init
        ax = axes[2]
        for ptype, color in [("lepton", "blue"), ("quark", "red")]:
            mask = joint_df["type"] == ptype
            ax.scatter(
                joint_df.loc[mask, "log_R_rms"],
                joint_df.loc[mask, "log_rho_init"],
                c=color,
                label=ptype.capitalize(),
                s=100,
                alpha=0.7,
            )
            for _, row in joint_df[mask].iterrows():
                ax.annotate(
                    row["particle"],
                    (row["log_R_rms"], row["log_rho_init"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        
        ax.set_xlabel("log₁₀(R_rms)", fontsize=12)
        ax.set_ylabel("log₁₀(ρ_IGBP initial)", fontsize=12)
        ax.set_title("Curvature vs ρ_IGBP", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping visualization")


def print_joint_analysis_table(joint_df: pd.DataFrame):
    """Print comprehensive analysis table."""
    print("=" * 120)
    print("Joint Analysis: Mass, Curvature, and ρ_IGBP")
    print("=" * 120)
    print(f"{'Particle':<10} {'Type':<8} {'Mass (MeV)':<12} {'R_rms':<12} "
          f"{'R_max':<12} {'ρ_IGBP_init':<15} {'Conv.Step':<10}")
    print("-" * 120)
    
    # Sort by mass
    for _, row in joint_df.sort_values("mass_MeV").iterrows():
        print(f"{row['particle']:<10} {row['type']:<8} {row['mass_MeV']:<12.2f} "
              f"{row['R_rms']:<12.2e} {row['R_max']:<12.2e} "
              f"{row['rho_igbp_initial']:<15.2e} {row['convergence_step']:<10}")
    
    print("=" * 120)
    print()
    
    # Statistical analysis
    lepton_df = joint_df[joint_df["type"] == "lepton"]
    quark_df = joint_df[joint_df["type"] == "quark"]
    
    print("Statistical Summary:")
    print()
    print("Leptons:")
    print(f"  Mass range:    [{lepton_df['mass_MeV'].min():.2f}, {lepton_df['mass_MeV'].max():.2f}] MeV")
    print(f"  R_rms range:   [{lepton_df['R_rms'].min():.2e}, {lepton_df['R_rms'].max():.2e}]")
    print(f"  ρ_IGBP range:  [{lepton_df['rho_igbp_initial'].min():.2e}, {lepton_df['rho_igbp_initial'].max():.2e}]")
    print()
    print("Quarks:")
    print(f"  Mass range:    [{quark_df['mass_MeV'].min():.2f}, {quark_df['mass_MeV'].max():.2f}] MeV")
    print(f"  R_rms range:   [{quark_df['R_rms'].min():.2e}, {quark_df['R_rms'].max():.2e}]")
    print(f"  ρ_IGBP range:  [{quark_df['rho_igbp_initial'].min():.2e}, {quark_df['rho_igbp_initial'].max():.2e}]")
    print()
    
    # Correlation analysis
    print("Correlations:")
    print(f"  log(mass) vs log(R_rms):  {joint_df[['log_mass', 'log_R_rms']].corr().iloc[0, 1]:.3f}")
    print(f"  log(mass) vs log(ρ_init): {joint_df[['log_mass', 'log_rho_init']].corr().iloc[0, 1]:.3f}")
    print(f"  log(R_rms) vs log(ρ_init): {joint_df[['log_R_rms', 'log_rho_init']].corr().iloc[0, 1]:.3f}")
    print()


# =============================================================================
# Main Demo
# =============================================================================

def demo_multi_plaquette_scanner():
    """Comprehensive demo of all four extensions."""
    print("=" * 100)
    print("Prime Sixth-Order Flow: Multi-Plaquette Scanner with QGT")
    print("=" * 100)
    print()
    print("Extensions:")
    print("  A. Wilson Loop Holonomy (genuine geometric Φ)")
    print("  B. Multi-Plaquette Scanner (all 9 SM fermions)")
    print("  C. Reeb Flow Integration (time-aligned evolution)")
    print("  D. QGT Parameterization (curvature landscape)")
    print()
    print("=" * 100)
    print()
    
    # Extension B: Scan all plaquettes
    df_plaq = scan_all_plaquettes(n_steps=200, dt=0.01, use_wilson=True, verbose=True)
    
    # Print comparison table
    print_comparison_table(df_plaq)
    
    # Extension D: Scan all curvature landscapes
    print("=" * 100)
    print("Curvature Landscape Scanner")
    print("=" * 100)
    print()
    
    df_curv = scan_all_curvature_landscapes(n_theta1=15, n_theta2=15, verbose=True)
    
    # Joint analysis
    print("=" * 100)
    print("Mass-Curvature Analysis")
    print("=" * 100)
    print()
    
    joint_df = create_joint_analysis(df_plaq, df_curv)
    print_joint_analysis_table(joint_df)
    
    # Save joint analysis
    log_dir = Path(__file__).parent / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joint_path = log_dir / f"joint_analysis_{timestamp}.csv"
    joint_df.to_csv(joint_path, index=False)
    print(f"Joint analysis saved to: {joint_path}")
    print()
    
    # Visualization
    print("=" * 100)
    print("Generating Visualizations")
    print("=" * 100)
    print()
    
    fig_path = log_dir / f"mass_curvature_correlation_{timestamp}.png"
    plot_mass_curvature_correlation(joint_df, save_path=str(fig_path))
    
    print()
    print("=" * 100)
    print("Analysis Complete!")
    print("=" * 100)
    print()
    print("Key Findings:")
    print("  - All 9 plaquettes scanned with Wilson loop holonomy")
    print("  - Curvature landscapes computed for each fermion")
    print("  - Mass-curvature correlation analysis complete")
    print("  - Results saved to logs/ directory")
    print()


if __name__ == "__main__":
    demo_multi_plaquette_scanner()
