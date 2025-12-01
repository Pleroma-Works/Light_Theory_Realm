"""
Pocket_U Lite: Geometry Diagnostics
====================================

Mandatory geometric diagnostics for each Standard Model particle using
Light Theory Realm primitives:
    - Quantum Geometric Tensor (Fisher metric + Berry curvature)
    - 5D Kaluza-Klein metric
    - Reeb flow dynamics (resonance density ξ and time direction R)

This module demonstrates that each particle is not just a number, but has
a complete geometric and topological fingerprint.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple
import sys
import os

# Import Light Theory Realm components
# Note: Using relative imports from parent package
try:
    from ..engine import CliffordEngine
    from ..qgt import CliffordQGT
    from ..experiments.prime_gauge.uplift import KaluzaKleinUplift
    from ..experiments.prime_gauge.reeb_flow import ReebFlowDynamics
except ImportError:
    # Fallback for direct execution / testing
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import CliffordEngine
    from qgt import CliffordQGT
    from experiments.prime_gauge.uplift import KaluzaKleinUplift
    from experiments.prime_gauge.reeb_flow import ReebFlowDynamics

from .plaquettes import get_plaquette

# ============================================================================
# Global Engine Instances (shared across all geometry computations)
# ============================================================================

_ENGINE = CliffordEngine(seed=137)  # Fixed seed for reproducibility
_QGT = CliffordQGT(_ENGINE)
_KK = KaluzaKleinUplift(_ENGINE)
_REEB = ReebFlowDynamics(_ENGINE)

# ============================================================================
# Trajectory Construction
# ============================================================================

def _build_plaquette_trajectory(
    primes: list[int], 
    steps: int = 16
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a parameterized trajectory ψ(θ) around a prime plaquette.
    
    This constructs a simple 1D rotation path in Clifford spinor space.
    
    Args:
        primes: Prime plaquette loop
        steps: Number of points along the trajectory
        
    Returns:
        Tuple of (thetas, psi_trajectory):
            - thetas: (steps,) parameter values
            - psi_trajectory: (steps, 4) spinor states
    """
    # Parameter range
    thetas = jnp.linspace(0.0, 2.0 * jnp.pi, steps)
    
    # Reference spinor (deterministic from plaquette)
    key = jax.random.PRNGKey(sum(primes))
    psi_0 = _ENGINE.random_spinor(key)
    psi_0 = psi_0 / jnp.linalg.norm(psi_0)
    
    # Simple bivector for rotation (gamma_0 ^ gamma_1)
    bivector = _ENGINE.wedge_product(_ENGINE.gammas[0], _ENGINE.gammas[1])
    
    # Scale by characteristic log-energy from plaquette
    p1, p2 = float(primes[0]), float(primes[1])
    angle_scale = jnp.log(p2 / p1)
    bivector = bivector * angle_scale
    
    # Define trajectory: ψ(θ) ≈ (I + iθB)·ψ₀ (first-order rotation)
    def psi_at_theta(theta):
        R = jnp.eye(4, dtype=jnp.complex128) + 1j * theta * bivector
        psi = R @ psi_0
        # Normalize and squeeze to vector (4,)
        return jnp.squeeze(psi / jnp.linalg.norm(psi))
    
    # Compute trajectory
    psi_trajectory = jax.vmap(psi_at_theta)(thetas)
    
    return thetas, psi_trajectory


# ============================================================================
# QGT Diagnostics
# ============================================================================

def compute_qgt_diagnostics(primes: list[int]) -> Dict:
    """
    Compute Quantum Geometric Tensor diagnostics for a plaquette.
    
    The QGT decomposes into:
        - Fisher metric g_μν (real, symmetric): information distance
        - Berry curvature Ω_μν (imaginary, antisymmetric): topological twist
    
    Args:
        primes: Prime plaquette loop
        
    Returns:
        Dictionary containing:
            - fisher: (d, d) Fisher information metric
            - berry: (d, d) Berry curvature
            - fisher_trace: Scalar trace of Fisher metric
            - berry_norm: Frobenius norm of Berry curvature
    """
    # Build trajectory
    thetas, psi_trajectory = _build_plaquette_trajectory(primes)
    
    # Use middle point for QGT computation
    mid_idx = len(thetas) // 2
    i0 = max(mid_idx - 1, 0)
    i1 = mid_idx
    i2 = min(mid_idx + 1, len(thetas) - 1)
    
    theta0, theta1, theta2 = thetas[i0], thetas[i1], thetas[i2]
    psi0, psi1, psi2 = psi_trajectory[i0], psi_trajectory[i1], psi_trajectory[i2]
    
    # Compute finite-difference derivative at mid-point
    if i0 < i1 and i1 < i2:
        dpsi_dtheta = (psi2 - psi0) / (theta2 - theta0)
    elif i1 < i2:
        dpsi_dtheta = (psi2 - psi1) / (theta2 - theta1)
    else:
        dpsi_dtheta = (psi1 - psi0) / (theta1 - theta0)
    
    # Shape (4, 1): n components × 1 parameter
    jac = dpsi_dtheta.reshape((-1, 1))
    
    # Compute QGT
    fisher, berry = _QGT.compute_full_qgt(psi1, jac)
    
    # Diagnostics
    fisher_trace = float(jnp.trace(fisher))
    berry_norm = float(jnp.linalg.norm(berry))
    
    return {
        "fisher": fisher,
        "berry": berry,
        "fisher_trace": fisher_trace,
        "berry_norm": berry_norm,
    }


# ============================================================================
# Kaluza-Klein Metric
# ============================================================================

def compute_kk_metric(primes: list[int], phi_scalar: float = 0.925) -> jnp.ndarray:
    """
    Compute the 5D Kaluza-Klein metric for a plaquette.
    
    The KK metric unifies the 4D information geometry with a 5th dimension:
        G_AB = [ g_μν + φ²A_μA_ν    φ²A_μ ]
               [     φ²A_ν           φ²   ]
    
    where:
        - g_μν: Fisher metric (base geometry)
        - A_μ: Berry connection (gauge field)
        - φ: Dilaton field (extra dimension size)
    
    Args:
        primes: Prime plaquette loop
        phi_scalar: Dilaton field value (default: 0.925)
        
    Returns:
        (d+1, d+1) 5D Kaluza-Klein metric
    """
    # Build trajectory
    thetas, psi_trajectory = _build_plaquette_trajectory(primes)
    
    # Use middle point
    mid_idx = len(thetas) // 2
    i0 = max(mid_idx - 1, 0)
    i1 = mid_idx
    i2 = min(mid_idx + 1, len(thetas) - 1)
    
    theta0, theta1, theta2 = thetas[i0], thetas[i1], thetas[i2]
    psi0, psi1, psi2 = psi_trajectory[i0], psi_trajectory[i1], psi_trajectory[i2]
    
    # Same finite-diff scheme as QGT diagnostics
    if i0 < i1 and i1 < i2:
        dpsi_dtheta = (psi2 - psi0) / (theta2 - theta0)
    elif i1 < i2:
        dpsi_dtheta = (psi2 - psi1) / (theta2 - theta1)
    else:
        dpsi_dtheta = (psi1 - psi0) / (theta1 - theta0)
    
    jac = dpsi_dtheta.reshape((-1, 1))
    
    # Construct 5D metric
    G_5D = _KK.construct_5d_metric(psi1, jac, phi_scalar=phi_scalar)
    
    return G_5D


# ============================================================================
# Reeb Flow Diagnostics
# ============================================================================

def compute_reeb_diagnostics(
    primes: list[int],
    fisher: jnp.ndarray = None,
    berry: jnp.ndarray = None
) -> Dict:
    """
    Compute Reeb flow diagnostics for a plaquette.
    
    Reeb flow describes the "time direction" and "dark energy" of the geometry:
        - ξ (xi): Resonance density (vacuum energy / dark energy)
        - R: Reeb vector (time direction in parameter space)
    
    The Reeb vector satisfies: i_R Ω = 0 (null space of Berry curvature)
    
    Args:
        primes: Prime plaquette loop
        fisher: Optional pre-computed Fisher metric
        berry: Optional pre-computed Berry curvature
        
    Returns:
        Dictionary containing:
            - xi: Resonance density (dark energy scalar)
            - R: Reeb vector (time direction)
            - R_norm: Norm of Reeb vector
    """
    # Compute resonance density from primes
    xi = _REEB.compute_resonance_density(primes)
    
    # If Fisher/Berry not provided, compute them
    if fisher is None or berry is None:
        qgt_diag = compute_qgt_diagnostics(primes)
        fisher = qgt_diag["fisher"]
        berry = qgt_diag["berry"]
    
    # Compute Reeb vector
    R = _REEB.compute_reeb_vector(fisher, berry)
    R_norm = float(jnp.linalg.norm(R))
    
    return {
        "xi": float(xi),
        "R": R,
        "R_norm": R_norm,
    }


# ============================================================================
# Complete Particle Geometry
# ============================================================================

def get_particle_geometry(name: str, phi_scalar: float = 0.925) -> Dict:
    """
    Get the complete geometric profile for a Standard Model particle.
    
    This is the main API for Pocket_U Lite geometry diagnostics. It returns
    the full geometric fingerprint of a particle, including:
        - QGT (Fisher + Berry)
        - 5D KK metric
        - Reeb flow (ξ + R)
    
    Args:
        name: Particle name (e.g., "e", "mu", "tau", "u", "d", etc.)
        phi_scalar: Dilaton field value for KK metric (default: 0.925)
        
    Returns:
        Dictionary containing:
            - name: Particle name
            - primes: Prime plaquette loop
            - fisher: (d, d) Fisher metric
            - berry: (d, d) Berry curvature
            - fisher_trace: Trace of Fisher metric
            - berry_norm: Frobenius norm of Berry curvature
            - G_5D: (d+1, d+1) Kaluza-Klein metric
            - xi: Resonance density (dark energy)
            - R: Reeb vector (time direction)
            - R_norm: Norm of Reeb vector
            
    Raises:
        KeyError: If particle name is not recognized
    """
    # Get plaquette
    primes = get_plaquette(name)
    
    # Compute QGT diagnostics
    qgt_diag = compute_qgt_diagnostics(primes)
    
    # Compute KK metric
    G_5D = compute_kk_metric(primes, phi_scalar)
    
    # Compute Reeb diagnostics (reuse Fisher/Berry from QGT)
    reeb_diag = compute_reeb_diagnostics(
        primes,
        fisher=qgt_diag["fisher"],
        berry=qgt_diag["berry"]
    )
    
    return {
        "name": name,
        "primes": primes,
        "fisher": qgt_diag["fisher"],
        "berry": qgt_diag["berry"],
        "fisher_trace": qgt_diag["fisher_trace"],
        "berry_norm": qgt_diag["berry_norm"],
        "G_5D": G_5D,
        "xi": reeb_diag["xi"],
        "R": reeb_diag["R"],
        "R_norm": reeb_diag["R_norm"],
    }
