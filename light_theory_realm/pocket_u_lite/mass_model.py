"""
Pocket_U Lite: Mass Model
==========================

Implements the Pocket_U v1 mass formula using Light Theory Realm primitives:
    m_phys(γ) = Λ · |M_geom(γ) + α·ζ_vac(γ)|

This module computes physical masses from prime plaquettes by:
1. Computing geometric mass via SU(3) Wilson loop tension
2. Computing vacuum screening via zeta correction
3. Combining with calibrated constants and particle-specific factors

All computations use only Light_Theory_Realm components (no THRML dependencies).
"""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from typing import Dict, List

from .plaquettes import (
    ALPHA, SCALE_MEV, DOWN_FACTOR, TOP_POLE_FACTOR,
    get_plaquette, get_pdg_mass, is_down_type, requires_pole_factor
)

# ============================================================================
# SU(3) Gell-Mann Generators (from THRML_Geometry/zeta_physics.py)
# ============================================================================

# Exact Gell-Mann matrices for SU(3) gauge group
GELL_MANN_GENERATORS = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3),
]

# ============================================================================
# Wilson Loop Computation
# ============================================================================

def compute_wilson_loop(primes: List[int], coupling_g: float = 1.0) -> np.ndarray:
    """
    Compute the SU(3) Wilson loop operator for a prime plaquette.
    
    The Wilson loop is constructed as a product of link operators:
        W = U_{p1→p2} · U_{p2→p3} · ... · U_{pN→p1}
    
    Each link operator is:
        U_{pi→pj} = exp(i·g·θ·λ_k)
    where:
        - θ = ln(pj/pi) is the log-energy distance
        - λ_k is the Gell-Mann generator with k = (pi·pj) mod 8
        - g is the coupling strength
    
    Args:
        primes: Closed prime plaquette loop (e.g., [2, 3, 5, 7, 2])
        coupling_g: SU(3) coupling strength (default: 1.0)
        
    Returns:
        3x3 complex Wilson loop matrix W
    """
    W = np.eye(3, dtype=complex)
    
    for i in range(len(primes) - 1):
        p_i, p_j = primes[i], primes[i + 1]
        
        # Generator index from prime product (mod 8 rule)
        gen_idx = (p_i * p_j) % 8
        
        # Log-energy angle
        theta = np.log(p_j / p_i)
        
        # Link operator: U = exp(i·g·θ·λ)
        U = expm(1j * coupling_g * theta * GELL_MANN_GENERATORS[gen_idx])
        
        # Accumulate Wilson loop
        W = W @ U
    
    return W


def geometric_mass(primes: List[int], coupling_g: float = 1.0) -> float:
    """
    Compute the geometric mass M_geom from Wilson loop tension.
    
    The geometric mass is extracted from the Wilson loop phase:
        M_geom = |angle(det(W))|
    
    If the determinant phase is too small (< 1e-6), we fall back to:
        M_geom = |trace(W)|
    
    Args:
        primes: Prime plaquette loop
        coupling_g: SU(3) coupling strength
        
    Returns:
        Dimensionless geometric mass M_geom
    """
    W = compute_wilson_loop(primes, coupling_g)
    
    # Primary: phase of determinant
    m_geom = np.abs(np.angle(np.linalg.det(W)))
    
    # Fallback: trace magnitude for small phases
    if m_geom < 1e-6:
        m_geom = np.abs(np.trace(W))
    
    return float(m_geom)


def vacuum_screening(primes: List[int], W: np.ndarray) -> float:
    """
    Compute the vacuum screening correction ζ_vac.
    
    The zeta vacuum correction is:
        ζ_vac = Re(Tr(W)) / p_char
    where p_char is the geometric mean of the unique primes in the loop.
    
    Args:
        primes: Prime plaquette loop
        W: Wilson loop matrix (from compute_wilson_loop)
        
    Returns:
        Dimensionless vacuum screening term ζ_vac
    """
    # Real part of Wilson loop trace
    trace_real = np.real(np.trace(W))
    
    # Geometric mean of unique primes (characteristic scale)
    unique_primes = list(set(primes[:-1]))  # Exclude closing prime
    p_char = np.exp(np.mean(np.log(unique_primes)))
    
    # Zeta screening term
    zeta_vac = trace_real / p_char
    
    return float(zeta_vac)


# ============================================================================
# Physical Mass Computation
# ============================================================================

def physical_mass(name: str, coupling_g: float = 1.0) -> Dict:
    """
    Compute the full physical mass for a Standard Model fermion.
    
    This is the main API for Pocket_U Lite mass predictions. It implements
    the complete Pocket_U v1 formula:
    
        m_phys = Λ · |M_geom + α·ζ_vac| · [factors]
    
    where:
        - Λ = SCALE_MEV (fundamental energy scale)
        - M_geom = geometric mass from Wilson loop tension
        - α = ALPHA (zeta screening coupling)
        - ζ_vac = vacuum screening correction
        - [factors] = particle-specific corrections (down-type, top pole)
    
    Args:
        name: Particle name (e.g., "e", "mu", "tau", "u", "d", etc.)
        coupling_g: SU(3) coupling strength (default: 1.0)
        
    Returns:
        Dictionary containing:
            - plaquette: Prime loop
            - m_geom_dim: Dimensionless geometric mass
            - m_vac_dim: Dimensionless vacuum correction (α·ζ_vac)
            - m_total_signed: Signed sum before absolute value
            - m_total_mag: Magnitude after absolute value
            - m_geom_MeV: Geometric mass in MeV
            - m_vac_MeV: Vacuum correction in MeV
            - m_phys_MeV: Final physical mass in MeV
            - screening_pct: Screening percentage
            - pole_factor: Pole amplification (if applicable)
            - pdg_mass_MeV: PDG target mass
            - error_pct: Prediction error percentage
            
    Raises:
        KeyError: If particle name is not recognized
    """
    # Get plaquette and PDG target
    primes = get_plaquette(name)
    pdg_mass = get_pdg_mass(name)
    
    # 1. Compute Wilson loop
    W = compute_wilson_loop(primes, coupling_g)
    
    # 2. Geometric mass
    m_geom = geometric_mass(primes, coupling_g)
    
    # 3. Vacuum screening
    zeta_vac = vacuum_screening(primes, W)
    m_vac = ALPHA * zeta_vac
    
    # 4. Combine (before absolute value)
    m_total_signed = m_geom + m_vac
    
    # 5. Apply particle-specific factors
    pole_factor = 1.0
    if requires_pole_factor(name):
        pole_factor = TOP_POLE_FACTOR
        m_total_signed *= pole_factor
    elif is_down_type(name):
        m_total_signed *= DOWN_FACTOR
    
    # 6. Apply absolute value (CRITICAL: this was missing in original code)
    m_total_mag = np.abs(m_total_signed)
    
    # 7. Scale to MeV
    m_phys = SCALE_MEV * m_total_mag
    
    # 8. Compute diagnostics
    screening_pct = -m_vac / m_geom * 100 if m_geom != 0 else 0
    error_pct = (m_phys - pdg_mass) / pdg_mass * 100 if pdg_mass != 0 else 0
    
    return {
        "name": name,
        "plaquette": primes,
        "m_geom_dim": float(m_geom),
        "m_vac_dim": float(m_vac),
        "m_total_signed": float(m_total_signed / pole_factor if pole_factor != 1.0 else m_total_signed),
        "m_total_mag": float(m_total_mag),
        "m_geom_MeV": float(SCALE_MEV * m_geom),
        "m_vac_MeV": float(SCALE_MEV * m_vac),
        "m_phys_MeV": float(m_phys),
        "screening_pct": float(screening_pct),
        "pole_factor": float(pole_factor) if pole_factor != 1.0 else None,
        "pdg_mass_MeV": float(pdg_mass),
        "error_pct": float(error_pct),
    }
