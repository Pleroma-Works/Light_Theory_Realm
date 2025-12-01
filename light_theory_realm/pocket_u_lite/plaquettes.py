"""
Pocket_U Lite: Standard Model Plaquettes and Calibrated Constants
==================================================================

This module contains the frozen snapshot from Pocket_U v1 (THEORY_V1_FINAL):
- Prime plaquettes for all 9 Standard Model fermions
- PDG target masses in MeV
- Calibrated universal constants (α, Λ, factors)

These values are the result of validated research in THRML_Geometry and are
provided here as a read-only reference for the public Light Theory Realm engine.
"""

# ============================================================================
# Calibrated Universal Constants (from THEORY_V1_FINAL.md)
# ============================================================================

# Zeta screening coupling
ALPHA = -3.812602

# Fundamental lattice energy scale (MeV) - the cosmological constant Λ
SCALE_MEV = 1731.918381

# S6 outer automorphism factor for down-type quarks
DOWN_FACTOR = 11.0 / 12.0

# Pole amplification factor for top quark
TOP_POLE_FACTOR = 97.419

# ============================================================================
# Standard Model Prime Plaquettes
# ============================================================================

PLAQUETTES = {
    # Leptons
    "e":   [2, 3, 5, 7, 2],      # Electron: vacuum bubble (~100% screened)
    "mu":  [2, 17, 23, 7, 2],    # Muon: stable knot (~2% screening)
    "tau": [3, 11, 13, 47, 3],   # Tau: pure geometry (anchor)
    
    # Up-type quarks
    "u": [13, 53, 73, 89, 13],   # Up: twin-prime angles
    "c": [11, 47, 83, 113, 11],  # Charm: exact match
    "t": [3, 11, 13, 47, 3],     # Top: uses tau plaquette + pole factor
    
    # Down-type quarks (with 11/12 factor)
    "d": [2, 5, 3, 7, 2],        # Down: 11/12 outer automorphism
    "s": [17, 61, 67, 83, 17],   # Strange: stable knot
    "b": [3, 19, 43, 59, 3],     # Bottom: exact match
}

# ============================================================================
# PDG Target Masses (MeV)
# ============================================================================

PDG_MASSES = {
    # Leptons
    "e":   0.51099895,
    "mu":  105.6583755,
    "tau": 1776.86,
    
    # Up-type quarks
    "u": 2.16,
    "c": 1270.0,
    "t": 172690.0,
    
    # Down-type quarks
    "d": 4.67,
    "s": 93.0,
    "b": 4180.0,
}

# ============================================================================
# Particle Classification
# ============================================================================

LEPTONS = ["e", "mu", "tau"]
UP_TYPE_QUARKS = ["u", "c", "t"]
DOWN_TYPE_QUARKS = ["d", "s", "b"]
QUARKS = UP_TYPE_QUARKS + DOWN_TYPE_QUARKS
ALL_PARTICLES = LEPTONS + QUARKS

# ============================================================================
# Helper Functions
# ============================================================================

def get_plaquette(name: str) -> list[int]:
    """
    Get the prime plaquette for a given particle.
    
    Args:
        name: Particle name (e.g., "e", "mu", "tau", "u", "d", etc.)
        
    Returns:
        List of primes forming the closed plaquette loop
        
    Raises:
        KeyError: If particle name is not recognized
    """
    return PLAQUETTES[name]


def get_pdg_mass(name: str) -> float:
    """
    Get the PDG target mass for a given particle in MeV.
    
    Args:
        name: Particle name
        
    Returns:
        PDG mass in MeV
        
    Raises:
        KeyError: If particle name is not recognized
    """
    return PDG_MASSES[name]


def is_down_type(name: str) -> bool:
    """Check if particle is a down-type quark (requires 11/12 factor)."""
    return name in DOWN_TYPE_QUARKS


def requires_pole_factor(name: str) -> bool:
    """Check if particle requires pole amplification (top quark)."""
    return name == "t"
