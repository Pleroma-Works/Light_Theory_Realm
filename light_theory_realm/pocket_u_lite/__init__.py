"""
Pocket_U Lite: Standard Model Toy
==================================

A read-only Standard Model demonstration built on Light Theory Realm.

This module provides:
    - Mass predictions for all 9 SM fermions from prime plaquettes
    - Mandatory geometry diagnostics (QGT, KK, Reeb) for each particle
    - User-friendly API for exploring the geometric Standard Model

Example:
    >>> from light_theory_realm import print_sm_table, get_particle_profile
    >>> 
    >>> # Print the full SM spectrum
    >>> print_sm_table()
    >>> 
    >>> # Get complete profile for electron
    >>> e_profile = get_particle_profile("e")
    >>> print(f"Mass: {e_profile['m_phys_MeV']:.3f} MeV")
    >>> print(f"Dark energy ξ: {e_profile['xi']:.4f}")
"""

from .sm_table import (
    get_sm_spectrum,
    print_sm_table,
    get_particle_profile,
)

from .geometry import get_particle_geometry

from .mass_model import physical_mass

from .plaquettes import (
    PLAQUETTES,
    PDG_MASSES,
    ALPHA,
    SCALE_MEV,
    ALL_PARTICLES,
)

from .koide import (
    koide_ratio,
    check_koide_triplet,
    predict_third_mass,
    koide_angle,
    descartes_angle,
)

__all__ = [
    # Main user-facing API
    "get_sm_spectrum",
    "print_sm_table",
    "get_particle_profile",
    
    # Geometry diagnostics
    "get_particle_geometry",
    
    # Mass model
    "physical_mass",
    
    # Koide equation utilities
    "koide_ratio",
    "check_koide_triplet",
    "predict_third_mass",
    "koide_angle",
    "descartes_angle",
    
    # Constants and data
    "PLAQUETTES",
    "PDG_MASSES",
    "ALPHA",
    "SCALE_MEV",
    "ALL_PARTICLES",
]
