"""
Light Theory Realm
==================

A geometry-first JAX engine for Light Theory experiments.

Like TensorFlow Quantum, but built around the Quantum Geometric Tensor,
gauge structure, and prime-gauge experiments instead of just circuits and gates.

Core Components:
    - CliffordEngine: Cl(1,3) algebra implementation
    - CliffordQGT: Quantum Geometric Tensor (Fisher + Berry)
    - KaluzaKleinUplift: 5D metric construction
    - PrimeLattice: Classical prime-gauge parameters
    - PrimeWilsonLoop: Quantum Wilson loop operators
    - ReebFlowDynamics: Dark Energy as vacuum resonance

Example:
    >>> from src import CliffordEngine, CliffordQGT
    >>> engine = CliffordEngine(seed=42)
    >>> qgt = CliffordQGT(engine)
    >>> fisher, berry = qgt.compute_full_qgt(psi, jacobian)
"""

__version__ = "0.1.1"

# Core engine (always available)
from .engine import CliffordEngine
from .qgt import CliffordQGT

__all__ = ["CliffordEngine", "CliffordQGT"]

# Experiments (optional, but should work with core dependencies)
try:
    from .experiments.prime_gauge.uplift import KaluzaKleinUplift
    __all__.append("KaluzaKleinUplift")
except Exception:
    pass

try:
    from .experiments.prime_gauge.prime_plaquette import PrimeLattice, PrimeWilsonLoop
    __all__.extend(["PrimeLattice", "PrimeWilsonLoop"])
except Exception:
    pass

try:
    from .experiments.prime_gauge.reeb_flow import ReebFlowDynamics
    __all__.append("ReebFlowDynamics")
except Exception:
    pass

# Pocket_U Lite (Standard Model toy - requires core dependencies only)
try:
    from .pocket_u_lite import (
        get_sm_spectrum,
        print_sm_table,
        get_particle_profile,
        get_particle_geometry,
        koide_ratio,
        koide_angle,
        predict_third_mass,
    )
    __all__.extend([
        "get_sm_spectrum",
        "print_sm_table",
        "get_particle_profile",
        "get_particle_geometry",
        "koide_ratio",
        "koide_angle",
        "predict_third_mass",
    ])
except ImportError:
    # Pocket_U Lite not available
    pass

# Models (require optional dependencies)
try:
    from .models.classical_clock import U1ClockTHRMLModel
    __all__.append("U1ClockTHRMLModel")
except Exception:
    # THRML not installed
    pass

try:
    from .models.quantum_qhbm import QuantumLayer, LatentDistribution
    __all__.extend(["QuantumLayer", "LatentDistribution"])
except Exception:
    # PennyLane not installed or has compatibility issues
    pass
