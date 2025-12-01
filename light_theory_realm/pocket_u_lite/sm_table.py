"""
Pocket_U Lite: Standard Model Table API
========================================

User-facing convenience functions for accessing the Standard Model spectrum
and particle profiles. This is the main public API for Pocket_U Lite.
"""

from typing import List, Dict
from .plaquettes import ALL_PARTICLES, get_pdg_mass
from .mass_model import physical_mass
from .geometry import get_particle_geometry


def get_sm_spectrum() -> List[Dict]:
    """
    Get the complete Standard Model mass spectrum.
    
    Computes mass predictions for all 9 Standard Model fermions
    (3 leptons + 6 quarks) and compares with PDG target values.
    
    Returns:
        List of dictionaries, one per particle, containing:
            - name: Particle name
            - mass_pdg: PDG target mass (MeV)
            - mass_pred: Predicted mass (MeV)
            - error_pct: Prediction error (%)
            - plaquette: Prime plaquette loop
            
    Example:
        >>> spectrum = get_sm_spectrum()
        >>> for particle in spectrum:
        ...     print(f"{particle['name']}: {particle['mass_pred']:.3f} MeV")
    """
    spectrum = []
    
    for name in ALL_PARTICLES:
        result = physical_mass(name)
        spectrum.append({
            "name": name,
            "mass_pdg": result["pdg_mass_MeV"],
            "mass_pred": result["m_phys_MeV"],
            "error_pct": result["error_pct"],
            "plaquette": result["plaquette"],
        })
    
    return spectrum


def print_sm_table() -> None:
    """
    Print a formatted table of the Standard Model mass spectrum.
    
    Displays all 9 fermions with PDG targets, predictions, and errors.
    This provides a quick visual verification of Pocket_U Lite's accuracy.
    
    Example Output:
        Pocket_U Lite: Standard Model Mass Spectrum
        ============================================
        
        Particle   PDG (MeV)    Predicted (MeV)   Error (%)   Plaquette
        --------   ---------    ---------------   ---------   ---------
        e              0.511          0.511           0.0%    [2,3,5,7,2]
        mu           105.658        105.240           0.4%    [2,17,23,7,2]
        ...
    """
    spectrum = get_sm_spectrum()
    
    # Header
    print("\nPocket_U Lite: Standard Model Mass Spectrum")
    print("=" * 80)
    print(f"{'Particle':<10} {'PDG (MeV)':>12} {'Predicted (MeV)':>18} {'Error (%)':>12}   {'Plaquette'}")
    print("-" * 80)
    
    # Rows
    for p in spectrum:
        plaq_str = str(p['plaquette'][:4])  # Show first 4 primes
        print(f"{p['name']:<10} {p['mass_pdg']:>12.3f} {p['mass_pred']:>18.3f} {p['error_pct']:>11.1f}%   {plaq_str}")
    
    print("=" * 80)
    print(f"Total particles: {len(spectrum)}")
    print(f"Average error: {sum(abs(p['error_pct']) for p in spectrum) / len(spectrum):.2f}%\n")


def get_particle_profile(name: str, phi_scalar: float = 0.925) -> Dict:
    """
    Get the complete physical and geometric profile for a particle.
    
    This is the main high-level API that combines:
        - Mass prediction (from mass_model)
        - Geometry diagnostics (from geometry)
    
    Args:
        name: Particle name (e.g., "e", "mu", "tau", "u", "d", etc.)
        phi_scalar: Dilaton field value for KK metric (default: 0.925)
        
    Returns:
        Dictionary containing all mass and geometry information:
        
        Mass fields:
            - name: Particle name
            - plaquette: Prime loop
            - m_phys_MeV: Physical mass (MeV)
            - pdg_mass_MeV: PDG target (MeV)
            - error_pct: Prediction error (%)
            - m_geom_MeV: Geometric mass component
            - m_vac_MeV: Vacuum screening component
            - screening_pct: Screening percentage
            - pole_factor: Pole amplification (if applicable)
            
        Geometry fields:
            - fisher: (d, d) Fisher information metric
            - berry: (d, d) Berry curvature
            - fisher_trace: Trace of Fisher metric
            - berry_norm: Frobenius norm of Berry curvature
            - G_5D: (d+1, d+1) Kaluza-Klein metric
            - xi: Resonance density (dark energy)
            - R: Reeb vector (time direction)
            - R_norm: Norm of Reeb vector
            
    Example:
        >>> profile = get_particle_profile("e")
        >>> print(f"Electron mass: {profile['m_phys_MeV']:.3f} MeV")
        >>> print(f"Screening: {profile['screening_pct']:.1f}%")
        >>> print(f"Dark energy ξ: {profile['xi']:.4f}")
        >>> print(f"Fisher metric shape: {profile['fisher'].shape}")
    """
    # Get mass prediction
    mass_result = physical_mass(name)
    
    # Get geometry diagnostics
    geom_result = get_particle_geometry(name, phi_scalar)
    
    # Merge results
    profile = {
        # Mass fields
        "name": mass_result["name"],
        "plaquette": mass_result["plaquette"],
        "m_phys_MeV": mass_result["m_phys_MeV"],
        "pdg_mass_MeV": mass_result["pdg_mass_MeV"],
        "error_pct": mass_result["error_pct"],
        "m_geom_MeV": mass_result["m_geom_MeV"],
        "m_vac_MeV": mass_result["m_vac_MeV"],
        "screening_pct": mass_result["screening_pct"],
        "pole_factor": mass_result["pole_factor"],
        
        # Geometry fields
        "fisher": geom_result["fisher"],
        "berry": geom_result["berry"],
        "fisher_trace": geom_result["fisher_trace"],
        "berry_norm": geom_result["berry_norm"],
        "G_5D": geom_result["G_5D"],
        "xi": geom_result["xi"],
        "R": geom_result["R"],
        "R_norm": geom_result["R_norm"],
    }
    
    return profile
