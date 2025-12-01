"""
Pocket_U Lite: Mass Table Demo
===============================

Demonstrates the complete Standard Model mass spectrum from Pocket_U Lite.

This script shows:
1. The full SM mass table with PDG targets and predictions
2. A detailed profile for the electron including geometry diagnostics
"""

from light_theory_realm import print_sm_table, get_particle_profile

def main():
    print("\n" + "="*80)
    print("Pocket_U Lite: Standard Model Mass Spectrum Demo")
    print("="*80)
    
    # Print the full SM mass table
    print_sm_table()
    
    # Get detailed profile for electron
    print("\n" + "="*80)
    print("Detailed Electron Profile")
    print("="*80 + "\n")
    
    e_profile = get_particle_profile("e")
    
    print(f"Particle: {e_profile['name']}")
    print(f"Primes: {e_profile['plaquette']}")
    print(f"\nMass Prediction:")
    print(f"  Physical mass: {e_profile['m_phys_MeV']:.6f} MeV")
    print(f"  PDG target:    {e_profile['pdg_mass_MeV']:.6f} MeV")
    print(f"  Error:         {e_profile['error_pct']:.2f}%")
    print(f"\nMass Breakdown:")
    print(f"  Geometric:     {e_profile['m_geom_MeV']:.3f} MeV")
    print(f"  Vacuum:        {e_profile['m_vac_MeV']:.3f} MeV")
    print(f"  Screening:     {e_profile['screening_pct']:.1f}%")
    
    print(f"\nGeometry Diagnostics:")
    print(f"  Fisher metric shape:  {e_profile['fisher'].shape}")
    print(f"  Fisher trace:         {e_profile['fisher_trace']:.6f}")
    print(f"  Berry curvature norm: {e_profile['berry_norm']:.6f}")
    print(f"  KK metric shape:      {e_profile['G_5D'].shape}")
    print(f"  Resonance density ξ:  {e_profile['xi']:.6f}")
    print(f"  Reeb vector norm:     {e_profile['R_norm']:.6f}")
    
    print("\n" + "="*80)
    print("Key Insight:")
    print("="*80)
    print("The electron is ~100% vacuum screened - it's a 'bubble' in the")
    print("vacuum energy, not a pure geometric knot. This is why it's so light!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
