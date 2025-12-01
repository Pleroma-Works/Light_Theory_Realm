"""
Pocket_U Lite: Geometry Showcase
=================================

Demonstrates the mandatory geometry diagnostics for all Standard Model particles.

This script shows that each particle has a unique geometric fingerprint:
- Fisher metric (information distance)
- Berry curvature (topological twist)
- 5D Kaluza-Klein metric
- Reeb flow (dark energy ξ and time direction R)
"""

from light_theory_realm import get_particle_profile
from light_theory_realm.pocket_u_lite import ALL_PARTICLES, LEPTONS, QUARKS

def print_particle_geometry(name: str):
    """Print geometric diagnostics for a single particle."""
    profile = get_particle_profile(name)
    
    print(f"\n{name.upper():>4} | Primes: {str(profile['plaquette'][:4]):<20} | "
          f"Mass: {profile['m_phys_MeV']:>10.3f} MeV")
    print(f"     | Fisher trace: {profile['fisher_trace']:>8.4f} | "
          f"Berry norm: {profile['berry_norm']:>8.4f} | "
          f"ξ: {profile['xi']:>6.4f}")


def main():
    print("\n" + "="*80)
    print("Pocket_U Lite: Geometric Fingerprints of the Standard Model")
    print("="*80)
    
    # Leptons
    print("\n" + "-"*80)
    print("LEPTONS: Light particles with high vacuum screening")
    print("-"*80)
    for name in LEPTONS:
        print_particle_geometry(name)
    
    # Quarks
    print("\n" + "-"*80)
    print("QUARKS: Heavier particles with geometric structure")
    print("-"*80)
    for name in QUARKS:
        print_particle_geometry(name)
    
    # Detailed comparison: electron vs tau
    print("\n" + "="*80)
    print("Detailed Comparison: Electron vs Tau")
    print("="*80)
    
    e_profile = get_particle_profile("e")
    tau_profile = get_particle_profile("tau")
    
    print(f"\n{'Property':<30} {'Electron':>15} {'Tau':>15}")
    print("-"*62)
    print(f"{'Mass (MeV)':<30} {e_profile['m_phys_MeV']:>15.3f} {tau_profile['m_phys_MeV']:>15.3f}")
    print(f"{'Screening (%)':<30} {e_profile['screening_pct']:>15.1f} {tau_profile['screening_pct']:>15.1f}")
    print(f"{'Fisher trace':<30} {e_profile['fisher_trace']:>15.6f} {tau_profile['fisher_trace']:>15.6f}")
    print(f"{'Berry norm':<30} {e_profile['berry_norm']:>15.6f} {tau_profile['berry_norm']:>15.6f}")
    print(f"{'Resonance ξ':<30} {e_profile['xi']:>15.6f} {tau_profile['xi']:>15.6f}")
    print(f"{'Reeb norm':<30} {e_profile['R_norm']:>15.6f} {tau_profile['R_norm']:>15.6f}")
    
    print("\n" + "="*80)
    print("Key Insights:")
    print("="*80)
    print("1. Electron: ~100% screened → 'vacuum bubble' → very light")
    print("2. Tau: ~0% screened → 'pure geometry' → heavy")
    print("3. Each particle has a unique geometric signature in:")
    print("   - Fisher metric (how fast parameters change)")
    print("   - Berry curvature (topological phase accumulation)")
    print("   - Resonance density ξ (vacuum energy / dark energy)")
    print("   - Reeb vector R (time direction in parameter space)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
