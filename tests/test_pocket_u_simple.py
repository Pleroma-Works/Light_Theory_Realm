"""
Simple test script for Pocket_U Lite mass predictions.
Tests only the mass model without geometry diagnostics.
"""

from light_theory_realm.pocket_u_lite.mass_model import physical_mass
from light_theory_realm.pocket_u_lite.plaquettes import ALL_PARTICLES, PDG_MASSES

def test_mass_predictions():
    print("\n" + "="*80)
    print("Pocket_U Lite: Mass Prediction Test")
    print("="*80 + "\n")
    
    print(f"{'Particle':<10} {'PDG (MeV)':>12} {'Predicted (MeV)':>18} {'Error (%)':>12}")
    print("-"*60)
    
    total_error = 0.0
    for name in ALL_PARTICLES:
        result = physical_mass(name)
        print(f"{name:<10} {result['pdg_mass_MeV']:>12.3f} {result['m_phys_MeV']:>18.3f} {result['error_pct']:>11.1f}%")
        total_error += abs(result['error_pct'])
    
    avg_error = total_error / len(ALL_PARTICLES)
    print("-"*60)
    print(f"Average error: {avg_error:.2f}%\n")

if __name__ == "__main__":
    test_mass_predictions()
