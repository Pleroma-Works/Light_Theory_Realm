"""
Reeb Flow Example
=================
Demonstrates the derivation of Dark Energy as vacuum resonance density.
"""

import jax
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.experiments.prime_gauge.reeb_flow import ReebFlowDynamics

def main():
    print("="*60)
    print("Reeb Flow / Dark Energy Example")
    print("="*60)
    
    engine = CliffordEngine(seed=42)
    reeb = ReebFlowDynamics(engine)
    
    # Compare Matter vs Vacuum
    test_cases = [
        ("Matter (Small Primes)", [2, 3, 5, 7]),
        ("Intermediate", [11, 13, 17, 19]),
        ("Vacuum (Large Primes)", [97, 101, 103, 107])
    ]
    
    print("\nResonance Density (ξ) Analysis:")
    print("-" * 60)
    
    results = []
    for name, primes in test_cases:
        xi = reeb.compute_resonance_density(primes)
        results.append((name, xi))
        print(f"{name:25s}: ξ = {xi:.6f}")
    
    # Calculate ratios
    print("\n" + "="*60)
    print("Dark Energy Dominance:")
    print("-" * 60)
    
    matter_xi = results[0][1]
    vacuum_xi = results[2][1]
    ratio = vacuum_xi / matter_xi
    
    print(f"Vacuum/Matter Ratio: {ratio:.2f}×")
    print(f"\nInterpretation:")
    print(f"  - Small primes: Generators clash → Low ξ → Curvature dominates")
    print(f"  - Large primes: Generators align → High ξ → Drift dominates")
    print(f"  - Empty space has {ratio:.0f}× more 'Dark Energy' than matter!")
    
    print("\n" + "="*60)
    print("This explains cosmic acceleration:")
    print("  The universe drifts toward algebraically resonant states.")
    print("="*60)

if __name__ == "__main__":
    main()
