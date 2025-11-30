"""
Prime Plaquette Example
=======================
Demonstrates how prime numbers define particle masses via 
the Wilson Loop operator.
"""

import jax
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeLattice, PrimeWilsonLoop

def main():
    print("="*60)
    print("Prime Plaquette Example")
    print("="*60)
    
    # Define three different prime plaquettes
    plaquettes = {
        "Electron": [2, 3, 5, 7],
        "Muon": [11, 13, 17, 19],
        "Tau": [23, 29, 31, 37]
    }
    
    engine = CliffordEngine(seed=137)
    
    for name, primes in plaquettes.items():
        print(f"\n{name} Plaquette: {primes}")
        print("-" * 40)
        
        # Track B: Classical Parameters
        lattice = PrimeLattice({i: p for i, p in enumerate(primes)})
        J_01, A_01 = lattice.compute_edge_params(0, 1)
        print(f"Classical (Track B):")
        print(f"  First Link Stiffness: J = {J_01:.4f}")
        print(f"  First Link Twist:     A = {A_01:.4f} rad")
        
        # Track A: Quantum Wilson Loop
        pwl = PrimeWilsonLoop(engine)
        psi = engine.random_spinor(jax.random.PRNGKey(42))
        W_expect = pwl.measure_loop(psi, primes)
        mass = jnp.angle(W_expect)
        
        print(f"Quantum (Track A):")
        print(f"  Wilson Loop:     <W> = {W_expect:.4f}")
        print(f"  Topological Mass: m = {mass:.4f} rad")
    
    print("\n" + "="*60)
    print("Key Insight:")
    print("  Different prime sequences → Different masses")
    print("  This is how Light Theory predicts particle masses!")
    print("="*60)

if __name__ == "__main__":
    main()
