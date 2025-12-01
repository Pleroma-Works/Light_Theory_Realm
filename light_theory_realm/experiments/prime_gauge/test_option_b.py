"""
Quick verification test for Option B: Full 2D Ricci Scalar Curvature
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from light_theory_realm.experiments.prime_gauge.nextgen_curvature import *
from light_theory_realm.engine import CliffordEngine
from light_theory_realm.qgt import CliffordQGT
import jax.numpy as jnp

def test_full_2d_curvature():
    """Test that the full 2D Ricci scalar computation works"""
    print("=" * 80)
    print("Testing Option B: Full 2D Ricci Scalar Curvature")
    print("=" * 80)
    print()
    
    # Setup
    engine = CliffordEngine(seed=42)
    primes = [2, 3, 5, 7]
    generators = build_edge_generators(primes, engine)[:2]
    base_psi = jnp.log(jnp.array(primes[:2], dtype=float))
    base_psi = base_psi / jnp.linalg.norm(base_psi)
    qgt = CliffordQGT(engine)
    
    theta = jnp.array([0.1, 0.2])
    
    print("Test 1: scalar_curvature_full_2d")
    print("-" * 80)
    try:
        R, g, Gamma = scalar_curvature_full_2d(theta, primes, engine, base_psi, generators, qgt)
        print(f"✓ Full 2D Ricci scalar computation successful!")
        print(f"  R = {R:.6e}")
        print(f"  Metric shape: {g.shape}")
        print(f"  Christoffel shape: {Gamma.shape}")
        print(f"  Metric symmetry check: {jnp.allclose(g, g.T)}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Test 2: compute_qgt_exponential with new curvature")
    print("-" * 80)
    try:
        theta_4d = jnp.array([0.1, 0.2, 0.3, 0.4])
        generators_4d = build_edge_generators(primes, engine)
        base_psi_4d = jnp.log(jnp.array(primes, dtype=float))
        base_psi_4d = base_psi_4d / jnp.linalg.norm(base_psi_4d)
        
        result = compute_qgt_exponential(theta_4d, generators_4d, base_psi_4d, engine, primes)
        print(f"✓ compute_qgt_exponential successful!")
        print(f"  R = {result['curvature_scalar']:.6e}")
        print(f"  Fisher shape: {result['fisher'].shape}")
        print(f"  Gamma shape: {result['Gamma'].shape}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_full_2d_curvature()
    sys.exit(0 if success else 1)
