"""
Basic QGT Example
=================
Demonstrates computing the Quantum Geometric Tensor (Fisher + Berry) 
for a simple parameterized quantum state.
"""

import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.qgt import CliffordQGT

def main():
    print("="*60)
    print("Basic QGT Example")
    print("="*60)
    
    # 1. Initialize the Clifford Engine
    engine = CliffordEngine(seed=42)
    qgt = CliffordQGT(engine)
    
    # 2. Define a parameterized state: rotation in the 01-plane
    psi_0 = engine.random_spinor(jax.random.PRNGKey(0))
    generator = engine.gammas[0] @ engine.gammas[1]  # Bivector S_01
    
    def get_psi(theta):
        """Rotate the initial state by angle theta."""
        angle = theta[0]
        U = jax.scipy.linalg.expm(-1j * angle * generator)
        return U @ psi_0
    
    # 3. Compute the Jacobian at theta = 0.5
    theta_val = jnp.array([0.5])
    jacobian = jax.jacfwd(get_psi)(theta_val).reshape(4, 1)
    psi_val = get_psi(theta_val)
    
    # 4. Compute QGT
    fisher, berry = qgt.compute_full_qgt(psi_val, jacobian)
    
    # 5. Display results
    print(f"\nParameter: θ = {theta_val[0]:.4f}")
    print(f"\nFisher Information (Metric):")
    print(f"  g_θθ = {fisher[0,0]:.6f}")
    print(f"\nBerry Curvature (Twist):")
    print(f"  Ω_θθ = {berry[0,0]:.6f}")
    
    print("\n" + "="*60)
    print("Interpretation:")
    print("  - Fisher: How fast the state changes with θ")
    print("  - Berry: Topological phase accumulated during rotation")
    print("="*60)

if __name__ == "__main__":
    main()
