import jax
import jax.numpy as jnp
import sys
import os

# Ensure the project root is in the path so we can import 'light_theory_realm'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.qgt import CliffordQGT
from light_theory_realm.experiments.prime_gauge.uplift import KaluzaKleinUplift
from light_theory_realm.experiments.prime_gauge.reeb_flow import ReebFlowDynamics
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeLattice

def main():
    print("===============================================")
    print("   LIGHT THEORY REALM: FULL STACK VERIFICATION")
    print("===============================================\n")

    # ---------------------------------------------------------
    # 1. ENGINE LAYER: Algebra & Grade Projection
    # ---------------------------------------------------------
    print("[1] Verifying Algebraic Core (CliffordEngine)...")
    try:
        engine = CliffordEngine(seed=42)
        print(f"    ✓ Engine Initialized (Dim: {engine.dim})")
        
        # Verify Wedge Product Logic
        # Test A: v ^ v = 0 (Antisymmetry)
        v0 = engine.gammas[0]
        w_self = engine.wedge_product(v0, v0)
        norm_self = jnp.linalg.norm(w_self)
        if norm_self < 1e-10:
            print(f"    ✓ Wedge Product Consistency: v^v ~ 0 (Norm: {norm_self:.2e})")
        else:
            print(f"    X Wedge Product FAILED: v^v Norm = {norm_self}")

        # Test B: v0 ^ v1 (Volume Creation)
        v1 = engine.gammas[1]
        w_cross = engine.wedge_product(v0, v1)
        
        # Check Grade Decomposition
        # v0^v1 should be pure Grade 2 (Bivector)
        g0 = jnp.linalg.norm(engine.project_grade(w_cross, 0))
        g1 = jnp.linalg.norm(engine.project_grade(w_cross, 1))
        g2 = jnp.linalg.norm(engine.project_grade(w_cross, 2))
        g3 = jnp.linalg.norm(engine.project_grade(w_cross, 3))
        
        if g0 < 1e-9 and g1 < 1e-9 and g2 > 0.1:
            print(f"    ✓ Grade Projection: v0^v1 is pure Bivector (Magnitude: {g2:.2f})")
        else:
            print(f"    X Grade Projection FAILED. Decomp: S={g0:.2e}, V={g1:.2e}, B={g2:.2e}")

    except Exception as e:
        print(f"    X CRITICAL ERROR in Engine: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 2. GEOMETRY LAYER: Quantum Geometric Tensor
    # ---------------------------------------------------------
    print("\n[2] Verifying Geometry (CliffordQGT)...")
    try:
        qgt = CliffordQGT(engine)
        
        # Create a test state and Jacobian
        psi = engine.random_spinor(jax.random.PRNGKey(1))
        # Simulate a Jacobian (4, 2) -> 2 parameters
        d_psi_1 = engine.random_spinor(jax.random.PRNGKey(2))
        d_psi_2 = engine.random_spinor(jax.random.PRNGKey(3))
        jacobian = jnp.hstack([d_psi_1, d_psi_2])
        
        fisher, berry = qgt.compute_full_qgt(psi, jacobian)
        
        print(f"    ✓ QGT Computed for 2 Parameters")
        print(f"      Fisher Metric shape: {fisher.shape} (Expected (2,2))")
        print(f"      Berry Curvature shape: {berry.shape} (Expected (2,2))")
        
        # Check properties (Fisher symmetric, Berry antisymmetric)
        sym_check = jnp.linalg.norm(fisher - fisher.T)
        anti_check = jnp.linalg.norm(berry + berry.T)
        
        if sym_check < 1e-10 and anti_check < 1e-10:
            print(f"    ✓ Geometric Symmetries Hold (Symmetric/Antisymmetric)")
        else:
            print(f"    X Symmetry Check FAILED: Sym={sym_check:.2e}, Anti={anti_check:.2e}")

    except Exception as e:
        print(f"    X CRITICAL ERROR in QGT: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 3. UPLIFT LAYER: Kaluza-Klein 5D
    # ---------------------------------------------------------
    print("\n[3] Verifying Theory (Kaluza-Klein Uplift)...")
    try:
        kk = KaluzaKleinUplift(engine)
        
        # Construct 5D metric from the previous 4D state
        G_5D = kk.construct_5d_metric(psi, jacobian, phi_scalar=0.9)
        
        print(f"    ✓ 5D Metric Constructed")
        print(f"      Shape: {G_5D.shape} (Expected (3,3) for 2 params + 1 extra dim)")
        # Shape explanation: N params = 2. Matrix is (N+1)x(N+1) = 3x3.
        
        # Check diagonal (should be positive for physical metric)
        diag = jnp.diag(G_5D)
        print(f"      Diagonal Elements: {diag}")

    except Exception as e:
        print(f"    X CRITICAL ERROR in Uplift: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------
    # 4. EXPERIMENTS: Prime Plaquettes & Reeb Flow
    # ---------------------------------------------------------
    print("\n[4] Verifying Experiments (Prime/Reeb)...")
    try:
        # Prime Lattice
        lattice = PrimeLattice({0:2, 1:3})
        J, A = lattice.compute_edge_params(0, 1)
        print(f"    ✓ Prime Lattice (2->3): Stiffness J={J:.4f}, Twist A={A:.4f}")
        
        # Reeb Flow
        reeb = ReebFlowDynamics(engine)
        # Test large primes (Vacuum resonance)
        xi = reeb.compute_resonance_density([17, 41, 73])
        print(f"    ✓ Reeb Resonance (Vacuum): xi={xi:.4f}")
        
        # Test Reeb Vector calculation
        # Use the Fisher/Berry we computed earlier
        R = reeb.compute_reeb_vector(fisher, berry)
        print(f"    ✓ Reeb Vector Computed: {R}")
        print(f"      Norm Check: {jnp.linalg.norm(R.T @ fisher @ R):.4f} (Target 1.0)")

    except Exception as e:
        print(f"    X CRITICAL ERROR in Experiments: {e}")
        import traceback
        traceback.print_exc()


    print("\n===============================================")
    print("   VERIFICATION COMPLETE")
    print("===============================================")

if __name__ == "__main__":
    main()
