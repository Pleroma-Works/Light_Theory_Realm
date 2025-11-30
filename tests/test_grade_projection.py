import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from light_theory_realm.engine import CliffordEngine

def test_grade_projection():
    print("="*60)
    print("Testing Grade Projection")
    print("="*60)
    
    engine = CliffordEngine(seed=42)
    
    # 1. Test Scalar Projection (Grade 0)
    print("\n1. Testing Scalar Projection (Grade 0)")
    scalar_val = 5.0 + 2.0j
    scalar_mv = scalar_val * jnp.eye(4, dtype=jnp.complex128)
    proj_0 = engine.project_grade(scalar_mv, 0)
    
    # Verify
    diff = jnp.linalg.norm(proj_0 - scalar_mv)
    print(f"Original Scalar Val: {scalar_val}")
    print(f"Projected Matrix:\n{proj_0}")
    print(f"Original Matrix:\n{scalar_mv}")
    print(f"Projected Norm Difference: {diff:.6e}")
    
    # Check scalar part method directly
    sp = engine.scalar_part(scalar_mv)
    print(f"Direct Scalar Part Call: {sp}")
    
    assert diff < 1e-10, "Scalar projection failed"
    print("✓ Scalar projection passed")
    
    # 2. Test Vector Projection (Grade 1)
    print("\n2. Testing Vector Projection (Grade 1)")
    # Create a vector: v = 2*g0 + 3*g1
    vec_mv = 2.0 * engine.gammas[0] + 3.0 * engine.gammas[1]
    proj_1 = engine.project_grade(vec_mv, 1)
    proj_0_from_vec = engine.project_grade(vec_mv, 0)
    
    diff_1 = jnp.linalg.norm(proj_1 - vec_mv)
    diff_0 = jnp.linalg.norm(proj_0_from_vec)
    
    print(f"Vector Projection Difference: {diff_1:.6e}")
    print(f"Scalar Projection of Vector (should be 0): {diff_0:.6e}")
    assert diff_1 < 1e-10, "Vector projection failed"
    assert diff_0 < 1e-10, "Vector leaked into scalar grade"
    print("✓ Vector projection passed")
    
    # 3. Test Bivector Projection (Grade 2)
    print("\n3. Testing Bivector Projection (Grade 2)")
    # Create a bivector: B = g0*g1
    bivec_mv = engine.geometric_product(engine.gammas[0], engine.gammas[1])
    proj_2 = engine.project_grade(bivec_mv, 2)
    proj_1_from_bivec = engine.project_grade(bivec_mv, 1)
    
    diff_2 = jnp.linalg.norm(proj_2 - bivec_mv)
    diff_1_leak = jnp.linalg.norm(proj_1_from_bivec)
    
    print(f"Bivector Projection Difference: {diff_2:.6e}")
    print(f"Vector Projection of Bivector (should be 0): {diff_1_leak:.6e}")
    assert diff_2 < 1e-10, "Bivector projection failed"
    assert diff_1_leak < 1e-10, "Bivector leaked into vector grade"
    print("✓ Bivector projection passed")
    
    # 4. Test Mixed Multivector Decomposition
    print("\n4. Testing Mixed Multivector Decomposition")
    # M = 1 + g0 + g0g1
    mixed_mv = scalar_mv + vec_mv + bivec_mv
    
    p0 = engine.project_grade(mixed_mv, 0)
    p1 = engine.project_grade(mixed_mv, 1)
    p2 = engine.project_grade(mixed_mv, 2)
    p3 = engine.project_grade(mixed_mv, 3)
    p4 = engine.project_grade(mixed_mv, 4)
    
    reconstructed = p0 + p1 + p2 + p3 + p4
    total_diff = jnp.linalg.norm(reconstructed - mixed_mv)
    
    print(f"Total Reconstruction Difference: {total_diff:.6e}")
    assert total_diff < 1e-10, "Multivector reconstruction failed"
    
    # Verify individual components
    assert jnp.linalg.norm(p0 - scalar_mv) < 1e-10, "Scalar component mismatch"
    assert jnp.linalg.norm(p1 - vec_mv) < 1e-10, "Vector component mismatch"
    assert jnp.linalg.norm(p2 - bivec_mv) < 1e-10, "Bivector component mismatch"
    assert jnp.linalg.norm(p3) < 1e-10, "Ghost trivector component found"
    
    print("✓ Mixed decomposition passed")
    print("\nAll Grade Projection tests passed!")

if __name__ == "__main__":
    with open("test_output.txt", "w") as f:
        sys.stdout = f
        try:
            test_grade_projection()
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
    
    # Print the file content to console for the agent to see (if small enough)
    with open("test_output.txt", "r") as f:
        print(f.read())
