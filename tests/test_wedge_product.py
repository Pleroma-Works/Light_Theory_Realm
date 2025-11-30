import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from light_theory_realm.engine import CliffordEngine

def test_wedge_product():
    """
    Comprehensive tests for the wedge product implementation.
    
    The wedge product A ∧ B is the grade-increasing part of the geometric product,
    defined as:
    A ∧ B = Σ_{r,s} <A_r * B_s>_{r+s}
    
    Properties tested:
    1. v ∧ v = 0 (null vector property)
    2. v1 ∧ v2 is grade-2 (bivector)
    3. v1 ∧ v2 = -v2 ∧ v1 (anticommutativity)
    4. (a ∧ b) ∧ c = a ∧ (b ∧ c) (associativity)
    5. Scalar wedge properties
    6. Grade preservation
    7. Mixed grade wedge products
    """
    print("=" * 70)
    print("Testing Wedge Product Implementation")
    print("=" * 70)
    
    engine = CliffordEngine(seed=42)
    
    # Test 1: v ∧ v = 0 for any vector v
    print("\n1. Null Vector Property: v ∧ v = 0")
    print("-" * 70)
    for i, gamma in enumerate(engine.gammas):
        result = engine.wedge_product(gamma, gamma)
        norm = jnp.linalg.norm(result)
        print(f"   g{i} ∧ g{i} norm: {norm:.6e}", end="")
        assert norm < 1e-10, f"Failed for gamma {i}"
        print(" ✓")
    print("✓ Null vector property verified for all basis vectors")
    
    # Test 2: v1 ∧ v2 is a bivector (grade 2)
    print("\n2. Grade Preservation: v1 ∧ v2 has grade-2 component only")
    print("-" * 70)
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    wedge_v1v2 = engine.wedge_product(v1, v2)
    
    for grade in range(5):
        proj = engine.project_grade(wedge_v1v2, grade)
        norm = jnp.linalg.norm(proj)
        expected = "✓ NONZERO" if grade == 2 else "✓ zero"
        print(f"   Grade {grade}: {norm:.6e} {expected}")
        if grade == 2:
            assert norm > 1e-10, f"Grade 2 should be non-zero"
        else:
            assert norm < 1e-10, f"Grade {grade} should be zero for v1 ∧ v2"
    print("✓ Grade preservation verified")
    
    # Test 3: Anticommutativity: v1 ∧ v2 = -v2 ∧ v1
    print("\n3. Anticommutativity: v1 ∧ v2 = -v2 ∧ v1")
    print("-" * 70)
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    wedge_v1v2 = engine.wedge_product(v1, v2)
    wedge_v2v1 = engine.wedge_product(v2, v1)
    diff = jnp.linalg.norm(wedge_v1v2 + wedge_v2v1)
    print(f"   ||v1 ∧ v2 + v2 ∧ v1||: {diff:.6e}")
    assert diff < 1e-10, "Anticommutativity failed"
    print("✓ Anticommutativity verified")
    
    # Test 4: Associativity: (a ∧ b) ∧ c = a ∧ (b ∧ c)
    print("\n4. Associativity: (a ∧ b) ∧ c = a ∧ (b ∧ c)")
    print("-" * 70)
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    v3 = engine.gammas[2]
    
    left = engine.wedge_product(engine.wedge_product(v1, v2), v3)
    right = engine.wedge_product(v1, engine.wedge_product(v2, v3))
    diff = jnp.linalg.norm(left - right)
    print(f"   ||(v1 ∧ v2) ∧ v3 - v1 ∧ (v2 ∧ v3)||: {diff:.6e}")
    assert diff < 1e-10, "Associativity failed"
    print("✓ Associativity verified")
    
    # Test 5: Scalar wedge property: s*I ∧ v = s*v
    print("\n5. Scalar Wedge Property: s*I ∧ v = s*v")
    print("-" * 70)
    scalar_val = 3.5 + 2.1j
    scalar = scalar_val * jnp.eye(4, dtype=jnp.complex128)
    v = engine.gammas[1]
    
    wedge_result = engine.wedge_product(scalar, v)
    expected = scalar_val * v
    diff = jnp.linalg.norm(wedge_result - expected)
    print(f"   ||(3.5+2.1j)I ∧ v - (3.5+2.1j)*v||: {diff:.6e}")
    assert diff < 1e-10, "Scalar wedge property failed"
    print("✓ Scalar wedge property verified")
    
    # Test 6: Bivector ∧ Bivector = Grade 4 (Pseudoscalar) component
    print("\n6. Grade Mixing: B1 ∧ B2 (bivector ∧ bivector)")
    print("-" * 70)
    b1 = engine.geometric_product(engine.gammas[0], engine.gammas[1])
    b2 = engine.geometric_product(engine.gammas[2], engine.gammas[3])
    
    wedge_b1b2 = engine.wedge_product(b1, b2)
    
    for grade in range(5):
        proj = engine.project_grade(wedge_b1b2, grade)
        norm = jnp.linalg.norm(proj)
        print(f"   Grade {grade}: {norm:.6e}")
    print("✓ Bivector wedge product computed")
    
    # Test 7: Mixed grade wedge product
    print("\n7. Mixed Grade: (scalar + vector) ∧ (scalar + bivector)")
    print("-" * 70)
    scalar_part = 2.0 * jnp.eye(4, dtype=jnp.complex128)
    vector_part = engine.gammas[0]
    mixed_a = scalar_part + vector_part
    
    scalar_part_b = 1.5 * jnp.eye(4, dtype=jnp.complex128)
    bivector_part = engine.geometric_product(engine.gammas[1], engine.gammas[2])
    mixed_b = scalar_part_b + bivector_part
    
    wedge_mixed = engine.wedge_product(mixed_a, mixed_b)
    
    print(f"   Mixed A = 2*I + g0")
    print(f"   Mixed B = 1.5*I + g1∧g2")
    print(f"   Result contains grades:")
    for grade in range(5):
        proj = engine.project_grade(wedge_mixed, grade)
        norm = jnp.linalg.norm(proj)
        if norm > 1e-10:
            print(f"      Grade {grade}: {norm:.6e}")
    print("✓ Mixed grade wedge product computed")
    
    # Test 8: Consistency with geometric product for pure grades
    print("\n8. Consistency: v1 ∧ v2 is part of v1 * v2")
    print("-" * 70)
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    
    geom_prod = engine.geometric_product(v1, v2)
    wedge_prod = engine.wedge_product(v1, v2)
    
    # For vectors v1*v2 = (v1·v2) + (v1∧v2), so wedge part is v1*v2 - scalar part
    scalar_part_gp = engine.scalar_part(geom_prod)
    scalar_matrix = scalar_part_gp * jnp.eye(4, dtype=jnp.complex128)
    expected_wedge = geom_prod - scalar_matrix
    
    diff = jnp.linalg.norm(wedge_prod - expected_wedge)
    print(f"   ||v1∧v2 - (v1*v2 - ⟨v1*v2⟩₀)||: {diff:.6e}")
    assert diff < 1e-10, "Consistency with geometric product failed"
    print("✓ Consistency with geometric product verified")
    
    print("\n" + "=" * 70)
    print("✓ All wedge product tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_wedge_product()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
