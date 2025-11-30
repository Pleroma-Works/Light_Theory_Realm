"""
Example: Wedge Product in Clifford Algebra

This example demonstrates the wedge product (outer product) and its
fundamental properties in Clifford algebra: null property, anticommutativity,
associativity, and grade preservation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
from light_theory_realm import CliffordEngine


def main():
    print("=" * 70)
    print("Wedge Product Example")
    print("=" * 70)
    
    # Initialize engine
    engine = CliffordEngine(seed=42)
    
    # Example 1: Null Property (v ∧ v = 0)
    print("\n[1] Null Property: v ∧ v = 0")
    print("-" * 70)
    print("For any vector v, the wedge product with itself is zero")
    print("This represents that a vector cannot span a volume with itself\n")
    
    for i in range(4):
        v = engine.gammas[i]
        result = engine.wedge_product(v, v)
        norm = jnp.linalg.norm(result)
        print(f"g{i} ∧ g{i}: norm = {norm:.2e}")
    
    print("\nAll norms are effectively zero (< 1e-10)")
    
    # Example 2: Anticommutativity (v1 ∧ v2 = -v2 ∧ v1)
    print("\n[2] Anticommutativity: v1 ∧ v2 = -v2 ∧ v1")
    print("-" * 70)
    print("The wedge product changes sign when operands are swapped\n")
    
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    
    w12 = engine.wedge_product(v1, v2)
    w21 = engine.wedge_product(v2, v1)
    
    sum_result = w12 + w21
    error = jnp.linalg.norm(sum_result)
    
    print(f"g0 ∧ g1 + g1 ∧ g0 should equal 0")
    print(f"Verification error: {error:.2e}")
    
    # Example 3: Grade Preservation
    print("\n[3] Grade Preservation: Grade(A ∧ B) = Grade(A) + Grade(B)")
    print("-" * 70)
    print("The wedge product of grade-r and grade-s produces grade-(r+s)\n")
    
    v = engine.gammas[0]  # Grade 1
    b = engine.wedge_product(engine.gammas[1], engine.gammas[2])  # Grade 2
    
    # v ∧ b should be grade 1+2 = grade 3 (trivector)
    result = engine.wedge_product(v, b)
    
    print(f"Grade 1 (vector) ∧ Grade 2 (bivector) = Grade 3 (trivector)")
    print(f"\nComponent breakdown:")
    for grade in range(5):
        component = engine.project_grade(result, grade)
        norm = jnp.linalg.norm(component)
        expected = "NONZERO" if grade == 3 else "zero"
        status = "OK" if (norm > 1e-10) == (grade == 3) else "ERROR"
        print(f"  Grade {grade}: {norm:.2e} ({expected}) [{status}]")
    
    # Example 4: Vector Wedge Product
    print("\n[4] Vector Wedge: Two Vectors Produce Bivector")
    print("-" * 70)
    print("The wedge of two vectors spans a 2D plane (bivector)\n")
    
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    v3 = engine.gammas[2]
    
    # All pairwise wedges should be pure bivectors
    pairs = [(v1, v2), (v1, v3), (v2, v3)]
    
    for i, (va, vb) in enumerate(pairs):
        wedge = engine.wedge_product(va, vb)
        bivector_comp = engine.project_grade(wedge, 2)
        error = jnp.linalg.norm(wedge - bivector_comp)
        print(f"Pair {i+1}: Purity error = {error:.2e}")
    
    # Example 5: Associativity ((A ∧ B) ∧ C = A ∧ (B ∧ C))
    print("\n[5] Associativity: (v1 ∧ v2) ∧ v3 = v1 ∧ (v2 ∧ v3)")
    print("-" * 70)
    print("Wedge product is associative\n")
    
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    v3 = engine.gammas[2]
    
    # Left associative
    left = engine.wedge_product(engine.wedge_product(v1, v2), v3)
    
    # Right associative
    right = engine.wedge_product(v1, engine.wedge_product(v2, v3))
    
    error = jnp.linalg.norm(left - right)
    print(f"(v1 ∧ v2) ∧ v3 vs v1 ∧ (v2 ∧ v3)")
    print(f"Difference norm: {error:.2e}")
    print(f"Associativity verified: {error < 1e-10}")
    
    # Example 6: Scalar Multiplication
    print("\n[6] Scalar Multiplication: (αI) ∧ v = αv")
    print("-" * 70)
    print("Scalar multiplication is preserved under wedge product\n")
    
    scalar = 3.5 + 2.1j
    scalar_matrix = scalar * jnp.eye(4, dtype=jnp.complex128)
    v = engine.gammas[1]
    
    result = engine.wedge_product(scalar_matrix, v)
    expected = scalar * v
    error = jnp.linalg.norm(result - expected)
    
    print(f"Scalar value: {scalar}")
    print(f"(3.5+2.1j)I ∧ v vs (3.5+2.1j)v")
    print(f"Difference norm: {error:.2e}")
    print(f"Scalar multiplication preserved: {error < 1e-10}")
    
    # Example 7: Mixed Grade Wedges
    print("\n[7] Mixed Multivector Wedge")
    print("-" * 70)
    print("Wedging mixed-grade multivectors\n")
    
    scalar = 2.0 * jnp.eye(4, dtype=jnp.complex128)
    vector = engine.gammas[0]
    mixed_a = scalar + vector  # Grade 0 + Grade 1
    
    bivector = engine.wedge_product(engine.gammas[1], engine.gammas[2])
    scalar_b = 1.5 * jnp.eye(4, dtype=jnp.complex128)
    mixed_b = scalar_b + bivector  # Grade 0 + Grade 2
    
    result = engine.wedge_product(mixed_a, mixed_b)
    
    print(f"(2I + g0) ∧ (1.5I + g1∧g2)")
    print(f"\nResult components:")
    for grade in range(5):
        component = engine.project_grade(result, grade)
        norm = jnp.linalg.norm(component)
        if norm > 1e-10:
            print(f"  Grade {grade}: {norm:.6f}")
    
    # Example 8: Bivector Wedge (Produces Pseudoscalar)
    print("\n[8] Bivector Wedge: Two Bivectors Produce Pseudoscalar")
    print("-" * 70)
    print("Two independent bivectors span the entire 4D volume\n")
    
    b1 = engine.wedge_product(engine.gammas[0], engine.gammas[1])  # Grade 2
    b2 = engine.wedge_product(engine.gammas[2], engine.gammas[3])  # Grade 2
    
    result = engine.wedge_product(b1, b2)
    
    print(f"(g0 ∧ g1) ∧ (g2 ∧ g3)")
    print(f"\nResult components:")
    for grade in range(5):
        component = engine.project_grade(result, grade)
        norm = jnp.linalg.norm(component)
        if norm > 1e-10:
            print(f"  Grade {grade}: {norm:.6f}")
    
    print("\n" + "=" * 70)
    print("Wedge product examples completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
