"""
Example: Grade Projection in Clifford Algebra

This example demonstrates how to decompose multivectors into their
geometric grades (scalar, vector, bivector, trivector, pseudoscalar)
and reconstruct them from individual components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
from light_theory_realm import CliffordEngine


def main():
    print("=" * 70)
    print("Grade Projection Example")
    print("=" * 70)
    
    # Initialize engine
    engine = CliffordEngine(seed=42)
    
    # Example 1: Pure Component Projection
    print("\n[1] Extracting Pure Grade Components")
    print("-" * 70)
    
    scalar = 5.0 + 2.0j
    scalar_matrix = scalar * jnp.eye(4, dtype=jnp.complex128)
    
    # Extract scalar component
    extracted_scalar = engine.project_grade(scalar_matrix, 0)
    scalar_part = engine.scalar_part(extracted_scalar)
    
    print(f"Original scalar value: {scalar}")
    print(f"Extracted scalar value: {scalar_part}")
    print(f"Match: {jnp.isclose(scalar, scalar_part)}")
    
    # Example 2: Vector Components
    print("\n[2] Extracting Vector Components")
    print("-" * 70)
    
    v1 = engine.gammas[0]  # First basis vector
    v2 = engine.gammas[1]  # Second basis vector
    
    mixed_vector = 2.0 * v1 + 3.0 * v2
    
    # Project onto grade 1 (vectors)
    vector_component = engine.project_grade(mixed_vector, 1)
    
    # Verify no scalar contamination
    scalar_contamination = engine.project_grade(mixed_vector, 0)
    scalar_norm = jnp.linalg.norm(scalar_contamination)
    
    print(f"Vector component extracted successfully")
    print(f"Scalar contamination norm: {scalar_norm:.2e} (should be 0)")
    
    # Example 3: Bivector Components
    print("\n[3] Extracting Bivector Components")
    print("-" * 70)
    
    # Create a bivector v1 ∧ v2
    bivector = engine.wedge_product(engine.gammas[0], engine.gammas[1])
    
    # Extract bivector component
    bivector_component = engine.project_grade(bivector, 2)
    
    # Verify it's pure bivector
    non_bivector_norms = []
    for grade in [0, 1, 3, 4]:
        component = engine.project_grade(bivector, grade)
        norm = jnp.linalg.norm(component)
        non_bivector_norms.append(norm)
    
    print(f"Bivector extracted with grade purity")
    print(f"Grade 0 (scalar) contamination: {non_bivector_norms[0]:.2e}")
    print(f"Grade 1 (vector) contamination: {non_bivector_norms[1]:.2e}")
    print(f"Grade 3 (trivector) contamination: {non_bivector_norms[2]:.2e}")
    print(f"Grade 4 (pseudoscalar) contamination: {non_bivector_norms[3]:.2e}")
    
    # Example 4: Mixed Multivector Decomposition
    print("\n[4] Decomposing Mixed Multivectors")
    print("-" * 70)
    
    # Create a mixed multivector: scalar + vector + bivector
    scalar_part = 2.0 * jnp.eye(4, dtype=jnp.complex128)
    vector_part = 1.5 * engine.gammas[0] + 2.5 * engine.gammas[1]
    bivector_part = engine.wedge_product(engine.gammas[2], engine.gammas[3])
    
    mixed = scalar_part + vector_part + bivector_part
    
    # Extract all grades
    g0 = engine.project_grade(mixed, 0)
    g1 = engine.project_grade(mixed, 1)
    g2 = engine.project_grade(mixed, 2)
    g3 = engine.project_grade(mixed, 3)
    g4 = engine.project_grade(mixed, 4)
    
    # Reconstruct
    reconstructed = g0 + g1 + g2 + g3 + g4
    reconstruction_error = jnp.linalg.norm(mixed - reconstructed)
    
    print(f"Mixed multivector components extracted:")
    print(f"  Grade 0 norm: {jnp.linalg.norm(g0):.6f}")
    print(f"  Grade 1 norm: {jnp.linalg.norm(g1):.6f}")
    print(f"  Grade 2 norm: {jnp.linalg.norm(g2):.6f}")
    print(f"  Grade 3 norm: {jnp.linalg.norm(g3):.6f}")
    print(f"  Grade 4 norm: {jnp.linalg.norm(g4):.6f}")
    print(f"\nReconstruction error: {reconstruction_error:.2e}")
    print(f"Perfect reconstruction: {reconstruction_error < 1e-10}")
    
    # Example 5: Grade Progression
    print("\n[5] Grade Progression Through Wedge Products")
    print("-" * 70)
    
    v1 = engine.gammas[0]
    v2 = engine.gammas[1]
    v3 = engine.gammas[2]
    v4 = engine.gammas[3]
    
    # Grade 1
    grade_1 = v1
    print(f"Grade 1 (vector): norm = {jnp.linalg.norm(grade_1):.6f}")
    
    # Grade 2: v1 ∧ v2
    grade_2 = engine.wedge_product(v1, v2)
    print(f"Grade 2 (bivector): norm = {jnp.linalg.norm(grade_2):.6f}")
    
    # Grade 3: v1 ∧ v2 ∧ v3
    grade_3 = engine.wedge_product(engine.wedge_product(v1, v2), v3)
    print(f"Grade 3 (trivector): norm = {jnp.linalg.norm(grade_3):.6f}")
    
    # Grade 4: v1 ∧ v2 ∧ v3 ∧ v4
    grade_4 = engine.wedge_product(
        engine.wedge_product(engine.wedge_product(v1, v2), v3),
        v4
    )
    print(f"Grade 4 (pseudoscalar): norm = {jnp.linalg.norm(grade_4):.6f}")
    
    # Grade 5 would be zero (dimension 4 limit)
    grade_5_attempt = engine.wedge_product(grade_4, v1)
    print(f"Grade 5+ attempt (should be 0): norm = {jnp.linalg.norm(grade_5_attempt):.6f}")
    
    print("\n" + "=" * 70)
    print("Grade projection examples completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
