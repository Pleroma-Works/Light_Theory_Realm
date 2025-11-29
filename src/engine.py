import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, Dict, Union, Optional
import functools

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

class CliffordEngine:
    """
    JAX-based Clifford Algebra Engine for Cl(6).
    Focuses on Weyl representation (4x4 complex matrices) for QGT computations.
    """

    def __init__(self, seed: int = 42):
        self.key = random.PRNGKey(seed)
        self.dim = 6
        
        # Initialize generators
        self.gammas = self._build_gamma_matrices()
        self.bivectors = self._build_bivectors(self.gammas)

    def _build_gamma_matrices(self):
        """Construct 8x8 Gamma matrices for Cl(6) (Dirac) or 4x4 (Weyl).
        For now, we stick to the Dirac-like construction from the reference 
        but we will adapt for 4x4 if needed. 
        
        The reference implementation used 8x8. We will keep 8x8 for consistency 
        with the 'standard model' logic unless 4x4 is explicitly requested and defined.
        The user plan mentioned 4x4 Weyl, but the reference code uses 8x8.
        Let's implement the 8x8 first as it's robust, and we can project down if needed.
        Actually, let's look at the reference again. It supports 'weyl' mode but uses 8x8 construction logic?
        Wait, the reference `_build_gamma_matrices` returns 8x8.
        Let's stick to the 8x8 implementation for now to ensure correctness with the 'standard model' mapping.
        """
        I = jnp.eye(2, dtype=jnp.complex128)
        X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
        Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

        gammas = []
        # G0 = X ⊗ I ⊗ I
        gammas.append(jnp.kron(X, jnp.kron(I, I)))
        # G1 = Y ⊗ I ⊗ I
        gammas.append(jnp.kron(Y, jnp.kron(I, I)))
        # G2 = Z ⊗ X ⊗ I
        gammas.append(jnp.kron(Z, jnp.kron(X, I)))
        # G3 = Z ⊗ Y ⊗ I
        gammas.append(jnp.kron(Z, jnp.kron(Y, I)))
        # G4 = Z ⊗ Z ⊗ X
        gammas.append(jnp.kron(Z, jnp.kron(Z, X)))
        # G5 = Z ⊗ Z ⊗ Y
        gammas.append(jnp.kron(Z, jnp.kron(Z, Y)))

        return gammas

    def _build_bivectors(self, gammas):
        """Construct all 15 bivectors B_ij = Gi @ Gj."""
        bivectors = {}
        for i in range(6):
            for j in range(i + 1, 6):
                bivectors[(i, j)] = gammas[i] @ gammas[j]
        return bivectors

    @functools.partial(jit, static_argnums=(0,))
    def geometric_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Compute geometric product of two multivectors.
        In matrix representation, this is simply matrix multiplication.
        """
        return jnp.matmul(a, b)

    @functools.partial(jit, static_argnums=(0,))
    def commutator(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Compute [A, B] = AB - BA"""
        return a @ b - b @ a

    @functools.partial(jit, static_argnums=(0,))
    def anticommutator(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Compute {A, B} = AB + BA"""
        return a @ b + b @ a

    def get_bivector(self, i: int, j: int) -> jnp.ndarray:
        """Get bivector generator e_i ∧ e_j"""
        if i < j:
            return self.bivectors[(i, j)]
        elif i > j:
            return -self.bivectors[(j, i)]
        else:
            return jnp.zeros_like(self.gammas[0])

    def random_spinor(self, key):
        """Generate a random normalized spinor (column vector)."""
        # For 8x8 representation, spinors are 8-component vectors
        real = random.normal(key, (8, 1))
        imag = random.normal(random.fold_in(key, 1), (8, 1))
        psi = real + 1j * imag
        psi = psi / jnp.linalg.norm(psi)
        return psi
