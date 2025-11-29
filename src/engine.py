import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, Dict, Union, Optional
import functools

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

class CliffordEngine:
    """
    JAX-based Clifford Algebra Engine for Cl(1,3) in the Weyl representation.
    Focuses on 4x4 complex matrices for QGT computations.
    """

    def __init__(self, seed: int = 42):
        self.key = random.PRNGKey(seed)
        self.dim = 4
        
        # Initialize generators
        self.gammas = self._build_gamma_matrices()
        self.bivectors = self._build_bivectors(self.gammas)

    def _build_gamma_matrices(self):
        """Construct 4x4 Gamma matrices for Cl(1,3) in the Weyl representation."""
        I = jnp.eye(2, dtype=jnp.complex128)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        
        # Pauli matrices
        s1 = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        s2 = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
        s3 = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        
        paulis = [s1, s2, s3]

        # Weyl basis
        # g0 = [[Z, I], [I, Z]]
        # gi = [[Z, si], [-si, Z]]
        g0 = jnp.block([[Z, I], [I, Z]])
        gammas = [g0]
        for i in range(3):
            gammas.append(jnp.block([[Z, paulis[i]], [-paulis[i], Z]]))

        return gammas

    def _build_bivectors(self, gammas):
        """Construct all 6 bivectors S_uv = 1/4 [Gu, Gv]."""
        bivectors = {}
        for i in range(4):
            for j in range(i + 1, 4):
                # The 1/4 factor is conventional for generators of Lorentz transformations
                bivectors[(i, j)] = 0.25 * self.commutator(gammas[i], gammas[j])
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
        """Get bivector generator S_uv"""
        if i < j:
            return self.bivectors[(i, j)]
        elif i > j:
            return -self.bivectors[(j, i)]
        else:
            return jnp.zeros_like(self.gammas[0])

    def random_spinor(self, key):
        """Generate a random normalized spinor (column vector)."""
        # For 4x4 representation, spinors are 4-component vectors
        real = random.normal(key, (4, 1))
        imag = random.normal(random.fold_in(key, 1), (4, 1))
        psi = real + 1j * imag
        psi = psi / jnp.linalg.norm(psi)
        return psi

