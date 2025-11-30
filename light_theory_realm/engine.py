import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, Dict, Union, Optional
import functools

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

class CliffordEngine:
    """
    JAX-based Clifford Algebra Engine for the spacetime algebra Cl(1,3).

    This engine implements the Clifford algebra using a 4x4 complex matrix
    representation (Weyl representation) of the gamma matrices. It provides
    fundamental operations for multivector manipulation, crucial for
    applications in quantum field theory and quantum information,
    particularly for Quantum Geometric Tensor (QGT) computations.

    Attributes:
        key (jax.random.PRNGKey): JAX PRNG key for random operations.
        dim (int): The dimension of the matrices used (fixed at 4 for Cl(1,3) Weyl representation).
        gammas (list[jnp.ndarray]): List of the four 4x4 complex gamma matrices.
        bivectors (dict[tuple[int, int], jnp.ndarray]): Dictionary of the 6 bivector generators.
        basis_by_grade (dict[int, list[jnp.ndarray]]): Dictionary mapping grade to basis elements.
    """

    def __init__(self, seed: int = 42):
        """
        Initializes the CliffordEngine with a given random seed.

        Args:
            seed (int): An integer seed for the JAX pseudo-random number generator.
                        Defaults to 42.
        """
        self.key = random.PRNGKey(seed)
        self.dim = 4
        
        # Initialize generators
        self.gammas = self._build_gamma_matrices()
        self.bivectors = self._build_bivectors(self.gammas)
        
        # Build full basis for grade projection
        self.basis_by_grade = self._build_graded_basis()

    def _build_gamma_matrices(self) -> list[jnp.ndarray]:
        """
        Constructs the 4x4 Gamma matrices for Cl(1,3) in the Weyl representation.

        Returns:
            list[jnp.ndarray]: A list containing the four 4x4 complex gamma matrices.
        """
        I = jnp.eye(2, dtype=jnp.complex128)
        Z = jnp.zeros((2, 2), dtype=jnp.complex128)
        
        # Pauli matrices
        s1 = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        s2 = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
        s3 = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        
        paulis = [s1, s2, s3]

        # Weyl basis construction
        g0 = jnp.block([[Z, I], [I, Z]])
        gammas = [g0]
        for i in range(3):
            gammas.append(jnp.block([[Z, paulis[i]], [-paulis[i], Z]]))

        return gammas

    def _build_bivectors(self, gammas: list[jnp.ndarray]) -> Dict[Tuple[int, int], jnp.ndarray]:
        """
        Constructs the 6 bivector generators S_uv from the gamma matrices.

        Args:
            gammas (list[jnp.ndarray]): A list of the four 4x4 complex gamma matrices.

        Returns:
            Dict[Tuple[int, int], jnp.ndarray]: Dictionary of bivector matrices.
        """
        bivectors = {}
        for i in range(4):
            for j in range(i + 1, 4):
                bivectors[(i, j)] = 0.25 * self.commutator(gammas[i], gammas[j])
        return bivectors

    def _build_graded_basis(self) -> Dict[int, list[jnp.ndarray]]:
        """
        Constructs the full basis of the Clifford algebra organized by grade.
        
        Returns:
            Dict[int, list[jnp.ndarray]]: Dictionary mapping grade (0-4) to basis elements.
        """
        # Grade 0: Scalar (Identity)
        I4 = jnp.eye(4, dtype=jnp.complex128)
        basis = {0: [I4]}
        
        # Grade 1: Vectors (Gammas)
        basis[1] = self.gammas
        
        # Grade 2: Bivectors
        basis[2] = list(self.bivectors.values())
        
        # Grade 3: Trivectors
        trivectors = []
        indices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        for idxs in indices:
            bv = self.bivectors[(idxs[0], idxs[1])]
            tv = self.geometric_product(bv, self.gammas[idxs[2]])
            trivectors.append(tv)
        basis[3] = trivectors
        
        # Grade 4: Pseudoscalar
        ps = self.geometric_product(trivectors[0], self.gammas[3])
        basis[4] = [ps]
        
        return basis

    @functools.partial(jit, static_argnums=(0,))
    def geometric_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the geometric product of two multivectors.

        Args:
            a (jnp.ndarray): The first multivector.
            b (jnp.ndarray): The second multivector.

        Returns:
            jnp.ndarray: The geometric product (a @ b).
        """
        return jnp.matmul(a, b)

    @functools.partial(jit, static_argnums=(0,))
    def commutator(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the commutator [A, B] = AB - BA.

        Args:
            a (jnp.ndarray): The first multivector.
            b (jnp.ndarray): The second multivector.

        Returns:
            jnp.ndarray: The commutator.
        """
        return a @ b - b @ a

    @functools.partial(jit, static_argnums=(0,))
    def anticommutator(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the anticommutator {A, B} = AB + BA.

        Args:
            a (jnp.ndarray): The first multivector.
            b (jnp.ndarray): The second multivector.

        Returns:
            jnp.ndarray: The anticommutator.
        """
        return a @ b + b @ a

    def get_bivector(self, i: int, j: int) -> jnp.ndarray:
        """
        Retrieves a bivector generator S_ij.

        Args:
            i (int): First index (0 to 3).
            j (int): Second index (0 to 3).

        Returns:
            jnp.ndarray: The bivector matrix S_ij.
        """
        if i < j:
            return self.bivectors[(i, j)]
        elif i > j:
            return -self.bivectors[(j, i)]
        else:
            return jnp.zeros_like(self.gammas[0])

    def random_spinor(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Generates a random, normalized 4-component complex spinor.

        Args:
            key (jax.random.PRNGKey): JAX PRNG key.

        Returns:
            jnp.ndarray: A 4x1 complex spinor.
        """
        real = random.normal(key, (4, 1), dtype=jnp.float64)
        imag = random.normal(random.fold_in(key, 1), (4, 1), dtype=jnp.float64)
        psi = real + 1j * imag
        psi = psi / jnp.linalg.norm(psi)
        return psi

    @functools.partial(jit, static_argnums=(0,))
    def reverse(self, multivector: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Clifford reverse of a multivector.

        Args:
            multivector (jnp.ndarray): The multivector.

        Returns:
            jnp.ndarray: The reversed multivector.
        """
        return jnp.transpose(multivector)

    @functools.partial(jit, static_argnums=(0,))
    def scalar_part(self, multivector: jnp.ndarray) -> jnp.complex128:
        """
        Extracts the scalar part (grade-0 component) of a multivector.

        Args:
            multivector (jnp.ndarray): The multivector.

        Returns:
            jnp.complex128: The scalar part.
        """
        return jnp.trace(multivector) / self.dim

    @functools.partial(jit, static_argnums=(0,))
    def adjoint(self, multivector: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Hermitian adjoint (conjugate transpose) of a multivector.

        Args:
            multivector (jnp.ndarray): The multivector.

        Returns:
            jnp.ndarray: The adjoint of the multivector.
        """
        return jnp.conjugate(jnp.transpose(multivector))

    @functools.partial(jit, static_argnums=(0, 2))
    def project_grade(self, multivector: jnp.ndarray, grade: int) -> jnp.ndarray:
        """
        Projects a multivector onto a specific geometric grade.
        
        Args:
            multivector (jnp.ndarray): The input multivector (4x4 matrix).
            grade (int): The target grade (0-4).
            
        Returns:
            jnp.ndarray: The component of the multivector of the specified grade.
        """
        if grade not in self.basis_by_grade:
             raise ValueError(f"Invalid grade: {grade}. Must be 0-4.")
             
        basis_elements = self.basis_by_grade[grade]
        projected = jnp.zeros_like(multivector)
        
        for basis in basis_elements:
            # B^-1 = B / (B * B)_scalar
            sq = self.geometric_product(basis, basis)
            s = self.scalar_part(sq)
            basis_inv = basis / s
            
            term = self.geometric_product(multivector, basis_inv)
            coeff = self.scalar_part(term)
            
            projected = projected + coeff * basis
            
        return projected

    @functools.partial(jit, static_argnums=(0,))
    def wedge_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Wedge (Outer) Product of two multivectors A ^ B.

        The wedge product represents the spanning of a subspace (volume creation).
        It is the grade-increasing part of the geometric product.
        
        Definition:
        A ^ B = sum_{r, s} < <A>_r * <B>_s >_{r+s}
        
        where <M>_k denotes the projection of multivector M onto grade k.

        Args:
            a (jnp.ndarray): First multivector (4x4 matrix).
            b (jnp.ndarray): Second multivector (4x4 matrix).

        Returns:
            jnp.ndarray: The wedge product A ^ B.
        """
        result = jnp.zeros_like(a)

        # Iterate through all grade combinations of inputs
        # Max grade in 4D is 4.
        for r in range(5):
            for s in range(5):
                target_grade = r + s
                
                # The wedge product is zero if the resulting grade exceeds the dimension (4)
                if target_grade <= 4:
                    # 1. Extract grade components
                    a_r = self.project_grade(a, r)
                    b_s = self.project_grade(b, s)

                    # 2. Compute geometric product of these components
                    # prod = A_r * B_s
                    prod = self.geometric_product(a_r, b_s)

                    # 3. Project the product strictly onto the target grade (r+s)
                    # term = < A_r * B_s >_{r+s}
                    term = self.project_grade(prod, target_grade)

                    result = result + term

        return result

    @functools.partial(jit, static_argnums=(0,))
    def inverse(self, multivector: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Clifford inverse of a multivector A.

        Args:
            multivector (jnp.ndarray): The multivector to invert.

        Returns:
            jnp.ndarray: The inverse of the multivector.
        """
        rev_multivector = self.reverse(multivector)
        denominator_multivector = self.geometric_product(multivector, rev_multivector)
        scalar_denominator = self.scalar_part(denominator_multivector)
        
        if jnp.isclose(scalar_denominator, 0.0):
            raise ValueError("Multivector is not invertible.")
        
        inv_multivector = rev_multivector / scalar_denominator
        return inv_multivector
