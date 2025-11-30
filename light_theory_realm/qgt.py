import jax
import jax.numpy as jnp
from jax import jit
import functools
from typing import Tuple
from .engine import CliffordEngine

class CliffordQGT:
    """
    Quantum Geometric Tensor (QGT) estimator using Clifford Algebra principles.

    The Quantum Geometric Tensor G_uv is a fundamental quantity in quantum
    mechanics and quantum information, characterizing the local geometry of
    the parameter space of a quantum state manifold. It is defined as:

    G_uv = <d_u psi | d_v psi> - <d_u psi | psi><psi | d_v psi>

    where |psi> is a quantum state, and d_u denotes differentiation with
    respect to a parameter theta_u. The real part of the QGT is the
    Fisher information metric, and its imaginary part is the Berry curvature.

    This class leverages a `CliffordEngine` instance for underlying algebraic
    operations on quantum states (spinors/multivectors).

    Attributes:
        engine (CliffordEngine): An instance of CliffordEngine for Clifford algebra operations.
    """
    
    def __init__(self, engine: CliffordEngine):
        """
        Initializes the CliffordQGT with a CliffordEngine instance.

        Args:
            engine (CliffordEngine): An initialized CliffordEngine object that provides
                                     Clifford algebra operations.
        """
        self.engine = engine

    @functools.partial(jit, static_argnums=(0,))
    def inner_product(self, psi_a: jnp.ndarray, psi_b: jnp.ndarray) -> jnp.complex128:
        """
        Computes the Hilbert space inner product <psi_a | psi_b> between two spinors.

        For column spinors, this is computed as the Hermitian inner product:
        psi_a^dagger @ psi_b (conjugate transpose of psi_a multiplied by psi_b).

        Args:
            psi_a (jnp.ndarray): The first spinor (e.g., a 4x1 complex JAX array).
            psi_b (jnp.ndarray): The second spinor (e.g., a 4x1 complex JAX array).

        Returns:
            jnp.complex128: The complex scalar result of the inner product.
        """
        # Use the engine's adjoint for Clifford compliance (psi_a^dagger)
        bra = self.engine.adjoint(psi_a)
        # Calculate the product <psi_a | psi_b> = bra * ket
        # This returns a (1,1) matrix for column spinors
        result_matrix = self.engine.geometric_product(bra, psi_b)
        # Extract the scalar value
        return result_matrix.reshape(-1)[0]

    @functools.partial(jit, static_argnums=(0,))
    def compute_qgt_components(self, psi: jnp.ndarray, d_psi_u: jnp.ndarray, d_psi_v: jnp.ndarray) -> Tuple[jnp.complex128, jnp.complex128]:
        """
        Computes the individual components (G_uv) of the Quantum Geometric Tensor.

        G_uv = <d_u psi | d_v psi> - <d_u psi | psi><psi | d_v psi>

        This method calculates the complex QGT component G_uv, and then separates
        it into its real (Fisher information) and imaginary (Berry curvature) parts.

        Args:
            psi (jnp.ndarray): The current quantum state (spinor), typically a 4x1 complex JAX array.
            d_psi_u (jnp.ndarray): The derivative of the state psi with respect to parameter u,
                                   same shape as psi.
            d_psi_v (jnp.ndarray): The derivative of the state psi with respect to parameter v,
                                   same shape as psi.

        Returns:
            Tuple[jnp.complex128, jnp.complex128]: A tuple containing:
                - fisher_uv (jnp.complex128): The real part of G_uv, representing the
                                              Fisher information metric component.
                - berry_uv (jnp.complex128): The imaginary part of G_uv, representing the
                                             Berry curvature component.
        """
        # Term 1: <d_u psi | d_v psi>
        term1 = self.inner_product(d_psi_u, d_psi_v)
        
        # Term 2: <d_u psi | psi><psi | d_v psi>
        # Note: <psi | d_v psi> is purely imaginary if psi is normalized and gauge fixed, 
        # but generally it's complex.
        term2 = self.inner_product(d_psi_u, psi) * self.inner_product(psi, d_psi_v)
        
        G_uv = term1 - term2
        
        fisher_uv = jnp.real(G_uv)
        berry_uv = jnp.imag(G_uv)
        
        return fisher_uv, berry_uv

    @functools.partial(jit, static_argnums=(0,))
    def compute_full_qgt(self, psi: jnp.ndarray, jacobian: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the full Quantum Geometric Tensor (QGT) matrices for a set of parameters.

        This method efficiently calculates the QGT matrix (G) from the state vector
        and its Jacobian with respect to the parameters. The full QGT is decomposed
        into its real (Fisher information matrix) and imaginary (Berry curvature matrix) parts.

        Args:
            psi (jnp.ndarray): The current quantum state vector, typically a (dim,) or (dim, 1) JAX array.
                               Assumed to be normalized.
            jacobian (jnp.ndarray): The Jacobian matrix of the state with respect to the parameters,
                                    with shape (dim, num_params). Each column d_psi_u represents
                                    the derivative of the state with respect to a parameter.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - fisher_matrix (jnp.ndarray): The real, symmetric (num_params, num_params) matrix,
                                               representing the Quantum Fisher Information Metric.
                - berry_matrix (jnp.ndarray): The real, antisymmetric (num_params, num_params) matrix,
                                              representing the Berry curvature tensor.
        """
        num_params = jacobian.shape[1]
        
        # Compute overlaps matrix O_uv = <d_u psi | d_v psi> = J^dagger @ J
        # Use engine methods for Clifford compliance
        jacobian_dag = self.engine.adjoint(jacobian)
        overlaps = self.engine.geometric_product(jacobian_dag, jacobian)
        
        # Compute Berry connection vector A_u = <psi | d_u psi>
        psi_dag = self.engine.adjoint(psi)
        connections = self.engine.geometric_product(psi_dag, jacobian)
        
        # Outer product term P_uv = <d_u psi | psi><psi | d_v psi> = A_u^* A_v
        # This forms a matrix where P_uv = connections[u].conj() * connections[v]
        # Note: 'connections' is already a row vector from the geometric product (1, N)
        # We need to flatten it to use jnp.outer effectively, or handle dims
        connections_flat = connections.reshape(-1)
        projector = jnp.outer(connections_flat.conj(), connections_flat)
        
        # The full Quantum Geometric Tensor G = O - P
        G = overlaps - projector
        
        fisher_matrix = jnp.real(G)
        berry_matrix = jnp.imag(G)
        
        return fisher_matrix, berry_matrix
