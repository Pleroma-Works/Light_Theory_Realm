import jax
import jax.numpy as jnp
from jax import jit
import functools
from typing import Tuple
from .engine import CliffordEngine

class CliffordQGT:
    """
    Quantum Geometric Tensor (QGT) estimator using Clifford Algebra.
    
    The QGT is defined as:
    G_uv = <d_u psi | d_v psi> - <d_u psi | psi><psi | d_v psi>
    
    In Clifford Algebra, the inner product <psi|phi> corresponds to the scalar part 
    of the geometric product (psi_dagger * phi).
    """
    
    def __init__(self, engine: CliffordEngine):
        self.engine = engine

    @functools.partial(jit, static_argnums=(0,))
    def inner_product(self, psi_a: jnp.ndarray, psi_b: jnp.ndarray) -> jnp.complex128:
        """
        Compute Hilbert space inner product <psi_a | psi_b>.
        For column spinors, this is psi_a^dagger @ psi_b.
        """
        return jnp.vdot(psi_a, psi_b)

    @functools.partial(jit, static_argnums=(0,))
    def compute_qgt_components(self, psi: jnp.ndarray, d_psi_u: jnp.ndarray, d_psi_v: jnp.ndarray) -> Tuple[jnp.complex128, jnp.complex128]:
        """
        Compute the components of the QGT G_uv.
        
        G_uv = <d_u psi | d_v psi> - <d_u psi | psi><psi | d_v psi>
        
        Returns:
            fisher_uv: Real part (Symmetric)
            berry_uv: Imaginary part (Antisymmetric)
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
        Compute the full QGT matrix for a set of parameters.
        
        Args:
            psi: State vector (dim,)
            jacobian: Jacobian matrix d_psi / d_theta (dim, num_params)
            
        Returns:
            fisher_matrix: (num_params, num_params) Real symmetric
            berry_matrix: (num_params, num_params) Real antisymmetric (representing imaginary part)
        """
        num_params = jacobian.shape[1]
        
        # Compute overlaps matrix O_uv = <d_u psi | d_v psi>
        # O = J^dagger @ J
        overlaps = jacobian.conj().T @ jacobian
        
        # Compute Berry connection vector A_u = <psi | d_u psi>
        # A = psi^dagger @ J
        connections = psi.conj().T @ jacobian
        
        # Outer product term P_uv = <d_u psi | psi><psi | d_v psi> = A_u^* A_v
        projector = jnp.outer(connections.conj(), connections)
        
        # QGT G = O - P
        G = overlaps - projector
        
        fisher_matrix = jnp.real(G)
        berry_matrix = jnp.imag(G)
        
        return fisher_matrix, berry_matrix
