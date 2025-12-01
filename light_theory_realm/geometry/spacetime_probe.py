# light_theory_realm/geometry/spacetime_probe.py

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Callable

# Relative imports for project components
from ..engine import CliffordEngine
from ..qgt import CliffordQGT
from ..experiments.prime_gauge.reeb_flow import ReebFlowDynamics
from ..experiments.prime_gauge.uplift import KaluzaKleinUplift

@dataclass
class GravityProbeResult:
    """
    A container for the geometric and topological properties of a point
    on the quantum state manifold, interpreted as a 'spacetime' probe.
    """
    fisher_metric: jnp.ndarray
    berry_curvature: jnp.ndarray
    rho_igbp: float
    xi_resonance: float
    reeb_vector: jnp.ndarray
    kk_metric: jnp.ndarray
    state_vector: jnp.ndarray
    jacobian: jnp.ndarray

def _calculate_rho_igbp(psi_fn: Callable, theta: jnp.ndarray, berry_curvature: jnp.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculates the IGBP-like scalar density (rho_IGBP).

    rho_IGBP is defined as the ratio of the squared norm of the entropic drive
    to the field strength of the Berry curvature. It measures how information
    density (entropic gradient) relates to the local topological twist.

    rho_IGBP = |nabla S_rep|^2 / (|Omega| + epsilon)

    Args:
        psi_fn (Callable): A function that takes parameters `theta` and returns a state vector `psi`.
        theta (jnp.ndarray): The parameters at which to evaluate the density.
        berry_curvature (jnp.ndarray): The Berry curvature matrix (Omega).
        epsilon (float): A small constant to prevent division by zero.

    Returns:
        float: The scalar value of rho_IGBP.
    """
    # Define representational entropy S_rep = -sum(p_i * log(p_i))
    def s_rep_fn(t):
        psi = psi_fn(t)
        # Probabilities p_i = |psi_i|^2
        probs = jnp.abs(psi)**2
        # Add epsilon to avoid log(0) for zero-probability states
        return -jnp.sum(probs * jnp.log(probs + epsilon))

    # Gradient of entropy wrt theta: nabla S_rep
    grad_s_rep = jax.grad(s_rep_fn)(theta)
    
    # Squared norm of the entropic gradient: |nabla S_rep|^2
    grad_norm_sq = jnp.dot(grad_s_rep, grad_s_rep)
    
    # Frobenius norm of Berry curvature: |Omega|
    berry_norm = jnp.linalg.norm(berry_curvature)
    
    # IGBP scalar density
    rho_igbp = grad_norm_sq / (berry_norm + epsilon)
    
    return float(rho_igbp)

def gravity_probe(psi_fn: Callable, theta: jnp.ndarray, primes: list, engine: CliffordEngine, phi_scalar: float = 1.0) -> GravityProbeResult:
    """
    Core 'poke gravity' primitive for Light Theory Realm.

    This function takes a parameterized quantum state and probes its geometric
    and topological properties at a specific point in the parameter manifold.
    It computes all the essential tensors and scalars that describe the local
    "spacetime" according to Light Theory principles.

    Args:
        psi_fn (Callable): A function that takes a JAX array `theta` of parameters
                           and returns a quantum state vector `psi`.
        theta (jnp.ndarray): The specific parameter vector at which to probe the geometry.
        primes (list): The list of prime numbers defining the plaquette or interaction,
                       used for calculating vacuum resonance.
        engine (CliffordEngine): An initialized CliffordEngine for backend operations.
        phi_scalar (float): The 'radion' field scalar for Kaluza-Klein uplift,
                            representing the size of the extra dimension.

    Returns:
        GravityProbeResult: A dataclass instance containing the computed Fisher metric,
                            Berry curvature, scalar densities, Reeb vector, and the
                            full Kaluza-Klein metric.
    """
    # 1. Instantiate necessary service classes with the engine
    qgt_computer = CliffordQGT(engine)
    reeb_computer = ReebFlowDynamics(engine)
    kk_computer = KaluzaKleinUplift(engine)

    # 2. Compute the state vector and its Jacobian
    # Ensure theta is a JAX array for differentiation
    theta = jnp.asarray(theta)
    
    # The state vector psi at the given parameters
    psi = psi_fn(theta)
    
    # The Jacobian of the state function w.r.t. parameters.
    # jax.jacfwd is typically more efficient for "tall" Jacobians.
    # The output shape of jacfwd is (output_dim, input_dim), e.g., (4, num_params) for a spinor.
    jacobian = jax.jacfwd(psi_fn)(theta)
    
    # Ensure jacobian is always (spinor_dim, num_params)
    # If psi is (spinor_dim, 1), jacobian from jacfwd will be (spinor_dim, 1, num_params).
    # We need to reshape it to (spinor_dim, num_params).
    spinor_dim = psi.shape[0]
    num_params = len(theta)
    jacobian = jacobian.reshape(spinor_dim, num_params)

    # 3. Compute core geometric tensors using CliffordQGT
    fisher_metric, berry_curvature = qgt_computer.compute_full_qgt(psi, jacobian)

    # 4. Compute scalar densities
    # IGBP-like density (information vs. curvature)
    rho_igbp = _calculate_rho_igbp(psi_fn, theta, berry_curvature)
    
    # Vacuum resonance density (commutativity of generators)
    xi_resonance_jax_array = reeb_computer.compute_resonance_density(primes)
    xi_resonance = float(xi_resonance_jax_array)

    # 5. Compute the time direction (Reeb vector)
    reeb_vector = reeb_computer.compute_reeb_vector(fisher_metric, berry_curvature)

    # 6. Compute the fully uplifted Kaluza-Klein metric
    # Note: construct_5d_metric internally recalculates QGT, but we pass psi/jacobian
    # to maintain a clear API contract. It gets the Fisher metric itself.
    kk_metric = kk_computer.construct_5d_metric(psi, jacobian, phi_scalar)

    # 7. Assemble and return the results
    return GravityProbeResult(
        fisher_metric=fisher_metric,
        berry_curvature=berry_curvature,
        rho_igbp=rho_igbp,
        xi_resonance=xi_resonance,
        reeb_vector=reeb_vector,
        kk_metric=kk_metric,
        state_vector=psi,
        jacobian=jacobian
    )
