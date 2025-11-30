import jax
import jax.numpy as jnp
from jax import random
import pytest

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.qgt import CliffordQGT

# Configure JAX to use float64 for tests
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def clifford_engine():
    """Fixture for CliffordEngine."""
    return CliffordEngine(seed=0)

@pytest.fixture
def clifford_qgt(clifford_engine):
    """Fixture for CliffordQGT."""
    return CliffordQGT(engine=clifford_engine)

class TestCliffordQGT:
    def test_inner_product(self, clifford_qgt):
        """
        Test the inner_product method with simple known spinors.
        """
        # Test 1: Orthogonal spinors
        psi_a = jnp.array([[1.0], [0.0], [0.0], [0.0]], dtype=jnp.complex128)
        psi_b = jnp.array([[0.0], [1.0], [0.0], [0.0]], dtype=jnp.complex128)
        assert jnp.isclose(clifford_qgt.inner_product(psi_a, psi_b), 0.0)

        # Test 2: Identical spinors (normalized)
        psi_c = jnp.array([[1.0/jnp.sqrt(2)], [1.0/jnp.sqrt(2)], [0.0], [0.0]], dtype=jnp.complex128)
        assert jnp.isclose(clifford_qgt.inner_product(psi_c, psi_c), 1.0)

        # Test 3: Complex spinors
        psi_d = jnp.array([[1.0 + 1.0j], [2.0 - 1.0j], [0.0], [0.0]], dtype=jnp.complex128)
        psi_e = jnp.array([[0.5 - 0.5j], [1.0 + 2.0j], [0.0], [0.0]], dtype=jnp.complex128)
        expected_ip = jnp.vdot(psi_d, psi_e)
        assert jnp.isclose(clifford_qgt.inner_product(psi_d, psi_e), expected_ip)

    def test_compute_qgt_components_simple(self, clifford_qgt):
        """
        Test compute_qgt_components with a trivial case where psi does not change.
        In this case, d_psi_u = d_psi_v = 0, so QGT should be 0.
        """
        psi = jnp.array([[1.0], [0.0], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_u = jnp.zeros_like(psi)
        d_psi_v = jnp.zeros_like(psi)

        fisher_uv, berry_uv = clifford_qgt.compute_qgt_components(psi, d_psi_u, d_psi_v)
        
        assert jnp.isclose(fisher_uv, 0.0)
        assert jnp.isclose(berry_uv, 0.0)

    def test_compute_qgt_components_basic(self, clifford_qgt):
        """
        Test compute_qgt_components with a simple non-trivial case.
        Consider a 2-level system for simplicity (using first two components of 4-vector).
        psi = [cos(theta/2), sin(theta/2)]
        d_psi/d_theta = [-sin(theta/2)/2, cos(theta/2)/2]

        Let psi = [1, 0, 0, 0]^T (theta=0)
        Then d_psi/d_theta = [0, 1/2, 0, 0]^T
        If d_psi_u = d_psi_v = [0, 1/2, 0, 0]^T and psi = [1, 0, 0, 0]^T
        <d_psi_u | d_psi_v> = 1/4
        <d_psi_u | psi> = 0
        <psi | d_psi_v> = 0
        So, G_uv = 1/4 (Fisher_uv = 1/4, Berry_uv = 0)
        """
        psi = jnp.array([[1.0], [0.0], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_u = jnp.array([[0.0], [0.5], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_v = jnp.array([[0.0], [0.5], [0.0], [0.0]], dtype=jnp.complex128)

        fisher_uv, berry_uv = clifford_qgt.compute_qgt_components(psi, d_psi_u, d_psi_v)

        assert jnp.isclose(fisher_uv, 0.25)
        assert jnp.isclose(berry_uv, 0.0)
    
    def test_compute_qgt_components_complex(self, clifford_qgt):
        """
        Test compute_qgt_components with a more complex scenario.
        Consider psi = [1/sqrt(2), 1/sqrt(2) * exp(i*phi), 0, 0]^T
        Let's take derivatives with respect to a parameter that affects phi.
        Suppose d_psi_u has an imaginary component and psi is normalized.
        
        psi = 1/sqrt(2) * [1, exp(1j*phi), 0, 0]^T
        d_psi_u = 1/sqrt(2) * [0, 1j*exp(1j*phi), 0, 0]^T * d_phi/d_u (let d_phi/d_u = 1 for simplicity)
        
        Let phi = 0
        psi = 1/sqrt(2) * [1, 1, 0, 0]^T
        d_psi_u = 1/sqrt(2) * [0, 1j, 0, 0]^T
        d_psi_v = 1/sqrt(2) * [0, 1j, 0, 0]^T (same derivative)
        
        <d_psi_u | d_psi_v> = (1/2) * (0*0 + (-1j)*(1j)) = (1/2) * (1) = 0.5
        <d_psi_u | psi> = (1/2) * (0*1 + (-1j)*1) = -0.5j
        <psi | d_psi_v> = (1/2) * (1*0 + 1*(1j)) = 0.5j
        
        Term 1 = 0.5
        Term 2 = <d_psi_u | psi> * <psi | d_psi_v> = (-0.5j) * (0.5j) = -0.25 * (j*j) = 0.25
        
        G_uv = Term1 - Term2 = 0.5 - 0.25 = 0.25
        Fisher_uv = 0.25, Berry_uv = 0.0
        
        Now, let's make d_psi_v orthogonal to d_psi_u (e.g., real derivative affecting first component)
        d_psi_u = 1/sqrt(2) * [0, 1j, 0, 0]^T
        d_psi_v = 1/sqrt(2) * [1, 0, 0, 0]^T
        
        <d_psi_u | d_psi_v> = 0
        <d_psi_u | psi> = -0.5j
        <psi | d_psi_v> = 0.5
        
        Term 1 = 0
        Term 2 = (-0.5j) * 0.5 = -0.25j
        
        G_uv = 0 - (-0.25j) = 0.25j
        Fisher_uv = 0.0, Berry_uv = 0.25
        """
        # Test Case 1: d_psi_u = d_psi_v (phi derivative)
        psi_c1 = jnp.array([[1.0/jnp.sqrt(2)], [1.0/jnp.sqrt(2)], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_u_c1 = jnp.array([[0.0], [1.0j/jnp.sqrt(2)], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_v_c1 = jnp.array([[0.0], [1.0j/jnp.sqrt(2)], [0.0], [0.0]], dtype=jnp.complex128)

        fisher_uv_c1, berry_uv_c1 = clifford_qgt.compute_qgt_components(psi_c1, d_psi_u_c1, d_psi_v_c1)
        assert jnp.isclose(fisher_uv_c1, 0.25)
        assert jnp.isclose(berry_uv_c1, 0.0)

        # Test Case 2: d_psi_u and d_psi_v are "orthogonal"
        psi_c2 = jnp.array([[1.0/jnp.sqrt(2)], [1.0/jnp.sqrt(2)], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_u_c2 = jnp.array([[0.0], [1.0j/jnp.sqrt(2)], [0.0], [0.0]], dtype=jnp.complex128)
        d_psi_v_c2 = jnp.array([[1.0/jnp.sqrt(2)], [0.0], [0.0], [0.0]], dtype=jnp.complex128)

        fisher_uv_c2, berry_uv_c2 = clifford_qgt.compute_qgt_components(psi_c2, d_psi_u_c2, d_psi_v_c2)
        assert jnp.isclose(fisher_uv_c2, 0.0)
        assert jnp.isclose(berry_uv_c2, 0.25)

    def test_compute_full_qgt_simple(self, clifford_qgt):
        """
        Test compute_full_qgt with a simple, analytically verifiable scenario.
        
        Scenario:
        psi = [1, 0, 0, 0]^T (normalized)
        jacobian (d_psi/d_theta) with 2 parameters:
        d_psi/d_theta_0 = [0, 1, 0, 0]^T
        d_psi/d_theta_1 = [0, 0, 0, 0]^T
        
        Expected analytical results:
        overlaps = jacobian^dagger @ jacobian = [[1, 0], [0, 0]]
        connections = psi^dagger @ jacobian = [0, 0]
        projector = outer(connections^dagger, connections) = [[0, 0], [0, 0]]
        G = overlaps - projector = [[1, 0], [0, 0]]
        
        fisher_matrix = real(G) = [[1, 0], [0, 0]]
        berry_matrix = imag(G) = [[0, 0], [0, 0]]
        """
        psi = jnp.array([[1.0], [0.0], [0.0], [0.0]], dtype=jnp.complex128)
        
        jacobian = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ], dtype=jnp.complex128)

        expected_fisher = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
        expected_berry = jnp.array([[0.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)

        fisher_matrix, berry_matrix = clifford_qgt.compute_full_qgt(psi, jacobian)

        assert jnp.allclose(fisher_matrix, expected_fisher)
        assert jnp.allclose(berry_matrix, expected_berry)