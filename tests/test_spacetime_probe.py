# tests/test_spacetime_probe.py

import jax
import jax.numpy as jnp
from jax import random
import pytest
from typing import Tuple

from light_theory_realm.engine import CliffordEngine
from light_theory_realm.geometry.spacetime_probe import gravity_probe, GravityProbeResult
from light_theory_realm.qgt import CliffordQGT # For independent QGT verification
from light_theory_realm.experiments.prime_gauge.reeb_flow import ReebFlowDynamics # For independent Reeb verification
from light_theory_realm.experiments.prime_gauge.uplift import KaluzaKleinUplift # For independent KK verification

# Configure JAX to use float64 for tests
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def clifford_engine():
    """Fixture for CliffordEngine."""
    return CliffordEngine(seed=0)

class TestGravityProbe:
    @pytest.fixture
    def single_param_psi_fn(self, clifford_engine: CliffordEngine):
        """
        Fixture for a simple single-parameter psi_fn for testing.
        psi(theta) = exp(i * theta * G) @ base_spinor
        """
        base_spinor = clifford_engine.random_spinor(jax.random.PRNGKey(42))
        G = clifford_engine.gammas[0] @ clifford_engine.gammas[1] # A fixed bivector generator

        @jax.jit
        def psi_fn_local(theta_vec: jnp.ndarray) -> jnp.ndarray:
            # theta_vec is expected to be a 1-element JAX array
            angle = theta_vec[0]
            U = jax.scipy.linalg.expm(1j * angle * G)
            return U @ base_spinor
        return psi_fn_local

    def test_gravity_probe_output_shapes_and_types(self, clifford_engine: CliffordEngine, single_param_psi_fn):
        """
        Tests that the gravity_probe function returns a GravityProbeResult
        with components of the expected shapes and general properties.
        """
        num_params = 1
        theta_test = jnp.array([0.5]) # A single parameter value
        test_primes: Tuple[int, ...] = (2, 3, 5) # A simple set of primes
        
        result = gravity_probe(single_param_psi_fn, theta_test, test_primes, clifford_engine)

        # Assert type of the result
        assert isinstance(result, GravityProbeResult)

        # Assert shapes of array components
        assert result.fisher_metric.shape == (num_params, num_params)
        assert result.berry_curvature.shape == (num_params, num_params)
        assert result.reeb_vector.shape == (num_params,)
        assert result.state_vector.shape == (4, 1) # Spinor is 4x1
        assert result.jacobian.shape == (4, num_params) # Jacobian should be (spinor_dim, num_params)
        
        # KK metric shape should be (num_params + 1, num_params + 1)
        assert result.kk_metric.shape == (num_params + 1, num_params + 1)

        # Assert scalar types and basic ranges
        assert isinstance(result.rho_igbp, float)
        assert isinstance(result.xi_resonance, float)
        assert result.rho_igbp >= 0.0 # Information density should be non-negative
        assert 0.0 <= result.xi_resonance <= 1.0 # Resonance density is between 0 and 1

        # Assert that the matrices are real (Fisher) and imaginary (Berry) parts
        assert jnp.allclose(result.fisher_metric, jnp.real(result.fisher_metric))
        assert jnp.allclose(result.berry_curvature, jnp.imag(result.berry_curvature))

        # Further basic checks (e.g., Fisher should be symmetric and positive semi-definite)
        assert jnp.allclose(result.fisher_metric, result.fisher_metric.T) # Symmetric
        # For positive semi-definite, check eigenvalues (can be complex if not Hermitian for small numerical errors)
        min_eig = jnp.min(jnp.linalg.eigvalsh(result.fisher_metric))
        assert min_eig >= -1e-9 # Allow small negative for numerical stability

    def test_gravity_probe_values_consistency(self, clifford_engine: CliffordEngine, single_param_psi_fn):
        """
        Tests consistency of calculated values with individual component calls.
        """
        num_params = 1
        theta_test = jnp.array([0.5])
        test_primes: Tuple[int, ...] = (2, 3, 5)
        
        result = gravity_probe(single_param_psi_fn, theta_test, test_primes, clifford_engine)

        # --- Independent Verification ---
        psi = single_param_psi_fn(theta_test)
        jacobian = jax.jacfwd(single_param_psi_fn)(theta_test)
        jacobian = jacobian.reshape(psi.shape[0], num_params) # Ensure correct shape

        # 1. QGT (Fisher & Berry)
        qgt_computer = CliffordQGT(clifford_engine)
        expected_fisher, expected_berry = qgt_computer.compute_full_qgt(psi, jacobian)
        assert jnp.allclose(result.fisher_metric, expected_fisher)
        assert jnp.allclose(result.berry_curvature, expected_berry)

        # 2. Reeb Resonance
        reeb_computer = ReebFlowDynamics(clifford_engine)
        expected_xi_resonance = reeb_computer.compute_resonance_density(test_primes)
        assert jnp.isclose(result.xi_resonance, expected_xi_resonance)

        # 3. Reeb Vector (shape and normalization)
        expected_reeb_vector = reeb_computer.compute_reeb_vector(expected_fisher, expected_berry)
        assert jnp.allclose(result.reeb_vector, expected_reeb_vector)
        # Check normalization with Fisher metric: R^T g R = 1
        norm_sq = jnp.dot(expected_reeb_vector.T, jnp.dot(expected_fisher, expected_reeb_vector))
        assert jnp.isclose(norm_sq, 1.0)

        # 4. Kaluza-Klein Metric
        kk_computer = KaluzaKleinUplift(clifford_engine)
        expected_kk_metric = kk_computer.construct_5d_metric(psi, jacobian) # Uses default phi_scalar=1.0
        assert jnp.allclose(result.kk_metric, expected_kk_metric)

        # Note: rho_igbp is calculated internally in spacetime_probe.py,
        # so direct independent verification is harder without replicating the logic.
        # We've already checked its basic properties (non-negative float).
