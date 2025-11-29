# import pytest
import jax
import jax.numpy as jnp
from src.engine import CliffordEngine
from src.qgt import CliffordQGT

def test_clifford_algebra_properties():
    """Verify basic Cl(6) properties."""
    engine = CliffordEngine(seed=42)
    
    # Verify anticommutation of gammas: {Gi, Gj} = 2 * delta_ij * I
    for i in range(6):
        for j in range(6):
            Gi = engine.gammas[i]
            Gj = engine.gammas[j]
            
            anti_comm = engine.anticommutator(Gi, Gj)
            
            if i == j:
                expected = 2.0 * jnp.eye(8, dtype=jnp.complex128)
            else:
                expected = jnp.zeros((8, 8), dtype=jnp.complex128)
                
            assert jnp.allclose(anti_comm, expected, atol=1e-10), f"Failed for {i}, {j}"

def test_qgt_properties():
    """Verify QGT symmetry and antisymmetry."""
    engine = CliffordEngine(seed=123)
    qgt = CliffordQGT(engine)
    
    # Create a dummy parameterized state: psi(theta)
    # Let's define a simple rotation: psi(theta) = exp(-i * theta * G12/2) * psi_0
    
    psi_0 = engine.random_spinor(jax.random.PRNGKey(0))
    G12 = engine.get_bivector(0, 1)
    
    # Define a function to get psi from params
    def get_psi(theta):
        # theta is a scalar for simplicity here
        # Rotor R = exp(-0.5 * theta * G12)
        # Since G12^2 = -1, exp(alpha G12) = cos(alpha) + G12 sin(alpha)
        angle = -0.5 * theta[0]
        rotor = jnp.cos(angle) * jnp.eye(8) + jnp.sin(angle) * G12
        return rotor @ psi_0

    # Compute Jacobian using JAX
    theta_val = jnp.array([0.5])
    
    # We need a Jacobian of shape (dim, num_params)
    # jax.jacfwd returns (dim, 1, num_params) if input is (1,)
    jacobian_fn = jax.jacfwd(get_psi)
    jacobian = jacobian_fn(theta_val).reshape(8, 1)
    psi_val = get_psi(theta_val)
    
    fisher, berry = qgt.compute_full_qgt(psi_val, jacobian)
    
    # For a single parameter, Fisher is scalar, Berry is 0
    assert fisher.shape == (1, 1)
    assert berry.shape == (1, 1)
    assert jnp.abs(berry[0, 0]) < 1e-10  # Berry curvature is 0 for 1D manifold
    assert fisher[0, 0] > 0  # Metric should be positive
    
    # Now let's try 2 parameters to see non-zero Berry curvature
    # psi(theta1, theta2) = exp(-theta1 G12) exp(-theta2 G23) psi_0
    G23 = engine.get_bivector(1, 2)
    
    def get_psi_2d(theta):
        t1, t2 = theta[0], theta[1]
        R1 = jnp.cos(-0.5*t1) * jnp.eye(8) + jnp.sin(-0.5*t1) * G12
        R2 = jnp.cos(-0.5*t2) * jnp.eye(8) + jnp.sin(-0.5*t2) * G23
        return R1 @ (R2 @ psi_0)

    theta_val_2d = jnp.array([0.5, 0.3])
    jacobian_2d = jax.jacfwd(get_psi_2d)(theta_val_2d).reshape(8, 2)
    psi_val_2d = get_psi_2d(theta_val_2d)
    
    fisher_2d, berry_2d = qgt.compute_full_qgt(psi_val_2d, jacobian_2d)
    
    # Verify Fisher symmetry
    assert jnp.allclose(fisher_2d, fisher_2d.T)
    
    # Verify Berry antisymmetry
    assert jnp.allclose(berry_2d, -berry_2d.T)
    
    # Check non-triviality (should be non-zero generally)
    # Note: might be zero at specific points, but unlikely for random state
    print(f"Fisher:\n{fisher_2d}")
    print(f"Berry:\n{berry_2d}")

if __name__ == "__main__":
    test_clifford_algebra_properties()
    test_qgt_properties()
    print("All tests passed!")
