# import pytest
import jax
import jax.numpy as jnp
from src.engine import CliffordEngine
from src.qgt import CliffordQGT

def test_clifford_algebra_properties():
    """Verify basic Cl(1,3) properties in Weyl representation."""
    engine = CliffordEngine(seed=42)
    
    # Verify anticommutation of gammas: {Gu, Gv} = 2 * eta_uv * I
    # For Cl(1,3), eta = diag(1, -1, -1, -1)
    metric = jnp.diag(jnp.array([1, -1, -1, -1], dtype=jnp.complex128))
    
    for i in range(4): # Now 4 gamma matrices
        for j in range(4):
            Gi = engine.gammas[i]
            Gj = engine.gammas[j]
            
            anti_comm = engine.anticommutator(Gi, Gj)
            
            expected = 2.0 * metric[i, j] * jnp.eye(4, dtype=jnp.complex128) # Now 4x4
                
            assert jnp.allclose(anti_comm, expected, atol=1e-10), f"Failed for {i}, {j}"

def test_qgt_properties():
    """Verify QGT symmetry and antisymmetry."""
    engine = CliffordEngine(seed=123)
    qgt = CliffordQGT(engine)
    
    # Create a dummy parameterized state: psi(theta)
    # Let's define a simple rotation: psi(theta) = exp(-i * theta * G12/2) * psi_0
    
    psi_0 = engine.random_spinor(jax.random.PRNGKey(0))
    
    # Using G0 and G1 for example. Bivectors are S_uv = 1/4 [Gu, Gv]
    S01 = engine.get_bivector(0, 1) # This is a bivector for Lorentz transformation
    
    # Let's pick a simple Hermitian generator:
    gen = engine.gammas[0] @ engine.gammas[1] # This is Hermitian
    gen2 = engine.gammas[1] @ engine.gammas[2] # Another Hermitian generator
    
    # Define a function to get psi from params
    def get_psi(theta):
        # theta is a scalar for simplicity here
        
        angle = theta[0] # Define angle here
        # U = exp(-1j * angle * gen)
        U = jax.scipy.linalg.expm(-1j * angle * gen)
        return U @ psi_0

    # Compute Jacobian using JAX
    theta_val = jnp.array([0.5])
    
    # We need a Jacobian of shape (dim, num_params)
    jacobian_fn = jax.jacfwd(get_psi)
    jacobian = jacobian_fn(theta_val).reshape(4, 1) # Now 4x1
    psi_val = get_psi(theta_val)
    
    fisher, berry = qgt.compute_full_qgt(psi_val, jacobian)
    
    # For a single parameter, Fisher is scalar, Berry is 0
    assert fisher.shape == (1, 1)
    assert berry.shape == (1, 1)
    assert jnp.abs(berry[0, 0]) < 1e-10  # Berry curvature is 0 for 1D manifold
    assert fisher[0, 0] > 0  # Metric should be positive
    
    # Now let's try 2 parameters to see non-zero Berry curvature
    # psi(theta1, theta2) = U(theta1) U(U2) psi_0
    
    def get_psi_2d(theta):
        t1, t2 = theta[0], theta[1]
        U1 = jax.scipy.linalg.expm(-1j * t1 * gen)
        U2 = jax.scipy.linalg.expm(-1j * t2 * gen2)
        return U1 @ (U2 @ psi_0)

    theta_val_2d = jnp.array([0.5, 0.3])
    jacobian_2d = jax.jacfwd(get_psi_2d)(theta_val_2d).reshape(4, 2) # Now 4x2
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
