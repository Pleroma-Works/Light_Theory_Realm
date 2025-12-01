
import jax
import jax.numpy as jnp
from light_theory_realm.engine import CliffordEngine
from light_theory_realm.qgt import CliffordQGT

def test_shapes():
    print("Testing shapes...")
    theta = jnp.array([0.1, 0.2, 0.3]) # 3 params
    
    def state_fn(th):
        # Returns a 4-component state
        return jnp.array([th[0], th[1], th[2], th[0] + th[1]])
    
    # Expected Jacobian: 4x3
    # [1, 0, 0]
    # [0, 1, 0]
    # [0, 0, 1]
    # [1, 1, 0]
    
    jacobian = jax.jacfwd(state_fn)(theta)
    print(f"Jacobian shape: {jacobian.shape}")
    print("Jacobian content:")
    print(jacobian)
    
    engine = CliffordEngine()
    qgt = CliffordQGT(engine)
    
    # Test QGT
    psi = state_fn(theta)
    # psi shape (4,)
    # jacobian shape (4, 3)
    
    # CliffordQGT expects (dim, params)
    # num_params should be 3
    
    try:
        fisher, berry = qgt.compute_full_qgt(psi, jacobian)
        print(f"Fisher shape: {fisher.shape}")
        print(f"Berry shape: {berry.shape}")
    except Exception as e:
        print(f"QGT failed: {e}")

if __name__ == "__main__":
    test_shapes()
