import jax
import jax.numpy as jnp
import sys
import os

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.engine import CliffordEngine
from src.qgt import CliffordQGT

class KaluzaKleinUplift(CliffordQGT):
    """
    Uplifts a 4D Information Manifold to 5D Kaluza-Klein Geometry.
    
    Inherits from CliffordQGT to access the fundamental quantum state geometry.
    """
    
    def get_berry_connection(self, psi: jnp.ndarray, jacobian: jnp.ndarray):
        """
        Extract the Berry Connection A_mu (The Gauge Potential).
        
        A_mu = -i * <psi | d_mu psi>
        
        In KK theory, this acts as the electromagnetic potential.
        """
        # <psi | d_mu psi>
        # jacobian is d_mu psi (shape: dim, num_params)
        overlap = psi.conj().T @ jacobian
        
        # A_mu should be real-valued for the metric construction in standard KK.
        # Physics convention: A_mu is the imaginary part of the overlap.
        A_mu = jnp.imag(overlap) 
        return A_mu

    def construct_5d_metric(self, psi: jnp.ndarray, jacobian: jnp.ndarray, phi_scalar: float = 1.0):
        """
        Construct the (N+1)x(N+1) Kaluza-Klein Metric Tensor.
        
        Args:
            psi: The state vector (Light String configuration).
            jacobian: The change in state wrt parameters.
            phi_scalar: The 'Radion' field (size of the extra dim). 
                        If phi -> 0, the extra dim creates a singularity (Hard Core).
        
        Returns:
            G_AB: The uplifted metric tensor.
        """
        # 1. Get Base Geometry (Fisher Metric g_uv)
        # We only need the real part (Fisher) for the base block
        fisher, _ = self.compute_full_qgt(psi, jacobian)
        
        # 2. Get Gauge Field (Berry Connection A_mu)
        A = self.get_berry_connection(psi, jacobian)
        
        # 3. Construct the KK Blocks
        # The scalar field squared (phi^2)
        phi2 = phi_scalar ** 2
        
        # Bottom Right: The scalar sector (phi^2)
        # Shape: (1, 1)
        G_scalar = jnp.array([[phi2]])
        
        # Top Right / Bottom Left: Mixing terms (phi^2 * A_mu)
        # Shape: (N, 1) and (1, N)
        G_cross = (phi2 * A).reshape(-1, 1)
        
        # Top Left: Modified Base Metric (g_uv + phi^2 A_u A_v)
        # This represents the "magnetic stress" added to the base geometry
        outer_A = jnp.outer(A, A)
        G_base = fisher + (phi2 * outer_A)
        
        # 4. Assemble the Unified Tensor
        # [[ Base+AA,  A ],
        #  [    A,    Phi ]]
        
        row1 = jnp.hstack([G_base, G_cross])
        row2 = jnp.hstack([G_cross.T, G_scalar])
        
        G_AB = jnp.vstack([row1, row2])
        
        return G_AB

# Usage Example
def demo_kk_uplift():
    # 1. Initialize Engine
    engine = CliffordEngine(seed=42)
    kk_builder = KaluzaKleinUplift(engine)
    
    # 2. Create a Dummy State (Light String)
    # Using the same dummy setup as your test_qgt.py
    psi_0 = engine.random_spinor(jax.random.PRNGKey(0))
    gen = engine.gammas[0] @ engine.gammas[1] 
    
    def get_psi(theta):
        angle = theta[0]
        U = jax.scipy.linalg.expm(-1j * angle * gen)
        return U @ psi_0
        
    theta_val = jnp.array([0.5])
    jacobian = jax.jacfwd(get_psi)(theta_val).reshape(4, 1)
    psi_val = get_psi(theta_val)
    
    # 3. Perform the Uplift
    # Lift the 1D parameter space to 2D (1D Space + 1D Prime Phase)
    G_5D = kk_builder.construct_5d_metric(psi_val, jacobian, phi_scalar=0.925) # Using IGBP crit as scalar size?
    
    print("4D Fisher Metric (Base):")
    fisher, _ = kk_builder.compute_full_qgt(psi_val, jacobian)
    print(fisher)
    
    print("\n5D Kaluza-Klein Metric (Uplifted):")
    print(G_5D)
    
    print("\nNote: The extra terms come from the Berry Connection 'gluing' the dimensions.")

if __name__ == "__main__":
    demo_kk_uplift()
