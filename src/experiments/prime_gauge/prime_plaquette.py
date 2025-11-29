import jax
import jax.numpy as jnp
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.engine import CliffordEngine

class PrimeLattice:
    """
    Track B: Classical Prime-Gauge Lattice Logic.
    
    Manages the assignment of Prime Numbers to lattice sites and computes
    the effective interaction strengths (J) and gauge twists (A) for the
    Thermodynamic Clock Model.
    """
    def __init__(self, primes_map: dict):
        """
        Args:
            primes_map: Dictionary mapping site index (int) to Prime Number (int).
                        e.g., {0: 2, 1: 3, 2: 5, ...}
        """
        self.primes_map = primes_map
        
    def compute_edge_params(self, i: int, j: int):
        """
        Compute the Log-Energy Coupling (J) and Gauge Twist (A) for edge i->j.
        
        J_ij = 1 / |ln(p_j/p_i)|  (Information Stiffness)
        A_ij = (2pi/8) * ((p_i * p_j) % 8) (Generator Twist)
        """
        p_i = self.primes_map[i]
        p_j = self.primes_map[j]
        
        # 1. Log-Energy Distance (The Metric)
        # Avoid division by zero if p_i == p_j (self-loop)
        if p_i == p_j:
            dist = 0.0
            J = 100.0 # Infinite stiffness for self
        else:
            dist = jnp.abs(jnp.log(p_j / p_i))
            J = 1.0 / (dist + 1e-6)
            
        # 2. Generator Twist (The Gauge Field)
        # Mod-8 Rule from Light Theory
        gen_idx = (p_i * p_j) % 8
        # Map 0-7 to 0-2pi
        A = (2 * jnp.pi / 8) * gen_idx
        
        return J, A

    def compute_plaquette_energy(self, site_indices: list, phases: jnp.ndarray):
        """
        Compute the energy of a closed loop (plaquette) in the Clock Model.
        
        E = - sum J_ij * cos(theta_i - theta_j - A_ij)
        """
        energy = 0.0
        path = site_indices + [site_indices[0]] # Close the loop
        
        for k in range(len(site_indices)):
            u, v = path[k], path[k+1]
            J, A = self.compute_edge_params(u, v)
            
            theta_u = phases[u]
            theta_v = phases[v]
            
            # The interaction term with gauge twist
            energy -= J * jnp.cos(theta_u - theta_v - A)
            
        return energy

class PrimeWilsonLoop:
    """
    Track A: Quantum Prime Wilson Loop Logic.
    
    Constructs the unitary operators corresponding to traversing a sequence
    of primes in the Information Manifold.
    """
    def __init__(self, engine: CliffordEngine):
        self.engine = engine
        
    def _get_generator(self, p_i, p_j):
        """
        Map prime pair to a Clifford generator.
        Rule: (p_i * p_j) % 8 -> Specific Pauli/Gamma matrix.
        """
        idx = (p_i * p_j) % 8
        
        # Mapping strategy:
        # 0: Identity (No twist)
        # 1-3: Paulis (spatial rotation)
        # 4-6: Bivectors (boosts/mixed)
        # 7: Pseudoscalar (duality)
        
        if idx == 0:
            return jnp.eye(4, dtype=jnp.complex128)
        elif idx <= 3:
            # Use the spatial gammas (1, 2, 3)
            # gammas[0] is time, gammas[1-3] are space
            return self.engine.gammas[idx] 
        elif idx <= 6:
            # Use bivectors (01, 02, 03) -> boosts
            # keys are (0,1), (0,2), (0,3)
            return self.engine.get_bivector(0, idx-3)
        else: # idx == 7
            # Pseudoscalar I = G0 G1 G2 G3
            # Or just use a mix
            return self.engine.gammas[0] @ self.engine.gammas[1]

    def construct_loop_operator(self, primes: list):
        """
        Construct the Wilson Loop Operator W = U_12 U_23 ... U_N1
        
        U_ij = exp(i * ln(p_j/p_i) * Generator)
        """
        W = jnp.eye(4, dtype=jnp.complex128)
        path = primes + [primes[0]]
        
        for k in range(len(primes)):
            p_i = path[k]
            p_j = path[k+1]
            
            # 1. The "Distance" is the phase angle
            theta = jnp.log(p_j / p_i)
            
            # 2. The "Direction" is the generator
            G = self._get_generator(p_i, p_j)
            
            # 3. The Link Unitary
            # U = exp(i * theta * G)
            # Note: G might not be hermitian in all cases for Clifford, 
            # but for unitary evolution we usually want exp(-i H t).
            # Here we follow the user's ansatz: exp(i * ln * G)
            U = jax.scipy.linalg.expm(1j * theta * G)
            
            # Path ordered product: U_new @ W_old (or W_old @ U_new depending on convention)
            # Usually W = U_n ... U_2 U_1
            W = U @ W
            
        return W

    def measure_loop(self, psi: jnp.ndarray, primes: list):
        """
        Measure the expectation value <psi | W | psi>.
        The phase of this value is the "Topological Mass".
        """
        W = self.construct_loop_operator(primes)
        expectation = psi.conj().T @ W @ psi
        
        # Return scalar (complex)
        return expectation[0, 0]

def demo_prime_plaquette():
    print("--- Prime Plaquette Injection Demo ---\n")
    
    # 1. Setup
    engine = CliffordEngine(seed=137)
    
    # Define a "Muon Plaquette" (Hypothetical prime sequence)
    # 2 -> 3 -> 5 -> 7 -> 2
    plaquette_primes = [2, 3, 5, 7]
    print(f"Plaquette: {plaquette_primes}")
    
    # --- Track B: Classical Clock Model ---
    print("\n[Track B] Classical Clock Model Logic:")
    lattice = PrimeLattice(primes_map={0:2, 1:3, 2:5, 3:7})
    
    # Calculate params for first link 2->3
    J_01, A_01 = lattice.compute_edge_params(0, 1)
    print(f"Link 2->3: J={J_01:.4f} (Stiffness), A={A_01:.4f} (Twist)")
    
    # Calculate params for 3->5
    J_12, A_12 = lattice.compute_edge_params(1, 2)
    print(f"Link 3->5: J={J_12:.4f}, A={A_12:.4f}")
    
    # --- Track A: Quantum Wilson Loop ---
    print("\n[Track A] Quantum Wilson Loop Logic:")
    qc_loop = PrimeWilsonLoop(engine)
    
    # Create a random state |psi>
    psi = engine.random_spinor(jax.random.PRNGKey(42))
    
    # Measure the loop
    W_expect = qc_loop.measure_loop(psi, plaquette_primes)
    print(f"Wilson Loop <W>: {W_expect:.4f}")
    
    # The Mass is the Phase
    mass_phase = jnp.angle(W_expect)
    print(f"Topological Mass (Phase): {mass_phase:.4f} rad")
    
    print("\n--- Conclusion ---")
    print("Track B sees 'Energy Cost' (J, A).")
    print("Track A sees 'Phase Accumulation' (Mass).")
    print("Light Theory asserts these are dual descriptions of the same geometry.")

if __name__ == "__main__":
    demo_prime_plaquette()
