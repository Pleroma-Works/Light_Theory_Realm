import jax
import jax.numpy as jnp
import pennylane as qml
from typing import List, Tuple

class LatentDistribution:
    """Classical probabilistic model for latent variables."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        
    def init_params(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        # e.g., logits for independent Bernoulli
        return jax.random.normal(key, (self.n_qubits,))
        
    def sample(self, params: jnp.ndarray, key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        # Bernoulli sampling
        logits = params
        probs = jax.nn.sigmoid(logits)
        return jax.random.bernoulli(key, probs, shape=(n_samples, self.n_qubits)).astype(jnp.float32)
        
    def entropy(self, params: jnp.ndarray) -> float:
        # Analytic entropy for factorized distribution
        probs = jax.nn.sigmoid(params)
        return -jnp.sum(probs * jnp.log(probs + 1e-10) + 
                        (1-probs) * jnp.log(1-probs + 1e-10))

class QuantumLayer:
    """
    JAX-based quantum layer using PennyLane.
    Implements the QHBM circuit ansatz.
    """
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit.jax", wires=n_qubits)
        
        # Define the QNode
        @qml.qnode(self.dev, interface="jax")
        def circuit(phi, z, observables):
            # 1. Prepare latent state |z>
            qml.BasisState(z, wires=range(self.n_qubits))
            
            # 2. Variational Circuit
            for l in range(self.n_layers):
                # Rotations
                for i in range(self.n_qubits):
                    qml.RY(phi[l, i, 0], wires=i)
                    qml.RZ(phi[l, i, 1], wires=i)
                
                # Entangling layer (CZ ring)
                for i in range(self.n_qubits):
                    qml.CZ(wires=[i, (i+1)%self.n_qubits])
            
            # 3. Measure
            return [qml.expval(o) for o in observables]
            
        self.qnode = circuit
        
    def init_params(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        # (layers, qubits, 2 params per qubit)
        return jax.random.uniform(key, (self.n_layers, self.n_qubits, 2), minval=0, maxval=2*jnp.pi)
        
    def forward(self, phi: jnp.ndarray, z_samples: jnp.ndarray, observables: List[qml.operation.Observable]) -> jnp.ndarray:
        """
        Compute expectation values for a batch of latent samples.
        """
        # vmap over the batch dimension (z_samples)
        # phi and observables are constant for the batch
        vmapped_circuit = jax.vmap(self.qnode, in_axes=(None, 0, None))
        
        return vmapped_circuit(phi, z_samples, observables)

def vqt_loss(theta, phi, latent_model, quantum_layer, hamiltonian, beta, key):
    """
    Variational Quantum Thermalizer Loss: F = E - TS
    """
    # 1. Sample Latents
    z_samples = latent_model.sample(theta, key, n_samples=128)
    
    # 2. Quantum Expectation (Energy)
    # Hamiltonian is a list of observables (e.g., Pauli terms)
    # We sum their expectations
    # Note: PennyLane handles Hamiltonian objects directly in expval, 
    # but for flexibility we might pass a list of terms.
    # Here assuming hamiltonian is a single PennyLane Hamiltonian object or list of observables.
    
    if isinstance(hamiltonian, qml.Hamiltonian):
        observables = [hamiltonian]
    else:
        observables = hamiltonian
        
    expectations = quantum_layer.forward(phi, z_samples, observables)
    # expectations shape: (n_samples, n_observables)
    
    # Sum over observables to get total energy per sample
    energies = jnp.sum(jnp.array(expectations), axis=1)
    E_mean = jnp.mean(energies)
    
    # 3. Classical Entropy
    S = latent_model.entropy(theta)
    
    # 4. Free Energy
    F = beta * E_mean - S
    return F
