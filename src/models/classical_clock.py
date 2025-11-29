import jax
import jax.numpy as jnp
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.block_sampling import Block, SamplingSchedule, sample_with_observation
from thrml.pgm import AbstractNode
from typing import List, Tuple

class U1ClockTHRMLModel:
    """
    U(1)-compatible THRML model using clock model encoding.
    
    Discretizes U(1) phases into q states, encodes as binary spins,
    and uses THRML/TSU backend for thermodynamic sampling.
    """
    
    def __init__(self, lattice_size: int, q: int = 8, J: float = 1.0):
        self.L = lattice_size
        self.N = lattice_size * lattice_size  # Number of sites
        self.q = q  # Number of clock states
        self.n_bits = int(jnp.ceil(jnp.log2(q)))  # Bits per site
        self.total_spins = self.N * self.n_bits
        self.J = J
        
        # Create THRML nodes (one per binary spin)
        # In THRML, we need to create AbstractNode instances
        # For simplicity, we'll create a simple node class
        class BinarySpinNode(AbstractNode):
            pass
        
        self.nodes = [BinarySpinNode() for _ in range(self.total_spins)]
        
        # Build edges (nearest-neighbor on lattice)
        self.edges = self._build_clock_edges()
        
        # Compute interaction weights
        self.weights = self._compute_clock_weights()
    
    def _build_clock_edges(self) -> List[Tuple[AbstractNode, AbstractNode]]:
        """Build edges for clock model on 2D lattice."""
        edges = []
        
        # For each lattice site pair (i,j)
        for i in range(self.L):
            for j in range(self.L):
                site_idx = i * self.L + j
                
                # Right neighbor
                right_idx = i * self.L + (j + 1) % self.L
                # Add edges between all binary spins of these sites
                for b in range(self.n_bits):
                    for b2 in range(self.n_bits):
                        node_i = self.nodes[site_idx * self.n_bits + b]
                        node_j = self.nodes[right_idx * self.n_bits + b2]
                        edges.append((node_i, node_j))
                
                # Down neighbor (similar)
                down_idx = ((i + 1) % self.L) * self.L + j
                for b in range(self.n_bits):
                    for b2 in range(self.n_bits):
                        node_i = self.nodes[site_idx * self.n_bits + b]
                        node_j = self.nodes[down_idx * self.n_bits + b2]
                        edges.append((node_i, node_j))
        
        return edges
    
    def _compute_clock_weights(self) -> jnp.ndarray:
        """
        Compute effective Ising weights that approximate clock model.
        
        This is the key: map cos(theta_i - theta_j) to binary spin interactions.
        """
        # Simplified: uniform coupling distributed across bits
        # In a full implementation, we would solve for weights that best approximate
        # the cosine interaction table.
        weights = []
        for _ in self.edges:
            # Distribute J across the n_bits^2 connections per edge
            weights.append(self.J / (self.n_bits ** 2))
        
        return jnp.array(weights)
    
    def sample_states(self, T: float, n_warmup: int, n_samples: int, steps_per_sample: int, key: jax.random.PRNGKey):
        """Sample clock configurations using THRML/TSU backend."""
        beta = jnp.array(1.0 / T)
        
        # Create THRML Ising model
        biases = jnp.zeros(self.total_spins)
        model = IsingEBM(self.nodes, self.edges, biases, self.weights, beta)
        
        # Block Gibbs sampling (one block per lattice site)
        free_blocks = [
            Block(self.nodes[i*self.n_bits:(i+1)*self.n_bits]) 
            for i in range(self.N)
        ]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        # Sample
        k_init, k_samp = jax.random.split(key)
        init_state = hinton_init(k_init, model, tuple(free_blocks), ())
        schedule = SamplingSchedule(
            n_warmup=n_warmup, 
            n_samples=n_samples, 
            steps_per_sample=steps_per_sample
        )
        
        # Use sample_with_observation (THRML's actual API)
        # We don't need observation for this simple case
        def identity_transform(state, _):
            return state
        
        from thrml.observers import NoOpObserver
        observer = NoOpObserver()
        init_mem = observer.init()
        
        _, final_state = sample_with_observation(
            k_samp, program, schedule, init_state, [], init_mem, observer
        )
        
        # Convert final state to array
        # final_state is a list of arrays, concatenate them
        return jnp.concatenate([s[-1] for s in final_state], axis=-1)
    
    def decode_phases(self, binary_configs: jnp.ndarray) -> jnp.ndarray:
        """Convert binary spin configs to phase angles."""
        n_samples = binary_configs.shape[0]
        phases = jnp.zeros((n_samples, self.N))
        
        for sample_idx in range(n_samples):
            for site_idx in range(self.N):
                # Extract n_bits binary spins for this site
                # Map -1/1 spins to 0/1 bits
                spins = (binary_configs[sample_idx, site_idx*self.n_bits:(site_idx+1)*self.n_bits] + 1) / 2
                
                # Convert to decimal
                k = sum(spins[b] * (2 ** (self.n_bits - 1 - b)) for b in range(self.n_bits))
                
                # Convert to phase
                phases = phases.at[sample_idx, site_idx].set(2 * jnp.pi * k / self.q)
        
        return phases
