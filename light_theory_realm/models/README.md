# Models – Geometric Architectures

This directory contains **two geometric deep learning architectures** that demonstrate Light Theory's dual-track approach to modeling physical and information-theoretic systems.

Both models are **geometric in spirit**: they encode symmetry, manifold structure, and thermodynamic principles directly into their state spaces and interactions.

---

## Overview: Two Tracks

Light Theory Realm provides two complementary architectures:

| Architecture | Backend | Paradigm | Geometry |
|--------------|---------|----------|----------|
| **Classical U(1) Clock** | THRML/TSU | Thermodynamic sampling | U(1) phase manifold on lattice |
| **Quantum QHBM/VQT** | PennyLane + JAX | Variational quantum | Hybrid classical-quantum state manifold |

Both instantiate **geometric deep learning principles**:
- They treat states as living on **curved geometric manifolds**
- They use **group structure** (U(1), SU(3)) to define interactions
- They expose **geometric quantities** (metric, curvature, free energy) as primary observables

---

## 1. Classical: `U1ClockTHRMLModel`

**File:** [`classical_clock.py`](classical_clock.py)

### What it does

Implements a **discretized U(1) phase field** on an L×L lattice using thermodynamic sampling:

1. **State space:** U(1) phases θ ∈ [0, 2π) at each lattice site
2. **Discretization:** q clock states (e.g., q=8 for 8 directions on the circle)
3. **Encoding:** Each clock state → binary spins (n_bits = ⌈log₂(q)⌉ bits per site)
4. **Interactions:** Nearest-neighbor coupling approximating H ∼ -J Σ cos(θᵢ - θⱼ)
5. **Sampling:** THRML Ising EBM with block Gibbs sampling

### Geometric interpretation

- **Manifold:** Product of circles S¹ × S¹ × ... (one per lattice site)
- **Group structure:** U(1) at each site; interactions respect phase differences
- **Metric:** Induced by the clock model Hamiltonian (cosine interactions)
- **Sampling:** Thermodynamic state unit (TSU) explores the manifold via Gibbs dynamics

### Why this is GDL

This is a **classical geometric architecture** that builds a generative model over U(1)-valued fields using energy-based modeling and thermodynamics.

Key GDL principles:
- **Symmetry-aware:** U(1) group structure is explicit
- **Manifold-native:** States live on a circle, not ℝ
- **Geometric interactions:** cos(θᵢ - θⱼ) is the canonical metric on S¹ × S¹

### Usage example

```python
from light_theory_realm.models.classical_clock import U1ClockTHRMLModel
import jax

# Create 8×8 lattice with 8 clock states
model = U1ClockTHRMLModel(lattice_size=8, q=8, J=1.0)

# Sample configurations at temperature T=2.0
key = jax.random.PRNGKey(42)
samples = model.sample_states(
    T=2.0,
    n_warmup=100,
    n_samples=10,
    steps_per_sample=5,
    key=key
)

# Decode to phase angles
phases = model.decode_phases(samples)
print(f"Sampled phases shape: {phases.shape}")  # (10, 64)
```

### Parameters

- `lattice_size`: L (creates L×L lattice)
- `q`: Number of clock states (discretization resolution)
- `J`: Coupling strength (energy scale)

### Implementation notes

- **Binary encoding:** Maps q clock states to ⌈log₂(q)⌉ binary spins
- **Edge construction:** Connects all bit pairs between nearest-neighbor sites
- **Weight approximation:** Currently uniform J/n_bits²; could be improved with cosine fitting
- **THRML integration:** Uses `IsingEBM`, `Block`, `SamplingSchedule`, and `sample_with_observation`

---

## 2. Quantum: `LatentDistribution` + `QuantumLayer` + `vqt_loss`

**File:** [`quantum_qhbm.py`](quantum_qhbm.py)

### What it does

Implements a **hybrid classical-quantum architecture** (QHBM/VQT-style):

1. **Classical latent:** Factorized Bernoulli distribution over n_qubits bits
2. **Quantum circuit:** PennyLane ansatz with:
   - Basis state preparation |z⟩ from latent bits
   - n_layers of (RY, RZ rotations + CZ entangling ring)
   - Observable measurements (e.g., Hamiltonian expectation values)
3. **Loss:** Variational free energy F = βE - S
   - E = quantum expectation of Hamiltonian
   - S = classical entropy of latent distribution

### Geometric interpretation

- **Manifold:** Space of quantum states parameterized by:
  - Classical logits θ (latent distribution)
  - Quantum angles φ (circuit parameters)
- **Metric:** Quantum geometric tensor (QGT) of the variational family
- **Free energy:** Scalar functional F(θ, φ) over the manifold
- **Optimization:** Gradient descent on F → variational thermalization

### Why this is GDL

This is a **quantum geometric architecture** that builds a variational family over quantum states, combining a classical latent geometry with a quantum circuit, using free energy as the organizing scalar.

Key GDL principles:
- **Manifold of states:** Variational family defines a smooth manifold in Hilbert space
- **Information geometry:** Free energy F = βE - S balances energetic fit and entropic spread
- **Differentiable:** Full JAX + PennyLane integration for gradient-based optimization

### Usage example

```python
from light_theory_realm.models.quantum_qhbm import (
    LatentDistribution, QuantumLayer, vqt_loss
)
import jax
import pennylane as qml

# Setup
n_qubits = 4
n_layers = 2
beta = 1.0

latent_model = LatentDistribution(n_qubits)
quantum_layer = QuantumLayer(n_qubits, n_layers)

# Initialize parameters
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
theta = latent_model.init_params(k1)  # Classical logits
phi = quantum_layer.init_params(k2)   # Quantum angles

# Define Hamiltonian (example: Ising chain)
hamiltonian = qml.Hamiltonian(
    coeffs=[1.0] * (n_qubits - 1),
    observables=[qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n_qubits-1)]
)

# Compute free energy
k3 = jax.random.split(key)[0]
F = vqt_loss(theta, phi, latent_model, quantum_layer, hamiltonian, beta, k3)
print(f"Free energy: {F:.4f}")

# Gradient descent (example)
grad_fn = jax.grad(vqt_loss, argnums=(0, 1))
dtheta, dphi = grad_fn(theta, phi, latent_model, quantum_layer, hamiltonian, beta, k3)
```

### Components

#### `LatentDistribution`
- **Purpose:** Classical probabilistic backbone
- **Distribution:** Factorized Bernoulli (independent qubits)
- **Methods:**
  - `init_params(key)`: Initialize logits
  - `sample(params, key, n_samples)`: Draw Bernoulli samples
  - `entropy(params)`: Analytic Shannon entropy

#### `QuantumLayer`
- **Purpose:** QHBM-like quantum circuit
- **Device:** `default.qubit.jax` (PennyLane)
- **Circuit:**
  1. `BasisState(z)` preparation
  2. n_layers × (RY, RZ rotations + CZ ring)
  3. Observable measurements
- **Methods:**
  - `init_params(key)`: Initialize rotation angles
  - `forward(phi, z_samples, observables)`: Batched expectation values

#### `vqt_loss`
- **Purpose:** Variational Quantum Thermalizer objective
- **Formula:** F = β⟨E⟩ - S
  - ⟨E⟩ = mean energy over latent samples
  - S = entropy of latent distribution
- **Inputs:**
  - `theta`: Classical parameters
  - `phi`: Quantum parameters
  - `hamiltonian`: PennyLane Hamiltonian or list of observables
  - `beta`: Inverse temperature
  - `key`: PRNG key for sampling

### Implementation notes

- **JAX integration:** Full autodiff support for both classical and quantum parameters
- **Batching:** `vmap` over latent samples for efficient expectation computation
- **Hamiltonian flexibility:** Accepts `qml.Hamiltonian` or list of observables
- **Entangling structure:** CZ ring (nearest-neighbor on a circle)

---

## Connection to Light Theory Realm

These two models instantiate Light Theory's dual-track approach to geometric modeling:

### Three-layer architecture

1. **Foundational geometry engine**
   - Clifford algebra + QGT + Kaluza-Klein + Reeb
   - (See [`light_theory_realm/engine.py`](../engine.py), [`qgt.py`](../qgt.py))

2. **Physical toy world**
   - Pocket_U Lite: prime plaquettes → fermion masses
   - (See [`pocket_u_lite/`](../pocket_u_lite/))

3. **Model architectures**
   - Classical U(1) lattice model (geometry of phases, THRML sampling)
   - Quantum thermalizer / QHBM (geometry of quantum states via hybrid variational family)

### Bridge to geometric deep learning

Both models are **geometric architectures** that bridge theory and practical ML:

- **Classical side:** Circle-valued spins (U(1)), nearest-neighbor coupling → compact Lie group with natural metric
- **Quantum side:** Variational manifold of quantum states; free energy as scalar functional; QGT machinery applies directly

These implementations demonstrate that Light Theory Realm provides **explicit geometric models** on both classical and quantum tracks, not just theoretical frameworks.

---

## Comparison table

| Feature | Classical U(1) Clock | Quantum QHBM/VQT |
|---------|---------------------|------------------|
| **State space** | U(1) phases on lattice | Quantum states in Hilbert space |
| **Parameters** | Lattice size, q, J | n_qubits, n_layers, θ, φ |
| **Backend** | THRML/TSU (Ising EBM) | PennyLane + JAX |
| **Sampling** | Block Gibbs (thermodynamic) | Latent Bernoulli + circuit |
| **Geometry** | S¹ product manifold | Variational state manifold |
| **Loss/Objective** | Ising energy | Free energy F = βE - S |
| **Symmetry** | U(1) phase invariance | Depends on Hamiltonian |
| **Differentiable** | Via JAX (THRML) | Via JAX + PennyLane |
| **Use case** | Classical phase transitions, XY model | Quantum ground states, VQE-like |

---

## Future directions

### Classical track
- **Improved weight fitting:** Solve inverse problem for cos(θᵢ - θⱼ) → binary weights
- **Larger lattices:** Benchmark scaling to L=16, 32
- **Observables:** Add magnetization, correlation functions
- **Integration:** Connect to Pocket_U geometry (prime plaquettes → lattice configurations)

### Quantum track
- **Hamiltonian learning:** Train φ to approximate ground state of target H
- **Entanglement structure:** Explore different ansätze (hardware-efficient, QAOA-like)
- **QGT computation:** Explicitly compute quantum Fisher matrix for the variational family
- **Hybrid optimization:** Alternate classical (θ) and quantum (φ) updates

### Cross-track
- **Unified interface:** Common API for sampling, geometry computation, and loss
- **Comparative studies:** Classical vs quantum for same physical system (e.g., Ising model)
- **Visualization:** Plot manifold geometry, curvature landscapes for both models

---

## Questions?

See the main [ARCHITECTURE.md](../../ARCHITECTURE.md) for how these models fit into the overall Light Theory Realm design.

For GDL + ML + physics context, see [Foundations of Light Theory](../../Foundations/Foundations_of_Light_Theory.md).
