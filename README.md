# Light Theory: The QGT Engine

**A JAX-based Clifford Algebra engine for measuring the geometry of information.**

### 🌌 The Core Idea
Modern AI and physical systems are not just functions; they are information-processing manifolds. To understand them, we need to measure their geometry.

**Light Theory** posits that the **Quantum Geometric Tensor (QGT)** is the fundamental "quanta" of this observation. It combines two distinct physical properties into a single object:

1.  **Fisher Information ($g_{uv}$):** The "Metric." How much does the system change when we move parameters? (The speed of learning).
2.  **Berry Curvature ($\Omega_{uv}$):** The "Twist." How much phase is accumulated during the move? (The topology of learning). 

This repository provides the minimal, high-precision engine required to compute these quantities using **Clifford Algebra**.

---

### ⚡ Features

*   **Clifford Algebra $Cl(1,3)$:** A fully differentiable implementation of the Weyl representation using JAX.
*   **Exact QGT Estimation:** Computes the full Quantum Geometric Tensor, decomposing it automatically into Fisher (Real) and Berry (Imaginary) components.
*   **Kaluza-Klein Uplift:** Constructs 5D geometric manifolds from 4D information geometry, using the Berry Connection as a gauge field to "glue" dimensions together.
*   **Prime Plaquette Logic:** Bridges Number Theory with Physics via log-energy distances and generator twists.
*   **Reeb Flow Dynamics:** Derives Dark Energy as vacuum resonance density and Time as null space flow.
*   **U(1) Dual Track Architectures:** Classical (THRML) and Quantum (PennyLane) implementations for complementary experiments.
*   **JAX-Accelerated:** Built on `jax.jit` and `float64` precision for differentiable physics experiments.

---

### 🚀 Quick Start

**1. The Engine**
We use Clifford Algebra because it naturally handles the complex geometry of spinors and bivectors required for the QGT.

```python
import jax.numpy as jnp
from src.engine import CliffordEngine
from src.qgt import CliffordQGT

# Initialize the Cl(1,3) Physics Engine
engine = CliffordEngine(seed=42)
qgt = CliffordQGT(engine)

# Define a parameterized state trajectory (e.g., a rotation)
def get_psi(theta):
    # ... your model logic here ...
    return psi
````

**2. Measuring the Geometry**
Feed your model's Jacobian into the QGT estimator to see the hidden geometry.

```python
# Compute the fundamental "Quanta" of the system
fisher, berry = qgt.compute_full_qgt(psi, jacobian)

print(f"Metric (Information Distance): {fisher}")
print(f"Curvature (Topological Twist): {berry}")
```

-----

### 📐 Theoretical Background

The Quantum Geometric Tensor $T$ describes the geometry of quantum-like states $|̈\psi(\lambda)\rangle$. It is defined as:

$$T_{uv} = \langle \partial_u \psi | \partial_v \psi \rangle - \langle \partial_u \psi | \psi \rangle \langle \psi | \partial_v \psi \rangle$$

In our Clifford implementation:

  *   **Real Part (Symmetric):** The Fubini-Study metric (Fisher Information).
  *   **Imaginary Part (Anti-symmetric):** The Berry Curvature (Topological Phase).

By measuring these, we can detect phase transitions, mode collapse, and topological obstructions in learning dynamics.

-----

### 🌀 Kaluza-Klein Uplift

The engine now supports **geometric unification** via Kaluza-Klein theory. You can uplift your 4D information manifold to a 5D geometry where the Berry Connection acts as a gauge field.

```python
from src.experiments.prime_gauge.uplift import KaluzaKleinUplift

# Initialize the KK builder
kk_builder = KaluzaKleinUplift(engine)

# Construct the 5D metric
# phi_scalar controls the "size" of the extra dimension
G_5D = kk_builder.construct_5d_metric(psi, jacobian, phi_scalar=0.925)

print("5D Kaluza-Klein Metric:")
print(G_5D)
```

**The 5D Metric Structure:**
$$
\hat{G}_{AB} = 
\begin{pmatrix} 
g_{\mu\nu} + \phi^2 A_\mu A_\nu & \phi^2 A_\mu \\ 
\phi^2 A_\nu & \phi^2 
\end{pmatrix}
$$

Where:
- $g_{\mu\nu}$: Fisher Information Metric (base geometry)
- $A_\mu$: Berry Connection (gauge field)
- $\phi$: Dilaton/Radion field (extra dimension size)

This unified tensor contains both the information geometry and topological forces in a single object, enabling:
- **Singularity Analysis:** Investigate where the geometry becomes degenerate
- **Eigenvalue Spectra:** Study the characteristic scales of the unified field
- **Prime Gauge Theory:** Connect geometric properties to number-theoretic structures

-----

### 🔢 Prime Plaquettes: The Bridge

The **Prime Plaquette** logic acts as the input specification that configures both the Classical and Quantum engines, bridging abstract Number Theory with Physics.

It defines the geometry of the system using sequences of Prime Numbers (e.g., $2 \to 3 \to 5 \to 7$).

#### 1. Track B: Classical Clock Model (Thermodynamic)
In the classical regime, primes determine the **Interaction Stiffness** and **Gauge Twist**:

*   **Stiffness ($J_{ij}$):** Derived from the Log-Energy distance.
    $$J_{ij} \propto \frac{1}{|\ln(p_j/p_i)|}$$
*   **Twist ($A_{ij}$):** Derived from the generator mismatch (Mod-8 rule).
    $$E = - \sum J_{ij} \cos(\theta_i - \theta_j - A_{ij})$$

#### 2. Track A: Quantum Coherent Model (JAX)
In the quantum regime, primes define the **Wilson Loop Operator**:

*   **The Operator ($W$):** A product of unitary link gates along the prime sequence.
    $$W = U_{12} U_{23} \dots U_{N1}$$
*   **The Observable:** The expectation value $\langle \psi | W | \psi \rangle$.
*   **Topological Mass:** The phase of this expectation value corresponds to the mass of the particle defined by the plaquette.

```python
from src.experiments.prime_gauge.prime_plaquette import PrimeLattice, PrimeWilsonLoop

# Track B: Calculate Classical Parameters
lattice = PrimeLattice(primes_map={0:2, 1:3})
J, A = lattice.compute_edge_params(0, 1)

# Track A: Measure Quantum Mass
qc_loop = PrimeWilsonLoop(engine)
mass_phase = jnp.angle(qc_loop.measure_loop(psi, [2, 3, 5, 7]))
```

-----

### 🌌 Dark Energy: The Reeb Flow

The final piece of the theoretical puzzle is the derivation of **Dark Energy** ($\xi$) and **Time** ($R$) from algebraic resonance.

#### 1. Resonance Density ($\xi$)
Dark Energy is not a mysterious fluid; it is the **Resonance Density of the Vacuum**.
*   **Matter (Small Primes):** Generators clash (non-commuting) $\to$ Low $\xi$ $\to$ Curvature dominates.
*   **Vacuum (Large Primes):** Generators commute $\to$ High $\xi \approx 1$ $\to$ Drift dominates.

$$\xi = \frac{1}{1 + \sum \|[G_i, G_j]\|}$$

#### 2. The Reeb Vector ($R$)
Time does not flow "forward"; it flows through the **Null Space** of the Berry Curvature (the path of least topological resistance).

$$i_R \Omega = 0$$

#### 3. Unified Dynamics
The evolution of the universe is governed by a Time-Natural Gradient Descent (TNGD) with Reeb Drift:

$$d\theta = -\eta g^{-1} \nabla L + \xi R$$

```python
from src.experiments.prime_gauge.reeb_flow import ReebFlowDynamics

reeb = ReebFlowDynamics(engine)

# Calculate Resonance (Dark Energy)
xi = reeb.compute_resonance_density([17, 41, 73]) # Large primes -> High xi

# Calculate Time Direction (Reeb Vector)
R = reeb.compute_reeb_vector(fisher, berry)
```

-----

### 🔬 U(1) Dual Track Architectures

We provide **two complementary implementations** of U(1) gauge theory, allowing experiments in both classical (thermodynamic) and quantum (coherent) regimes.

#### Track B: Classical Clock Model (THRML)
**File:** `src/models/classical_clock.py`

Uses **Extropic's THRML library** for thermodynamic computing:
- Binary encoding of q-state clock variables (q=8 or q=16)
- Ising EBM with cosine interactions: $E = -\sum J_{ij} \cos(\theta_i - \theta_j - A_{ij})$
- Thermodynamic sampling via TSU backend
- Prime-derived parameters: $J_{ij}$ (stiffness), $A_{ij}$ (gauge twist)

```python
from src.models.classical_clock import U1ClockTHRMLModel

model = U1ClockTHRMLModel(lattice_size=2, q=8, J=1.0)
samples = model.sample_states(T=2.0, n_warmup=100, n_samples=100, 
                               steps_per_sample=10, key=key)
phases = model.decode_phases(samples)
```

**Result:** Vortex formation energy cost in the classical thermodynamic limit.

#### Track A: Quantum QHBM (PennyLane)
**File:** `src/models/quantum_qhbm.py`

Uses **PennyLane** with JAX backend for quantum simulation:
- Latent Distribution: Factorized Bernoulli (classical entropy)
- Quantum Circuit: Ry/Rz rotations + CZ entanglers (quantum correlations)
- VQT Loss: $F = \beta E - TS$ (Variational Quantum Thermalizer)
- Wilson Loop observables for topological mass measurement

```python
from src.models.quantum_qhbm import QuantumLayer, LatentDistribution, vqt_loss

ql = QuantumLayer(n_qubits=4, n_layers=2)
ld = LatentDistribution(n_qubits=4)

loss = vqt_loss(theta, phi, ld, ql, hamiltonian, beta=1.0, key=key)
```

**Result:** Topological mass as the phase of the Wilson Loop expectation value.

#### Integration Demo
**File:** `src/experiments/dual_track_demo.py`

Demonstrates both tracks solving the same Prime Plaquette problem:

```bash
python src/experiments/dual_track_demo.py
```

**Output:**
```
Target Plaquette: [2, 3, 5, 7]

Track B: Edge 2→3: J=2.4663, A=4.7124 rad
Track A: <W> = 1.0595+0.0927j, Mass = 0.0872 rad

✓ Both tracks successfully bridge Number Theory with Physics
```

-----

### 🔮 Future Directions

This engine is the foundational layer for **Pleroma Works**. Upcoming modules will utilize this QGT engine to explore:

  *   **Prime Gauge Theory:** Investigating how the Berry Connection relates to prime number distributions and the IGBP (Inverse Generalized Bertrand's Postulate) framework.
  *   **Thermodynamic Sampling:** Using curvature to diagnose sampler health (e.g., measuring "information bounce").
  *   **Hybrid Architectures:** Geometry-aware optimization for neural networks.
  *   **The Koide Equation:** Investigating how Clifford geometric structures might naturally give rise to the specific mass relations observed in the Koide formula.