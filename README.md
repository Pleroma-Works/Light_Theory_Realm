# Light Theory: The QGT Engine

**A JAX-based Clifford Algebra engine for measuring the geometry of information.**

### The Core Idea
Modern AI and physical systems are not just functions; they are information-processing manifolds. To understand them, we need to measure their geometry.

**Light Theory** posits that the **Quantum Geometric Tensor (QGT)** is the fundamental "quanta" of this observation. It combines two distinct physical properties into a single object:

1.  **Fisher Information ($g_{uv}$):** The "Metric." How much does the system change when we move parameters? (The speed of learning).
2.  **Berry Curvature ($\Omega_{uv}$):** The "Twist." How much phase is accumulated during the move? (The topology of learning). 

This repository provides the minimal, high-precision engine required to compute these quantities using **Clifford Algebra**.

---

### Visual Overview

**Engine + Experiments**
```
CliffordEngine --> CliffordQGT --> Fisher metric + Berry curvature
                                      |                     |
                                      v                     v
                           prime_plaquette.py      reeb_flow.py
                                      |
                                      v
                              uplift.py (5D KK)
```

**Prime Plaquette Flow**
```
primes: 2 --> 3 --> 5 --> 7 --> 2
 edges:  J (log distance) , A (generator twist)

J, A  --> THRML clock energy      --> classical observables
J, A  --> Wilson loop exp(i*ln*p) --> mass phase
```

**Dual Track Architecture**
```
Track B (THRML)          Track A (PennyLane)
----------------         --------------------
Prime lattice J, A       Prime sequence
Ising/clock sampler      Quantum circuit (Ry/Rz, CZ)
Samples -> phases        Wilson loop expectation
Energy / vortices        Mass phase
            \            /
             \-- Compare / bridge --/
```

**Kaluza-Klein Uplift (sketch)**
```
Inputs: Fisher g_uv, Berry connection A_u, scalar phi

4D base block: g_uv
Cross terms:   phi^2 * A_u
Scalar:        phi^2
Assemble -> 5D metric:
[ g + phi^2 A A^T , phi^2 A ]
[ phi^2 A^T       ,  phi^2  ]
```

### Features

*   **Clifford Algebra $Cl(1,3)$:** A fully differentiable implementation of the Weyl representation using JAX.
*   **Exact QGT Estimation:** Computes the full Quantum Geometric Tensor, decomposing it automatically into Fisher (Real) and Berry (Imaginary) components.
*   **Kaluza-Klein Uplift:** Constructs 5D geometric manifolds from 4D information geometry, using the Berry Connection as a gauge field to "glue" dimensions together.
*   **Prime Plaquette Logic:** Bridges Number Theory with Physics via log-energy distances and generator twists.
*   **Reeb Flow Dynamics:** Derives Dark Energy as vacuum resonance density and Time as null space flow.
*   **U(1) Dual Track Architectures:** Classical (THRML) and Quantum (PennyLane) implementations for complementary experiments.
*   **JAX-Accelerated:** Built on `jax.jit` and `float64` precision for differentiable physics experiments.

---

### Quick Start

For detailed installation instructions and how to set up your development environment, please refer to the [INSTALL.md](INSTALL.md) guide. This includes information on installing JAX with CPU/GPU support, managing optional dependencies, and running tests.

**1. The Engine**
We use Clifford Algebra because it naturally handles the complex geometry of spinors and bivectors required for the QGT.

```python
import jax.numpy as jnp
from light_theory_realm import CliffordEngine, CliffordQGT

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

**3. Grade Projection: Decomposing Multivectors**

The Clifford algebra decomposes any multivector into its geometric grades (scalar, vector, bivector, trivector, pseudoscalar). Use grade projection to extract individual components.

```python
# Create a mixed multivector
v1 = engine.gammas[0]  # Vector (grade 1)
v2 = engine.gammas[1]  # Vector (grade 1)
bivector = engine.wedge_product(v1, v2)  # Bivector (grade 2)

scalar_part = 2.0 * jnp.eye(4, dtype=jnp.complex128)
mixed = scalar_part + v1 + bivector

# Extract individual grades
grade_0 = engine.project_grade(mixed, 0)  # Scalar component
grade_1 = engine.project_grade(mixed, 1)  # Vector component
grade_2 = engine.project_grade(mixed, 2)  # Bivector component

# Reconstruct the original
reconstructed = grade_0 + grade_1 + grade_2
print(f"Reconstruction error: {jnp.linalg.norm(reconstructed - mixed)}")
```

**4. Wedge Product: The Outer Product**

The wedge product (outer product) combines multivectors to produce higher-grade components. It respects Clifford algebra axioms: v ∧ v = 0, v1 ∧ v2 = -v2 ∧ v1.

```python
# Wedge product of two vectors produces a bivector
v1 = engine.gammas[0]
v2 = engine.gammas[1]
bivector = engine.wedge_product(v1, v2)

# Verify null property: v ∧ v = 0
self_wedge = engine.wedge_product(v1, v1)
print(f"v ∧ v norm: {jnp.linalg.norm(self_wedge)}")  # Should be 0

# Verify anticommutativity: v1 ∧ v2 = -v2 ∧ v1
reverse_wedge = engine.wedge_product(v2, v1)
print(f"v1 ∧ v2 + v2 ∧ v1 norm: {jnp.linalg.norm(bivector + reverse_wedge)}")  # Should be 0
```

-----

### Theoretical Background

The Quantum Geometric Tensor $T$ describes the geometry of quantum-like states $|̈\psi(\lambda)\rangle$. It is defined as:

$$T_{uv} = \langle \partial_u \psi | \partial_v \psi \rangle - \langle \partial_u \psi | \psi \rangle \langle \psi | \partial_v \psi \rangle$$

In our Clifford implementation:

  *   **Real Part (Symmetric):** The Fubini-Study metric (Fisher Information).
  *   **Imaginary Part (Anti-symmetric):** The Berry Curvature (Topological Phase).

By measuring these, we can detect phase transitions, mode collapse, and topological obstructions in learning dynamics.

-----

### Kaluza-Klein Uplift

The engine now supports **geometric unification** via Kaluza-Klein theory. You can uplift your 4D information manifold to a 5D geometry where the Berry Connection acts as a gauge field.

```python
from light_theory_realm.experiments.prime_gauge.uplift import KaluzaKleinUplift

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

### Prime Plaquettes: The Bridge

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
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeLattice, PrimeWilsonLoop

# Track B: Calculate Classical Parameters
lattice = PrimeLattice(primes_map={0:2, 1:3})
J, A = lattice.compute_edge_params(0, 1)

# Track A: Measure Quantum Mass
qc_loop = PrimeWilsonLoop(engine)
mass_phase = jnp.angle(qc_loop.measure_loop(psi, [2, 3, 5, 7]))
```

-----

### Dark Energy: The Reeb Flow

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
from light_theory_realm.experiments.prime_gauge.reeb_flow import ReebFlowDynamics

reeb = ReebFlowDynamics(engine)

# Calculate Resonance (Dark Energy)
xi = reeb.compute_resonance_density([17, 41, 73]) # Large primes -> High xi

# Calculate Time Direction (Reeb Vector)
R = reeb.compute_reeb_vector(fisher, berry)
```

-----

### U(1) Dual Track Architectures

We provide **two complementary implementations** of U(1) gauge theory, allowing experiments in both classical (thermodynamic) and quantum (coherent) regimes.

#### Track B: Classical Clock Model (THRML)
**File:** `src/models/classical_clock.py`

Uses **Extropic's THRML library** for thermodynamic computing:
- Binary encoding of q-state clock variables (q=8 or q=16)
- Ising EBM with cosine interactions: $E = -\sum J_{ij} \cos(\theta_i - \theta_j - A_{ij})$
- Thermodynamic sampling via TSU backend
- Prime-derived parameters: $J_{ij}$ (stiffness), $A_{ij}$ (gauge twist)

```python
from light_theory_realm.models.classical_clock import U1ClockTHRMLModel

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
from light_theory_realm.models.quantum_qhbm import QuantumLayer, LatentDistribution, vqt_loss

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

Both tracks successfully bridge Number Theory with Physics
```

-----

### Future Directions

This engine is the foundational layer for **Pleroma Works**. Upcoming modules will utilize this QGT engine to explore:

  *   **Prime Gauge Theory:** Investigating how the Berry Connection relates to prime number distributions and the IGBP (Inverse Generalized Bertrand's Postulate) framework.
  *   **Thermodynamic Sampling:** Using curvature to diagnose sampler health (e.g., measuring "information bounce").
  *   **Hybrid Architectures:** Geometry-aware optimization for neural networks.
  *   **The Koide Equation:** Investigating how Clifford geometric structures might naturally give rise to the specific mass relations observed in the Koide formula.

---

## Citing Light Theory Realm

If you use Light Theory Realm in your research, please cite:

```bibtex
@software{jeannoel2025lighttheory,
  author = {Jean-Noel II, Dimitry},
  title = {Light Theory Realm: A Geometry-First Framework for Quantum Machine Learning},
  year = {2025},
  url = {https://github.com/Pleroma-Works/Light_Theory_Realm},
  note = {Version 0.1.0}
}
```

### Acknowledgments

This work builds upon and is inspired by:

**TensorFlow Quantum** - For pioneering the hybrid quantum-classical ML framework:
```bibtex
@article{broughton2021tensorflowquantum,
  title={TensorFlow Quantum: A Software Framework for Quantum Machine Learning},
  author={Broughton, Michael and Verdon, Guillaume and McCourt, Trevor and Martinez, Antonio J. and Yoo, Jae Hyeon and Isakov, Sergei V. and Massey, Philip and Halavati, Ramin and Niu, Murphy Yuezhen and Zlokapa, Alexander and Peters, Evan and Lockwood, Owen and Skolik, Andrea and Jerbi, Sofiene and Dunjko, Vedran and Leib, Martin and Streif, Michael and Von Dollen, David and Chen, Hongxiang and Cao, Shuxiang and Wiersema, Roeland and Huang, Hsin-Yuan and McClean, Jarrod R. and Babbush, Ryan and Boixo, Sergio and Bacon, Dave and Ho, Alan K. and Neven, Hartmut and Mohseni, Masoud},
  year={2021},
  eprint={2003.02989},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  doi={10.48550/arXiv.2003.02989},
  url={https://arxiv.org/abs/2003.02989}
}
```

**PennyLane** - For the differentiable quantum programming framework used in Track A:
```bibtex
@article{bergholm2018pennylane,
  title={PennyLane: Automatic differentiation of hybrid quantum-classical computations},
  author={Bergholm, Ville and Izaac, Josh and Schuld, Maria and Gogolin, Christian and Ahmed, Shahnawaz and Ajith, Vishnu and Alam, M. Sohaib and Alonso-Linaje, Guillermo and AkashNarayanan, B. and Asadi, Ali and others},
  journal={arXiv preprint arXiv:1811.04968},
  year={2018},
  url={https://arxiv.org/abs/1811.04968}
}
```

**Extropic's THRML** - For the thermodynamic computing backend used in Track B:
```bibtex
@misc{jelinčič2025efficientprobabilistichardwarearchitecture,
  title={An efficient probabilistic hardware architecture for diffusion-like models}, 
  author={Andraž Jelinčič and Owen Lockwood and Akhil Garlapati and Guillaume Verdon and Trevor McCourt},
  year={2025},
  eprint={2510.23972},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.23972}, 
}
```

---

## License

This project is licensed under the Apache License 2.0 – see the [LICENSE](LICENSE) file for details.
