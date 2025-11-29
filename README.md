# Pleroma Works — Research Overview
### Physics-Aware AI • Information Geometry • Scientific Discovery  
**Author:** Dimitry Jean-Noel II  
**Purpose:** High-level overview of current research programs, architectures, and experimental directions.

---

## 🌌 Overview

Pleroma Works is an independent research studio exploring **AI as a physical process**.

The core working hypothesis is that modern AI systems are not just “functions with real-valued weights and a loss,” but **physical, information-processing systems** whose behavior can and should be described using tools from:

- information geometry (Fisher metric, Quantum Geometric Tensor),  
- gauge theory (scalar and non-abelian fields),  
- topology (Berry curvature, Chern numbers), and  
- thermodynamics / statistical mechanics.

In this framing, objects like QGT, Berry curvature, Chern numbers, and scalar information gauges are treated as **physically meaningful observables** of the underlying process, whether that process is a thermodynamic sampler, a quantum-inspired model, or a large neural network.

Machine learning models play a dual role:
1. **Practical systems we want to understand and improve**, and  
2. **Controllable, high-dimensional systems** where field-theoretic and information-geometric ideas can be tested and falsified.

The goal is to build **open, reproducible tools** that turn these ideas into concrete experiments and diagnostics for scientific discovery.

---

# 1. Light Theory — An Information-Geometric Field Theory

**Light Theory** is a physics-first framework that describes how geometric and topological quantities structure the dynamics of complex systems.

### Core Objects

- **Quantum Geometric Tensor (QGT)**  
  Encodes both a Riemannian metric and Berry curvature on state space.

- **Information-Gauge Scalar (ρ\_IGBP)**  
  A scalar field intended to capture “information bounce protection” and stability properties, analogous to order parameters in phase transitions.

- **Topological Invariants**  
  Chern-like integers and Berry-phase integrals that characterize global structure and discrete transitions.

### Working Hypothesis

> For a wide class of complex systems, including AI models, thermodynamic samplers, and quantum-inspired architectures, these geometric and topological quantities are not just mathematical decorations but **physically meaningful observables** of how information and energy flow through the system.

### How AI Enters

AI systems (e.g., neural networks, variational models, samplers) are treated as:

- concrete, high-dimensional systems where QGT, Berry curvature, and ρ\_IGBP can be computed,
- testbeds where Light Theory’s predictions about phase structure, stability regimes, and topological transitions can be probed,
- targets for **geometry-aware design**, where architectures and optimizers are chosen to respect or exploit underlying field structure.

Prototype implementations include:

- QGT and Berry-curvature estimators in JAX,  
- κ-anisotropy and representational-entropy diagnostics,  
- geometry-aware update rules (e.g., TNGD-like flows).

---

# 2. Thermodynamic Sampler Validation

This module focuses on validating **thermodynamic and Ising-like samplers** as physical processes.

### Goals

- Move beyond scalar metrics (ESS, autocorrelation) to **field-aware diagnostics**.  
- Detect:
  - hidden phase transitions,  
  - mode collapse,  
  - unphysical trajectories,  
  - misaligned temperature schedules.

### Observables

- **Fisher information spectra** and spectral anisotropy (κ).  
- **Berry-curvature–style integrals** over control parameters (e.g., temperature, external fields).  
- **Approximate Chern-like invariants** along loops in control space.  

### Infrastructure

- GPU-accelerated experimental harnesses,  
- configuration → receipt → CSV logging → plotting pipelines,  
- reproducible protocols that can be run on classical GPUs or thermodynamic hardware (e.g., THRML) with minimal changes.

The result is a **physics-aware test suite** for samplers, intended to be portable to future non-von-Neumann compute substrates.

---

# 3. Prime-Gauge Search — Structured Scientific Discovery

**Prime-Gauge Search** is an experiment-as-code pipeline for exploring structured relationships between:

- prime-indexed plaquettes,  
- SU(3)-like gauge structures (Wilson loops),  
- and mass spectra (with Koide-style constraints).

### High-Level Idea

1. Construct prime-labeled plaquettes and map them to SU(3)-like matrices.  
2. Form Wilson loops and extract phases / eigenvalues.  
3. Define losses that encode physical desiderata (e.g., Koide proximity, mass ratios).  
4. Search over structures algorithmically and record candidate configurations.

This is not presented as a finished “theory of everything,” but as a **structured search space plus tooling** that invites further exploration, including:

- symbolic regression (e.g., PySR) on discovered patterns,  
- integration with LLM-based coding agents for automated heuristic generation,  
- cross-checking against known particle data.

---

# 4. Hybrid Quantum / Thermodynamic Architectures

This line of work develops **hybrid models** that blend:

- **Quantum-inspired components** (e.g., variational quantum circuits, U(1) or SU(2) symmetry),  
- **Classical thermodynamic components** (Boltzmann energies, Ising layers, denoising thermodynamic models).

The goal is to:

- realize systems where Light Theory’s geometric objects are naturally defined,  
- experiment with models that sit closer to physical hardware constraints,  
- explore whether certain gauge / topological structures lead to qualitatively different learning or inference behavior.

Architecturally, this includes:

- U(1) QHBM-like constructions with variational thermal layers,  
- coupled classical–quantum latent spaces,  
- experiments relating sampler dynamics, geometry, and model performance.

---

# 5. QHE-Style Benchmarks for Representations

Inspired by the **Quantum Hall Effect**, this benchmark family treats certain model parameter or representation spaces as synthetic manifolds and asks:

- Can we see **plateaus** and **integer-like jumps** in suitable response functions?  
- Can we define an effective “Hall impedance” \( Z_{\mathrm{eff}} \) from information-geometric quantities?  
- Do these structures correspond to qualitative changes in representation (e.g., robustness, generalization behavior)?

### Ingredients

- Loops or surfaces in parameter / hyperparameter space.  
- QGT- and curvature-based integrals along these loops.  
- Derived scalars that behave like topological response coefficients.

These experiments do not claim to make ML literally a quantum Hall system, but rather test whether **similar topological structures** appear in information-processing systems under Light Theory’s lens.

---

# 6. Research Stack Diagram

```text
Pleroma Works Research Stack
┌──────────────────────────────────────────────┐
│ 1. Physical & Mathematical Foundations      │
│    - Information geometry (QGT, Fisher)     │
│    - Gauge theory (SU(3)/SU(2)/U(1))        │
│    - Topology (Berry phase, Chern numbers)  │
├──────────────────────────────────────────────┤
│ 2. Core Engines                             │
│    - JAX (GPU numerical experiments)        │
│    - THRML / thermodynamic computing        │
│    - Symbolic tools (e.g., regression)      │
├──────────────────────────────────────────────┤
│ 3. Modules                                  │
│    - Light Theory (field framework)         │
│    - Thermodynamic Sampler Validation       │
│    - Prime-Gauge Search                     │
│    - Hybrid Quantum/Thermal Models          │
│    - QHE-Style Benchmarks                   │
├──────────────────────────────────────────────┤
│ 4. Applications                             │
│    - AI as physical process (diagnostics)   │
│    - Scientific discovery tooling           │
│    - Open research & reproducible infra     │
└──────────────────────────────────────────────┘
