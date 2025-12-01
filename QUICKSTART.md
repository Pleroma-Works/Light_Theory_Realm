# Quick Start Guide

Get up and running with Light Theory Realm in 5 minutes.

## Installation

```bash
pip install light-theory-realm
```

For GPU support:
```bash
pip install light-theory-realm[gpu]
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

---

## Command-Line Interface (CLI)

Once installed, use the `light-realm` command to explore without writing code:

```bash
# Show all Standard Model fermion masses
light-realm sm-table

# Analyze Koide relation for PDG leptons
light-realm koide 0.511 105.66 1776.86

# Predict tau from electron and muon masses
light-realm koide-predict 0.511 105.66

# Koide analysis using Pocket_U Lite predictions
light-realm koide-from-pocket
```

For help:
```bash
light-realm --help
light-realm sm-table --help
```

---

## Your First QGT Calculation

```python
import jax.numpy as jnp
from light_theory_realm import CliffordEngine, CliffordQGT

# Initialize engine
engine = CliffordEngine(seed=42)
qgt = CliffordQGT(engine)

# Create a state and its derivative
psi = engine.random_spinor(engine.key)
d_psi = engine.random_spinor(engine.key)

# Compute geometry
fisher, berry = qgt.compute_full_qgt(psi, d_psi.reshape(-1, 1))

print(f"Fisher metric (information distance): {fisher[0,0]:.6f}")
print(f"Berry curvature (topological twist): {berry[0,0]:.6f}")
```

**What this does**: Computes the Quantum Geometric Tensor, decomposing it into Fisher (metric) and Berry (curvature) components.

---

## Standard Model Masses

```python
from light_theory_realm import print_sm_table, get_particle_profile

# Print all 9 fermion masses
print_sm_table()

# Get detailed electron profile
e = get_particle_profile("e")
print(f"Electron: {e['m_phys_MeV']:.3f} MeV")
print(f"Screening: {e['screening_pct']:.1f}%")
print(f"Dark energy ξ: {e['xi']:.4f}")
```

**What this does**: Shows Standard Model mass predictions from prime plaquettes with ~2.3% average error.

---

## Koide Equation

```python
from light_theory_realm import koide_ratio, predict_third_mass, koide_angle

# Check Koide relation
Q = koide_ratio(0.511, 105.66, 1776.86)
print(f"Q = {Q:.6f}")  # Should be ≈ 0.666667

# Predict tau from e and μ
m_tau, Q_check = predict_third_mass(0.511, 105.66)
print(f"Predicted tau: {m_tau:.2f} MeV")

# Geometric angle
result = koide_angle(0.511, 105.66, 1776.86)
print(f"Angle: {result['theta_deg']:.2f}° (45° expected)")
```

**What this does**: Analyzes the mysterious Koide formula showing geometric relationships between lepton masses.

---

## Prime Plaquettes

```python
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeWilsonLoop

# Create Wilson loop calculator
wilson = PrimeWilsonLoop(engine)

# Compute Wilson loop for electron plaquette
plaquette = [2, 3, 5, 7]
psi = engine.random_spinor(engine.key)
W_expect = wilson.measure_loop(psi, plaquette)

print(f"Wilson loop: {W_expect:.4f}")
print(f"Topological mass (phase): {jnp.angle(W_expect):.4f} rad")
```

**What this does**: Demonstrates how prime number sequences generate topological mass through Wilson loops.

---

## Next Steps

- **Examples**: See [`examples/`](examples/) for complete demos:
  - `pocket_u_mass_table.py` - Standard Model mass spectrum
  - `koide_analysis.py` - Koide equation analysis
  - `pocket_u_geometry_showcase.py` - Geometric fingerprints

- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md) for complete function documentation

- **Full Documentation**: See [README.md](README.md) for theoretical background and detailed explanations

---

## Common Use Cases

### 1. Measuring Learning Geometry
Use QGT to measure how fast your model learns (Fisher) and how much topological structure it has (Berry).

### 2. Standard Model Physics
Explore how prime numbers encode particle masses through geometric Wilson loops.

### 3. Koide Analysis
Investigate geometric relationships between particle masses and test the 45° angle prediction.

### 4. Kaluza-Klein Uplift
Construct 5D geometric manifolds from 4D information geometry using Berry connection as gauge field.

---

## Troubleshooting

**Import Error**: Make sure you've installed the package:
```bash
pip install -e .  # If working from source
```

**JAX GPU Issues**: See [INSTALL.md](INSTALL.md) for GPU-specific installation instructions.

**Example Scripts Not Found**: Make sure you're running from the repository root directory.

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Pleroma-Works/Light_Theory_Realm/issues)
- **Documentation**: [README.md](README.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
