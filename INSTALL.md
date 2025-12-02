# Installation Guide

> **Note for v0.1.2:** This document describes the dual-track architecture. In the current release:
> - Core engine, Pocket_U Lite, and the models layer are shipped
> - Package uses flat layout: `light_theory_realm/` (no `src/` subpackage)
> - Track A (PennyLane / QHBM) is optional via extras: `pip install light-theory-realm[pennylane]`
> - Track B (THRML) is still installed from source (`thrml` repo), not via PyPI
> - `light_theory_realm/experiments/*` and `notebooks/*` contain examples and tutorials

## Requirements

- Python 3.10+
- JAX (with CPU or GPU support)
- NumPy

## Core Dependencies

```bash
pip install jax jaxlib numpy
```

## Optional Dependencies

The project supports **two complementary experimental tracks** with independent optional dependencies. Choose based on your use case.

### Track A: Quantum Coherent Model (PennyLane)
**Purpose:** Quantum circuit simulation with JAX backend for Wilson Loop observables and topological mass measurement.

**When to use:** If you want to run quantum experiments with PennyLane integration.

```bash
pip install pennylane
```

**Related files:**
- `light_theory_realm/models/quantum_qhbm.py` - Quantum Layer implementation
- `light_theory_realm/experiments/dual_track_demo.py` - Dual track comparison

### Track B: Classical Thermodynamic Model (THRML)
**Purpose:** Thermodynamic computing backend for classical clock models with Ising sampler.

**When to use:** If you want to run classical thermodynamic experiments with prime-derived parameters.

**Note:** THRML is not yet on PyPI. Install from source:

```bash
git clone https://github.com/extropic-ai/thrml
cd thrml
pip install -e .
```

**Related files:**
- `light_theory_realm/models/classical_clock.py` - Classical Clock Model implementation
- `light_theory_realm/experiments/dual_track_demo.py` - Dual track comparison

### Development Tools
```bash
pip install pytest pytest-cov
```

**When to use:** If you're running tests or contributing to the project.

## Installation

### Option A: Install from PyPI (Recommended for Users)
```bash
pip install light-theory-realm

# With Track A (Quantum) optional dependencies
pip install light-theory-realm[pennylane]

# With all optional dependencies (Note: THRML still requires manual source install)
pip install light-theory-realm[all]  # includes pytest and pennylane
```

**Import style:**
```python
from light_theory_realm import CliffordEngine, CliffordQGT
```

**Note:** Track B (THRML) must be installed separately from source (see Optional Dependencies section).

### Option B: Development Install (For Contributors)
Clone and install in editable mode:
```bash
git clone https://github.com/Pleroma-Works/Light_Theory_Realm
cd Light_Theory_Realm
pip install -e .

# Then install optional dependencies as needed
pip install -e ".[pennylane]"      # For Track A
pip install -e ".[all]"             # For dev + Track A
```

**Import style (same as Option A):**
```python
from light_theory_realm import CliffordEngine, CliffordQGT
```

### Option C: Direct Clone (No Install)
For running examples directly:
```bash
git clone https://github.com/Pleroma-Works/Light_Theory_Realm
cd Light_Theory_Realm
python examples/basic_qgt.py
```

**Note:** Examples use relative imports and will work without installation. Core engine works without optional deps.

## Verification

Test the installation:
```bash
python -c "from light_theory_realm import CliffordEngine; print('Core engine ready')"
python -c "from light_theory_realm import CliffordQGT; print('QGT module ready')"
python -c "from light_theory_realm import __version__; print(f'Version {__version__}')"
```

Or run the test script:
```bash
python test_install.py
```

## Verify New Features

Test Grade Projection and Wedge Product functionality:

```bash
# Test grade projection (decompose multivectors into geometric grades)
python test_grade_projection.py

# Test wedge product (outer product with Clifford algebra axioms)
python test_wedge_product.py

# Full end-to-end stack verification (all 4 layers)
python verify_full_stack.py
```

Expected output for each test:
- Grade projection: 4 tests PASS (scalar, vector, bivector, mixed decomposition)
- Wedge product: 8 tests PASS (null property, grade preservation, anticommutativity, associativity, scalar property, bivector products, mixed grades, consistency)
- Full stack: 9 tests PASS (algebraic core, geometry layer, theory layer, experiments)

## Quick Start

See `examples/` directory for standalone scripts:
- `basic_qgt.py` - Compute Fisher + Berry for a simple system
- `prime_plaquette.py` - Calculate mass from prime sequences
- `reeb_flow.py` - Demonstrate Dark Energy derivation
- `grade_projection_example.py` - Decompose multivectors into grades
- `wedge_product_example.py` - Compute wedge products with axiom verification
