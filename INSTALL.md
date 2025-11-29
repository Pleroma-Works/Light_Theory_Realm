# Installation Guide

## Requirements

- Python 3.9+
- JAX (with CPU or GPU support)
- NumPy

## Core Dependencies

```bash
pip install jax jaxlib numpy
```

## Optional Dependencies

### For Track B (Classical THRML)
```bash
# THRML is installed from local repo at:
# \\wsl.localhost\Ubuntu\home\jdimi\projects\to_observe_proof\thrml_repo
```

### For Track A (Quantum PennyLane)
```bash
pip install pennylane
```

## Installation

### Option 1: Direct Use (Recommended)
Clone and use directly:
```bash
cd Light_Theory_Realm
python src/experiments/dual_track_demo.py
```

### Option 2: Development Install
For development with editable install:
```bash
pip install -e .
```

## Verification

Test the installation:
```bash
python -c "from src.engine import CliffordEngine; print('✓ Core engine ready')"
python -c "from src.qgt import CliffordQGT; print('✓ QGT module ready')"
```

## Quick Start

See `examples/` directory for standalone scripts:
- `basic_qgt.py` - Compute Fisher + Berry for a simple system
- `prime_plaquette.py` - Calculate mass from prime sequences
- `reeb_flow.py` - Demonstrate Dark Energy derivation
