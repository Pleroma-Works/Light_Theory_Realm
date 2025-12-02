# Changelog

All notable changes to Light Theory Realm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-12-02

### Added
- **Tutorial notebook series (10 notebooks)** under `notebooks/`:
  - 01_intro_hello_light_theory_realm.ipynb
  - 02_clifford_algebra_101.ipynb
  - 03_basic_qgt_single_parameter.ipynb
  - 04_standard_model_mass_explorer.ipynb
  - 05_koide_playground.ipynb
  - 06_geometric_fingerprints_particles.ipynb
  - 07_prime_plaquettes_topological_mass.ipynb
  - 08_dark_energy_reeb_flow.ipynb
  - 09_u1_clock_playground.ipynb
  - 10_quantum_qhbm_vqt_tutorial.ipynb
- **Notebook-learning docs**:
  - “📓 Interactive Notebooks” section in `README.md`
  - “📓 Interactive Learning Path” section in `GETTING_STARTED.md`

### Changed
- **Installation docs** updated to reflect current dual-track architecture (core engine, Pocket_U Lite, models, and optional THRML / PennyLane backends).
- **Executive summary** updated to mark v0.1.2 as the current release and highlight the interactive notebook series.

### Fixed
- Improved robustness of Reeb-flow / geometry usage in notebooks by handling `primes` as tuples when passed into `compute_reeb_diagnostics`.

## [0.1.1] - 2025-12-02

### Added
- **Models layer documentation** - Comprehensive README for classical_clock.py and quantum_qhbm.py
- **Layer 4: Models** section in ARCHITECTURE.md
- **Models reference** in main README.md

### Changed
- **Architecture documentation** - Added explicit models layer with classical and quantum models
- **Main documentation** - Updated with models section and links

### Removed
- **Old comparison document** - Removed `docs/light_theory_comparison.md` (superseded by current architecture)

## [0.1.0] - 2025-12-01

### Added
- **Initial PyPI release** - Package available as `light-theory-realm` on PyPI
- **Command-line interface** with 5 subcommands:
  - `light-realm sm-table` - Standard Model fermion mass table (Pocket_U Lite)
  - `light-realm profile` - Full geometric profile for particles
  - `light-realm koide` - Compute Koide ratio and Foot angle
  - `light-realm koide-predict` - Predict third mass from first two using Koide
  - `light-realm koide-from-pocket` - Koide analysis using Pocket_U Lite predictions
- **Pocket_U Lite mass model** - Reproduces 9 Standard Model fermion masses with ~2.3% average error
- **Clifford algebra engine** - JAX-based Cl(1,3) implementation with grade projections
- **Quantum Geometric Tensor** - Computes Fisher information metric and Berry curvature
- **Geometric fingerprint system** - Full profile for each particle (metric, curvature, KK uplift, Reeb flow)
- **Koide paper** - 17-page analysis "Koide relations from prime plaquettes" with comprehensive bibliography
- **Complete documentation**:
  - Foundations of Light Theory (beginner-friendly)
  - Light Mechanics (full mathematical treatment)
  - Executive Summary (one-pager for decision makers)
  - Architecture documentation
  - API reference
  - Testing guide
  - Contributing guidelines

### Technical Details
- **Dependencies**: JAX, NumPy, SciPy
- **License**: Apache 2.0
- **Python compatibility**: Python 3.8+
- **Platform**: CPU/GPU ready via JAX

### Files
- `light_theory_realm/` - Core Python package
- `examples/` - Usage examples and demonstrations
- `tests/` - Unit tests (7/7 passing)
- `paper/` - Koide paper source, PDF, and supporting data
- `Foundations/` - Beginner and advanced theory documentation

## Pre-Release History

### 2025-11-30
- CLI implementation completed
- Documentation reorganization
- Architecture documentation finalized

### 2025-11-29
- Pocket_U Lite mass model implementation
- Geometric fingerprint system
- Initial test suite

### 2025-11-28
- Clifford algebra engine
- Quantum Geometric Tensor implementation
- Basic framework structure

[Unreleased]: https://github.com/Pleroma-Works/Light_Theory_Realm/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Pleroma-Works/Light_Theory_Realm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Pleroma-Works/Light_Theory_Realm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Pleroma-Works/Light_Theory_Realm/releases/tag/v0.1.0
