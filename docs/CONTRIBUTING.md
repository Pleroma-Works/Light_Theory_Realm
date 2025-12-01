# Contributing to Light Theory Realm

Thanks for your interest! Light Theory Realm is an open-source project and contributions are welcome.

---

## Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/yourusername/Light_Theory_Realm
   cd Light_Theory_Realm
   ```

2. **Install in editable mode**
   ```bash
   pip install -e .
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

---

## Code Style

### Docstrings
Use NumPy-style docstrings:

```python
def compute_qgt(psi: jnp.ndarray, jacobian: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Brief one-liner description.
    
    Longer explanation of what this does and why.
    
    Args:
        psi: State vector, shape (n,)
        jacobian: Derivative matrix, shape (n, d)
    
    Returns:
        Tuple of (fisher, berry):
            - fisher: (d, d) Fisher information metric
            - berry: (d, d) Berry curvature
            
    Example:
        >>> fisher, berry = compute_qgt(psi, jac)
    """
```

### Type Hints
Add type hints to all public functions:

```python
def add(a: float, b: float) -> float:
    return a + b
```

### Imports
- Standard library first
- Third-party (jax, numpy, etc.)
- Local imports last
- Sort within each group

```python
import jax
import jax.numpy as jnp
from typing import Dict, Tuple

from .engine import CliffordEngine
from .qgt import CliffordQGT
```

### Comments
Only comment non-obvious code:

```python
# ✅ Good: explains WHY
# Central difference (more accurate than forward/backward at midpoint)
dpsi = (psi_next - psi_prev) / (theta_next - theta_prev)

# ❌ Bad: explains WHAT (code already says this)
# Compute derivative
dpsi = (psi_next - psi_prev) / (theta_next - theta_prev)
```

---

## Adding a Feature

1. **Create a branch**
   ```bash
   git checkout -b feature/my-cool-thing
   ```

2. **Write code + tests**
   - Add feature in appropriate module
   - Add tests in `tests/`
   - Ensure `pytest` passes

3. **Update documentation**
   - Add docstrings
   - Update `README.md` if user-facing
   - Update `API_REFERENCE.md` if public API changes

4. **Commit with clear message**
   ```bash
   git commit -m "Add feature: XYZ

   - What it does
   - Why it matters
   - Any breaking changes
   "
   ```

5. **Push & open PR**
   ```bash
   git push origin feature/my-cool-thing
   ```

---

## Testing

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test
```bash
pytest tests/test_qgt.py::TestCliffordQGT::test_inner_product -v
```

### Check coverage
```bash
pytest --cov=light_theory_realm tests/
```

### Add a new test
```python
# In tests/test_my_feature.py

def test_my_feature():
    """Describe what you're testing."""
    result = my_function(input_data)
    assert result == expected_output
```

### Expected coverage
- Target: ≥80% on core modules
- Current: See `pytest --cov` output
- Gaps are OK if documented

---

## Reporting Bugs

1. **Check existing issues** first
2. **Create issue with:**
   - Clear title
   - Minimal reproducible example
   - Expected vs actual behavior
   - Environment (Python version, JAX version, OS)

Example:
```
Title: Profile CLI command hangs on import

Environment:
- Python 3.11
- JAX 0.8.1
- Ubuntu 22.04

Minimal example:
```python
light-realm profile e
```

Expected: Shows electron profile
Actual: Hangs indefinitely
```

---

## Submitting Pull Requests

- **Describe what and why** in PR description
- **Link related issues** (Fixes #123)
- **One feature per PR** if possible
- **Keep changes focused** (easier to review)
- **Tests must pass** before merge

Template:
```
## What
Brief description of the changes

## Why
Why is this change needed?

## How
How does it work?

## Testing
How was this tested?

Fixes #123
```

---

## Architecture & Extension Points

### 4-Layer Stack

```
[ Algebra Layer ]    CliffordEngine      
                     ├── gammas (Cl(1,3) basis)
                     ├── wedge_product()
                     └── project_grade()

[ Geometry Layer ]   CliffordQGT
                     ├── compute_full_qgt()
                     └── inner_product()

[ Theory Layer ]     Kaluza-Klein Uplift + Reeb Flow
                     ├── construct_5d_metric()
                     └── compute_reeb_vector()

[ Experiment Layer ] Pocket_U + Koide + Prime Plaquettes
                     ├── physical_mass()
                     ├── koide_ratio()
                     └── get_particle_profile()
```

### Where to Add Features

- **New algebraic operation?** → `CliffordEngine`
- **New geometric computation?** → `CliffordQGT` or new module in `experiments/`
- **New particle or mass model?** → Extend `Pocket_U Lite` plaquettes
- **New Koide-like relation?** → Extend `pocket_u_lite/koide.py`

---

## Documentation

### For Users
- `README.md` — Project overview
- `QUICKSTART.md` — Get started in 5 min
- `INSTALL.md` — Installation guide
- `Light_Realm_Easy_Mode.md` — Non-physicist intro
- `Light_Realm_Multidimensional.md` — Deep dive

### For Developers
- `API_REFERENCE.md` — Complete API docs
- `ARCHITECTURE.md` — Design & structure
- `CONTRIBUTING.md` — This file

### When to update docs
- After API changes
- After adding a new public function
- After changing behavior
- After adding examples

---

## Questions?

- Open an issue with `[question]` tag
- Check existing issues
- Read the README & docs first

---

## Code of Conduct

Be respectful. We're all here to learn and build something cool.

---

## License

By contributing, you agree your code is licensed under Apache 2.0 (see LICENSE file).
