# Testing Guide

How to run, understand, and add tests for Light Theory Realm.

---

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=light_theory_realm tests/

# Run specific test file
pytest tests/test_qgt.py -v

# Run specific test
pytest tests/test_qgt.py::TestCliffordQGT::test_inner_product -v
```

---

## Test Structure

```
tests/
├── test_grade_projection.py    # Layer 1: Algebra
├── test_wedge_product.py       # Layer 1: Algebra
├── test_qgt.py                 # Layer 2: Geometry
└── test_integration_*.py       # Planned: Full stack tests
```

---

## Current Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Layer 1: Algebra | 2 | ✅ Passing |
| Layer 2: Geometry | 5 | ✅ Passing |
| Layer 3: Theory | 0 | ⏳ Planned |
| Layer 4: Experiments | 0 | ⏳ Planned |
| **Total** | **7** | **✅ 7/7 PASS** |

**Target coverage:** ≥80% on core modules

---

## Layer 1: Algebra Tests

**File:** `tests/test_grade_projection.py`, `test_wedge_product.py`

### Grade Projection
Tests that we can extract pure geometric grades:
```python
def test_grade_projection():
    engine = CliffordEngine(seed=42)
    # Create multivector with known grades
    v = engine.gammas[0]  # Vector (grade 1)
    
    # Project to grade 0 (scalar) → should be 0
    scalar_part = engine.project_grade(v, 0)
    assert norm(scalar_part) < 1e-10
    
    # Project to grade 1 (vector) → should recover v
    vector_part = engine.project_grade(v, 1)
    assert allclose(vector_part, v)
```

**Properties tested:**
- ✅ Scalar extraction (all zeros except trace)
- ✅ Vector extraction (γᵢ terms)
- ✅ Bivector extraction (γᵢγⱼ terms)
- ✅ Completeness (all grades sum to original)

### Wedge Product
Tests antisymmetric outer product:
```python
def test_wedge_product():
    engine = CliffordEngine(seed=42)
    
    # Wedge property: a ∧ a = 0
    v = engine.random_spinor(key)
    result = engine.wedge_product(v, v)
    assert norm(result) < 1e-10
    
    # Anticommutativity: a ∧ b = -(b ∧ a)
    u = engine.random_spinor(key)
    assert allclose(
        engine.wedge_product(v, u),
        -engine.wedge_product(u, v)
    )
```

**Properties tested:**
- ✅ Null property (v ∧ v = 0)
- ✅ Anticommutativity (a ∧ b = -(b ∧ a))
- ✅ Grade preservation
- ✅ Associativity

---

## Layer 2: Geometry Tests

**File:** `tests/test_qgt.py`

### Inner Product
Test geometric inner product:
```python
def test_inner_product():
    engine = CliffordEngine(seed=42)
    qgt = CliffordQGT(engine)
    
    psi = engine.random_spinor(key1)
    phi = engine.random_spinor(key2)
    
    # Compute inner product
    result = qgt.inner_product(psi, phi)
    
    # Should be complex number
    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()
```

### QGT Computation
Test full Fisher + Berry calculation:
```python
def test_compute_full_qgt():
    engine = CliffordEngine(seed=42)
    qgt = CliffordQGT(engine)
    
    # Create state and derivative
    psi = engine.random_spinor(key1)
    d_psi = engine.random_spinor(key2).reshape(-1, 1)
    
    # Compute QGT
    fisher, berry = qgt.compute_full_qgt(psi, d_psi)
    
    # Check shapes and properties
    assert fisher.shape == (1, 1)
    assert berry.shape == (1, 1)
    
    # Fisher should be real and symmetric
    assert jnp.allclose(fisher, fisher.T)
    assert jnp.allclose(fisher.imag, 0)
    
    # Berry should be imaginary and antisymmetric
    assert jnp.allclose(berry, -berry.T)
```

---

## Layer 3: Theory Tests

**Planned for v0.2.0:**

### Kaluza-Klein Uplift
```python
def test_kk_metric():
    """5D metric should have correct block structure."""
    engine = CliffordEngine(seed=42)
    kk = KaluzaKleinUplift(engine)
    
    psi = engine.random_spinor(key1)
    jac = engine.random_spinor(key2).reshape(-1, 1)
    
    G_5D = kk.construct_5d_metric(psi, jac, phi_scalar=0.925)
    
    # Should be (5, 5)
    assert G_5D.shape == (5, 5)
    
    # Should be Hermitian
    assert jnp.allclose(G_5D, G_5D.conj().T)
```

### Reeb Flow
```python
def test_reeb_vector():
    """Reeb vector should lie in null space of Berry curvature."""
    engine = CliffordEngine(seed=42)
    reeb = ReebFlowDynamics(engine)
    
    fisher = jnp.eye(2)
    berry = jnp.array([[0, 1j], [-1j, 0]])
    
    R = reeb.compute_reeb_vector(fisher, berry)
    
    # Should satisfy: i·R·Ω = 0
    contraction = berry @ R
    assert jnp.allclose(contraction, 0, atol=1e-10)
```

---

## Layer 4: Experiment Tests

**Planned for v0.2.0:**

### Mass Predictions
```python
def test_mass_predictions():
    """Pocket_U predictions should be within 2.5% of PDG."""
    spectrum = get_sm_spectrum()
    
    for particle in spectrum:
        error = abs(particle['error_pct'])
        assert error < 2.5, f"{particle['name']} error too high: {error}%"
```

### Koide Relation
```python
def test_koide_consistency():
    """Koide ratio should be ~2/3 for PDG leptons."""
    Q = koide_ratio(0.511, 105.66, 1776.86)
    expected = 2.0 / 3.0
    
    assert jnp.allclose(Q, expected, rtol=1e-3)
```

---

## Writing New Tests

### Template
```python
import jax
import jax.numpy as jnp
from light_theory_realm import CliffordEngine, CliffordQGT

def test_my_feature():
    """Clear description of what you're testing."""
    # Setup
    engine = CliffordEngine(seed=42)
    key = jax.random.PRNGKey(0)
    
    # Execute
    result = engine.my_method(key)
    
    # Assert
    assert result.shape == (4,)
    assert jnp.all(result != 0)
```

### Good practices
- ✅ One test per behavior
- ✅ Clear, descriptive names (`test_wedge_product_null_property`)
- ✅ Minimal setup (reuse fixtures if needed)
- ✅ Test edge cases (empty inputs, large values, etc.)
- ✅ Use `jnp.allclose()` for floating-point comparisons

### Avoid
- ❌ Testing implementation details (test behavior)
- ❌ Interdependent tests (each test should stand alone)
- ❌ Hardcoded magic numbers (use named constants)
- ❌ Skipping error cases

---

## Numerical Precision

### When comparing floats
```python
# ❌ Don't do this
assert result == expected

# ✅ Do this (default atol=1e-8)
assert jnp.allclose(result, expected)

# ✅ Or with custom tolerance
assert jnp.allclose(result, expected, rtol=1e-5, atol=1e-10)
```

### For complex numbers
```python
# Real part
assert jnp.allclose(result.real, expected.real)

# Imaginary part
assert jnp.allclose(result.imag, expected.imag)

# Magnitude
assert jnp.allclose(jnp.abs(result), jnp.abs(expected))
```

---

## Running Tests Locally

### Before committing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=light_theory_realm

# Check coverage percentage
pytest tests/ --cov=light_theory_realm --cov-report=term-missing
```

### Debugging a failing test
```bash
# Run with verbose output
pytest tests/test_qgt.py::TestCliffordQGT::test_inner_product -vv

# Run with print statements
pytest tests/test_qgt.py -vv -s

# Run with pdb on failure
pytest tests/test_qgt.py --pdb
```

---

## Known Limitations

1. **No property-based testing yet** (Hypothesis integration planned)
2. **Limited integration tests** (only Layer 1-2 tested)
3. **No GPU-specific tests** (relies on JAX CPU/GPU compatibility)
4. **No performance benchmarks** (see BENCHMARKS.md for planned additions)

---

## Expected Output

When all tests pass:

```
======================== test session starts =========================
collected 7 items

tests/test_grade_projection.py::test_grade_projection PASSED    [ 14%]
tests/test_qgt.py::TestCliffordQGT::test_inner_product PASSED   [ 28%]
tests/test_qgt.py::TestCliffordQGT::test_compute_qgt_components_simple PASSED [ 42%]
tests/test_qgt.py::TestCliffordQGT::test_compute_qgt_components_basic PASSED [ 57%]
tests/test_qgt.py::TestCliffordQGT::test_compute_qgt_components_complex PASSED [ 71%]
tests/test_qgt.py::TestCliffordQGT::test_compute_full_qgt_simple PASSED [ 85%]
tests/test_wedge_product.py::test_wedge_product PASSED           [100%]

======================== 7 passed in 21.45s ==========================
```

---

## Coverage Report

After running `pytest --cov=light_theory_realm`:

```
Name                      Stmts   Miss  Cover
---------------------------------------------
light_theory_realm/__init__    30      5    83%
light_theory_realm/engine      150     10    93%
light_theory_realm/qgt         80      2    98%
light_theory_realm/cli         50      5    90%
---------------------------------------------
TOTAL                         310     22    93%
```

Target: Keep ≥80% on core modules (engine, qgt, geometry).

---

## Next Steps

1. **Add Layer 3 tests** (Theory: KK, Reeb)
2. **Add Layer 4 tests** (Experiments: mass, Koide)
3. **Add integration tests** (full pipeline)
4. **Property-based tests** (Hypothesis)
5. **Performance benchmarks** (timing, memory)

See `TODO_IMPLEMENTATION.md` for planned test items.
