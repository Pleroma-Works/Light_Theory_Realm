# API Reference - Light Theory Realm

Complete API documentation for the Light Theory Realm framework.

## Table of Contents
1. [CliffordEngine](#cliffordengine)
2. [CliffordQGT](#cliffordqgt)
3. [Kaluza-Klein Uplift](#kaluza-klein-uplift)
4. [Prime Gauge Experiments](#prime-gauge-experiments)

---

## CliffordEngine

The core Clifford algebra engine implementing Cl(1,3) using JAX.

### Class: CliffordEngine

```python
from light_theory_realm import CliffordEngine

engine = CliffordEngine(seed=42)
```

#### Constructor

```python
def __init__(self, seed: int = 42)
```

**Parameters**:
- `seed` (int, default=42): Random seed for JAX PRNG

**Attributes**:
- `key`: JAX PRNG key
- `dim` (int): Dimension of matrices (fixed at 4 for Cl(1,3))
- `gammas` (list[jnp.ndarray]): Four 4x4 gamma matrices
- `bivectors` (dict): Six 4x4 bivector matrices
- `basis_by_grade` (dict): 16-element basis organized by grade

---

### Grade Projection

#### Method: project_grade

```python
def project_grade(self, multivector: jnp.ndarray, grade: int) -> jnp.ndarray
```

Decomposes a multivector onto a specific geometric grade.

**Parameters**:
- `multivector` (jnp.ndarray): Input multivector (4x4 complex matrix)
- `grade` (int): Target grade (0=scalar, 1=vector, 2=bivector, 3=trivector, 4=pseudoscalar)

**Returns**:
- jnp.ndarray: Component of the multivector at the specified grade

**Formula**:
$$M_k = \sum_{B_i \in \text{Basis}_k} \langle M \cdot B_i^{-1} \rangle_0 \cdot B_i$$

**Example**:
```python
# Extract scalar component
scalar = engine.project_grade(multivector, 0)

# Extract vector component
vector = engine.project_grade(multivector, 1)

# Extract bivector component
bivector = engine.project_grade(multivector, 2)

# Reconstruct multivector
reconstructed = sum(engine.project_grade(multivector, g) for g in range(5))
```

**Properties**:
- Grade 0: Scalar (diagonal multiples of identity)
- Grade 1: Vectors (linear combinations of gamma matrices)
- Grade 2: Bivectors (log-generators of rotations)
- Grade 3: Trivectors (dual to vectors)
- Grade 4: Pseudoscalar (volume element)

---

### Wedge Product

#### Method: wedge_product

```python
def wedge_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray
```

Computes the wedge (outer) product of two multivectors.

**Parameters**:
- `a` (jnp.ndarray): First multivector (4x4 complex matrix)
- `b` (jnp.ndarray): Second multivector (4x4 complex matrix)

**Returns**:
- jnp.ndarray: Wedge product A ∧ B

**Formula**:
$$A \wedge B = \sum_{r=0}^{4} \sum_{s=0}^{4} \langle A_r B_s \rangle_{r+s}$$

where r + s ≤ 4.

**Example**:
```python
v1 = engine.gammas[0]
v2 = engine.gammas[1]

# Vector wedge produces bivector
bivector = engine.wedge_product(v1, v2)

# Null property: v ∧ v = 0
result = engine.wedge_product(v1, v1)
assert jnp.linalg.norm(result) < 1e-10

# Anticommutativity: v1 ∧ v2 = -v2 ∧ v1
result = engine.wedge_product(v1, v2) + engine.wedge_product(v2, v1)
assert jnp.linalg.norm(result) < 1e-10
```

**Properties**:
- Null vector: v ∧ v = 0
- Anticommutative: A ∧ B = -B ∧ A
- Associative: (A ∧ B) ∧ C = A ∧ (B ∧ C)
- Grade-preserving: Grade(A ∧ B) = Grade(A) + Grade(B)

---

### Basic Operations

#### Method: geometric_product

```python
def geometric_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray
```

Computes the geometric product (matrix multiplication).

**Returns**: A * B (jnp.ndarray)

---

#### Method: scalar_part

```python
def scalar_part(self, multivector: jnp.ndarray) -> jnp.complex128
```

Extracts the scalar (grade-0) part of a multivector.

**Returns**: Trace divided by dimension

**Example**:
```python
scalar = engine.scalar_part(multivector)
# Useful for extracting coefficients in projections
```

---

#### Method: reverse

```python
def reverse(self, multivector: jnp.ndarray) -> jnp.ndarray
```

Computes the Clifford reverse (matrix transpose).

**Returns**: Transpose of the multivector matrix

---

#### Method: adjoint

```python
def adjoint(self, multivector: jnp.ndarray) -> jnp.ndarray
```

Computes the Hermitian adjoint (conjugate transpose).

**Returns**: Conjugate transpose of the multivector matrix

---

#### Method: inverse

```python
def inverse(self, multivector: jnp.ndarray) -> jnp.ndarray
```

Computes the Clifford inverse.

**Formula**: A^(-1) = A^reverse / scalar_part(A * A^reverse)

**Raises**: ValueError if multivector is not invertible

---

#### Method: commutator

```python
def commutator(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray
```

Computes the commutator [A, B] = AB - BA.

**Returns**: jnp.ndarray

---

#### Method: anticommutator

```python
def anticommutator(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray
```

Computes the anticommutator {A, B} = AB + BA.

**Returns**: jnp.ndarray

---

### Basis Access

#### Method: get_bivector

```python
def get_bivector(self, i: int, j: int) -> jnp.ndarray
```

Retrieves a bivector generator S_ij.

**Parameters**:
- `i` (int): First index (0-3)
- `j` (int): Second index (0-3)

**Returns**: Bivector matrix S_ij

---

#### Property: gammas

List of four 4x4 gamma matrices (generators of Cl(1,3)).

---

### Utilities

#### Method: random_spinor

```python
def random_spinor(self, key: jax.random.PRNGKey) -> jnp.ndarray
```

Generates a random normalized 4-component complex spinor.

**Parameters**:
- `key` (jax.random.PRNGKey): JAX random key

**Returns**: 4x1 normalized complex spinor (jnp.ndarray)

---

## CliffordQGT

Quantum Geometric Tensor computation using the Clifford engine.

### Class: CliffordQGT

```python
from light_theory_realm import CliffordQGT

qgt = CliffordQGT(engine)
```

#### Constructor

```python
def __init__(self, engine: CliffordEngine)
```

**Parameters**:
- `engine` (CliffordEngine): Initialized Clifford algebra engine

---

### Core Methods

#### Method: compute_full_qgt

```python
def compute_full_qgt(self, psi: jnp.ndarray, jacobian: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Computes the full Quantum Geometric Tensor decomposed into Fisher metric and Berry curvature.

**Parameters**:
- `psi` (jnp.ndarray): State vector (4x1 or 4-element)
- `jacobian` (jnp.ndarray): Jacobian matrix (4 x n_params) where n_params is the number of parameters

**Returns**:
- `fisher_metric` (jnp.ndarray): (n_params x n_params) symmetric matrix
- `berry_curvature` (jnp.ndarray): (n_params x n_params) antisymmetric matrix

**Formula**:
$$T_{uv} = g_{uv} + i\Omega_{uv}$$

where g_uv is the Fisher metric and Omega_uv is the Berry curvature.

**Example**:
```python
psi = engine.random_spinor(key)
jacobian = jnp.hstack([d_psi_du, d_psi_dv])  # n_params=2

fisher, berry = qgt.compute_full_qgt(psi, jacobian)

print(f"Fisher metric shape: {fisher.shape}")  # (2, 2)
print(f"Berry curvature shape: {berry.shape}")  # (2, 2)
```

---

#### Method: compute_qgt_components

```python
def compute_qgt_components(self, psi: jnp.ndarray, d_psi_u: jnp.ndarray, d_psi_v: jnp.ndarray) -> Tuple[jnp.complex128, jnp.complex128]
```

Computes QGT components for two specific parameter directions.

**Parameters**:
- `psi` (jnp.ndarray): State vector (4x1)
- `d_psi_u` (jnp.ndarray): Derivative w.r.t. first parameter (4x1)
- `d_psi_v` (jnp.ndarray): Derivative w.r.t. second parameter (4x1)

**Returns**:
- `fisher_uv` (jnp.complex128): Fisher metric component
- `berry_uv` (jnp.complex128): Berry curvature component

---

#### Method: inner_product

```python
def inner_product(self, psi_a: jnp.ndarray, psi_b: jnp.ndarray) -> jnp.complex128
```

Computes the inner product between two spinors.

**Returns**: Conjugate dot product of spinors

---

## Kaluza-Klein Uplift

### Class: KaluzaKleinUplift

```python
from light_theory_realm.experiments.prime_gauge.uplift import KaluzaKleinUplift

kk = KaluzaKleinUplift(engine)
```

Uplifts 4D information manifolds to 5D Kaluza-Klein geometry.

#### Constructor

```python
def __init__(self, engine: CliffordEngine)
```

---

#### Method: construct_5d_metric

```python
def construct_5d_metric(self, psi: jnp.ndarray, jacobian: jnp.ndarray, phi_scalar: float = 0.9) -> jnp.ndarray
```

Constructs the 5D Kaluza-Klein metric tensor.

**Parameters**:
- `psi` (jnp.ndarray): Quantum state (4x1)
- `jacobian` (jnp.ndarray): Parameter derivatives (4 x n_params)
- `phi_scalar` (float, default=0.9): Dilaton field controlling extra dimension size

**Returns**:
- `G_5D` (jnp.ndarray): (n_params+1) x (n_params+1) metric tensor

**Metric Structure**:
$$\hat{G}_{AB} = \begin{pmatrix} g_{\mu\nu} + \phi^2 A_\mu A_\nu & \phi^2 A_\mu \\ \phi^2 A_\nu & \phi^2 \end{pmatrix}$$

**Example**:
```python
G_5D = kk.construct_5d_metric(psi, jacobian, phi_scalar=0.9)
print(f"5D metric shape: {G_5D.shape}")  # (3, 3) for 2 parameters
```

---

## Prime Gauge Experiments

### Class: PrimeLattice

```python
from light_theory_realm.experiments.prime_gauge.prime_plaquette import PrimeLattice

lattice = PrimeLattice({0: 2, 1: 3})
```

Manages prime number lattice with edge parameters.

#### Method: compute_edge_params

```python
def compute_edge_params(self, i: int, j: int) -> Tuple[float, float]
```

Computes classical parameters for edge between nodes i and j.

**Returns**:
- `J` (float): Stiffness parameter
- `A` (float): Twist angle (radians)

**Example**:
```python
J, A = lattice.compute_edge_params(0, 1)
print(f"Stiffness: {J:.4f}, Twist: {A:.4f}")
```

---

### Class: ReebFlowDynamics

```python
from light_theory_realm.experiments.prime_gauge.reeb_flow import ReebFlowDynamics

reeb = ReebFlowDynamics(engine)
```

Implements Reeb vector flow for dark energy and time evolution.

#### Method: compute_resonance_density

```python
def compute_resonance_density(self, primes: list) -> float
```

Computes vacuum resonance density for given prime sequence.

**Parameters**:
- `primes` (list): List of prime numbers

**Returns**:
- `xi` (float): Resonance density (0 to 1)

**Example**:
```python
xi = reeb.compute_resonance_density([17, 41, 73])
print(f"Resonance density: {xi:.4f}")
```

---

#### Method: compute_reeb_vector

```python
def compute_reeb_vector(self, fisher_metric: jnp.ndarray, berry_curvature: jnp.ndarray) -> jnp.ndarray
```

Extracts the Reeb vector (time direction) from geometric tensors.

**Parameters**:
- `fisher_metric` (jnp.ndarray): (n x n) symmetric matrix
- `berry_curvature` (jnp.ndarray): (n x n) antisymmetric matrix

**Returns**:
- `R` (jnp.ndarray): Reeb vector (n,)

**Properties**:
- Norm invariant: ||R^T @ Fisher @ R|| = 1.0
- Represents direction of minimal topological resistance

**Example**:
```python
R = reeb.compute_reeb_vector(fisher, berry)
norm = jnp.linalg.norm(R.T @ fisher @ R)
print(f"Reeb norm invariant: {norm:.4f}")  # Should be ~1.0
```

---

## Testing

### Test Modules

**test_grade_projection.py**: Grade projection functionality (4 tests)
```bash
python test_grade_projection.py
```

**test_wedge_product.py**: Wedge product properties (8 tests)
```bash
python test_wedge_product.py
```

**tests/test_qgt.py**: Quantum geometric tensor (5 tests)
```bash
python -m pytest tests/test_qgt.py -v
```

**verify_full_stack.py**: End-to-end integration (9 tests)
```bash
python verify_full_stack.py
```

---

## Type Hints

All methods use JAX-compatible type hints:
- `jnp.ndarray`: JAX array (complex128 for Clifford algebra)
- `jnp.complex128`: Complex scalar
- `jax.random.PRNGKey`: JAX random key

---

## Performance Notes

- All operations are JIT-compiled with `@functools.partial(jit, ...)`
- Uses `float64` precision for numerical stability
- Suitable for real-time quantum geometric calculations
- Scales efficiently with parameter count

---

## Mathematical Conventions

- Clifford algebra: Cl(1,3) with Weyl representation
- Signature: (1, -1, -1, -1)
- Grade definitions: 0=scalar, 1=vector, 2=bivector, 3=trivector, 4=pseudoscalar
- Wedge product: Anti-commutative outer product
- QGT: Complex tensor with real (Fisher) and imaginary (Berry) parts

