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
- jnp.ndarray: Wedge product A âˆ§ B

**Formula**:
$$A \wedge B = \sum_{r=0}^{4} \sum_{s=0}^{4} \langle A_r B_s \rangle_{r+s}$$

where r + s â‰¤ 4.

**Example**:
```python
v1 = engine.gammas[0]
v2 = engine.gammas[1]

# Vector wedge produces bivector
bivector = engine.wedge_product(v1, v2)

# Null property: v âˆ§ v = 0
result = engine.wedge_product(v1, v1)
assert jnp.linalg.norm(result) < 1e-10

# Anticommutativity: v1 âˆ§ v2 = -v2 âˆ§ v1
result = engine.wedge_product(v1, v2) + engine.wedge_product(v2, v1)
assert jnp.linalg.norm(result) < 1e-10
```

**Properties**:
- Null vector: v âˆ§ v = 0
- Anticommutative: A âˆ§ B = -B âˆ§ A
- Associative: (A âˆ§ B) âˆ§ C = A âˆ§ (B âˆ§ C)
- Grade-preserving: Grade(A âˆ§ B) = Grade(A) + Grade(B)

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

#   P o c k e t _ U   L i t e   &   K o i d e   A P I   D o c u m e n t a t i o n  
  
 T h i s   d o c u m e n t   c o n t a i n s   t h e   A P I   a d d i t i o n s   f o r   P o c k e t _ U   L i t e   a n d   K o i d e   u t i l i t i e s .  
 A p p e n d   t h i s   c o n t e n t   t o   A P I _ R E F E R E N C E . m d   a f t e r   t h e   " P r i m e   G a u g e   E x p e r i m e n t s "   s e c t i o n .  
  
 - - -  
  
 # #   P o c k e t _ U   L i t e :   S t a n d a r d   M o d e l   T o y  
  
 # # #   O v e r v i e w  
  
 P o c k e t _ U   L i t e   p r o v i d e s   r e a d - o n l y   a c c e s s   t o   S t a n d a r d   M o d e l   m a s s   p r e d i c t i o n s   f r o m   p r i m e   p l a q u e t t e s .  
  
 ` ` ` p y t h o n  
 f r o m   l i g h t _ t h e o r y _ r e a l m   i m p o r t   (  
         g e t _ s m _ s p e c t r u m ,  
         p r i n t _ s m _ t a b l e ,  
         g e t _ p a r t i c l e _ p r o f i l e ,  
         p h y s i c a l _ m a s s ,  
 )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   g e t _ s m _ s p e c t r u m  
  
 ` ` ` p y t h o n  
 d e f   g e t _ s m _ s p e c t r u m ( )   - >   L i s t [ D i c t ]  
 ` ` `  
  
 G e t   t h e   c o m p l e t e   S t a n d a r d   M o d e l   m a s s   s p e c t r u m   f o r   a l l   9   f e r m i o n s .  
  
 * * R e t u r n s * * :  
 -   L i s t   o f   d i c t i o n a r i e s ,   o n e   p e r   p a r t i c l e ,   c o n t a i n i n g :  
     -   ` n a m e ` :   P a r t i c l e   n a m e   ( e . g . ,   " e " ,   " m u " ,   " t a u " )  
     -   ` m a s s _ p d g ` :   P D G   t a r g e t   m a s s   ( M e V )  
     -   ` m a s s _ p r e d ` :   P r e d i c t e d   m a s s   ( M e V )  
     -   ` e r r o r _ p c t ` :   P r e d i c t i o n   e r r o r   ( % )  
     -   ` p l a q u e t t e ` :   P r i m e   p l a q u e t t e   l o o p  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 s p e c t r u m   =   g e t _ s m _ s p e c t r u m ( )  
 f o r   p a r t i c l e   i n   s p e c t r u m :  
         p r i n t ( f " { p a r t i c l e [ ' n a m e ' ] } :   { p a r t i c l e [ ' m a s s _ p r e d ' ] : . 3 f }   M e V " )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   p r i n t _ s m _ t a b l e  
  
 ` ` ` p y t h o n  
 d e f   p r i n t _ s m _ t a b l e ( )   - >   N o n e  
 ` ` `  
  
 P r i n t   a   f o r m a t t e d   t a b l e   o f   t h e   S t a n d a r d   M o d e l   m a s s   s p e c t r u m .  
  
 * * O u t p u t   E x a m p l e * * :  
 ` ` `  
 P o c k e t _ U   L i t e :   S t a n d a r d   M o d e l   M a s s   S p e c t r u m  
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =  
 P a r t i c l e       P D G   ( M e V )         P r e d i c t e d   ( M e V )       E r r o r   ( % )       P l a q u e t t e  
 - - - - - - - -       - - - - - - - - -         - - - - - - - - - - - - - - -       - - - - - - - - -       - - - - - - - - -  
 e                             0 . 5 1 1                     0 . 5 1 1                       0 . 0 %         [ 2 , 3 , 5 , 7 ]  
 m u                       1 0 5 . 6 5 8                 1 0 5 . 2 4 1                       0 . 4 %         [ 2 , 1 7 , 2 3 , 7 ]  
 . . .  
 A v e r a g e   e r r o r :   2 . 3 1 %  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   g e t _ p a r t i c l e _ p r o f i l e  
  
 ` ` ` p y t h o n  
 d e f   g e t _ p a r t i c l e _ p r o f i l e ( n a m e :   s t r ,   p h i _ s c a l a r :   f l o a t   =   0 . 9 2 5 )   - >   D i c t  
 ` ` `  
  
 G e t   c o m p l e t e   p h y s i c a l   a n d   g e o m e t r i c   p r o f i l e   f o r   a   p a r t i c l e .  
  
 * * P a r a m e t e r s * * :  
 -   ` n a m e `   ( s t r ) :   P a r t i c l e   n a m e   ( " e " ,   " m u " ,   " t a u " ,   " u " ,   " d " ,   " s " ,   " c " ,   " b " ,   " t " )  
 -   ` p h i _ s c a l a r `   ( f l o a t ,   d e f a u l t = 0 . 9 2 5 ) :   D i l a t o n   f i e l d   f o r   K K   m e t r i c  
  
 * * R e t u r n s * * :  
 D i c t i o n a r y   c o n t a i n i n g :  
  
 * * M a s s   f i e l d s * * :  
 -   ` m _ p h y s _ M e V ` :   P h y s i c a l   m a s s   ( M e V )  
 -   ` p d g _ m a s s _ M e V ` :   P D G   t a r g e t   ( M e V )  
 -   ` e r r o r _ p c t ` :   P r e d i c t i o n   e r r o r   ( % )  
 -   ` m _ g e o m _ M e V ` :   G e o m e t r i c   m a s s   c o m p o n e n t  
 -   ` m _ v a c _ M e V ` :   V a c u u m   s c r e e n i n g   c o m p o n e n t  
 -   ` s c r e e n i n g _ p c t ` :   S c r e e n i n g   p e r c e n t a g e  
  
 * * G e o m e t r y   f i e l d s * * :  
 -   ` f i s h e r ` :   ( d ,   d )   F i s h e r   i n f o r m a t i o n   m e t r i c  
 -   ` b e r r y ` :   ( d ,   d )   B e r r y   c u r v a t u r e  
 -   ` f i s h e r _ t r a c e ` :   T r a c e   o f   F i s h e r   m e t r i c  
 -   ` b e r r y _ n o r m ` :   F r o b e n i u s   n o r m   o f   B e r r y   c u r v a t u r e  
 -   ` G _ 5 D ` :   ( d + 1 ,   d + 1 )   K a l u z a - K l e i n   m e t r i c  
 -   ` x i ` :   R e s o n a n c e   d e n s i t y   ( d a r k   e n e r g y )  
 -   ` R ` :   R e e b   v e c t o r   ( t i m e   d i r e c t i o n )  
 -   ` R _ n o r m ` :   N o r m   o f   R e e b   v e c t o r  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 e _ p r o f i l e   =   g e t _ p a r t i c l e _ p r o f i l e ( " e " )  
 p r i n t ( f " M a s s :   { e _ p r o f i l e [ ' m _ p h y s _ M e V ' ] : . 3 f }   M e V " )  
 p r i n t ( f " S c r e e n i n g :   { e _ p r o f i l e [ ' s c r e e n i n g _ p c t ' ] : . 1 f } % " )  
 p r i n t ( f " D a r k   e n e r g y   Î ¾ :   { e _ p r o f i l e [ ' x i ' ] : . 4 f } " )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   p h y s i c a l _ m a s s  
  
 ` ` ` p y t h o n  
 d e f   p h y s i c a l _ m a s s ( n a m e :   s t r )   - >   D i c t  
 ` ` `  
  
 C o m p u t e   p h y s i c a l   m a s s   f o r   a   p a r t i c l e   u s i n g   t h e   P o c k e t _ U   v 1   f o r m u l a .  
  
 * * F o r m u l a * * :  
 $ $ m _ { \ t e x t { p h y s } }   =   \ L a m b d a   \ c d o t   | M _ { \ t e x t { g e o m } }   +   \ a l p h a   \ z e t a _ { \ t e x t { v a c } } | $ $  
  
 * * P a r a m e t e r s * * :  
 -   ` n a m e `   ( s t r ) :   P a r t i c l e   n a m e  
  
 * * R e t u r n s * * :  
 D i c t i o n a r y   w i t h   d e t a i l e d   m a s s   b r e a k d o w n :  
 -   ` n a m e ` :   P a r t i c l e   n a m e  
 -   ` p l a q u e t t e ` :   P r i m e   l o o p  
 -   ` m _ p h y s _ M e V ` :   P h y s i c a l   m a s s   ( M e V )  
 -   ` p d g _ m a s s _ M e V ` :   P D G   t a r g e t   ( M e V )  
 -   ` e r r o r _ p c t ` :   P r e d i c t i o n   e r r o r   ( % )  
 -   ` m _ g e o m _ M e V ` :   G e o m e t r i c   m a s s   c o m p o n e n t  
 -   ` m _ v a c _ M e V ` :   V a c u u m   s c r e e n i n g   c o m p o n e n t  
 -   ` s c r e e n i n g _ p c t ` :   S c r e e n i n g   p e r c e n t a g e  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 r e s u l t   =   p h y s i c a l _ m a s s ( " m u " )  
 p r i n t ( f " M u o n :   { r e s u l t [ ' m _ p h y s _ M e V ' ] : . 3 f }   M e V " )  
 p r i n t ( f " G e o m e t r i c :   { r e s u l t [ ' m _ g e o m _ M e V ' ] : . 3 f }   M e V " )  
 p r i n t ( f " S c r e e n i n g :   { r e s u l t [ ' s c r e e n i n g _ p c t ' ] : . 1 f } % " )  
 ` ` `  
  
 - - -  
  
 # #   K o i d e   E q u a t i o n   U t i l i t i e s  
  
 # # #   O v e r v i e w  
  
 T o o l s   f o r   a n a l y z i n g   t h e   K o i d e   f o r m u l a   a n d   i t s   g e o m e t r i c   i n t e r p r e t a t i o n s .  
  
 ` ` ` p y t h o n  
 f r o m   l i g h t _ t h e o r y _ r e a l m   i m p o r t   (  
         k o i d e _ r a t i o ,  
         c h e c k _ k o i d e _ t r i p l e t ,  
         p r e d i c t _ t h i r d _ m a s s ,  
         k o i d e _ a n g l e ,  
         d e s c a r t e s _ a n g l e ,  
 )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   k o i d e _ r a t i o  
  
 ` ` ` p y t h o n  
 d e f   k o i d e _ r a t i o ( m 1 :   f l o a t ,   m 2 :   f l o a t ,   m 3 :   f l o a t )   - >   f l o a t  
 ` ` `  
  
 C a l c u l a t e   t h e   K o i d e   r a t i o   Q   f o r   t h r e e   m a s s e s .  
  
 * * F o r m u l a * * :  
 $ $ Q   =   \ f r a c { m _ 1   +   m _ 2   +   m _ 3 } { ( \ s q r t { m _ 1 }   +   \ s q r t { m _ 2 }   +   \ s q r t { m _ 3 } ) ^ 2 } $ $  
  
 * * P a r a m e t e r s * * :  
 -   ` m 1 ,   m 2 ,   m 3 ` :   P a r t i c l e   m a s s e s   ( s a m e   u n i t s )  
  
 * * R e t u r n s * * :  
 -   ` Q `   ( f l o a t ) :   K o i d e   r a t i o   ( s h o u l d   b e   â 0 Æ  2 / 3   f o r   l e p t o n s )  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 Q   =   k o i d e _ r a t i o ( 0 . 5 1 1 ,   1 0 5 . 6 6 ,   1 7 7 6 . 8 6 )     #   e ,   Î ¼ ,   Ï   
 p r i n t ( f " Q   =   { Q : . 6 f } " )     #   Q   â 0 Æ  0 . 6 6 1  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   c h e c k _ k o i d e _ t r i p l e t  
  
 ` ` ` p y t h o n  
 d e f   c h e c k _ k o i d e _ t r i p l e t (  
         m 1 :   f l o a t ,  
         m 2 :   f l o a t ,  
         m 3 :   f l o a t ,  
         t a r g e t _ Q :   f l o a t   =   2 / 3 ,  
         t o l e r a n c e :   f l o a t   =   0 . 0 1  
 )   - >   D i c t [ s t r ,   f l o a t ]  
 ` ` `  
  
 V a l i d a t e   i f   t h r e e   m a s s e s   s a t i s f y   t h e   K o i d e   r e l a t i o n .  
  
 * * R e t u r n s * * :  
 D i c t i o n a r y   w i t h :  
 -   ` Q ` :   C a l c u l a t e d   r a t i o  
 -   ` t a r g e t ` :   T a r g e t   Q   v a l u e  
 -   ` d e v i a t i o n ` :   | Q   -   t a r g e t |  
 -   ` r e l a t i v e _ e r r o r ` :   d e v i a t i o n   /   t a r g e t  
 -   ` v a l i d ` :   T r u e   i f   w i t h i n   t o l e r a n c e  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 r e s u l t   =   c h e c k _ k o i d e _ t r i p l e t ( 0 . 5 1 1 ,   1 0 5 . 6 6 ,   1 7 7 6 . 8 6 )  
 p r i n t ( f " V a l i d :   { r e s u l t [ ' v a l i d ' ] } " )  
 p r i n t ( f " E r r o r :   { r e s u l t [ ' r e l a t i v e _ e r r o r ' ] * 1 0 0 : . 2 f } % " )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   p r e d i c t _ t h i r d _ m a s s  
  
 ` ` ` p y t h o n  
 d e f   p r e d i c t _ t h i r d _ m a s s (  
         m 1 :   f l o a t ,  
         m 2 :   f l o a t ,  
         t a r g e t _ Q :   f l o a t   =   2 / 3  
 )   - >   T u p l e [ f l o a t ,   f l o a t ]  
 ` ` `  
  
 P r e d i c t   t h e   t h i r d   m a s s   g i v e n   t w o   m a s s e s   a n d   t a r g e t   K o i d e   r a t i o .  
  
 * * P a r a m e t e r s * * :  
 -   ` m 1 ,   m 2 ` :   T w o   k n o w n   m a s s e s  
 -   ` t a r g e t _ Q ` :   T a r g e t   K o i d e   r a t i o   ( d e f a u l t :   2 / 3 )  
  
 * * R e t u r n s * * :  
 -   ` m 3 _ s o l u t i o n ` :   P r e d i c t e d   t h i r d   m a s s  
 -   ` Q _ c h e c k ` :   V e r i f i c a t i o n   t h a t   Q ( m 1 ,   m 2 ,   m 3 )   â 0 Æ  t a r g e t _ Q  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 #   P r e d i c t   t a u   m a s s   f r o m   e l e c t r o n   a n d   m u o n  
 m _ t a u ,   Q   =   p r e d i c t _ t h i r d _ m a s s ( 0 . 5 1 1 ,   1 0 5 . 6 6 )  
 p r i n t ( f " P r e d i c t e d   t a u :   { m _ t a u : . 2 f }   M e V " )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   k o i d e _ a n g l e  
  
 ` ` ` p y t h o n  
 d e f   k o i d e _ a n g l e ( m 1 :   f l o a t ,   m 2 :   f l o a t ,   m 3 :   f l o a t )   - >   D i c t [ s t r ,   f l o a t ]  
 ` ` `  
  
 C a l c u l a t e   t h e   g e o m e t r i c   a n g l e   f r o m   F o o t ' s   v e c t o r   i n t e r p r e t a t i o n .  
  
 * * T h e o r y * * :  
 I n   f l a v o r   s p a c e ,   t h e   m a s s   v e c t o r   v   =   ( â Æam â   ,   â Æam â   ,   â Æam â  ’)   m a k e s   a n   a n g l e   Î ¸   w i t h   t h e   d e m o c r a t i c   v e c t o r   u   =   ( 1 ,   1 ,   1 ) .   F o r   Q   =   2 / 3 ,   Î ¸   =   4 5 Â °   e x a c t l y !  
  
 * * R e t u r n s * * :  
 D i c t i o n a r y   w i t h :  
 -   ` t h e t a _ r a d ` :   A n g l e   i n   r a d i a n s  
 -   ` t h e t a _ d e g ` :   A n g l e   i n   d e g r e e s  
 -   ` c o s _ t h e t a ` :   C o s i n e   o f   t h e   a n g l e  
 -   ` Q ` :   K o i d e   r a t i o  
 -   ` i s _ 4 5 _ d e g ` :   T r u e   i f   a n g l e   â 0 Æ  4 5 Â °   ( w i t h i n   1 Â ° )  
 -   ` d e v i a t i o n _ d e g ` :   D i s t a n c e   f r o m   4 5 Â °  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 r e s u l t   =   k o i d e _ a n g l e ( 0 . 5 1 1 ,   1 0 5 . 6 6 ,   1 7 7 6 . 8 6 )  
 p r i n t ( f " A n g l e :   { r e s u l t [ ' t h e t a _ d e g ' ] : . 2 f } Â ° " )  
 p r i n t ( f " I s   4 5 Â ° ?   { r e s u l t [ ' i s _ 4 5 _ d e g ' ] } " )  
 ` ` `  
  
 - - -  
  
 # # #   F u n c t i o n :   d e s c a r t e s _ a n g l e  
  
 ` ` ` p y t h o n  
 d e f   d e s c a r t e s _ a n g l e ( Q :   f l o a t )   - >   D i c t [ s t r ,   f l o a t ]  
 ` ` `  
  
 C a l c u l a t e   t h e   D e s c a r t e s   c i r c l e   i n t e r s e c t i o n   a n g l e   f r o m   K o i d e   r a t i o .  
  
 * * F o r m u l a * * :   Ï     =   a r c c o s ( Q )  
  
 F o r   Q   =   2 / 3 ,   Ï     â 0 Æ  4 8 . 1 9 Â ° .  
  
 * * R e t u r n s * * :  
 D i c t i o n a r y   w i t h :  
 -   ` p h i _ r a d ` :   D e s c a r t e s   a n g l e   i n   r a d i a n s  
 -   ` p h i _ d e g ` :   D e s c a r t e s   a n g l e   i n   d e g r e e s  
 -   ` Q ` :   I n p u t   K o i d e   r a t i o  
  
 * * E x a m p l e * * :  
 ` ` ` p y t h o n  
 r e s u l t   =   d e s c a r t e s _ a n g l e ( 2 / 3 )  
 p r i n t ( f " D e s c a r t e s   a n g l e :   { r e s u l t [ ' p h i _ d e g ' ] : . 2 f } Â ° " )  
 ` ` `  
  
 - - -  
 