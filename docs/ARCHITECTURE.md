# Architecture – Light Theory Realm

A deep dive into the design, structure, and philosophy of Light Theory Realm.

---

## Philosophy

**One sentence:** Treat models and particles as geometric shapes living on curved manifolds.

Light Theory Realm is built on the premise that:
1. Information-processing systems (AI models, quantum states, particle physics) aren't just functions
2. They evolve along **curved mathematical manifolds**
3. Those manifolds have measurable **geometry** (metric + curvature)
4. The same mathematical language works for both AI and physics

---

## 4-Layer Architecture

```
┌─────────────────────────────────────────────┐
│  Layer 4: EXPERIMENTS                       │
│  (Pocket_U, Koide, Prime Plaquettes)        │
├─────────────────────────────────────────────┤
│  Layer 3: THEORY                            │
│  (Kaluza-Klein Uplift, Reeb Flow Dynamics)  │
├─────────────────────────────────────────────┤
│  Layer 2: GEOMETRY                          │
│  (Quantum Geometric Tensor: Fisher + Berry) │
├─────────────────────────────────────────────┤
│  Layer 1: ALGEBRA                           │
│  (Clifford Algebra Cl(1,3), Spinors)        │
└─────────────────────────────────────────────┘
```

Each layer is **usable independently** but designed to compose.

---

## Layer 1: Algebra

**File:** `light_theory_realm/engine.py`

**Purpose:** Foundational mathematical objects

**Provides:**
- Clifford Algebra Cl(1,3) in Weyl representation
- Gamma matrices (γ₀, γ₁, γ₂, γ₃)
- Spinor operations (random, inner product, etc.)
- Wedge product (antisymmetric outer product)
- Grade projection (extract scalar, vector, bivector, etc. parts)

**Key Classes:**
```python
class CliffordEngine:
    def __init__(seed: int)
    def random_spinor(key) -> (4,) complex spinor
    def wedge_product(a, b) -> outer product
    def project_grade(v, grade) -> extract grade component
    def geometric_product(a, b) -> Clifford product
```

**Design Decisions:**
- ✅ JAX-first: all operations differentiable
- ✅ Fixed seed for reproducibility
- ✅ Weyl representation (chiral basis) for physics relevance
- ✅ No external Clifford library (self-contained)

**When to extend:**
- Need new Clifford algebra (e.g., Cl(3,1) for different signature)?
- Need new spinor operations?
- → Extend `CliffordEngine`

---

## Layer 2: Geometry

**File:** `light_theory_realm/qgt.py`

**Purpose:** Measure geometric structure of manifolds

**Key Idea:**
Given a parameterized state ψ(θ), measure:
1. **Fisher metric** g_ij = ⟨∂ᵢψ|∂ⱼψ⟩ − ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩
   - How "stretched" is the manifold?
   - Information distance
2. **Berry curvature** Ω_ij = Im⟨ψ|∂ᵢψ⟩⟨∂ⱼψ|ψ⟩
   - How much "twist" when moving around?
   - Topological phase

**Key Classes:**
```python
class CliffordQGT:
    def compute_full_qgt(psi, jacobian) -> (fisher, berry)
    def inner_product(psi, phi) -> complex
```

**Inputs:**
- `psi`: State vector, shape (n,) or (n, 1)
- `jacobian`: Derivative matrix, shape (n, d) where d = # parameters

**Outputs:**
- `fisher`: (d, d) real symmetric matrix
- `berry`: (d, d) imaginary antisymmetric matrix

**Design Decisions:**
- ✅ Works with any state, any dimension
- ✅ Automatic Gram-Schmidt orthogonalization
- ✅ Numerically stable
- ✅ Fully differentiable

**When to extend:**
- Need different metric (e.g., Fubini-Study)?
- Need different curvature formula?
- → Extend or create new geometry class

---

## Layer 3: Theory

**Files:** 
- `light_theory_realm/experiments/prime_gauge/uplift.py`
- `light_theory_realm/experiments/prime_gauge/reeb_flow.py`

**Purpose:** Connect geometry to physics

### 3A: Kaluza-Klein Uplift

**Idea:** Unify 4D information geometry with a 5th dimension

```
G_5D = [ g_μν + φ²A_μA_ν    φ²A_μ ]
       [      φ²A_ν         φ²   ]
```

Where:
- `g_μν` = Fisher metric (base geometry)
- `A_μ` = Berry connection (gauge field)
- `φ` = Dilaton (extra dimension)

**Why?** Treats Fisher (metric) and Berry (curvature) as unified structure.

**Key Class:**
```python
class KaluzaKleinUplift:
    def construct_5d_metric(psi, jacobian, phi_scalar) -> (5, 5) matrix
```

### 3B: Reeb Flow Dynamics

**Idea:** Interpret dark energy and time as geometric objects

```
Dark Energy (ξ):
    ξ = resonance density of vacuum
    ~ commutator clashing of prime generators
    Range: [0, 1]
    
Time Direction (R):
    R = Reeb vector (lives in null space of Berry curvature)
    i·R·Ω = 0
```

**Key Class:**
```python
class ReebFlowDynamics:
    def compute_reeb_vector(fisher, berry) -> (d,) vector
    def compute_resonance(primes) -> float in [0,1]
```

**Why?** Offers geometric interpretation of dark energy & time flow.

---

## Layer 4: Experiments

**Directory:** `light_theory_realm/pocket_u_lite/`

**Purpose:** Demonstrate theory on real data

### 4A: Prime Plaquettes
**File:** `plaquettes.py`

Assign each SM fermion a prime sequence (plaquette):
- Electron: [2, 3, 5, 7]
- Muon: [2, 17, 23, 7]
- Tau: [3, 11, 13, 47]
- ... etc

**Why primes?**
- Basis elements (gauge generators)
- Canonical loop in SU(3)
- Encode topological structure

### 4B: Pocket_U Lite
**Files:** `mass_model.py`, `sm_table.py`, `geometry.py`

Predict fermion masses using:
```
m_phys = Λ · |M_geom + α·ζ_vac|
```

Where:
- `Λ` = energy scale
- `M_geom` = geometric mass from Wilson loop
- `ζ_vac` = vacuum screening
- `α` = coupling strength

**Accuracy:** ~2.3% average error on 9 fermions

### 4C: Koide Utilities
**File:** `koide.py`

Mysterious relation in lepton masses:
```
Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² ≈ 2/3
```

Functions:
```python
def koide_ratio(m1, m2, m3) -> Q
def koide_angle(m1, m2, m3) -> theta (≈ 45°)
def predict_third_mass(m1, m2) -> m3 predicted
```

---

## Data Flow

```
User Input (e.g., "electron")
         ↓
Get plaquette [2, 3, 5, 7]
         ↓
Compute mass (mass_model.py)
    ├→ Wilson loop phase
    ├→ Geometric mass component
    └→ Physical mass prediction
         ↓
Compute geometry (geometry.py)
    ├→ Build state trajectory ψ(θ)
    ├→ Compute QGT (Fisher + Berry)
    ├→ Construct 5D KK metric
    └→ Compute Reeb (ξ + R)
         ↓
Merge results → Full particle profile
         ↓
Display or return to user
```

---

## Dependency Graph

```
CliffordEngine (Layer 1)
    ↓
CliffordQGT (Layer 2)
    ├→ KaluzaKleinUplift (Layer 3)
    └→ ReebFlowDynamics (Layer 3)
         ↓
Pocket_U geometry (Layer 4)
    ├→ mass_model
    ├→ koide utilities
    └→ CLI interface
```

**Key:** No upward dependencies (clean layering)

---

## Extension Points

### 1. Add a new Clifford algebra
```python
# In engine.py
class CliffordEngine:
    def __init__(self, seed, signature=(1,3)):  # Make configurable
        self.signature = signature
        # Initialize gammas based on signature
```

### 2. Add a new geometry computation
```python
# New file: experiments/my_geometry.py
class MyGeometry:
    def compute(psi, jacobian):
        # Custom geometric quantity
        pass
```

### 3. Add a new particle to Pocket_U
```python
# In pocket_u_lite/plaquettes.py
PLAQUETTES = {
    "e": [2, 3, 5, 7],
    # ... add new particle
    "new_particle": [p1, p2, p3, p4]
}
```

### 4. Add a new experiment
```python
# New directory: experiments/my_experiment/
# New class that uses Layers 1-3
```

---

## Testing Strategy

**By layer:**
- **Layer 1:** Test algebra (wedge product, grade projection, etc.)
- **Layer 2:** Test QGT computation against known states
- **Layer 3:** Test KK metric and Reeb vector numerically
- **Layer 4:** Test mass predictions against PDG data

**Integration:**
- Full pipeline: primes → mass → geometry → profile

**Coverage target:** ≥80% on core modules

---

## Performance Notes

- **Startup:** ~2-3 seconds (JAX initialization)
- **QGT computation:** <0.1 seconds per state
- **Mass prediction:** <0.01 seconds per particle
- **Full profile:** ~1 second per particle (geometry-heavy)

**Optimization opportunities:**
- Batch multiple states
- Cache frequently used computations
- GPU acceleration for large batches

---

## Known Limitations

1. **Numerical precision:** High-dimensional spinors (~D>10) may lose precision
2. **Jacobian computation:** Requires finite differences (no autodiff yet)
3. **Geometry module:** Circular import issue in global initialization (v0.2.0 fix)
4. **THRML integration:** Track B requires external installation

---

## Design Philosophy

✅ **What we chose:**
- Minimal, self-contained code
- Explicit over implicit
- JAX-first (differentiable by default)
- Layer separation (easy to understand & modify)
- Open-source friendly

❌ **What we avoided:**
- Heavy dependencies (no TensorFlow, PyTorch required)
- Hidden magic (all computations visible)
- Complex inheritance hierarchies
- Pre-computed lookups (compute on demand)

---

## Future Directions (v0.2.0+)

1. **Refactor geometry.py** to avoid circular imports
2. **Add GPU kernels** for batch operations
3. **Integrate dual-track models** (PennyLane + THRML)
4. **Extended theory** (string theory interpretation, etc.)
5. **Visualization layer** (geometry plots, manifold rendering)

---

## Questions?

See `CONTRIBUTING.md` for how to ask questions and propose extensions.
