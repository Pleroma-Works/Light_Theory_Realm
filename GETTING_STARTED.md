# Getting Started with Light Theory Realm

Welcome! This guide takes you from zero to running your first geometry computation in 30 minutes.

---

## ⏱️ 5-Minute Quickstart

### Install
```bash
pip install light-theory-realm
```

### Run Your First Command
```bash
# View Standard Model masses predicted by Pocket_U Lite
light-realm sm-table

# Analyze the mysterious Koide relation
light-realm koide 0.511 105.66 1776.86
```

**That's it!** You're using a geometry engine that measures particle masses using prime numbers and Clifford algebra.

---

## ⏱️ 15-Minute Deep Dive

### The 4-Layer Stack

Light Theory Realm has 4 layers. Understand each in ~3 minutes:

#### Layer 1: Algebra
```python
from light_theory_realm import CliffordEngine

engine = CliffordEngine(seed=42)

# Create a random spinor (4D complex vector)
import jax
key = jax.random.PRNGKey(0)
psi = engine.random_spinor(key)

# Extract pure grades (scalar, vector, bivector parts)
scalar = engine.project_grade(psi, grade=0)
vector = engine.project_grade(psi, grade=1)

# Compute wedge product (antisymmetric outer product)
v1 = engine.random_spinor(jax.random.PRNGKey(1))
v2 = engine.random_spinor(jax.random.PRNGKey(2))
wedge = engine.wedge_product(v1, v2)
print(f"Wedge satisfies v ∧ v = 0: {engine.wedge_product(v1, v1)}")
```

**What it measures:** Pure algebraic structure (no geometry yet)

#### Layer 2: Geometry
```python
from light_theory_realm import CliffordQGT

qgt = CliffordQGT(engine)

# Create a state and its derivative
psi = engine.random_spinor(jax.random.PRNGKey(0))
d_psi = engine.random_spinor(jax.random.PRNGKey(1)).reshape(-1, 1)

# Measure geometry: Fisher (metric) + Berry (curvature)
fisher, berry = qgt.compute_full_qgt(psi, d_psi)

print(f"Fisher metric shape: {fisher.shape}")
print(f"Fisher (info distance): {float(fisher[0,0]):.4f}")
print(f"Berry curvature (twist): {float(berry[0,0]):.4f}")
```

**What it measures:** How systems change (Fisher) and twist (Berry)

#### Layer 3: Theory
```python
# This layer bridges geometry to physics
# (Implementation details in experiments/)

# It computes:
# - 5D Kaluza-Klein metric (unifies Fisher + Berry)
# - Reeb flow (dark energy ξ + time R)

# You typically don't use this directly
# → Use Layer 4 (experiments) instead
```

**What it measures:** Unified geometry + physical interpretation

#### Layer 4: Experiments
```python
from light_theory_realm import print_sm_table, get_particle_profile

# The easiest layer - just use the results!

# View all 9 fermion masses (with predictions)
print_sm_table()

# Get full profile for a particle
profile = get_particle_profile("e")
print(f"Electron mass: {profile['m_phys_MeV']:.3f} MeV")
print(f"Error vs PDG: {profile['error_pct']:.2f}%")
```

**What it measures:** Real particle masses and geometric fingerprints

---

## ⏱️ 30-Minute Hands-On

### Modify Pocket_U and Watch Geometry Change

Let's create a custom particle and see its geometry:

```python
import jax
import jax.numpy as jnp
from light_theory_realm import CliffordEngine, CliffordQGT

# Step 1: Create a simple "pseudo-particle" from a custom plaquette
custom_primes = [2, 5, 11, 13]  # Random prime sequence

# Step 2: Build a state trajectory around it
engine = CliffordEngine(seed=sum(custom_primes))
key = jax.random.PRNGKey(sum(custom_primes))
psi_0 = engine.random_spinor(key)

# Step 3: Create a rotation trajectory
thetas = jnp.linspace(0, 2*jnp.pi, 16)

bivector = engine.wedge_product(engine.gammas[0], engine.gammas[1])
bivector = bivector * jnp.log(custom_primes[1] / custom_primes[0])

def psi_at_theta(theta):
    R = jnp.eye(4, dtype=jnp.complex128) + 1j * theta * bivector
    psi = R @ psi_0
    return jnp.squeeze(psi / jnp.linalg.norm(psi))

psi_trajectory = jax.vmap(psi_at_theta)(thetas)

# Step 4: Compute geometry at the midpoint
mid_idx = len(thetas) // 2
psi_mid = psi_trajectory[mid_idx]
psi_prev = psi_trajectory[mid_idx - 1]
psi_next = psi_trajectory[mid_idx + 1]

# Finite-difference derivative
dpsi = (psi_next - psi_prev) / 2
jac = dpsi.reshape((-1, 1))

# Step 5: Measure the geometry
qgt = CliffordQGT(engine)
fisher, berry = qgt.compute_full_qgt(psi_mid, jac)

print(f"Custom particle geometry:")
print(f"  Fisher trace (info distance): {float(jnp.trace(fisher)):.4f}")
print(f"  Berry norm (twist): {float(jnp.linalg.norm(berry)):.4f}")

# Step 6: Modify the primes and see how geometry changes
print("\nGeometry scales with prime separation:")
for sep in [2, 5, 10]:
    custom_primes = [2, 2+sep, 11, 13]
    angle_scale = jnp.log((2+sep) / 2.0)
    bivector_scaled = bivector * (angle_scale / jnp.log(custom_primes[1]/custom_primes[0]))
    # ... recompute and show different geometry
```

**Insight:** Changing primes changes the geometric fingerprint!

---

## 📚 What's Next?

### For Quick Demos
- Check out `examples/` folder
- Run: `python examples/koide_analysis.py`
- Run: `python examples/pocket_u_mass_table.py`

### For Deep Understanding
- Read: [Light_Realm_Easy_Mode.md](Light_Realm_Easy_Mode.md)
- Read: [Light_Realm_Multidimensional.md](Light_Realm_Multidimensional.md)
- Read: [ARCHITECTURE.md](ARCHITECTURE.md)

### For Contributing
- Read: [CONTRIBUTING.md](CONTRIBUTING.md)
- Check: [TODO_IMPLEMENTATION.md](TODO_IMPLEMENTATION.md) for open tasks

### For Reference
- [API_REFERENCE.md](API_REFERENCE.md) - All public functions
- [INSTALL.md](INSTALL.md) - Installation options
- [QUICKSTART.md](QUICKSTART.md) - Quick reference

---

## ❓ Common Questions

### Q: Do I need to understand physics?
**A:** No! Think of it as:
- Layer 1-2: Pure math (geometry on manifolds)
- Layer 3-4: Physics flavor (but math works independently)

### Q: What if I get an error?
**A:** Check:
1. Python 3.10+ installed? `python --version`
2. JAX installed? `python -c "import jax; print(jax.__version__)"`
3. Light Theory Realm installed? `pip install light-theory-realm`

### Q: How accurate are the mass predictions?
**A:** ~2.3% average error on 9 fermions. That's remarkable for a toy model!

### Q: Can I extend Pocket_U with new particles?
**A:** Yes! See `CONTRIBUTING.md` → "Adding a Feature"

### Q: What's the profile command error?
**A:** Known issue in v0.1.1 (geometry module initialization).
Workaround: Use `light-realm sm-table` for now.
Fix coming in v0.2.0.

---

## 🎯 Challenge: Build a Custom Experiment

Once you're comfortable:

1. **Create** a new prime plaquette sequence
2. **Compute** its geometry (mass + Fisher + Berry)
3. **Compare** to other particles
4. **Hypothesize** about patterns
5. **Share** your findings!

---

## 🔗 Resources

- **GitHub:** https://github.com/Pleroma-Works/Light_Theory_Realm
- **Issues:** Report bugs there
- **Discussions:** Start a discussion for questions

---

## ✨ You're ready!

You now understand:
- ✅ The 4-layer stack
- ✅ How to run the CLI
- ✅ How to use the Python API
- ✅ How to modify & explore

**Next:** Try the examples, then read the deep documentation.

Happy exploring! 🚀
