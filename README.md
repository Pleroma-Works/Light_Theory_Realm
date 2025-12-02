# Light Theory Realm

[![PyPI version](https://img.shields.io/pypi/v/light-theory-realm.svg)](https://pypi.org/project/light-theory-realm/)

> A geometry-first engine that measures the **shape of information** in particles and AI models using Clifford algebra and the Quantum Geometric Tensor (QGT).

- 🧮 **Core engine**: JAX-based Cl(1,3) Clifford algebra + Quantum Geometric Tensor
- 🧬 **Standard Model toy**: 9 fermion masses with ~2.3% average error from prime plaquettes
- 📐 **Geometry everywhere**: Fisher information, Berry curvature, Kaluza–Klein uplift, Reeb flow

---

## 📚 Start Here

### 1. Choose your depth

- **For non-physicists / engineers**  
  👉 [Foundations of Light Theory](Foundations/Foundations_of_Light_Theory.md)  
  Plain-language explanation of "geometry of information", curvature, and prime barcodes.

- **For physicists / theorists**  
  👉 [Light Mechanics](Foundations/Light_Mechanics.md)  
  Full mathematical treatment: Cl(1,3), QGT, information field equations, KK uplift, Reeb flow, geometrodynamics.

- **For decision makers / hiring managers**  
  👉 [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)  
  One-pager with results, metrics, and impact.

### 2. Or just run it

```bash
pip install light-theory-realm

# Standard Model mass table (Pocket_U Lite)
light-realm sm-table

# Koide relation check for leptons
light-realm koide 0.511 105.66 1776.86
```

📦 **PyPI**: [light-theory-realm](https://pypi.org/project/light-theory-realm/)

More first steps: [GETTING_STARTED.md](GETTING_STARTED.md)

---

## ⚙️ What this library actually does

At a high level:

1. **Algebra layer – CliffordEngine**
   Implements Cl(1,3) (gamma matrices, grades, wedge product) in JAX.

2. **Geometry layer – CliffordQGT**
   Given a state and its Jacobian, computes the full **Quantum Geometric Tensor**:

   * Fisher information (metric)
   * Berry curvature (geometric twist)

3. **Theory layer – geometry as dynamics**
   Builds higher structures on top of the QGT:

   * Information field equations (Einstein–Fisher-like)
   * 5D Kaluza–Klein uplift of the information metric
   * Contact geometry + Reeb flow (time and "dark-energy-like" terms)

4. **Experiments – Pocket_U Lite & friends**
   Uses prime plaquettes as discrete "labels" for excitations and shows that:

   * 9 Standard Model fermion masses can be reproduced at ~2–3% error
   * Each particle has a full **geometric fingerprint** (metric, curvature, KK, Reeb)

For a detailed architecture diagram and module-level docs, see:
👉 [ARCHITECTURE.md](ARCHITECTURE.md) and [API_REFERENCE.md](API_REFERENCE.md)

---

## 🔍 Minimal code examples

### Compute Fisher + Berry for your own model

```python
import jax
import jax.numpy as jnp
from light_theory_realm import CliffordEngine, CliffordQGT

engine = CliffordEngine(seed=42)
qgt = CliffordQGT(engine)

# Example: a tiny 1-parameter family of states
def get_psi(theta):
    # user-defined: returns a normalized 4-component spinor
    # ...
    return psi

theta0 = 0.1
eps = 1e-4

psi = get_psi(theta0)
dpsi = (get_psi(theta0 + eps) - get_psi(theta0 - eps)) / (2 * eps)
jac = dpsi.reshape(-1, 1)

fisher, berry = qgt.compute_full_qgt(psi, jac)
print("Fisher metric:", fisher)
print("Berry curvature:", berry)
```

### Inspect a particle's geometric fingerprint

```python
from light_theory_realm import get_particle_profile

e = get_particle_profile("e")  # electron

print(f"Mass (MeV):       {e['m_phys_MeV']:.3f}")
print(f"Error vs PDG:     {e['error_pct']:.2f}%")
print(f"Dark energy ξ:    {e['xi']:.4f}")
print(f"Fisher trace:     {e['fisher_trace']:.6f}")
print(f"Berry norm:       {e['berry_norm']:.6f}")
```

More examples in [`examples/`](examples/).

---

## 🧭 Who is this for?

* **ML / AI researchers**
  Use the QGT engine to probe the geometry of models: sharp directions, curvature, phase transitions, etc.

* **Theoretical physicists**
  Treat this as a concrete playground for geometrodynamics and holographic information geometry.

* **Independent / self-taught researchers**
  Everything is open, tested, and documented; you can run and modify any experiment locally.

---

## 🧪 Tests & quality

* 7/7 unit tests passing across algebra + geometry layers
* JAX-native, differentiable, GPU-ready
* Pocket_U Lite reproduces 9 fermion masses with ~2.3% average error

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for how to run and extend tests.

---

## 🤝 Contributing

Contributions, questions, and critiques are welcome.

* How to get started: [GETTING_STARTED.md](GETTING_STARTED.md)
* How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md)
* Open implementation tasks: [TODO_IMPLEMENTATION.md](TODO_IMPLEMENTATION.md)
* Release history: [CHANGELOG.md](CHANGELOG.md)

---

## 📜 Citation

If you use Light Theory Realm in research:

```bibtex
@software{jeannoel2025lighttheory,
  author = {Jean-Noel II, Dimitry},
  title = {Light Theory Realm: A Geometry-First Framework for Information-Geometric Physics},
  year = {2025},
  organization = {Pleroma Works, LLC},
  url = {https://github.com/Pleroma-Works/Light_Theory_Realm},
  orcid = {0009-0009-6082-8647},
  note = {Version 0.1.0}
}
```

**Related paper**: [Koide relations from prime plaquettes](paper/koide_from_prime_plaquettes.pdf) – 17-page analysis of lepton mass relations using the Pocket_U Lite framework.

**Author contact:**
Dimitry Jean-Noel II  
Pleroma Works, LLC  
djean@botwpbc.com  
djeannoelii@gmail.com  
ORCID: [0009-0009-6082-8647](https://orcid.org/0009-0009-6082-8647)

---

## License

This project is licensed under the Apache License 2.0 – see the [LICENSE](LICENSE) file for details.
