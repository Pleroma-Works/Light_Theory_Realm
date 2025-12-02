# Notebook Series – Light Theory Realm (v0.1.2)

A 10-part, hands-on path from first spinor to quantum free energy. Levels progress from beginner to advanced; each notebook is runnable standalone and pairs with CLI/Python APIs.

## Index

1. **01_intro_hello_light_theory_realm.ipynb** — Beginner  
   First QGT + Standard Model snapshot; quick tour of Pocket_U Lite.  
   Provenance: `examples/pocket_u_mass_table.py`, QUICKSTART snippets.

2. **02_clifford_algebra_101.ipynb** — Beginner  
   Wedge products, grade projections, multivectors; intuitive geometric algebra demo.  
   Provenance: `examples/wedge_product_example.py`, `grade_projection_example.py`.

3. **03_basic_qgt_single_parameter.ipynb** — Beginner  
   Fisher & Berry along a 1D curve; rotating spinor example.  
   Provenance: `examples/basic_qgt.py`.

4. **04_standard_model_mass_explorer.ipynb** — Beginner  
   Pocket_U Lite mass table, screening, and plots; geometric fingerprints overview.  
   Provenance: `examples/pocket_u_mass_table.py`.

5. **05_koide_playground.ipynb** — Beginner  
   Koide ratio, Foot angle, interactive lepton tuner; predict third mass.  
   Provenance: `examples/koide_analysis.py`, `pocket_u_lite/koide.py`.

6. **06_geometric_fingerprints_particles.ipynb** — Intermediate  
   ξ, Fisher trace, Berry norm, KK metrics; electron vs tau comparison.  
   Provenance: `examples/pocket_u_geometry_showcase.py`, `pocket_u_lite/sm_table.py`.

7. **07_prime_plaquettes_topological_mass.ipynb** — Intermediate  
   Prime sequences → Wilson loop phase; topological vs physical mass; prime sweep sensitivity.  
   Provenance: `experiments/prime_gauge/prime_plaquette.py`, `mass_model.py`.

8. **08_dark_energy_reeb_flow.ipynb** — Intermediate  
   Reeb resonance density ξ across Matter/Intermediate/Vacuum prime sets; ratio to vacuum.  
   Provenance: `experiments/prime_gauge/reeb_flow.py`.

9. **09_u1_clock_playground.ipynb** — Intermediate  
   U(1) clock model sampling (THRML) with phase visualizations; fallback to random phases if THRML absent.  
   Provenance: `models/classical_clock.py`.

10. **10_quantum_qhbm_vqt_tutorial.ipynb** — Advanced  
    Quantum QHBM/VQT free-energy minimization; PennyLane path with surrogate fallback if backend missing.  
    Provenance: `models/quantum_qhbm.py`.

## How to run

```bash
pip install light-theory-realm
# Optional extras
pip install light-theory-realm[pennylane]   # quantum track
# THRML needs source install if you want native clock sampling

jupyter notebook notebooks/01_intro_hello_light_theory_realm.ipynb
```

Each notebook resets imports and includes small robustness patches (e.g., `compute_reeb_diagnostics` idempotent monkeypatch) to keep reruns safe. For advanced backends (PennyLane/THRML), comment out the fallbacks to use full functionality.

## Release scope (v0.1.2)
- Adds this notebook series + README/GETTING_STARTED links
- Small robustness fixes (tuple primes in Reeb flow)
- No breaking API changes
