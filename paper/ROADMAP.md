# Koide Paper Project Roadmap

## Project Overview
**Title:** Koide from Prime Plaquettes: Geometric Fixed Points in an Information Manifold  
**Location:** `\\wsl.localhost\Ubuntu\home\jdimi\projects\to_observe_proof\Light_Theory_Realm\paper`  
**Status:** LaTeX skeleton created, data identified, ready for content development

---

## File Structure

```
Light_Theory_Realm/paper/
├── koide_from_prime_plaquettes.tex  ✅ Created (skeleton)
├── koide_data_summary.md             ✅ Created (data inventory)
├── references.bib                    ✅ Exists (from geometrodynamic_probes)
├── figures/                          📁 Ready for plots
│   ├── koide_q_distribution.pdf     ⏳ To generate
│   ├── parameter_space_flow.pdf     ⏳ To generate
│   ├── geometric_diagnostics.pdf    ⏳ To generate
│   ├── quark_extensions.pdf         ⏳ To generate
│   └── ckm_matrix.pdf                ⏳ To generate
└── tables/                           📁 To create
    ├── lepton_masses.tex             ⏳ To generate
    ├── quark_masses.tex              ⏳ To generate
    ├── koide_statistics.tex          ⏳ To generate
    └── plaquette_assignments.tex     ⏳ To generate
```

---

## Data Sources Mapped

### From THRML_Geometry
**Logs:** `THRML_Geometry/thrml_geom/pocket_u/logs/` (75 files)
- ✅ Wilson loop analysis: `koide_geometric_analysis_results.json`
- ✅ Flavor space angles: `flavor_space_angle_results.json`
- ✅ Latest Koide run: `koide_training_results_20251201_095459.txt`
- ⏳ Need to extract: Koide-blind runs (lambda_koide=0)

**Experiments:** `THRML_Geometry/thrml_geom/pocket_u/experiments/`
- ✅ Analysis scripts identified
- ✅ CKM calculations available
- ✅ Quark Koide extensions documented

---

## Section Development Plan

### Phase 1: Introduction & Background (Sections 1-2)
**Estimated:** 2-3 pages  
**Content:**
- [ ] Define Koide formula with experimental value
- [ ] Cite original Koide (1981) paper
- [ ] Brief history of Koide explanations
- [ ] Introduce Light Theory framework
- [ ] State paper scope and claims

**Key Citations Needed:**
- Koide (1981) original paper
- PDG lepton mass values
- Recent Koide geometric interpretations (2024-2025)

---

### Phase 2: Light Theory Framework (Section 3)
**Estimated:** 3-4 pages  
**Content:**
- [ ] QGT definition (Fisher + Berry)
- [ ] Sixth-order operator L₆ = (∇³)†∇³
- [ ] Prime plaquettes as Wilson loops
- [ ] Reeb flow and contact geometry

**Cross-Reference:**
- Light Mechanics (technical details)
- Geometrodynamic Probes (experimental context)

---

### Phase 3: Prime-Plaquette Mass Model (Section 4)
**Estimated:** 4-5 pages  
**Content:**
- [ ] Plaquette assignments table
  - e: [2,3,5,7]
  - μ: [5,7,11,13]
  - τ: [3,11,13,47]
- [ ] Mass operator construction
- [ ] Shared parameters (Λ, α, ζ_vac)
- [ ] Koide Q as derived observable

**Tables:**
- Table 1: Plaquette assignments for leptons
- Table 2: Shared parameters and scales

---

### Phase 4: Prime-Plaquette Koide Geometry (Section 5)
**Estimated:** 3-4 pages  
**Content:**
- [ ] Analysis of runs with validated plaquettes (λ_K = 0.01)
- [ ] Rigidity of Q across hyperparameter noise
- [ ] Mass prediction accuracy
- [ ] Geometric diagnostics (Fisher/Berry/Reeb)

**Figures:**
- Figure 1: Q stability histogram (all runs)
- Figure 2: Mass errors vs Q value

**Data Needed:**
- Compile stats for all 63 runs
- Demonstrate Q convergence to ~0.667

---

### Phase 5: Plaquette Robustness and Stability (Section 6)
**Estimated:** 4-5 pages  
**Content:**
- [ ] Comparison with "wrong" plaquette assignments
- [ ] Sensitivity analysis
- [ ] Why these specific primes?
- [ ] Geometric signature of the "Koide leaf"

**Figures:**
- Figure 3: Q values for alternative plaquette triples
- Figure 4: Mass errors for alternative plaquettes
- Figure 5: Geometric diagnostics comparison

**Data Available:**
- `koide_formula_test.py` results
- Need to run: Comparative scan of alternative plaquettes

---

### Phase 6: Quark Extensions (Section 7)
**Estimated:** 3-4 pages  
**Content:**
- [ ] Quark plaquette assignments
- [ ] Mass predictions for u,c,t and d,s,b
- [ ] Koide-like combinations
  - Heavy quarks (c,t,b): Q_heavy
  - Light quarks (u,d,s): Q_light
- [ ] CKM mixing angles

**Tables:**
- Table 3: Quark masses (predicted vs PDG)
- Table 4: Quark Koide ratios

**Figures:**
- Figure 6: Quark Koide scatter plot
- Figure 7: CKM matrix comparison

**Data Available:**
- `ckm_results.json`
- Quark mass predictions
- Cabibbo angle: 12.34° (predicted) vs 13.04° (experimental)

---

### Phase 7: Limitations (Section 8)
**Estimated:** 2-3 pages  
**Content:**
- [ ] Plaquette choice sensitivity
- [ ] Regularization issues
- [ ] Neutrino mass failures
- [ ] Why this is a toy model

**Honest Assessment:**
- Neutrino predictions are wrong (MeV scale vs eV)
- Light quark Koide fails
- Curvature-mass correlation weak
- Prime number justification speculative

---

### Phase 8: Discussion (Section 9)
**Estimated:** 3-4 pages  
**Content:**
- [ ] Summary of what works
- [ ] Koide as geometric fixed point
- [ ] Holographic interpretation
- [ ] Comparison with other models

**Key Points:**
- Q ≈ 2/3 is a *region*, not a point
- Reeb flow + curvature signature
- Not claiming to solve Koide, but showing geometric mechanism

---

### Phase 9: Outlook (Section 10)
**Estimated:** 2 pages  
**Content:**
- [ ] Better parameter scans
- [ ] Alternative plaquette schemes
- [ ] ML applications
- [ ] Light Theory roadmap

---

## Bibliography Requirements

### Essential Citations
1. **Koide Original:** Y. Koide, Phys. Lett. B (1981)
2. **PDG:** Particle Data Group (2024)
3. **Light Mechanics:** This work (reference to foundations)
4. **Geometrodynamic Probes:** This work (experimental paper)

### Recent Koide Work (2024-2025)
5. Phase coherence / topological soliton (preprints.org 2025)
6. Inverse Compton radii (reddit/physics 2024)
7. Descartes circles (arXiv)

### QGT & Information Geometry
8. QGT in condensed matter (recent reviews)
9. Natural gradient descent (ML applications)
10. Berry curvature (topological materials)

### Geometric Mass Generation
11. Symmetric Mass Generation (2024 workshop)
12. Geometric torsion (Pinčák et al. 2025)

---

## Timeline Estimate

### Week 1-2: Data Extraction & Figure Generation
- [ ] Extract Koide-blind run data
- [ ] Generate all figures (6-7 total)
- [ ] Create all tables (4-5 total)
- [ ] Compile statistics

### Week 3-4: Content Writing
- [ ] Sections 1-3 (Introduction, Background, Framework)
- [ ] Sections 4-6 (Model, Koide-blind, Koide-influenced)

### Week 5-6: Experimental Sections & Polish
- [ ] Section 7 (Quark extensions)
- [ ] Sections 8-10 (Limitations, Discussion, Outlook)
- [ ] Bibliography completion
- [ ] Proofread and polish

### Week 7: Finalization
- [ ] Generate PDF
- [ ] Internal review
- [ ] Prepare for arXiv submission

**Total Estimated Time:** 6-7 weeks

---

## Success Criteria

### Minimum Viable Paper
- ✅ LaTeX compiles without errors
- ✅ All sections have content (no TODOs)
- ✅ At least 5 figures
- ✅ At least 3 tables
- ✅ Bibliography complete (≥15 references)
- ✅ Honest about limitations
- ✅ Reproducible (data/code available)

### Stretch Goals
- 🎯 Comparison with other Koide models
- 🎯 Holographic bulk interpretation
- 🎯 ML applications section
- 🎯 Supplementary material with full data

---

## Next Immediate Steps

1. **Create figures directory**
   ```bash
   mkdir -p Light_Theory_Realm/paper/figures
   mkdir -p Light_Theory_Realm/paper/tables
   ```

2. **Extract Koide-blind data**
   - Scan all 75 log files
   - Filter for lambda_koide=0
   - Compile Q statistics

3. **Generate first figure**
   - Koide Q distribution histogram
   - Use matplotlib/seaborn
   - Save as PDF for LaTeX

4. **Start Section 1**
   - Write introduction
   - Define Koide formula
   - State paper scope

---

**Status:** Ready to begin content development  
**Next Action:** Extract Koide-blind data and generate Figure 1
