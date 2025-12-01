# Koide Paper Data Summary

## Data Sources

### THRML_Geometry Logs (75 files)
Location: `\\wsl.localhost\Ubuntu\home\jdimi\projects\to_observe_proof\THRML_Geometry\thrml_geom\pocket_u\logs`

#### Key Files:
1. **koide_geometric_analysis_results.json** - Wilson loop analysis for 3 generations
   - Generation 1 (e): plaquette [2,3,5,7,2], Wilson trace=2.056
   - Generation 2 (μ): plaquette [5,7,11,13,5], Wilson trace=1.775
   - Generation 3 (τ): plaquette [3,11,13,47,3], Wilson trace=-0.777

2. **flavor_space_angle_results.json** - Koide angle analysis
   - Koide ratio Q = 0.661 (target 0.667)
   - Flavor space angle θ = 44.76° (predicted 45°)
   - Error: 9.98° (18.2%)

3. **koide_training_results_20251201_095459.txt** - Latest Koide-influenced run
   - 100 epochs with lambda_koide ramping 0.01 → 0.10
   - Lepton masses: e=0.51, μ=105.24, τ=1776.86 MeV
   - Errors: e=0.0%, μ=0.4%, τ=0.0%
   - CKM Cabibbo angle: 12.34° (experimental ~13.04°)

### Experimental Scripts
Location: `\\wsl.localhost\Ubuntu\home\jdimi\projects\to_observe_proof\THRML_Geometry\thrml_geom\pocket_u\experiments`

#### Relevant Scripts:
- `koide_geometric_analysis.py` - Wilson loop + flavor space analysis
- `flavor_space_angle_analysis.py` - Koide angle verification
- `calculate_quark_koide.py` - Quark sector extensions
- `calculate_ckm_mixing.py` - CKM matrix calculations

## Key Results to Include

### Koide-Blind Regime
- Need to identify runs with lambda_koide = 0
- Extract Q distribution across runs
- Document geometric diagnostics (Fisher, Berry, Reeb)

### Koide-Influenced Regime
- Multiple runs with lambda_koide ∈ [0.01, 0.10]
- Track Q convergence during training
- Analyze parameter space regions where Q ≈ 2/3

### Quark Sector
- Plaquette assignments for u,c,t and d,s,b
- Koide-like combinations (heavy vs light quarks)
- CKM mixing angle results

## Figures Needed

1. **Koide Q Distribution** - Histogram of Q values from Koide-blind runs
2. **Parameter Space Flow** - Trajectory toward Q=2/3 in Koide-influenced runs
3. **Geometric Diagnostics** - Fisher/Berry/Reeb patterns vs Q value
4. **Quark Extensions** - Koide-like ratios for quark triplets
5. **CKM Matrix** - Predicted vs experimental mixing angles

## Tables Needed

1. **Lepton Masses** - Predicted vs PDG for e, μ, τ
2. **Quark Masses** - Predicted vs PDG for all 6 flavors
3. **Koide Statistics** - Q values across different regimes
4. **Plaquette Assignments** - Prime sequences for all fermions
5. **Geometric Diagnostics** - Fisher/Berry/Reeb norms per particle

## Next Steps

1. Extract Koide-blind run data (lambda_koide=0)
2. Compile Koide-influenced run statistics
3. Generate publication-quality figures
4. Create result tables
5. Write experimental sections
