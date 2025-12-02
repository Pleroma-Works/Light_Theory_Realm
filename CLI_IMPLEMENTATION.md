# Light Theory Realm – CLI Implementation Notes

**Date:** 2025-11-30  
**Status:** Complete and Verified  
**Version:** v0.1.1

---

## Summary

Added a professional command-line interface (CLI) to Light Theory Realm for non-programmers and researchers to explore the framework without writing code.

---

## Files Created/Modified

### Created
- `light_theory_realm/cli.py` (250 lines)
  - Main CLI module with 5 subcommands
  - Proper error handling and help text
  - Uses only `argparse` (built-in, no new dependencies)

### Modified
- `pyproject.toml` (+3 lines)
  - Added `[project.scripts]` entry: `light-realm = "light_theory_realm.cli:main"`
  
- `QUICKSTART.md` (+20 lines)
  - Added "Command-Line Interface (CLI)" section
  - Included example usage for all 4 working commands

---

## CLI Subcommands

### ✅ Working Commands

**1. `light-realm sm-table`**
```bash
light-realm sm-table
```
- Prints all 9 Standard Model fermions
- Shows PDG masses vs Pocket_U Lite predictions
- Shows error percentages and plaquettes
- Use case: Quick overview of model accuracy

**2. `light-realm koide <m1> <m2> <m3>`**
```bash
light-realm koide 0.511 105.66 1776.86
```
- Computes Koide ratio and Foot angle
- Three masses in MeV (PDG leptons shown)
- Output: Q ≈ 0.6667, θ ≈ 45°
- Use case: Validate Koide hypothesis

**3. `light-realm koide-predict <m1> <m2>`**
```bash
light-realm koide-predict 0.511 105.66
```
- Predicts third lepton mass using Koide
- Predicts τ ≈ 1776.99 MeV (PDG: 1776.86)
- Use case: Test predictive power

**4. `light-realm koide-from-pocket`**
```bash
light-realm koide-from-pocket
```
- Uses Pocket_U Lite's own predicted masses
- Computes Koide Q and angle automatically
- Use case: See Koide emerge from geometry

### ⚠️ Temporarily Disabled

**5. `light-realm profile <particle>`**
- Reason: JAX tracer issue in geometry.py (known JAX limitation with vmap + conditionals)
- Timeline: Will fix in v0.2.0
- Note: User sees clear message explaining why

---

## Installation & Usage

After `pip install -e .`, the CLI is available globally:

```bash
light-realm --help
light-realm sm-table
light-realm koide 0.511 105.66 1776.86
```

---

## Design Decisions

### Why This CLI Works
✅ **Focused scope:** Only "showcase" features (Pocket_U Lite + Koide)
✅ **No steep learning curve:** Simple arguments, clear output
✅ **Minimal dependencies:** Uses only `argparse` (built-in)
✅ **Professional UX:** Proper help text, error handling
✅ **Extensible:** Easy to add new subcommands

### What We Intentionally Skipped
- Low-level engine controls (too technical for CLI)
- Complex geometric output (better in Python notebooks)
- Particle creation (Pocket_U is read-only by design)

### Architecture
```
light_theory_realm/cli.py
├── build_parser()          → Creates ArgumentParser with all subcommands
├── cmd_sm_table()          → Handler for 'sm-table'
├── cmd_koide()             → Handler for 'koide'
├── cmd_koide_predict()     → Handler for 'koide-predict'
├── cmd_koide_from_pocket() → Handler for 'koide-from-pocket'
├── cmd_profile()           → Disabled (shows helpful message)
└── main()                  → Entry point called by console_scripts
```

---

## Known Issues & Workarounds

### Issue: `profile` Command Disabled
**Root Cause:** JAX tracer bug in geometry.py  
```python
# In geometry.py:107, inside vmap:
if i == 0:  # ← Cannot use Python control flow on traced values
    ...
```

**Workaround:** Use Python directly for full profiles (see QUICKSTART.md)  
**Fix:** Refactor with `jax.lax.cond` for v0.2.0

### Issue: GPU Warnings
JAX prints informational warnings about GPU drivers on startup. This is normal and not an error.

---

## Test Verification

All tests pass:
```bash
$ cd /home/jdimi/projects/to_observe_proof/Light_Theory_Realm
$ python -m pytest tests/ -v
# Result: 7/7 PASSED ✅
```

CLI commands verified:
```bash
$ light-realm --help                               ✅
$ light-realm sm-table                             ✅
$ light-realm koide 0.511 105.66 1776.86          ✅
$ light-realm koide-predict 0.511 105.66          ✅
$ light-realm koide-from-pocket                   ✅
```

---

## Example Output

### sm-table
```
Pocket_U Lite: Standard Model Mass Spectrum
================================================================================
Particle      PDG (MeV)    Predicted (MeV)    Error (%)   Plaquette
--------------------------------------------------------------------------------
e                 0.511              0.511        -0.0%   [2, 3, 5, 7]
mu              105.658            105.241        -0.4%   [2, 17, 23, 7]
tau            1776.860           1776.860         0.0%   [3, 11, 13, 47]
...
```

### koide
```
Koide Analysis
============================================================
  Masses          : 0.511, 105.66, 1776.86 MeV
  Q               : 0.66665909
  Foot angle      : 44.999674°
  θ (radians)     : 0.78539248
```

### koide-predict
```
Koide Prediction
============================================================
  Input masses    : 0.511, 105.66 MeV
  Predicted m3    : 1776.99 MeV
  Q               : 0.66666667
  Foot angle      : 45.000000°
```

---

## Future Enhancements (v0.2.0+)

1. **Fix `profile` command**
   - Refactor geometry.py to use `jax.lax.cond`
   - Enable full particle profile output

2. **Add `qgt-demo` command**
   - Simple QGT calculation from CLI
   - Show Fisher and Berry for example state

3. **Add `help-topic` command**
   - Educational explanations
   - "What is Koide?", "What is Berry curvature?", etc.

4. **Add batch processing**
   - Accept mass values from file
   - Output CSV with results

---

## Distribution Notes

✅ **PyPI Ready:** CLI automatically included  
✅ **GitHub Release:** Include in release notes  
✅ **Documentation:** CLI section added to QUICKSTART.md  
✅ **Examples:** Shown in multiple doc sections  

---

## Performance

- **Startup time:** ~2-3 seconds (JAX initialization)
- **sm-table:** <1 second after startup
- **koide analysis:** <0.1 seconds
- **Memory:** ~150-200 MB (JAX/JAX lib baseline)

---

## For Contributors

To add a new CLI subcommand:

1. Create handler function: `def cmd_mynewcommand(args) -> int:`
2. Add subparser: `p = subparsers.add_parser('my-command', help='...')`
3. Add handler: `p.set_defaults(func=cmd_mynewcommand)`
4. Test: `light-realm my-command --help`

Example:
```python
def cmd_demo(args: argparse.Namespace) -> int:
    print("Hello from my command!")
    return 0

p = subparsers.add_parser("demo", help="Demo command")
p.set_defaults(func=cmd_demo)
```

---

## Conclusion

The CLI adds a professional polish layer to Light Theory Realm, making it accessible to non-programmers while maintaining clean code and minimal dependencies. It successfully showcases the core "showroom" features (Pocket_U + Koide) and provides a foundation for future enhancements.

**Status:** ✅ Complete and Production-Ready

---

*Created: 2025-11-30*  
*Author: Dimitry Jean-Noel II*  
*Project: Light Theory Realm (v0.1.1)*
