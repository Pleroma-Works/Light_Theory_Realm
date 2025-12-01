# v0.1.0 Release Checklist

**Date:** 2025-11-30  
**Target:** v0.1.0-beta or v0.1.0-stable

---

## Pre-Release Verification

- [x] All tests passing (7/7)
- [x] No import errors
- [x] CLI commands work (4/5 verified)
- [x] Documentation links correct
- [x] README updated with Easy/Multidimensional links
- [x] Executive summary created
- [x] Examples verified (basic_qgt, pocket_u, sm-table working)
- [ ] pyproject.toml reviewed
- [ ] Version number confirmed (0.1.0)
- [ ] CHANGELOG entry added

---

## Code Quality

- [x] Core engine stable
- [x] Tests: 7/7 passing
- [x] No regressions from CLI addition
- [ ] Code coverage measured (target: ≥80%)
- [ ] Type hints on public functions (optional for beta)
- [ ] Known issues documented

---

## Documentation Status

- [x] README.md updated
- [x] INSTALL.md comprehensive
- [x] QUICKSTART.md with CLI section
- [x] API_REFERENCE.md complete
- [x] Light_Realm_Easy_Mode.md created
- [x] Light_Realm_Multidimensional.md created
- [x] EXECUTIVE_SUMMARY.md created
- [x] CLI_IMPLEMENTATION.md created
- [x] TODO_IMPLEMENTATION.md updated
- [ ] CONTRIBUTING.md created (not needed for beta)

---

## CLI Verification

- [x] `light-realm --help` works
- [x] `light-realm sm-table` works
- [x] `light-realm koide 0.511 105.66 1776.86` works
- [x] `light-realm koide-predict 0.511 105.66` works
- [x] `light-realm koide-from-pocket` works
- [ ] `light-realm profile <particle>` (disabled, documented)

---

## Package & Distribution

- [ ] pyproject.toml keywords reviewed
- [ ] Classifiers accurate
- [ ] Author info complete
- [ ] Build wheel locally: `python -m build`
- [ ] Test installation from wheel
- [ ] Verify CLI available globally after install
- [ ] Create TestPyPI account (optional)
- [ ] Do dry-run to TestPyPI (optional)

---

## Release Decision

**Current Status:** Ready for Decision

Choose one:
- [ ] Release as v0.1.0-beta (this weekend)
- [ ] Polish for 1 week, release as v0.1.0-stable
- [ ] Comprehensive release (1 week, full docs)

**Recommended:** v0.1.0-beta (get feedback from community)

---

## Post-Release

- [ ] Create GitHub release
- [ ] Write release announcement
- [ ] Share with Prime Intellect
- [ ] Collect community feedback
- [ ] Plan v0.2.0 based on feedback
