# P0 algebra corrections

Base: 2997a902716c2c222acf3eb73626a81dae0e2ffe. Isolated correction branch;
original working checkout and prior audit evidence remain unchanged.

Reversion is now the complex-linear grade involution with sign
(-1)^(r(r-1)/2), not matrix transpose or Hermitian adjoint. General inversion
uses the faithful 4x4 complex matrix representation, with JIT-compatible
linear algebra. `inverse_checked` is a host-only finite/conditioning guard;
the low-level JIT inverse intentionally returns nonfinite results on singular
input. Near-singular numerical conditioning still requires caller policy.

AlgebraSpec records signature, scalar field, carrier size and ordered basis.
StatisticalTensors keeps the real positive metric and full symmetric cubic,
chart and source identity. The indexed map T_ab c gamma^c retains two free
indices and is injective for independent vector matrices. It is not a
Clifford trivector or a replacement for the original tensor. Frame transport
remains the caller's responsibility; arrays prevent ordinary mutation but
this is not an adversarial security container.

`normalized_geometry` explicitly distinguishes FS=Re Q, QFI=4 Re Q and
F=-2 s Im Q for A=s i<psi|dpsi>. Legacy methods preserve their previous
(Re Q, Im Q) outputs. No other legacy particle or gauge claims are validated.
The engine exposes its actual complex dtype to adapters. SciPy, already
imported by the package's mass module, is now a declared direct dependency.

Verification (CPU, Python 3.13.2):

```
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=. /home/jdimi/miniconda3/bin/python -m pytest tests/p0 tests/test_grade_projection.py tests/test_wedge_product.py tests/test_qgt.py -q
```

11 tests passed. The four new tests cover all 16 blade involutions, 256
reversion product identities, general mixed-multivector inverses and JIT,
singular rejection, a trace-free cubic erased by naive contraction but
recovered by the indexed interface, and analytic two-parameter QGT factors.
This is scoped verification, not a full repository or physical validation.
Review: same-agent code review; no independent reviewer is claimed.
