"""Explicit algebra conventions and loss-preserving statistical tensor boundary.

No physical signature, matter representation or dynamics is selected here.
Tensor input remains authoritative; a cubic is not an exterior trivector.
"""
from dataclasses import dataclass
from itertools import permutations
import numpy as np

@dataclass(frozen=True)
class AlgebraSpec:
    signature: tuple[int, ...]
    scalar_field: str
    matrix_dimension: int
    basis_id: str

    def __post_init__(self):
        if not self.signature or any(s not in (-1, 1) for s in self.signature):
            raise ValueError("nondegenerate normalized signature required")
        if self.scalar_field not in ('real', 'complex') or self.matrix_dimension <= 0 or not self.basis_id:
            raise ValueError("explicit scalar field, carrier dimension and basis required")

@dataclass(frozen=True)
class StatisticalTensors:
    metric: np.ndarray
    cubic: np.ndarray
    chart_id: str
    source_id: str

    def __post_init__(self):
        g, t = np.asarray(self.metric), np.asarray(self.cubic)
        if np.iscomplexobj(g) or np.iscomplexobj(t):
            raise ValueError("real statistical tensors required; no silent complex projection")
        g, t = np.array(g, dtype=float, copy=True), np.array(t, dtype=float, copy=True)
        if g.ndim != 2 or g.shape[0] != g.shape[1] or t.shape != (g.shape[0],)*3:
            raise ValueError("metric and cubic dimensions disagree")
        if not self.chart_id or not self.source_id or not np.all(np.isfinite(g)) or not np.all(np.isfinite(t)):
            raise ValueError("finite data and chart/source provenance required")
        if not np.allclose(g,g.T,rtol=0,atol=1e-12) or np.min(np.linalg.eigvalsh(g)) <= 0:
            raise ValueError("positive definite symmetric statistical metric required")
        if any(not np.allclose(t,t.transpose(p),rtol=0,atol=1e-12) for p in permutations(range(3))):
            raise ValueError("statistical cubic must be symmetric")
        g.setflags(write=False);t.setflags(write=False)
        object.__setattr__(self,'metric',g);object.__setattr__(self,'cubic',t)

    def indexed_cubic_operators(self, vector_basis: np.ndarray) -> np.ndarray:
        """Return T_ab c gamma^c with TWO free tensor indices, retaining T.

        Caller transports the basis with the chart. This is an injective linear
        encoding when the supplied vector matrices are linearly independent;
        it neither antisymmetrizes T nor changes its tensor transformation law.
        """
        b=np.asarray(vector_basis)
        n=self.metric.shape[0]
        if b.ndim!=3 or b.shape[0]!=n or b.shape[1]!=b.shape[2] or not np.all(np.isfinite(b)):
            raise ValueError("one finite square vector matrix per tensor coordinate required")
        if np.linalg.matrix_rank(b.reshape(n,-1))!=n:
            raise ValueError("vector encoding would lose information")
        return np.einsum('abc,cij->abij',self.cubic,b)
