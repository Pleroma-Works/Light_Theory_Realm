import itertools
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from light_theory_realm.engine import CliffordEngine
from light_theory_realm.contracts import StatisticalTensors
from light_theory_realm.qgt import CliffordQGT

def test_involutions_and_anti_automorphism():
    e=CliffordEngine();I=jnp.eye(4,dtype=e.cdtype);blades=[]
    for mask in range(16):
        b=I
        for i in range(4):
            if mask>>i&1:b=b@e.gammas[i]
        blades.append(b)
        r=mask.bit_count()
        np.testing.assert_allclose(e.reverse(b),(-1)**(r*(r-1)//2)*b,atol=1e-12)
        np.testing.assert_allclose(e.grade_involution(b),(-1)**r*b,atol=1e-12)
        np.testing.assert_allclose(e.clifford_conjugate(b),(-1)**(r*(r+1)//2)*b,atol=1e-12)
    for a,b in itertools.product(blades,repeat=2):
        np.testing.assert_allclose(e.reverse(a@b),e.reverse(b)@e.reverse(a),atol=1e-12)
    np.testing.assert_allclose(e.reverse(1j*I),1j*I,atol=1e-12)

def test_general_inverse_and_jit_domain():
    e=CliffordEngine();rng=np.random.default_rng(1701);I=jnp.eye(4)
    for m in [2*I+e.gammas[0]]+[jnp.array(4*np.eye(4)+rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))) for _ in range(5)]:
        inv=jax.jit(e.inverse)(m)
        np.testing.assert_allclose(m@inv,I,atol=1e-11);np.testing.assert_allclose(inv@m,I,atol=1e-11)
        np.testing.assert_allclose(e.inverse_checked(m),inv,atol=1e-12)
    with pytest.raises(ValueError):e.inverse_checked(jnp.zeros((4,4)))
    with pytest.raises(ValueError):e.inverse_checked(jnp.eye(3))
    assert not np.all(np.isfinite(e.inverse(jnp.zeros((4,4)))))

def test_tracefree_cubic_is_preserved_by_indexed_contract():
    e=CliffordEngine();t=np.zeros((3,)*3);t[0,0,0]=1
    for p in set(itertools.permutations((0,1,1))):t[p]=-1
    fields=StatisticalTensors(np.eye(3),t,'orthonormal','fixture')
    gamma=np.array([1j*g for g in e.gammas[1:]])
    full=fields.indexed_cubic_operators(gamma)
    recovered=np.einsum('cij,abij->abc',gamma.conj(),full)/4
    np.testing.assert_allclose(recovered,t,atol=1e-12)
    naive=np.einsum('abc,aij,bjk,ckl->il',t,gamma,gamma,gamma)
    np.testing.assert_allclose(naive,0,atol=1e-12)
    with pytest.raises(ValueError):fields.indexed_cubic_operators(np.repeat(gamma[:1],3,axis=0))
    with pytest.raises(ValueError):StatisticalTensors(-np.eye(3),t,'chart','fixture')

def test_qgt_normalization_contract():
    e=CliffordEngine();q=CliffordQGT(e);th=.7;ph=.4
    state=lambda z:jnp.array([jnp.cos(z[0]/2),jnp.exp(1j*z[1])*jnp.sin(z[0]/2)])
    z=jnp.array([th,ph]);psi=state(z);J=jax.jacfwd(state)(z)
    r=q.normalized_geometry(psi,J)
    np.testing.assert_allclose(r['quantum_fisher'],np.diag([1,np.sin(th)**2]),atol=1e-12)
    np.testing.assert_allclose(r['berry_curvature'][0,1],np.sin(th)/2,atol=1e-12)
    np.testing.assert_allclose(q.normalized_geometry(psi,J,connection_sign=1)['berry_curvature'],-r['berry_curvature'],atol=1e-12)
