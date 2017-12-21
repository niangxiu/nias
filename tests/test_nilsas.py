# Tests on nilsas using some apps such as Lorenz 63 are also contained in this file, 
# rather than in the app's corresponding test files

from __future__ import division
import numpy as np
import sys, os
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))


from nilsas.utility import qr_transpose
def test_qr():
    A       = np.random.rand(4,6)
    Q, R    = qr_transpose(A)
    
    # check shape
    assert R.shape[0] == R.shape[1] == Q.shape[0] == A.shape[0]
    assert Q.shape[1] == A.shape[1]
    
    # check R is upper triangular
    assert np.allclose(R, np.triu(R))

    # check Q is orthogonal
    assert np.allclose(np.dot(Q, Q.T), np.eye(R.shape[0]))


from nilsas.utility import remove_orth_projection
def test_remove_orth_projection():
    w       = np.random.rand(5,7)
    q, _    = qr_transpose(w)
    p       = np.random.rand(7)
    p_new, b = remove_orth_projection(p, q)
    assert np.allclose(p_new, p-np.dot(b,q))
    assert np.allclose(np.zeros(5), np.dot(q,p_new))


from nilsas.interface import adjoint_terminal_condition
def test_terminal_condition():
    M = 3
    m = 5
    f_tmn = np.random.rand(m)
    w_tmn, yst_tmn, vst_tmn = adjoint_terminal_condition(M, f_tmn)

    # check shape
    assert w_tmn.shape[0] == M
    assert w_tmn.shape[1] == yst_tmn.shape[0] == vst_tmn.shape[0] == m

    # check if w_tmn is orthogonal to f_tmn
    assert np.allclose(np.dot(w_tmn, f_tmn) , np.zeros(M))

    # check yst_tmn is f_tmn
    assert np.allclose(yst_tmn, f_tmn)

    # check vst_tmn is zero
    assert np.allclose(vst_tmn, np.zeros(m))


from nilsas.nilsas import vector_bundle
import apps.lorenz as lrz
@pytest.fixture(scope ='module')
def vecbd_lorenz():
    u0          = [0,1,5]
    parameter   = (30, 10)
    M_modes     = 2
    K_segment   = 40
    nstep_per_segment   = 200
    runup_steps = 4000
    m           = 3

    forward, interface, segment =  vector_bundle( 
           lrz.run_forward, lrz.run_adjoint, u0, parameter, 
           M_modes, K_segment, nstep_per_segment, runup_steps)

    return forward, interface, segment, M_modes, m, K_segment, nstep_per_segment


def test_forward(vecbd_lorenz):
    # forward has member variables:
    # u:    shape(K, nstep_per_segment, m)
    # f:    shape(K, nstep_per_segment, m)
    # fu:   shape(K, nstep_per_segment, m, m)
    # fs:   shape(K, nstep_per_segment, ns, m)
    # J:    shape(K, nstep_per_segment,)
    # Ju:   shape(K, nstep_per_segment, m)
    # Js:   shape(K, nstep_per_segment, ns)
    fw, _, _, M_modes, m, K_segment, nstep_per_segment = vecbd_lorenz

    # test shape
    assert fw.u.shape[0] == fw.f.shape[0] == fw.fu.shape[0] == fw.fs.shape[0] \
            == fw.J.shape[0] == fw.Ju.shape[0] == fw.Js.shape[0] == K_segment
    assert fw.u.shape[1] == fw.f.shape[1] == fw.fu.shape[1] == fw.fs.shape[1] \
            == fw.J.shape[1] == fw.Ju.shape[1] == fw.Js.shape[1] == nstep_per_segment + 1
    assert fw.u.shape[2] == fw.f.shape[2] == fw.fu.shape[2] == fw.fs.shape[3] \
            == fw.Ju.shape[2] == fw.fu.shape[3] == m
    assert fw.fs.shape[2] == fw.Js.shape[2] # should be number of parameters


def test_segment(vecbd_lorenz):
    forward, interface, segment, M_modes, m, K_segment, nstep_per_segment = vecbd_lorenz
    # test shape
    pass
