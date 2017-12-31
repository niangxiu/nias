from __future__ import division
import numpy as np
from numpy import nan
import sys, os
import pytest
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))
from .test_vector_bundle import vecbd_lorenz_explicit
from nilsas.nilsas import nilsas_min, nilsas_main
import apps.lorenz as lorenz


def test_nilsas_matrix(vecbd_lorenz_explicit):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz_explicit
    ay, C, Cinv, B = nilsas_min( sg.C, itf.R, sg.dy, itf.by ) 

    assert B.shape == ((K_segment-1)*M_modes, K_segment*M_modes)
    assert C.shape == (K_segment*M_modes, K_segment*M_modes)

    for i in range(K_segment-1):
        B[i*M_modes:(i+1)*M_modes, i*M_modes:(i+1)*M_modes] \
                -= np.eye(M_modes)
        B[i*M_modes:(i+1)*M_modes, (i+1)*M_modes:(i+2)*M_modes] \
                += itf.R[i+1]
    assert np.allclose(B.todense(), np.zeros(B.shape))

    for i in range(K_segment):
        C[i*M_modes:(i+1)*M_modes, i*M_modes:(i+1)*M_modes] \
                -= sg.C[i]
        Cinv[i*M_modes:(i+1)*M_modes, i*M_modes:(i+1)*M_modes] \
                -= np.linalg.inv(sg.C[i])
    assert np.allclose(C.todense(), np.zeros(C.shape))
    assert np.allclose(Cinv.todense(), np.zeros(Cinv.shape))


def test_nilsas_min():
    C = np.array([[[1]],[[4]]])
    d = np.array([[2],[6]])
    R = np.array([[[nan]],[[2]],[[nan]]])
    b = np.array([[nan],[3],[nan]])

    a, C_, Cinv, B = nilsas_min( C, R, d, b )
    assert np.allclose(a, np.array([[-1], [-2]]))
    assert np.allclose(B.todense(), np.array([1,-2]))
    assert np.allclose(C_.todense(), np.array([[1,0],[0,4]]))
    assert np.allclose(Cinv.todense(), np.array([[1,0],[0,0.25]]))


@pytest.fixture(scope='module')
def vecbd_lorenz_v(vecbd_lorenz_explicit):
    # provide a vector bundle where the segment has y, vstpm, dv, vpm, v
    forward, interface, segment, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz_explicit
    ay, _, _, _ = nilsas_min(segment.C, interface.R, segment.dy, interface.by )
    segment.y_vstpm_dv(ay, forward.f)
    av, _, _, _ = nilsas_min(segment.C, interface.R, segment.dv, interface.bv)
    segment.vpm_v(av, forward.f, forward.Jtild)
    return forward, interface, segment, M_modes, m, K_segment, nstep_per_segment, dt



def test_v(vecbd_lorenz_v):
    forward, interface, segment, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz_v
    v = segment.v   # shape(K, nstep_per_segment, m)

    # check continuity and norm
    assert np.allclose(v[1:,0], v[:-1,-1])
    vnorm = np.linalg.norm(v.reshape([-1,3]), axis=-1)
    np.testing.assert_approx_equal(
            np.average(vnorm[:int(vnorm.shape[0]/2)]),
            np.average(vnorm[int(vnorm.shape[0]/2):]),
            significant=1)
    # plot  
    # fig = plt.figure()
    # plt.plot(np.linalg.norm(v.reshape([-1,3]), axis=-1))
    # plt.savefig('vnorm.png')
    # plt.close(fig)


def test_nilsas_main():
    u0          = [0,1,5]
    parameter   = (30, 10)
    M_modes     = 2
    K_segment   = 40
    nstep_per_segment   = 500
    runup_steps = 4000
    dt          = 0.001
    
    Javg, grad = nilsas_main(
        lorenz.run_forward, lorenz.run_adjoint, u0, parameter, M_modes,
        K_segment, nstep_per_segment, runup_steps, dt, 
        lorenz.step_PTA, lorenz.adjoint_step_explicit)

    assert 0.8 <= grad[0] <= 1.2
