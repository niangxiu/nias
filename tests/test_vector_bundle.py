# Tests on nilsas using some apps such as Lorenz 63 are also contained in this file, 
# rather than in the app's corresponding test files
# tests on v and y are in test_nilsas.py

from __future__ import division
import numpy as np
import sys, os
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))
from nilsas.interface import adjoint_terminal_condition
from nilsas.nilsas import vector_bundle
import apps.lorenz as lorenz
from nilsas.utility import angle


def test_terminal_condition_function():
    M_modes = 3
    m = 5
    f_tmn = np.random.rand(m)
    w_tmn, vst_tmn = adjoint_terminal_condition(M_modes, f_tmn)

    # check shape
    assert w_tmn.shape == (M_modes, m)
    assert vst_tmn.shape == f_tmn.shape == (m,)
    # check if w_tmn is orthogonal to f_tmn
    assert np.allclose(np.dot(w_tmn[:-1], f_tmn) , np.zeros(M_modes-1))
    # check the last column of w is f_tmn
    assert np.allclose(w_tmn[-1], f_tmn/np.linalg.norm(f_tmn))
    # check vst_tmn is zero
    assert np.allclose(vst_tmn, np.zeros(m))


@pytest.fixture(scope ='module')
def vecbd_lorenz_explicit():
    # uses the explicit PTA integrations
    u0 = [0,1,5]
    parameter = (30, 10)
    M_modes = 2
    K_segment = 100
    nstep_per_segment = 100
    runup_steps = 4000
    m = 3         # not needed for the vector_bundle function
    dt = 0.001
    ns = 2

    forward, interface, segment =  vector_bundle( 
           lorenz.run_forward, lorenz.run_adjoint, u0, parameter, 
           M_modes, K_segment, nstep_per_segment, runup_steps, dt,
           lorenz.step_PTA, lorenz.adjoint_step_explicit)

    return forward, interface, segment, M_modes, m, K_segment, nstep_per_segment, dt, ns


def test_forward_explicit(vecbd_lorenz_explicit):
    fw, _, _, M_modes, m, K_segment, nstep_per_segment, dt, ns \
            = vecbd_lorenz_explicit
    # test shape
    assert fw.u.shape == (K_segment, nstep_per_segment+1, m)
    assert fw.f.shape == (K_segment, nstep_per_segment+1, m)
    assert fw.fu.shape == (K_segment, nstep_per_segment+1, m, m)
    assert fw.fs.shape == (K_segment, nstep_per_segment+1, ns, m)
    assert fw.J.shape == (K_segment, nstep_per_segment+1)
    assert fw.Ju.shape == (K_segment, nstep_per_segment+1, m)
    assert fw.Js.shape == (K_segment, nstep_per_segment+1, ns)
    # test continuity
    assert np.allclose(fw.u[1:,0], fw.u[:-1,-1])
    assert np.allclose(fw.f[1:,0], fw.f[:-1,-1])
    assert np.allclose(fw.fu[1:,0], fw.fu[:-1,-1])
    assert np.allclose(fw.fs[1:,0], fw.fs[:-1,-1])
    assert np.allclose(fw.J[1:,0], fw.J[:-1,-1])
    assert np.allclose(fw.Ju[1:,0], fw.Ju[:-1,-1])
    assert np.allclose(fw.Js[1:,0], fw.Js[:-1,-1])
    # plot f
    # fig = plt.figure()
    # plt.plot(np.linalg.norm(fw.f.reshape([-1,3]), axis=-1))
    # plt.savefig('fnorm.png')
    # plt.close(fig)


def test_segment(vecbd_lorenz_explicit):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt, ns \
            = vecbd_lorenz_explicit
    # check shape
    assert sg.w.shape == (K_segment, nstep_per_segment+1, M_modes, m)  
    assert sg.vst.shape == (K_segment, nstep_per_segment+1, m)
    assert sg.C.shape == (K_segment, M_modes, M_modes)
    assert sg.dwv.shape == sg.dwf.shape == (K_segment, M_modes)
    assert sg.dvf.shape == (K_segment,)

    # check satisfy governing equation
    w = sg.w[0]     # shape(nstep, M, m)
    vst = sg.vst[0] # shape(nstep, m)
    fu = fw.fu[0]   # shape(nstep, m, m)
    Ju = fw.Ju[0]
    f  = fw.f[0]
    assert np.allclose( (w[1:] - w[:-1]) / dt, \
            (-fu[:-1,np.newaxis,:,:] * w[1:,:,:,np.newaxis]).sum(axis=-2) )
    assert np.allclose( (vst[1:] - vst[:-1]) / dt, \
            (-fu[:-1] * vst[1:,:,np.newaxis]).sum(axis=-2) - Ju[:-1] )


def test_terminal(vecbd_lorenz_explicit):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt, ns \
            = vecbd_lorenz_explicit
    w_tmn   = itf.w_right[-1]
    vst_tmn = itf.vst_right[-1]
    f_tmn   = fw.f[-1,-1]
    assert w_tmn.shape == (M_modes, m)
    assert f_tmn.shape == vst_tmn.shape == (m,)
    # check w_tmn is orthogonal to f_tmn, w_tmn[-1] is direction of f
    assert np.allclose(np.dot(w_tmn[:-1], f_tmn) , np.zeros(M_modes))
    assert np.allclose(w_tmn[-1], f_tmn/np.linalg.norm(f_tmn))
    # check vst_tmn is zero
    assert np.allclose(vst_tmn, np.zeros(m))
    # check if the left values are the same as the right
    assert np.allclose(np.abs(itf.Q[-1]), np.abs(itf.w_right[-1]))
    assert np.allclose(itf.vst_left[-1], itf.vst_right[-1])


def test_segment_interface_continuous(vecbd_lorenz_explicit):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt, ns \
            = vecbd_lorenz_explicit
    assert np.allclose(itf.w_right[:-1], sg.w[:,0])
    assert np.allclose(itf.Q[1:], sg.w[:,-1])
    assert np.allclose(itf.vst_right[:-1], sg.vst[:,0])
    assert np.allclose(itf.vst_left[1:], sg.vst[:,-1])
    assert np.allclose(fw.f[1:,0], fw.f[:-1,-1])


def test_othorgonal_w_f(vecbd_lorenz_explicit):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt, ns\
            = vecbd_lorenz_explicit
    for i in range(K_segment):
        w   = sg.w[i]
        vst = sg.vst[i]
        fu  = fw.fu[i]  # shape(nsteps, m, m)
        Ju  = fw.Ju[i]
        f   = fw.f[i]
        _   = (w[:,:-1] * f[:,np.newaxis,:]).sum(axis=-1)
        assert np.allclose(_, np.zeros(_.shape))


def test_rescale(vecbd_lorenz_explicit):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt, ns \
            = vecbd_lorenz_explicit
    for i in range(K_segment + 1):
        Q = itf.Q[i] # shape(M_modes, m)
        vst_left = itf.vst_left[i] # shape(m,)
        vst_right = itf.vst_right[i] # shape(m,)
        bv = itf.bv[i]
        assert np.allclose(np.dot(Q, Q.T), np.eye(Q.shape[0]))
        assert np.allclose(np.dot(Q, vst_left), np.zeros(Q.shape[0]))
        assert np.allclose(np.dot(Q, vst_right), bv)
