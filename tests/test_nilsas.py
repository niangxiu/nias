# Tests on nilsas using some apps such as Lorenz 63 are also contained in this file, 
# rather than in the app's corresponding test files

from __future__ import division
import numpy as np
import sys, os
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))


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
    m           = 3         # not needed for the vector_bundle function
    dt          = 0.001

    forward, interface, segment =  vector_bundle( 
           lrz.run_forward, lrz.run_adjoint, u0, parameter, 
           M_modes, K_segment, nstep_per_segment, runup_steps, dt)

    return forward, interface, segment, M_modes, m, K_segment, nstep_per_segment, dt


def test_forward(vecbd_lorenz):
    # forward has member variables:
    # u:    shape(K, nstep_per_segment, m)
    # f:    shape(K, nstep_per_segment, m)
    # fu:   shape(K, nstep_per_segment, m, m)
    # fs:   shape(K, nstep_per_segment, ns, m)
    # J:    shape(K, nstep_per_segment,)
    # Ju:   shape(K, nstep_per_segment, m)
    # Js:   shape(K, nstep_per_segment, ns)
    fw, _, _, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz

    # test shape
    assert fw.u.shape[0] == fw.f.shape[0] == fw.fu.shape[0] == fw.fs.shape[0] \
            == fw.J.shape[0] == fw.Ju.shape[0] == fw.Js.shape[0] == K_segment
    assert fw.u.shape[1] == fw.f.shape[1] == fw.fu.shape[1] == fw.fs.shape[1] \
            == fw.J.shape[1] == fw.Ju.shape[1] == fw.Js.shape[1] == nstep_per_segment + 1
    assert fw.u.shape[2] == fw.f.shape[2] == fw.fu.shape[2] == fw.fs.shape[3] \
            == fw.Ju.shape[2] == fw.fu.shape[3] == m
    assert fw.fs.shape[2] == fw.Js.shape[2] # should be number of parameters


def test_segment(vecbd_lorenz):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz
    # check shape
    # w:        shape(K, nstep_per_segment, M, m)  
    # yst, vst: shape(K, nstep_per_segment, m)
    # C:        shape(K, M, M)
    # dy, dv:   shape(K, M)
    assert sg.w.shape[0] == sg.yst.shape[0] == sg.vst.shape[0] \
            == sg.C.shape[0] == sg.dy.shape[0] == sg.dv.shape[0] \
            == K_segment
    assert sg.w.shape[1] == sg.yst.shape[1] == sg.vst.shape[1] \
            == nstep_per_segment+1
    assert sg.w.shape[2] == sg.C.shape[1] == sg.C.shape[2] \
            == sg.dy.shape[1] == sg.dv.shape[1] \
            == M_modes
    assert sg.w.shape[3] == sg.yst.shape[2] == sg.vst.shape[2] \
            == m

    # check satisfy governing equation
    w = sg.w[0]     # shape(nstep, M, m)
    vst = sg.vst[0] # shape(nstep, m)
    yst = sg.yst[0] # shape(nstep, m)
    fu = fw.fu[0]   # shape(nstep, m, m)
    Ju = fw.Ju[0]
    f  = fw.f[0]
    assert np.allclose( (w[1:] - w[:-1]) / dt, \
            (-fu[:-1,np.newaxis,:,:] * w[1:,:,:,np.newaxis]).sum(axis=-2) )
    assert np.allclose( (yst[1:] - yst[:-1]) / dt, \
            (-fu[:-1] * yst[1:,:,np.newaxis]).sum(axis=-2) )
    assert np.allclose( (vst[1:] - vst[:-1]) / dt, \
            (-fu[:-1] * vst[1:,:,np.newaxis]).sum(axis=-2) - Ju[:-1] )


def test_interface_terminal(vecbd_lorenz):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz
    w_tmn   = itf.w_right[-1]
    yst_tmn = itf.yst_right[-1]
    vst_tmn = itf.vst_right[-1]
    f_tmn   = fw.f[-1,-1]
    # check shape
    assert w_tmn.shape[0] == M_modes
    assert f_tmn.shape[0] == w_tmn.shape[1] == yst_tmn.shape[0] \
            == vst_tmn.shape[0] == m
    # check if w_tmn is orthogonal to f_tmn
    assert np.allclose(np.dot(w_tmn, f_tmn) , np.zeros(M_modes))
    # check yst_tmn is f_tmn
    assert np.allclose(yst_tmn, f_tmn)
    # check vst_tmn is zero
    assert np.allclose(vst_tmn, np.zeros(m))
    # check if the left values are the same as the right
    assert np.allclose(itf.Q[-1],           itf.w_right[-1])
    assert np.allclose(itf.yst_left[-1],    itf.yst_right[-1])
    assert np.allclose(itf.vst_left[-1],    itf.vst_right[-1])


def test_segment_interface_continuous(vecbd_lorenz):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz
    assert np.allclose(itf.w_right[:-1],    sg.w[:,0])
    assert np.allclose(itf.Q[1:],           sg.w[:,-1])
    assert np.allclose(itf.yst_right[:-1],  sg.yst[:,0])
    assert np.allclose(itf.yst_left[1:],    sg.yst[:,-1])
    assert np.allclose(itf.vst_right[:-1],  sg.vst[:,0])
    assert np.allclose(itf.vst_left[1:],    sg.vst[:,-1])
    assert np.allclose(fw.f[1:,0],          fw.f[:-1,-1])


def test_orthogonal_w_f(vecbd_lorenz):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz
    for i in range(K_segment):
        w   = sg.w[i]
        vst = sg.vst[i]
        yst = sg.yst[i]
        fu  = fw.fu[i]  # shape(nsteps, m, m)
        Ju  = fw.Ju[i]
        f   = fw.f[i]
        _   = (w * f[:,np.newaxis,:]).sum(axis=-1)
        assert np.allclose( _, np.zeros(_.shape) )


def test_rescale(vecbd_lorenz):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz
    for i in range(K_segment + 1):
        Q   = itf.Q[i]          # shape(M_modes, m)
        vst = itf.vst_left[i]   # shape(m,)
        yst = itf.yst_left[i]   # shape(m,)
        assert np.allclose(np.dot(Q, Q.T), np.eye(Q.shape[0]))
        assert np.allclose(np.dot(Q, vst), np.zeros(Q.shape[0]))
        assert np.allclose(np.dot(Q, yst), np.zeros(Q.shape[0]))
