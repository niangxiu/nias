from __future__ import division
import numpy as np
import sys, os
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdb import set_trace

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

import apps.lorenz as lorenz
import nilsas.segment as segment
from nilsas.interface import adjoint_terminal_condition
from nilsas.forward import Forward


nstep  = 5000
u0      = [1,3,5]
m       = 3         # dimension of system
M       = 2         # number of homogeneous adjoints
ns      = 2         # number of parameters
dt      = 0.001


@pytest.fixture(scope="module")
def trajectory():
    forward = Forward()
    return forward.run1seg(u0, nstep, lorenz.step_PTA, lorenz.derivatives, lorenz.dudt)


def test_run_primal(trajectory):
    
    u, f, fu, fs, J, Ju, Js = trajectory
    assert u.shape[0] == f.shape[0] == J.shape[0]  == fu.shape[0] == Ju.shape[0] == fs.shape[0] == nstep + 1
    assert u.shape[1] == f.shape[1] == fu.shape[1] == fu.shape[2] == Ju.shape[1] == fs.shape[2] == m
    assert fs.shape[1] == ns
    assert -20  <= np.average(u[:,0]) <= 20
    assert 10   <= np.average(u[:,2]) <= 100


def test_run_adjoint(trajectory):
    u, f, fu, fs, J, Ju, Js = trajectory
    w_tmn, vst_tmn = adjoint_terminal_condition(M, f[-1])
    w, vst = segment.get_wvst(w_tmn, vst_tmn, fu, Ju, lorenz.adjoint_step_explicit)
    assert w.shape[0] == vst.shape[0] == nstep + 1
    assert w.shape[1] == M
    assert w.shape[2] == vst.shape[1] == m
    # test if w remains orthorgonal to f
    _ =  (w[:,:-1] * f[:,np.newaxis,:]).sum(axis=-1)  
    assert np.allclose(_ , np.zeros(_.shape))    


@pytest.fixture(scope="module")
def trajectory_implicit():
    # this implicit scheme comes to a fake stable point if nstep=1e5
    forward = Forward()
    return forward.run1seg(u0, nstep, lorenz.step_PTA_backward_Euler, lorenz.derivatives, lorenz.dudt)


def test_run_primal2(trajectory_implicit):
    # the trajectory computed by implicit PTA has a fake stable point    
    u, f, fu, fs, J, Ju, Js = trajectory_implicit
    assert u.shape[0] == f.shape[0] == J.shape[0]  == fu.shape[0] \
            == Ju.shape[0] == fs.shape[0] 
    assert u.shape[1] == f.shape[1] == fu.shape[1] == fu.shape[2] \
            == Ju.shape[1] == fs.shape[2] 
    assert fs.shape[1] 
    assert -40  <= np.average(u[:,0]) <= 40
    assert 10   <= np.average(u[:,2]) <= 100


def test_run_adjoint2(trajectory_implicit):
    u, f, fu, fs, J, Ju, Js = trajectory_implicit
    w_tmn, vst_tmn = adjoint_terminal_condition(M, f[-1])
    w, vst = segment.get_wvst(w_tmn, vst_tmn, fu, Ju, lorenz.adjoint_step_implicit)
    assert w.shape[0] == vst.shape[0] 
    assert w.shape[2] == vst.shape[1] 
    # test if w remains orthorgonal to f
    _ =  (w[:,:-1] * f[:,np.newaxis,:]).sum(axis=-1)  
    assert np.allclose(_ , np.zeros(_.shape))    


