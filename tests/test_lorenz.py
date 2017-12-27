from __future__ import division
import numpy as np
import sys, os
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from apps.lorenz import run_forward, run_adjoint 
from nilsas.interface import adjoint_terminal_condition
from nilsas.forward import Forward

nstep  = 10000
u0      = [1,3,5]
m       = 3         # dimension of system
M       = 2         # number of homogeneous adjoints
ns      = 2         # number of parameters
base_parameter = (30, 10)
dt      = 0.001

@pytest.fixture(scope="module")
def trajectory():
    return run_forward(u0, base_parameter, nstep, dt)


def test_run_primal(trajectory):
    
    u, f, fu, fs, J, Ju, Js = trajectory
    assert u.shape[0] == f.shape[0] == J.shape[0]  == fu.shape[0] == Ju.shape[0] == fs.shape[0] == nstep + 1
    assert u.shape[1] == f.shape[1] == fu.shape[1] == fu.shape[2] == Ju.shape[1] == fs.shape[2] == m
    assert fs.shape[1] == ns
    assert -20  <= np.average(u[:,0]) <= 20
    assert 10   <= np.average(u[:,2]) <= 100

    # fig = plt.figure()
    # plt.plot(u[:,0], u[:,1])
    # plt.savefig('lorenz_x_y.png')
    # plt.close(fig)    

    # fig = plt.figure()
    # plt.plot(u[:,0], u[:,2])  
    # plt.savefig('lorenz_x_z.png')
    # plt.close(fig)    


def test_run_adjoint(trajectory):
    u, f, fu, fs, J, Ju, Js = trajectory
    w_tmn, yst_tmn, vst_tmn = adjoint_terminal_condition(M, f[-1])
    w, yst, vst = run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju, dt)
    assert w.shape[0] == yst.shape[0] == vst.shape[0] == nstep + 1
    assert w.shape[1] == M
    assert w.shape[2] == yst.shape[1] == vst.shape[1] == m
    # test if w remains orthorgonal to f
    _ =  (w * f[:,np.newaxis,:]).sum(axis=-1)  
    assert np.allclose(_ , np.zeros(_.shape))    

