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
u0      = [0,1,5]
m       = 3         # dimension of system
M       = 2         # number of homogeneous adjoints
ns      = 2         # number of parameters

@pytest.fixture(scope="module")
def trajectory():
    return run_forward(u0, nstep)


def test_primal(trajectory):
    u, f, fu, fs, J, Ju, Js = trajectory
    assert u.shape[0] == f.shape[0] == J.shape[0]  == fu.shape[0] == Ju.shape[0] == fs.shape[0] == nstep + 1
    assert u.shape[1] == f.shape[1] == fu.shape[1] == fu.shape[2] == Ju.shape[1] == fs.shape[2] == m
    assert fs.shape[1] == ns
    assert -20  <= np.average(u[:,0]) <= 20
    assert 10   <= np.average(u[:,2]) <= 100


def test_adjoint(trajectory):
    u, f, fu, fs, J, Ju, Js = trajectory
    w_tmn, yst_tmn, vst_tmn = adjoint_terminal_condition(M, f[-1])
    w, yst, vst = run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju)
    assert w.shape[0] == yst.shape[0] == vst.shape[0] == nstep + 1
    assert w.shape[1] == M
    assert w.shape[2] == yst.shape[1] == vst.shape[1] == m
    _ =  (w * f[:,np.newaxis,:]).sum(axis=-1)  
    assert np.allclose(_ , np.zeros(_.shape))    


def test_forward(trajectory):
    # u:    shape(K, nstep_per_segment, m)
    # f:    shape(K, nstep_per_segment, m)
    # fu:   shape(K, nstep_per_segment, m, m)
    # fs:   shape(K, nstep_per_segment, ns, m)
    # J:    shape(K, nstep_per_segment,)
    # Ju:   shape(K, nstep_per_segment, m)
    # Js:   shape(K, nstep_per_segment, ns)
    u_, _, _, _, _, _, _ = trajectory
    u0 = u_[-1]
    nstep_per_segment = 400
    K_segment = 20
    m = 3
    ns = 2
    fw = Forward()
    fw.run(run_forward, u0, nstep_per_segment, K_segment)
    assert fw.u.shape[0] == fw.f.shape[0] == fw.fu.shape[0] == fw.fs.shape[0] \
            == fw.J.shape[0] == fw.Ju.shape[0] == fw.Js.shape[0] == K_segment
    assert fw.u.shape[1] == fw.f.shape[1] == fw.fu.shape[1] == fw.fs.shape[1] \
            == fw.J.shape[1] == fw.Ju.shape[1] == fw.Js.shape[1] == nstep_per_segment + 1
    assert fw.u.shape[2] == fw.f.shape[2] == fw.fu.shape[2] == fw.fs.shape[3] \
            == fw.Ju.shape[2] == fw.fu.shape[3] == m
    assert fw.fs.shape[2] == fw.Js.shape[2] == ns
    # for i in range(K_segment):
        # plt.plot(fw.u[i,:,0], fw.u[i,:,2])
    # plt.savefig('trajec2.png')
    # plt.close('all')
    
