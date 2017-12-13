from __future__ import division
import numpy as np
import sys, os
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from apps.lorenz import solve_primal, solve_adjoint 
from nilsas.nilsas import adjoint_terminal_condition

nsteps  = 1000
u0      = [0,1,5]
m       = 3         # dimension of system
M       = 2         # number of homogeneous adjoints

@pytest.fixture(scope="module")
def trajectory():
    return solve_primal(u0, nsteps)


def test_primal(trajectory):
    u, f, J, fu, Ju, fs = trajectory
    assert u.shape[0] == f.shape[0] == J.shape[0]  == fu.shape[0] == Ju.shape[0] == fs.shape[0] == nsteps + 1
    assert u.shape[1] == f.shape[1] == fu.shape[1] == fu.shape[2] == Ju.shape[1] == fs.shape[1] == m
    assert -20  <= np.average(u[:,0]) <= 20
    assert 0    <= np.average(u[:,2]) <= 100


def test_adjoint(trajectory):
    u, f, J, fu, Ju, fs = trajectory
    w_tmn, yst_tmn, vst_tmn = adjoint_terminal_condition(M, f[-1])
    w, yst, vst = solve_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju)
    assert w.shape[0] == yst.shape[0] == vst.shape[0] == nsteps + 1
    assert w.shape[1] == M
    assert w.shape[2] == yst.shape[1] == vst.shape[1] == m
    _ =  (w * f[:,np.newaxis,:]).sum(axis=-1)  
    assert np.amax(np.absolute(_)) <= 1e-11
