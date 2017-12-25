from __future__ import division
import numpy as np
from numpy import nan
import sys, os
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))
from .test_vector_bundle import vecbd_lorenz

from nilsas.nilsas import nilsas_min
def test_nilsas_matrix(vecbd_lorenz):
    fw, itf, sg, M_modes, m, K_segment, nstep_per_segment, dt = vecbd_lorenz
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
