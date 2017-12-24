from __future__ import division
import numpy as np
import sys, os
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))
from nilsas.utility import qr_transpose, remove_orth_projection, stackv


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


def test_remove_orth_projection():
    w       = np.random.rand(5,7)
    q, _    = qr_transpose(w)
    p       = np.random.rand(7)
    p_new, b = remove_orth_projection(p, q)
    assert np.allclose(p_new, p-np.dot(b,q))
    assert np.allclose(np.zeros(5), np.dot(q,p_new))


def test_stackv():
    u = np.random.rand(4,5)
    v = np.array([])
    s = stackv(u, v)
    assert s.shape == (1,4,5) and np.all(s[0] == u)
    s = stackv(v, u)
    assert s.shape == (1,4,5) and np.all(s[0] == u)

    u = np.random.rand(6,4,5)
    v = np.random.rand(4,5)
    s = stackv(u, v)
    assert s.shape == (7,4,5) and np.all(s[:6] == u) and np.all(s[-1] == v)
    s = stackv(v, u)
    assert s.shape == (7,4,5) and np.all(s[0] == v) and np.all(s[1:] == u)
