# Tests on nilsas using some apps such as Lorenz 63 are also contained in this file, 
# rather than in the app's corresponding test files

from __future__ import division
import numpy as np
import sys, os

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


from 
