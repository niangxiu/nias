# this file has some misc utility functions

from __future__ import division
import numpy as np

def qr_transpose(A):
    # B[i] are orthonormal sets, 
    # from which we can form linear combination that recovers A[j], 
    # with coefficents in R[:,j].
    # B.T * R = A.T
    assert A.ndim == 2
    Q, R = np.linalg.qr(A.T)
    B = Q.T
    return B, R


def remove_orth_projection(p, w)
    # removes from p its orthogonal projection onto span( w[j], j=... )
    # p:        shape(m,)
    # w:        shape(M, m)
    # b:        shape(M,)
    # p_new:    shape(m,)
    assert w.ndim == 2
    assert p.ndim == 1
    assert p.shape[0] == w.shape[1] >= w.shape[0]
    b = np.dot(w, p)
    p_new = p - np.dot(b, w)
    return p_new, b
