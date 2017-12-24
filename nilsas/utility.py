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


def remove_orth_projection(p, q):
    # removes from p its orthogonal projection onto span( w[j], j=... )
    # requires w[j] be an othonormal set
    # p:        shape(m,)
    # w:        shape(M, m)
    # b:        shape(M,)
    # p_new:    shape(m,)
    assert q.ndim == 2
    assert p.ndim == 1
    assert p.shape[0] == q.shape[1] >= q.shape[0]
    assert np.allclose(np.dot(q, q.T), np.eye(q.shape[0]))
    b = np.dot(q, p)
    p_new = p - np.dot(b, q)
    return p_new, b


def stackv(u, v):
    # basically np.vstack on two arrays, but
    # 1. allow one of the argument be empty
    # 2. add a newaxis for the arrays with 1 less dimension
    if u.shape == (0,):
        return v[np.newaxis]
    elif v.shape == (0,):
        return u[np.newaxis]
    elif u.shape != (0,) and v.shape != (0,):
        if u.shape[1:] == v.shape:
            return np.concatenate((u, v[np.newaxis]), axis=0)
        elif u.shape == v.shape[1:]:
            return np.concatenate((u[np.newaxis], v), axis=0)
        else:
            raise ValueError('stackv two arrays of wrong shape')
    else:
        raise ValueError('stackv two empty arrays')

