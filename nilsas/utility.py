# this file has some misc utility functions

from __future__ import division
import numpy as np

def qr_transpose(A):
    # B[i] are orthonormal sets
    # with coefficents in R[:,j], we form linear combination that recovers A[j]:
    # B.T * R = A.T
    Q, R = np.linalg.qr(A.T)
    B = Q.T
    return B, R
