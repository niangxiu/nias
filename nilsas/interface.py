# the class Interface is for the adjoint quatities at interfaces

import numpy as np
from .utility import qr_transpose, remove_orth_projection, stackv

def adjoint_terminal_condition(M_modes, f_tmn):
    """
    Args: 
        M_modes: number of homogeneous adjoint
        m: the dimension of the dynamical system
        f_tmn: f at the end of the trajectory
    Returns:
        w: terminal conditions for h-adjoints, the last being direction of f
        vst: terminal conditions fo i-adjoint, all zeros
    """
    m = f_tmn.shape[0]
    assert M_modes <= m 
    assert f_tmn.ndim == 1

    f_unit = f_tmn / np.linalg.norm(f_tmn)
    W = np.random.rand(M_modes-1, m)
    W = W - np.dot(W, f_unit)[:,np.newaxis] * f_unit
    w, _ = qr_transpose(W)
    w = stackv(w, f_unit)
    vst_tmn = np.zeros(m)

    return w, vst_tmn


class Interface:

    def __init__(self):
        # Q <- w_right:             shape(K+1, M, m)
        # R:                        shape(K+1, M, M)
        # vst_left <- vst_right:    shape(K+1, m)
        # bv:                   shape(K+1, M)
        self.w_right    = np.array([])
        self.Q          = np.array([])
        self.R          = np.array([])
        self.vst_right  = np.array([])
        self.vst_left   = np.array([])
        self.bv         = np.array([])


    def terminal(self, M_modes, forward):
        # get the right-of-interface values at t = t_n = T
        assert self.w_right.shape == self.vst_right.shape == (0,)
        w_right, vst_right = adjoint_terminal_condition(
                M_modes, forward.f[-1,-1])

        Q, R = qr_transpose(w_right)
        vst_left, bv = remove_orth_projection(vst_right, Q)
        assert np.allclose(np.abs(Q), np.abs(w_right))
        assert np.allclose(vst_left, vst_right)

        self.w_right = stackv(w_right, self.w_right)
        self.vst_right = stackv(vst_right, self.vst_right)
        self.Q = stackv(Q, self.Q)
        self.R = stackv(R, self.R)
        self.vst_left = stackv(vst_left, self.vst_left)
        self.bv = stackv(bv, self.bv)

        assert self.w_right.shape[0] == 1
        assert self.w_right.ndim == 3


    def interface_right(self, segment):
        # get the right-of-interface values at t_i < T
        # may not need this function if computer memory is running short
        w_right         = segment.w[0,0]
        vst_right       = segment.vst[0,0]
        self.w_right    = stackv(w_right,   self.w_right)
        self.vst_right  = stackv(vst_right, self.vst_right)


    def rescale(self, f_interface, w_right=None, vst_right=None):
        # compute left-of-interface values from right-of-interface values
        if w_right is None:
            w_right = self.w_right[0]
            vst_right = self.vst_right[0]

        assert w_right.ndim == 2
        assert f_interface.ndim == vst_right.ndim == 1

        Q, R = qr_transpose(w_right)
        vst_left, bv = remove_orth_projection(vst_right, Q)

        self.Q          = stackv(Q, self.Q)
        self.R          = stackv(R, self.R)
        self.vst_left   = stackv(vst_left, self.vst_left)
        self.bv         = stackv(bv, self.bv)
