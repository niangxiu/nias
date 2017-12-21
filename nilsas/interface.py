import numpy as np
from .utility import qr_transpose, remove_orth_projection, stackv

def adjoint_terminal_condition(M_modes, f_tmn):
    # inputs -  M_modes:    number of homogeneous adjoint
    #           m:          the dimension of the dynamical system
    #           f_tmn:      f at the end of the trajectory
    
    m = f_tmn.shape[0]
    assert M_modes <= m 

    W_  = np.random.rand(M_modes, m)
    W__ = W_ - np.dot(W_, f_tmn)[:,np.newaxis] * f_tmn / np.dot(f_tmn, f_tmn)
    w,_ = qr_transpose(W__)
    yst_tmn = f_tmn
    vst_tmn = np.zeros(m)

    return w, yst_tmn, vst_tmn


class Interface:

    def __init__(self):
        # Q <- w_right:             shape(K+1, M, m)
        # R:                        shape(K+1, M, M)
        # yst_left <- yst_right:    shape(K+1, m)
        # vst_left <- vst_right:    shape(K+1, m)
        # by, bv:                   shape(K+1, M)
        self.w_right    = np.array([])
        self.Q          = np.array([])
        self.R          = np.array([])
        self.yst_right  = np.array([])
        self.yst_left   = np.array([])
        self.by         = np.array([])
        self.vst_right  = np.array([])
        self.vst_left   = np.array([])
        self.bv         = np.array([])


    def terminal(self, M_modes, forward):
        # get the right-of-interface values at t = t_n = T
        assert self.w_right.shape == self.yst_right.shape == self.vst_right.shape == (0,)
        w_right, yst_right, vst_right = adjoint_terminal_condition(M_modes, forward.f[-1,-1])

        Q, R         = qr_transpose(w_right)
        yst_left, by = remove_orth_projection(yst_right, Q)
        vst_left, bv = remove_orth_projection(vst_right, Q)
        assert np.allclose(Q, w_right)
        assert np.allclose(yst_left, yst_right)
        assert np.allclose(vst_left, vst_right)

        self.w_right    = stackv(w_right,   self.w_right)
        self.yst_right  = stackv(yst_right, self.yst_right)
        self.vst_right  = stackv(vst_right, self.vst_right)
        self.Q          = stackv(Q,         self.Q)
        self.R          = stackv(R,         self.R)
        self.yst_left   = stackv(yst_left,  self.yst_left)
        self.by         = stackv(by,        self.by)
        self.vst_left   = stackv(vst_left,  self.vst_left)
        self.bv         = stackv(bv,        self.bv)

        assert self.w_right.shape[0] == 1
        assert self.w_right.ndim == 3


    def interface_right(self, segment ):
        # get the right-of-interface values at t_i < T
        # may not need this function if computer memory is running short
        w_right         = segment.w[0,0]
        yst_right       = segment.yst[0,0]
        vst_right       = segment.vst[0,0]
        self.w_right    = stackv(w_right,   self.w_right)
        self.yst_right  = stackv(yst_right, self.yst_right)
        self.vst_right  = stackv(vst_right, self.vst_right)


    def rescale(self, w_right=None, yst_right=None, vst_right=None):
        # compute left-of-interface values from right-of-interface values
        if w_right is None:
            Q, R         = qr_transpose(self.w_right[0])
            yst_left, by = remove_orth_projection(self.yst_right[0], Q)
            vst_left, bv = remove_orth_projection(self.vst_right[0], Q)
        else:
            Q, R         = qr_transpose(w_right)
            yst_left, by = remove_orth_projection(yst_right, Q)
            vst_left, bv = remove_orth_projection(vst_right, Q)

        self.Q          = stackv(Q,         self.Q)
        self.R          = stackv(R,         self.R)
        self.yst_left   = stackv(yst_left,  self.yst_left)
        self.by         = stackv(by,        self.by)
        self.vst_left   = stackv(vst_left,  self.vst_left)
        self.bv         = stackv(bv,        self.bv)
