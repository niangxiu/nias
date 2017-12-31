import numpy as np
from .utility import stackv

def get_C_cts(w):
    # compute the covariant matrix using w on all time steps
    # w is on a segment, shape (nstep_per_segment, M, m)
    assert w.ndim == 3
    assert w.shape[1] <= w.shape[2]
    M_modes = w.shape[1]
    C = np.zeros([M_modes, M_modes])
    for i in range(M_modes):
        for j in range(M_modes):
            C[i,j] = np.sum( w[:, i, :] * w[:, j, :] )
            C[j,i] = C[i,j]
    return C


def get_d_cts(w, p):
    # p is either vstar, shape (nstep_per_segment, m)
    assert p.ndim == 2
    assert w.shape[2] == p.shape[1]
    M_modes = w.shape[1]
    d = np.zeros(M_modes)
    for i in range(M_modes):
        d[i] = np.sum( w[:, i, :] * p )
    return d


def get_d_cts_all_seg(w, p):
    # p: shape (K_segment, nstep_per_segment, m)
    # w: shape (K_segment, nstep_per_segment, M, m)
    assert p.ndim == 3 and w.ndim == 4
    K_segment = p.shape[0]
    M_modes = w.shape[-2]
    d = np.zeros([K_segment, M_modes])
    for i in range(K_segment):
        d[i] = get_d_cts(w[i], p[i])
    return d


class Segment:

    def __init__(self):
        # w: shape(K, nstep_per_segment, M, m)  
        # vst, v: shape(K, nstep_per_segment, m)
        # C: shape(K, M, M)
        # dv: shape(K, M)
        
        self.w = np.array([])
        self.vst = np.array([])
        self.C = np.array([])
        self.dv = np.array([]) 


    def run1seg(self, run_adjoint, interface, forward, dt, stepfunc):
        j_current_segment = -(self.w.shape[0] + 1)

        w_tmn   = interface.Q[0]
        vst_tmn = interface.vst_left[0]
        fu      = forward.fu[j_current_segment]
        Ju      = forward.Ju[j_current_segment]

        w, vst = run_adjoint(w_tmn, vst_tmn, fu, Ju, dt, stepfunc)
        C = get_C_cts(w)
        dv = get_d_cts(w, vst)
   
        self.w = stackv(w,     self.w)
        self.vst = stackv(vst,   self.vst)
        self.C = stackv(C,     self.C)
        self.dv = stackv(dv,   self.dv)

    
    def get_v(self, av, f, Jtild):
        assert av.shape == self.dy.shape
        self.v = self.vst\
                + (self.w * av[:,np.newaxis,:,np.newaxis]).sum(axis=-2)
