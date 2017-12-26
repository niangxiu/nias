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
    # p is either ystar or vstar, shape (nstep_per_segment, m)
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
        # w:        shape(K, nstep_per_segment, M, m)  
        # yst, vst: shape(K, nstep_per_segment, m)
        # C:        shape(K, M, M)
        # dy, dv_:  shape(K, M)
        
        self.w      = np.array([])
        self.yst    = np.array([])
        self.vst    = np.array([])
        self.C      = np.array([])
        self.dy     = np.array([])
        self.dv_    = np.array([]) # not the pm component, only for debug 
        # self.y, vstpm, dv, vpm, v


    def run1seg(self, run_adjoint, interface, forward, dt):
        j_current_segment = -(self.w.shape[0] + 1)

        w_tmn   = interface.Q[0]
        yst_tmn = interface.yst_left[0]
        vst_tmn = interface.vst_left[0]
        fu      = forward.fu[j_current_segment]
        Ju      = forward.Ju[j_current_segment]

        w, yst, vst = run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju, dt)
        C   = get_C_cts(w)
        dy  = get_d_cts(w, yst)
        dv_ = get_d_cts(w, vst)
   
        self.w      = stackv(w,     self.w)
        self.yst    = stackv(yst,   self.yst)
        self.vst    = stackv(vst,   self.vst)
        self.C      = stackv(C,     self.C)
        self.dy     = stackv(dy,    self.dy)
        self.dv_    = stackv(dv_,   self.dv_)

    
    def y_vstpm_dv(self, ay, f):
        assert ay.shape == self.dy.shape
        assert f.shape  == self.vst.shape

        self.y      = self.yst \
                + (self.w * ay[:,np.newaxis,:,np.newaxis]).sum(axis=-2)
        self.vstpm  = self.vst \
                - ((f*self.vst).sum(axis=-1) / (f*self.y).sum(axis=-1))[:,:,np.newaxis] * self.y
        self.dv     = get_d_cts_all_seg(self.w, self.vstpm)


    def vpm_v(self, av, f, Jtild):
        assert av.shape == self.dy.shape
        self.vpm    = self.vstpm \
                + (self.w * av[:,np.newaxis,:,np.newaxis]).sum(axis=-2)
        self.v      = self.vpm \
                - (Jtild / (f*self.y).sum(axis=-1))[:,:,np.newaxis] * self.y
