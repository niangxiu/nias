from __future__ import division
import numpy as np
import sys

sys.path.append("../")
from nilsas import *
from apps.lorenz import solve_primal, solve_adjoint 

# from nilsas.utility import qr_transpose
# A = np.array(np.random.rand(4,6))
# Q, R = qr_transpose(A)

u, f, J, fu, Ju, fs = solve_primal([0,1,5], 10)

from nilsas.nilsas import adjoint_terminal_condition
w_tmn, yst_tmn, vst_tmn = adjoint_terminal_condition(2, f[-1])
# print(w_tmn, yst_tmn, vst_tmn)

w, yst, vst = solve_adjoint(w_tmn, yst_tmn, vst_tmn, fu, fs)
# print(w)
# print(yst)
# print(vst)
