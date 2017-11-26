from numpy import *
from matplotlib.pyplot import *


def ddt(uwvs):
    u = uwvs[0]
    [x, y, z] = u
    w = uwvs[1]
    vstar = uwvs[2]
    dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    Df = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
    dwdt = np.dot(Df, w.T)
    dfdrho = np.array([0, x, 0])
    dvstardt = np.dot(Df, vstar) + dfdrho
    return np.array([dudt, dwdt.T, dvstardt])

def RK1(u, w, vstar):
    # integrate u, w, and vstar to the next time step
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs) 
    uwvs_new = uwvs + k0 
    return uwvs_new

def RK4(u, w, vstar):
    # integrate u, w, and vstar to the next time step
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs) 
    k1 = dt * ddt(uwvs + 0.5 * k0)
    k2 = dt * ddt(uwvs + 0.5 * k1)
    k3 = dt * ddt(uwvs + k2)
    uwvs_new = uwvs + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return uwvs_new


