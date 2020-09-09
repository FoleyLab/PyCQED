### Import all libraries and define various parameters here!
import numpy as np
from numpy.lib import scimath as SM

w0 = 2.45/27.211
Nv = 100
hg = np.linspace(0.01, 0.5, Nv)
hgam = np.linspace(0.1, 100, Nv)

mag_abs = np.zeros((Nv,Nv))
mag_re = np.zeros((Nv,Nv))
mag_im = np.zeros((Nv,Nv))
ci = 0+1j
for i in range(0, Nv):
    c_val = hg[i] / 27.211
    print(" ")
    for j in range(0, Nv):
        gamma_val = hgam[j] * 1e-3 / 27.211

        Ep = w0 - ci*gamma_val/4 + SM.sqrt(c_val**2 - gamma_val**2/16.)
        Em = w0 - ci*gamma_val/4 - SM.sqrt(c_val**2 - gamma_val**2/16.)
        dE = (Ep-Em)
        print(gamma_val, c_val, np.real(Ep), np.imag(Ep), np.real(dE), np.imag(dE), np.abs(dE))
