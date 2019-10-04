#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

### Constants
kw = 1.011e-14

### Variables
### Concentration and volume of strong acid
ca = 0.1
va = 1.
### Concentrations of strong bases
cb_c = 0.2
cb_d = 0.1

### array of pH values
pH = np.linspace(1, 12.2, 100)

### Empty arrays of volumes of dilute and con. base
### to achieve a given pH
vb_d = np.zeros_like(pH)
vb_c = np.zeros_like(pH)

### Function to compute pH curve
def TT(Ca, Cb, Va, pH):
    H = 10**(-1*pH)
    Vb = (-H**2 + Ca*H + kw)/(H**2 + Cb*H - kw) * Va
    return Vb
    
### Loop to cal TT function for each pH
for i in range(0,len(pH)):
    vb_d[i] = TT(ca, cb_d, va, pH[i])
    vb_c[i] = TT(ca, cb_c, va, pH[i])

### Plot titration curves
plt.plot(vb_c, pH, 'red', vb_d, pH, 'blue')
plt.xlabel("Volume of Base")
plt.ylabel("pH")
plt.show()    