#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:32:40 2018

@author: jay
"""
import td_ada

### Somewhat similar to coupling strength and dissipation from HNPs
gamma= 0.01
eps = 0.075
mu_x = 0.01
mu_y = 0.001
mu_z = 0.02
N = 3
dt = 0.1
mu = np.zeros((N,3))
r  = np.zeros((N,3))
coords = np.zeros((N,3))
### coordinates of particles stored in a dictionary
coords[0][0] = 0
coords[1][0] = 1.5e-10
coords[2][0] = 3.0e-10

for i in range(0,N):
    mu[i][0] = mu_x
    mu[i][1] = mu_y
    mu[i][2] = mu_zz
    
