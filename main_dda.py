#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:32:40 2018

@author: jay
"""
import td_ada
import numpy as np

### Somewhat similar to coupling strength and dissipation from HNPs
gamma= 0.01
eps = 0.075
mu_x = 0.01
mu_y = 0.001
mu_z = 0.02
N = 6
dt = 0.05
tau = 150.
### Not sure if this is needed r  = np.zeros((N,3))

### arrays of vector quantities for each particle
### x,y,z position of each particle
coords = np.zeros((N,3))
### dipole expectation value of each particle
muexp = np.zeros((N,3))
mu = np.zeros(3)
mutmp = np.zeros(3)
rtmp = np.zeros(3)
### density matrix of each particle, each
### 2x2 DM is unrolled as a vector
D = np.zeros((N,4),dtype=complex)
Dtmp = np.zeros((2,2),dtype=complex)

### Mu matrix - same for each particle
### only z-component for now 
MUZ= np.zeros((2,2))
MUZ[0][1] = mu_z
MUZ[1][0] = mu_z

### H0 matrix - same for each particle
H0 = np.zeros((2,2))
H0[1][1] = eps

### Vint matrix - will be computed on the fly
Vint = np.zeros((2,2))

### initialize each system in its ground state
for i in range(0,N):
    D[i][0] = 1.+0j

### coordinates of particle... this instance
### the particles are separated by 1.5 Angstroms along 
### the x-axis
coords[0][1] = 0
coords[1][1] = 0.5
coords[2][1] = 1.0
coords[3][1] = 1.5
coords[4][1] = 2.0
coords[5][1] = 2.5

mu[0] = mu_x
mu[1] = mu_y
mu[2] = mu_z
    
### compute center of mass of sysstem    
R_com = td_ada.COM(coords)
Nsteps = 10000
ez = np.zeros(Nsteps)
time = np.zeros(Nsteps)
r1 = np.zeros(Nsteps)
r2 = np.zeros(Nsteps)
r3 = np.zeros(Nsteps)
p_init = np.ones(Nsteps)
### Propagation loop
DMU= np.zeros((2,2),dtype=complex)

for i in range(0,Nsteps):
    ### update time
    time[i] = i*dt
    ### update electric field
    ez[i] = td_ada.EField(i*dt, tau)
    ### loop over particles and update
    ### density matrix for each in turn
    for j in range(0,N):
        ### silly way to copy the density matrix to Dtmp
        Dtmp[0][0] = D[j][0]
        Dtmp[0][1] = D[j][1]
        Dtmp[1][0] = D[j][2]
        Dtmp[1][1] = D[j][3]
        
        Vint[0][0]= 0
        Vint[0][1]= 0
        Vint[1][0] = 0
        Vint[1][1] = 0
        
        for k in range(0,N):
            if k!=j:
                mutmp[0] = muexp[k][0]
                mutmp[1] = muexp[k][1]
                mutmp[2] = muexp[k][2]
                rtmp = td_ada.SepVector(coords, k, j)
                #print("rtmp",rtmp)
                #print("mutmp",mutmp)
                Vtmp = td_ada.DipoleDipole(mu, mutmp, rtmp)
                Vint[0][1] = Vint[0][1] + Vtmp
                Vint[1][0] = Vint[1][0] + Vtmp
        #print("Vint",Vint)      
        ### update this DM
        Dtmp = td_ada.RK4(H0, MUZ, Vint, gamma, Dtmp, dt, dt*i, tau)
        #print(Dtmp) 
        ### get product of MUZ and Dtmp
        DMU = np.matmul(Dtmp, MUZ)
        muexp[j][0] = DMU[0][0] + DMU[1][1]
        
        ### copy back to master D
        D[j][0] = Dtmp[0][0]
        D[j][1] = Dtmp[0][1]
        D[j][2] = Dtmp[1][0]
        D[j][3] = Dtmp[1][1]
        #print(D)
        
    #r1[i] = np.real(muexp[0][0])
    #r2[i] = np.real(muexp[3][0])
    #r3[i] = np.real(muexp[4][0])
    #print(" t, D(t): ",i*dt, np.real(D[0][0]))
    r1[i] = np.real(D[0][0])
    r2[i] = np.real(D[3][0])
    r3[i] = np.real(D[4][0])
    
plt.plot(time, r1, 'red', time, r2, 'b--', time, r3, 'g--')
plt.show()

