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
tau = 150.
### Not sure if this is needed r  = np.zeros((N,3))

### arrays of vector quantities for each particle
### x,y,z position of each particle
coords = np.zeros((N,3))
### dipole expectation value of each particle
muexp = np.zeros((N,3))
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
coords[0][0] = 0
coords[1][0] = 1.5e-10
coords[2][0] = 3.0e-10


for i in range(0,N):
    mu[i][0] = mu_x
    mu[i][1] = mu_y
    mu[i][2] = mu_z
    
### compute center of mass of sysstem    
R_com = COM(coords)

ez = np.zeros(10000)
time = np.zeros(10000)
r1 = np.zeros(10000)
r2 = np.zeros(10000)
r3 = np.zeros(10000)
### Propagation loop
for i in range(0,10000):
    ### update time
    time[i] = i*dt
    ### update electric field
    ez[i] = EField(i*dt, tau)
    ### loop over particles and update
    ### density matrix for each in turn
    for j in range(0,N):
        ### silly way to copy the density matrix to Dtmp
        Dtmp[0][0] = D[j][0]
        Dtmp[0][1] = D[j][1]
        Dtmp[1][0] = D[j][2]
        Dtmp[1][1] = D[j][3]
        
        ### update this DM
        Dtmp = RK4(H0, MUZ, Vint, gamma, Dtmp, dt, dt*i)
        
        ### copy back to master D
        D[j][0] = Dtmp[0][0]
        D[j][1] = Dtmp[0][1]
        D[j][2] = Dtmp[1][0]
        D[j][3] = Dtmp[1][1]
        
    r1[i] = np.real(D[0][3])
    r2[i] = np.real(D[1][3])
    r3[i] = np.real(D[2][3])
    
    
    
plt.plot(time, r1, 'red', time, r2, 'blue', time, r3, 'green')
plt.show()

