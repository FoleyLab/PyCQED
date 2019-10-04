#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:32:40 2018

@author: jay
"""
import td_ada
import numpy as np
from matplotlib import pyplot as plt
### Somewhat similar to coupling strength and dissipation from HNPs
#gamma= 0.01
#eps = 0.075
#mu_x = 0.01
#mu_y = 0.001
#mu_z = 0.02
#### Parameters for r=2.5 nm gold NP
mu_x = 0
mu_y = 0
mu_z = 59.5
eps = 0.0850
gamma = 0.007382

N = 7
dt = 0.5
tau = 150.
### Not sure if this is needed r  = np.zeros((N,3))

### arrays of vector quantities for each particle
### x,y,z position of each particle
coords = np.zeros((N,3))
disp_com = np.zeros((N,3))
### dipole expectation value of each particle
muexp = np.zeros((N,3),dtype=complex)
mu = np.zeros(3,dtype=complex)
mutmp = np.zeros(3,dtype=complex)
rtmp = np.zeros(3)
### density matrix of each particle, each
### 2x2 DM is unrolled as a vector
D = np.zeros((N,4),dtype=complex)
Dtmp = np.zeros((2,2),dtype=complex)

### Mu matrix - same for each particle
### only z-component for now 
MUZ= np.zeros((2,2),dtype=complex)
MUZ[0][1] = mu_z
MUZ[1][0] = mu_z

### H0 matrix - same for each particle
H0 = np.zeros((2,2))
H0[1][1] = eps

### Vint matrix - will be computed on the fly
Vint = np.zeros((2,2),dtype=complex)

### initialize each system in its ground state
for i in range(0,N):
    D[i][0] = 1.+0j

### coordinates of particle... this instance
### the particles are separated by 1.5 Angstroms along 
### the x-axis
coords[0][2] = -240.
coords[1][2] = -160.
coords[2][2] = -80
coords[3][2] =  0
coords[4][2] =  80
coords[5][2] =  160
coords[6][2] = 240

mu[0] = mu_x
mu[1] = mu_y
mu[2] = mu_z
    
### compute center of mass of sysstem    
R_com = td_ada.COM(coords)
### compute displacement vectors from COM for all atoms
for i in range(0,N):
    disp_com[i][0] = coords[i][0] - R_com[0]
    disp_com[i][1] = coords[i][1] - R_com[1]
    disp_com[i][2] = coords[i][2] - R_com[2]

print("R com is ",R_com)
print("disp_com is ",disp_com)    
Nsteps = 5000
ez = np.zeros(Nsteps)
### array for total dipole moment in COM frame
mu_of_t = np.zeros(Nsteps,dtype=complex)
mu_com_temp = 0.+0j
time = np.zeros(Nsteps)
energy = np.zeros(Nsteps)
r1 = np.zeros(Nsteps)
r2 = np.zeros(Nsteps)
r3 = np.zeros(Nsteps)
p_init = np.ones(Nsteps)

### Propagation loop
DMU= np.zeros((2,2),dtype=complex)

for i in range(1,Nsteps):
    energy[i] = np.pi*2*i/(Nsteps*dt)
    ### update time
    time[i] = i*dt
    ### update electric field
    ez[i] = td_ada.EField(i*dt, tau)
    ### loop over particles and update
    ### density matrix for each in turn
    mu_com_temp = 0+0j
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
                #print("mu ",mu)
                Vtmp = td_ada.DipoleDipole(mu, mutmp, rtmp)
                Vint[0][1] = Vint[0][1] + Vtmp
                Vint[1][0] = Vint[1][0] + Vtmp
        #print("Vint",Vint)      
        ### update this DM
        Dtmp = td_ada.RK4(H0, MUZ, Vint, gamma, Dtmp, dt, dt*i, tau)
        #print(Dtmp) 
        ### get product of MUZ and Dtmp
        DMU = np.matmul(Dtmp, MUZ)
        muexp[j][2] = DMU[0][0] + DMU[1][1]
        mu_com_temp = mu_com_temp + (disp_com[j][2] + muexp[j][2])
        #print("mu_com",mu_com_temp)
        
        ### copy back to master D
        D[j][0] = Dtmp[0][0]
        D[j][1] = Dtmp[0][1]
        D[j][2] = Dtmp[1][0]
        D[j][3] = Dtmp[1][1]
        #print(D)
        
    r1[i] = np.real(muexp[0][2])
    r2[i] = np.real(muexp[1][2])
    r3[i] = np.real(muexp[2][2])
    #print(" t, D(t): ",i*dt, np.real(D[0][0]))
    mu_of_t[i] = mu_com_temp
    '''
    r1[i] = np.real(D[0][0])
    r2[i] = np.real(D[3][0])
    r3[i] = np.real(D[4][0])

    '''
    
alpha = np.fft.fft(mu_of_t)/np.fft.fft(ez)
#omega = np.fft.fftfreq(time)

plt.plot(energy, alpha*np.conj(alpha), 'red')
plt.xlim(0,0.1)
#plt.ylim(0,100)
#plt.show()
#plt.plot(time, r1, 'red', time, r2, 'b--',time, r3, 'purple', time, mu_of_t, 'g--')
# time, r3, 'g--', time, mu_of_t/20., 'black' )
plt.show()

