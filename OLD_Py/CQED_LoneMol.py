#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:00:57 2018

@author: jay
"""

import numpy as np
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt

gam_diss_m = 0.0012
gam_deph_m = 0.001

en_mol = 0.075
dt = 0.1
gamma = np.zeros(2)
gamma[0] = 0.
gamma[1] = gam_diss_m
gam_deph = gam_deph_m

## Transform operators to basis that diagonalizes Htot
def Transform(v, O):
    vtO = np.dot(LA.inv(v),O)
    Odiag = np.dot(vtO,v)
    return Odiag


### should create a function that takes a wavefunction in vector
### format and computes a density matrix
def Form_Rho(Psi):

    D = np.outer(Psi,np.conj(Psi))
    return D

### Creates basis vector for state k
### k=0 -> ground state, k=1 -> first excited-state, etc
def CreateBas(dim, k):
    bas = np.zeros(dim)
    bas[k] = 1
    return bas

### Lindblad operator that models relaxation to the ground state
def L_Diss(D, gamma):
    dim = len(D)
    LD = np.zeros_like(D)
    ### need |g><g|
    bra_1 = CreateBas(dim, 0)
    gm = Form_Rho(bra_1)
    
    for k in range(1,dim):
        gam = gamma[k]
        bra_k = CreateBas(dim, k)
        km = Form_Rho(bra_k)
        
        ### first term 2*gam*<k|D|k>|g><g|
        t1 = 2*gam*D[k][k]*gm
        ### second term is |k><k|*D
        t2 = np.dot(km,D)
        ### third term is  D*|k><k|
        t3 = np.dot(D, km)
        LD = LD + t1 - gam*t2 - gam*t3
        
    return LD

### Lindblad operator that models dephasing
def L_Deph(D, gam):
    dim = len(D)
    LD = np.zeros_like(D)
    
    for k in range(1,dim):
        bra_k = CreateBas(dim, k)
        km = Form_Rho(bra_k)
        
        ### first term 2*gam*<k|D|k>|k><k|
        t1 = 2*gam*D[k][k]*km
        ### second term is |k><k|*D
        t2 = np.dot(km,D)
        ### third term is  D*|k><k|
        t3 = np.dot(D, km)
        LD = LD + t1 - gam*t2 - gam*t3
        
    return LD


### Take commutator of H and D to give Ddot
def DDot(H, D):
    ci = 0.+1j
    return -ci*(np.dot(H,D) - np.dot(D, H))

def Hamiltonian(H, t):
    return H

                    
### assumes molecule basis states are Psi[1][idx]
def H_Molecule(Psi, omega):
    dim = len(Psi[0])

    ### create an array of zeros for the Hamiltonian to be filled
    Hp = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            ### first check to make sure molecule and photon basis states match
            if (Psi[1][i]==Psi[1][j] and Psi[0][i]==Psi[0][j]):
                ### evaluate a-hat|Psi[0][j]>
                if (Psi[1][j]==0):
                    Hp[i][j] = -omega/2.
                else:
                    Hp[i][j] = omega/2.
    
    return Hp
    
    
    
def RK4(H, D, h, t):
    k1 = np.zeros_like(D)
    k2 = np.zeros_like(D)
    k3 = np.zeros_like(D)
    k4 = np.zeros_like(D)
    D1 = np.zeros_like(D)
    D2 = np.zeros_like(D)
    D3 = np.zeros_like(D)
    D4 = np.zeros_like(D)
    
    ### Get k1
    H1 = Hamiltonian(H, t)
    D1 = D    
    k1 = h*DDot(H1,D1) + h*L_Diss(D1, gamma) + h*L_Deph(D1, gam_deph)
    
    ### Update H and D and get k2
    H2 = Hamiltonian(H, t+h/2)
    D2 = D+k1/2.
    k2 = h*DDot(H2, D2) + h*L_Diss(D2, gamma) + h*L_Deph(D2, gam_deph)
    
    ### UPdate H and D and get k3
    H3 = H2
    D3 = D+k2/2
    k3 = h*DDot(H3, D3) + h*L_Diss(D3, gamma) + h*L_Deph(D3, gam_deph)
    
    ### Update H and D and get K4
    H4 = Hamiltonian(H, t+h)
    D4 = D+k3
    k4 = h*DDot(H4, D4) + h*L_Diss(D4, gamma) + h*L_Deph(D4, gam_deph)
    
    Df = D + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4)
    return Df

### Form Basis states
### Initialize basis vectors
#Psi = np.zeros(2)

en_mol = 0.075

Hmol = np.zeros((2,2))
Hmol[0][0] =  -1*en_mol/2.
Hmol[1][1] =  en_mol/2.


Ntime = 10000
p1 = np.zeros(Ntime,dtype=complex)
p2 = np.zeros(Ntime,dtype=complex)

#pd5 = np.zeros(Ntime,dtype=complex)
#pd6 = np.zeros(Ntime,dtype=complex)
t = np.zeros(Ntime)

Psi = np.zeros(2, dtype=complex)

#Psi = np.sqrt(1/4)*v[0]+np.sqrt(1/4)*v[1]+np.sqrt(1/4)*v[2] + np.sqrt(1/4)*v[3]
Psi[0] = 0.
Psi[1] = 1.

'''
print(Psi)

Psi[0] = np.sqrt(0.5/7)+0j
Psi[1] = np.sqrt(1.5/7)+0j
Psi[2] = np.sqrt(2.0/7)+0j
Psi[3] = np.sqrt(3.0/7)+0j

#Psi[4] = np.sqrt(3.5/10)+0j
#Psi[5] = np.sqrt(1./10)+0j
'''

D = Form_Rho(Psi)

# Should diagonalize and show populations in diagonalized basis
#    to see if dynamics are dramatically different 
#
 
for i in range(0,Ntime):
    Dp1 = RK4(Hmol, D, dt, i*dt)
    t[i] = dt*i
    p1[i] = Dp1[0][0]
    p2[i] = Dp1[1][1]
    print(t[i],np.real(p1[i]),np.real(p2[i]))
    D = Dp1


#plt.plot(t, np.real(p1), 'pink', t, np.real(p2), 'r--', t, np.real(p3), 'b--', t, np.real(p4), 'purple') # t, np.real(p5), 'purple', t, np.real(p6), 'orange')
plt.plot(t, np.real(p1), 'black', t, np.real(p2), 'r--') # t, np.real(pd3), 'blue', t, np.real(pd4), 'g--') # t, np.real(pd5), 'purple', t, np.real(pd6), 'orange')
plt.show()

#np.savetxt("pop_e.txt",np.real(p2))

for i in range(0,len(p1)):
    if p1[i]>0.5:
        print("half life in fs is ",0.0248*t[i])
        print("decay constant in a.u. is ",0.5*np.log(2.)/t[i])
        print("decay constant in ev   is ",0.5*27*np.log(2.)/t[i])
        break
