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
def L_Diss(D, gam):
    dim = len(D)
    LD = np.zeros_like(D)
    ### need |g><g|
    bra_1 = CreateBas(dim, 0)
    gm = Form_Rho(bra_1)
    
    for k in range(1,dim):
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

### assumes photon basis states are Psi[0][idx]
def H_Photon(Psi, omega):
    dim = len(Psi[0])
    ### create an array of zeros for the Hamiltonian to be filled
    Hp = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            ### first check to make sure molecule and photon basis states match
            if (Psi[1][i]==Psi[1][j] and Psi[0][i]==Psi[0][j]):
                ### evaluate a-hat|Psi[0][j]>
                if (Psi[0][j]==0):
                    Hp[i][j] = omega/2.
                else:
                    Hp[i][j] = omega/2. + np.sqrt(Psi[0][j])*omega
    
    return Hp
                    
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
    
    
def H_Interaction(Psi, g):
    dim = len(Psi[0])
    ### hg (sig^+ a + sig^- a^+ )
    ### create an array of zeros for the Hamiltonian to be filled
    Hp = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            pbra = Psi[0][i]
            mbra = Psi[1][i] 
            ### for term 1, action of sig^- a^+ |Psi[][j]>
            if (Psi[1][j]>0):
              pket = Psi[0][j]+1
              mket = Psi[1][j]-1
              t1 = np.sqrt(pket)*g
              if (pbra==pket and mbra==mket):
                  Hp[i][j] = Hp[i][j] + t1
                  
            ### for term 2, action of sig^+ a |Psi[][j]>
            if (Psi[0][j]>0):
                pket = Psi[0][j]-1
                mket = Psi[1][j]+1
                t2 = np.sqrt(pket+1)*g
                if (pbra==pket and mbra==mket):
                    Hp[i][j] = Hp[i][j] + t2
    
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
    k1 = h*DDot(H1,D1) + h*L_Diss(D1, 0.001) + h*L_Deph(D1, 0.2)
    
    ### Update H and D and get k2
    H2 = Hamiltonian(H, t+h/2)
    D2 = D+k1/2.
    k2 = h*DDot(H2, D2) + h*L_Diss(D2, 0.001) + h*L_Deph(D2, 0.2)
    
    ### UPdate H and D and get k3
    H3 = H2
    D3 = D+k2/2
    k3 = h*DDot(H3, D3) + h*L_Diss(D3, 0.001) + h*L_Deph(D3, 0.2)
    
    ### Update H and D and get K4
    H4 = Hamiltonian(H, t+h)
    D4 = D+k3
    k4 = h*DDot(H4, D4) + h*L_Diss(D4, 0.001) + h*L_Deph(D4, 0.2)
    
    Df = D + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4)
    return Df


Psi = np.zeros((2,6))
print(Psi)

idx = 0
for i in range(0,3):
    for j in range(0,2):
        Psi[0][idx] = i
        Psi[1][idx] = j
        idx = idx+1
        
print(Psi)
Hphot = H_Photon(Psi, 0.5)
print(Hphot)

Hmol = H_Molecule(Psi, 0.5)
print(Hmol)
  
Hi = H_Interaction(Psi, 0.5)
print(Hi)

Htot = Hphot + Hmol + Hi
Ntime = 4000
p1 = np.zeros(Ntime,dtype=complex)
p2 = np.zeros(Ntime,dtype=complex)
p3 = np.zeros(Ntime,dtype=complex)
p4 = np.zeros(Ntime,dtype=complex)
p5 = np.zeros(Ntime,dtype=complex)
p6 = np.zeros(Ntime,dtype=complex)
t = np.zeros(Ntime)

Psi = np.zeros(6, dtype=complex)

Psi[0] = np.sqrt(0.5/10)+0j
Psi[1] = np.sqrt(1.5/10)+0j
Psi[2] = np.sqrt(2./10)+0j
Psi[3] = np.sqrt(1.5/10)+0j
Psi[4] = np.sqrt(4./10)+0j
Psi[5] = np.sqrt(0.5/10)+0j
D = Form_Rho(Psi)

bra_1 = CreateBas(6, 1)

rho_1 = Form_Rho(bra_1)



for i in range(0,Ntime):
    Dp1 = RK4(Htot, D, 0.01, i*0.01)
    t[i] = 0.01*i
    p1[i] = Dp1[0][0]
    p2[i] = Dp1[1][1]
    p3[i] = Dp1[2][2]
    p4[i] = Dp1[3][3]
    p5[i] = Dp1[4][4]
    p6[i] = Dp1[5][5]
    D = Dp1
    ##trp = np.trace(D)
    ##print(np.real(trp))


plt.plot(t, np.real(p1), 'black', t, np.real(p2), 'red', t, np.real(p3), 'blue', t, np.real(p4), 'green', t, np.real(p5), 'purple', t, np.real(p6), 'orange')
plt.show()

