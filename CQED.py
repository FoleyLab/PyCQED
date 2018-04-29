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
    k1 = h*DDot(H1,D1)
    
    ### Update H and D and get k2
    H2 = Hamiltonian(H, t+h/2)
    D2 = D+k1/2.
    k2 = h*DDot(H2, D2)
    
    ### UPdate H and D and get k3
    H3 = H2
    D3 = D+k2/2
    k3 = h*DDot(H3, D3)
    
    ### Update H and D and get K4
    H4 = Hamiltonian(H, t+h)
    D4 = D+k3
    k4 = h*DDot(H4, D4)
    
    Df = D + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4)
    return Df


Psi = np.zeros((2,4))
print(Psi)

idx = 0
for i in range(0,2):
    for j in range(0,2):
        Psi[0][idx] = i
        Psi[1][idx] = j
        idx = idx+1
        

Hphot = H_Photon(Psi, 0.5)
print(Hphot)

Hmol = H_Molecule(Psi, 0.5)
print(Hmol)
  
Hi = H_Interaction(Psi, 0.2)
print(Hi)

Htot = Hphot + Hmol + Hi
Ntime = 4000
p23 = np.zeros(Ntime,dtype=complex)
p2 = np.zeros(Ntime,dtype=complex)
p3 = np.zeros(Ntime,dtype=complex)
p32 = np.zeros(Ntime,dtype=complex)
t = np.zeros(Ntime)

D = np.zeros((4,4),dtype=complex)
D[1][1] = 1.+0j
for i in range(0,Ntime):
    Dp1 = RK4(Htot, D, 0.01, i*0.01)
    t[i] = 0.01*i
    p23[i] = Dp1[1][2]
    p2[i] = Dp1[1][1]
    p3[i] = Dp1[2][2]
    p32[i] = Dp1[2][1]
    D = Dp1


plt.plot(t, np.imag(p23), 'red', t, np.real(p3), 'blue')
plt.show()
    