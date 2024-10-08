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

### give particle index i and j, return
### separation vector between the two
def SepVector(coords, i, j):
    r = np.zeros(3)
    for l in range(0,3):
        r[l] = coords[j][l] - coords[i][l]
    return r

def COM(coords):
    ### all masses are equal, so we can
    ### just say total mass is equal to the number of particles
    ### and that each particle has mass 1
    M = len(coords)
    R = np.zeros(3)
    for l in range(0,len(coords)):
        for j in range(0,3):
            R[j] = R[j] + coords[l][j]
    return R/M

### Function to compute dipole-dipole coupling element
def DipoleDipole(mu_i, mu_j, r):

  ## refractive index of vaccuum
  n = 1.
  normr = math.sqrt(np.dot(r,r))
  oer2 = 1./(normr**2)
  pre = 1./(n**2 * normr**3)
  
  t1 = np.dot(mu_i,mu_j)
  t2 = np.dot(mu_i,r)
  t3 = np.dot(r,mu_j)
  
  Vint = pre*(t1 - 3*oer2*t2*t3)

  return Vint

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
        bra_k = CreateBas(dim, k)
        km = Form_Rho(bra_k)
        
        ### first term 2*gam*<k|D|k>|g><g|
        t1 = 2*gamma*D[k][k]*gm
        ### second term is |k><k|*D
        t2 = np.dot(km,D)
        ### third term is  D*|k><k|
        t3 = np.dot(D, km)
        LD = LD + t1 - gamma*t2 - gamma*t3
        
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

def EField(t, tau):
    Ef = 0.
    if t<tau:
        Ef = 0.00001*np.sin(t*np.pi/tau)*np.sin(t*np.pi/tau)*np.sin(0.07423*t)
    return Ef

## Interaction is going to require the coordinates of atom i and atom j
## along with dipole moment expectation value of atom j and dipole matrix
## of atom i    
def H_Interaction(Psi, g):

    return Hp


### RK4 routine
def RK4(H0, mu, Vint, gamma, D, h, t, tau):
    k1 = np.zeros_like(D)
    k2 = np.zeros_like(D)
    k3 = np.zeros_like(D)
    k4 = np.zeros_like(D)
    D1 = np.zeros_like(D)
    D2 = np.zeros_like(D)
    D3 = np.zeros_like(D)
    D4 = np.zeros_like(D)
    Df = np.zeros_like(D)
    
    ### Get k1
    H1 = H0 - EField(t, tau)*mu + Vint
    D1 = D    
    k1 = h*DDot(H1,D1) + h*L_Diss(D1, gamma)
    
    ## Update H and D and get k2
    H2 = H0 - EField(t+h/2, tau)*mu + Vint
    D2 = D+k1/2.
    k2 = h*DDot(H2, D2) + h*L_Diss(D2, gamma)
    
    ### UPdate H and D and get k3
    H3 = H2
    D3 = D+k2/2
    k3 = h*DDot(H3, D3) + h*L_Diss(D3, gamma) 
    
    ### Update H and D and get K4
    H4 = H0 - EField(t+h, tau)*mu + Vint
    D4 = D+k3
    k4 = h*DDot(H4, D4) + h*L_Diss(D4, gamma)
    
    Df = D + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4)
    return Df


'''   
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

Psi = np.zeros((2,4))

### Define basis states
idx = 0
for i in range(0,2):
    for j in range(0,2):
        Psi[0][idx] = i
        Psi[1][idx] = j
        idx = idx+1
        
### Print basis states for verificatio
print(Psi)
### Form photon Hamiltonian in basis of Psi
Hphot = H_Photon(Psi, en_hnp)

### Form molecular Hamiltonian in basis of Psi
Hmol = H_Molecule(Psi, en_mol)
print("Hhnp")
print(Hphot)
print("Hmolecule")
print(Hmol)


Hi = H_Interaction(Psi, g_coup)
Htot = Hphot + Hmol + Hi
vals, vecs = LA.eig(Htot)
idx = vals.argsort()[::1]
vals = vals[idx]
v = vecs[:,idx]

  
Hdiag = Transform(v, Htot)
print(Hdiag)


Ntime = 10000
p1 = np.zeros(Ntime,dtype=complex)
p2 = np.zeros(Ntime,dtype=complex)
p3 = np.zeros(Ntime,dtype=complex)
p4 = np.zeros(Ntime,dtype=complex)
#p5 = np.zeros(Ntime,dtype=complex)
#p6 = np.zeros(Ntime,dtype=complex)
pd1 = np.zeros(Ntime,dtype=complex)
pd2 = np.zeros(Ntime,dtype=complex)
pd3 = np.zeros(Ntime,dtype=complex)
pd4 = np.zeros(Ntime,dtype=complex)
#pd5 = np.zeros(Ntime,dtype=complex)
#pd6 = np.zeros(Ntime,dtype=complex)
t = np.zeros(Ntime)

Psi = np.zeros(4, dtype=complex)

#Psi = np.sqrt(1/4)*v[0]+np.sqrt(1/4)*v[1]+np.sqrt(1/4)*v[2] + np.sqrt(1/4)*v[3]
#Psi[1] = np.sqrt(2/6)
#Psi[2] = np.sqrt(3/6)
#Psi[3] = np.sqrt(1/6)
Psi[1] = np.sqrt(1.)
#Psi[2] = np.sqrt(1./3)
#Psi[3] = np.sqrt(1.)


print(Psi)

Psi[0] = np.sqrt(0.5/7)+0j
Psi[1] = np.sqrt(1.5/7)+0j
Psi[2] = np.sqrt(2.0/7)+0j
Psi[3] = np.sqrt(3.0/7)+0j

#Psi[4] = np.sqrt(3.5/10)+0j
#Psi[5] = np.sqrt(1./10)+0j


D = Form_Rho(Psi)

bra_1 = CreateBas(4, 1)

rho_1 = Form_Rho(bra_1)

# Should diagonalize and show populations in diagonalized basis
#    to see if dynamics are dramatically different 
#
 
for i in range(0,Ntime):
    Dp1 = RK4(Htot, D, dt, i*dt)
    t[i] = dt*i
    p1[i] = Dp1[0][0]
    p2[i] = Dp1[1][1]
    p3[i] = Dp1[2][2]
    p4[i] = Dp1[3][3]
 #   p5[i] = Dp1[4][4]
 #   p6[i] = Dp1[5][5]
    D = Dp1
    DD = Transform(v, Dp1)
    pd1[i] = DD[0][0]
    pd2[i] = DD[1][1]
    pd3[i] = DD[2][2]
    pd4[i] = DD[3][3]
    print(t[i],np.real(p1[i]),np.real(p2[i]),np.real(p3[i]),np.real(p4[i]),np.real(pd1[i]),np.real(pd2[i]),np.real(pd3[i]),np.real(pd4[i]))
#    pd5[i] = DD[4][4]
#    pd6[i] = DD[5][5]
    ##trp = np.trace(D)
    ##print(np.real(trp))


plt.plot(t, np.real(p1), 'pink', t, np.real(p2), 'r--', t, np.real(p3), 'b--', t, np.real(p4), 'purple') # t, np.real(p5), 'purple', t, np.real(p6), 'orange')
plt.plot(t, np.real(pd1), 'black', t, np.real(pd2), 'r--', t, np.real(pd3), 'blue', t, np.real(pd4), 'g--') # t, np.real(pd5), 'purple', t, np.real(pd6), 'orange')
plt.show()

#np.savetxt("time.txt",t)
#np.savetxt("p1_g0.07_ghnp_0.0001_gdeph_0.005.txt",np.real(pd1))
#print("Try higher photon numnber states!!!")
for i in range(0,len(p1)):
    if pd1[i]>0.5:
        print("half life is ",t[i])
        print("half life in fs is ",0.0248*t[i])
        print("decay constant in a.u. is ",0.5*np.log(2.)/t[i])
        print("decay constant in ev   is ",0.5*27*np.log(2.)/t[i])
        break
    
'''
