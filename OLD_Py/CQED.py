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

ci=0+1j
### Somewhat similar to coupling strength and dissipation from HNPs
gam_diss_np = 0.0001 #.001
gam_deph_np = 0 #.0001

gam_diss_m = 0 #.0012
gam_deph_m = 0 #.0001

en_hnp = 0.075 -0.0002*ci
en_mol = 0.075 #-0.01*ci

dt = 0.1
gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_m
gamma[2] = gam_diss_np
gamma[3] = gam_diss_m+gam_diss_np

#g_coup = 0.065
g_coup = 0.03
gam_deph = 0.001

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

### assumes photon basis states are Psi[0][idx]
def H_Photon(Psi, omega):
    dim = len(Psi[0])
    ### create an array of zeros for the Hamiltonian to be filled
    Hp = np.zeros((dim,dim),dtype=complex)
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
    Hp = np.zeros((dim,dim),dtype=complex)
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
    Hp = np.zeros((dim,dim),dtype=complex)
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

def RK4_NH_SE(H, C, h, V, dc, t):
    ci = 0+1j
    k1 = np.zeros_like(C)
    k2 = np.zeros_like(C)
    k3 = np.zeros_like(C)
    k4 = np.zeros_like(C)
    C1 = np.zeros_like(C)
    C2 = np.zeros_like(C)
    C3 = np.zeros_like(C)
    C4 = np.zeros_like(C)
    
    C1 = np.copy(C)    
    k1 = np.copy(-ci * h * np.dot(H, C1) - h * V * np.dot(dc, C1))
    
    C2 = np.copy(C+k1/2.)
    k2 = np.copy(-ci * h * np.dot(H, C2) - h * V * np.dot(dc, C2))
    #k2 = h*DDot(H2, D2) + h*L_Diss(D2, gamma) + h*L_Deph(D2, gam_deph)
    
    ### UPdate H and D and get k3
    C3 = np.copy(C+k2/2)
    k3 = np.copy(-ci * h * np.dot(H, C3) - h * V * np.dot(dc, C3))
    #k3 = h*DDot(H3, D3) + h*L_Diss(D3, gamma) + h*L_Deph(D3, gam_deph)
    
    ### Update H and D and get K4
    C4 = np.copy(C+k3)
    k4 = np.copy(-ci * h * np.dot(H, C4) - h * V * np.dot(dc, C4))
    #k4 = h*DDot(H4, D4) + h*L_Diss(D4, gamma) + h*L_Deph(D4, gam_deph)
    
    Cf = np.copy(C + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4))
    
    return Cf

def RK4(H, D, h, t):
    ci = 0+1j
    k1 = np.zeros_like(D)
    k2 = np.zeros_like(D)
    k3 = np.zeros_like(D)
    k4 = np.zeros_like(D)
    D1 = np.zeros_like(D)
    D2 = np.zeros_like(D)
    D3 = np.zeros_like(D)
    D4 = np.zeros_like(D)
    
    ### Get k1
    H1 = np.copy(np.real(H))
    G1 = np.copy(np.imag(H))
    
    D1 = np.copy(D)    
    k1 = np.copy(h*DDot(H1,D1)  + h*(np.dot(G1,D1) + np.dot(D1,G1))) # h*L_Diss(D1, gamma) + h*L_Deph(D1, gam_deph)
    #k1 = h*DDot(H1, D1) + h*L_Diss(D1, gamma) + h*L_Deph(D1, gam_deph)
    ### Update H and D and get k2
    H2 = np.copy(H)
    D2 = np.copy(D+k1/2.)
    k2 = np.copy(h*DDot(H1,D2)  + h*(np.dot(G1,D2) + np.dot(D2,G1)))
    #k2 = h*DDot(H2, D2) + h*L_Diss(D2, gamma) + h*L_Deph(D2, gam_deph)
    
    ### UPdate H and D and get k3
    H3 = np.copy(H2)
    D3 = np.copy(D+k2/2)
    k3 = np.copy(h*DDot(H1,D3)  + h*(np.dot(G1,D3) + np.dot(D3,G1)))
    #k3 = h*DDot(H3, D3) + h*L_Diss(D3, gamma) + h*L_Deph(D3, gam_deph)
    
    ### Update H and D and get K4
    H4 = np.copy(H)
    D4 = np.copy(D+k3)
    k4 = np.copy(h*DDot(H1,D4)  + h*(np.dot(G1,D4) + np.dot(D4,G1)))
    #k4 = h*DDot(H4, D4) + h*L_Diss(D4, gamma) + h*L_Deph(D4, gam_deph)
    
    Df = np.copy(D + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4))
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

print("idx is ",idx)
print("vals is ",vals)

  
Hdiag = Transform(v, Htot)
print(Hdiag)


Ntime = 100000
p1 = np.zeros(Ntime,dtype=complex)
p2 = np.zeros(Ntime,dtype=complex)
p3 = np.zeros(Ntime,dtype=complex)
p4 = np.zeros(Ntime,dtype=complex)

p1zp = np.zeros(Ntime,dtype=complex)
p2zp = np.zeros(Ntime,dtype=complex)
p3zp = np.zeros(Ntime,dtype=complex)
p4zp = np.zeros(Ntime,dtype=complex)

t = np.zeros(Ntime)

Psi = np.zeros(4, dtype=complex)
Psi_zp = np.zeros_like(Psi)


print(Psi)


Psi[2] = np.sqrt(1)+0j
Psi_zp[2] = np.sqrt(1)+0j

'''
D = Form_Rho(Psi)

bra_1 = CreateBas(4, 1)

rho_1 = Form_Rho(bra_1)
'''

# Should diagonalize and show populations in diagonalized basis
#    to see if dynamics are dramatically different 
#

V = 3.207337830262267e-05

Htot = np.array([[ 3.81084841e-02-3.03331099e-09j,  0.00000000e+00+0.00000000e+00j,
   0.00000000e+00+0.00000000e+00j,  0.00000000e+00-1.05879118e-22j],
 [ 0.00000000e+00+0.00000000e+00j,  1.27721221e-01-1.36906639e-04j,
   0.00000000e+00+8.67361738e-19j,  0.00000000e+00+0.00000000e+00j],
 [ 0.00000000e+00+0.00000000e+00j,  2.77555756e-17+8.67361738e-19j,
   1.29397650e-01-4.68425806e-05j,  0.00000000e+00+0.00000000e+00j],
 [-2.16840434e-19-7.94093388e-23j,  0.00000000e+00+0.00000000e+00j,
   0.00000000e+00+0.00000000e+00j,  2.19010387e-01-1.83746186e-04j]])
dc = np.array([[ 0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j,
  -0.00000000e+00+0.00000000e+00j, -1.58222275e-03-1.37409398e-06j],
 [ 0.00000000e+00-0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  -1.81684114e+01+6.86804157e-01j, -0.00000000e+00+0.00000000e+00j],
 [ 0.00000000e+00-0.00000000e+00j,  1.81374193e+01-1.26368155e+00j,
   0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j],
 [ 1.58222227e-03+1.84005188e-06j,  0.00000000e+00-0.00000000e+00j,
   0.00000000e+00-0.00000000e+00j,  0.00000000e+00+0.00000000e+00j]])
 
Htot_zp = np.array([[ 8.31270428e-02-9.18776428e-05j,  0.00000000e+00+0.00000000e+00j,
   0.00000000e+00+0.00000000e+00j, -1.08420217e-19+5.29395592e-23j],
 [ 0.00000000e+00+0.00000000e+00j,  1.72739780e-01-2.28781248e-04j,
   1.38777878e-17-1.73472348e-18j,  0.00000000e+00+0.00000000e+00j],
 [ 0.00000000e+00+0.00000000e+00j,  0.00000000e+00-4.33680869e-18j,
   1.74416209e-01-1.38717190e-04j,  0.00000000e+00+0.00000000e+00j],
 [ 0.00000000e+00-8.60267837e-23j,  0.00000000e+00+0.00000000e+00j,
   0.00000000e+00+0.00000000e+00j,  2.64028946e-01-2.75620795e-04j]])
 
dc_zp = np.array([[ 0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j,
  -0.00000000e+00+0.00000000e+00j, -1.58222275e-03-1.37409398e-06j],
 [ 0.00000000e+00-0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  -1.81684114e+01+6.86804157e-01j, -0.00000000e+00+0.00000000e+00j],
 [ 0.00000000e+00-0.00000000e+00j,  1.81374193e+01-1.26368155e+00j,
   0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j],
 [ 1.58222227e-03+1.84005188e-06j,  0.00000000e+00-0.00000000e+00j,
   0.00000000e+00-0.00000000e+00j,  0.00000000e+00+0.00000000e+00j]])

    
''' DENSITY MATRIX PROPAGATION '''
#for i in range(0,Ntime):
#    Dp1 = RK4(Htot, D, dt, i*dt)
#    t[i] = dt*i
#    pd1[i] = Dp1[0,0]
#    pd2[i] = Dp1[1,1]
#    pd3[i] = Dp1[2,2]
#    pd4[i] = Dp1[3,3]
 #   p5[i] = Dp1[4][4]
 #   p6[i] = Dp1[5][5]
#    D = Dp1
    #DD = Transform(v, Dp1)
    #pd1[i] = DD[0,0]
    #pd2[i] = DD[1,1]
    #pd3[i] = DD[2,2]
    #pd4[i] = DD[2,2]
    #print(t[i],np.real(p1[i]),np.real(p2[i]),np.real(p3[i]),np.real(p4[i]),np.real(pd1[i]),np.real(pd2[i]),np.real(pd3[i]),np.real(pd4[i]))
#    pd5[i] = DD[4][4]
#    pd6[i] = DD[5][5]
    ##trp = np.trace(D)
    ##print(np.real(trp))


''' Wavefunction propagation '''
for i in range(0,Ntime):
    Psip1 = RK4_NH_SE(Htot, Psi, dt, V, dc, i*dt)
    t[i] = dt*i
    Psi = np.copy(Psip1)
    DD = Form_Rho(Psi)
    
    p1[i] = DD[0,0]
    p2[i] = DD[1,1]
    p3[i] = DD[2,2]
    p4[i] = DD[3,3]
    
    Psizpp1 = RK4_NH_SE(Htot_zp, Psi_zp, dt, V, dc_zp, i*dt)
    Psi_zp = np.copy(Psizpp1)
    DDzp = Form_Rho(Psi_zp)
    p1zp[i] = DDzp[0,0]
    p2zp[i] = DDzp[1,1]
    p3zp[i] = DDzp[2,2]
    p4zp[i] = DDzp[3,2]

plt.plot(t, np.real(p2), 'black', t, np.real(p2zp), 'r--') #, t, np.real(p3), 'blue') # t, np.real(p5), 'purple', t, np.real(p6), 'orange')
#plt.plot(t, np.real(pd1), 'black', t, np.real(pd2), 'r--', t, np.real(pd3), 'blue', t, np.real(pd4), 'g--') # t, np.real(pd5), 'purple', t, np.real(pd6), 'orange')
plt.ylim(0,1.2)
plt.show()


print(np.real(p2))
#print(Dp1[0,0])
#print(Dp1[1,1])
#print(Dp1[2,2])
#print(Dp1[3,3])
#np.savetxt("time.txt",t)
#np.savetxt("p1_g0.07_ghnp_0.0001_gdeph_0.005.txt",np.real(pd1))
#print("Try higher photon numnber states!!!")
'''
for i in range(0,len(p1)):
    if pd1[i]>0.5:
        print("half life is ",t[i])
        print("half life in fs is ",0.0248*t[i])
        print("decay constant in a.u. is ",0.5*np.log(2.)/t[i])
        print("decay constant in ev   is ",0.5*27*np.log(2.)/t[i])
        break
'''
