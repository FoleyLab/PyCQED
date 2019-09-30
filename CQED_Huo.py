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

### Somewhat similar to coupling strength and dissipation from HNPs
gam_diss_np = 0.0001
gam_deph_np = 0.005

gam_diss_m = 0.01
gam_deph_m = 0.005

gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_m
gamma[2] = gam_diss_np
gamma[3] = gam_diss_m+gam_diss_np

g_coup = 0.1

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

def E_of_R(R):
    Ai = np.array([0.049244, 0.010657, 0.428129, 0.373005])
    Bi = np.array([0.18, 0.18, 0.18, 0.147])
    Ri = np.array([-0.75, 0.85, -1.15, 1.25])
    Di = np.array([0.073, 0.514])
    
    v = Ai + Bi*(R - Ri)**2
    
    Eg = 0.5*(v[0] + v[1]) - np.sqrt(Di[0]**2 + 0.25 * (v[0] - v[1])**2)
    Ee = 0.5*(v[2] + v[3]) - np.sqrt(Di[1]**2 + 0.25 * (v[2] - v[3])**2)
    print("Eg is ",Eg)
    #return [Ai[0],Ai[1]]
    return [Eg, Ee]
    

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
        
        
rlist = np.linspace(-1.5, 1.5, 50)

E_ground = []
E_excite = []
for r in rlist:
    PES = E_of_R(r)
    E_ground.append(PES[0]*27.211)
    E_excite.append(PES[1]*27.211)
    
plt.plot(rlist, E_ground, 'red')
plt.plot(rlist, E_excite, 'blue')
plt.show()

        
        
'''
### Print basis states for verificatio
#print(Psi)
### Form photon Hamiltonian in basis of Psi
Hphot = H_Photon(Psi, 0.075)

### Form molecular Hamiltonian in basis of Psi
Hmol = H_Molecule(Psi, 0.075)

Nvals = 100
g_coup_min = 0.00
d_g = (0.04-g_coup_min)/(Nvals-1)

Hi = H_Interaction(Psi, 0.)
Htot = Hphot + Hmol + Hi
vals, vecs = LA.eig(Htot)
idx = vals.argsort()[::1]
vals = vals[idx]
v = vecs[:,idx]
lone_one = vals[0]
lone_two = vals[1]
lone_three = vals[2]
lone_four = vals[3]

c_one = np.zeros(Nvals)
c_two = np.zeros(Nvals)
c_three = np.zeros(Nvals)
c_four = np.zeros(Nvals)
u_one = np.zeros(Nvals)
u_two = np.zeros(Nvals)
u_three = np.zeros(Nvals)
u_four = np.zeros(Nvals)
ga = np.zeros(Nvals)

for i in range(0,Nvals):
  ### Form interaction Hamiltoninan in basis of Psi
  Hi = H_Interaction(Psi, d_g*i+g_coup_min)
  ga[i] = d_g*i+g_coup_min
  #print(Hi)
  ### Form total Hamiltonian
  Htot = Hphot + Hmol + Hi

  ## Diagonalize Total Hamiltonian, store vectors in v array
  vals, vecs = LA.eig(Htot)
  idx = vals.argsort()[::1]
  vals = vals[idx]
  v = vecs[:,idx]

  c_one[i] = vals[0]
  c_two[i] = vals[1]
  c_three[i]= vals[2]
  c_four[i] = vals[3]
  u_one[i] = lone_one
  u_two[i] = lone_two
  u_three[i] = lone_three
  u_four[i] = lone_four
  
  #Hdiag = Transform(v, Htot)
  #print(Hdiag)


plt.plot(ga, c_one, 'black', ga, c_two, 'red', ga, c_three, 'blue', ga, c_four, 'green')
plt.show()
'''

'''
Ntime = 4000
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
Psi[1] = np.sqrt(2/6)
Psi[2] = np.sqrt(3/6)
Psi[3] = np.sqrt(1/6)
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
    Dp1 = RK4(Htot, D, 0.05, i*0.05)
    t[i] = 0.01*i
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
#    pd5[i] = DD[4][4]
#    pd6[i] = DD[5][5]
    ##trp = np.trace(D)
    ##print(np.real(trp))


#plt.plot(t, np.real(p1), 'pink', t, np.real(p2), 'r--', t, np.real(p3), 'b--', t, np.real(p4), 'purple') # t, np.real(p5), 'purple', t, np.real(p6), 'orange')
plt.plot(t, np.real(pd1), 'black', t, np.real(pd2), 'r--', t, np.real(pd3), 'b--', t, np.real(pd4), 'g--') # t, np.real(pd5), 'purple', t, np.real(pd6), 'orange')
plt.show()
'''
