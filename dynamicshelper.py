#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:58:08 2019

@author: foleyj10
"""
import numpy as np
from numpy import linalg as LA
import math
from numpy.polynomial.hermite import *

''' For the Hamiltonians '''
### Function to return the ground and excited-state electronic energy as a function of
### the nuclear coordinate $R$
def E_of_R(R):
    Ai = np.array([0.049244, 0.010657, 0.428129, 0.373005])
    Bi = np.array([0.18, 0.18, 0.18, 0.147])
    Ri = np.array([-0.75, 0.85, -1.15, 1.25])
    Di = np.array([0.073, 0.514])
    
    v = Ai + Bi*(R - Ri)**2
    
    Eg = 0.5*(v[0] + v[1]) - np.sqrt(Di[0]**2 + 0.25 * (v[0] - v[1])**2)
    Ee = 0.5*(v[2] + v[3]) - np.sqrt(Di[1]**2 + 0.25 * (v[2] - v[3])**2)
    return [Eg, Ee]

### Forms the electronic Hamiltonian matrix at a given value of the nuclear coordinate
### FOR 4x4 case relevant for 1-mode polariton case
def H_e(Hmat, R):
    PES = E_of_R(R)
    Hmat[0,0] = PES[0]
    Hmat[1,1] = PES[0]
    Hmat[2,2] = PES[1]
    Hmat[3,3] = PES[1]
    return Hmat

### Form bare photon hamiltonian with frequency omega c for 4x4 case 
### relevant for 1-mode polariton case
def H_p(Hmat, omega):
    Hmat[0,0] = 0.5 * omega
    Hmat[1,1] = 1.5 * omega
    Hmat[2,2] = 0.5 * omega
    Hmat[3,3] = 1.5 * omega
    return Hmat

### form coupling hamiltonian with coupling strength gamma_c for 4x4 
### case relevant for 1-mode polariton case
def H_ep(Hmat, g):
    Hmat[1,2] = g
    Hmat[2,1] = g
    return Hmat
''' The following functions are helpers for the quantum and classical dynamics '''

''' Quantum dynamics first '''
### should create a function that takes a wavefunction in vector
### format and computes a density matrix
def Form_Rho(Psi):

    D = np.outer(Psi,np.conj(Psi))
    return D

def RK4(H, D, h, gamma, gam_deph):
    k1 = np.zeros_like(D)
    k2 = np.zeros_like(D)
    k3 = np.zeros_like(D)
    k4 = np.zeros_like(D)
    D1 = np.zeros_like(D)
    D2 = np.zeros_like(D)
    D3 = np.zeros_like(D)
    D4 = np.zeros_like(D)
    
    ### Get k1
    D1 = np.copy(D)    
    k1 = h*DDot(H,D1) + h*L_Diss(D1, gamma) + h*L_Deph(D1, gam_deph)
    
    ### Update H and D and get k2
    D2 = np.copy(D+k1/2.)
    k2 = h*DDot(H, D2) + h*L_Diss(D2, gamma) + h*L_Deph(D2, gam_deph)
    
    ### UPdate H and D and get k3
    D3 = np.copy(D+k2/2)
    k3 = h*DDot(H, D3) + h*L_Diss(D3, gamma) + h*L_Deph(D3, gam_deph)
    
    ### Update H and D and get K4
    D4 = np.copy(D+k3)
    k4 = h*DDot(H, D4) + h*L_Diss(D4, gamma) + h*L_Deph(D4, gam_deph)
    
    Df = D + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4)
    return Df

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

def DDot(H, D):
    ci = 0.+1j
    return -ci*(np.dot(H,D) - np.dot(D, H))

def TrHD(H, D):
    N = len(H)
    HD = np.dot(H,D)
    som = 0
    for i in range(0,N):
        som = som + HD[i,i]
    return np.real(som)

### Transform density matrix from local to polariton basis
### at a given R... return (diagonal) polariton Hamiltonian
### and transformation vecs, also
def Transform_L_to_P(r, Dl, Hp, Hep):
    He = np.zeros((4,4))
    He = H_e(He, r)
    Htot = He + Hp + Hep
    ### get eigenvalues/vectors of total Hamiltonian at this R
    vals, vecs = LA.eig(Htot)
    ### sort the eigenvectors
    idx = vals.argsort()[::1]
    vals = vals[idx]
    v = vecs[:,idx]
    ### transform Htot with v^-1
    vt0 = np.dot(LA.inv(v),Htot)
    ### finish transformation to polariton basis, Hpl
    Hpl = np.dot(vt0,v)
    ### now do transformation for density matrix from local to polariton basis
    dt0 = np.dot(LA.inv(v), Dl)
    Dpl = np.dot(dt0,v)
    ### return Hpl and Dpl
    return [Hpl, Dpl, v]

### Transform density matrix from polariton to local basis
### at a given R... return (off-diagonal) local Hamiltonian
### and transformation vecs, also
def Transform_P_to_L(r, Dp, Hp, Hep):
    He = np.zeros((4,4))
    He = H_e(He, r)
    Htot = He + Hp + Hep
    ### get eigenvalues/vectors of total Hamiltonian at this R
    vals, vecs = LA.eig(Htot)
    ### sort the eigenvectors
    idx = vals.argsort()[::1]
    vals = vals[idx]
    v = vecs[:,idx]
    ### now do transformation for density matrix from local to polariton basis
    dt0 = np.dot(v, Dp)
    Dl = np.dot(dt0,LA.inv(v))
    ### return Hpl and Dpl
    return [Htot, Dl, v]


def Erhenfest(r_curr, v_curr, mass, D, Hp, Hep, Hel, gamma, gam_deph, dr, dt):

    ''' Electronic part 1 '''
    ### Get forward-displaced electronic Hamiltonian
    Hel = H_e(Hel, r_curr+dr)
    Hf = Hp + Hep + Hel
    ### Get forward-dispaced density matrix
    D = RK4(Hf, D, dt, gamma, gam_deph)
    ### Get forward-displaced energy
    Ef = TrHD(Hf, D)
    ### Get back-displaced electronic Hamiltonian
    Hel = H_e(Hel, r_curr-dr)
    Hb = Hp + Hep + Hel
    D = RK4(Hb, D, dt, gamma, gam_deph)
    ### Get back-displaced energy
    Eb = TrHD(Hb, D)
    
    ''' Nuclear part 1'''
    ### Get force from finite-difference gradient
    F = (Eb - Ef)/(2*dr)
    ### Get acceleration from force
    a_curr = F / mass
    ### now get r in the future... r_fut
    r_fut = r_curr + v_curr*dt + 1/2 * a_curr*dt**2
    
    ''' Electronic part 2 '''
    ### now update electronic Hamiltonian
    Hel = H_e(Hel, r_fut+dr)
    Hf = Hp + Hep + Hel
    ### update electronic density matrix 
    D = RK4(Hf, D, dt, gamma, gam_deph)
    ### get forward-displaced energy
    Ef = TrHD(Hf,D)
    ### Get back-displaced electronic Hamiltonian
    Hel = H_e(Hel, r_curr-dr)
    Hb = Hp + Hep + Hel
    D = RK4(Hb, D, dt, gamma, gam_deph)
    #p_g = D[0,0]
    #p_e = D[1,1]
    #c_01 = D[0,1]
    #c_10 = D[1,0]
    ### Get back-displaced energy
    Eb = TrHD(Hb, D)
    
    ''' Nuclear part 2'''
    ### Get force from finite-difference gradient
    F = (Eb - Ef)/(2*dr)
    ### Get acceleration from force
    a_fut = F / mass
    v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
    ### return a list with new position and velocity
    return [r_fut, v_fut, D]

def VelocityVerlet(spline,  mass, r_curr, v_curr, dt):
    ### compute acceleration ... first we need force
    F_curr = -1 * spline(r_curr)
    ### now get acceleration from a = F/m
    a_curr = F_curr / mass
    ### now get r in the future... r_fut
    r_fut = r_curr + v_curr*dt + 1/2 * a_curr*dt**2
    ### now get the future force: -d/dr E(r_fut)
    F_fut = -1 * spline(r_fut)
    ### now that I have new Force, compute new acceleration
    a_fut = F_fut / mass
    ### now that I have acceleration in the future, let me calculate velocity 
    ### in the future
    v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
    ### return a list with new position and velocity
    return [r_fut, v_fut]

def dfdx(ft, xt):
    dx = xt[1]-xt[0]
    ftp = np.zeros_like(ft)
    for i in range(0,len(ft)):
        if (i<(len(ft)-1)):
            rise = ft[i+1]-ft[i]
            ftp[i] = rise/dx
        else:
            rise = ft[i]-ft[i-1]
            ftp[i] = rise/dx
    
    return ftp

### Kinetic energy operator on wavefunction
def TPhi(ft, xt, m):
    ftp = dfdx(ft, xt)
    ftpp = dfdx(ftp, xt)
    return -1/(2*m)*ftpp

### Get action of Hamiltonian on Phi and multiply by negative i... this
### gives time-derivative of Phi
def Phi_Dot(ft, xt, m, vx):
    ci = 0+1j
    return -1*ci*(TPhi(ft, xt, m) + vx*ft)

### Kinetic energy squared operator
def T2Phi(ft, xt, m):
    ftp = dfdx(ft, xt)
    ftpp = dfdx(ftp, xt)
    ftppp = dfdx(ftpp, xt)
    ftpppp = dfdx(ftppp, xt)
    return 1/(4*m*m)*ftpppp

### returns the kinetic energy functional of a trial
### wavefunction (called ft within the function)
def T_Functional(ft, xt, m):
    tphi = TPhi(ft, xt, m)
    dx = xt[1] - xt[0]
    num = 0
    denom = 0
    for i in range(0, len(ft)):
        num = num + ft[i]*tphi[i]*dx
        denom = denom + ft[i]*ft[i]*dx

    return num/denom

def T2_Functional(ft, xt, m):
    t2phi = T2Phi(ft, xt, m)
    dx = xt[1] - xt[0]
    num = 0
    denom = 0
    for i in range(0, len(ft)):
        num = num + ft[i]*t2phi[i]*dx
        denom = denom + ft[i]*ft[i]*dx

    return num/denom


hbar = 1
pi = np.pi
r0 = -1

def HO_En(K, m, n):
    return np.sqrt(K/m) * (n + 1/2)


def HO_Func(K, m,  n, r, r0):
    w = np.sqrt(K/m)
    psi = []
    herm_coeff = []
    
    for i in range(n):
        herm_coeff.append(0)
        
    herm_coeff.append(1)
    
    for x in r:
        psi.append(math.exp(-m*w*(x-r0)**2/(2*hbar)) * hermval((m*w/hbar)**0.5 * (x-r0), herm_coeff))
        
    # normalization factor for the wavefunction:
    psi = np.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*hbar))**0.25)
    
    return psi

r2 = np.linspace(-1,0,500)
vx_g0 = 1/2 * k_g0 * (r2-rmin_g0)**2
psi_g0 = HO_Func(k_g0, M, 0, r2, rmin_g0)

vx_phi2 = 1/2 * k_phi2 * (r2-rmin_phi2)**2
psi_phi2 = HO_Func(k_phi2, M, 0, r2, rmin_phi2)




def Fourier(x, fx, n, k, m, r0):
    tfn = HO_Func(k, m, n, x, r0)
    som = 0
    dx = x[1]-x[0]
    for i in range(0,len(x)):
        som = som + fx[i] * tfn[i] * dx
    return som


    