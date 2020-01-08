import numpy as np
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt

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
PsiBas = np.zeros((2,4))

### Define basis states
idx = 0
for i in range(0,2):
    for j in range(0,2):
        PsiBas[0][idx] = i
        PsiBas[1][idx] = j
        idx = idx+1


en_hnp = 0.075
en_mol = 0.075

dt = 0.05
gamma = np.zeros(4)
gam_deph = 0.005        
### Print basis states for verificatio
### Form photon Hamiltonian in basis of Psi
Hphot = H_Photon(PsiBas, en_hnp)

### Form molecular Hamiltonian in basis of Psi
Hmol = H_Molecule(PsiBas, en_mol)
Ntime = 50000
p1 = np.zeros(Ntime,dtype=complex)
p2 = np.zeros(Ntime,dtype=complex)
p3 = np.zeros(Ntime,dtype=complex)
p4 = np.zeros(Ntime,dtype=complex)
pd1 = np.zeros(Ntime,dtype=complex)
pd2 = np.zeros(Ntime,dtype=complex)
pd3 = np.zeros(Ntime,dtype=complex)
pd4 = np.zeros(Ntime,dtype=complex)
t = np.zeros(Ntime)

gam_deph_np = 0.03
gam_deph_m = 0.001
d_diss = (0.03-0.01)/401
d_g    = (0.07 - 0.001)/401
gam_diss_m = 0.01

for k in range(0,400):
    print(" ")
    gam_diss_np =  0.01+k*d_diss
    gamma[0] = 0.
    gamma[1] = gam_diss_m
    gamma[2] = gam_diss_np
    gamma[3] = gam_diss_m+gam_diss_np

    for j in range(0,400):

        g_coup = 0.001 + d_g * j

        Hi = H_Interaction(PsiBas, g_coup)
        Htot = Hphot + Hmol + Hi
        vals, vecs = LA.eig(Htot)
        idx = vals.argsort()[::1]
        vals = vals[idx]
        v = vecs[:,idx]

  
        Hdiag = Transform(v, Htot)


        Psi = np.zeros(4, dtype=complex)

        Psi[1] = np.sqrt(1.)
        D = Form_Rho(Psi)

        for i in range(0,Ntime):
            Dp1 = RK4(Htot, D, dt, i*dt)
            t[i] = dt*i
            p1[i] = Dp1[0][0]
            D = Dp1
            DD = Transform(v, Dp1)
            pd1[i] = DD[0][0]
            #print(t[i],pd1[i])
            if np.real(pd1[i])>0.5:
                hl = t[i]
                break
        
        print(gam_diss_np, g_coup, hl)
''' 
np.savetxt("time.txt",t)
np.savetxt("p1_g0.07_ghnp_0.0001_gdeph_0.005.txt",np.real(pd1))
print("Try higher photon numnber states!!!")
for i in range(0,len(p1)):
    if pd1[i]>0.5:
        print("half life is ",t[i])
        break
'''    
