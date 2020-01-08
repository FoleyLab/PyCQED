import numpy as np
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import sys


'''
ri = -0.5
vi = 0.0001282*1.5
gam_diss_np = 0.00005
gam_deph_np = 0.0000

gam_diss_m = 0.00005
gam_deph = 0.000
'''

ri = float(sys.argv[1])
vi = float(sys.argv[2])
gam_diss_np = float(sys.argv[3])
gam_diss_m = float(sys.argv[4])
gam_deph_np = 0.
gam_deph = 0.


en_hnp = 0.075
en_mol = 0.075

def E_of_R(R):
    Ai = np.array([0.049244, 0.010657, 0.428129, 0.373005])
    Bi = np.array([0.18, 0.18, 0.18, 0.147])
    Ri = np.array([-0.75, 0.85, -1.15, 1.25])
    Di = np.array([0.073, 0.514])
    
    v = Ai + Bi*(R - Ri)**2
    
    Eg = 0.5*(v[0] + v[1]) - np.sqrt(Di[0]**2 + 0.25 * (v[0] - v[1])**2)
    Ee = 0.5*(v[2] + v[3]) - np.sqrt(Di[1]**2 + 0.25 * (v[2] - v[3])**2)
    return [Eg, Ee]
    
rlist = np.linspace(-2.0, 2.0, 50)

E_ground = []
E_excite = []
for r in rlist:
    PES = E_of_R(r)
    E_ground.append(PES[0])
    E_excite.append(PES[1])

### form spline for ground-state surface
Eg_spline = InterpolatedUnivariateSpline(rlist, E_ground, k=3)
Fg_spline = Eg_spline.derivative()

### form spline for excited-state surface
Ee_spline = InterpolatedUnivariateSpline(rlist, E_excite, k=3)
Fe_spline = Ee_spline.derivative()

#ture spline for the ground state
Cg_spline = Fg_spline.derivative()
### get the force constant by evaluating the curvature spline at R = -0.7 
k_g = Cg_spline(-0.7)
print(" Force constant of left well on |g> surface is",k_g)
### This is the reduced mass of this rotational mode
M = 1009883
### This is R_eq on the left well of |g>
#ri = -0.7
#vi = np.sqrt( np.sqrt(k_g/M) / (2*M))
#print("vi is ",vi)


gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_np
gamma[2] = gam_diss_m
gamma[3] = gam_diss_m+gam_diss_np

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

omc = 2.18/27.211 
### hbar g_c in atomic units
gc = 0.136/27.211

### form empty arrays for lower and upper polariton surface
PPES = np.zeros((len(rlist),4))

### Htot
He = np.zeros((4,4))
Hp = np.zeros((4,4))
Hep = np.zeros((4,4))
Htot = np.zeros((4,4))

Hp = H_p(Hp, omc)
Hep = H_ep(Hep, gc)

for i in range(0,len(rlist)):
    r = rlist[i]
    He = H_e(He, r)
    Htot = He + Hp + Hep
    #PES = E_of_R(r)
    #Htot[0][0] = PES[0] + 1.5 * omc
    #Htot[1][1] = PES[1] + 0.5 * omc
    tmpH = np.copy(Htot)
    vals, vecs = LA.eig(Htot)
    idx = vals.argsort()[::1]
    vals = vals[idx]
    for j in range(0,4):
        PPES[i,j] = vals[j]
    
dt = 0.75
N_time = 40000
time = np.zeros(N_time)
v_of_t = np.zeros(N_time)
r_of_t = np.zeros(N_time)
e_of_t = np.zeros(N_time)
p_of_t = np.zeros((N_time, 4))
dr = 0.1

### density matrix
D = np.zeros((4,4),dtype=complex)
D[1,1] = 1.+0j
### Hamiltonian matrix
for i in range(0,N_time):
    #### Update nuclear coordinate first
    time[i] = i*dt
    res = Erhenfest(ri, vi, M, D, Hp, Hep, He, gamma, gam_deph, dr, dt)
    #res = VelocityVerlet(Fg_spline, Fe_spline, np.real(D[0,0]), np.real(D[1,1]), M, ri, vi, dt)
    ri = res[0]
    vi = res[1]
    r_of_t[i] = ri
    v_of_t[i] = vi
    D = res[2]
    Htot = H_e(He, ri) + Hp + Hep
    e_of_t[i] = TrHD(Htot, D)
    for j in range(0,4):
        p_of_t[i,j] = np.real(D[j,j])

    
#plt.plot(rlist, Eg_spline(rlist), 'b--', label='|g> Spline')
#plt.plot(rlist, Ee_spline(rlist), 'g--', label='|e> Spline')
plt.plot(rlist, 27.211*PPES[:,0], 'b')
plt.plot(rlist, 27.211*PPES[:,1], 'g')
plt.plot(rlist, 27.211*PPES[:,2], 'y')
plt.plot(rlist, 27.211*PPES[:,3], 'r')
plt.plot(r_of_t, 27.211*e_of_t, 'red', label='Trajectory')
plt.legend()
plt.show()

