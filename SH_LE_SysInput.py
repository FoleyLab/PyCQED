### Import all libraries and define various parameters here!
import numpy as np
import dynamicshelper as dh
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.animation as animation
import sys


''' Some key parameters for the simulation! '''
### dissipation parameters for electronic and photonic system


### initial position
ri_init = float(sys.argv[1])
### initial velocity
vi_init = float(sys.argv[2])
### photonic mode dissipation rate in meV, gamma
gamp = float(sys.argv[3]) 
### convert to a.u.
gam_diss_np = gamp * 1e-3 / 27.211

### photonic mode energy in eV
omp = float(sys.argv[4])
### convert to a.u.
omc = omp/27.211
### coupling strength in eV
gp = float(sys.argv[5])
### convert to a.u.
gc = gp/27.211
gam_deph_np = 0.0000

gam_diss_m = 0.00000
gam_deph = 0.0000

### Set T to 300 K in atomic units
T = 0.00095 # 300 K is 0.00095 in atomic units
### Set friction constant so that vibrational relaxation happens on a ~3 ps timescale
### Note this value of g_n = 0.000011 was arrived at with some experimentation; 
### we can probably determine the correct value analytically.
g_n = 0.000011


au_to_ps = 2.4188e-17 * 1e12
### This is the reduced mass of this rotational mode for the dynamics... in atomic units
M = 1009883

### Number of updates for dynamics
N_time = 4000000 
### N_thresh controls when you start taking the average position
N_thresh = N_time / 4
### position displacement increment for dynamics (a.u.)
dr = 0.001
### time displacement increment for dynamics (a.u.)
### This is about 0.003 f.s., similar to timestep used in Ehrenfest dynamics for 
### ACS Nano 2016, 10, 5452-5458
dt = 0.12 

### initial state for light/matter system (will index the density matrix in local basis)
pn = 2

### array of dissipation parameters to be passed to RK4 function
gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_np
gamma[2] = gam_diss_m
gamma[3] = gam_diss_m+gam_diss_np



### various arrays for dynamics

time = np.zeros(N_time)
r_of_t = np.zeros(N_time)

v_of_t = np.zeros(N_time)
e_of_t = np.zeros(N_time)
p_of_t = np.zeros((N_time,4))

''' The following parameters and arrays pertain 
    to the polaritonic suraces!  '''
    
### array of values along the reactive coordinate R that the 
### surfaces will be explicitly evaluated at... this is in atomic units
rlist = np.linspace(-2.0, 2.0, 50)


### Htot
He = np.zeros((4,4))
Hp = np.zeros((4,4))
Hep = np.zeros((4,4))
Htot = np.zeros((4,4))
PPES = np.zeros((len(rlist),4))



''' Now let's evaluate the polaritonic PES and store the values to the PPES array! '''
Hp = dh.H_p(Hp, omc)
Hep = dh.H_ep(Hep, gc)

#### Get H_e(r) and diagonlize to get the polaritonic potential energy surfaces
for i in range(0,len(rlist)):
    r = rlist[i]
    He = dh.H_e(He, r)
    Htot = He + Hp + Hep
    tmpH = np.copy(Htot)
    vals, vecs = LA.eig(Htot)
    idx = vals.argsort()[::1]
    vals = vals[idx]
    for j in range(0,4):
        PPES[i,j] = vals[j]

ri = ri_init
vi = vi_init
### density matrix in local
Dl = np.zeros((4,4),dtype=complex)
Dl[pn,pn] = 1.+0j
gs_r = []
    
for i in range(0,N_time):
    #### Update nuclear coordinate first
    time[i] = i*dt
    #res = dh.Erhenfest_v2(ri, vi, M, Dl, Hp, Hep, He, gamma, gam_deph, dr, dt)
    res = dh.FSSH_Update(ri, vi, M, g_n, T, Dl, Hp, Hep, He, gamma, gam_deph, dr, dt, pn)
    ri = res[0]
    vi = res[1]
    r_of_t[i] = ri
    v_of_t[i] = vi
    e_of_t[i] = res[2] #i_spline(ri)
    Dl = res[3]
    pn = res[4]
    p_of_t[i,0] = np.real(Dl[0,0])
    p_of_t[i,1] = np.real(Dl[1,1])
    p_of_t[i,2] = np.real(Dl[2,2])
    p_of_t[i,3] = np.real(Dl[3,3])
        
    #### Don't start taking average immediately, allow some time to elapse first.
    if (i>N_thresh):
        gs_r.append(ri)

### if we have accumulated a trajectory on the gs surface        
if (len(gs_r)>1):
    avg_r = sum(gs_r) / len(gs_r)
    #print("avg r is ",avg_r)
    if (avg_r>0.0):
        iso_res = 1
    else:
        iso_res = 0

print(ri_init, vi_init, gamp, omp, gp, avg_r, iso_res)

