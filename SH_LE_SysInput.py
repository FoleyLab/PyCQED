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

ri_init = float(sys.argv[1])
vi_init = float(sys.argv[2])

gam_diss_np = float(sys.argv[3])
gam_deph_np = 0.0000

gam_diss_m = 0.00000
gam_deph = 0.0000



au_to_ps = 2.4188e-17 * 1e12
### This is the reduced mass of this rotational mode for the dynamics... in atomic units
M = 1009883

### hbar omega_c in atomic units
omc = 2.45/27.211 
### hbar g_c in atomic units
gc = 0.02/27.211


### Number of updates for dynamics
N_time = 4000000

### position displacement increment for dynamics (a.u.)
dr = 0.01 
### time displacement increment for dynamics (a.u.)
dt = 0.1

### initial polariton state
pn = 2

### array of dissipation parameters to be passed to RK4 function
gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_np
gamma[2] = gam_diss_m
gamma[3] = gam_diss_m+gam_diss_np



### various arrays for dynamics

time = np.zeros(N_time)
r_of_t = np.zeros((N_time,10))
hf_error_of_t = np.zeros((N_time, 10))
tot_error_of_t = np.zeros((N_time, 10))

v_of_t = np.zeros((N_time,10))
e_of_t = np.zeros((N_time,10))
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
        
### form spline for ground-state surface
i_spline = InterpolatedUnivariateSpline(rlist, PPES[:,1], k=3)
Fi_spline = i_spline.derivative()
  
g_spline = InterpolatedUnivariateSpline(rlist, PPES[:,0], k=3)
Fg_spline = g_spline.derivative()  


#[Ht, Dl, vec] = dh.Transform_P_to_L(ri[0], Dpl, Hp, Hep)
#print(Dl)
#HD = np.dot(Ht,Dl)
#print(Ht)


flag = 1
T = 0.00095 # boiling point of CO in atomic units
g_n = 0.000011
#ri_val = [-0.6615679318398704, -0.698020918719673, -0.7325842045116059, -0.7304937149739744, -0.6940536701380835, -0.6836965102136909, -0.6998506709530399, -0.7449253281970559, -0.6904318143316741, -0.6939445601745601]
#vi_val = [3.33752906715916e-05, -1.7604569932905628e-05, 7.215162825258178e-07, -3.308478778918541e-05, 3.0711164130420224e-06, -2.104531948592309e-05, -1.4907596794662185e-06, 1.2388055205752432e-05, 5.501228084142224e-05, -3.3319508968582436e-06]


pn = 2
ri = ri_init
vi = vi_init
### density matrix in polariton basis!
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
    r_of_t[i,j] = ri
    v_of_t[i,j] = vi
    e_of_t[i,j] = res[2] #i_spline(ri)
    Dl = res[3]
    pn = res[4]
    p_of_t[i,0] = np.real(Dl[0,0])
    p_of_t[i,1] = np.real(Dl[1,1])
    p_of_t[i,2] = np.real(Dl[2,2])
    p_of_t[i,3] = np.real(Dl[3,3])
        
    if (pn==0):
        gs_r.append(ri)


### if we have accumulated a trajectory on the gs surface        
if (len(gs_r)>1):
    avg_r = sum(gs_r) / len(gs_r)
    print("avg r is ",avg_r)
    if (avg_r>0.0):
        iso_res = 1
    else:
        iso_res = 0
### if we haven't decayed to the gs, the simulation is indeterminant
else:
    iso_res = 0.5

print(ri_init, vi_init, gam_diss_np, iso_res)

