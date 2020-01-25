#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:26:29 2020

@author: jay
"""

from polaritonic import polaritonic
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time
import sys

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
gc = gp/27.211

au_to_ps = 2.4188e-17 * 1e12

### Number of updates!
N_time = 4000000

### N_thresh controls when you start taking the average position
N_thresh = int( N_time / 4)

r_array = []

options = {
        'Number_of_Photons': 1,
        'Photon_Energys': [omc],
        'Coupling_Strengths': [gc], 
        'Photon_Lifetimes': [gam_diss_np],
        'Initial_Position': ri_init,
        'Initial_Velocity': vi_init,
        'Mass': 1009883,
        ### temperature in a.u.
        'Temperature': 0.00095,
        ### friction in a.u.
        'Friction': 0.000011,
        ### specify initial state as a human would, not a computer...
        ### i.e. 1 is the ground state... it will be shifted down by -1 so
        ### that it makes sense to the python index convention
        'Initial_Local_State': 3
        
        }

### instantiate
polt = polaritonic(options)

polt.Transform_L_to_P()
polt.D_polariton = np.zeros((polt.N_basis_states,polt.N_basis_states),dtype=complex)
polt.D_polariton[polt.initial_state,polt.initial_state] = 1+0j
polt.D_local = np.outer(polt.transformation_vecs_L_to_P[:,polt.initial_state], np.conj(polt.transformation_vecs_L_to_P[:,polt.initial_state])) 



rlist = np.linspace(-1.5, 1.5, 200)
PES = np.zeros((len(rlist),polt.N_basis_states))
Local_En = np.zeros(2000)
Pol_En = np.zeros(2000)


for r in range(0,len(rlist)):
    polt.R = rlist[r]
    polt.H_e()
    polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
    polt.Transform_L_to_P()
    polt.Transform_P_to_L()
    Local_En[r] = polt.TrHD(polt.H_total, polt.D_local)
    #Pol_En[r] = polt.TrHD(polt.H_polariton, polt.D_polariton)
    
    for i in range(0,polt.N_basis_states):
        PES[r,i] = polt.polariton_energies[i]

'''
plt.plot(rlist, Local_En, 'green')
plt.plot(rlist, PES[:,polt.initial_state], 'b--')
plt.show()
'''
polt.R = ri_init





start = time.time()
### We won't track dynamic quantities for the trial so just comment this out for now!
#sim_time[0] = 0
#r_of_t[0] = polt.R
polt.H_e()
polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
polt.Energy = polt.TrHD(polt.H_total, polt.D_local)
#e_of_t[0] = polt.Energy

anidx = 1

for i in range(0,N_time):
    #### Update nuclear coordinate first
    polt.FSSH_Update()
    
    ### store dynamics data every 200 updates
    #if i%200==0:
    #    sim_time[anidx] = i*polt.dt
    #    r_of_t[anidx] = polt.R
    #    e_of_t[anidx] = polt.Energy
    #    anidx = anidx + 1
        
    if i>N_thresh:
        r_array.append(polt.R)
        
end = time.time()

time_taken = end-start
        
avg_r = sum(r_array) / len(r_array)

if avg_r>0:
    iso_res = 1
else:
    iso_res = 0

print(time_taken/60., ri_init, vi_init, gamp, omp, gp, avg_r, iso_res)
    


