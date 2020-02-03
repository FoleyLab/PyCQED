#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:26:29 2020

@author: jay
"""

from polaritonic import polaritonic
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib.animation as animation
import time
import sys

### initial position
ri_init = float(sys.argv[1])
#ri_init = -0.66156
### initial velocity
vi_init = float(sys.argv[2])
#vi_init = 3.3375e-5
### photonic mode dissipation rate in meV, gamma
gamp = float(sys.argv[3]) 
#gamp = 0.1
### convert to a.u.
gam_diss_np = gamp * 1e-3 / 27.211

### photonic mode energy in eV
omp = float(sys.argv[4])
#omp = 2.45
### convert to a.u.
omc = omp/27.211
### coupling strength in eV
gp = float(sys.argv[5])
#gp = 0.02
gc = gp/27.211

au_to_ps = 2.4188e-17 * 1e12

### get prefix for data file names
prefix = sys.argv[6]
#prefix = "test"
### filename to write nuclear trajectory to
nuc_traj_fn = "Data/" + prefix + '_nuc_traj.txt'
### filename to wrote PES to
pes_fn = "Data/" + prefix + '_pes.txt'
### filename to write electronic/polaritonic dynamics to
ed_fn = "Data/" + prefix + '_electronic.txt'
### filename to write photonic contributions of each state to
pc_fn = "Data/" + prefix + '_photon_contribution.txt'


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

polt.Transform_L_to_P()

rlist = np.linspace(-1.5, 1.5, 500)
PES = np.zeros((len(rlist),polt.N_basis_states))

### Get PES of polaritonic system and write to file pes_fn
pes_file = open(pes_fn, "w")
pc_file = open(pc_fn, "w")
for r in range(0,len(rlist)):
    wr_str = "\n"
    pc_str = "\n"
    polt.R = rlist[r]
    wr_str = wr_str + str(polt.R) + " "
    pc_str = pc_str + str(polt.R) + " "
    polt.H_e()
    polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
    polt.Transform_L_to_P()
    v = polt.transformation_vecs_L_to_P
    
    for i in range(0,polt.N_basis_states):
        PES[r,i] = polt.polariton_energies[i]
        v_i = v[:,i]
        cv_i = np.conj(v_i)
        
        wr_str = wr_str + str(polt.polariton_energies[i]) + " "
        
        ### loop over all photon indices in basis states
        pc = 0
        for j in range(0,polt.N_basis_states):
            if polt.gamma_diss[j]>0:
                pc = pc + cv_i[j] * v_i[j]
        pc_str = pc_str + str(pc) + " "
            
            
    pes_file.write(wr_str)
    pc_file.write(pc_str)

### Close PES file
pes_file.close()
pc_file.close()


### Go back to r_init
polt.R = ri_init

polt.H_e()
polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
polt.Energy = polt.TrHD(polt.H_total, polt.D_local)
polt.Transform_L_to_P()

### open files for writing data about electronic and nuclear dynamics
electronic_file = open(ed_fn, "w")
nuclear_file = open(nuc_traj_fn, "w")
### Write initial electronic and nuclear configurations
e_str = "\n"
n_str = "\n"
### both strings need the time
e_str = e_str + str(0*polt.dt) + " "
n_str = n_str + str(0*polt.dt) + " "
n_str = n_str + str(polt.R) + " " + str(polt.Energy)
for j in range(0,polt.N_basis_states):
    e_str = e_str + str(np.real(polt.D_local[j,j])) + " "
for j in range(0,polt.N_basis_states):
    e_str = e_str + str(np.real(polt.D_polariton[j,j])) + " "

electronic_file.write(e_str)
nuclear_file.write(n_str)

### start timing
start = time.time()
### start dynamics
for i in range(1,N_time):
    #### Update nuclear coordinate first
    polt.FSSH_Update()
    
    ### store dynamics data every 200 updates
    if i%500==0:
        e_str = "\n"
        n_str = "\n"
        ### both strings need the time
        e_str = e_str + str(i*polt.dt) + " "
        n_str = n_str + str(i*polt.dt) + " " + str(polt.R) + " " + str(polt.Energy) + " "
        nuclear_file.write(n_str)
        
        ### nuc string needs R and E
        
        for j in range(0,polt.N_basis_states):
            e_str = e_str + str(np.real(polt.D_local[j,j])) + " "
        for j in range(0,polt.N_basis_states):
            e_str = e_str + str(np.real(polt.D_polariton[j,j])) + " "
        
        electronic_file.write(e_str)
        
    if i>N_thresh:
        r_array.append(polt.R)
        
end = time.time()

time_taken = end-start

electronic_file.close()
nuclear_file.close()
        
avg_r = sum(r_array) / len(r_array)

if avg_r>0:
    iso_res = 1
else:
    iso_res = 0

print(time_taken/60., ri_init, vi_init, gamp, omp, gp, avg_r, iso_res)
    


