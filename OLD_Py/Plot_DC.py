#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:31:21 2020

@author: foleyj10
"""

### Import all libraries and define various parameters here!
import numpy as np
from polaritonic import polaritonic
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

'''
ri_init = -0.66156
vi_init = 3.3375e-5
### lifetime
gamp = 0. 
gam_diss_np = gamp * 1e-3 / 27.211

### photonic mode energy in eV
omp = 2.45
### convert to a.u.
omc = omp/27.211
### coupling strength in eV
gp = 0.02
gc = gp/27.211

au_to_ps = 2.4188e-17 * 1e12

### get prefix for data file names
prefix = "gam_0.0"
### filename to write nuclear trajectory to
nuc_traj_fn = "Data/" + prefix + '_nuc_traj.txt'
### filename to wrote PES to
pes_fn = "Data/" + prefix + '_pes.txt'
### filename to write electronic/polaritonic dynamics to
ed_fn = "Data/" + prefix + '_electronic.txt'
### filename to write photonic contributions of each state to
pc_fn = "Data/" + prefix + '_photon_contribution.txt'

### hellman-Feynman file 
hf_fn = "Data/" + prefix + "_hf.txt"

options = {
        'Number_of_Photons': 1,
        'Complex_Frequency': True,
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
### write forces and derivative coupling
polt.Write_Forces(hf_fn)
'''

hf_0 = "Data/gam_0.0_hf.txt"
hf_1 = "Data/gam_1.0_hf.txt"
hf_10 = "Data/gam_50.0_hf.txt"
hf_100 = "Data/gam_100.0_hf.txt"


### read text file
dc_0 = np.loadtxt(hf_0,dtype=complex)
dc_1 = np.loadtxt(hf_1,dtype=complex)
dc_10 = np.loadtxt(hf_10,dtype=complex)
dc_100 = np.loadtxt(hf_100,dtype=complex)


#plt.plot(dc_0[:,0], dc_0[:,3], 'b*', label='$\gamma$=0')
plt.plot(dc_0[:,0], dc_1[:,3], 'red', label='$\gamma$=1 meV')
plt.plot(dc_0[:,0], dc_10[:,3], 'green', label='$\gamma$=50 meV')
plt.plot(dc_0[:,0], dc_100[:,3], 'blue', label='$\gamma$=100 meV')
plt.legend()
plt.xlim(-.65,-.55)
#plt.ylim(-20.,20)
#plt.xlabel("R (a.u.)")
#plt.ylabel("Energy (eV)")
#plt.savefig("Fig_Surface.eps")
plt.show()