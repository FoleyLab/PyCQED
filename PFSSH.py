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

### dummi initial position and velocity values
ri_init = -0.66156
vi_init = 3.3375e-5

### will we scale the velocity?
sv_condition = str(sys.argv[1])
if sv_condition=='Scale_False':
    sv_bool = False
else:
    sv_bool = True
dc_condition = str(sys.argv[2])
if dc_condition=='Real_DC':
    dc_bool = False
else:
    dc_bool = True
    
### Number of repeates
N_repeats = int(sys.argv[3])
### photonic mode dissipation rate in meV, gamma
gamp = float(sys.argv[4]) 
#gamp = 100.0
### convert to a.u.
gam_diss_np = gamp * 1e-3 / 27.211

### photonic mode energy in eV
omp = float(sys.argv[5])
#omp = 2.45
### convert to a.u.
omc = omp/27.211
### coupling strength in eV
gp = float(sys.argv[6])
#gp = 0.02
gc = gp/27.211

au_to_ps = 2.4188e-17 * 1e12

### get prefix for data file names
prefix = sys.argv[7]
#prefix = "g_100.0_test"
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
### derivative coupling data
dc_fn = "Data/" + prefix + "_dc.txt"
### inner-product data
ip_fn = "Data/" + prefix + "_ip.txt"

### Number of updates!
N_time = 4000000

### N_thresh controls when you start taking the average position
N_thresh = int( N_time / 4)



options = {
        'Number_of_Photons': 1,
        'Complex_Frequency': True,
        'Complex_Derivative_Coupling': dc_bool,
        'Scale_Velocity': sv_bool,
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

''' old way of initializing!!!! 
    deprecated as of 8/28/2020 
polt.R = -0.6
polt.H_e()
polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
polt.Transform_L_to_P()
polt.Derivative_Coupling()
#print("H")
#print(polt.H_polariton)
#print("dc")
#print(polt.dc)
#print("C")
#print(polt.C_polariton)
#print("V")
#print(polt.V)
'''

### New way of doing things is to write all this stuff to file 
### read the relevant quantities and fit splines to the data...
### the splines will then be passed to NH_FSSH
polt.Write_PES(pes_fn, pc_fn, dc_fn, ip_fn)

### fit PES splines
pes_v = np.loadtxt(pes_fn, dtype=complex)
spline_axis = np.real(pes_v[:,0])
 
# g0
re_g0_spline = InterpolatedUnivariateSpline(spline_axis, np.real(pes_v[:,1]), k=3)
im_g0_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(pes_v[:,1]), k=3)

# LP
re_LP_spline = InterpolatedUnivariateSpline(spline_axis, np.real(pes_v[:,2]), k=3)
im_LP_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(pes_v[:,2]), k=3)

# UP
re_UP_spline = InterpolatedUnivariateSpline(spline_axis, np.real(pes_v[:,3]), k=3)
im_UP_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(pes_v[:,3]), k=3)

# e1
re_e1_spline = InterpolatedUnivariateSpline(spline_axis, np.real(pes_v[:,4]), k=3)
im_e1_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(pes_v[:,4]), k=3)

### compute force splines for the g0, lp, and up surfaces
re_g0_force = re_g0_spline.derivative()

re_lp_force = re_LP_spline.derivative()

re_up_force = re_UP_spline.derivative()

### fit derivative coupling splines
dc = np.loadtxt(dc_fn,dtype=complex)

re_dc_23_spline = InterpolatedUnivariateSpline(spline_axis, np.real(dc[:,1]), k=3)
im_dc_23_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(dc[:,1]), k=3)

re_dc_32_spline = InterpolatedUnivariateSpline(spline_axis, np.real(dc[:,2]), k=3)
im_dc_32_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(dc[:,2]), k=3)


### Return to a randomly-chosen initial position and velocity!
### Repeat N_repeats number of times
electronic_file = open(ed_fn, "w")
nuclear_file = open(nuc_traj_fn, "w")
for j in range(0,N_repeats):
    
    
    polt.Initialize_Phase_Space()
    
    ri_init = polt.R
    vi_init = polt.V
    
    ### initialize on the UP surface
    polt.active_index = 2 #polt.initial_state
    polt.Energy = re_UP_spline(polt.R)
    
    ### uncomment if you wish to print the initial conditions
    #print("  Initial Position is ",polt.R)
    #print("  Initial Velocity is ",polt.V)
    ''' deprecated as of 8/28/20 
    ### Build local and polariton matrices!
    polt.H_e()
    polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
    polt.Transform_L_to_P()
    #polt.C_polariton = np.zeros((polt.N_basis_states,polt.N_basis_states),dtype=complex)
    ### hard-coding this!
    polt.C_polariton[polt.initial_state] = 1+0j
    #polt.D_local = np.outer(polt.transformation_vecs_L_to_P[:,polt.initial_state], np.conj(polt.transformation_vecs_L_to_P[:,polt.initial_state])) 
    polt.Energy = polt.Energy_Expectation(polt.H_polariton, polt.C_polariton)
    '''
    
    ### un-comment open files for writing data about electronic and nuclear dynamics
    if N_repeats==1:
        #electronic_file = open(ed_fn, "w")
        #nuclear_file = open(nuc_traj_fn, "w")
        polt.Write_Trajectory(0, nuclear_file, electronic_file)
    
    
    ### start timing
    start = time.time()
    
    ### start dynamics
    r_array = []
    #print("repeat is ", j+1,"active index is ",polt.active_index)
    #print("r_array is ",r_array)
    for i in range(1,N_time):
        
        #### Call FSSH update to update nuclear and electronic coordinates
        polt.NH_FSSH(re_up_force, re_lp_force, re_g0_force, 
                     re_dc_23_spline, im_dc_23_spline, re_dc_32_spline, im_dc_32_spline,
                     re_g0_spline, im_g0_spline, re_LP_spline, im_LP_spline,
                     re_UP_spline, im_UP_spline, re_e1_spline, im_e1_spline)
        
        ### Uncomment if you wish to write trajectory data!
        if N_repeats == 1 and i%500==0:
            
            polt.Write_Trajectory(i, nuclear_file, electronic_file)
            #print("repeat is ", j+1,"active index is ",polt.active_index)
            
        ### After a while, start accumulating the position so you can average it!
        if i>N_thresh:
            r_array.append(polt.R)
    end = time.time()
    time_taken = end-start
    
    ### uncomment if you are writing trajectory data!
    #if N_repeats==2:
    #    electronic_file.close()
    #    nuclear_file.close()

    
    avg_r = sum(r_array) / len(r_array)
    
    #### consider trans if clearly in right basin (r>0.5)
    #### consider cis if clearly in the left basin (r<-0.5)
    #### and consider 50/50 if vib relaxation occurs in the middle
    #### basin, which could occur on the case of large gamma for example
    if avg_r>0.5:
        iso_res = 1
    elif avg_r<-0.5:
        iso_res = -1
    else:
        iso_res = 0
    
        
    print(time_taken/60., ri_init, vi_init, gamp, omp, gp, avg_r, iso_res)
    
if N_repeats==1:
    electronic_file.close()
    nuclear_file.close()
    

