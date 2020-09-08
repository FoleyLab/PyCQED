### Import all libraries and define various parameters here!
import numpy as np
from polaritonic import polaritonic
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from scipy.interpolate import InterpolatedUnivariateSpline


rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

ri_init = -0.66156
vi_init = 3.3375e-5
'''NOTE 1:  when you change gamp, you change the lifetime in Hamiltonian!'''
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
'''NOTE 2:  when you prefix, it will change the file name that the data is written to, and in the next cell that 
   plots the data, this file name will automatically be read from.  Make sure you change the prefix accordingly each 
   time you change the gamp parameter.'''
prefix = "gam_0.0"
### filename to wrote PES to
pes_fn = "Data/" + prefix + '_pes.txt'
### filename to write photonic contributions of each state to
pc_fn = "Data/" + prefix + '_photon_contribution.txt'

### hellman-Feynman file 
hf_fn = "Data/" + prefix + "_hf.txt"
dc_fn = "Data/" + prefix + "_dc.txt"
ip_fn = "Data/" + prefix + "_ip.txt"
filename = prefix + ".eps"

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
#polt.Transform_L_to_P()

Nv = 20
hg = np.linspace(0.01, 0.2, Nv)
hgam = np.linspace(0.1, 100, Nv)

mag_abs = np.zeros((Nv,Nv))
mag_re = np.zeros((Nv,Nv))
mag_im = np.zeros((Nv,Nv))

for i in range(0, Nv):
    c_val = hg[i] / 27.211
    print(" ")
    for j in range(0, Nv):
        gamma_val = hgam[j] * 1e-3 / 27.211

        
        options['Photon_Lifetimes'] = [gamma_val]
        options['Coupling_Strengths'] =  [c_val]
        
        polt = polaritonic(options)
        #polt.gamma_photon = 1000 * 1e-3 / 27.211
        polt.R = -1.25
        ### get local slope curvature of each surface at R = -1.25
        polt.init_slope_and_curve()
        ### write the potential energy surfaces, the derivative couplings, and the polaritonic inner-product and left/right
        ### eigenvector data to files
        polt.Write_PES(pes_fn, pc_fn, dc_fn, ip_fn)
        dc = np.loadtxt(dc_fn,dtype=complex)
        spline_axis = np.real(dc[:,0])
        eval_spline = np.linspace(-1.5, 0, 200)
        
        #re_dc_23_spline = InterpolatedUnivariateSpline(spline_axis, np.real(dc[:,1]), k=3)
        #im_dc_23_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(dc[:,1]), k=3)
        re_dc_32_spline = InterpolatedUnivariateSpline(spline_axis, np.real(dc[:,2]), k=3)
        im_dc_32_spline = InterpolatedUnivariateSpline(spline_axis, np.imag(dc[:,2]), k=3)
        
        abs_list = (re_dc_32_spline(eval_spline)**2 + im_dc_32_spline(eval_spline)**2)**(0.5)
        re_list = np.abs(re_dc_32_spline(eval_spline))
        im_list = np.abs(im_dc_32_spline(eval_spline))
        #print(np.amax(abs_list))
        mag_abs[i,j] = np.amax(abs_list)
        mag_re[i,j] = np.amax(re_list)
        mag_im[i,j] = np.amax(im_list)
        print(gamma_val, c_val, np.amax(abs_list))
