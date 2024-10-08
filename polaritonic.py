#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:58:08 2019

@author: foleyj10

NOTE: As of 02/22/2020, this class assumes that the Hamiltonian matrix can be
non-Hermitian due to the fact that the photonic finite lifetime can be 
represented as a complex frequency:
    omega = omega_r - i * gamma
where omega_r is the ordinary "real" frequency and gamma is related to the lifetime,
i.e. it is the same gamma we feed to the Lindblad operator.
The polariton basis is defined by diagonalizing this non-Hermitian total Hamiltonian
and the derivative couplings are computed using this basis.  

To compute the Liouville-Lindblad equation of motion, the commutator
[H_total, D] only includes the REAL part of the H_total.

The polaritonic potential energy surfaces are computed as the absolute magnitude 
of the complex energies that result from diagonalizing the non-Hermitian H_total

The nuclear forces arise from applying the Hellman-Feynman theorem in the local basis:
    F = -Tr(H'_local D^act_local)
    where H'_local = (H_local(R + dr) - H_local(R - dr))/2dr
and since the H_local(R + dr) and the H_local(R - dr) both have identical
non-Hermtian parts, they cancel and the Force will be real!

FURTHER NOTE: polaritonic_bup.py contains the original versions of functions
that neglected the complex frequency; one can refer to it if we decide this 
non-Hermitian business is bad!

"""
import numpy as np
from numpy import linalg as LA
from scipy import linalg
import math
from numpy.polynomial.hermite import *
from scipy.interpolate import InterpolatedUnivariateSpline


class polaritonic:
    
    ### initializer
    #def __init__(self, mode, inputfile):
    def __init__(self, args):
        
        ### Get basic options from input dictionary, 
        ### get dimensionality and build appropriate local basis
        self.parse_options(args)
        ### allocate space for the Hamiltonian matrices
        ### local basis first
        self.H_electronic = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        self.H_photonic = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)

        self.H_interaction = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        self.H_total = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        
        ### polaritonic basis Hamiltonian
        self.H_polariton = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        
        ### Wavefunction arrays
        self.C_local = np.zeros(self.N_basis_states,dtype=complex)
        self.C_polariton = np.zeros(self.N_basis_states,dtype=complex)        
        ### Population arrays - real vectores!  Will drop imaginary part when
        ### assigning their entries!
        self.population_local = np.zeros(self.N_basis_states)
        self.population_polariton = np.zeros(self.N_basis_states)
        
        self.transformation_vecs_L_to_P = np.zeros((self.N_basis_states, self.N_basis_states),dtype=complex)
        self.l_transformation_vecs_L_to_P = np.zeros_like(self.transformation_vecs_L_to_P)
        self.polariton_energies = np.zeros(self.N_basis_states,dtype=complex)
        
        self.idx = np.zeros(self.N_basis_states)
        self.lidx = np.zeros(self.N_basis_states)
        
        ### Hamiltonians
        self.H_e()
        #print(self.H_electronic)
        self.H_p()
        #print(self.H_photonic)
        self.H_ep()
        #print(self.H_interaction)
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        
        self.C_polariton[self.initial_state] = 1+0j
        
        self.en_up_old = 0+0j
        self.en_up_new = 0+0j
        self.en_lp_old = 0+0j
        self.en_lp_new = 0+0j        
        self.slope_up_old = 0+0j
        self.slope_up_new = 0+0j
        self.slope_lp_old = 0+0j
        self.slope_lp_new = 0+0j
        self.curve_up_old = 0+0j
        self.curve_up_new = 0+0j
        self.curve_lp_old = 0+0j
        self.curve_lp_new = 0+0j
        self.init_slope_and_curve()
        
        ### transform to polariton basis
        self.Transform_L_to_P(0.01)
        ### some helper data to determine consistent ordering of polariton states

                
        ''' Get Total Energy of Initial State!  This will be a complex number! '''        
        self.Energy = self.polariton_energies[self.initial_state]
        
        
        ### derivative coupling
        self.dc = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        ### Hprime matrix
        self.Hprime = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        ### differences between polaritonic surface values... we will 
        ### take the differences between the absolute magnitudes of the energies so
        ### this will be a real vector
        self.Delta_V_jk = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        

        
        ### RK4 SE Variables
        self.kc1 = np.zeros_like(self.C_local)
        self.kc2 = np.zeros_like(self.C_local)
        self.kc3 = np.zeros_like(self.C_local)
        self.kc4 = np.zeros_like(self.C_local)
        self.C1 = np.zeros_like(self.C_local)
        self.C2 = np.zeros_like(self.C_local)
        self.C3 = np.zeros_like(self.C_local)
        self.C4 = np.zeros_like(self.C_local)
        
        
        
        
    ''' Next two methods used to build the local basis '''   
    def printTheArray(self):  
        
        for i in range(0, self.NPhoton+1):  
            #print(arr[i], end = " ")
            self.local_basis[self.basis_cnt,i] = self.temp_basis[i]
            
        self.basis_cnt = self.basis_cnt + 1
        
        return 1
    
    ''' Function to generate all binary strings that represent the local basis
        e.g. for a 1-photon 1-molecule system: 
            |00> represents the molecule in the ground state and no photon in the cavity
            |10> represents the molecule in the excited-state and no photon in the cavity
            |01> represents the molecule in the ground state and 1 photon in the cavity
            |11> represents the molecule in the excited-state and 1 photon in the cavity '''
    def generateAllBinaryStrings(self, i):  
        #print("idx",idx)
        
        if i == self.NPhoton+1: 
            self.printTheArray()
            return 
        # First assign "0" at ith position  
        # and try for all other permutations  
        # for remaining positions
        self.temp_basis[i] = 0
        self.generateAllBinaryStrings(i + 1)  
        
        # And then assign "1" at ith position  
        # and try for all other permutations  
        # for remaining positions  
        self.temp_basis[i] = 1
        self.generateAllBinaryStrings(i + 1)
        
        return 1
    
    ''' Method that parses input dictionary '''
    def parse_options(self, args):
        ### are we scaling velocities among switches/
        if 'Scale_Velocity' in args:
            self.scale = args['Scale_Velocity']
        ### default is to scale!
        else:
            self.scale = 'True'
        #print("scaling condition",self.scale)
        ### are we disregarding the imaginary part of the derivative coupling
        if 'Complex_Derivative_Coupling' in args:
            self.complex_dc = args['Complex_Derivative_Coupling']
        ### default is to keep complex part!
        else:
            self.complex_dc = 'True'
        #print("complex condition",self.complex_dc)
        ### are we treating the photon frequencies as complex quantities?
        if 'Complex_Frequency' in args: 
            self.Complex_Frequency = args['Complex_Frequency']
        else:
            self.Complex_Frequency = False
            
        if 'Initial_Position' in args:
            self.R = args['Initial_Position']
        ### arbitrary defaul initial position
        else:
            self.R = -0.678
        
        if 'Initial_Velocity' in args:
            self.V = args['Initial_Velocity']
        ### arbitray defaul initial velocity
        else:
            self.V = -3.00e-5
        if 'Temperature' in args:
            self.T = args['Temperature']
        else:
            self.T = 0.00095
        if 'Friction' in args:
            self.gamma_nuc = args['Friction']
        else:
            self.gamma_nuc = 0.000012
        ### read mass
        if 'Mass' in args:
            self.M = args['Mass']
        ### Default mass is 1009883 a.u.
        else: 
            self.M = 1009883
        if 'Time_Step' in args:
            self.dt = args['Time_Step']
        else:
            self.dt = 0.12
        if 'Space_Step' in args:
            self.dr = args['Space_Step']
        else:
            self.dr = 0.001
        ### how many photonic modes
        if 'Number_of_Photons' in args:
            self.NPhoton = args['Number_of_Photons']
        else:
            print(" Defaulting to one photon mode")
            self.NPhoton = 1
        ### what are the energies of each photonic mode
        if 'Photon_Energys' in args:
            self.omc = np.array(args['Photon_Energys'])
        ### Default energy is 2.45 eV
        else:
            self.omc = np.zeros(self.NPhoton)
            for i in range(0,self.NPhoton):
                self.omc[i] = 2.45/27.211
        ### what is the coupling strength between each photonic ode
        ### and the molecule
        if 'Coupling_Strengths' in args:
            self.gc = np.array(args['Coupling_Strengths'])
        ### Default coupling strength is 0
        else:
            self.gc = np.zeros(self.NPhoton)
        if 'Photon_Lifetimes' in args:
            self.gamma_photon = np.array(args['Photon_Lifetimes'])
        ### Default lifetime is 0.1 meV
        else:
            self.gamma_photon = np.zeros(self.NPhoton)
            for i in range(0,self.NPhoton):
                self.gamma_photon[i] = 0.1/27211.
        ''' Currently we will assume there is always just 1 molecule in the cavity,
            so that the total number of states is determined based on the fact that there
            is 1 2-level molecule NPhoton 2-level photons, so that the total number of 
            basis states will be 2^(NPhoton + 1).  we will organize the basis
            attribute as follows: it will be an array with dimensions [basis_states, (Nphoton+1)]'''
        self.N_basis_states = 2**(self.NPhoton+1)
        #print("N_basis_states",self.N_basis_states)
        self.local_basis = np.zeros((self.N_basis_states,self.NPhoton+1))
        #print("local_basis",self.local_basis)
        self.temp_basis = np.zeros(self.NPhoton+1)
        #print("temp_basis",self.temp_basis)
        self.basis_cnt = 0
        ### What state will we be in in the local basis?
        if 'Initial_Local_State' in args:
            self.initial_state = args['Initial_Local_State']-1
            if self.initial_state > self.N_basis_states:
                self.initial_state = 0
        else:
            ### if not specified, assume the ground state
            self.initial_state = 0
        ### Active surface is defined in the polariton basis
        ### so it can be different from the initial local state, in principle
        ### however, typically, the two will be very similar so we
        ### will default to using the same state index for both!
        if 'Active_Surface' in args:
            self.active_index = args['Active_Surface']-1
            if self.active_index > self.N_basis_states:
                self.active_index = 0
        else:
            self.active_index = self.initial_state
        
        self.generateAllBinaryStrings(0)
        
        ### get photon dissipation rates for Lindblad operator
        self.gamma_diss = np.zeros(self.N_basis_states)
        for i in range(0,self.N_basis_states):
            for j in range(1,self.NPhoton+1):
                if self.local_basis[i,j]==1:
                    self.gamma_diss[i] = self.gamma_diss[i] + self.gamma_photon[j-1]
                    
        
        return 1
    
    ''' Methods used to build the Hamiltonian at the current value of self.R '''
    def E_of_R(self):
        Ai = np.array([0.049244, 0.010657, 0.428129, 0.373005])
        Bi = np.array([0.18, 0.18, 0.18, 0.147])
        Ri = np.array([-0.75, 0.85, -1.15, 1.25])
        Di = np.array([0.073, 0.514])
        
        v = Ai + Bi*(self.R - Ri)**2
        self.Eg = 0.5*(v[0] + v[1]) - np.sqrt(Di[0]**2 + 0.25 * (v[0] - v[1])**2)
        self.Ee = 0.5*(v[2] + v[3]) - np.sqrt(Di[1]**2 + 0.25 * (v[2] - v[3])**2)
        
        return 1
    '''  Computes electronic Hamiltonian matrix '''
    def H_e(self):
        
        ### get self.Eg and self.Ee
        self.E_of_R()
        
        for i in range(0,self.N_basis_states):
            if self.local_basis[i,0] == 0:
                self.H_electronic[i,i] = self.Eg+0j
            elif self.local_basis[i,0] == 1:
                self.H_electronic[i,i] = self.Ee+0j
                
        return 1
    
    '''  Computes photonic Hamiltonian with complex frequencies '''
    def H_p(self):
        ci = 0+1j
        for i in range(0,self.N_basis_states):
            cval = 0+0j
            for j in range(1,self.NPhoton+1):
                if self.local_basis[i,j] == 0:
                    cval = cval + 0.0 * (self.omc[j-1] - ci * self.gamma_photon[j-1]/2.)
                elif self.local_basis[i,j] == 1:
                    cval = cval + 1.0 * (self.omc[j-1] - ci * self.gamma_photon[j-1]/2.)
            self.H_photonic[i,i] = cval            
            
        return 1
    
    '''  Computes molecule-photon coupling Hamiltonian matrix  '''
    def H_ep(self):
        
        for i in range(0,self.N_basis_states):
            
            for j in range(0,self.N_basis_states):
                
                val = 0+0j
                ### only proceed if one molecular basis state is \g> and the other |e>
                if self.local_basis[i,0] != self.local_basis[j,0]:
                    
                    for k in range(1,self.NPhoton+1):
                        
                        t1 = self.action(j, 't1', k )
                        t2 = self.action(j, 't2', k )
                        t3 = self.action(j, 't3', k )
                        t4 = self.action(j, 't4', k )
                        
                        if np.array_equal(t1, self.local_basis[i,:]):
                            val = val + self.gc[k-1]
                        if np.array_equal(t2, self.local_basis[i,:]):
                            val = val + self.gc[k-1]
                        if np.array_equal(t3, self.local_basis[i,:]):
                            val = val + self.gc[k-1]
                        if np.array_equal(t4, self.local_basis[i,:]):
                            val = val + self.gc[k-1]


                self.H_interaction[i,j] = val
        return 1
    
    ''' function that operates on basis state with string of 2nd
        quantized operators and returns the resulting basis state
        essentially we have 4 different strings to consider 
        t1 = b_i^+ a_e^+ a_g
        t2 = b_i^+ a_g^+ a_e
        t3 = b_i   a_e^+ a_g
        t4 = b_i   a_g^+ a_e '''
    def action(self, state_indx, term, photon_indx):
        ### flag that indicates if action sends state to zero or not
        ### zero==1 means state sent to zero, zero==0 means state is valid
        zero=0
        ### make temporary copy of basis state
        tmp_bas = np.copy(self.local_basis[state_indx,:])
        ### decide how to act on it
        if term=='t1':
            ### first annihilate a_g and create a_e^+
            if tmp_bas[0]==0:
                tmp_bas[0] = 1
            else:
                zero=1
            ### now create b_i^+
            if tmp_bas[photon_indx]==0:
                tmp_bas[photon_indx]=1
            else:
                zero=1
        elif term=='t2':
            ### first annihilate a_e and create a_g^+
            if tmp_bas[0]==1:
                tmp_bas[0] = 0
            else:
                zero=1
            ### now create b_i^+
            if tmp_bas[photon_indx]==0:
                tmp_bas[photon_indx]=1
            else:
                zero=1
        elif term=='t3':
            ### first annihilate a_g and create a_e^+
            if tmp_bas[0]==0:
                tmp_bas[0] = 1
            else:
                zero=1
            ### now annihilate b_i
            if tmp_bas[photon_indx]==1:
                tmp_bas[photon_indx]=0
            else:
                zero=1
        elif term=='t4':
            ### first annihilate a_e and create a_g^+
            if tmp_bas[0]==1:
                tmp_bas[0] = 0
            else:
                zero=1
            ### now annihilate b_i
            if tmp_bas[photon_indx]==1:
                tmp_bas[photon_indx]=0
            else:
                zero=1
        ### If invalid option given, nullify state and print warning
        else:
            print("Warning: Invalid operator string specified!")
            zero=1
        ### This is a pretty silly way to indicate the state has been 'killed',
        ### but it will make sure that comparisons of this tmp_bas state
        ### with any other state in the local_basis will come up false
        if zero:
            tmp_bas[0] = -1
        
        return tmp_bas
    
    ''' get initial slopes and curvatures of UP and LP surface.
        we will assume at the initial point, the LP/UP are in increasing
        energy order, and we will attempt to pick the ordering
        such that slope and curvature vary smoothly.'''
    def init_slope_and_curve(self):
        dim = 5
        delta_r = 0.01
        r_vals = np.array([self.R - 2*delta_r,self.R - 1*delta_r,self.R,self.R + 1*delta_r,self.R + 2*delta_r])
        LP_E = np.zeros(dim)
        UP_E = np.zeros(dim)
        ### get a stencil of UP and LP energies
        for i in range(0,dim):
            self.R = r_vals[i]
            self.H_e()
            self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
            vals, vecs = LA.eig(self.H_total)
            idx = vals.argsort()[::1]
            vals = vals[idx]
            LP_E[i] = np.real(vals[1])
            UP_E[i] = np.real(vals[2])
        
        ### fit spline to upper and lower polariton surface
        LP_spline = InterpolatedUnivariateSpline(r_vals, LP_E, k=3)
        UP_spline = InterpolatedUnivariateSpline(r_vals, UP_E, k=3)
        
        ### differentiate upper and lower polariton surfaces
        LP_slope = LP_spline.derivative()
        UP_slope = UP_spline.derivative()
        
        ### 2nd derivative of UP and LP surfaces
        LP_curve = LP_slope.derivative()
        UP_curve = UP_slope.derivative()
        
        ### store quantities
        self.en_up_old = UP_spline(self.R)
        self.en_up_new = UP_spline(self.R+delta_r)
        self.en_lp_old = LP_spline(self.R)
        self.en_lp_new = LP_spline(self.R+delta_r)
        self.slope_up_old = UP_slope(self.R)
        self.slope_up_new = UP_slope(self.R+delta_r)
        self.slope_lp_old = LP_slope(self.R)
        self.slope_lp_new = LP_slope(self.R+delta_r)
        self.curve_up_old = UP_curve(self.R)
        self.curve_up_new = UP_curve(self.R+delta_r)
        self.curve_lp_old = LP_slope(self.R)
        self.curve_lp_new = LP_slope(self.R+delta_r)
        
        return 1
        
            
        
    
    ''' Diagonalize self.H_total, which is typically a non-Hermitian Hamiltonian in the 
        local basis and store the eigenvalues and eigenvectors 
        to the attributes 
        self.polariton_energies and 
        self.transofrmation_vecs_L_to_P...
        Will also store the diagonalized Hamiltonian in the attribute
        self.H_polariton and transform the local density matrix to the 
        corresponding polariton basis and store it in
        self.D_polariton
        '''
    def Transform_L_to_P(self, delta_r):
        #print(first)
        ### left vectors
        lvals, lvecs = linalg.eig(self.H_total, left=True, right=False)
        ### right vectors
        #vals, vecs = linalg.eig(polt.H_total, left=False, right=True)
        vals, vecs = LA.eig(self.H_total)
        ### sort the right eigenvectors
        idx = vals.argsort()[::1]
        tvals = vals[idx]
        
        ### estimate UP energy based on prior UP energy and slope and curvature
        self.en_up_new = self.en_up_old + self.slope_up_old * delta_r + 0.5 * self.curve_up_old * delta_r ** 2
        self.en_lp_new = self.en_lp_old + self.slope_lp_old * delta_r + 0.5 * self.curve_lp_old * delta_r ** 2
        
        ### error 1 being smaller implies current odering is good!
        error_1 = np.abs(self.en_lp_new - tvals[1]) + np.abs(self.en_up_new - tvals[2])
        ### error 2 being smaller implies we need to switch the UP/LP indices!
        error_2 = np.abs(self.en_lp_new - tvals[2]) + np.abs(self.en_up_new - tvals[1])
        if error_2<error_1:
            UP_idx = idx[1]
            idx[1] = idx[2]
            idx[2] = UP_idx
        
        v = np.copy(vecs[:,idx])
        vals = np.copy(vals[idx])
        
        ### sort the left eigenvectors
        lvals = np.copy(lvals[idx])
        lv = np.copy(lvecs[:,idx])
        
        
        #v = np.copy(vecs)
        #lv = np.copy(lvecs)
        
        ### store eigenvectors and eigenvalues
        self.transformation_vecs_L_to_P = np.copy(v)
        self.l_transformation_vecs_L_to_P = np.copy(lv)
        self.polariton_energies = np.copy(vals)
        ### transform Htot with v^-1
        vt0 = np.dot(LA.inv(v),self.H_total)
        ### finish transformation to polariton basis, Hpl
        self.H_polariton = np.copy(np.dot(vt0,v))
        
        ### now update the en/slope/curve quantities
        self.en_lp_new = vals[1]
        self.en_up_new = vals[2]
        self.slope_lp_new = (self.en_lp_new-self.en_lp_old)/delta_r
        self.slope_up_new = (self.en_up_new-self.en_up_old)/delta_r
        self.curve_lp_new = (self.slope_lp_new-self.slope_lp_old)/delta_r
        self.curve_up_new = (self.slope_up_new-self.slope_up_old)/delta_r
        self.slope_lp_old = self.slope_lp_new
        self.slope_up_old = self.slope_up_new
        self.curve_lp_old = self.curve_lp_new
        self.curve_up_old = self.curve_up_new
        

        return 1
    
    

    '''  4th-order Runge-Kutta Algorithm for solving non-Hermitian 
         Schrodinger Equation in polariton basis! '''
    def RK4_NH_SE(self):
        ci = 0+1j
        ### make a copy of the polariton Hamiltonian
        H = np.copy(self.H_polariton)
        ### make a copy of the derivative coupling matrix in the polariton basis
        d = np.copy(self.dc)
        
        
        ### Copy current density matrix
        self.C1 = np.copy(self.C_polariton)  
        
        ### First partial update
        self.kc1 = np.copy(-ci * self.dt * np.dot(H,self.C1) - 
                           self.dt * self.V * np.dot(d, self.C1))
        
        

        ### Update D and get k2
        self.C2 = np.copy(self.C_polariton+self.kc1/2.)
        ### Second partial update
        self.kc2 = np.copy(-ci * self.dt * np.dot(H,self.C2) - 
                           self.dt * self.V * np.dot(d, self.C2))
        
        
        ### UPdate D and get k3
        self.C3 = np.copy(self.C_polariton+self.kc2/2)
        self.kc3 = np.copy(-ci * self.dt * np.dot(H,self.C3) - 
                           self.dt * self.V * np.dot(d, self.C3))
        
        ### Update H and D and get K4
        self.C4 = np.copy(self.C_polariton+self.kc3)
        self.kc4 = np.copy(-ci * self.dt * np.dot(H,self.C4) - 
                           self.dt * self.V * np.dot(d, self.C4))
        
        self.C_polariton = np.copy(self.C_polariton + 
                                   (1/6.)*(self.kc1 + 2.*self.kc2 + 2*self.kc3 + self.kc4))
        
        return 1


    ''' Non-Hermitian FSSH update '''
    def NH_FSSH(self, up_force, lp_force, g0_force, dc_23_re, dc_23_im, dc_32_re, dc_32_im,
                pes_g0_re, pes_g0_im, pes_lp_re, pes_lp_im, pes_up_re, pes_up_im, pes_e1_re, pes_e1_im):
        ### set up some quantities for surface hopping first!
        ci = 0+1j
        switch=0
        sign = 1
        starting_act_idx = self.active_index
        ### allocate a few arrays we will need
        pop_cur = np.zeros(self.N_basis_states,dtype=complex)
        pop_fut = np.zeros(self.N_basis_states,dtype=complex)
        pop_dot = np.zeros(self.N_basis_states,dtype=complex)
        ### Will take the real part of the pop_dot to compute gik, 
        ### so it is a real vector!
        gik = np.zeros(self.N_basis_states)
        
        ''' Update nuclear degrees of freedom first '''
        #  Force from spline - will use the appropriate spline
        #  based on which surface we are on!
        if self.active_index==2:
            F_curr = -1*up_force(self.R)
        elif self.active_index==1:
            F_curr = -1*lp_force(self.R)
        else:
            F_curr = -1*g0_force(self.R)
        
        ### Depricated as of 8/28/2020     
        #F_curr = np.real(self.Hellman_Feynman())
        
        ### get perturbation of force for Langevin dynamics
        rp_curr = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1) * 0
    
        ### get acceleration
        ### Langevin acceleration
        a_curr = (F_curr + rp_curr) / self.M - self.gamma_nuc * self.V
        ### bbk update to velocity and position
        v_halftime = self.V + a_curr * self.dt / 2
        
        ### update R
        self.R = self.R + v_halftime * self.dt
        
        if self.active_index==2:
            F_fut = -1*up_force(self.R)
            self.Energy = pes_up_re(self.R)
        elif self.active_index==1:
            F_fut = -1*lp_force(self.R)
            self.Energy = pes_lp_re(self.R)
        else:
            F_fut = -1*g0_force(self.R)
            self.Energy = pes_g0_re(self.R)
        
        ### Depricated as of 8/28/2020
        #Hellman-Feynman force at updated geometry 
        #F_fut = np.real(self.Hellman_Feynman())
        
        ### get new random force 
        rp_fut = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1) * 0
        ### get acceleration
        a_fut = (F_fut + rp_fut) / self.M - self.gamma_nuc * v_halftime
        ### get updated velocity
        ### vv update
        ###v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
        ### bbk update
        ### updated velocity assuming we are on the same surface
        self.V = (v_halftime + a_fut * self.dt/2) * (1/(1 + self.gamma_nuc * self.dt/2))
        
        ### are we moving forward or backwards?
        if self.V>0:
            forward=1
        else:
            forward=-1
        
        
        ### Get derivative coupling at updated position!
        #### depricated as of 8/28/2020
        #self.Derivative_Coupling()
        ### now use splines for derivative coupling!
        self.dc[1,2] = dc_23_re(self.R)+ci*dc_23_im(self.R)
        self.dc[2,1] = dc_32_re(self.R)+ci*dc_32_im(self.R)
        
        ### now use splines to update the polaritonic Hamiltonian
        self.H_polariton[0,0] = pes_g0_re(self.R)+ci*pes_g0_im(self.R)
        self.H_polariton[1,1] = pes_lp_re(self.R)+ci*pes_lp_im(self.R)
        self.H_polariton[2,2] = pes_up_re(self.R)+ci*pes_up_im(self.R)
        self.H_polariton[3,3] = pes_e1_re(self.R)+ci*pes_e1_im(self.R)
        
        ### populations before updates
        for i in range(0,self.N_basis_states):
            pop_cur[i] = self.population_polariton[i]
        
        ### Update wavefunction
        self.RK4_NH_SE()
        
        ### update populations in polariton basis
        trace = 0.
        dot_trace = 0.
        for i in range(1,self.N_basis_states):
            p_i = np.real( np.conj(self.C_polariton[i]) * self.C_polariton[i])
            self.population_polariton[i] = p_i
            pop_fut[i] = p_i
            trace += p_i
            pop_dot[i] = (p_i - pop_cur[i])/self.dt
            dot_trace += pop_dot[i]
        
        pop_dot[0] = -1*dot_trace
        pop_fut[0] = 1 - trace
        self.population_polariton[0] = 1 - trace
        #print("population_polariton",self.population_polariton)
        
        ''' Compute probabilities for switching surfaces '''
        for i in range(0,self.active_index):
            ### if population in the active state is greater than a number vvvvery close to zero, it 
            ### can go in the denmoniator
            if self.population_polariton[self.active_index]>1e-13:
                g = np.real( pop_dot[i] / self.population_polariton[self.active_index] * self.dt)
            ### if it is zero, divide by something realllllly small instead!
            else:
                g = np.real( pop_dot[i] / 1e-13 * self.dt)
            if (g<0):
                g = 0          
            
            gik[i] = g
        
        ### decide if we want to hop to state k, if any
        thresh = np.random.random(1)
        ### this logic comes from the conditions in Eq.  (10) on page 4 
        ### from J. Chem. Phys. 138, 164106 (2013); doi: 10.1063/1.4801519
        ### are we in state Phi_3?
        if (self.active_index==2):

            #### switch to 0 if probability is larger than thresh
            if gik[0]>thresh:
                self.active_index = 0
                switch=1
                #print("switched from 3->1")
            #### otherwise if cumulative probability of switching to 1 is larger than thresh
            if (gik[0]+gik[1])>thresh:
                self.active_index = 1
                switch=1
                #print("switched from 3->2")
            else:
                switch = 0
        ### are we in state Phi_2? 
        elif (self.active_index==1):
            if gik[0]>thresh:
                self.active_index = 0
                switch = 1
                #print("switched from 2->1")
            else:
                switch = 0
        else:
            switch = 0 
        ### if we switched surfaces, we need to check some things about the
        ### momentum on the new surface... this comes from consideration of
        ### Eq. (7) and (8) on page 392 of Subotnik's Ann. Rev. Phys. Chem.
        ### on Surface Hopping
        ### check to see if we want to scale or not by value of self.scale!
        if switch and self.active_index==1 and self.scale:
            ### momentum on surface j (starting surface)
            Pj = self.V*self.M
            ### This number should always be positive!
            ### We will discard the imaginary part
            init = starting_act_idx
            fin = self.active_index
            ''' deprecated as of 8/28/2020 '''
            ####delta_V = np.real(self.Delta_V_jk[starting_act_idx,self.active_index])
            
            delta_V = np.real(self.H_polariton[init,init] - self.H_polariton[fin, fin])
            ### speed on surface k (new surface)
            Vk_mag = np.sqrt(2 * self.M * (Pj**2/(2*self.M) + delta_V)) / self.M
            Vk = forward * Vk_mag
            
            Pk = self.M * Vk
            ### First estimate of Delta P
            Delta_P = Pk - Pj
        
            #print("DP ",Delta_P,"Pj ", Pj,"Pk ", Pk, "dc_ij ", self.dc[starting_act_idx, self.active_index])    
            ### We will re-scale the updated momentum so that the following is true: 
            ### Pj = Pk + deltaP * dc
            ### if dc vanishes (as will happen for j->0), do not rescale the velocity
            ### why?  bc these transitions are due to the photon leaving the cavity, 
            ### and the photon should carry energy away with it so we don't want to conserve energy!
            if np.isclose(self.dc[starting_act_idx, self.active_index],0+0j):
                self.V = Pj/self.M
            elif self.active_index==0:
                self.V = Pj/self.M
            ### if derivative coupling is positive, then this hop cannot happen!
            elif np.real(self.dc[starting_act_idx, self.active_index])>0:
                self.active_index = starting_act_idx
                self.V = Pj/self.M
            ### hops with negative derivative coupling are allowed, rescale the velocity
            ### appropriately.
            else:
                ### Re-scale Delta P with derivative coupling vector
                scaled_Delta_P = Delta_P / np.real(self.dc[starting_act_idx, self.active_index])
                ### now compute the re-scaled Pk
                Pk_rescaled = Pj - scaled_Delta_P
                ### assign the corresponding re-scaled velocity to self.V
                self.V = Pk_rescaled / self.M
                #print("Pj ", Pj,"Pk ", Pk, "Pk_rs", Pk_rescaled, "dc_ij ", self.dc[starting_act_idx, self.active_index])
                         
        
        return 1 

    ''' Computes Hellman-Feynman force '''
    def Hellman_Feynman(self):
        ### get transformation vector at current R
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        self.Transform_L_to_P(self.dr)
        c = self.transformation_vecs_L_to_P[:,self.active_index]
        ### update energy attribute while we are at it!
        self.Energy = self.polariton_energies[self.active_index]
        cs = np.conj(c)
        
        ### get dH/dR
        self.R = self.R + self.dr
        self.H_e()
        Hf = np.copy(self.H_electronic)
        self.R = self.R - 2*self.dr
        self.H_e()
        Hb = np.copy(self.H_electronic)
        Hp = (Hf - Hb)/(2*self.dr)
        ### Return to innocence
        self.R = self.R + self.dr
        tmp = np.dot(Hp, c)
        F = -1*np.dot(cs,tmp)
        return F
    
    ''' Compute energy expectation value for a given Hamiltonian and c-vector '''
    def Energy_Expectation(self, H, c):
        ct = np.conj(c)
        Hc = np.dot(H, c)
        num = np.dot(ct, Hc)
        den = np.dot(ct, c)
        #print(num)
        #print(den)
        return num/den
         
    ''' Computes Derivative Coupling Matrix '''
    def Derivative_Coupling(self):
        
        ### Compute H_prime
        ### Forward displacement
        self.R = self.R + self.dr
        
        ### update H_e
        self.H_e()
        ### H at forward step
        Hf = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        ### backward displacement
        self.R = self.R - 2*self.dr
        ### update electronic H
        self.H_e()
        Hb = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        Hp = np.copy((Hf - Hb)/(2*self.dr))
        ### Compute H
        self.R = self.R + self.dr
        self.H_e()
        ### H at forward step
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        self.Transform_L_to_P(self.dr)

        
        for j in range(0, self.N_basis_states): #self.N_basis_states):
            for k in range(0, self.N_basis_states): #self.N_basis_states):
                if j!=k:
                    Vjj = self.polariton_energies[j]
                    Vkk = self.polariton_energies[k]
                    cj = self.transformation_vecs_L_to_P[:,j]
                    ck = self.transformation_vecs_L_to_P[:,k]
                    tmp = np.dot(Hp, ck)
                    Fjk = -1*np.dot(np.conj(cj), tmp)
                    ### check if we should disregard the imaginary part
                    if self.complex_dc:
                        self.dc[j,k] = Fjk/(Vkk-Vjj)
                    else:
                        self.dc[j,k] = np.real(Fjk/(Vkk-Vjj))
                        
                    self.Delta_V_jk[j,k] = Vjj-Vkk
        return 1
        
    
    ''' Function to initialize position and velocity for a nuclear trajectory; will use
        hard-coded values of the Mass and Force Constant for the time  being but should be
        generalized!  '''
    def Initialize_Phase_Space(self):
        
        from random import seed
        from random import gauss
        import time
        k = 0.31246871517560126
        M = self.M
        ### alpha from Harmonic Oscillator eigenfunction
        a = np.sqrt(k*M)
        ### normalization from Harmonic oscillator eigenfunction
        N = (a/np.pi)**(1/4.)
        ### definition of sigma from Gaussian function, which 
        ### is equivalent to position uncertainty
        sig = np.sqrt(1/(2*a))
        ### momentum uncertainty for Harmonic Oscillator ground-state
        p_unc = np.sqrt(a - a**2 * np.sqrt(np.pi)/(2*a**(3/2)))
        ### velocity uncertainty for the same
        v_unc = p_unc / M
        
        ### seed random number generator
        seed(time.time())
        ### Get random value of R
        self.R = gauss(-0.7, sig)
        ### Get random value of V
        self.V = gauss(0, v_unc)
        
        return 1
    
    ''' The following methods will write different quantities to file! 
        Write_PES will 
        - take file-name strings for the potential energy surface
          and for the photon-contribution "surface"
        - compute both of these surfaces
        - write it to a file with names given by the respective file-name strings
    '''
    def Write_PES(self, pes_fn, pc_fn, dc_fn, ptdc_fn, ip_fn):
        
        rlist = np.linspace(-1.25, 1.25, 2000)
        pes_dr = rlist[1]-rlist[0]
        
        ### temporary arrays for old eigenvectors
        g0_old = np.zeros(self.N_basis_states, dtype='complex')
        LP_old = np.zeros(self.N_basis_states, dtype='complex')
        UP_old = np.zeros(self.N_basis_states, dtype='complex')
        e1_old = np.zeros(self.N_basis_states, dtype='complex')
        
        H_old = np.zeros((self.N_basis_states, self.N_basis_states), dtype=complex)
        
        ### Get PES of polaritonic system and write to file pes_fn
        pes_file = open(pes_fn, "w")
        pc_file = open(pc_fn, "w")
        dc_file = open(dc_fn, "w")
        ptdc_file = open(ptdc_fn, "w")
        ip_file = open(ip_fn, "w")

        for r in range(0,len(rlist)):
            wr_str = " "
            pc_str = " "
            self.R = rlist[r]
            wr_str = wr_str + str(self.R) + " "
            pc_str = pc_str + str(self.R) + " "
            self.H_e()
            self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
            self.Transform_L_to_P(pes_dr)
            v = np.copy(self.transformation_vecs_L_to_P)
            lv =  np.copy(self.l_transformation_vecs_L_to_P)
            
            g0 = np.copy(v[:,0])
            LP = np.copy(v[:,1])
            UP = np.copy(v[:,2])
            e1 = np.copy(v[:,3])
            lg0 = np.copy(lv[:,0])
            lLP = np.copy(lv[:,1])
            lUP = np.copy(lv[:,2])
            le1 = np.copy(lv[:,3])
            
            ip_22 = np.dot(np.conj(LP), LP)
            ip_33 = np.dot(np.conj(UP), UP)
            ip_23 = np.dot(np.conj(LP), UP)
            ip_32 = np.dot(np.conj(UP), LP)
            
            lLPvrLP = np.sum(np.conj(lLP)-LP)
            lUPvrUP = np.sum(np.conj(lUP)-UP)
            ip_str = str(self.R) + " " + str(ip_22) + " " + str(ip_33) + " " + str(ip_23) + " " + str(ip_32)
            ip_str = ip_str + " " + str(lLPvrLP) + " " + str(lUPvrUP) + "\n"
            ip_file.write(ip_str)
            for i in range(0,self.N_basis_states):
                v_i = np.copy(v[:,i])
                cv_i = np.copy(np.conj(v_i))
                
                wr_str = wr_str + str(np.real(self.polariton_energies[i])) + " "
                
                ### loop over all photon indices in basis states
                pc = 0
                for j in range(0,self.N_basis_states):
                    if self.gamma_diss[j]>0:
                        pc = pc + np.real(cv_i[j] * v_i[j])
                        
                pc_str = pc_str + str(pc) + " "
            
            ### compute derivative couplings
            if (r>0):
                ### first the "true" way, e.g. <LP | d/dR UP>
                #dg0 = (g0-g0_old)/pes_dr
                dLP = np.copy((LP-LP_old)/pes_dr)
                dUP = np.copy((UP-UP_old)/pes_dr)
                #de1 = (e1-e1_old)/pes_dr
                
                #d12 = np.dot(np.conj(g0),dLP)
                #d13 = np.dot(np.conj(g0),dUP)
                #d14 = np.dot(np.conj(g0),de1)
                
                d23 = np.dot(np.conj(LP),dUP)
                d32 = np.dot(np.conj(UP),dLP)
                #d24 = np.dot(np.conj(LP),de1)
                
                #d34 = np.dot(np.conj(UP),de1)
                
                d_str = str(self.R) + " " + str(d23) + " " + str(d32) + "\n"
                dc_file.write(d_str)
                
                ### Now the perturbative way!
                ### derivative of H
                Vlp = self.polariton_energies[1]
                Vup = self.polariton_energies[2]
                Hp = np.copy((self.H_total - H_old)/pes_dr)
                tmp_23 = np.dot(Hp, UP)
                num_pt_23 = np.dot(np.conj(LP), tmp_23)
                tmp_32 = np.dot(Hp, LP)
                num_pt_32 = np.dot(np.conj(UP), tmp_32)
                pt_32 = -1 * num_pt_32/(Vlp - Vup)
                pt_23 = -1 * num_pt_23/(Vup - Vlp)
                pt_str = str(self.R) + " " + str(pt_23) + " " + str(pt_32) + "\n"
                ptdc_file.write(pt_str)

            g0_old = np.copy(g0)
            LP_old = np.copy(LP)
            UP_old = np.copy(UP)
            e1_old = np.copy(e1)
                
                
                
        
            wr_str = wr_str + "\n"
            pc_str = pc_str + "\n"        
            pes_file.write(wr_str)
            pc_file.write(pc_str)
        
        ### Close PES file
        pes_file.close()
        pc_file.close()
        dc_file.close()
        ip_file.close()
        ptdc_file.close()
        
        return 1
    
    ''' This method will write the nuclear and electronic trajectories
        to an open-file that is passed as an argument... it also takes the
        current time-step number (note time, but the integer time-step number) '''
    def Write_Trajectory(self, step, nuclear_file, electronic_file):
        

        e_str = " "
        n_str = " "
        e_str = e_str + str(step*self.dt) + " "
        n_str = n_str + str(step*self.dt) + " "
        n_str = n_str + str(np.real(self.R)) + " " + str(np.real(self.Energy)) + " " 
        n_str = n_str + str(np.real(self.V)) + " " 
        
        #for j in range(0,self.N_basis_states):
        #    e_str = e_str + str(np.real(self.D_local[j,j])) + " "
            
        for j in range(0,self.N_basis_states):
            cj = self.C_polariton[j]
            pj = np.real( np.conj(cj) * cj)
            e_str = e_str + str(pj) + " " 
            e_str = e_str + str(np.real(self.population_polariton[j])) + " "
            #e_str = e_str + str(np.real(self.C_polariton[j])) + " " 
            #e_str = e_str + str(np.imag(self.C_polariton[j])) + " "
            
        e_str = e_str + "\n"
        n_str = n_str + "\n"
        electronic_file.write(e_str)
        nuclear_file.write(n_str)
        
        return 1

    ''' This method computes the forces using centered finite differences
        of the energy on surface 2 and 3, and also the Hellman-Feynman force on 
        polariton surfaces 2 and 3, and prints to file. It also computes the
        derivative coupling vectors for 3->2 and writes to file. '''
    def Write_Forces(self, prefix):
        
        init_active_index = self.active_index
        hf_file = open(prefix, "w")
        rlist = np.linspace(-1.0, 1.0, 500)
        
        for r in range(0,len(rlist)):
            wr_str = " "
            self.R = rlist[r]
            wr_str = wr_str + str(self.R) + " "
            
            
            ### Get derivative coupling at current position... 
            self.Derivative_Coupling()
            
            ### Get Hellman-Feynman force on surface Phi_3
            self.active_index = 2
            ### Hellman-Feynman Forces
            F_33 = self.Hellman_Feynman()
            ### Get Hellman-Feynman force on surface Phi_2
            self.active_index = 3
            F_22 = self.Hellman_Feynman()
            
            
            d_32 = self.dc[2,1]
            
            wr_str = wr_str + str(F_22) + " " + str(F_33) + " " + str(d_32) + "\n"
            hf_file.write(wr_str)
        
        
        self.active_index = init_active_index 
        hf_file.close()
        return 1
    

