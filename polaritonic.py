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
import math
from numpy.polynomial.hermite import *


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
        
        ### Density matrix  arrays
        self.D_local = np.zeros((self.N_basis_states, self.N_basis_states),dtype=complex)
        self.D_polariton = np.zeros((self.N_basis_states, self.N_basis_states),dtype=complex)
        
        ### Wavefunction arrays
        self.C_local = np.zeros(self.N_basis_states,dtype=complex)
        self.C_polariton = np.zeros(self.N_basis_states,dtype=complex)        
        ### Population arrays - real vectores!  Will drop imaginary part when
        ### assigning their entries!
        self.population_local = np.zeros(self.N_basis_states)
        self.population_polariton = np.zeros(self.N_basis_states)
        
        self.transformation_vecs_L_to_P = np.zeros((self.N_basis_states, self.N_basis_states),dtype=complex)
        self.polariton_energies = np.zeros(self.N_basis_states,dtype=complex)
        
        ### Hamiltonians
        self.H_e()
        #print(self.H_electronic)
        self.H_p()
        #print(self.H_photonic)
        self.H_ep()
        #print(self.H_interaction)
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        
        ### Density Matrices
        self.D_local[self.initial_state,self.initial_state] = 1+0j
        self.C_polariton[self.initial_state] = 1+0j
        
        
        ### derivative coupling
        self.dc = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        ### Hprime matrix
        self.Hprime = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        ### differences between polaritonic surface values... we will 
        ### take the differences between the absolute magnitudes of the energies so
        ### this will be a real vector
        self.Delta_V_jk = np.zeros((self.N_basis_states,self.N_basis_states),dtype=complex)
        
        self.Transform_L_to_P()
        
        ### RK4 Liouville Variables
        self.k1 = np.zeros_like(self.D_local)
        self.k2 = np.zeros_like(self.D_local)
        self.k3 = np.zeros_like(self.D_local)
        self.k4 = np.zeros_like(self.D_local)
        self.D1 = np.zeros_like(self.D_local)
        self.D2 = np.zeros_like(self.D_local)
        self.D3 = np.zeros_like(self.D_local)
        self.D4 = np.zeros_like(self.D_local)
        
        ### RK4 SE Variables
        self.kc1 = np.zeros_like(self.C_local)
        self.kc2 = np.zeros_like(self.C_local)
        self.kc3 = np.zeros_like(self.C_local)
        self.kc4 = np.zeros_like(self.C_local)
        self.C1 = np.zeros_like(self.C_local)
        self.C2 = np.zeros_like(self.C_local)
        self.C3 = np.zeros_like(self.C_local)
        self.C4 = np.zeros_like(self.C_local)
        
        
        
        
        
        ''' Form identity matrix and use to create a set of projectors for 
            each basis state... this will be used in the Lindblad operators! '''
        self.DM_Bas = np.identity(self.N_basis_states,dtype=complex)
        
        self.DM_Projector = np.zeros((self.N_basis_states, self.N_basis_states, self.N_basis_states,self.N_basis_states),dtype=complex)
        
        for i in range(0, self.N_basis_states):
            for j in range(0, self.N_basis_states):
                self.DM_Projector[:,:,i,j] = np.outer(self.DM_Bas[i,:], np.conj(self.DM_Bas[j,:]))
        
        ''' Get Total Energy of Initial State!  This will be a complex number! '''        
        self.Energy = self.TrHD(self.H_total, self.D_local)
            
        
        
        
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
            self.gamma_nuc = 0.000011
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
    def Transform_L_to_P(self):
        vals, vecs = LA.eig(self.H_total)
        ### sort the eigenvectors
        idx = vals.argsort()[::1]
        vals = vals[idx]
        v = vecs[:,idx]
        ### store eigenvectors and eigenvalues
        self.transformation_vecs_L_to_P = np.copy(v)
        self.polariton_energies = np.copy(vals)
        ### transform Htot with v^-1
        vt0 = np.dot(LA.inv(v),self.H_total)
        ### finish transformation to polariton basis, Hpl
        self.H_polariton = np.dot(vt0,v)
        ### now do transformation for density matrix from local to polariton basis
        dt0 = np.dot(LA.inv(v), self.D_local)
        self.D_polariton = np.dot(dt0,v)
        ### return Hpl and Dpl

        return 1
    
    ''' Transform from polariton to local basis if needed! '''
    def Transform_P_to_L(self):
        ### now do transformation for density matrix from local to polariton basis
        dt0 = np.dot(self.transformation_vecs_L_to_P, self.D_polariton)
        self.D_local = np.dot(dt0, LA.inv(self.transformation_vecs_L_to_P))
        ### return Hpl and Dpl
        return 1
    
    ''' Propagate Density matrix in local basis using the Liouville-Lindblad
        equation of motion.  Note that we will use only the REAL part of 
        self.H_total in the commutators.  However, the EOM will still be sensitive
        to the magnitude of the imaginary part of the photonic frequency through
        the derivative coupling term and through the lindblad operator! 
        '''
    def RK4_NA(self):
        ci = 0+1j
        ### make a copy of the real part of H_total for each commutator!
        H_real = np.copy(np.real(self.H_total))
        self.D1 = np.copy(self.D_local)    
        self.k1 = np.copy(self.dt * self.DDot(H_real,self.D1) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D1) + 
                          self.dt * self.L_Diss(self.D1))# uncomment for dephasing + 
                          #self.dt * self.L_Deph(self.D1))
        
        ### Update H and D and get k2
        self.D2 = np.copy(self.D_local+self.k1/2.)
        self.k2 = np.copy(self.dt * self.DDot(H_real, self.D2) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D2) + 
                          self.dt * self.L_Diss(self.D2)) #uncomment for dephasing + 
                          #self.dt * self.L_Deph(self.D2))
        
        ### UPdate H and D and get k3
        self.D3 = np.copy(self.D_local+self.k2/2)
        self.k3 = np.copy(self.dt*self.DDot(H_real, self.D3) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D3) + 
                          self.dt * self.L_Diss(self.D3)) # uncomment for dephasing + 
                          #self.dt * self.L_Deph(self.D3)
        
        ### Update H and D and get K4
        self.D4 = np.copy(self.D_local+self.k3)
        self.k4 = np.copy(self.dt * self.DDot(H_real, self.D4) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D4) + 
                          self.dt * self.L_Diss(self.D4)) # uncomment for dephasing+ 
                          #self.dt * self.L_Deph(self.D4)
        
        self.D_local = np.copy(self.D_local + (1/6.)*(self.k1 + 2.*self.k2 + 2*self.k3 + self.k4))
        
        return 1
    
    '''  4th-order Runge-Kutta Algorithm for solving non-Hermitian 
         Lioville Equation in local basis! '''
    def RK4_NH_Lioville(self):
        ci = 0+1j
        ### make a copy of the real part of H_total for each commutator!
        ### This is the Hermitian part of the Hamiltonian
        H_H = np.copy(np.real(self.H_total))
        
        ### make a copy of the imaginary part of H_total for each 
        ### anti-commutator... this is the anti-Hermitian part of H
        H_A = np.copy(np.imag(self.H_total))
        
        ### Copy current density matrix
        self.D1 = np.copy(self.D_local)  
        
        ### First partial update
        self.k1 = np.copy(-ci*self.dt * self.comm(H_H, self.D1) -
                          ci*self.dt * self.anti_comm(H_A, self.D1)
                          - self.dt * self.V * self.comm(self.dc, self.D1))
        
        

        ### Update D and get k2
        self.D2 = np.copy(self.D_local+self.k1/2.)
        ### Second partial update
        self.k2 = np.copy(-ci*self.dt * self.comm(H_H, self.D2) -
                          ci*self.dt * self.anti_comm(H_A, self.D2)
                          - self.dt * self.V * self.comm(self.dc, self.D2))
        
        
        ### UPdate D and get k3
        self.D3 = np.copy(self.D_local+self.k2/2)
        self.k3 = np.copy(-ci*self.dt * self.comm(H_H, self.D3) -
                          ci*self.dt * self.anti_comm(H_A, self.D3)
                          - self.dt * self.V * self.comm(self.dc, self.D3))
        
        ### Update H and D and get K4
        self.D4 = np.copy(self.D_local+self.k3)
        self.k4 = np.copy(-ci*self.dt * self.comm(H_H, self.D4) -
                          ci*self.dt * self.anti_comm(H_A, self.D4)
                          - self.dt * self.V * self.comm(self.dc, self.D4))
        
        self.D_local = np.copy(self.D_local + (1/6.)*(self.k1 + 2.*self.k2 + 2*self.k3 + self.k4))
        
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


    ### Lindblad operator that models relaxation to the ground state
    def L_Diss(self, D):
        dim = len(D)
        LD = np.zeros_like(D)
        ### need |g><g|
        gm = np.copy(self.DM_Projector[:,:,0,0])
    
        for k in range(1,dim):
            gam = self.gamma_diss[k]
            km = np.copy(self.DM_Projector[:,:,k,k])
            ### first term 2*gam*<k|D|k>|g><g|
            t1 = np.copy(2 * gam * D[k,k] * gm)
            ### second term is |k><k|*D
            t2 = np.dot(km,D)
            ### third term is  D*|k><k|
            t3 = np.dot(D, km)
            LD = np.copy(LD + t1 - gam*t2 - gam*t3)
            
        return LD
    
    ''' Commutator method '''
    def comm(self, A, B):
        return (np.dot(A,B) - np.dot(B,A))
    
    ''' Anti-Commutator method '''
    def anti_comm(self, A, B):
        return (np.dot(A,B) + np.dot(B,A))
    
    ''' performs commutator portion of Liouville equation '''
    def DDot(self, H, D):
        ci = 0.+1j
        return -ci*(np.dot(H,D) - np.dot(D, H))

    def TrHD(self, H, D):
        N = len(H)
        HD = np.dot(H,D)
        som = 0
        for i in range(0,N):
            som = som + HD[i,i]
        return som

    ''' Non-Hermitian FSSH update '''
    def NH_FSSH(self):
        ### set up some quantities for surface hopping first!
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
        #  Hellman-Feynman force
        F_curr = self.Hellman_Feynman()
        
        ### get perturbation of force for Langevin dynamics
        rp_curr = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1)
    
        ### get acceleration
        ### Langevin acceleration
        a_curr = (F_curr + rp_curr) / self.M - self.gamma_nuc * self.V
        ### bbk update to velocity and position
        v_halftime = self.V + a_curr * self.dt / 2
        
        ### update R
        self.R = self.R + v_halftime * self.dt
        
        ### Hellman-Feynman force at updated geometry 
        F_fut = self.Hellman_Feynman()
        
        ### get new random force 
        rp_fut = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1)
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
        self.Derivative_Coupling()
        
        ### populations before updates
        for i in range(0,self.N_basis_states):
            pop_cur[i] = self.population_polariton[i]
        
        ### Update wavefunction
        self.RK4_NH_SE()
        
        ### update populations in polariton basis
        trace = 0.
        dot_trace = 0.
        for i in range(1,self.N_basis_states):
            c_i = self.C_polariton[i]
            p_i = np.real( np.dot(np.conj(c_i),c_i))
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
            ### if population in the active state is greater than zero, it 
            ### can go in the denmoniator
            if self.population_polariton[self.active_index]>0:
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
        if (self.active_index==2):
            
            #### is the smallest probability greater than the threshol?
            if gik[0]>thresh:
                self.active_index = 0
                switch=1
            elif (gik[0]+gik[1])>thresh:
                self.active_index = 1
                switch=1
        ### only one relevant probability      
        elif (self.active_index==1):
            if gik[0]>thresh:
                self.active_index = 0
                switch = 1
        else:
            switch = 0
                
        
        ### if we switched surfaces, we need to check some things about the
        ### momentum on the new surface... this comes from consideration of
        ### Eq. (7) and (8) on page 392 of Subotnik's Ann. Rev. Phys. Chem.
        ### on Surface Hopping
        ''' skip this for now! '''
        if 0: #switch:
            ### momentum on surface j (starting surface)
            Pj = self.V*self.M
            ### This number should always be positive!
            ### We will discard the imaginary part
            delta_V = np.real(self.Delta_V_jk[starting_act_idx,self.active_index])
            
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

    '''
    def FSSH_Update(self):
        ### use this to check if a switch occurs this iteration!
        switch=0
        sign = 1
        starting_act_idx = self.active_index
        ### allocate a few arrays we will need
        pop_fut = np.zeros(self.N_basis_states,dtype=complex)
        pop_dot = np.zeros(self.N_basis_states,dtype=complex)
        ### Will take the real part of the pop_dot to compute gik, 
        ### so it is a real vector!
        gik = np.zeros(self.N_basis_states)
    
        ### Get density matrix in local basis corresponding to the
        ### current active index (which refers to a surface in the polariton basis)
        
        ### Get transformation vectors at current R
        self.Transform_L_to_P()
        
        ### Get dH/dR in local basis... only the Electronic part 
        ### changes with R so we will only update/include the electronic term!
        self.R = self.R + self.dr
        self.H_e()
        
        Hf = np.copy(self.H_electronic) # + self.H_photonic + self.H_interaction)

        
        self.R = self.R - 2*self.dr
        self.H_e()
        
        Hb = np.copy(self.H_electronic) # + self.Hc_photonic + self.H_interaction)

        Hprime = np.copy((Hf-Hb)/(2*self.dr))
        
        ### go back to r_curr
        self.R = self.R + self.dr
        
        ### Get total Hamiltonian at current position
        self.H_e()
        ### This H_total will be used in RK4...
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        
        ### Get derivative coupling at current position... 
        ### Note transformation vecs are still at current value of self.R,
        ### they were not recomputed at any displaced values
        self.Derivative_Coupling(Hprime)
        
        ### compute populations in all states from active down to ground
        
        ### update populations in local basis
        for i in range(0,self.N_basis_states):
            self.population_local[i] = np.real(self.D_local[i, i])
            self.population_polariton[i] = np.real(self.D_polariton[i, i])
            
        ### update density matrix
        #self.RK4_NA()
        ### This will update all populations except D_local[0,0]
        self.RK4_NH()
        ### get updated density matrix in polariton basis so we can compute population
        ### in the future in the polariton basis!
        self.Transform_L_to_P()
        
        ### get future populations in polariton basis
        trace = 0
        for i in range(1,self.N_basis_states):
            pop_fut[i] = np.real(self.D_polariton[i, i])
            trace += np.real(self.D_polariton[i,i])
            
        pop_fut[0] = 1 - trace
        ### D_polariton[0,0] = D_local[0,0] 
        self.D_polariton[0,0] = 1 - trace
        self.D_local[0,0] = 1 - trace
        
        ###  review from here on 3/21/2020 to make sure FSSH is sensible 
        ### get change in populations in local basis to get hoppin
        for i in range(0,self.active_index+1):
            pop_dot[i] = np.real(pop_fut[i] - self.population_polariton[i])/self.dt
            ### don't divide by zero!!!
            if self.population_polariton[self.active_index]>0:
                g = np.real( pop_dot[i] / self.population_polariton[self.active_index] * self.dt)
            ### divide by something realllllly small instead!
            else:
                g = np.real( pop_dot[i] / 1e-13 * self.dt)
            if (g<0):
                g = 0
                
            ### Get cumulative probability
            if (i==0):
                gik[i] = g
            else:
                gik[i] = g + gik[i-1]
        
        ### decide if we want to hop to state k, if any
        thresh = np.random.random(1)
        ### This logic comes from the conditions in Eq.  (10) on page 4 
        ### from J. Chem. Phys. 138, 164106 (2013); doi: 10.1063/1.4801519
        if (self.active_index>1):
            for i in range(self.active_index-1,0,-1):
                if (gik[i]>=thresh[0] and gik[i-1]<thresh[0]):
                    ### Which direction does dc vector point?
                    if self.dc[self.active_index,i]<0:
                        sign = -1
                    else:
                        sign = 1
                    self.active_index = i
                    switch=1
        if (self.active_index==1):
            if (gik[0]>=thresh[0]):
                ### Which direction does dc vector point?
                if self.dc[self.active_index,0]<0:
                    sign = -1
                else:
                    sign = 1
                self.active_index = 0
                switch=1
                
        ### Now that we have considered switching active surfaces, compute D_act
        ### so that we can get the force on the active surface!
        D_act = np.outer(self.transformation_vecs_L_to_P[:,self.active_index], 
                         np.conj(self.transformation_vecs_L_to_P[:,self.active_index]))
            
        ### get perturbation of force for Langevin dynamics
        rp_curr = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1)
        
        ### get (negative of) Hellman-Feynamn force on active surface... H_prime hasn't changed
        ### from when we computed dc bc the nuclei have not moved yet!
        ### Note that the imaginary part of this trace should be zero anyway,
        ### we are just dropping the imaginary part so that F_curr will
        ### be a float data type not a complex data type!
        F_curr = np.real(self.TrHD(Hprime, D_act))

        ### get acceleration
        ### Langevin acceleration
        a_curr = (-1 * F_curr + rp_curr) / self.M - self.gamma_nuc * self.V
        ### bbk update to velocity and position
        v_halftime = self.V + a_curr * self.dt / 2
        
        ### update R
        self.R = self.R + v_halftime * self.dt
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        
        ### Get total Hamiltonian (maybe non-Hermitian) at new geometry!
        H_act = np.copy(self.H_total)
        
        ### get derivative of Hpl at r_fut
        
        ### Get H' again for FUTURE Hellman-Feynman force.
        ### Again it only depends on H_electronic bc H_photonic and H_interaction
        ### don't change with geometry
        self.R = self.R + self.dr
        self.H_e()
        Hf = np.copy(self.H_electronic)
        
        ### backward step
        self.R = self.R - 2*self.dr
        self.H_e()
        Hb = np.copy(self.H_electronic)
        ### derivative
        Hprime = np.copy((Hf-Hb)/(2*self.dr))
        
        ### return R to r_fut and make sure H_electronic 
        ### is what it should be... H_total should not have changed!
        self.R = self.R + self.dr
        self.H_e()
        ### get force from Hellman-Feynman theorem
        ### Note that the imaginary part of this trace should be zero anyway,
        ### we are just dropping the imaginary part so that F_fut will
        ### be a float data type not a complex data type!
        
        # note the full force is also real and agrees with Hellman-Feynman force #
        F_fut = np.real(self.TrHD(Hprime, D_act))
        
        ### get energy at r_fut on active polariton surface...
        ### Energy may be complex!
        self.Energy = self.TrHD(H_act, D_act)
        
        ### get random force 
        rp_fut = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1)
        ### get acceleration
        a_fut = (-1 * F_fut + rp_fut) / self.M - self.gamma_nuc * v_halftime
        ### get updated velocity
        ### vv update
        ###v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
        ### bbk update
        ### updated velocity assuming we are on the same surface
        self.V = (v_halftime + a_fut * self.dt/2) * (1/(1 + self.gamma_nuc * self.dt/2))
        if self.V>0:
            forward=1
        else:
            forward=-1
        ### if we switched surfaces, we need to check some things about the
        ### momentum on the new surface... this comes from consideration of
        ### Eq. (7) and (8) on page 392 of Subotnik's Ann. Rev. Phys. Chem.
        ### on Surface Hopping
        if switch:
            ### momentum on surface j (starting surface)
            Pj = self.V*self.M
            ### This number should always be positive!
            ### We will discard the imaginary part
            delta_V = np.real(self.Delta_V_jk[starting_act_idx,self.active_index])
            
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
        '''

    ''' Computes derivative coupling matrix in the LOCAL basis... this
        local basis corresponds to the potentially non-Hermitian total Hamiltonian...
        now depricated!  
    def Derivative_Coupling(self, H_prime):
        ### Compute derivative coupling in local basis
        for i in range(0, self.N_basis_states):
            D_ii = np.outer(self.transformation_vecs_L_to_P[:,i], 
                         np.conj(self.transformation_vecs_L_to_P[:,i]))
            Vii = self.TrHD(self.H_total, D_ii)
            
            for j in range(i, self.N_basis_states):
                if (i!=j):
                    D_jj = np.outer(self.transformation_vecs_L_to_P[:,j], 
                         np.conj(self.transformation_vecs_L_to_P[:,j]))
                    D_ij = np.outer(self.transformation_vecs_L_to_P[:,i], 
                         np.conj(self.transformation_vecs_L_to_P[:,j]))
                    
                    Vjj = self.TrHD(self.H_total, D_jj)
                    #D_ij = np.copy(self.DM_Projector[:,:,i,j])
                    cup = self.TrHD(H_prime, D_ij)
                    self.dc[i,j] = -1*cup/(Vjj-Vii)
                    self.dc[j,i] = -1*cup/(Vii-Vjj)
                    self.Delta_V_jk[i,j] = Vii-Vjj
                    self.Delta_V_jk[j,i] = Vjj-Vii
                    
        return 1
    '''
    ''' Computes Hellman-Feynman force '''
    def Hellman_Feynman(self):
        ### get transformation vector at current R
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        self.Transform_L_to_P()
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
        self.Transform_L_to_P()

        
        for j in range(0, self.N_basis_states): #self.N_basis_states):
            for k in range(0, self.N_basis_states): #self.N_basis_states):
                if j!=k:
                    Vjj = self.polariton_energies[j]
                    Vkk = self.polariton_energies[k]
                    cj = self.transformation_vecs_L_to_P[:,j]
                    ck = self.transformation_vecs_L_to_P[:,k]
                    tmp = np.dot(Hp, ck)
                    Fjk = -1*np.dot(np.conj(cj), tmp)
                    self.dc[j,k] = Fjk/(Vkk-Vjj)
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
    def Write_PES(self, pes_fn, pc_fn):
        
        rlist = np.linspace(-1.5, 1.5, 1000)
        
        ### Get PES of polaritonic system and write to file pes_fn
        pes_file = open(pes_fn, "w")
        pc_file = open(pc_fn, "w")
        
        for r in range(0,len(rlist)):
            wr_str = " "
            pc_str = " "
            self.R = rlist[r]
            wr_str = wr_str + str(self.R) + " "
            pc_str = pc_str + str(self.R) + " "
            self.H_e()
            self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
            self.Transform_L_to_P()
            v = self.transformation_vecs_L_to_P
            
            for i in range(0,self.N_basis_states):
                v_i = v[:,i]
                cv_i = np.conj(v_i)
                
                wr_str = wr_str + str(self.polariton_energies[i]) + " "
                
                ### loop over all photon indices in basis states
                pc = 0
                for j in range(0,self.N_basis_states):
                    if self.gamma_diss[j]>0:
                        pc = pc + np.real(cv_i[j] * v_i[j])
                        
                pc_str = pc_str + str(pc) + " "
        
            wr_str = wr_str + "\n"
            pc_str = pc_str + "\n"        
            pes_file.write(wr_str)
            pc_file.write(pc_str)
        
        ### Close PES file
        pes_file.close()
        pc_file.close()
        
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
        rlist = np.linspace(-0.75,  -0.50, 500)
        
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
    

