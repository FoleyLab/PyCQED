#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:58:08 2019

@author: foleyj10
"""
import numpy as np
from numpy import linalg as LA
import math
from numpy.polynomial.hermite import *


class polaritonic:
    
    ### initializer
    #def __init__(self, mode, inputfile):
    def __init__(self, args):
        
        '''
        self.H_electronic = np.zeros((dim,dim))
        self.H_photonic = np.zeros((dim,dim))
        self.H_interaction = np.zeros((dim,dim))
        self.H_total = np.zeros((dim,dim))
        self.H_polariton = np.zeros((dim,dim))
        '''
        ### Get basic options from input dictionary, 
        ### get dimensionality and build appropriate local basis
        self.parse_options(args)
        ### allocate space for the Hamiltonian matrices
        ### local basis first
        self.H_electronic = np.zeros((self.N_basis_states,self.N_basis_states))
        self.H_photonic = np.zeros((self.N_basis_states,self.N_basis_states))
        self.H_interaction = np.zeros((self.N_basis_states,self.N_basis_states))
        self.H_total = np.zeros((self.N_basis_states,self.N_basis_states))
        ### polaritonic basis Hamiltonian
        self.H_polariton = np.zeros((self.N_basis_states,self.N_basis_states))
        
        ### Density matrix  arrays
        self.D_local = np.zeros((self.N_basis_states, self.N_basis_states),dtype=complex)
        self.D_polariton = np.zeros((self.N_basis_states, self.N_basis_states),dtype=complex)
        
        ### Population arrays 
        self.population_local = np.zeros(self.N_basis_states)
        self.population_polariton = np.zeros(self.N_basis_states)
        
        self.transformation_vecs_L_to_P = np.zeros((self.N_basis_states, self.N_basis_states))
        self.polariton_energies = np.zeros(self.N_basis_states)
        
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
        
        ### derivative coupling
        self.dc = np.zeros((self.N_basis_states,self.N_basis_states))
        
        self.Transform_L_to_P()
        
        ### RK4 Variables
        self.k1 = np.zeros_like(self.D_local)
        self.k2 = np.zeros_like(self.D_local)
        self.k3 = np.zeros_like(self.D_local)
        self.k4 = np.zeros_like(self.D_local)
        self.D1 = np.zeros_like(self.D_local)
        self.D2 = np.zeros_like(self.D_local)
        self.D3 = np.zeros_like(self.D_local)
        self.D4 = np.zeros_like(self.D_local)
        
        self.DM_Bas = np.identity(self.N_basis_states,dtype=complex)
        
        self.DM_Projector = np.zeros((self.N_basis_states, self.N_basis_states, self.N_basis_states,self.N_basis_states),dtype=complex)
        
        for i in range(0, self.N_basis_states):
            for j in range(0, self.N_basis_states):
                self.DM_Projector[:,:,i,j] = np.outer(self.DM_Bas[i,:], np.conj(self.DM_Bas[j,:]))
                
        self.Energy = self.TrHD(self.H_total, self.D_local)
            
        
        
        
    ''' Next two methods used to build the local basis '''   
    def printTheArray(self):  
        
        for i in range(0, self.NPhoton+1):  
            #print(arr[i], end = " ")
            self.local_basis[self.basis_cnt,i] = self.temp_basis[i]
            
        self.basis_cnt = self.basis_cnt + 1
        
        return 1
    
    # Function to generate all binary strings
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
    
    def H_e(self):
        
        ### get self.Eg and self.Ee
        self.E_of_R()
        
        for i in range(0,self.N_basis_states):
            if self.local_basis[i,0] == 0:
                self.H_electronic[i,i] = self.Eg
            elif self.local_basis[i,0] == 1:
                self.H_electronic[i,i] = self.Ee
                
        return 1
    
    def H_p(self):
        
        
        for i in range(0,self.N_basis_states):
            val = 0
            for j in range(1,self.NPhoton+1):
                if self.local_basis[i,j] == 0:
                    val = val + 0.5 * self.omc[j-1]
                elif self.local_basis[i,j] == 1:
                    val = val + 1.5 * self.omc[j-1]
            self.H_photonic[i,i] = val
            
        return 1
    
    def H_ep(self):
        
        for i in range(0,self.N_basis_states):
            
            for j in range(0,self.N_basis_states):
                
                val = 0
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
    
    ### function that operates on basis state with string of 2nd
    ### quantized operators and returns the resulting basis state
    ### essentially we have 4 different strings to consider 
    ### t1 = b_i^+ a_e^+ a_g
    ### t2 = b_i^+ a_g^+ a_e
    ### t3 = b_i   a_e^+ a_g
    ### t4 = b_i   a_g^+ a_e
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
    
    def Transform_P_to_L(self):
        ### now do transformation for density matrix from local to polariton basis
        dt0 = np.dot(self.transformation_vecs_L_to_P, self.D_polariton)
        self.D_local = np.dot(dt0, LA.inv(self.transformation_vecs_L_to_P))
        ### return Hpl and Dpl
        return 1
  
    def RK4_NA(self): #H, D, h, gamma, gam_deph, V, dc):
        ci = 0+1j
        ### Get k1
        self.D1 = np.copy(self.D_local)    
        self.k1 = np.copy(self.dt * self.DDot(self.H_total,self.D1) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D1) + 
                          self.dt * self.L_Diss(self.D1))# uncomment for dephasing + 
                          #self.dt * self.L_Deph(self.D1))
        
        ### Update H and D and get k2
        self.D2 = np.copy(self.D_local+self.k1/2.)
        self.k2 = np.copy(self.dt * self.DDot(self.H_total, self.D2) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D2) + 
                          self.dt * self.L_Diss(self.D2)) #uncomment for dephasing + 
                          #self.dt * self.L_Deph(self.D2))
        
        ### UPdate H and D and get k3
        self.D3 = np.copy(self.D_local+self.k2/2)
        self.k3 = np.copy(self.dt*self.DDot(self.H_total, self.D3) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D3) + 
                          self.dt * self.L_Diss(self.D3)) # uncomment for dephasing + 
                          #self.dt * self.L_Deph(self.D3)
        
        ### Update H and D and get K4
        self.D4 = np.copy(self.D_local+self.k3)
        self.k4 = np.copy(self.dt * self.DDot(self.H_total, self.D4) - 
                          ci * self.dt * self.V * self.DDot(self.dc, self.D4) + 
                          self.dt * self.L_Diss(self.D4)) # uncomment for dephasing+ 
                          #self.dt * self.L_Deph(self.D4)
        
        self.D_local = np.copy(self.D_local + (1/6.)*(self.k1 + 2.*self.k2 + 2*self.k3 + self.k4))
        
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


    def DDot(self, H, D):
        ci = 0.+1j
        return -ci*(np.dot(H,D) - np.dot(D, H))



    def TrHD(self, H, D):
        N = len(H)
        HD = np.dot(H,D)
        som = 0
        for i in range(0,N):
            som = som + HD[i,i]
        return np.real(som)
    

    def FSSH_Update(self): 
        ### allocate a few arrays we will need
        pop_fut = np.zeros(self.N_basis_states)
        pop_dot = np.zeros(self.N_basis_states)
        gik = np.zeros(self.N_basis_states)
    
        ### Get density matrix in local basis corresponding to the
        ### current active index (which refers to a surface in the polariton basis)
        
        ### Get transformation vectors at current R
        self.Transform_L_to_P()
        ### Get corresponding density matrix in local basis
        D_act = np.outer(self.transformation_vecs_L_to_P[:,self.active_index], 
                         np.conj(self.transformation_vecs_L_to_P[:,self.active_index]))
        
        ### Get dH/dR in local basis
        self.R = self.R + self.dr
        self.H_e()
        Hf = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)

        
        self.R = self.R - 2*self.dr
        self.H_e()
        Hb = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)

        Hprime = np.copy((Hf-Hb)/(2*self.dr))
        
        ### go back to r_curr
        self.R = self.R + self.dr
        
        ### Get total Hamiltonian at current position
        self.H_e()
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
        self.RK4_NA() 
        ### get updated density matrix in polariton basis as well
        self.Transform_L_to_P()
        
        ### get future populations in local basis
        for i in range(0,self.N_basis_states):
            pop_fut[i] = np.real(self.D_polariton[i, i])
             
        
        ### get change in populations in local basis to get hoppin
        for i in range(0,self.active_index+1):
            pop_dot[i] = np.real(pop_fut[i] - self.population_polariton[i])/self.dt
            g = np.real( pop_dot[i] / self.population_polariton[self.active_index] * self.dt)
            if (g<0):
                g = 0
            
            ### Get cumulative probability
            if (i==0):
                gik[i] = g
            else:
                gik[i] = g + gik[i-1]
        #print(gik)
        
        ### decide if we want to hop to state k, if any
        thresh = np.random.random(1)
        #print(gik[0],gik[1],gik[2],gik[3],thresh[0])
        #print(gik[0],gik[1],gik[2],gik[3],thresh[0])
    
        if (self.active_index>1):
            for i in range(self.active_index-1,0,-1):
                if (gik[i]>=thresh[0] and gik[i-1]<thresh[0]):
                    #print("hopping from state",self.active_index,"to ",i)
                    self.active_index = i
        if (self.active_index==1):
            if (gik[0]>=thresh[0]):
                #print("hopping from state",self.active_index,"to ",0)
                self.active_index = 0
    
        
            
        ### use parameters set above to get initial perturbation of force for Langevin dynamics
        rp_curr = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1)
        
        ### get force on active surface
        F_curr = self.TrHD(Hprime, D_act)
        #print("F_curr",F_curr)

        ### get acceleration
        ### Langevin acceleration
        a_curr = (-1 * F_curr + rp_curr) / self.M - self.gamma_nuc * self.V
        ### bbk update to velocity and position
        v_halftime = self.V + a_curr * self.dt / 2
        
        ### update R
        self.R = self.R + v_halftime * self.dt
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        
        H_act = np.copy(self.H_total)
        
        ### get derivative of Hpl at r_fut
        
        ### forward step first
        self.R = self.R + self.dr
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        Hf = np.copy(self.H_total)
        
        ### backward step
        self.R = self.R - 2*self.dr
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        Hb = np.copy(self.H_total)
        ### derivative
        Hprime = np.copy((Hf-Hb)/(2*self.dr))
        
        ### return R to r_curr
        self.R = self.R + self.dr
        self.H_e()
        self.H_total = np.copy(self.H_electronic + self.H_photonic + self.H_interaction)
        ### get force from Hellman-Feynman theorem
        F_fut = self.TrHD(Hprime, D_act)
        ### get energy at r_fut on active polariton surface
        self.Energy = self.TrHD(H_act, D_act)
        
        ### get random force 
        rp_fut = np.sqrt(2 * self.T * self.gamma_nuc * self.M / self.dt) * np.random.normal(0,1)
        ### get acceleration
        a_fut = (-1 * F_fut + rp_fut) / self.M - self.gamma_nuc * v_halftime
        ### get updated velocity
        ### vv update
        ###v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
        ### bbk update
        self.V = (v_halftime + a_fut * self.dt/2) * (1/(1 + self.gamma_nuc * self.dt/2))
        return 1

    
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
                    self.dc[i,j] = cup/(Vjj-Vii)
                    self.dc[j,i] = cup/(Vii-Vjj)
                    
        return 1

'''

    
    
    def Hopping_Rate(dc, Dl, v, dt, idx_j, idx_k):
        arg = 2 * v * np.real(dc[idx_j, idx_k] * Dl[idx_k, idx_j]) * dt
        
        if (arg<0):
            rate = 0
            
        else:
            rate = arg / np.real(Dl[idx_j, idx_j])
    
        #print(dc[idx_j, idx_k], np.real(Dl[idx_k, idx_j]), np.real(Dl[idx_j, idx_j]), v[0], rate)
        return rate
        
    ### Transform density matrix from polariton to local basis
    ### at a given R... return (off-diagonal) local Hamiltonian
    ### and transformation vecs, also

    def Transform_P_to_L(r, Dp, Hp, Hep):
        He = np.zeros((4,4))
        He = H_e(He, r)
        Htot = He + Hp + Hep
        ### get eigenvalues/vectors of total Hamiltonian at this R
        vals, vecs = LA.eig(Htot)
        ### sort the eigenvectors
        idx = vals.argsort()[::1]
        vals = vals[idx]
        v = vecs[:,idx]
        ### now do transformation for density matrix from local to polariton basis
        dt0 = np.dot(v, Dp)
        Dl = np.dot(dt0,LA.inv(v))
        ### return Hpl and Dpl
        return [Htot, Dl, v]

    

def HF_Force(Hp, Hep, He, r, dr, D):
    H0 = Hp + Hep
    ### get forward Hamiltonian
    #He = H_e(He, r+dr)
    #def Transform_L_to_P(r, Dl, Hp, Hep):
    [Hf, Df, v] = Transform_L_to_P(r+dr, D, Hp, Hep)
    #print(Hf[0,0], Hf[1,1], Hf[2,2], Hf[3,3])
    [Hb, Db, v] = Transform_L_to_P(r-dr, D, Hp, Hep)
    
    ### return Hpl and Dpl
    #return [Hpl, Dpl, v]
    
    
    #Hf = np.copy(H0+He)
    #### get backwards Hamilonian
    #He = H_e(He, r-dr)
    #Hb = np.copy(H0+He)
    Hprime = (Hf - Hb)/(2*dr)
    #[Ht, Dl, vec] = Transform_P_to_L(r, D, Hp, Hep)
    hf_force = TrHD(Hprime, D)
    #ef Transform_P_to_L(r, Dp, Hp, Hep):
    return hf_force

def Dp_Force(Hp, Hep, He, r, dr, D):
    [Ht, Df, vec] = Transform_P_to_L(r+dr, D, Hp, Hep)
    [Ht, Db, vec] = Transform_P_to_L(r-dr, D, Hp, Hep)
    He = H_e(He, r)
    Htot = Hp + He + Hep
    Dprime = (Df - Db)/(2*dr)
    residual_force = TrHD(Htot, Dprime)
    return residual_force
    
'''  

''' Quantum dynamics first '''
### should create a function that takes a wavefunction in vector
### format and computes a density matrix
'''

### Lindblad operator that models dephasing
def L_Deph(D, gam):
    dim = len(D)
    LD = np.zeros_like(D)
    
    for k in range(1,dim):
        bra_k = CreateBas(dim, k)
        km = Form_Rho(bra_k, bra_k)
        
        ### first term 2*gam*<k|D|k>|k><k|
        t1 = 2*gam*D[k][k]*km
        ### second term is |k><k|*D
        t2 = np.dot(km,D)
        ### third term is  D*|k><k|
        t3 = np.dot(D, km)
        LD = LD + t1 - gam*t2 - gam*t3
        
    return LD



### evaluate the hopping rate from state j -> k


        
            
            
    
    
        
'''
'''
def Erhenfest(r_curr, v_curr, mass, D, Hp, Hep, Hel, gamma, gam_deph, dr, dt):

    ### Electronic part 1 ###
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
    
    ###  Nuclear part 1 ###
    ### Get force from finite-difference gradient
    F = (Eb - Ef)/(2*dr)
    ### Get acceleration from force
    a_curr = F / mass
    ### now get r in the future... r_fut
    r_fut = r_curr + v_curr*dt + 1/2 * a_curr*dt**2
    
    ###  Electronic part 2 ###
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
    
    ###Nuclear part 2###
    ### Get force from finite-difference gradient
    F = (Eb - Ef)/(2*dr)
    ### Get acceleration from force
    a_fut = F / mass
    v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
    ### return a list with new position and velocity
    return [r_fut, v_fut, D]
'''
'''
def VelocityVerlet(spline,  mass, r_curr, v_curr, dt):
    ### compute acceleration ... first we need force
    F_curr = -1 * spline(r_curr)
    ### now get acceleration from a = F/m
    a_curr = F_curr / mass
    ### now get r in the future... r_fut
    r_fut = r_curr + v_curr*dt + 1/2 * a_curr*dt**2
    ### now get the future force: -d/dr E(r_fut)
    F_fut = -1 * spline(r_fut)
    ### now that I have new Force, compute new acceleration
    a_fut = F_fut / mass
    ### now that I have acceleration in the future, let me calculate velocity 
    ### in the future
    v_fut = v_curr + 1/2 * (a_curr + a_fut)*dt
    ### return a list with new position and velocity
    return [r_fut, v_fut]

def dfdx(ft, xt):
    dx = xt[1]-xt[0]
    ftp = np.zeros_like(ft)
    for i in range(0,len(ft)):
        if (i<(len(ft)-1)):
            rise = ft[i+1]-ft[i]
            ftp[i] = rise/dx
        else:
            rise = ft[i]-ft[i-1]
            ftp[i] = rise/dx
    
    return ftp

def df2dx2(ft, xt):
    dx = xt[1]-xt[0]
    ftp = np.zeros(len(ft),dtype=complex)
    N = len(ft)
    ftp[0] = 0+0j
    ftp[N-1]= 0+0j
    for i in range(1,len(ft)-1):
        ftp[i] = (ft[i+1] - 2*ft[i] + ft[i-1])/(dx**2)
    return ftp

### Kinetic energy operator on wavefunction
def TPhi(ft, xt, m):
    ftpp = df2dx2(ft, xt)
    return -1/(2*m)*ftpp

### Get action of Hamiltonian on Phi and multiply by negative i... this
### gives time-derivative of Phi
def Phi_Dot(ft, xt, m, vx):
    ci = 0+1j
    return -1*ci*(TPhi(ft, xt, m) + vx*ft)

### Kinetic energy squared operator
def T2Phi(ft, xt, m):
    ftp = dfdx(ft, xt)
    ftpp = dfdx(ftp, xt)
    ftppp = dfdx(ftpp, xt)
    ftpppp = dfdx(ftppp, xt)
    return 1/(4*m*m)*ftpppp

### returns the kinetic energy functional of a trial
### wavefunction (called ft within the function)
def T_Functional(ft, xt, m):
    tphi = TPhi(ft, xt, m)
    dx = xt[1] - xt[0]
    num = 0
    denom = 0
    for i in range(0, len(ft)):
        num = num + ft[i]*tphi[i]*dx
        denom = denom + ft[i]*ft[i]*dx

    return num/denom

def T2_Functional(ft, xt, m):
    t2phi = T2Phi(ft, xt, m)
    dx = xt[1] - xt[0]
    num = 0
    denom = 0
    for i in range(0, len(ft)):
        num = num + ft[i]*t2phi[i]*dx
        denom = denom + ft[i]*ft[i]*dx

    return num/denom


'''
'''

pi = np.pi
hbar = 1
def HO_En(K, m, n):
    return np.sqrt(K/m) * (n + 1/2)


def HO_Func(K, m,  n, r, r0):
    w = np.sqrt(K/m)
    psi = []
    herm_coeff = []
    
    for i in range(n):
        herm_coeff.append(0)
        
    herm_coeff.append(1)
    
    for x in r:
        psi.append(math.exp(-m*w*(x-r0)**2/(2*hbar)) * hermval((m*w/hbar)**0.5 * (x-r0), herm_coeff))
        
    # normalization factor for the wavefunction:
    psi = np.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*hbar))**0.25)
    
    return psi

'''

'''
r2 = np.linspace(-1,0,500)
vx_g0 = 1/2 * k_g0 * (r2-rmin_g0)**2
psi_g0 = HO_Func(k_g0, M, 0, r2, rmin_g0)

vx_phi2 = 1/2 * k_phi2 * (r2-rmin_phi2)**2
psi_phi2 = HO_Func(k_phi2, M, 0, r2, rmin_phi2)


def Fourier(x, fx, n, k, m, r0):
    tfn = HO_Func(k, m, n, x, r0)
    som = 0
    dx = x[1]-x[0]
    for i in range(0,len(x)):
        som = som + fx[i] * tfn[i] * dx
    return som


    
'''