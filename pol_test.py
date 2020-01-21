#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:26:29 2020

@author: jay
"""

from polaritonic import polaritonic


options = {
        'Number_of_Photons': 1,
        'Photon_Energys': [2.5/27.211],
        'Coupling_Strengths': [0.03/27.211], 
        'Initial_Position': -0.60,
        'Initial_Velocity': -3.00e-5,
        ### specify initial state as a human would, not a computer...
        ### i.e. 1 is the ground state... it will be shifted down by -1 so
        ### that it makes sense to the python index convention
        'Initial_Local_State': 3
        
        }

polt = polaritonic(options)
#print(polt.transformation_vecs_L_to_P)
print("Local")
print(polt.H_total)
print("Polariton")
print(polt.H_polariton)

print(polt.DM_Projector[:,:,2])

#res = polt.action(1,'t3', 2)
#print(res)
#action(self, state_indx, term, photon_indx):