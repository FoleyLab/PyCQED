#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:26:29 2020

@author: jay
"""

from polaritonic import polaritonic


options = {
        'Number_of_Photons': 1,
        'Photon_Energys': [2.45/27.211],
        'Coupling_Strengths': [0.02/27.211], 
        'Initial_Position': -0.678,
        'Initial_Velocity': -3.00e-5
        
        }

polt = polaritonic(options)
print(polt.local_basis)

#res = polt.action(1,'t3', 2)
#print(res)
#action(self, state_indx, term, photon_indx):