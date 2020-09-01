#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:00:26 2020

@author: foleyj10
"""
import numpy as np
from matplotlib import pyplot as plt

#file_path = "Data/hg_0.1_run1000_electronic.txt"
#a = np.loadtxt(file_path)
file_path2 = "Data/hg_0.1_run7_electronic.txt"
a = np.loadtxt(file_path2)


plt.plot(a[:,0], a[:,1], 'red', label='g0')
plt.plot(a[:,0], a[:,2], 'b--', label='g0 pop')
plt.plot(a[:,0], a[:,3], 'blue', label='LP')
plt.plot(a[:,0], a[:,4], 'r--', label='LP pop')
plt.plot(a[:,0], a[:,5], 'black', label='UP')
plt.plot(a[:,0], a[:,6], 'g--', label='UP pop')
plt.plot(a[:,0], a[:,3]+a[:,5], 'purple')
plt.xlim(0,50000)
plt.legend()
plt.show()
