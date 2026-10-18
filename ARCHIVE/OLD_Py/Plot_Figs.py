#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:31:21 2020

@author: foleyj10
"""

### Import all libraries and define various parameters here!
import numpy as np

from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14


fp1 = "Data/lossy_SMA_gam_50meV_photon_contribution.txt"
fp2 = "Data/lossy_SMA_gam_50meV_pes.txt"
fp3 = "Data/SMA_gam_1meV_nuc_traj.txt" 

PC = np.loadtxt(fp1)
PES = np.loadtxt(fp2)
tr = np.loadtxt(fp3)

#plt.plot(p1[:,0], p1[:,6], label='1 pp22')
#plt.plot(p1[:,0], p1[:,7], label='1 pp33')
#plt.plot(tr[:,0], (2*tr[:,2]/0.18258794364057063-1.4), label='trajectory')

#plt.legend()
#plt.show()





fig, ax = plt.subplots()
cm = plt.cm.get_cmap('rainbow')
im = ax.scatter(PES[:,0], 27.211*PES[:,1], c=PC[:,1], cmap=cm, s=4) # rlist, 27.211*PPES[:,1], c=LP_p, cmap=cm )
im = ax.scatter(PES[:,0], 27.211*PES[:,2], c=PC[:,2], cmap=cm, s=4)
im = ax.scatter(PES[:,0], 27.211*PES[:,3], c=PC[:,3], cmap=cm, s=4)
im = ax.scatter(PES[:,0], 27.211*PES[:,4], c=PC[:,4], cmap=cm, s=4)
im = ax.plot(tr[:,1], tr[:,2]*27.211, 'black', label='Trajectory')
#ax.legend()
#cbar = fig.colorbar(im, ticks=[0.1, 0.5, 0.9])
#cbar.ax.set_yticklabels(['excitonic', 'polaritonic', 'photonic'])
#plt.xlim(-1.,1.)
plt.ylim(1.,6.5)
plt.xlabel("R (a.u.)")
plt.ylabel("Energy (eV)")
#plt.savefig("Fig_Surface.eps")
plt.show()