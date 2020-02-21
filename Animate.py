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
from matplotlib import animation
from matplotlib import cm
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
ci = 0+1j

### go through and read nuclear dynamics trajectories from each data file

print(" What is the prefix for your data files?")
prefix = input()

r_of_t = np.zeros((2,8000))
t_of_t = np.zeros(8000)
e_of_t = np.zeros((2,8000))
for i in range(1,2):
    #file_path = "Data/" + prefix + str(i) + "_nuc_traj.txt"
    file_path = "Data/test_gam_0.1_meV_nuc_traj.txt"
    print(file_path)
    a = np.loadtxt(file_path)
    #print(len(a[:,0]))
    
    t_of_t[:] = a[:,0]
    r_of_t[i-1,:] = a[:,1]
    e_of_t[i-1,:] = a[:,2]

dt = 0.12

rlist = np.zeros(500)
PPES = np.zeros((500,4))

file_path = "Data/" + prefix + "_pes.txt"
b = np.loadtxt(file_path)

rlist[:] = b[:,0]
PPES[:,0] = b[:,1]
PPES[:,1] = b[:,2]
PPES[:,2] = b[:,3]
PPES[:,3] = b[:,4]
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-2, 2), ylim=(0.5, 6.5))
ax.grid()

### these are the different curves/points that will be plotted
line, = ax.plot([], [], 'o', lw=2)
linee1, = ax.plot([], [], lw=2)
linep2, = ax.plot([], [], lw=2)
linep1, = ax.plot([], [], lw=2)
lineg0, = ax.plot([], [], lw=2)

time_template = 'time = %.1e ps'
time_text = ax.text(0.05, 0.02, '', transform=ax.transAxes)

### initializes the figure object
def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


### function that defines the points/curves in the animation given a current
### value of the time counter, i
def animate(i):
    thisx = r_of_t[:,i]
    thisy = 27.211*e_of_t[:,i]     
    line.set_data(thisx, thisy)
    linee1.set_data(rlist, 27.211*PPES[:,3])
    linep2.set_data(rlist, 27.211*PPES[:,2])
    linep1.set_data(rlist, 27.211*PPES[:,1])
    lineg0.set_data(rlist, 27.211*PPES[:,0])
    time_text.set_text(time_template % (i*dt))
    return line, linee1, linep2, linep1, lineg0, time_text

### driver for the animation... argument fig refers to the figure object,
### argument animate refers to the animate function above, the range argument 
### defines which values of the time counter will be used 
### (here the timestep dt is really small, so we only plot every 100 timesteps
### to make the animation reasonable
ani = animation.FuncAnimation(fig, animate, range(1, len(t_of_t), 1),
                             interval=dt, blit=True, init_func=init)

#ani.save('test.html')
#ani.save('traj_gam_5p0.gif', writer='imagemagick')
#ani.save('test.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
plt.show()
'''

fp1 = "Data/test_photon_contribution.txt"
fp2 = "Data/test_pes.txt"
#fp3 = "Data/ntest1_nuc_traj.txt" 


#rt = np.zeros(80)
#rho1 = np.zeros((80,4))
#rho2 = np.zeros((80,4))
p1 = np.loadtxt(fp1)
p2 = np.loadtxt(fp2)



fig, ax = plt.subplots()
cm = plt.cm.get_cmap('rainbow')
im = ax.scatter(p2[:,0], 27.211*p2[:,1], c=p1[:,1], cmap=cm, s=4) 
im = ax.scatter(p2[:,0], 27.211*p2[:,2], c=p1[:,2], cmap=cm, s=4)
im = ax.scatter(p2[:,0], 27.211*p2[:,3], c=p1[:,3], cmap=cm, s=4)
im = ax.scatter(p2[:,0], 27.211*p2[:,4], c=p1[:,4], cmap=cm, s=4)
cbar = fig.colorbar(im, ticks=[0.1, 0.5, 0.9])
cbar.ax.set_yticklabels(['excitonic', 'polaritonic', 'photonic'])
plt.xlim(-1.,1.)
#plt.ylim(3.5,6.)
plt.xlabel("R (a.u.)")
plt.ylabel("Energy (eV)")
#plt.savefig(filename)
plt.show()

'''



