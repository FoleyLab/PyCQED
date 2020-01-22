#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:26:29 2020

@author: jay
"""

from polaritonic import polaritonic
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

au_to_ps = 2.4188e-17 * 1e12
options = {
        'Number_of_Photons': 1,
        'Photon_Energys': [2.45/27.211],
        'Coupling_Strengths': [0.02/27.211], 
        'Photon_Lifetimes': [0.1/27211.],
        'Initial_Position': -0.6940536701380835,
        'Initial_Velocity': 3.0711164130420224e-06,
        'Mass': 1009883,
        ### temperature in a.u.
        'Temperature': 0.00095,
        ### friction in a.u.
        'Photon_Lifetime': 0.00001,
        'Friction': 0.000011,
        ### specify initial state as a human would, not a computer...
        ### i.e. 1 is the ground state... it will be shifted down by -1 so
        ### that it makes sense to the python index convention
        'Initial_Local_State': 3
        
        }

polt = polaritonic(options)
print(polt.dr)
print(polt.dt)


rlist = np.linspace(-1, 1, 200)
PES = np.zeros((len(rlist),polt.N_basis_states))

for r in range(0,len(rlist)):
    polt.R = rlist[r]
    polt.H_e()
    polt.H_total = np.copy(polt.H_electronic + polt.H_photonic + polt.H_interaction)
    polt.Transform_L_to_P()
    for i in range(0,polt.N_basis_states):
        PES[r,i] = polt.polariton_energies[i]


polt.R = -0.6940536701380835
print(polt.R,polt.V,polt.Energy)

N_time = 100000
sim_time = np.zeros(N_time)
r_of_t = np.zeros(N_time)
e_of_t = np.zeros(N_time)
start = time.time()
for i in range(0,N_time):
    #### Update nuclear coordinate first
    sim_time[i] = i*polt.dt
    r_of_t[i] = polt.R
    e_of_t[i] = polt.Energy
    polt.FSSH_Update()



fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-1, 1), ylim=(0, 10))
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
    thisx = r_of_t[i]
    thisy = 27.211*e_of_t[i]     
    line.set_data(thisx, thisy)
    linee1.set_data(rlist, 27.211*PES[:,3])
    linep2.set_data(rlist, 27.211*PES[:,2])
    linep1.set_data(rlist, 27.211*PES[:,1])
    lineg0.set_data(rlist, 27.211*PES[:,0])
    time_text.set_text(time_template % (i*polt.dt * au_to_ps))
    return line, linee1, linep2, linep1, lineg0, time_text

### driver for the animation... argument fig refers to the figure object,
### argument animate refers to the animate function above, the range argument 
### defines which values of the time counter will be used 
### (here the timestep dt is really small, so we only plot every 100 timesteps
### to make the animation reasonable
ani = animation.FuncAnimation(fig, animate, range(1, len(r_of_t),100),
                             interval=polt.dt, blit=True, init_func=init)
plt.show()
