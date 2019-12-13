#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:24:42 2019

@author: jay
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

### create an array of 20 bond lengths spanning 0.5 - 3.5 angstroms
### but store in atomic units of length... note 1 angstrom = 1.88973 a.u. of length
r_array = np.linspace(0.561,3.087,10)*1.88973
print(r_array)

### fill in this array with your actual energy values... there should be 20 values in total!!!
E_array = [-109.0350050004,-112.3137052599, -112.6994904875, -112.5943503269, -112.4432190905, -112.3517396124, 
          -112.3147098978, -112.2973399066, -112.0120671242, -111.9948057317]


# TO see the plot of the PES, uncomment the following lines
#plt.title('Potential Energy Surface')
#plt.xlabel('Bond Length (a.u.)')
#plt.ylabel('Potential Energy (a.u.)')
#plt.plot(r_array, E_array, 'red')
#plt.show()

### use cubic spline interpolation
order = 3
### form the interpolator object for the data
sE = InterpolatedUnivariateSpline(r_array, E_array, k=order)
### form a much finer grid
r_fine = np.linspace(1.06,5.0,200)
### compute the interpolated/extrapolated values for E on this grid
E_fine = sE(r_fine)
### plot the interpolated data
#plt.title('Potential Energy Surface Using Spline Function')
#plt.xlabel('Bond Length (a.u.)')
#plt.ylabel('Potential Energy (a.u.)')
#plt.plot(r_fine, E_fine, 'blue')
#plt.show()

### minimum energy
minE = min(E_fine)

### take the derivative of potential
fE = sE.derivative()
### force is the negative of the derivative
F_fine = -1*fE(r_fine)

### plot the forces
#plt.title('Bond Length vs. Force')
#plt.xlabel('Bond Length (a.u.)')
#plt.ylabel('Force (a.u.)')
#plt.plot(r_fine, np.abs(F_fine), 'black')
#plt.xlim(1,5)
#plt.show()

### Find index of the PES where it has its global minimum
r_eq_idx = np.argmin(E_fine)
### find the value of the separation corresponding to that index
r_eq = r_fine[r_eq_idx]
### print equilibrium bond-length in atomic units and in angstroms
print("Equilibrium bond length is ",r_eq," atomic units")
print("Equilibrium bond length is ",r_eq*0.529," Angstroms")

### get second derivative of potential energy curve... recall that we fit a spline to
### to the first derivative already and called that spline function fE.
cE = fE.derivative()

### evaluate the second derivative at r_eq to get k
k = cE(r_eq)

### define reduced mass of CO as m_C * m_O /(m_C + m_O) where mass is in atomic units (electron mass = 1)
mu = 13625.

### define "ground-state" velocity:
v = np.sqrt( np.sqrt(k/mu)/(2*mu))

### get random position and velocity for CO within a reasonable range
r_init = np.random.uniform(0.75*r_eq,2*r_eq)
v_init = np.random.uniform(-2*v,2*v)

### print initial position and velocity
print("Initial separation is ",r_init, "atomic units")
print("Initial velocity is   ",v_init, "atomic units")
### establish time-step for integration to be 0.02 atomic units... this is about 0.01 femtoseconds
dt = 0.5

### get force on particle 
F_init = -1*fE(r_init)

### parameters for Langevin Equation: temperature and drag
### note that in atomic units, the Boltzmann constant is unity
temperature = 0.1 # boiling point of CO in atomic units
gamma = 0.0001 #;

### use parameters set above to get initial perturbation of force for Langevin dynamics
rp_init = np.sqrt(2*temperature*gamma*mu/dt)*np.random.normal(0,1)

def BBK(r_curr, v_init, rp_curr, gamma_val, temperature_val, mu, f_interp, dt):
    
    ### get acceleration at current time: (force + current perturbation on force)/mass - drag
    a_curr = (-1*f_interp(r_curr) + rp_curr)/mu - gamma_val*v_init
### THE ABOVE IS EQUATION 6 IN THE ASSIGNMENT DAS POSTED
    
    ### update velocity for half time step, for such a small time step can approximate dv = adt
### THINK ABOUT IT: DO YOU REALLY HAVE TO DO THIS STEP?
    v_halftime = v_init + a_curr*dt/2
   
    ### use current acceleration and velocity to update position
### WRITE CODE IMPLEMENTING THE FORMULA YOU OBTAINED, AS PART OF EXERCISE 1, FOR r(t+dt) HERE
    r_fut = r_curr + v_halftime * dt 
    
    ### random force in future
    rp_fut = np.sqrt(2*temperature_val*gamma_val*mu/dt)*np.random.normal(0,1)
    
    ### use rp_fut to get future acceleration a_fut (a_tilde at the future time in the assignment), 
    ### note that we cannot take future drag into account as we have not calculated our future velocity yet
### CODE IMPLEMENTING EQUATION 7 IN THE ASSIGNMENT DAS POSTED GOES HERE
    a_fut = (-1*f_interp(r_fut) + rp_fut)/mu - gamma_val*v_halftime
    
    ### use current and future acceleration to get future velocity v_fut
    ### note that we have to "correct" our formula relative to the formula for velocity Verlet
    ### as we have not included our future drag in our future acceleration
    
    v_fut = (v_halftime + a_fut * dt/2) * (1/(1 + gamma * dt/2))
### WRITE CODE IMPLEMENTING THE FORMULA YOU OBTAINED, AS PART OF EXERCISE 1, FOR r(t+dt) HERE
    #r_fut = r_curr + ((v_init + a_curr)*(dt/2))     
    
    result = [r_fut, v_fut, rp_fut]
    
    return result

### how many updates do you want to perform?
N_updates = 40000

### Now use r_init and v_init and run velocity verlet update N_updates times, plot results
### these arrays will store the time, the position vs time, and the velocity vs time
r_vs_t = np.zeros(N_updates)
v_vs_t = np.zeros(N_updates)
e_vs_t = np.zeros(N_updates)
t_array = np.zeros(N_updates)

### first entry is the intial position and velocity
r_vs_t[0] = r_init
v_vs_t[0] = v_init
e_vs_t[0] = (sE(r_init)-minE)+0.5*mu*v_init**2

### first BBK update
result_array = BBK(r_init, v_init, rp_init, gamma, temperature, mu, fE, dt)

### do the update N_update-1 more times
for i in range(1,N_updates):
    tmp = BBK(result_array[0], result_array[1], result_array[2], gamma, temperature, mu, fE, dt)
    result_array = tmp
    t_array[i] = dt*i
    r_vs_t[i] = result_array[0]
    v_vs_t[i] = result_array[1]
    e_vs_t[i] = (sE(result_array[0])-minE)+0.5*mu*v_init**2

### Plot the trajectory of bondlength vs time:
plt.title('Change in Bond Length vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Bond Length (a.u.)')
plt.plot(t_array, r_vs_t, 'red')
plt.show()
'''
### plot the phase space trajectory of position vs momentum
plt.title('Position vs. Momentum')
plt.xlabel('Position (a.u.)')
plt.ylabel('Momentum (a.u.)')
plt.plot(mu*v_vs_t, r_vs_t, 'blue')
plt.show()

### plot the total energy vs. time: does the average energy seem to converge?
plt.title('Total Energy vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (Hartrees)')
plt.plot(t_array,e_vs_t,'magenta')
plt.show()

### mean energy for next to last and last ten thousand time steps: does each equal k_B*T? are they similar to each other?
print("Mean Energy for next to last 100000 time steps of the trajectory is ",np.mean(e_vs_t[(N_updates-20000):(N_updates-10000)])," Hartrees")
print("Mean Energy for last 100000 time steps of the trajectory is ",np.mean(e_vs_t[(N_updates-10000):N_updates])," Hartrees")

'''