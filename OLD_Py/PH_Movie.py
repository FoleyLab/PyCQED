### Import all libraries and define various parameters here!
import numpy as np
import dynamicshelper as dh
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
#import matplotlib.animation as animation

''' Some key parameters for the simulation! '''
### dissipation parameters for electronic and photonic system
gam_diss_np = 0.000005
gam_deph_np = 0.0000

gam_diss_m = 0.00000
gam_deph = 0.0000

### Initial position, velocity, timestep, and R-step for dynamics
ri = -0.7
vi = 0.0001282*1.5

### This is the reduced mass of this rotational mode for the dynamics... in atomic units
M = 1009883

### hbar omega_c in atomic units
omc = 2.18/27.211 
### hbar g_c in atomic units
gc = 0.136/27.211


### Number of updates for dynamics
N_time = 500000

### position displacement increment for dynamics (a.u.)
dr = 0.1 
### time displacement increment for dynamics (a.u.)
dt = 0.75


### array of dissipation parameters to be passed to RK4 function
gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_np
gamma[2] = gam_diss_m
gamma[3] = gam_diss_m+gam_diss_np



### various arrays for dynamics
'''
time = np.zeros(N_time)
r_of_t = np.zeros(N_time)
v_of_t = np.zeros(N_time)
e_of_t = np.zeros(N_time)
p_of_t = np.zeros((N_time,4))
'''
''' The following parameters and arrays pertain 
    to the polaritonic suraces!  '''
    
### array of values along the reactive coordinate R that the 
### surfaces will be explicitly evaluated at... this is in atomic units
rlist = np.linspace(-2.0, 2.0, 50)


### Htot
He = np.zeros((4,4))
Hp = np.zeros((4,4))
Hep = np.zeros((4,4))
Htot = np.zeros((4,4))
PPES = np.zeros((len(rlist),4))



''' Now let's evaluate the polaritonic PES and store the values to the PPES array! '''
Hp = dh.H_p(Hp, omc)
Hep = dh.H_ep(Hep, gc)

#### Get H_e(r) and diagonlize to get the polaritonic potential energy surfaces
for i in range(0,len(rlist)):
    r = rlist[i]
    He = dh.H_e(He, r)
    Htot = He + Hp + Hep
    tmpH = np.copy(Htot)
    vals, vecs = LA.eig(Htot)
    idx = vals.argsort()[::1]
    vals = vals[idx]
    for j in range(0,4):
        PPES[i,j] = vals[j]
    

### Plot the surfaces
'''
plt.plot(rlist, 27.211*PPES[:,0], 'b')
plt.plot(rlist, 27.211*PPES[:,1], 'g')
plt.plot(rlist, 27.211*PPES[:,2], 'y')
plt.plot(rlist, 27.211*PPES[:,3], 'r')
plt.xlim(-1.5,1.5)
plt.ylim(0,10)
plt.show()
'''

### density matrix
D = np.zeros((4,4),dtype=complex)
D[2,2] = 1.+0j
He = dh.H_e(He, ri)
### Hamiltonian matrix
'''
for i in range(0,N_time):
    #### Update nuclear coordinate first
    time[i] = i*dt
    res = dh.Erhenfest(ri, vi, M, D, Hp, Hep, He, gamma, gam_deph, dr, dt)
    #res = VelocityVerlet(Fg_spline, Fe_spline, np.real(D[0,0]), np.real(D[1,1]), M, ri, vi, dt)
    ri = res[0]
    vi = res[1]
    r_of_t[i] = ri
    v_of_t[i] = vi
    D = res[2]
    Htot = dh.H_e(He, ri) + Hp + Hep
    e_of_t[i] = dh.TrHD(Htot, D)
    for j in range(0,4):
        p_of_t[i,j] = np.real(D[j,j])

'''
plt.plot(rlist, 27.211*PPES[:,0], 'b')
plt.plot(rlist, 27.211*PPES[:,1], 'g')
plt.plot(rlist, 27.211*PPES[:,2], 'y')
plt.plot(rlist, 27.211*PPES[:,3], 'r')
#plt.plot(r_of_t, 27.211*e_of_t, 'red', label='Trajectory')
#plt.legend()
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-2, 2), ylim=(0, 6))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
linee1, = ax.plot([], [], lw=2)
linep2, = ax.plot([], [], lw=2)
linep1, = ax.plot([], [], lw=2)
lineg0, = ax.plot([], [], lw=2)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    #thisx = [0, x1[i], x2[i]]
    #thisy = [0, y1[i], y2[i]]
    thisx = [r_of_t[i]]
    thisy = [27.211*e_of_t[i]]
    
    
    line.set_data(thisx, thisy)
    linee1.set_data(rlist, 27.211*PPES[:,3])
    linep2.set_data(rlist, 27.211*PPES[:,2])
    linep1.set_data(rlist, 27.211*PPES[:,1])
    lineg0.set_data(rlist, 27.211*PPES[:,0])
    time_text.set_text(time_template % (i*dt))
    return line, linee1, linep2, linep1, lineg0, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(r_of_t)),
                              interval=dt*0.01, blit=True, init_func=init)
plt.show()



