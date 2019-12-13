### Import all libraries and define various parameters here!
import numpy as np
import dynamicshelper as dh
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.animation as animation

''' Some key parameters for the simulation! '''
### dissipation parameters for electronic and photonic system
gam_diss_np = 0.01
gam_deph_np = 0.0000

gam_diss_m = 0.00000
gam_deph = 0.0000



au_to_ps = 2.4188e-17 * 1e12
### This is the reduced mass of this rotational mode for the dynamics... in atomic units
M = 1009883

### hbar omega_c in atomic units
omc = 2.45/27.211 
### hbar g_c in atomic units
gc = 0.02/27.211


### Number of updates for dynamics
N_time = 40000 #00

### position displacement increment for dynamics (a.u.)
dr = 0.01 
### time displacement increment for dynamics (a.u.)
dt = 0.25

### initial polariton state
pn = 2

### array of dissipation parameters to be passed to RK4 function
gamma = np.zeros(4)
gamma[0] = 0.
gamma[1] = gam_diss_np
gamma[2] = gam_diss_m
gamma[3] = gam_diss_m+gam_diss_np



### various arrays for dynamics

time = np.zeros(N_time)
r_of_t = np.zeros((N_time,3))
hf_error_of_t = np.zeros((N_time, 3))
tot_error_of_t = np.zeros((N_time, 3))

oo = np.zeros((N_time, 1))
tt = np.zeros((N_time, 1))
v_of_t = np.zeros((N_time,3))
e_of_t = np.zeros((N_time,3))
p_of_t = np.zeros((N_time,4))

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
        
### form spline for ground-state surface
i_spline = InterpolatedUnivariateSpline(rlist, PPES[:,1], k=3)
Fi_spline = i_spline.derivative()
  
g_spline = InterpolatedUnivariateSpline(rlist, PPES[:,0], k=3)
Fg_spline = g_spline.derivative()  

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



#[Ht, Dl, vec] = dh.Transform_P_to_L(ri[0], Dpl, Hp, Hep)
#print(Dl)
#HD = np.dot(Ht,Dl)
#print(Ht)


flag = 1
T = 0.01 # boiling point of CO in atomic units
g_n = 0.0001
ri_val = [-0.6615679318398704, -0.698020918719673, -0.7325842045116059]
vi_val = [3.33752906715916e-05, -1.7604569932905628e-05, 7.215162825258178e-07]
iso_res = np.zeros(len(vi_val))


for j in range(0,3):
    pn = 2
    ### density matrix in polariton basis!
    Dl = np.zeros((4,4),dtype=complex)
    Dl[pn,pn] = 1.+0j
    ri = ri_val[j]
    vi = vi_val[j]
    gs_r = []
    
    for i in range(0,N_time):
        #### Update nuclear coordinate first
        time[i] = i*dt
        #res = dh.Erhenfest_v2(ri, vi, M, Dl, Hp, Hep, He, gamma, gam_deph, dr, dt)
        res = dh.FSSH_Update(ri, vi, M, g_n, T, Dl, Hp, Hep, He, gamma, gam_deph, dr, dt, pn)
        ri = res[0]
        vi = res[1]
        r_of_t[i,j] = ri
        v_of_t[i,j] = vi
        e_of_t[i,j] = res[2] #i_spline(ri)
        Dl = res[3]
        pn = res[4]
        p_of_t[i,0] = np.real(Dl[0,0])
        p_of_t[i,1] = np.real(Dl[1,1])
        p_of_t[i,2] = np.real(Dl[2,2])
        p_of_t[i,3] = np.real(Dl[3,3])
        
        if (pn==0):
            gs_r.append(ri)
    ### if we have accumulated a trajectory on the gs surface        
    if (len(gs_r)>1):
        avg_r = sum(gs_r) / len(gs_r)
        if (avg_r>0.1):
            iso_res[j] = 1
        else:
            iso_res[j] = 0
    ### if we haven't decayed to the gs, the simulation is indeterminant
    else:
        iso_res[j] = 0.5

print(iso_res)
'''
plt.plot(time*au_to_ps, p_of_t[:,0], 'r')
plt.plot(time*au_to_ps, p_of_t[:,1], 'y')
plt.plot(time*au_to_ps, p_of_t[:,2], 'g')
plt.plot(time*au_to_ps, p_of_t[:,3], 'b')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-3, 3), ylim=(0, 10))
ax.grid()

line, = ax.plot([], [], 'o', lw=2)
line1, = ax.plot([], [], 'o', lw=2)
linee1, = ax.plot([], [], lw=2)
linep2, = ax.plot([], [], lw=2)
linep1, = ax.plot([], [], lw=2)
lineg0, = ax.plot([], [], lw=2)

time_template = 'time = %.1e ps'
time_text = ax.text(0.05, 0.02, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = r_of_t[i,:]
    thisy = 27.211*e_of_t[i,:]     
    line.set_data(thisx, thisy)
    linee1.set_data(rlist, 27.211*PPES[:,3])
    linep2.set_data(rlist, 27.211*PPES[:,2])
    linep1.set_data(rlist, 27.211*PPES[:,1])
    lineg0.set_data(rlist, 27.211*PPES[:,0])
    time_text.set_text(time_template % (i*dt * au_to_ps))
    return line, linee1, linep2, linep1, lineg0, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(r_of_t),100),
                              interval=dt, blit=True, init_func=init)
plt.show()





#import numpy as np
#from matplotlib import pyplot as plt
'''
''' get reasonable starting positions and velocities!
k = 0.31246871517560126
M = 1009883

a = np.sqrt(k*M)

x = np.linspace(-0.5,0.5,500)
N = (a/np.pi)**(1/4.)
sig = np.sqrt(1/(2*a))
pre = 1/(sig * np.sqrt(2*np.pi))

psi0 = N * np.exp(-a/2 * x**2)
g = pre * np.exp(-0.5 * (x/sig)**2)

xf = np.linspace(-sig, sig, 100)
f = 0*xf + 7
#plt.plot(x, psi0, 'red')
plt.plot(x, psi0**2, 'blue')
plt.plot(x, g, 'r--')
plt.plot(xf, f, 'green')

p_unc = np.sqrt(a - a**2 * np.sqrt(np.pi)/(2*a**(3/2)))
v_unc = p_unc / M
print("Momentum uncertainty",p_unc)
print("Velocity uncertainty",p_unc/M)

from random import seed
from random import gauss
# seed random number generator
seed(1)
# generate some Gaussian values
for _ in range(10):
    value_x = gauss(-0.7, sig)
    value_p = gauss(0, v_unc)
    print(value_x, value_p)
    

plt.show()
print(sig)
'''