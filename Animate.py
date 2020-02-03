### Import all libraries and define various parameters here!
import numpy as np

from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import matplotlib.animation as animation
import time
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

### go through and read nuclear dynamics trajectories from each data file
'''
r_of_t = np.zeros((15,20000))
t_of_t = np.zeros(20000)
e_of_t = np.zeros((15,20000))

for i in range(1,2):
    #file_path = "Data/test" + str(i) + "_nuc_traj.txt"
    file_path = "Data/test_nuc_traj.txt"
    print(file_path)
    a = np.loadtxt(file_path)
    #print(len(a[:,0]))
    
    t_of_t[:] = a[:,0]
    r_of_t[i-1,:] = a[:,1]
    e_of_t[i-1,:] = a[:,2]
'''
r_of_t = [-0.6615,-0.6615 ]
e_of_t = [0.17681792512919203,0.17681792512919203 ]
t_of_t = [0., 0.12]
dt = 0.12

rlist = np.zeros(500)
PES = np.zeros((500,4))
CES = np.zeros((500,4))
file_path = "Data/test" + "_pes.txt"
b = np.loadtxt(file_path)

rlist[:] = b[:,0]
PES[:,0] = b[:,1]
PES[:,1] = b[:,2]
PES[:,2] = b[:,3]
PES[:,3] = b[:,4]

file_path = "Data/test" + "_photon_contribution.txt"
c = np.loadtxt(file_path)
CES[:,0] = b[:,1]
CES[:,1] = b[:,2]
CES[:,2] = b[:,3]
CES[:,3] = b[:,4]

fig, ax = plt.subplots()
cm = plt.cm.get_cmap('rainbow')
im = ax.scatter(rlist, 27.211*PES[:,0], c=CES[:,0], cmap=cm, s=4) 
im = ax.scatter(rlist, 27.211*PES[:,1], c=CES[:,1], cmap=cm, s=4)
im = ax.scatter(rlist, 27.211*PES[:,2], c=CES[:,2], cmap=cm, s=4) # rlist, 27.211*PPES[:,1], c=LP_p, cmap=cm )
im = ax.scatter(rlist, 27.211*PES[:,3], c=CES[:,3], cmap=cm, s=4)
cbar = fig.colorbar(im, ticks=[0.1, 0.5, 0.9])
cbar.ax.set_yticklabels(['excitonic', 'polaritonic', 'photonic'])
plt.xlim(-1.,1.)
#plt.ylim(3.5,6.)
plt.xlabel("R (a.u.)")
plt.ylabel("Energy (eV)")
plt.legend()
#plt.savefig(filename)
plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-3, 3), ylim=(0, 10))
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
plt.show()

'''
