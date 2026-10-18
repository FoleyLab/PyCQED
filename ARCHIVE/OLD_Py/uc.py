import numpy as np
from numpy import pi,sin,cos,sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
 
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.gca(projection='3d')
 
x = np.linspace(0,2,4)
y = np.linspace(0,2,4)
 
X,Y = np.meshgrid(x,y)
 
#Wave speed
c = 1
 
#Initial time
t0 = 0
 
#Time increment
dt = 0.03
 
#Try every combination
p = 5 #1 #5 #2 #5
q = 6 #1 #5 #3 #3
 
w = pi*c*sqrt(p**2+q**2)
 
### finite-differences first derivative

def d2fdx2(ft, xt, yt):
    h = xt[1]-xt[0]
    ftpp = np.linspace_like(ft)
    for i in range(0, len(yt)):
        for j in range(0, len(xt)):
            if (j>0 and j<len(ft)-1):
                rise = ft[j+1][i]-2*ft[j][i]-ft[j-1][i]
                ftpp[j][i] = rise/(h*h)
            ### last i    
            elif j>0:
                rise = ft[j][i]-2*ft[j-1][i]-ft[j-2][i]
                ftpp[j][i] = rise/(h*h)
                ### first i
            else:
                rise = ft[j+2][i] - 2*ft[j+1][i]-ft[j][i]
                ftpp[j][i] = rise/(h*h)
                
    return ftpp


def d2fdy2(ft, xt, yt):
    h = yt[1]-yt[0]
    ftpp = np.linspace_like(ft)
    for i in range(0, len(xt)):
        for j in range(0, len(yt)):
            if (j>0 and j<len(ft)-1):
                rise = ft[i][j+1]-2*ft[i][j]-ft[i][j-1]
                ftpp[i][j] = rise/(h*h)
            ### last i    
            elif j>0:
                rise = ft[i][j]-2*ft[i][j-1]-ft[i][j-2]
                ftpp[i][j] = rise/(h*h)
                ### first i
            else:
                rise = ft[i][j+2] - 2*ft[i][j+1]-ft[i][j]
                ftpp[i][j] = rise/(h*h)
                
    return ftpp

def HPsi(Psi, xt, yt):
    Psidot = np.linspace_like(ft)
    Psidot = -0.5*d2fdx2(Psi, xt, yt) -0.5*d2fdy2(Psi, xt, yt)
    return Psidot


def RK4(Psi, xt, yt, h, t):
    k1 = np.zeros_like(Psi)
    k2 = np.zeros_like(Psi)
    k3 = np.zeros_like(Psi)
    k4 = np.zeros_like(Psi)
    Psi1 = np.zeros_like(Psi)
    Psi2 = np.zeros_like(Psi)
    Psi3 = np.zeros_like(Psi)
    Psi4 = np.zeros_like(Psi)
    

    Psi1 = Psi    
    k1 = h*HPsi(Psi, xt, yt)
    
    Psi2 = Psi+k1/2.
    k2 = h*HPsi(Psi, xt, yt)
    
    ### UPdate H and D and get k3
    Psi3 = Psi+k2/2
    k3 = h*HPsi(Psi, xt, yt)
    

    Psi4 = Psi+k3
    k4 = h*HPsi(Psi, xt, yt)
    
    Psif = Psi + (1/6.)*(k1 + 2.*k2 + 2*k3 + k4)
    return Psif

def u(x,y,t):
    #return (cos(w*t)+sin(w*t))*sin(pi*p*x)*sin(q*pi*y)
    ci = 0.+1j
    return np.exp(ci*w*t)*np.exp(ci*pi*p*x)*np.exp(-2*y)

#Building the datapoints
a = []
for i in range(1):
    z = np.real(u(X,Y,t0))
    t0 = t0 + dt
    a.append(z)
 
print(z[2][0:3])
#Adding the colorbar 
m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
m.set_array(a[0])
cbar = plt.colorbar(m)
 
k = 0
def animate(i):
    global k
    Z = a[k]
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
    #ax1.contour(X,Y,Z)
    ax1.set_zlim(0,5)
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
plt.show()