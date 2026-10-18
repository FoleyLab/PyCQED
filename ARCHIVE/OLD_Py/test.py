
import numpy as np
from matplotlib import pyplot as plt
import math as mt

lam = np.linspace(100e-9,50000e-9,20600)
BBA = np.zeros_like(lam)
h=6.626*10**(-34) #Js
k=1.38*10**(-23) #mmKg/ssK
c=2.998*10**(8) #m/s
sig = 5.67*10**(-8)
b = 2.8977*10**(-3)
Ts=input()
T = float(Ts)
Wien = b/T
SB = sig*T**4

for L in range(0, len(lam)):
    BB = (2*h*c**2)/lam[L]**5
    BB = BB/((mt.exp(h*c/(lam[L]*k*T)))-1)

    BBA[L] = BB
    
plt.plot(lam*1*1e9, BBA, 'red')

plt.xlabel('wave length nm')
plt.ylabel('power density W/m^2/m')
plt.show()

def rect_rule(f, a):
    total = 0.
    dx = a[1]-a[0]
    for i in range(0,len(f)):
       total = total + dx*f[i]
    return total

max_y = max(BBA)  
max_x = lam[BBA.argmax()]

def error(a, b):   #a is calculated b is from Blackbody spectrum
    return ((abs(b-a))/a)*100

    
print(T, 'K')
print('stefan-Boltzman = ', SB,'W/m^2')
print('Area_BB = ', rect_rule(BBA,lam)*mt.pi,'W/m^2')
print('error = ', error(SB, rect_rule(BBA,lam)*mt.pi))
print('Wiens max wave length =', Wien*1*1e9,'nm')
print('Max wave length of BB = ', max_x*1*1e9, max_y,'nm')
print('error = ', error(Wien, max_x))