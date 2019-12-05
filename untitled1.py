
import numpy as np
import matplotlib.pyplot as plt

tau = 3
tau_s = 3e-12
gam = 1/tau * np.pi * 2
au = 2.4188e-17

tau_au = tau_s / au

gam_au = 1/tau_au * np.pi * 2

print(gam_au)
print(gam)
t = np.linspace(0,3,1000)
f = np.exp(-gam*t)

plt.plot(t, f, 'red')
plt.show()
