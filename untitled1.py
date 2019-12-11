
import numpy as np
import matplotlib.pyplot as plt


def CreateBas(dim, k):
    bas = np.zeros(dim)
    bas[k] = 1
    return bas


def TrHD(H, D):
    N = len(H)
    HD = np.dot(H,D)
    som = 0
    for i in range(0,N):
        som = som + HD[i,i]
    return np.real(som)

def Form_Rho(Psii, Psij):

    D = np.outer(Psii,np.conj(Psij))
    return D


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

mu = np.zeros((2,2))
D = np.zeros((2,2),dtype=complex)

dm = 0.5
mu[0,1] = dm
mu[1,0] = 2*dm

print(mu)

Psi1 = CreateBas(2, 0)
Psi2 = CreateBas(2,1)

print(Psi1)
print(Psi2)

D = Form_Rho(Psi1, Psi2)
print("length is ",len(D))

val = TrHD(mu, D)
print(val)


