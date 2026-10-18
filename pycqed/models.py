"""Analytic electronic models implementing :class:`~pycqed.electronic.ElectronicModel`.

These are used for regression tests and examples; production runs plug an ML model in
through :class:`~pycqed.electronic.CallableModel`.
"""

from __future__ import annotations

import numpy as np

from .electronic import ElectronicData


class DoubleWellModel:
    """The 1-D isomerization model hard-coded in the legacy ``polaritonic.E_of_R``.

    Each adiabatic surface is the lower root of a 2x2 diabatic problem built from
    displaced harmonic wells.  The legacy code had no dipoles, only a constant coupling
    ``g``; here ``mu_eg`` is a constant vector (default unit length along z) and the
    permanent dipoles are zero, so ``CavityMode.from_coupling_strength`` reproduces it.
    """

    A = np.array([0.049244, 0.010657, 0.428129, 0.373005])
    B = np.array([0.18, 0.18, 0.18, 0.147])
    R0 = np.array([-0.75, 0.85, -1.15, 1.25])
    D = np.array([0.073, 0.514])

    def __init__(self, mu_eg=(0.0, 0.0, 1.0), mu_gg=(0.0, 0.0, 0.0), mu_ee=(0.0, 0.0, 0.0)):
        self.mu = np.zeros((2, 2, 3))
        self.mu[0, 0] = mu_gg
        self.mu[1, 1] = mu_ee
        self.mu[0, 1] = self.mu[1, 0] = mu_eg

    def evaluate(self, R) -> ElectronicData:
        x = float(np.asarray(R).reshape(-1)[0])
        v = self.A + self.B * (x - self.R0) ** 2
        dv = 2.0 * self.B * (x - self.R0)
        E = np.empty(2)
        dE = np.empty((2, 1))
        for s, (i, j) in enumerate(((0, 1), (2, 3))):
            root = np.sqrt(self.D[s] ** 2 + 0.25 * (v[i] - v[j]) ** 2)
            E[s] = 0.5 * (v[i] + v[j]) - root
            dE[s, 0] = 0.5 * (dv[i] + dv[j]) - 0.25 * (v[i] - v[j]) * (dv[i] - dv[j]) / root
        return ElectronicData(energies=E, dipoles=self.mu.copy(), energy_gradients=dE,
                              dipole_gradients=np.zeros((2, 2, 3, 1)), nac=np.zeros(1))


class DiabaticTwoStateModel:
    """Adiabatic data generated from a 1-D two-state *diabatic* model.

    Diabatic potentials default to Tully's simple avoided crossing:
    V11 = A(1 - exp(-B|x|)) sign(x), V22 = -V11, V12 = C exp(-D x^2).
    The diabatic dipole matrix ``mu_diabatic`` (2, 2, 3) is R-independent.  The
    adiabatic states are |e> = (cos t, sin t), |g> = (-sin t, cos t) in the diabatic
    basis with tan 2t = 2 V12 / (V11 - V22), which gives a non-zero intrinsic
    coupling d_ge = <g|grad e> = grad t and R-dependent adiabatic dipoles.  Because
    everything is known in closed form, this model is the reference for the
    chain-rule polariton NAC tests.
    """

    def __init__(self, A=0.01, B=1.6, C=0.005, D=1.0, offset=0.0, mu_diabatic=None):
        self.A, self.B, self.C, self.D, self.offset = A, B, C, D, offset
        if mu_diabatic is None:
            mu_diabatic = np.zeros((2, 2, 3))
            mu_diabatic[0, 0] = (0.0, 0.0, 0.8)
            mu_diabatic[1, 1] = (0.0, 0.0, -0.5)
            mu_diabatic[0, 1] = mu_diabatic[1, 0] = (0.0, 0.0, 1.2)
        self.mu_diabatic = np.asarray(mu_diabatic, dtype=float)

    def diabatic(self, x: float):
        """Return (V, dV) as 2x2 arrays."""
        A, B, C, D = self.A, self.B, self.C, self.D
        s = np.sign(x) if x != 0 else 1.0
        V11 = s * A * (1.0 - np.exp(-B * abs(x)))
        dV11 = A * B * np.exp(-B * abs(x))
        V12 = C * np.exp(-D * x * x)
        dV12 = -2.0 * D * x * V12
        V = np.array([[V11, V12], [V12, -V11]]) + self.offset * np.eye(2)
        dV = np.array([[dV11, dV12], [dV12, -dV11]])
        return V, dV

    def rotation(self, x: float):
        """Return (W, dW, t, dt): columns of W are |g>, |e> in the diabatic basis."""
        V, dV = self.diabatic(x)
        X, Y = V[0, 0] - V[1, 1], 2.0 * V[0, 1]
        dX, dY = dV[0, 0] - dV[1, 1], 2.0 * dV[0, 1]
        t = 0.5 * np.arctan2(Y, X)
        dt = 0.5 * (X * dY - Y * dX) / (X * X + Y * Y)
        c, s = np.cos(t), np.sin(t)
        W = np.array([[-s, c], [c, s]])
        dW = dt * np.array([[-c, -s], [-s, c]])
        return W, dW, t, dt

    def evaluate(self, R) -> ElectronicData:
        x = float(np.asarray(R).reshape(-1)[0])
        V, dV = self.diabatic(x)
        W, dW, _, dt = self.rotation(x)
        E = np.diag(W.T @ V @ W).copy()
        dE = np.diag(W.T @ dV @ W).copy()[:, None]
        mu = np.einsum("ia,ijx,jb->abx", W, self.mu_diabatic, W)
        dmu = (np.einsum("ia,ijx,jb->abx", dW, self.mu_diabatic, W)
               + np.einsum("ia,ijx,jb->abx", W, self.mu_diabatic, dW))
        return ElectronicData(energies=E, dipoles=mu, energy_gradients=dE,
                              dipole_gradients=dmu[..., None], nac=np.array([dt]))
