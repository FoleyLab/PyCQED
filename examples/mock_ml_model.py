"""Stand-in for an ML electronic-structure model: a heteronuclear diatomic in 3-D.

``predict(R)`` takes the flat Cartesian coordinates of two atoms (6 numbers, bohr) and
returns only *values*: energies (Hartree) and dipole 3-vectors (a.u.) along the bond.
No gradients are returned, so :class:`pycqed.electronic.CallableModel` finite-differences
them, which is the path an ML model without autograd will take.
"""

import numpy as np

MASSES = np.repeat([1837.15, 19.0 * 1822.89], 3)  # H-like and F-like atoms, a.u.


def _morse(r, D, a, r0):
    return D * (1.0 - np.exp(-a * (r - r0))) ** 2


def predict(R):
    R = np.asarray(R, dtype=float).reshape(2, 3)
    bond = R[1] - R[0]
    r = np.linalg.norm(bond)
    e = bond / r
    return {
        "E_g": _morse(r, 0.17, 1.0, 1.75),
        "E_e": 0.10 + _morse(r, 0.06, 0.9, 2.05),
        "mu_gg": (0.5 + 0.1 * (r - 1.75)) * e,
        "mu_ee": (-0.4 + 0.2 * (r - 1.75)) * e,
        "mu_eg": 1.5 * np.exp(-0.5 * (r - 1.9) ** 2) * e,
    }
