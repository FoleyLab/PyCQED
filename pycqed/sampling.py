"""Initial conditions: Wigner sampling of harmonic ground states."""

from __future__ import annotations

import numpy as np


def wigner_1d(R0: float, mass: float, force_constant: float, rng=None, size=None):
    """Sample (R, P) from the ground-state Wigner distribution of a 1-D harmonic well.

    With a = sqrt(k M) = M omega: sigma_R = 1/sqrt(2a), sigma_P = sqrt(a/2) (hbar = 1).
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    a = np.sqrt(force_constant * mass)
    R = rng.normal(R0, np.sqrt(1.0 / (2.0 * a)), size)
    P = rng.normal(0.0, np.sqrt(a / 2.0), size)
    return R, P


def wigner_normal_modes(R0, masses, hessian, rng=None, freq_cutoff: float = 1e-6):
    """Sample (R, P) from the ground-state Wigner distribution of the normal modes.

    ``hessian`` is the Cartesian Hessian (ndof, ndof) at the minimum ``R0``; modes with
    frequency below ``freq_cutoff`` (translations, rotations) are left at zero.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    R0 = np.asarray(R0, dtype=float).reshape(-1)
    m = np.broadcast_to(np.asarray(masses, dtype=float), R0.shape)
    inv_sqrt_m = 1.0 / np.sqrt(m)
    w2, L = np.linalg.eigh(hessian * inv_sqrt_m[:, None] * inv_sqrt_m[None, :])
    q = np.zeros_like(w2)
    p = np.zeros_like(w2)
    for i, x in enumerate(w2):
        if x > freq_cutoff ** 2:
            w = np.sqrt(x)
            q[i] = rng.normal(0.0, np.sqrt(1.0 / (2.0 * w)))
            p[i] = rng.normal(0.0, np.sqrt(w / 2.0))
    return R0 + inv_sqrt_m * (L @ q), np.sqrt(m) * (L @ p)
