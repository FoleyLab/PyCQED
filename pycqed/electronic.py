"""Interface between external electronic-structure (e.g. ML) models and the cavity code.

An electronic model maps nuclear coordinates ``R`` (flat array of length ``ndof``,
typically 3N Cartesians in bohr) to the two-level matter data in its *adiabatic*
basis {|g>, |e>}:

* energies ``E_g(R), E_e(R)``
* dipole vectors ``mu_gg(R), mu_ee(R), mu_eg(R)`` (3-vectors, a.u.)
* their gradients with respect to ``R``
* optionally the intrinsic non-adiabatic coupling ``d_ge = <g|grad e>``.

The phase of |e> is arbitrary at each geometry, so ``mu_eg`` and ``d_ge`` are only
defined up to a common sign.  :func:`align_phase` fixes that sign relative to a
reference geometry; it must be applied along trajectories and inside finite
differences, otherwise a sign flip appears as a delta-function gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

_DIPOLE_KEYS = ("mu_gg", "mu_ee", "mu_eg")


@dataclass(frozen=True)
class ElectronicData:
    """Two-level matter data at one geometry.

    Shapes (``ndof`` = number of nuclear degrees of freedom):

    * ``energies``          (2,)            [E_g, E_e]
    * ``dipoles``           (2, 2, 3)       dipoles[a, b] = <a|mu|b>, symmetric in a, b
    * ``energy_gradients``  (2, ndof)
    * ``dipole_gradients``  (2, 2, 3, ndof)
    * ``nac``               (ndof,)         d_ge = <g|grad e>  (d_eg = -d_ge)
    * ``dipole_squared``    (2, 2, 3, 3) or None: exact <a|mu_i mu_j|b>, used for the
      dipole self-energy when available (otherwise the projected square is used)
    * ``dipole_squared_gradients`` (2, 2, 3, 3, ndof) or None
    """

    energies: np.ndarray
    dipoles: np.ndarray
    energy_gradients: np.ndarray
    dipole_gradients: np.ndarray
    nac: np.ndarray
    dipole_squared: np.ndarray | None = None
    dipole_squared_gradients: np.ndarray | None = None

    @property
    def ndof(self) -> int:
        return self.energy_gradients.shape[-1]

    def flip_excited_phase(self) -> "ElectronicData":
        """Return the same data in the gauge |e> -> -|e>."""
        sign = np.array([[1.0, -1.0], [-1.0, 1.0]])
        mu2 = self.dipole_squared
        dmu2 = self.dipole_squared_gradients
        return replace(
            self,
            dipoles=self.dipoles * sign[:, :, None],
            dipole_gradients=self.dipole_gradients * sign[:, :, None, None],
            nac=-self.nac,
            dipole_squared=None if mu2 is None else mu2 * sign[:, :, None, None],
            dipole_squared_gradients=None if dmu2 is None else dmu2 * sign[:, :, None, None, None],
        )


@runtime_checkable
class ElectronicModel(Protocol):
    """Anything with ``evaluate(R) -> ElectronicData``."""

    def evaluate(self, R: np.ndarray) -> ElectronicData: ...


def align_phase(data: ElectronicData, reference: ElectronicData | None,
                dR: np.ndarray | None = None, tol: float = 1e-12) -> tuple[ElectronicData, bool]:
    """Choose the sign of |e> so that ``mu_eg`` (or ``d_ge``) is continuous with ``reference``.

    A flip of |e> changes the sign of ``mu_eg``, its gradient and ``d_ge`` together, so the
    test sums their overlaps with the reference (gradients and NAC taken per bohr).  ``dR``
    is the displacement from the reference geometry; when given, ``mu_eg`` is compared with
    its first-order extrapolation, so a transition dipole that genuinely passes through zero
    is not mistaken for a phase flip.

    Returns ``(aligned_data, flipped)``.
    """
    if reference is None:
        return data, False
    predicted = reference.dipoles[1, 0]
    if dR is not None:
        predicted = predicted + reference.dipole_gradients[1, 0] @ np.asarray(dR).reshape(-1)
    overlap = (float(np.dot(data.dipoles[1, 0], predicted))
               + float(np.sum(data.dipole_gradients[1, 0] * reference.dipole_gradients[1, 0]))
               + float(np.dot(data.nac, reference.nac)))
    if overlap < -tol:
        return data.flip_excited_phase(), True
    return data, False


def _dipole_matrix(mu_gg, mu_ee, mu_eg) -> np.ndarray:
    mu = np.empty((2, 2) + np.shape(mu_gg))
    mu[0, 0] = mu_gg
    mu[1, 1] = mu_ee
    mu[0, 1] = mu_eg
    mu[1, 0] = mu_eg
    return mu


class CallableModel:
    """Wrap a plain function (e.g. an ML predictor) as an :class:`ElectronicModel`.

    ``fn(R)`` must return a mapping with keys ``E_g, E_e`` (floats) and
    ``mu_gg, mu_ee, mu_eg`` (3-vectors).  It may also return

    * ``dE_g, dE_e``                 gradients, shape (ndof,)
    * ``dmu_gg, dmu_ee, dmu_eg``     Jacobians, shape (3, ndof)
    * ``nac_ge``                     <g|grad e>, shape (ndof,), in the same |e> phase as ``mu_eg``

    Any missing gradient is obtained by central finite differences with step ``fd_step``
    (2*ndof extra calls per evaluation, made only if something is missing).  The phase of
    ``mu_eg`` at the displaced points is aligned to the central point before differencing.
    """

    def __init__(self, fn: Callable[[np.ndarray], Mapping[str, object]], fd_step: float = 1e-4):
        self.fn = fn
        self.fd_step = fd_step

    def _values(self, R):
        out = self.fn(R)
        E = np.array([out["E_g"], out["E_e"]], dtype=float)
        mu = _dipole_matrix(*(np.asarray(out[k], dtype=float) for k in _DIPOLE_KEYS))
        return out, E, mu

    def evaluate(self, R: np.ndarray) -> ElectronicData:
        R = np.asarray(R, dtype=float)
        ndof = R.size
        out, E, mu = self._values(R)

        have_dE = "dE_g" in out and "dE_e" in out
        have_dmu = all("d" + k in out for k in _DIPOLE_KEYS)
        dE = np.array([out["dE_g"], out["dE_e"]], dtype=float) if have_dE else None
        dmu = (_dipole_matrix(*(np.asarray(out["d" + k], dtype=float) for k in _DIPOLE_KEYS))
               if have_dmu else None)

        if not (have_dE and have_dmu):
            fd_E = np.zeros((2, ndof))
            fd_mu = np.zeros((2, 2, 3, ndof))
            h = self.fd_step
            for k in range(ndof):
                Rp, Rm = R.copy(), R.copy()
                Rp[k] += h
                Rm[k] -= h
                _, Ep, mup = self._values(Rp)
                _, Em, mum = self._values(Rm)
                for m in (mup, mum):
                    if np.dot(m[1, 0], mu[1, 0]) < 0.0:
                        m[0, 1] *= -1.0
                        m[1, 0] *= -1.0
                fd_E[:, k] = (Ep - Em) / (2 * h)
                fd_mu[..., k] = (mup - mum) / (2 * h)
            dE = fd_E if dE is None else dE
            dmu = fd_mu if dmu is None else dmu

        nac = np.asarray(out.get("nac_ge", np.zeros(ndof)), dtype=float).reshape(ndof)
        return ElectronicData(energies=E, dipoles=mu, energy_gradients=dE,
                              dipole_gradients=dmu, nac=nac)
