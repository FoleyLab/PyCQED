"""Cavity-QED Hamiltonians in the matter-adiabatic x Fock product basis.

Basis ordering: index = a * (n_fock + 1) + n with a in {0: g, 1: e} and n = 0..n_fock.

With kappa = sqrt(omega / 2) * lambda and M_ab = e . mu_ab (all R-dependent):

* ``"JC"``   H = diag(E) + w b+b + kappa M_eg (s+ b + s- b+)
* ``"Rabi"`` H = diag(E) + w b+b + kappa M_eg (s+ + s-)(b + b+)
* ``"PF"``   H = diag(E) + w b+b + kappa M (b + b+) + (lambda^2 / 2) Q

where w = omega - i gamma / 2 and Q is the dipole self-energy matrix, (M @ M) by default
or the exact projected <a|(e . mu)^2|b> when the electronic model provides it.  All
couplings are real, so H is complex symmetric (H.T == H).

Every R-dependent term is linear in the matrix elements of the matter operators, so the
gradient of H is assembled from the same basis tensors (see :meth:`CavityHamiltonian.gradient`).
"""

from __future__ import annotations

import numpy as np

from .cavity import CavityMode, ProjectedMatter

COUPLING_MODELS = ("JC", "Rabi", "PF")


def _unit(a: int, b: int) -> np.ndarray:
    m = np.zeros((2, 2))
    m[a, b] = 1.0
    return m


class CavityHamiltonian:
    """Builds H(R), its gradient, and the electronic NAC operator for one cavity mode."""

    def __init__(self, cavity: CavityMode, model: str = "PF", dse: bool = True):
        if model not in COUPLING_MODELS:
            raise ValueError(f"model must be one of {COUPLING_MODELS}, got {model!r}")
        self.cavity = cavity
        self.model = model
        self.dse = dse and model == "PF"

        nph = cavity.n_fock + 1
        self.dim = 2 * nph
        b = np.diag(np.sqrt(np.arange(1, nph)), k=1)
        I_ph = np.eye(nph)
        self.annihilation = np.kron(np.eye(2), b)
        self.number = np.kron(np.eye(2), b.T @ b)
        self.photon_hamiltonian = cavity.complex_omega * self.number

        # P[a, b] = |a><b| x 1 ; T[a, b] = |a><b| x (photon operator for that term)
        self.P = np.array([[np.kron(_unit(a, c), I_ph) for c in range(2)] for a in range(2)])
        T = np.zeros((2, 2, self.dim, self.dim))
        q = b + b.T
        if model == "JC":
            T[1, 0] = np.kron(_unit(1, 0), b)      # s+ b
            T[0, 1] = np.kron(_unit(0, 1), b.T)    # s- b+
        else:
            pairs = ((0, 1), (1, 0)) if model == "Rabi" else ((0, 0), (0, 1), (1, 0), (1, 1))
            for a, c in pairs:
                T[a, c] = np.kron(_unit(a, c), q)
        self.T = T

        self.labels = [f"{'ge'[a]},{n}" for a in range(2) for n in range(nph)]

    # --- pieces ------------------------------------------------------------------------
    def _dse_matrix(self, matter: ProjectedMatter) -> np.ndarray:
        if matter.dipole_squared is not None:
            return matter.dipole_squared
        return matter.dipole @ matter.dipole

    def _dse_gradients(self, matter: ProjectedMatter) -> np.ndarray:
        if matter.dipole_squared_gradients is not None:
            return matter.dipole_squared_gradients
        M, dM = matter.dipole, matter.dipole_gradients
        return np.einsum("ack,cb->abk", dM, M) + np.einsum("ac,cbk->abk", M, dM)

    # --- public ------------------------------------------------------------------------
    def matrix(self, matter: ProjectedMatter) -> np.ndarray:
        """H(R) as a (dim, dim) complex array."""
        cav = self.cavity
        H = np.einsum("ab,abij->ij", matter.hamiltonian, self.P).astype(complex)
        H += self.photon_hamiltonian
        H += cav.bilinear_prefactor * np.einsum("ab,abij->ij", matter.dipole, self.T)
        if self.dse:
            H += 0.5 * cav.coupling ** 2 * np.einsum("ab,abij->ij", self._dse_matrix(matter), self.P)
        return H

    def gradient(self, matter: ProjectedMatter) -> np.ndarray:
        """dH/dR_k of the matrix elements, shape (ndof, dim, dim), real."""
        cav = self.cavity
        dH = np.einsum("abk,abij->kij", matter.hamiltonian_gradients, self.P)
        dH += cav.bilinear_prefactor * np.einsum("abk,abij->kij", matter.dipole_gradients, self.T)
        if self.dse:
            dH += 0.5 * cav.coupling ** 2 * np.einsum("abk,abij->kij", self._dse_gradients(matter), self.P)
        return dH

    def basis_coupling(self, matter: ProjectedMatter) -> np.ndarray:
        """(D_el x 1)_k = <a n|d/dR_k|b m>, shape (ndof, dim, dim), real antisymmetric."""
        return np.einsum("abk,abij->kij", matter.nac_matrix, self.P)

    def photon_content(self, U: np.ndarray) -> np.ndarray:
        """<n> for each column of U, using the ordinary (Hermitian) norm."""
        num = np.einsum("ij,ik,kj->j", U.conj(), self.number, U).real
        return num / np.einsum("ij,ij->j", U.conj(), U).real
