"""Cavity mode parameters and projection of vector dipoles onto the mode polarization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electronic import ElectronicData


@dataclass(frozen=True)
class CavityMode:
    """A single lossy cavity mode.

    * ``omega``         real photon energy (Hartree)
    * ``coupling``      light-matter coupling magnitude lambda (a.u.)
    * ``polarization``  polarization vector (normalized on construction)
    * ``gamma``         photon loss rate (Hartree); the mode energy is omega - i gamma / 2
    * ``n_fock``        highest photon number kept (basis |0> ... |n_fock>)

    The bilinear coupling is sqrt(omega / 2) * lambda * (e . mu), with the *real* omega.
    """

    omega: float
    coupling: float
    polarization: tuple = (0.0, 0.0, 1.0)
    gamma: float = 0.0
    n_fock: int = 1

    def __post_init__(self):
        eps = np.asarray(self.polarization, dtype=float)
        norm = np.linalg.norm(eps)
        if eps.shape != (3,) or norm == 0.0:
            raise ValueError("polarization must be a non-zero 3-vector")
        object.__setattr__(self, "polarization", tuple(eps / norm))
        if self.n_fock < 1:
            raise ValueError("n_fock must be >= 1")

    @property
    def epsilon(self) -> np.ndarray:
        return np.array(self.polarization)

    @property
    def complex_omega(self) -> complex:
        return self.omega - 0.5j * self.gamma

    @property
    def bilinear_prefactor(self) -> float:
        return np.sqrt(self.omega / 2.0) * self.coupling

    @classmethod
    def from_coupling_strength(cls, g: float, omega: float, mu_eg: float, **kwargs) -> "CavityMode":
        """Build a mode whose JC coupling sqrt(omega/2) lambda mu_eg equals ``g``."""
        return cls(omega=omega, coupling=g / (np.sqrt(omega / 2.0) * mu_eg), **kwargs)


@dataclass(frozen=True)
class ProjectedMatter:
    """Two-level matter data with all dipoles scalarized as e . mu.

    Shapes: ``energies`` (2,), ``energy_gradients`` (2, ndof), ``dipole`` (2, 2),
    ``dipole_gradients`` (2, 2, ndof), ``nac`` (ndof,), ``dipole_squared`` (2, 2) or None,
    ``dipole_squared_gradients`` (2, 2, ndof) or None.
    """

    energies: np.ndarray
    energy_gradients: np.ndarray
    dipole: np.ndarray
    dipole_gradients: np.ndarray
    nac: np.ndarray
    dipole_squared: np.ndarray | None = None
    dipole_squared_gradients: np.ndarray | None = None

    @property
    def ndof(self) -> int:
        return self.energy_gradients.shape[-1]

    @property
    def hamiltonian(self) -> np.ndarray:
        return np.diag(self.energies)

    @property
    def hamiltonian_gradients(self) -> np.ndarray:
        """(2, 2, ndof): gradient of the (diagonal) matter Hamiltonian matrix."""
        dh = np.zeros((2, 2, self.ndof))
        dh[0, 0] = self.energy_gradients[0]
        dh[1, 1] = self.energy_gradients[1]
        return dh

    @property
    def nac_matrix(self) -> np.ndarray:
        """(2, 2, ndof): D_ab = <a|grad b>."""
        D = np.zeros((2, 2, self.ndof))
        D[0, 1] = self.nac
        D[1, 0] = -self.nac
        return D


def project(data: ElectronicData, cavity: CavityMode) -> ProjectedMatter:
    """Scalarize the dipoles of ``data`` along the cavity polarization."""
    eps = cavity.epsilon
    mu2 = dmu2 = None
    if data.dipole_squared is not None:
        mu2 = np.einsum("abij,i,j->ab", data.dipole_squared, eps, eps)
        if data.dipole_squared_gradients is not None:
            dmu2 = np.einsum("abijk,i,j->abk", data.dipole_squared_gradients, eps, eps)
        else:
            raise ValueError("dipole_squared given without its gradients")
    return ProjectedMatter(
        energies=np.asarray(data.energies, dtype=float),
        energy_gradients=np.asarray(data.energy_gradients, dtype=float),
        dipole=np.einsum("abi,i->ab", data.dipoles, eps),
        dipole_gradients=np.einsum("abik,i->abk", data.dipole_gradients, eps),
        nac=np.asarray(data.nac, dtype=float),
        dipole_squared=mu2,
        dipole_squared_gradients=dmu2,
    )
