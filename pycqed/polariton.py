"""Polaritonic eigenstates, gradients and non-adiabatic couplings.

H(R) is complex symmetric, so its right eigenvectors can be *c-normalized*
(u_J^T u_K = delta_JK, no complex conjugation).  The left eigenvectors are then simply
U^T, the normalization fixes each eigenvector up to a sign, and the formulas below
reduce to the usual Hermitian ones when gamma = 0.

With dH the gradient of the matrix elements and D = D_el x 1 the electronic
derivative coupling in the |a(R)> x |n> basis, the polaritonic quantities are

    grad E_J = [UL dH U]_JJ
    d_JK     = [UL dH U]_JK / (E_K - E_J) + [UL D U]_JK          (J != K)

The second term is the chain-rule contribution of the intrinsic matter coupling
<g|grad e>; it vanishes when the electronic model does not supply one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cavity import ProjectedMatter
from .hamiltonian import CavityHamiltonian


class ExceptionalPointWarning(RuntimeWarning):
    """Raised when an eigenvector is close to self-orthogonal (u^T u -> 0)."""


@dataclass
class PolaritonSolution:
    """Eigen-decomposition of H(R) plus everything FSSH needs at that geometry.

    * ``energies`` (n,) complex
    * ``U`` (dim, n) right eigenvectors (columns); ``UL`` (n, dim) left eigenvectors, UL @ U = 1
    * ``grad_E`` (n, ndof) complex
    * ``nac`` (n, n, ndof) complex, nac[J, K] = <J|grad K>
    * ``H`` (dim, dim), ``dH`` (ndof, dim, dim), ``D`` (ndof, dim, dim): local-basis data
    """

    energies: np.ndarray
    U: np.ndarray
    UL: np.ndarray
    grad_E: np.ndarray
    nac: np.ndarray
    H: np.ndarray
    dH: np.ndarray
    D: np.ndarray
    matter: ProjectedMatter
    labels: list = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return self.energies.size

    def forces(self) -> np.ndarray:
        """(n, ndof) forces -Re grad E on every surface."""
        return -self.grad_E.real

    def with_signs(self, signs: np.ndarray) -> "PolaritonSolution":
        """Flip the sign of selected eigenvectors (and everything that depends on them)."""
        if np.all(signs > 0):
            return self
        self.U = self.U * signs[None, :]
        self.UL = self.UL * signs[:, None]
        self.nac = self.nac * (signs[:, None] * signs[None, :])[:, :, None]
        return self


# --- linear-algebra helpers ------------------------------------------------------------

def c_normalize(U: np.ndarray, ep_tol: float = 1e-3) -> np.ndarray:
    """Scale columns so that u^T u = 1; warn near an exceptional point."""
    cnorm = np.einsum("ij,ij->j", U, U)
    hnorm = np.einsum("ij,ij->j", U.conj(), U).real
    ratio = np.abs(cnorm) / hnorm
    if np.any(ratio < ep_tol):
        warnings.warn(f"near-exceptional point: min |u^T u| / u^H u = {ratio.min():.2e}",
                      ExceptionalPointWarning, stacklevel=3)
    return U / np.sqrt(cnorm)[None, :]


def left_vectors(U: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """UL = U^T when the c-normalized columns are c-orthogonal, else inv(U)."""
    UL = U.T
    if np.max(np.abs(UL @ U - np.eye(U.shape[1]))) > tol:
        UL = np.linalg.inv(U)
    return UL


def track_order(U: np.ndarray, prev_UL: np.ndarray) -> np.ndarray:
    """Permutation of columns of U that maximizes |overlap| with the previous states."""
    overlap = np.abs(prev_UL @ U)
    _, cols = linear_sum_assignment(-overlap)
    return cols


def sign_convention(U: np.ndarray, prev_UL: np.ndarray | None) -> np.ndarray:
    """Signs (+-1 per column) for a smooth gauge: positive overlap with the previous step,
    or largest component with positive real part when there is no previous step."""
    if prev_UL is not None:
        ov = np.einsum("ji,ij->j", prev_UL, U).real
        return np.where(ov < 0.0, -1.0, 1.0)
    big = U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])]
    return np.where(big.real < 0.0, -1.0, 1.0)


def diagonalize(H: np.ndarray, prev_UL: np.ndarray | None = None, ep_tol: float = 1e-3):
    """Eigen-decompose complex-symmetric H; returns (E, U, UL) ordered and sign-fixed.

    Without ``prev_UL`` states are sorted by Re E; with it they follow the previous states
    by maximum overlap (so labels survive true crossings).
    """
    E, U = np.linalg.eig(H)
    U = c_normalize(U, ep_tol)
    if prev_UL is None:
        order = np.argsort(E.real, kind="stable")
    else:
        order = track_order(U, prev_UL)
    E, U = E[order], U[:, order]
    U = U * sign_convention(U, prev_UL)[None, :]
    return E, U, left_vectors(U)


def sandwich(UL: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """UL @ A_k @ U for a stack A (ndof, dim, dim); returns (n, n, ndof)."""
    return np.moveaxis(UL[None] @ A @ U[None], 0, -1)


def couplings_from_matrices(E: np.ndarray, G: np.ndarray, Dt: np.ndarray,
                            degeneracy_tol: float = 1e-12) -> np.ndarray:
    """Combine G = UL dH U and Dt = UL D U (both (n, n, ...)) into d_JK.

    Pairs closer than ``degeneracy_tol`` in energy get no Hellmann-Feynman term (those are
    exact crossings of uncoupled states, e.g. different JC manifolds).
    """
    gap = E[None, :] - E[:, None]
    small = np.abs(gap) < degeneracy_tol
    np.fill_diagonal(small, True)
    inv_gap = np.where(small, 0.0, 1.0 / np.where(small, 1.0, gap))
    extra = (slice(None), slice(None)) + (None,) * (G.ndim - 2)
    d = G * inv_gap[extra] + Dt
    idx = np.arange(E.size)
    d[idx, idx] = 0.0
    return d


# --- solvers ---------------------------------------------------------------------------

class NumericalSolver:
    """Full numerical diagonalization; works for JC, Rabi and Pauli-Fierz."""

    def __init__(self, hamiltonian: CavityHamiltonian, ep_tol: float = 1e-3):
        self.ham = hamiltonian
        self.ep_tol = ep_tol

    def solve(self, matter: ProjectedMatter, prev: PolaritonSolution | None = None) -> PolaritonSolution:
        H = self.ham.matrix(matter)
        dH = self.ham.gradient(matter)
        D = self.ham.basis_coupling(matter)
        E, U, UL = diagonalize(H, None if prev is None else prev.UL, self.ep_tol)
        G = sandwich(UL, dH, U)
        idx = np.arange(E.size)
        grad_E = G[idx, idx]
        nac = couplings_from_matrices(E, G, sandwich(UL, D, U))
        labels = prev.labels if prev is not None else [f"P{j}" for j in range(E.size)]
        return PolaritonSolution(E, U, UL, grad_E, nac, H, dH, D, matter, labels)


class AnalyticJCSolver:
    """Closed-form Jaynes-Cummings solution, one 2x2 block per excitation number.

    State order: [G0, LP1, UP1, LP2, UP2, ..., LP_n, UP_n, E_n] where block N couples
    |g,N> and |e,N-1>, UP_N = cos t |g,N> + sin t |e,N-1> has energy (a+b)/2 + s with
    s = sqrt(((a-b)/2)^2 + c^2), and |e,n_fock> is uncoupled.  The branch of s is chosen
    continuous with ``prev`` (this matters below the exceptional point).
    """

    def __init__(self, hamiltonian: CavityHamiltonian, ep_tol: float = 1e-3):
        if hamiltonian.model != "JC":
            raise ValueError("AnalyticJCSolver only applies to the JC Hamiltonian")
        self.ham = hamiltonian
        self.ep_tol = ep_tol
        nf = hamiltonian.cavity.n_fock
        self.labels = ["G0"] + [f"{p}{N}" for N in range(1, nf + 1) for p in ("LP", "UP")] + [f"e,{nf}"]

    def solve(self, matter: ProjectedMatter, prev: PolaritonSolution | None = None) -> PolaritonSolution:
        ham, cav = self.ham, self.ham.cavity
        nf, nph, dim = cav.n_fock, cav.n_fock + 1, self.ham.dim
        ndof = matter.ndof
        w = cav.complex_omega
        kappa = cav.bilinear_prefactor
        Eg, Ee = matter.energies
        dEg, dEe = matter.energy_gradients
        Meg, dMeg = matter.dipole[1, 0], matter.dipole_gradients[1, 0]

        E = np.zeros(dim, dtype=complex)
        U = np.zeros((dim, dim), dtype=complex)
        grad_E = np.zeros((dim, ndof), dtype=complex)
        X = np.zeros((dim, dim, ndof), dtype=complex)

        def basis(a, n):
            return a * nph + n

        E[0], U[basis(0, 0), 0], grad_E[0] = Eg, 1.0, dEg
        E[-1], U[basis(1, nf), -1], grad_E[-1] = Ee + nf * w, 1.0, dEe

        for N in range(1, nf + 1):
            lp, up = 2 * N - 1, 2 * N
            a, b = Eg + N * w, Ee + (N - 1) * w
            c = kappa * Meg * np.sqrt(N)
            delta = 0.5 * (a - b)
            s = np.sqrt(delta * delta + c * c)
            if prev is not None:
                s_prev = 0.5 * (prev.energies[up] - prev.energies[lp])
                if abs(-s - s_prev) < abs(s - s_prev):
                    s = -s
            cos2, sin2 = delta / s, c / s
            if abs(1.0 + cos2) >= abs(1.0 - cos2):
                ct = np.sqrt(0.5 * (1.0 + cos2))
                st = sin2 / (2.0 * ct)
            else:
                st = np.sqrt(0.5 * (1.0 - cos2))
                ct = sin2 / (2.0 * st)
            if abs(ct) ** 2 + abs(st) ** 2 > 1.0 / self.ep_tol:
                warnings.warn(f"near-exceptional point in JC block N={N}", ExceptionalPointWarning,
                              stacklevel=2)

            E[up], E[lp] = 0.5 * (a + b) + s, 0.5 * (a + b) - s
            U[basis(0, N), up], U[basis(1, N - 1), up] = ct, st
            U[basis(0, N), lp], U[basis(1, N - 1), lp] = -st, ct

            d_delta = 0.5 * (dEg - dEe)
            d_c = kappa * np.sqrt(N) * dMeg
            d_s = (delta * d_delta + c * d_c) / s
            grad_E[up] = 0.5 * (dEg + dEe) + d_s
            grad_E[lp] = 0.5 * (dEg + dEe) - d_s
            d_theta = (delta * d_c - c * d_delta) / (2.0 * s * s)
            X[lp, up] = d_theta
            X[up, lp] = -d_theta

        UL = U.T
        D = ham.basis_coupling(matter)
        nac = X + sandwich(UL, D, U)
        idx = np.arange(dim)
        nac[idx, idx] = 0.0
        sol = PolaritonSolution(E, U, UL, grad_E, nac, ham.matrix(matter), ham.gradient(matter),
                                D, matter, list(self.labels))
        return sol.with_signs(sign_convention(U, None if prev is None else prev.UL))


def make_solver(hamiltonian: CavityHamiltonian, kind: str = "numerical", **kwargs):
    """Factory: ``kind`` is ``"numerical"`` or ``"analytic"`` (JC only)."""
    if kind == "numerical":
        return NumericalSolver(hamiltonian, **kwargs)
    if kind == "analytic":
        return AnalyticJCSolver(hamiltonian, **kwargs)
    raise ValueError(f"unknown solver {kind!r}")
