import warnings

import numpy as np
import pytest

from conftest import matter_at
from pycqed.cavity import CavityMode, ProjectedMatter
from pycqed.hamiltonian import COUPLING_MODELS, CavityHamiltonian
from pycqed.polariton import AnalyticJCSolver, ExceptionalPointWarning, NumericalSolver, sandwich


def match(ref, other):
    """Permutation and signs mapping states of ``other`` onto ``ref``."""
    perm = np.argmax(np.abs(ref.UL @ other.U), axis=1)
    assert len(set(perm)) == ref.n_states
    signs = np.sign(np.einsum("ji,ij->j", ref.UL, other.U[:, perm]).real)
    return perm, signs


def reordered_nac(sol, perm, signs):
    return sol.nac[np.ix_(perm, perm)] * (signs[:, None] * signs[None, :])[:, :, None]


@pytest.mark.parametrize("n_fock", [1, 3, 6])
@pytest.mark.parametrize("gamma", [0.0, 0.002])
def test_analytic_jc_matches_numerical(tully, n_fock, gamma):
    cav = CavityMode(omega=0.02, coupling=0.02, gamma=gamma, n_fock=n_fock)
    ham = CavityHamiltonian(cav, "JC")
    ana, num = AnalyticJCSolver(ham), NumericalSolver(ham)
    for x in (-2.0, -0.3, 0.0, 0.45, 1.7):
        pm = matter_at(tully, x, cav)
        a, n = ana.solve(pm), num.solve(pm)
        perm, signs = match(a, n)
        assert np.allclose(a.energies, n.energies[perm], atol=1e-10)
        assert np.allclose(a.grad_E, n.grad_E[perm], atol=1e-10)
        assert np.allclose(a.nac, reordered_nac(n, perm, signs), atol=1e-9)


@pytest.mark.parametrize("model", COUPLING_MODELS)
def test_nac_matches_eigenvector_finite_difference(tully, cavity, model):
    """d = UL grad U + UL D U, with grad U from gauge-aligned neighbouring solutions."""
    solver = NumericalSolver(CavityHamiltonian(cavity, model))
    x, h = 0.21, 1e-5
    sol = solver.solve(matter_at(tully, x, cavity))
    up = solver.solve(matter_at(tully, x + h, cavity), prev=sol)
    dn = solver.solve(matter_at(tully, x - h, cavity), prev=sol)
    fd = sol.UL @ (up.U - dn.U) / (2 * h) + sandwich(sol.UL, sol.D, sol.U)[..., 0]
    np.fill_diagonal(fd, 0.0)
    assert np.allclose(sol.nac[..., 0], fd, atol=1e-6)


@pytest.mark.parametrize("model", COUPLING_MODELS)
def test_chain_rule_against_fixed_diabatic_basis(tully, cavity, model):
    """Express H in the R-independent diabatic x Fock basis, where the NAC is a pure
    Hellmann-Feynman expression, and compare with the adiabatic-basis chain rule."""
    ham = CavityHamiltonian(cavity, model)
    solver = NumericalSolver(ham)
    eye = np.eye(cavity.n_fock + 1)

    def H_dia(x):
        W = np.kron(tully.rotation(x)[0], eye)
        return W @ ham.matrix(matter_at(tully, x, cavity)) @ W.T

    x, h = -0.15, 1e-5
    sol = solver.solve(matter_at(tully, x, cavity))
    W = np.kron(tully.rotation(x)[0], eye)
    U_dia = W @ sol.U
    dH_dia = (H_dia(x + h) - H_dia(x - h)) / (2 * h)
    G = U_dia.T @ dH_dia @ U_dia
    gap = sol.energies[None, :] - sol.energies[:, None]
    np.fill_diagonal(gap, 1.0)
    ref = G / gap
    np.fill_diagonal(ref, 0.0)

    intrinsic = sandwich(sol.UL, sol.D, sol.U)[..., 0]
    assert np.max(np.abs(intrinsic)) > 1e-2          # the chain-rule term is really exercised
    assert np.allclose(sol.nac[..., 0], ref, atol=1e-6)
    assert np.allclose(np.diag(G), sol.grad_E[:, 0], atol=1e-8)


def test_jc_intrinsic_nac_couples_manifolds_not_lp_up(tully, cavity):
    """In JC, <g|grad e> links |g,n> and |e,n>, which belong to different excitation
    manifolds: it couples LP1/UP1 to G0 but contributes nothing to <LP1|grad UP1>."""
    sol = AnalyticJCSolver(CavityHamiltonian(cavity, "JC")).solve(matter_at(tully, 0.1, cavity))
    intrinsic = sandwich(sol.UL, sol.D, sol.U)[..., 0]
    G0, LP1, UP1 = (sol.labels.index(k) for k in ("G0", "LP1", "UP1"))
    assert abs(intrinsic[LP1, UP1]) < 1e-14
    assert abs(intrinsic[G0, LP1]) > 1e-2 and abs(intrinsic[G0, UP1]) > 1e-2


def test_sign_gauge_is_continuous(tully, cavity):
    solver = NumericalSolver(CavityHamiltonian(cavity, "PF"))
    sol = None
    for x in np.linspace(-3, 3, 400):
        new = solver.solve(matter_at(tully, x, cavity), prev=sol)
        if sol is not None:
            assert np.all(np.einsum("ji,ij->j", sol.UL, new.U).real > 0.5)
        sol = new


def test_exceptional_point_warning():
    gamma = 0.004
    cav = CavityMode(omega=0.1, coupling=1.0, gamma=gamma, n_fock=1)
    kappa = cav.bilinear_prefactor
    M = np.array([[0.0, 1.0], [1.0, 0.0]]) * (gamma / 4) / kappa * (1 + 1e-9)
    pm = ProjectedMatter(energies=np.array([0.0, 0.1]), energy_gradients=np.zeros((2, 1)),
                         dipole=M, dipole_gradients=np.zeros((2, 2, 1)), nac=np.zeros(1))
    for solver in (NumericalSolver, AnalyticJCSolver):
        with pytest.warns(ExceptionalPointWarning):
            solver(CavityHamiltonian(cav, "JC")).solve(pm)
    far = ProjectedMatter(pm.energies, pm.energy_gradients, 10 * M, pm.dipole_gradients, pm.nac)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NumericalSolver(CavityHamiltonian(cav, "JC")).solve(far)


def test_analytic_solver_rejects_other_models(cavity):
    with pytest.raises(ValueError):
        AnalyticJCSolver(CavityHamiltonian(cavity, "PF"))
