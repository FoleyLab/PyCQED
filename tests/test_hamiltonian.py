import numpy as np
import pytest

from conftest import matter_at
from pycqed.cavity import CavityMode, project
from pycqed.hamiltonian import COUPLING_MODELS, CavityHamiltonian
from pycqed.models import DoubleWellModel


@pytest.mark.parametrize("model", COUPLING_MODELS)
def test_complex_symmetric(tully, cavity, model):
    ham = CavityHamiltonian(cavity, model)
    H = ham.matrix(matter_at(tully, 0.3, cavity))
    assert H.shape == (2 * (cavity.n_fock + 1),) * 2
    assert np.allclose(H, H.T)
    assert np.allclose(np.diag(H).imag, -0.5 * cavity.gamma * np.diag(ham.number))


def test_jc_conserves_excitations(tully, cavity):
    ham = CavityHamiltonian(cavity, "JC")
    H = ham.matrix(matter_at(tully, -0.4, cavity))
    n_exc = ham.number + ham.P[1, 1]
    assert np.allclose(H @ n_exc, n_exc @ H)
    rabi = CavityHamiltonian(cavity, "Rabi").matrix(matter_at(tully, -0.4, cavity))
    assert not np.allclose(rabi @ n_exc, n_exc @ rabi)


@pytest.mark.parametrize("model", COUPLING_MODELS)
def test_gradient_matches_finite_difference(tully, cavity, model):
    ham = CavityHamiltonian(cavity, model)
    x, h = 0.37, 1e-5
    dH = ham.gradient(matter_at(tully, x, cavity))[0]
    fd = (ham.matrix(matter_at(tully, x + h, cavity)) - ham.matrix(matter_at(tully, x - h, cavity))) / (2 * h)
    assert np.allclose(dH, fd, atol=1e-8)


def test_pauli_fierz_dse_by_hand(tully):
    cav = CavityMode(omega=0.05, coupling=0.03, n_fock=2)
    pm = matter_at(tully, 0.2, cav)
    H = CavityHamiltonian(cav, "PF").matrix(pm)
    M, lam, kappa = pm.dipole, cav.coupling, np.sqrt(cav.omega / 2) * cav.coupling
    g0, e0, g1 = 0, 3, 1                      # basis index a * (n_fock + 1) + n
    assert np.isclose(H[g0, g0], pm.energies[0] + 0.5 * lam ** 2 * (M[0, 0] ** 2 + M[0, 1] ** 2))
    assert np.isclose(H[e0, g0], 0.5 * lam ** 2 * (M[1, 0] * M[0, 0] + M[1, 1] * M[1, 0]))
    assert np.isclose(H[g1, g0], kappa * M[0, 0])           # permanent-dipole term
    assert np.isclose(H[e0 + 1, g0], kappa * M[1, 0])       # |e,1><g,0| counter-rotating
    no_dse = CavityHamiltonian(cav, "PF", dse=False).matrix(pm)
    assert np.isclose(no_dse[g0, g0], pm.energies[0])


def test_polarization_projection():
    model = DoubleWellModel(mu_eg=(0.0, 0.0, 2.0))
    data = model.evaluate([0.1])
    perp = CavityMode(omega=0.09, coupling=0.05, polarization=(1.0, 0.0, 0.0))
    tilt = CavityMode(omega=0.09, coupling=0.05, polarization=(1.0, 0.0, 1.0))
    assert project(data, perp).dipole[1, 0] == 0.0
    assert np.isclose(project(data, tilt).dipole[1, 0], 2.0 / np.sqrt(2.0))


def test_exact_dipole_squared_hook(tully):
    cav = CavityMode(omega=0.05, coupling=0.03, n_fock=1, polarization=(0, 0, 1))
    data = tully.evaluate([0.2])
    from dataclasses import replace
    mu2 = np.zeros((2, 2, 3, 3))
    mu2[..., 2, 2] = np.array([[7.0, 1.0], [1.0, 5.0]])
    data2 = replace(data, dipole_squared=mu2, dipole_squared_gradients=np.zeros((2, 2, 3, 3, 1)))
    H = CavityHamiltonian(cav, "PF").matrix(project(data2, cav))
    assert np.isclose(H[0, 0], data.energies[0] + 0.5 * cav.coupling ** 2 * 7.0)
