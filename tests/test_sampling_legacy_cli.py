import numpy as np
import pytest

from conftest import ROOT
from pycqed.cavity import CavityMode, project
from pycqed.cli import run
from pycqed.hamiltonian import CavityHamiltonian
from pycqed.models import DoubleWellModel
from pycqed.sampling import wigner_1d, wigner_normal_modes


def test_wigner_widths():
    M, k = 1009883.0, 0.31246871517560126
    a = np.sqrt(k * M)
    R, P = wigner_1d(-0.7, M, k, rng=0, size=40000)
    assert np.isclose(R.std(), np.sqrt(1 / (2 * a)), rtol=0.02)
    assert np.isclose(P.std(), np.sqrt(a / 2), rtol=0.02)
    Rs, Ps = zip(*(wigner_normal_modes([-0.7], [M], np.array([[k]]), rng=i) for i in range(4000)))
    assert np.isclose(np.std(Ps), np.sqrt(a / 2), rtol=0.05)


def test_legacy_eigenvalues_reproduced(monkeypatch):
    """The Rabi / n_fock = 1 limit reproduces the legacy ARCHIVE/polaritonic.py Hamiltonian."""
    monkeypatch.syspath_prepend(str(ROOT / "ARCHIVE"))
    import polaritonic as legacy_mod
    omc, gc, gam = 2.45 / 27.211, 0.02 / 27.211, 1.0e-3 / 27.211
    legacy = legacy_mod.polaritonic({"Number_of_Photons": 1, "Photon_Energys": [omc],
                                     "Coupling_Strengths": [gc], "Photon_Lifetimes": [gam]})
    cav = CavityMode.from_coupling_strength(gc, omc, 1.0, gamma=gam, n_fock=1)
    ham, model = CavityHamiltonian(cav, "Rabi"), DoubleWellModel()
    for x in np.linspace(-1.2, 1.2, 13):
        legacy.R = x
        legacy.H_e()
        H_old = legacy.H_electronic + legacy.H_photonic + legacy.H_interaction
        old = np.sort_complex(np.linalg.eigvals(H_old))
        new = np.sort_complex(np.linalg.eigvals(ham.matrix(project(model.evaluate([x]), cav))))
        assert np.allclose(old, new, atol=1e-12)


def test_cli_runs(tmp_path):
    config = {
        "model": {"entry_point": "examples.mock_ml_model:predict"},
        "cavity": {"hamiltonian": "PF", "omega_eV": 2.85, "gamma_meV": 5.0, "coupling": 0.02, "n_fock": 2},
        "dynamics": {"dt": 2.0, "n_steps": 5, "n_substeps": 4},
        "initial": {"sampling": "fixed", "R": [0, 0, 0, 0, 0, 1.95], "masses": [1837.0] * 3 + [34635.0] * 3,
                    "active": 2},
        "run": {"n_traj": 2, "seed": 1, "output_dir": str(tmp_path), "prefix": "t"},
    }
    summary = run(config)
    assert len(summary) == 2
    data = np.load(tmp_path / "t_0001.npz")
    assert data["R"].shape == (6, 6)
