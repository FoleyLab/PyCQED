import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycqed.cavity import CavityMode, project  # noqa: E402
from pycqed.models import DiabaticTwoStateModel  # noqa: E402


@pytest.fixture
def tully():
    """Two-state model with R-dependent dipoles and a non-zero intrinsic NAC."""
    return DiabaticTwoStateModel()


@pytest.fixture
def cavity():
    return CavityMode(omega=0.02, coupling=0.02, gamma=0.002, n_fock=3)


def matter_at(model, x, cav):
    return project(model.evaluate(np.atleast_1d(x)), cav)
