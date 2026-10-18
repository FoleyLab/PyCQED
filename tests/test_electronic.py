import numpy as np

from pycqed.electronic import CallableModel, align_phase
from pycqed.models import DoubleWellModel


def as_function(model, gradients):
    """Expose an analytic model through the dict interface an ML model would use."""
    def fn(R):
        d = model.evaluate(R)
        out = dict(E_g=d.energies[0], E_e=d.energies[1], mu_gg=d.dipoles[0, 0],
                   mu_ee=d.dipoles[1, 1], mu_eg=d.dipoles[1, 0], nac_ge=d.nac)
        if gradients:
            out.update(dE_g=d.energy_gradients[0], dE_e=d.energy_gradients[1],
                       dmu_gg=d.dipole_gradients[0, 0], dmu_ee=d.dipole_gradients[1, 1],
                       dmu_eg=d.dipole_gradients[1, 0])
        return out
    return fn


def test_analytic_models_gradients(tully):
    for model in (tully, DoubleWellModel()):
        for x in (-0.8, 0.05, 0.6):
            exact = model.evaluate([x])
            fd = CallableModel(as_function(model, gradients=False), fd_step=1e-5).evaluate([x])
            assert np.allclose(exact.energy_gradients, fd.energy_gradients, atol=1e-8)
            assert np.allclose(exact.dipole_gradients, fd.dipole_gradients, atol=1e-7)


def test_callable_uses_supplied_gradients(tully):
    calls = []

    def fn(R):
        calls.append(1)
        return as_function(tully, gradients=True)(R)

    d = CallableModel(fn).evaluate([0.3])
    assert len(calls) == 1
    assert np.allclose(d.energy_gradients, tully.evaluate([0.3]).energy_gradients)


def test_finite_difference_survives_random_phase(tully):
    rng = np.random.default_rng(0)
    base = as_function(tully, gradients=False)

    def flaky(R):
        out = base(R)
        if rng.random() < 0.5:
            out["mu_eg"] = -out["mu_eg"]
        return out

    fd = CallableModel(flaky, fd_step=1e-5).evaluate([0.2])
    exact = tully.evaluate([0.2])
    if np.dot(fd.dipoles[1, 0], exact.dipoles[1, 0]) < 0:
        fd = fd.flip_excited_phase()
    assert np.allclose(fd.dipole_gradients, exact.dipole_gradients, atol=1e-7)


def test_align_phase(tully):
    ref = tully.evaluate([0.10])
    new = tully.evaluate([0.11])
    same, flipped = align_phase(new, ref, [0.01])
    assert not flipped and same is new
    back, flipped = align_phase(new.flip_excited_phase(), ref, [0.01])
    assert flipped
    assert np.allclose(back.dipoles, new.dipoles) and np.allclose(back.nac, new.nac)


def test_align_phase_through_genuine_zero(tully):
    """mu_eg of the Tully model changes sign smoothly; that must not count as a flip."""
    xs = np.linspace(-1.0, 1.0, 801)
    ref, flips, signs = None, 0, []
    for a, b in zip(xs[:-1], xs[1:]):
        d, f = align_phase(tully.evaluate([b]), ref, [b - a] if ref is not None else None)
        flips += f
        ref = d
        signs.append(np.sign(d.dipoles[1, 0, 2]))
    assert flips == 0
    assert len(set(signs)) == 2
