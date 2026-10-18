import numpy as np
import pytest

from pycqed.cavity import CavityMode
from pycqed.fssh import FSSH, FSSHSettings
from pycqed.io import scan_pes


def hermitian_engine(tully, **kw):
    cav = CavityMode(omega=0.02, coupling=0.02, gamma=0.0, n_fock=2)
    return FSSH(tully, cav, FSSHSettings(dt=1.0, **kw), coupling="PF")


def test_hermitian_limit_conserves_norm_and_energy(tully):
    eng = hermitian_engine(tully)
    hops = 0
    for seed in range(6):
        st = eng.initialize([-6.0], [20.0], 2000.0, active=2, rng=seed)
        rec = eng.run(st, 1000, stride=5)
        a = rec.arrays()
        hops += rec.counters["hops"]
        assert np.allclose(a["norm"], 1.0, atol=1e-10)
        assert np.ptp(a["total_energy"]) < 2e-6
    assert hops > 0


@pytest.mark.parametrize("rescale", ["nac", "velocity"])
def test_hop_rescaling_conserves_energy(tully, rescale):
    eng = hermitian_engine(tully, rescale=rescale)
    st = eng.initialize([0.0], [15.0], 2000.0, active=2, rng=0)
    before = st.kinetic + st.potential
    g = np.zeros(st.solution.n_states)
    g[1] = 1.0                                    # force a downward hop 2 -> 1
    event = eng._attempt_hop(st, g)
    assert event["kind"] == "hop" and st.active == 1
    assert np.isclose(st.kinetic + st.potential, before, rtol=0, atol=1e-12)


def test_frustrated_hop(tully):
    eng = hermitian_engine(tully, frustrated="reverse")
    st = eng.initialize([0.0], [0.5], 2000.0, active=0, rng=0)
    P0 = st.P.copy()
    g = np.zeros(st.solution.n_states)
    g[3] = 1.0                                    # far uphill
    event = eng._attempt_hop(st, g)
    assert event["kind"] == "frustrated" and st.active == 0
    assert np.allclose(st.P, -P0)


def test_photon_loss_cascades_down_fock_ladder(tully):
    """From UP2, a photon jump must land in the one-excitation manifold, not the ground state."""
    cav = CavityMode(omega=0.02, coupling=0.02, gamma=0.02, n_fock=2)
    eng = FSSH(tully, cav, FSSHSettings(dt=1.0, n_substeps=5), coupling="JC", solver="analytic")
    labels = eng.solver.labels
    firsts = []
    for seed in range(5):
        st = eng.initialize([-6.0], [0.0], 2000.0, active=labels.index("UP2"), rng=seed)
        rec = eng.run(st, 1500, stride=50)
        jumps = [e for e in rec.events if e["kind"] == "jump"]
        assert jumps
        firsts.append(labels[jumps[0]["dst"]])
        assert labels[st.active] == "G0"
        assert rec.arrays()["photon_number"][-1] < 1e-6
    assert all(f in ("LP1", "UP1") for f in firsts)


def test_random_phase_of_model_is_harmless(tully):
    rng_flip = np.random.default_rng(123)

    class Flaky:
        def evaluate(self, R):
            d = tully.evaluate(R)
            return d.flip_excited_phase() if rng_flip.random() < 0.5 else d

    cav = CavityMode(omega=0.02, coupling=0.02, gamma=0.001, n_fock=2)
    runs = []
    for model in (tully, Flaky()):
        eng = FSSH(model, cav, FSSHSettings(dt=1.0, n_substeps=10), coupling="PF")
        st = eng.initialize([-5.0], [20.0], 2000.0, active=2, rng=7)
        rec = eng.run(st, 800, stride=20)
        runs.append((rec.arrays(), rec.events, rec.counters))
    (a, ev_a, c_a), (b, ev_b, c_b) = runs
    assert c_b["phase_flips"] > 0
    assert np.allclose(a["R"], b["R"]) and np.array_equal(a["active"], b["active"])
    assert [e["kind"] for e in ev_a] == [e["kind"] for e in ev_b]


def test_langevin_noise_and_friction(tully):
    cav = CavityMode(omega=0.02, coupling=0.0, n_fock=1)
    damped = FSSH(tully, cav, FSSHSettings(dt=1.0, n_substeps=1, friction=1e-3), coupling="PF")
    st = damped.initialize([0.0], [20.0], 2000.0, active=0, rng=0)
    rec = damped.run(st, 200)
    assert np.all(np.diff(rec.arrays()["total_energy"]) < 0)
    hot = FSSH(tully, cav, FSSHSettings(dt=1.0, n_substeps=1, friction=1e-3, temperature=1e-3), coupling="PF")
    st2 = hot.initialize([0.0], [20.0], 2000.0, active=0, rng=0)
    hot.run(st2, 200)
    assert not np.isclose(st2.P[0], st.P[0])


def test_scan_pes(tully):
    cav = CavityMode(omega=0.02, coupling=0.02, gamma=0.001, n_fock=2)
    res = scan_pes(tully, cav, np.linspace(-2, 2, 101), coupling="Rabi")
    assert res["energies"].shape == (101, 6)
    assert np.all((res["photon"] >= -1e-12) & (res["photon"] <= 2 + 1e-12))
    slope_fd = np.gradient(res["energies"].real, res["path"][:, 0], axis=0)
    assert np.allclose(res["slope"].real[5:-5], slope_fd[5:-5], atol=2e-4)
