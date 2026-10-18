"""Fewest-switches surface hopping on (non-Hermitian) polaritonic surfaces.

One nuclear step of :meth:`FSSH.step`:

1. Velocity Verlet (or BBK Langevin) on the active polariton surface F = -Re grad E_active.
2. New electronic data from the model, phase-aligned to the previous step, then a new
   polariton solution tracked from the previous one.
3. Electronic wavefunction psi propagated in the |a(R)> x |n> basis over ``n_substeps``
   with H, v.dH and v.D interpolated linearly between the two geometries:
   psi <- expm((-i H - v.D) h) psi.  Fewest-switches probabilities are accumulated over
   the substeps from the polariton amplitudes c = UL psi:
   g(a -> k) += h * max(0, -2 Re[c_k^* (v.d)_ka c_a]) / |c_a|^2.
4. Photon loss as a quantum jump: with probability 1 - |psi|^2, psi <- b psi / |b psi| and
   the new active state is drawn from |c_J|^2 (no momentum rescaling, the photon carries
   the energy away); otherwise psi is renormalized.
5. If no jump occurred, at most one hop, with momentum rescaled along the (phase-rotated)
   NAC, or along P, to conserve Re E + KE.  Frustrated hops keep or reverse P along it.
6. Optional energy-based decoherence correction (Granucci-Persico).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .cavity import CavityMode, project
from .electronic import ElectronicData, ElectronicModel, align_phase
from .hamiltonian import CavityHamiltonian
from .polariton import PolaritonSolution, couplings_from_matrices, diagonalize, make_solver


@dataclass
class FSSHSettings:
    dt: float                          # nuclear time step (a.u.)
    n_substeps: int = 20               # electronic substeps per nuclear step
    rescale: str = "nac"               # "nac" | "velocity" | "none"
    frustrated: str = "keep"           # "keep" | "reverse"
    decoherence: float | None = None   # EDC constant C (Hartree), e.g. 0.1; None disables
    photon_loss: bool = True           # quantum jumps; False leaves psi un-normalized (norm = survival)
    friction: float = 0.0              # Langevin friction (1 / a.u. time)
    temperature: float = 0.0           # k_B T (Hartree)

    def __post_init__(self):
        if self.rescale not in ("nac", "velocity", "none"):
            raise ValueError(f"bad rescale option {self.rescale!r}")
        if self.frustrated not in ("keep", "reverse"):
            raise ValueError(f"bad frustrated option {self.frustrated!r}")


@dataclass
class TrajectoryState:
    t: float
    R: np.ndarray
    P: np.ndarray
    masses: np.ndarray
    psi: np.ndarray
    active: int
    electronic: ElectronicData
    solution: PolaritonSolution
    rng: np.random.Generator
    counters: dict = field(default_factory=lambda: dict(hops=0, frustrated=0, jumps=0, phase_flips=0))

    @property
    def amplitudes(self) -> np.ndarray:
        return self.solution.UL @ self.psi

    @property
    def populations(self) -> np.ndarray:
        p = np.abs(self.amplitudes) ** 2
        return p / p.sum()

    @property
    def kinetic(self) -> float:
        return float(np.sum(self.P ** 2 / (2.0 * self.masses)))

    @property
    def potential(self) -> float:
        return float(self.solution.energies[self.active].real)


class FSSH:
    """Surface-hopping engine for one electronic model in one cavity mode."""

    def __init__(self, model: ElectronicModel, cavity: CavityMode, settings: FSSHSettings,
                 coupling: str = "PF", solver: str = "numerical", dse: bool = True):
        self.model = model
        self.cavity = cavity
        self.settings = settings
        self.ham = CavityHamiltonian(cavity, coupling, dse=dse)
        self.solver = make_solver(self.ham, solver)

    # --- set-up -------------------------------------------------------------------------
    def evaluate(self, R, ref_data: ElectronicData | None = None,
                 prev: PolaritonSolution | None = None, dR=None):
        data, flipped = align_phase(self.model.evaluate(R), ref_data, dR)
        return data, self.solver.solve(project(data, self.cavity), prev), flipped

    def initialize(self, R, P, masses, active: int, rng=None, psi=None) -> TrajectoryState:
        R = np.array(R, dtype=float).reshape(-1)
        P = np.array(P, dtype=float).reshape(-1)
        masses = np.broadcast_to(np.asarray(masses, dtype=float), R.shape).copy()
        data, sol, _ = self.evaluate(R)
        if psi is None:
            psi = sol.U[:, active].copy()
        psi = np.asarray(psi, dtype=complex)
        psi = psi / np.linalg.norm(psi)
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        return TrajectoryState(0.0, R, P, masses, psi, active, data, sol, rng)

    # --- one step ----------------------------------------------------------------------
    def step(self, st: TrajectoryState) -> list:
        """Advance ``st`` by one nuclear step in place; returns a list of event dicts."""
        s = self.settings
        dt, m = s.dt, st.masses
        events = []

        def noise():
            if s.friction > 0.0 and s.temperature > 0.0:
                return np.sqrt(2.0 * s.friction * m * s.temperature / dt) * st.rng.standard_normal(m.size)
            return 0.0

        sol0, psi0, P0 = st.solution, st.psi, st.P
        F0 = sol0.forces()[st.active]
        P_half = P0 + 0.5 * dt * (F0 - s.friction * P0 + noise())
        R1 = st.R + dt * P_half / m
        data1, sol1, flipped = self.evaluate(R1, st.electronic, sol0, R1 - st.R)
        st.counters["phase_flips"] += int(flipped)
        F1 = sol1.forces()[st.active]
        P1 = (P_half + 0.5 * dt * (F1 + noise())) / (1.0 + 0.5 * dt * s.friction)

        psi1, g = self._propagate(sol0, sol1, P0 / m, P1 / m, psi0, st.active)
        st.R, st.P, st.electronic, st.solution, st.t = R1, P1, data1, sol1, st.t + dt

        jumped = False
        norm2 = float(np.vdot(psi1, psi1).real)
        if s.photon_loss:
            if self.cavity.gamma > 0.0 and st.rng.random() < 1.0 - norm2:
                jumped = True
                psi1 = self.ham.annihilation @ psi1
                old = st.active
                c = sol1.UL @ psi1
                p = np.abs(c) ** 2
                st.active = int(st.rng.choice(p.size, p=p / p.sum()))
                st.counters["jumps"] += 1
                events.append(dict(kind="jump", t=st.t, src=old, dst=st.active))
            psi1 = psi1 / np.linalg.norm(psi1)
        st.psi = psi1

        if not jumped:
            event = self._attempt_hop(st, g)
            if event is not None:
                events.append(event)
        if s.decoherence is not None:
            self._decohere(st)
        return events

    def _propagate(self, sol0, sol1, v0, v1, psi, active):
        """Substep propagation of psi; returns (psi, hop probabilities out of ``active``)."""
        n = self.settings.n_substeps
        h = self.settings.dt / n
        g = np.zeros(sol0.n_states)
        prev_UL = sol0.UL
        for j in range(n):
            x = (j + 0.5) / n
            H = (1 - x) * sol0.H + x * sol1.H
            v = (1 - x) * v0 + x * v1
            vdH = np.einsum("k,kij->ij", v, (1 - x) * sol0.dH + x * sol1.dH)
            vD = np.einsum("k,kij->ij", v, (1 - x) * sol0.D + x * sol1.D)
            E, U, UL = diagonalize(H, prev_UL)
            prev_UL = UL
            c = UL @ psi
            vd = couplings_from_matrices(E, UL @ vdH @ U, UL @ vD @ U)
            rho = abs(c[active]) ** 2
            if rho > 1e-14:
                flux = -2.0 * (c.conj() * vd[:, active] * c[active]).real
                flux[active] = 0.0
                g += h * np.maximum(flux, 0.0) / rho
            psi = expm((-1j * H - vD) * h) @ psi
        return psi, g

    def _attempt_hop(self, st: TrajectoryState, g: np.ndarray):
        u = st.rng.random()
        cum = np.cumsum(g)
        if cum[-1] <= u:
            return None
        target = int(np.searchsorted(cum, u, side="right"))
        sol, a = st.solution, st.active
        dE = float((sol.energies[target] - sol.energies[a]).real)
        mode = self.settings.rescale
        if mode == "none":
            st.active = target
            st.counters["hops"] += 1
            return dict(kind="hop", t=st.t, src=a, dst=target)

        m = st.masses
        if mode == "nac":
            d = sol.nac[a, target]
            eta = 0.5 * np.angle(np.sum(d * d))
            direction = (d * np.exp(-1j * eta)).real
        else:
            direction = st.P / m
        if not np.any(direction):
            direction = st.P / m
        A = float(np.sum(direction ** 2 / (2.0 * m)))
        B = float(np.sum(st.P * direction / m))
        disc = B * B - 4.0 * A * dE
        if disc < 0.0:
            st.counters["frustrated"] += 1
            if self.settings.frustrated == "reverse":
                st.P = st.P - (B / A) * direction
            return dict(kind="frustrated", t=st.t, src=a, dst=target)
        sgn = 1.0 if B >= 0.0 else -1.0
        alpha = (B - sgn * np.sqrt(disc)) / (2.0 * A)
        st.P = st.P - alpha * direction
        st.active = target
        st.counters["hops"] += 1
        return dict(kind="hop", t=st.t, src=a, dst=target)

    def _decohere(self, st: TrajectoryState):
        sol, a = st.solution, st.active
        c = sol.UL @ st.psi
        total = np.sum(np.abs(c) ** 2)
        if abs(c[a]) ** 2 < 1e-14:
            return
        ke = max(st.kinetic, 1e-12)
        gaps = np.abs((sol.energies - sol.energies[a]).real)
        tau = np.where(gaps > 0, (1.0 + self.settings.decoherence / ke) / np.maximum(gaps, 1e-300), np.inf)
        damp = np.exp(-self.settings.dt / tau)
        damp[a] = 1.0
        c = c * damp
        rest = np.sum(np.abs(np.delete(c, a)) ** 2)
        c[a] *= np.sqrt(max(total - rest, 0.0)) / abs(c[a])
        psi = sol.U @ c
        st.psi = psi * (np.linalg.norm(st.psi) / np.linalg.norm(psi))

    # --- driver ------------------------------------------------------------------------
    def run(self, st: TrajectoryState, n_steps: int, stride: int = 1):
        """Propagate ``n_steps`` and return a :class:`~pycqed.io.TrajectoryRecord`."""
        from .io import TrajectoryRecord

        rec = TrajectoryRecord(self.ham)
        rec.append(st)
        for i in range(1, n_steps + 1):
            rec.events.extend(self.step(st))
            if i % stride == 0:
                rec.append(st)
        rec.counters = dict(st.counters)
        return rec
