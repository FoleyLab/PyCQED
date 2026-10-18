"""Trajectory recording and potential-energy-surface scans."""

from __future__ import annotations

import numpy as np

from .cavity import CavityMode, project
from .electronic import ElectronicModel, align_phase
from .hamiltonian import CavityHamiltonian
from .polariton import make_solver


class TrajectoryRecord:
    """Time series collected by :meth:`pycqed.fssh.FSSH.run`."""

    FIELDS = ("t", "R", "P", "active", "energies", "populations", "norm",
              "photon_number", "kinetic", "total_energy")

    def __init__(self, hamiltonian: CavityHamiltonian):
        self.ham = hamiltonian
        self.rows = {k: [] for k in self.FIELDS}
        self.events = []
        self.counters = {}

    def append(self, st):
        psi = st.psi
        norm2 = float(np.vdot(psi, psi).real)
        r = self.rows
        r["t"].append(st.t)
        r["R"].append(st.R.copy())
        r["P"].append(st.P.copy())
        r["active"].append(st.active)
        r["energies"].append(st.solution.energies.real.copy())
        r["populations"].append(st.populations)
        r["norm"].append(norm2)
        r["photon_number"].append(float(np.vdot(psi, self.ham.number @ psi).real) / norm2)
        r["kinetic"].append(st.kinetic)
        r["total_energy"].append(st.kinetic + st.potential)

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.rows.items()}

    def save(self, path):
        """Write all time series, events and counters to a compressed ``.npz``."""
        ev = self.events
        np.savez_compressed(
            path, **self.arrays(),
            event_kind=np.array([e["kind"] for e in ev], dtype="U10"),
            event_t=np.array([e["t"] for e in ev]),
            event_src=np.array([e["src"] for e in ev], dtype=int),
            event_dst=np.array([e["dst"] for e in ev], dtype=int),
            **{f"count_{k}": v for k, v in self.counters.items()},
        )


def scan_pes(model: ElectronicModel, cavity: CavityMode, path, coupling: str = "PF",
             solver: str = "numerical", dse: bool = True) -> dict:
    """Polaritonic surfaces along a path of geometries (replaces legacy ``Write_PES``).

    ``path`` has shape (npts, ndof) (or (npts,) for 1-D models).  States are tracked by
    overlap and the eigenvector gauge is kept smooth.  Returns arrays:

    * ``energies``  (npts, n) complex
    * ``slope``     (npts, n)    t . grad E, with t the unit path tangent
    * ``nac``       (npts, n, n) t . d_JK
    * ``photon``    (npts, n)    <n> of each polariton
    """
    path = np.asarray(path, dtype=float)
    if path.ndim == 1:
        path = path[:, None]
    ham = CavityHamiltonian(cavity, coupling, dse=dse)
    solve = make_solver(ham, solver).solve
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    out = {k: [] for k in ("energies", "slope", "nac", "photon")}
    data_prev, sol, R_prev = None, None, path[0]
    for R, t in zip(path, tangent):
        data, _ = align_phase(model.evaluate(R), data_prev, R - R_prev)
        sol = solve(project(data, cavity), sol)
        data_prev, R_prev = data, R
        out["energies"].append(sol.energies)
        out["slope"].append(sol.grad_E @ t)
        out["nac"].append(sol.nac @ t)
        out["photon"].append(ham.photon_content(sol.U))
    res = {k: np.asarray(v) for k, v in out.items()}
    res["path"] = path
    res["labels"] = list(sol.labels)
    return res
