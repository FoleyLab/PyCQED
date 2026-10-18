"""Electronic amplitudes for one passage through the R = -0.588 LP/UP crossing at fixed velocity.

No hopping and no nuclear feedback: R(t) = R0 + v t, start in UP.  The same path is fed to
(i) the legacy RK4 (C' = -i H C - v d C with the spline tables of Write_PES), once with the
LAPACK-gauge NAC and once with the smooth-gauge NAC, and (ii) pycqed's own substep
propagator.  Prints the populations after the passage and the Landau-Zener estimate.

    python analysis/legacy_comparison/single_passage.py LEGACY_DIR
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycqed.cli import build_engine  # noqa: E402

DT, R0, R1 = 0.12, -0.66, -0.52
VELOCITIES = (3e-5, 6e-5, 9e-5)


def legacy_run(data_dir, tag, v):
    pes = np.loadtxt(data_dir / f"{tag}_pes.txt", dtype=complex)
    dc = np.loadtxt(data_dir / f"{tag}_dc.txt", dtype=complex)
    E = [Spline(pes[:, 0].real, pes[:, j].real, k=3) for j in range(1, 5)]
    G = [Spline(pes[:, 0].real, pes[:, j].imag, k=3) for j in range(1, 5)]
    d23 = [Spline(dc[:, 0].real, f(dc[:, 1]), k=3) for f in (np.real, np.imag)]
    d32 = [Spline(dc[:, 0].real, f(dc[:, 2]), k=3) for f in (np.real, np.imag)]
    C = np.array([0, 0, 1, 0], dtype=complex)
    R = R0
    while R < R1:
        R += v * DT
        H = np.diag([E[j](R) + 1j * G[j](R) for j in range(4)])
        d = np.zeros((4, 4), dtype=complex)
        d[1, 2] = d23[0](R) + 1j * d23[1](R)
        d[2, 1] = d32[0](R) + 1j * d32[1](R)
        f = lambda c: -1j * DT * H @ c - DT * v * d @ c  # noqa: E731  (legacy RK4_NH_SE)
        k1 = f(C); k2 = f(C + k1 / 2); k3 = f(C + k2 / 2); k4 = f(C + k3)  # noqa: E702
        C = C + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.abs(C) ** 2


def new_run(v):
    with open(ROOT / "examples" / "legacy_double_well.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    eng = build_engine(cfg)
    m = cfg["initial"]["masses"]
    st = eng.initialize([R0], [m * v], m, 2)
    data, sol, psi = st.electronic, st.solution, st.psi
    R = R0
    while R < R1:
        R += v * DT
        data, sol1, _ = eng.evaluate(np.array([R]), data, sol, np.array([v * DT]))
        psi, _ = eng._propagate(sol, sol1, np.array([v]), np.array([v]), psi, 2)
        sol = sol1
    return np.abs(sol.UL @ psi) ** 2


def landau_zener(v):
    """P(diabatic) = exp(-2 pi V12^2 / (v |d(E_e - E_g)/dR|)) at the crossing."""
    with open(ROOT / "examples" / "legacy_double_well.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    eng = build_engine(cfg)
    r = np.linspace(-0.62, -0.56, 6001)
    gaps = []
    for x in r:
        dat = eng.model.evaluate([x])
        gaps.append(dat.energies[1] - dat.energies[0] - eng.cavity.omega)
    gaps = np.array(gaps)
    i = np.argmin(abs(gaps))
    slope = abs(np.gradient(gaps, r)[i])
    V12 = eng.cavity.bilinear_prefactor * 1.0
    return np.exp(-2 * np.pi * V12 ** 2 / (v * slope)), r[i]


def main():
    data_dir = Path(sys.argv[1]) / "Data"
    print(f"{'v (au)':>8} {'LZ P_diab':>10} | {'code':<26} {'G':>7} {'LP':>7} {'UP':>7} {'norm':>7}")
    for v in VELOCITIES:
        pz, rc = landau_zener(v)
        rows = [("legacy, LAPACK-gauge NAC", legacy_run(data_dir, "legacy_chunk00", v)),
                ("legacy, smooth-gauge NAC", legacy_run(data_dir, "legacy_smooth_chunk00", v)),
                ("pycqed", new_run(v))]
        for name, p in rows:
            # legacy G population is 1 - trace in PFSSH; here report the raw amplitudes + norm
            print(f"{v:8.0e} {pz:10.3f} | {name:<26} {p[0]:7.4f} {p[1]:7.4f} {p[2]:7.4f} {p.sum():7.4f}")
    print(f"(diabatic crossing at R = {rc:.4f}; P_diab is the LZ probability to end on LP)")


if __name__ == "__main__":
    main()
