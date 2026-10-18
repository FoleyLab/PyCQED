"""Command-line driver: ``pycqed-fssh config.toml`` (or ``python scripts/run_fssh.py config.toml``).

See ``examples/*.toml`` for the configuration format.  Energies may be given in Hartree
(``omega``), eV (``omega_eV``) or meV (``omega_meV``); everything else is atomic units.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import tomllib
from pathlib import Path

import numpy as np

from .cavity import CavityMode
from .constants import EV_TO_HARTREE, MEV_TO_HARTREE
from .electronic import CallableModel
from .fssh import FSSH, FSSHSettings
from .sampling import wigner_1d, wigner_normal_modes


def _energy(section: dict, key: str, default=None):
    if key in section:
        return float(section[key])
    if key + "_eV" in section:
        return float(section[key + "_eV"]) * EV_TO_HARTREE
    if key + "_meV" in section:
        return float(section[key + "_meV"]) * MEV_TO_HARTREE
    if default is None:
        raise KeyError(f"missing {key!r} (or {key}_eV / {key}_meV)")
    return default


def load_model(cfg: dict):
    """``entry_point = "module:attr"``; a class is instantiated with ``kwargs``, a plain
    function is wrapped in :class:`CallableModel`."""
    module, _, attr = cfg["entry_point"].partition(":")
    obj = getattr(importlib.import_module(module), attr)
    kwargs = cfg.get("kwargs", {})
    if isinstance(obj, type):
        return obj(**kwargs)
    if hasattr(obj, "evaluate"):
        return obj
    return CallableModel(obj, fd_step=cfg.get("fd_step", 1e-4))


def build_cavity(cfg: dict) -> CavityMode:
    omega = _energy(cfg, "omega")
    extra = dict(polarization=tuple(cfg.get("polarization", (0.0, 0.0, 1.0))),
                 gamma=_energy(cfg, "gamma", 0.0), n_fock=int(cfg.get("n_fock", 1)))
    if "coupling" in cfg:
        return CavityMode(omega=omega, coupling=float(cfg["coupling"]), **extra)
    return CavityMode.from_coupling_strength(_energy(cfg, "g"), omega, float(cfg["mu_ref"]), **extra)


def build_engine(config: dict) -> FSSH:
    cav_cfg = config["cavity"]
    dyn = dict(config["dynamics"])
    dyn.pop("n_steps", None)
    settings = FSSHSettings(**dyn)
    return FSSH(load_model(config["model"]), build_cavity(cav_cfg), settings,
                coupling=cav_cfg.get("hamiltonian", "PF"), solver=cav_cfg.get("solver", "numerical"),
                dse=cav_cfg.get("dse", True))


def initial_conditions(cfg: dict, rng: np.random.Generator):
    kind = cfg.get("sampling", "fixed")
    if kind == "fixed":
        return np.asarray(cfg["R"], dtype=float), np.asarray(cfg.get("P", np.zeros(len(cfg["R"]))), dtype=float)
    if kind == "wigner_1d":
        R, P = wigner_1d(cfg["center"], cfg["masses"], cfg["force_constant"], rng)
        return np.array([R]), np.array([P])
    if kind == "wigner_hessian":
        return wigner_normal_modes(cfg["R"], cfg["masses"], np.load(cfg["hessian_file"]), rng)
    raise ValueError(f"unknown sampling {kind!r}")


def run(config: dict) -> list:
    engine = build_engine(config)
    init, runcfg = config["initial"], config.get("run", {})
    n_steps = int(config["dynamics"]["n_steps"])
    out_dir = Path(runcfg.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = runcfg.get("prefix", "traj")
    seeds = np.random.SeedSequence(runcfg.get("seed")).spawn(int(runcfg.get("n_traj", 1)))

    summary = []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        R, P = initial_conditions(init, rng)
        st = engine.initialize(R, P, init["masses"], int(init["active"]), rng)
        start = time.time()
        rec = engine.run(st, n_steps, stride=int(runcfg.get("stride", 1)))
        rec.save(out_dir / f"{prefix}_{i:04d}.npz")
        row = dict(traj=i, wall_s=round(time.time() - start, 2), R_final=st.R.tolist(),
                   active_final=st.active, **rec.counters)
        print(json.dumps(row), flush=True)
        summary.append(row)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="TOML configuration file")
    args = parser.parse_args(argv)
    with open(args.config, "rb") as fh:
        run(tomllib.load(fh))


if __name__ == "__main__":
    main()
