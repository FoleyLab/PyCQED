"""Run the pycqed ensemble for the legacy double-well scenario, in parallel.

Each trajectory i uses exactly the random stream that ``pycqed.cli.run`` would give it
(SeedSequence(seed).spawn(n_traj)[i]), so the results are those of
``pycqed-fssh examples/legacy_double_well.toml`` with ``n_traj``/``n_steps`` changed,
just spread over several processes.  The Wigner initial conditions are also written
to ``ics.txt`` (R, P per line) for the legacy driver.

    python analysis/legacy_comparison/run_new.py --n-traj 100 --n-steps 400000 --out DIR
"""

import argparse
import json
import sys
import time
import tomllib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pycqed.cli import build_engine, initial_conditions  # noqa: E402


def load_config(n_traj, n_steps):
    with open(ROOT / "examples" / "legacy_double_well.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["dynamics"]["n_steps"] = n_steps
    cfg["run"]["n_traj"] = n_traj
    return cfg


def child_rngs(cfg):
    seeds = np.random.SeedSequence(cfg["run"].get("seed")).spawn(int(cfg["run"]["n_traj"]))
    return [np.random.default_rng(s) for s in seeds]


def run_one(args):
    i, cfg, out = args
    rng = child_rngs(cfg)[i]
    R, P = initial_conditions(cfg["initial"], rng)
    engine = build_engine(cfg)
    st = engine.initialize(R, P, cfg["initial"]["masses"], int(cfg["initial"]["active"]), rng)
    start = time.time()
    rec = engine.run(st, int(cfg["dynamics"]["n_steps"]), stride=int(cfg["run"]["stride"]))
    rec.save(Path(out) / f"new_{i:04d}.npz")
    return dict(traj=i, wall_s=round(time.time() - start, 1), R_final=float(st.R[0]),
                active_final=st.active, **rec.counters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=400000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.n_traj, a.n_steps)

    ics = []
    for rng in child_rngs(cfg):
        R, P = initial_conditions(cfg["initial"], rng)
        ics.append((R[0], P[0]))
    np.savetxt(a.out / "ics.txt", np.array(ics), fmt="%.17e")

    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(run_one, [(i, cfg, a.out) for i in range(a.n_traj)]):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
