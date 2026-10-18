"""Run the patched legacy PFSSH.py on the matched initial conditions, in parallel.

``--legacy-dir`` is a scratch copy of ARCHIVE/polaritonic.py / ARCHIVE/PFSSH.py with
``legacy_fixes.patch`` applied (never the files in ARCHIVE/ themselves).  The
ensemble is split into chunks, one PFSSH.py process per chunk, and merged into
``legacy[_smooth].npz`` in ``--out``.

    python analysis/legacy_comparison/run_legacy.py --legacy-dir SCRATCH/legacy \
        --ics DIR/ics.txt --n-steps 400000 --out DIR [--smooth-gauge]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# python PFSSH.py Scale_True Real_DC N gamma_meV omega_eV g_eV prefix  (ARCHIVE/run.sh / the brief)
SCENARIO = ["Scale_True", "Real_DC", None, "1.0", "2.45", "0.02"]


def run_chunk(legacy_dir, ics, n_steps, out, tag, seed, smooth):
    ic_file = out / f"{tag}_ics.txt"
    np.savetxt(ic_file, ics, fmt="%.17e")
    args = list(SCENARIO)
    args[2] = str(len(ics))
    # PFSSH.py loops i in range(1, N_time): N_time = n_steps + 1 gives n_steps steps
    cmd = [sys.executable, "PFSSH.py", *args, tag, str(ic_file), str(n_steps + 1),
           str(out / f"{tag}.npz"), str(seed)]
    env = dict(os.environ, LEGACY_SMOOTH_GAUGE="1" if smooth else "0")
    with open(out / f"{tag}.log", "w") as log:
        subprocess.run(cmd, cwd=legacy_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out / f"{tag}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", type=Path, required=True)
    ap.add_argument("--ics", type=Path, required=True)
    ap.add_argument("--n-steps", type=int, default=400000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smooth-gauge", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.legacy_dir / "Data").mkdir(exist_ok=True)

    name = "legacy_smooth" if a.smooth_gauge else "legacy"
    ics = np.loadtxt(a.ics, ndmin=2)
    chunks = np.array_split(np.arange(len(ics)), a.workers)
    with ThreadPoolExecutor(a.workers) as pool:
        files = list(pool.map(
            lambda c: run_chunk(a.legacy_dir, ics[c[1]], a.n_steps, a.out, f"{name}_chunk{c[0]:02d}",
                                a.seed + c[0], a.smooth_gauge),
            [(k, idx) for k, idx in enumerate(chunks) if len(idx)]))

    parts = [np.load(f) for f in files]
    offset, hops = 0, []
    for p in parts:
        h = p["hops"].copy()
        h[:, 0] += offset
        hops.append(h)
        offset += len(p["R"])
    np.savez_compressed(a.out / f"{name}.npz", t=parts[0]["t"],
                        **{k: np.concatenate([p[k] for p in parts]) for k in ("R", "V", "active", "pop", "frustrated", "ics")},
                        hops=np.concatenate(hops))
    print(f"wrote {a.out / (name + '.npz')}")


if __name__ == "__main__":
    main()
