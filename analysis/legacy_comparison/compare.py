"""Compare the pycqed and patched-legacy ensembles; writes figures + summary.json here.

    python analysis/legacy_comparison/compare.py RUN_DIR [LEGACY_DIR]

RUN_DIR holds ``new_*.npz`` (run_new.py) and ``legacy.npz`` / ``legacy_smooth.npz``
(run_legacy.py).  LEGACY_DIR (optional) is the patched scratch copy; its
``Data/legacy_chunk00_dc.txt`` / ``Data/legacy_smooth_chunk00_dc.txt`` spline tables are
used for the NAC figure.
"""

import json
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from pycqed.cli import build_cavity, load_model  # noqa: E402
from pycqed.constants import AU_TIME_TO_FS, HARTREE_TO_EV  # noqa: E402
from pycqed.io import scan_pes  # noqa: E402

STATES = ["G", "LP", "UP", "e1"]
# reference categorical slots 1-3 (validated all-pairs for three series)
STYLE = {
    "new": dict(color="#2a78d6", label="pycqed (Tully FSSH + quantum jumps, analytic NAC)"),
    "legacy": dict(color="#eb6834", label="legacy, patched A1-A5 + B1 (spline NAC, LAPACK gauge)"),
    "legacy_smooth": dict(color="#1baf7a", label="legacy, patched + smooth-gauge spline NAC"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_new(run):
    files = sorted(run.glob("new_*.npz"))
    d = [np.load(f) for f in files]
    t = d[0]["t"]
    hops = []
    for i, x in enumerate(d):
        for k, tt, s, dd in zip(x["event_kind"], x["event_t"], x["event_src"], x["event_dst"]):
            hops.append((i, str(k), float(tt), int(s), int(dd)))
    return dict(t=t, R=np.array([x["R"][:, 0] for x in d]), active=np.array([x["active"] for x in d]),
                pop=np.array([x["populations"] for x in d]), events=hops,
                counters={k: int(sum(int(x[f"count_{k}"]) for x in d)) for k in ("hops", "frustrated", "jumps", "phase_flips")})


def load_legacy(run, name):
    x = np.load(run / f"{name}.npz")
    dt = 0.12
    # hop log rows: (traj, step_counter, src, dst); step_counter k is the (k+1)-th step
    events = [(int(i), "hop" if dd != 0 else "sink", (int(k) + 1) * dt, int(s), int(dd)) for i, k, s, dd in x["hops"]]
    return dict(t=x["t"], R=x["R"], active=x["active"], pop=x["pop"], events=events,
                counters=dict(frustrated=int(x["frustrated"].sum())), ics=x["ics"])


def fractions(active):
    n = active.shape[0]
    f = np.stack([(active == s).mean(0) for s in range(4)], axis=-1)
    return f, np.sqrt(f * (1 - f) / n)


def hop_table(events, n):
    tab = {}
    for _, kind, _, s, d in events:
        key = f"{kind} {STATES[s]}->{STATES[d]}"
        tab[key] = tab.get(key, 0) + 1
    tab["total"] = len(events)
    tab["trajectories"] = n
    return dict(sorted(tab.items()))


def first_event_times(events, n, t_end):
    out = np.full(n, t_end)
    for i, kind, t, _, _ in events:
        if kind != "frustrated":  # a frustrated hop (frustrated="keep") changes nothing
            out[i] = min(out[i], t)
    return out


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main():
    run = Path(sys.argv[1])
    legacy_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    data = {"new": load_new(run), "legacy": load_legacy(run, "legacy")}
    if (run / "legacy_smooth.npz").exists():
        data["legacy_smooth"] = load_legacy(run, "legacy_smooth")
    n = data["new"]["R"].shape[0]
    t_fs = data["new"]["t"] * AU_TIME_TO_FS
    for k, v in data.items():
        assert np.allclose(v["t"], data["new"]["t"]), k

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelcolor": INK})

    # --- active-state fractions -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, s in zip(axes, (2, 1, 0)):
        for k, v in data.items():
            f, e = fractions(v["active"])
            ax.fill_between(t_fs, f[:, s] - e[:, s], f[:, s] + e[:, s], color=STYLE[k]["color"], alpha=0.15, lw=0)
            ax.plot(t_fs, f[:, s], color=STYLE[k]["color"], lw=2, label=STYLE[k]["label"])
        ax.set_title(f"fraction of trajectories on {STATES[s]}", color=INK)
        ax.set_xlabel("time (fs)")
        ax.set_ylim(-0.02, 1.02)
        style_axes(ax)
    axes[0].set_ylabel("active-state fraction")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(HERE / "active_fractions.png", dpi=150)
    plt.close(fig)

    # --- mean R -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, v in data.items():
        m, e = v["R"].mean(0), v["R"].std(0) / np.sqrt(n)
        ax.fill_between(t_fs, m - e, m + e, color=STYLE[k]["color"], alpha=0.15, lw=0)
        ax.plot(t_fs, m, color=STYLE[k]["color"], lw=2, label=STYLE[k]["label"])
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("<R> (bohr)")
    ax.set_title("ensemble-mean reaction coordinate (band = 1 s.e.m.)", color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(HERE / "mean_R.png", dpi=150)
    plt.close(fig)

    # --- surfaces and NAC ---------------------------------------------------------
    with open(ROOT / "examples" / "legacy_double_well.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    r = np.linspace(-1.25, 1.25, 2000)
    pes = scan_pes(load_model(cfg["model"]), build_cavity(cfg["cavity"]), r, coupling="Rabi")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw=dict(height_ratios=(1.3, 1)))
    for j, name in enumerate(STATES[:3]):
        a1.plot(r, pes["energies"][:, j].real * HARTREE_TO_EV, color=INK, lw=1.2)
        a1.annotate(name, (r[-1], pes["energies"][-1, j].real * HARTREE_TO_EV), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK)
    a1.axvspan(-0.7 - 0.06, -0.7 + 0.06, color=GRID, lw=0)
    a1.annotate("start (Wigner, 2 sigma)", (-0.7, 0.6), ha="center", color=MUTED, fontsize=8)
    a1.set_ylabel("Re E (eV)")
    a1.set_title("polaritonic surfaces (pycqed scan) and LP-UP coupling d_LP,UP", color=INK)
    style_axes(a1)
    a2.plot(r, pes["nac"][:, 1, 2].real, color=STYLE["new"]["color"], lw=2, label="pycqed analytic")
    if legacy_dir is not None:
        for k, fn in (("legacy", "legacy_chunk00_dc.txt"), ("legacy_smooth", "legacy_smooth_chunk00_dc.txt")):
            p = legacy_dir / "Data" / fn
            if p.exists():
                dc = np.loadtxt(p, dtype=complex)
                a2.plot(dc[:, 0].real, dc[:, 1].real, color=STYLE[k]["color"], lw=1.5,
                        ls="--" if k == "legacy_smooth" else "-",
                        label="legacy spline, smooth gauge" if k == "legacy_smooth" else "legacy spline, LAPACK gauge")
    a2.set_xlabel("R (bohr)")
    a2.set_ylabel("Re d_LP,UP (1/bohr)")
    style_axes(a2)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "surfaces_nac.png", dpi=150)
    plt.close(fig)

    # --- numbers ------------------------------------------------------------------
    summary = dict(n_traj=n, n_steps=int(round(data["new"]["t"][-1] / 0.12)), t_final_fs=float(t_fs[-1]))
    checkpoints = [int(np.argmin(abs(t_fs - x))) for x in (100, 250, 500, 750, t_fs[-1])]
    for k, v in data.items():
        f, e = fractions(v["active"])
        fe = first_event_times(v["events"], n, v["t"][-1])
        summary[k] = dict(
            hops=hop_table(v["events"], n), counters=v["counters"],
            fractions={f"{t_fs[c]:.0f} fs": {STATES[s]: round(float(f[c, s]), 3) for s in range(3)} for c in checkpoints},
            mean_R={f"{t_fs[c]:.0f} fs": round(float(v["R"][:, c].mean()), 4) for c in checkpoints},
            median_first_event_fs=float(np.median(fe) * AU_TIME_TO_FS),
            n_never_left_UP=int((v["active"] == 2).all(axis=1).sum()),
        )
    # trajectory-level agreement before the first stochastic event in either code
    fe_new = first_event_times(data["new"]["events"], n, data["new"]["t"][-1])
    fe_leg = first_event_times(data["legacy"]["events"], n, data["legacy"]["t"][-1])
    t = data["new"]["t"]
    dev = []
    for i in range(n):
        m = t < min(fe_new[i], fe_leg[i])
        if m.sum() > 1:
            dev.append(np.abs(data["new"]["R"][i, m] - data["legacy"]["R"][i, m]).max())
    summary["pre_event_R_max_abs_diff_bohr"] = dict(max=float(np.max(dev)), median=float(np.median(dev)), n=len(dev))
    diff = {}
    for k in data:
        if k == "new":
            continue
        fn, en = fractions(data["new"]["active"])
        fk, ek = fractions(data[k]["active"])
        z = np.abs(fn - fk)[:, :3] / np.sqrt(en ** 2 + ek ** 2 + 1e-12)[:, :3]
        diff[k] = dict(max_abs_fraction_diff={STATES[s]: round(float(np.abs(fn - fk)[:, s].max()), 3) for s in range(3)},
                       max_z={STATES[s]: round(float(z[:, s].max()), 1) for s in range(3)},
                       max_abs_meanR_diff=round(float(np.abs(data["new"]["R"].mean(0) - data[k]["R"].mean(0)).max()), 4))
    summary["new_minus"] = diff

    # --- source attribution diagnostics ---------------------------------------------
    diag = {}
    i100 = int(np.argmin(abs(t_fs - 100)))
    for k in data:
        if k == "new":
            ev = data[k]["events"]
            up = [e for e in ev if e[1] == "hop" and e[3] == 1 and e[4] == 2]
            diag[k] = dict(upward_LP_to_UP_hops=len(up), trajectories_with_upward_hop=len({e[0] for e in up}))
            continue
        x = np.load(run / f"{k}.npz")
        a, p = x["active"], x["pop"]
        norm = 1.0 - p[..., 0]          # PFSSH stores G as 1 - trace, so this is the surviving norm
        pa = np.take_along_axis(p, a[..., None], -1)[..., 0]
        enh = (norm / np.maximum(pa, 1e-12))[a > 0]
        sinks = [e for e in data[k]["events"] if e[1] == "sink"]
        stride_t = x["t"][1] - x["t"][0]
        Rs = np.array([x["R"][i, min(int(round(tt / stride_t)), x["R"].shape[1] - 1)] for i, _, tt, _, _ in sinks])
        near = (np.abs(Rs + 0.588) < 0.03) | (np.abs(Rs - 0.365) < 0.03)
        diag[k] = dict(
            mean_norm_at_100fs=round(float(norm[:, i100].mean()), 3),
            mean_norm_final=round(float(norm[:, -1].mean()), 3),
            sink_hops=len(sinks), sink_hops_first_100fs=int(sum(tt * AU_TIME_TO_FS < 100 for _, _, tt, _, _ in sinks)),
            sink_hops_within_0p03_bohr_of_a_crossing=int(near.sum()),
            sink_rate_over_jump_rate_N_over_p_active=dict(median=round(float(np.median(enh)), 2), mean=round(float(enh.mean()), 2)),
        )
    summary["diagnostics"] = diag

    # compact ensemble time series for re-plotting without re-running
    np.savez_compressed(HERE / "ensemble.npz", t_fs=t_fs,
                        **{f"{k}_R": v["R"].astype(np.float32) for k, v in data.items()},
                        **{f"{k}_active": v["active"].astype(np.int8) for k, v in data.items()},
                        **{f"{k}_events": np.array([(i, s, d, tt) for i, _, tt, s, d in v["events"]]) for k, v in data.items()},
                        **{f"{k}_event_kind": np.array([e[1] for e in v["events"]]) for k, v in data.items()})
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
