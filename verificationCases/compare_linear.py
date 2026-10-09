#!/usr/bin/env python3
"""Compare simulated J0-mode amplitude histories with the exact linear theory.

Usage::

    python verificationCases/compare_linear.py RUN_DIR [RUN_DIR ...] --out ANALYSIS_DIR

For each run, the simulated amplitude ``a(t)`` (the Hankel projection of the free surface on
``J0``) is compared with ``a0 Z(t)``, the exact linear response integrated from the same state
(``linear_theory/initial_value.py``), and with the Floquet theory (``linear_theory/faraday_linear``):

* unforced runs (``Gamma = 0``): the complex rate ``s`` from a damped-cosine fit, applied to the
  simulation and to the linear response over the same window, and the normal-mode root ``s``;
* forced runs: the Floquet exponent ``mu`` from ``a(t + T) = -exp(mu T) a(t)``, applied to both
  histories over the same window, and the Kumar-Tuckerman exponent ``mu(Gamma)``. A
  three-component Floquet fit (``floquet_multipliers``) is reported as a supplementary diagnostic,
  and forced runs that share mesh and time step give the neutral forcing by interpolation.

Each comparison is also made against the linear system advanced with the solver's own time scheme
(``initial_value.response_bdf2``); that difference contains only the spatial error.

It also reports the largest pointwise difference ``|a - a0 Z|/|a0|``, the relative volume drift and,
for runs with ``Bo = Gamma = 0``, the energy-budget residual
``[E(t) - E(0) + int_0^t D dt] / (A(0) - pi R_c^2)``, with ``E = kinetic energy + area`` (unit
surface tension, no gravity work when ``Bo = Gamma = 0``) and ``D`` the viscous dissipation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "linear_theory"))
sys.path.insert(0, str(HERE.parent / "postProcess"))

import faraday_linear as fl  # noqa: E402
import initial_value as iv  # noqa: E402
from linear_fits import fit_damped_cosine, floquet_exponent, floquet_multipliers  # noqa: E402

BOTTOM = "free-slip"


def read_table(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        header = fh.readline().lstrip("#").strip().split("\t")
    data = np.loadtxt(path, comments="#", ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def analyse(run: Path, out_dir: Path, fit_start_periods: float, fit_end_periods: float | None,
            allow_incomplete: bool) -> dict:
    params = json.loads((run / "parameters.json").read_text())
    status = json.loads((run / "run_status.json").read_text())
    nd = params["dimensionless"]
    case_id = params["case"]["case_id"]
    diag = read_table(run / "diagnostics.txt")
    t, a = diag["time"], diag["amp"]
    planned = params["time_stepping"]["t_end"]
    complete = status.get("state") == "completed" and t[-1] >= planned - 0.5 * params[
        "time_stepping"]["dt"]
    if not complete and not allow_incomplete:
        raise SystemExit(f"{run}: run {case_id!r} is not complete (state {status.get('state')!r}, "
                         f"t = {t[-1]:.6g} of {planned:.6g}); use --allow-incomplete to analyse it")
    Oh, Bo, d, Gamma, Omega = nd["Oh"], nd["Bo"], nd["depth"], nd["Gamma"], nd["Omega"]
    a0 = nd["a0"]
    T = 2 * math.pi / Omega

    Z = iv.response(t, Oh=Oh, Bo=Bo, d=d, bottom=BOTTOM, Gamma=Gamma, Omega=Omega)
    a_lin = a0 * Z
    # the same linear system advanced with the solver's time scheme: the difference is spatial
    dt = params["time_stepping"]["dt"]
    t_b, Z_b = iv.response_bdf2(dt, len(t) - 1, Oh=Oh, Bo=Bo, d=d, bottom=BOTTOM, Gamma=Gamma,
                                Omega=Omega)
    if not np.allclose(t_b, t, rtol=0.0, atol=1e-8 * max(1.0, t[-1])):
        raise SystemExit(f"{run}: output times are not the uniform grid n*dt")
    a_bdf2 = a0 * Z_b
    out = {
        "case_id": case_id,
        "run": str(run),
        "status": status.get("state"),
        "complete": bool(complete),
        "remesh_count": status.get("remesh_count"),
        "Oh": Oh, "Bo": Bo, "depth": d, "Gamma": Gamma, "Omega": Omega, "a0": a0,
        "mesh": params["mesh"],
        "dt": params["time_stepping"]["dt"],
        "steps_per_forcing_period": params["case"]["time_stepping"]["steps_per_forcing_period"],
        "t_end": float(t[-1]),
        "max_abs_diff_over_a0": float(np.max(np.abs(a - a_lin)) / abs(a0)),
        "max_abs_diff_bdf2_over_a0": float(np.max(np.abs(a - a_bdf2)) / abs(a0)),
        "axis_vs_projection_max_over_a0": float(
            np.max(np.abs((diag["z_axis"] - d) - a)) / abs(a0)),
        "volume_relative_drift": float(np.max(np.abs(diag["volume"] / diag["volume"][0] - 1))),
    }
    t_start = fit_start_periods * T
    t_end = None if fit_end_periods is None else fit_end_periods * T
    if Gamma == 0.0:
        s_th = fl.normal_mode(Oh, Bo, d, BOTTOM)
        kw = dict(t_start=t_start, t_end=t_end, omega_guess=s_th.imag, rate_guess=s_th.real)
        fit_sim = fit_damped_cosine(t, a, **kw)
        fit_lin = fit_damped_cosine(t, a_lin, **kw)
        fit_bdf2 = fit_damped_cosine(t, a_bdf2, **kw)
        out["normal_mode"] = {"s_real": s_th.real, "s_imag": s_th.imag}
        out["fit_simulation"] = fit_sim
        out["fit_linear_response"] = fit_lin
        out["fit_linear_response_bdf2"] = fit_bdf2
        out["rate_rel_error"] = fit_sim["s_real"] / fit_lin["s_real"] - 1
        out["frequency_rel_error"] = fit_sim["s_imag"] / fit_lin["s_imag"] - 1
        out["rate_rel_error_spatial"] = fit_sim["s_real"] / fit_bdf2["s_real"] - 1
        out["frequency_rel_error_spatial"] = fit_sim["s_imag"] / fit_bdf2["s_imag"] - 1
    if Gamma == 0.0 and Bo == 0.0:  # no body-force work: kinetic + surface energy is dissipated
        energy = diag["kinetic_energy"] + diag["area"]
        diss = diag["dissipation"]
        work = np.concatenate([[0.0], np.cumsum(0.5 * (diss[1:] + diss[:-1]) * np.diff(t))])
        excess = diag["area"][0] - math.pi * nd["radius"] ** 2
        out["energy_budget_max_residual"] = float(
            np.max(np.abs(energy - energy[0] + work)) / excess)
    if Gamma != 0.0:
        s_th = fl.normal_mode(Oh, Bo, d, BOTTOM)
        fold = fl.fold_point(Omega, 20, Oh, Bo, d, BOTTOM, s_th)
        mu_th = fl.mu_KT(Gamma, Omega, 20, Oh, Bo, d, BOTTOM, fold=fold)
        fit_sim = floquet_exponent(t, a, period=T, t_start=t_start, t_end=t_end)
        fit_lin = floquet_exponent(t, a_lin, period=T, t_start=t_start, t_end=t_end)
        out["Gamma_c"] = fl.gamma_c(Oh, Bo, d, BOTTOM, Omega, alpha=1)
        out["mu_floquet"] = mu_th
        out["fit_simulation"] = fit_sim
        out["fit_linear_response"] = fit_lin
        out["mu_abs_error"] = fit_sim["mu"] - fit_lin["mu"]
        # Supplementary diagnostic (not part of the acceptance criteria): a three-component Floquet
        # fit that separates the decaying companion mode which biases the single-mode estimate.
        modes_sim = floquet_multipliers(t, a, period=T, t_start=t_start, t_end=t_end)
        modes_lin = floquet_multipliers(t, a_lin, period=T, t_start=t_start, t_end=t_end)
        modes_bdf2 = floquet_multipliers(t, a_bdf2, period=T, t_start=t_start, t_end=t_end)
        out["modes_fit_simulation"] = modes_sim
        out["modes_fit_linear_response"] = modes_lin
        out["modes_fit_linear_response_bdf2"] = modes_bdf2
        out["modes_mu_error_vs_floquet"] = modes_sim["mu"] - mu_th
        out["modes_mu_error_spatial"] = modes_sim["mu"] - modes_bdf2["mu"]
    out["linear_response_file"] = f"{case_id}_linear_response.txt"
    np.savetxt(out_dir / out["linear_response_file"], np.column_stack([t, a, a_lin]),
               header="time  amp_simulation  amp_linear_response")
    return out


def thresholds(results: list[dict]) -> list[dict]:
    """Neutral forcing from forced runs that share mesh and time step, by linear interpolation.

    For each estimator, ``mu(Gamma)`` is interpolated to zero between the two bracketing amplitudes,
    for the simulations and for the exact linear response analysed identically; the exact
    ``Gamma_c`` is the Floquet value. Like-for-like comparison isolates the solver from the
    estimator and interpolation bias.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if r["Gamma"] != 0.0:
            m = r["mesh"]
            key = (m["h_interface"], m["h_axis"], m["h_max"], m["grading"], r["dt"])
            groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["Gamma"])
        G = np.array([r["Gamma"] for r in rs])
        entry = {"mesh_and_dt": list(key), "Gammas": G.tolist(), "Gamma_c_exact": rs[0]["Gamma_c"]}
        series = {
            "single_simulation": [r["fit_simulation"]["mu"] for r in rs],
            "single_linear_response": [r["fit_linear_response"]["mu"] for r in rs],
            "modes_simulation": [r["modes_fit_simulation"]["mu"] for r in rs],
            "modes_linear_response": [r["modes_fit_linear_response"]["mu"] for r in rs],
            "floquet_exact": [r["mu_floquet"] for r in rs],
        }
        for name, mu in series.items():
            mu = np.array(mu)
            idx = np.nonzero(np.diff(np.sign(mu)) != 0)[0]
            if idx.size == 0:
                entry[name] = None
                continue
            i = int(idx[0])
            entry[name] = float(G[i] - mu[i] * (G[i + 1] - G[i]) / (mu[i + 1] - mu[i]))
        if entry["single_simulation"] is not None and entry["single_linear_response"] is not None:
            entry["single_like_for_like_rel_error"] = (
                entry["single_simulation"] / entry["single_linear_response"] - 1)
        if entry["modes_simulation"] is not None:
            entry["modes_rel_error_vs_exact"] = entry["modes_simulation"] / entry["Gamma_c_exact"] - 1
        out.append(entry)
    return out


def plot(results: list[dict], stem: Path) -> None:
    import shutil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "serif"
    if shutil.which("latex"):
        matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        matplotlib.rcParams["mathtext.fontset"] = "cm"

    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, max(len(results), 2)))
    for res, c in zip(results, colours):
        t, a, a_lin = np.loadtxt(stem.parent / res["linear_response_file"], unpack=True)
        T = 2 * math.pi / res["Omega"]
        label = rf"$\Gamma={res['Gamma']:g}$, $h={res['mesh']['h_interface']:g}$, " \
                rf"$T/\Delta t={res['steps_per_forcing_period']}$"
        axes[0].plot(t / T, a / abs(res["a0"]), "-", color=c, lw=3, label=label)
        axes[0].plot(t / T, a_lin / abs(res["a0"]), "--", color="k", lw=1.5)
        axes[1].semilogy(t / T, np.abs(a - a_lin) / abs(res["a0"]) + 1e-16, "-", color=c, lw=3)
    axes[0].set_ylabel(r"$a/|a_0|$", fontsize=40, labelpad=15)
    axes[1].set_ylabel(r"$|a - a_0 Z|/|a_0|$", fontsize=40, labelpad=15)
    axes[1].set_xlabel(r"$t/T_f$", fontsize=40, labelpad=15)
    axes[0].legend(fontsize=22, frameon=False, loc="best")
    for ax in axes:
        ax.tick_params(which="both", direction="out", width=3, labelsize=30, pad=10)
        ax.tick_params(which="major", length=12)
        ax.tick_params(which="minor", length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(3)
    fig.tight_layout()
    # stems such as 'faraday_h0.025-n100' contain dots, so append rather than with_suffix()
    fig.savefig(stem.parent / f"{stem.name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(stem.parent / f"{stem.name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-start", type=float, default=4.0,
                    help="start of the fitting window, in forcing periods")
    ap.add_argument("--fit-end", type=float, default=None,
                    help="end of the fitting window, in forcing periods (default: end of run)")
    ap.add_argument("--name", default="linear_comparison")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="analyse runs that did not reach their planned end time")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = [analyse(run, args.out, args.fit_start, args.fit_end, args.allow_incomplete)
               for run in args.runs]
    summary = {"runs": results, "thresholds": thresholds(results)}
    (args.out / f"{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    plot(results, args.out / args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
