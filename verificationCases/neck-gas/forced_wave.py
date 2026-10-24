"""The gas traction on the free surface against exact linear theory (gates G1 and G2 of the neck
gas).

``run`` makes one finite-element case of the ``J0`` layer (``Oh``, depth ``d``, ``Bo = 0``,
``Gamma = 0``) with the pinned traction field (``FaradayProblem(gas=True)``) and steps it at a fixed
step with the traction set before every step, as the driver sets it:

* ``uniform``: ``T`` everywhere, from a flat layer at rest (G1: the liquid stays at rest and its
  pressure rises by ``T``) or from ``a0 J0(r)`` (G1: a uniform ``T`` does no work, so the free decay
  is that of ``T = 0``);
* ``j0``: ``T = eps J0(r) sin(omega t_n)`` at every node, from the state at the start of each step
  (lagged by one step, the staggered coupling), from a flat layer at rest (G2).

``gate`` runs the matrix of both gates in fresh processes and compares (``results.json``):

* G1: ``max |u|`` at rest, ``(p_T - p_0)/T - 1`` nodewise, and the largest difference of the
  amplitudes of the free decays with ``T = 100`` and ``T = 0`` over their largest amplitude;
* G2 at ``omega = omega0`` (resonance) and ``0.6 omega0``, two meshes and two steps: the amplitude
  ``a(t)`` against ``initial_value.forced_response_bdf2`` with the lag of the coupling (the time
  scheme's own response, so that the difference is the spatial error): the best-fit scale (within
  1 %) and the best-fit delay times ``omega`` (within 1e-3 rad); the delay of ``a(t)`` behind the
  response without the lag against the delay between the two theories (the predicted lag, one step,
  ``omega dt``), within 1e-3 rad, and its first order in ``dt``; and, for the record, the delay
  behind the exact continuous response (``forced_response``), which adds the scheme's dispersion.

Usage::

    python -I forced_wave.py gate --out DIR [--quick]
    python -I forced_wave.py run --out DIR --mode j0 --h 0.05 --dt 0.1257 --steps 200 \\
        --eps 1e-4 --omega 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "verificationCases" / "linear_theory"))
import numpy as np  # noqa: E402

OH, DEPTH = 0.04, 2.0 * math.pi


def run(args) -> dict:
    from scipy.special import j0

    from faraday_singularity.mesh import MeshSettings
    from faraday_singularity.parameters import CaseParameters, nondimensionalise
    from faraday_singularity.problem import FaradayProblem

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = sys.argv[:1]  # pyoomph parses the command line of its own (--h would be --help)
    phys = {"fluid": {"Oh": OH, "Bo": 0.0}, "container": {"mode": 1, "depth": DEPTH},
            "forcing": {"Gamma": 0.0, "detuning": 1.0, "phase": 0.0},
            "initial": {"amplitude": args.a0}}
    nd = nondimensionalise(CaseParameters.from_dict(phys))
    mesh = MeshSettings(h_interface=args.h, h_axis=args.h, h_max=0.4, grading=0.3,
                        interface_points=401)
    pb = FaradayProblem(nd, mesh=mesh, gas=True)
    pb.set_output_directory(str(out))
    rec = {"t": [], "amp": [], "args": vars(args)}

    def traction(t):
        if args.mode == "uniform":
            return float(args.T)
        return lambda r, z: args.eps * j0(r) * math.sin(args.omega * t)

    with pb:
        pb.initialise()
        rec["t"].append(0.0)
        rec["amp"].append(pb.observables()["amp"])
        t = 0.0
        for _ in range(args.steps):
            pb.set_gas_traction(traction(t))  # from the state at the start of the step
            pb.run(t + args.dt, timestep=args.dt, outstep=False, newton_solver_tolerance=1e-10)
            t = pb.get_current_time(dimensional=False, as_float=True)
            rec["t"].append(t)
            rec["amp"].append(pb.observables()["amp"])
        bulk = pb.get_mesh("liquid")
        fi = bulk.get_nodal_field_indices()
        p, u = [], []
        for n in bulk.nodes():
            u.append(math.hypot(n.value(fi["velocity_x"]), n.value(fi["velocity_y"])))
            if n.nvalue() > fi["pressure"]:
                p.append(n.value(fi["pressure"]))
        rec["pressure"] = p
        rec["u_max"] = max(u)
        r, z, T = pb.gas_traction()
        rec["traction_nodes"] = len(T)
        rec["traction_range"] = [float(np.min(T)), float(np.max(T))]
    (out / "result.json").write_text(json.dumps(rec))
    return rec


def _sub(out: Path, **kw) -> dict:
    """``run`` in a fresh process."""
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "run", "--out", str(out)]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(out.parent),
                          check=False, timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-3000:])
    return json.loads((out / "result.json").read_text())


def fit_delay(t, a, t_ref, a_ref, span: float) -> tuple[float, float, float]:
    """``(scale, delay, rms)`` of the best fit ``a(t) = scale a_ref(t - delay)`` (``a_ref`` a cubic
    spline through its samples), the delay within ``[-span, span]``."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize_scalar

    spl = CubicSpline(t_ref, a_ref)
    t, a = np.asarray(t), np.asarray(a)
    sel = t >= span + t_ref[0]

    def misfit(d):
        x = spl(t[sel] - d)
        s = float(np.dot(a[sel], x) / np.dot(x, x))
        return float(np.sum((a[sel] - s * x) ** 2)), s

    res = minimize_scalar(lambda d: misfit(d)[0], bounds=(-span, span), method="bounded",
                          options={"xatol": 1e-12 * max(span, 1.0)})
    err, s = misfit(res.x)
    return s, float(res.x), math.sqrt(err / max(1, int(sel.sum())))


def gate(args) -> dict:
    import initial_value as iv
    from faraday_linear import omega0

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    w0 = omega0(0.0, DEPTH)
    T0 = 2 * math.pi / w0
    quick = args.quick
    h_static = 0.1
    res = {"omega0": w0, "Oh": OH, "depth": DEPTH, "G1": {}, "G2": []}
    # G1: at rest under a uniform traction
    base = _sub(out / "g1-rest-T0", mode="uniform", T=0.0, a0=0.0, h=h_static, dt=0.05, steps=3)
    for T in (10.0, 100.0):
        r = _sub(out / f"g1-rest-T{T:g}", mode="uniform", T=T, a0=0.0, h=h_static, dt=0.05,
                 steps=3)
        dp = (np.array(r["pressure"]) - np.array(base["pressure"])) / T - 1.0
        res["G1"][f"rest_T{T:g}"] = {"u_max": r["u_max"], "u_max_T0": base["u_max"],
                                     "dp_over_T_minus_1_max": float(np.max(np.abs(dp))),
                                     "pass": bool(np.max(np.abs(dp)) <= 1e-10
                                                  and r["u_max"] <= 1e-10)}
    steps = 25 if quick else 100
    dec = {T: _sub(out / f"g1-decay-T{T:g}", mode="uniform", T=T, a0=0.01, h=h_static,
                   dt=T0 / 50, steps=steps) for T in (0.0, 100.0)}
    a0, a1 = np.array(dec[0.0]["amp"]), np.array(dec[100.0]["amp"])
    diff = float(np.max(np.abs(a1 - a0)) / np.max(np.abs(a0)))
    res["G1"]["decay_T100_vs_T0"] = {"max_rel_difference": diff, "steps": steps,
                                     "pass": bool(diff <= 1e-10)}
    # G2: the forced wave
    eps = 1e-4
    meshes = (0.1,) if quick else (0.05, 0.025)
    per = (25, 50) if quick else (50, 100)
    periods = 2.0 if quick else 4.0
    for fw in (1.0, 0.6):
        omega = fw * w0
        forcing = (lambda w: (lambda t: eps * math.sin(w * t)))(omega)
        for h in meshes:
            for n in per:
                dt = T0 / n
                steps = int(round(periods * n))
                r = _sub(out / f"g2-w{fw:g}-h{h:g}-n{n}", mode="j0", eps=eps, omega=omega, a0=0.0,
                         h=h, dt=dt, steps=steps)
                t, a = np.array(r["t"]), np.array(r["amp"])
                _, z_lag = iv.forced_response_bdf2(dt, steps, Oh=OH, Bo=0.0, d=DEPTH,
                                                   forcing=forcing, lag=1)
                _, z_now = iv.forced_response_bdf2(dt, steps, Oh=OH, Bo=0.0, d=DEPTH,
                                                   forcing=forcing, lag=0)
                tt = np.linspace(0.0, t[-1], 20 * steps + 1)
                z_ex = iv.forced_response(tt, Oh=OH, Bo=0.0, d=DEPTH, forcing=forcing)
                span = 3.0 * dt
                s_lag, d_lag, rms_lag = fit_delay(t, a, t, z_lag, span)
                s_now, d_now, _ = fit_delay(t, a, t, z_now, span)
                s_th, d_th, _ = fit_delay(t, z_lag, t, z_now, span)
                s_ex, d_ex, _ = fit_delay(t, a, tt, z_ex, span)
                row = {"omega_over_omega0": fw, "h": h, "steps_per_period": n, "dt": dt,
                       "steps": steps, "max_abs_amp": float(np.max(np.abs(a))),
                       "max_rel_diff_vs_lagged_theory": float(np.max(np.abs(a - z_lag))
                                                              / np.max(np.abs(z_lag))),
                       "scale_vs_lagged": s_lag, "phase_vs_lagged": omega * d_lag,
                       "rms_vs_lagged": rms_lag,
                       "lag_phase": omega * d_now, "predicted_lag_phase": omega * d_th,
                       "omega_dt": omega * dt, "omega_dt_half": 0.5 * omega * dt,
                       "lag_over_dt": d_now / dt, "predicted_lag_over_dt": d_th / dt,
                       "scale_vs_exact": s_ex, "phase_vs_exact": omega * d_ex,
                       "lag_theory_scale": s_th}
                row["pass_amplitude"] = bool(abs(s_lag - 1.0) <= 0.01)
                row["pass_phase"] = bool(abs(omega * d_lag) <= 1e-3)
                row["pass_lag"] = bool(abs(omega * (d_now - d_th)) <= 1e-3)
                res["G2"].append(row)
    # the lag is first order in dt: its delay over dt the same at both steps
    for fw in (1.0, 0.6):
        for h in meshes:
            rows = [x for x in res["G2"] if x["omega_over_omega0"] == fw and x["h"] == h]
            if len(rows) == 2:
                a_, b_ = rows
                res.setdefault("G2_order", []).append({
                    "omega_over_omega0": fw, "h": h,
                    "lag_ratio": a_["lag_phase"] / b_["lag_phase"],
                    "dt_ratio": a_["dt"] / b_["dt"],
                    "pass": bool(abs(a_["lag_phase"] / b_["lag_phase"] - a_["dt"] / b_["dt"])
                                 <= 0.05 * a_["dt"] / b_["dt"])})
    res["G1_pass"] = all(v["pass"] for v in res["G1"].values())
    res["G2_pass"] = all(x["pass_amplitude"] and x["pass_phase"] and x["pass_lag"]
                         for x in res["G2"]) and all(x["pass"] for x in res.get("G2_order", []))
    (out / "results.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=("uniform", "j0"), required=True)
    r.add_argument("--T", type=float, default=0.0)
    r.add_argument("--eps", type=float, default=0.0)
    r.add_argument("--omega", type=float, default=1.0)
    r.add_argument("--a0", type=float, default=0.0)
    r.add_argument("--h", type=float, default=0.05)
    r.add_argument("--dt", type=float, required=True)
    r.add_argument("--steps", type=int, required=True)
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    g.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
        return 0
    res = gate(args)
    print(json.dumps({k: res[k] for k in ("G1_pass", "G2_pass")}))
    return 0 if res["G1_pass"] and res["G2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
