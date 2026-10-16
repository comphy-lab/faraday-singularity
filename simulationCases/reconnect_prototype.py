#!/usr/bin/env python3
"""Reconnection prototype: load a landing state, cut the neck, project, and take a few steps.

Example::

    .venv-pyoomph-f912ac10-py313/bin/python simulationCases/reconnect_prototype.py CASE.json \\
        --state RUN/states/stop_00001379.dump --outdir OUT --delta-ratio 1.05 --steps 20

``CASE.json`` is the ``pyoomph-case-v1`` file of the run that wrote the state, or one derived from
it in the keys a restart may change and in its ``reconnect`` block
(``run_case.PROTOTYPE_DERIVED_KEYS``: the name, the description ``completion.value``, and the
collapse keys ``keep_stop_state``, ``landing``, ``neck_stop`` and ``reconnect``); the physics, mesh
and every other collapse setting must be the ones the state was computed with. The run that wrote
the state is the directory above ``states/``: its ``run-manifest.json`` must be a normalised one
with a clean source commit, and its case is compared with ``CASE.json``
(``run_case.restart_record``); the manifest of the prototype records that lineage. Its
``collapse.reconnect`` block, if any, gives the defaults of the cut, which the options below
override. The state is a driver dump with its JSON beside it, normally a landing state: one a run
with ``collapse.landing`` keeps for a cut radius, or the state a run with
``collapse.keep_stop_state`` keeps at its neck-radius stop. The cut radius is ``--delta`` or
``--delta-ratio`` times the neck radius of the state; it must lie between ``min_landing_ratio`` and
``max_landing_ratio`` times that radius, so a study in ``delta`` needs a landing state for each
``delta``. A state written after a cut is refused: one cut per run, and continuing a post-cut state
is not supported yet.

The cut (``faraday_singularity.surgery.cut``) writes ``cut_diagnostics.json`` and keeps the
``precut``/``postcut`` states under ``states/``; a refused or failed cut writes the partial record
there instead, with the stage reached (``failed_at``), and ``run_status.json`` names it. The run
then ends: the problem is not restored, since ``states/precut_<delta>.dump`` holds the state before
the cut. Then ``--steps`` steps follow, the first ``dt0``
long and each at most ``growth`` times the last, and at most ``courant`` times the smallest element
of the free surface near the cut (``h_cut``: the cut's tip size, or the elements of both chains
there if smaller, as where the focus of the run is finer) over the largest free-surface speed
there. After every step ``steps.txt`` gets the time after the cut ``tau = t - t_cut`` and after the
pinch ``tau_p = t - t_p`` (``t_p`` fitted to the neck samples of the landing state, ``pinch`` in
``cut_diagnostics.json``; NaN when the state carries none; a study in ``delta`` compares at one
common ``t_p``, since fits over different windows differ), the budgets (volume, kinetic energy,
axial momentum ``P_z``), the three tips on the axis (bubble bottom and top, cavity tip) with their
axial speeds, the bridge length, the bubble volume and the open chain's jet measurements;
``profiles/postcut_<step>_open.txt`` and ``..._bubble.txt`` hold both chains (``r z u_r u_z``).
The run ends after ``--steps``, at ``tau_end`` after the cut, or at a stop the mesh cannot follow,
checked right after the cut and after every step (``reconnect.post_cut_stops``: the bridge
closing, the chains touching, the bubble's tips meeting, the bubble draining, a new neck narrower
than ``delta`` on either chain, a fold on the bubble; and the collapse driver's neck and fold stops
on the open chain). ``run_status.json`` records how it ended. This is a development tool, not a
production runner: it does not admit restarts through ``run_case.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_case  # noqa: E402  (sets the MPI transport before pyoomph is imported)
from faraday_singularity.driver import FOLD_SEARCH, CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.provenance import (component_commit, create_run_manifest,  # noqa: E402
                                            pyoomph_revision)
from faraday_singularity.reconnect import (Chain, CutGeometry, ReconnectSettings,  # noqa: E402
                                           neck_radius, post_cut_stops, revolved_volume,
                                           split_collapse_block)

COLUMNS = ("step", "time", "tau", "tau_p", "dt", "newton_tol", "volume", "kinetic_energy",
           "momentum_z", "divergence_sq", "bubble_volume", "z_bubble_bottom", "w_bubble_bottom",
           "z_bubble_top", "w_bubble_top", "z_cavity_tip", "w_cavity_tip", "bridge", "bubble_r_max",
           "bubble_z_r_max", "axis_curvature", "r_j", "z_b", "Q_j", "V_j", "h_near", "U",
           "h_cut", "U_cut", "remesh_count", "wall")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    size = ap.add_mutually_exclusive_group(required=True)
    size.add_argument("--delta", type=float, help="cut radius")
    size.add_argument("--delta-ratio", type=float, help="cut radius over the state's neck radius")
    ap.add_argument("--steps", type=int, default=20)
    for name in ("fill", "restart"):
        ap.add_argument("--" + name)
    for name in ("dt0", "dt_growth", "courant", "tau_end", "cap_window_factor",
                 "cap_splice_factor"):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=float)
    ap.add_argument("--curvature-sizing", dest="curvature_sizing", type=int)
    ap.add_argument("--check", action="store_true", help="validate the inputs and stop")
    args, _ = ap.parse_known_args(argv)
    return args


def settings_from(case: dict, args) -> tuple[CollapseSettings, dict, float]:
    period = nondimensionalise(CaseParameters.from_dict(case["physics"])).forcing_period
    block, recon, _keep, _landing = split_collapse_block(case["collapse"])
    if "snapshot_dt_periods" in block:
        block["snapshot_dt"] = block.pop("snapshot_dt_periods") * period
    t_end = float(case["completion"]["t_max"]) * period
    dt = period / case["time_stepping"]["steps_per_forcing_period"]
    collapse = CollapseSettings.from_dict(block, t_end=t_end, dt_max=dt)
    recon = dict(recon or {})
    for name in ("fill", "restart", "dt0", "dt_growth", "courant", "tau_end", "cap_window_factor",
                 "cap_splice_factor", "curvature_sizing"):
        value = getattr(args, name)
        if value is not None:
            recon[name] = value
    return collapse, recon, period


def write_profile(path: Path, ch: dict, header: dict):
    import numpy as np

    np.savetxt(path, np.column_stack([ch["r"], ch["z"], ch["u_r"], ch["u_z"]]),
               header=json.dumps(header) + "\nr z u_r u_z", comments="# ")


def near_cut(segs, z_cut: float, reach: float) -> tuple[float, float]:
    """``(U, h)``: the largest free-surface speed of both chains within ``reach`` of the cut, and
    their smallest element length there (twice the node spacing: the elements are quadratic)."""
    import numpy as np

    U, h = 1e-3, math.inf
    for c in segs:
        near = np.hypot(c["r"], c["z"] - z_cut) < reach
        if near.any():
            U = max(U, float(np.hypot(c["u_r"], c["u_z"])[near].max()))
        edges = near[:-1] | near[1:]
        if edges.any():
            h = min(h, 2.0 * float(np.hypot(np.diff(c["r"]), np.diff(c["z"]))[edges].min()))
    return U, h


def chain_record(segs) -> dict:
    """The three tips on the axis, the bubble's widest point and volume."""
    import numpy as np

    bubble = next(c for c in segs if not c["open"])
    cavity = next(c for c in segs if c["open"])
    k = int(np.argmax(bubble["r"]))
    return {"z_bubble_bottom": float(bubble["z"][0]), "w_bubble_bottom": float(bubble["u_z"][0]),
            "z_bubble_top": float(bubble["z"][-1]), "w_bubble_top": float(bubble["u_z"][-1]),
            "z_cavity_tip": float(cavity["z"][0]), "w_cavity_tip": float(cavity["u_z"][0]),
            "bridge": float(cavity["z"][0] - bubble["z"][-1]),
            "bubble_volume": revolved_volume(np.column_stack([bubble["r"], bubble["z"]])),
            "bubble_r_max": float(bubble["r"][k]), "bubble_z_r_max": float(bubble["z"][k])}


def main(argv=None) -> int:
    args = parse_args(argv)
    case = run_case.load_case(args.case)
    state = args.state.resolve()
    meta_path = state.with_suffix(".json")
    if not state.is_file() or not meta_path.is_file():
        raise SystemExit("the state needs its dump and the driver's JSON beside it")
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    collapse, recon_block, _period = settings_from(case, args)
    # validated with a provisional radius; the actual one may depend on the state's neck
    recon_block.setdefault("delta", args.delta if args.delta is not None else 1.0)
    if args.delta is not None:
        recon_block["delta"] = args.delta
    settings = ReconnectSettings.from_dict(recon_block)
    # the run that wrote the state: a clean recorded source, and this case or one derived from it;
    # not a state written after a cut
    lineage = run_case.restart_record(state, case, args.case,
                                      derived_keys=run_case.PROTOTYPE_DERIVED_KEYS)
    if args.check:
        print(json.dumps({"valid": True, "settings": settings.to_dict(), "state": lineage}))
        return 0
    out = args.outdir
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"{out}: output directory is not empty")
    if out.resolve().is_relative_to(state.parent.parent):
        raise SystemExit("the output cannot be inside the run that wrote the state")
    if sha256(state) != lineage["sha256"] or sha256(meta_path) != lineage["metadata_sha256"]:
        raise SystemExit("the state changed during admission")
    out.mkdir(parents=True, exist_ok=True)
    params = CaseParameters.from_dict(case["physics"])
    nd = nondimensionalise(params)
    mesh = MeshSettings(**{k: case["mesh"][k] for k in run_case.MESH_KEYS if k in case["mesh"]})
    remesh = bool(case["mesh"].get("remeshing", True))
    tol = float(case["tolerances"]["newton"])
    manifest = {"schema": "faraday-reconnect-prototype-v1", "case_id": case.get("case_id"),
                "case_file": str(args.case.resolve()), "case_sha256": sha256(args.case),
                "state": lineage,
                "component_commit": component_commit(ROOT), "pyoomph": pyoomph_revision(),
                "argv": list(sys.argv if argv is None else argv),
                "units": "dimensionless; length 1/k; time in capillary units"}
    create_run_manifest(out, manifest)

    from pyoomph.output.states import copy_state_file

    from faraday_singularity.problem import FaradayProblem
    from faraday_singularity.surgery import CutFailed, CutRefused

    status = {"state": "running", "started": time.time()}
    run_case.write_json(out / "run_status.json", status)
    problem = FaradayProblem(nd, mesh=mesh, remesh=remesh, reconnect=settings)
    problem.set_output_directory(str(out))
    try:
        with problem:
            problem.initialise()
            local = out / "restart" / state.name
            local.parent.mkdir(exist_ok=True)
            copy_state_file(str(state), str(local))
            problem.load_state(str(local))
            meta = json.loads(meta_path.read_text())
            driver = CollapseDriver(problem, collapse, out)
            try:
                driver.restore(meta)
                status.update(run(problem, driver, settings, args, out, tol, collapse))
            except (CutRefused, CutFailed) as exc:
                # the run ends here; the problem is not restored (a failed remesh can leave it
                # half rebuilt, and a failing reload would hide this failure): the precut dump
                # holds the state before the cut
                status.update(state="refused" if isinstance(exc, CutRefused) else "failed",
                              detail=str(exc), cut_diagnostics="cut_diagnostics.json",
                              failed_at=(exc.record or {}).get("failed_at"))
                if isinstance(exc, CutFailed):
                    status["precut"] = exc.precut
            finally:
                driver.close()
                problem.close_diagnostics()
    except BaseException as exc:
        status.update(state="failed", detail=f"{type(exc).__name__}: {exc}")
        status["ended"] = time.time()
        run_case.write_json(out / "run_status.json", status)
        raise
    status["ended"] = time.time()
    status["wall_seconds"] = status["ended"] - status["started"]
    run_case.write_json(out / "run_status.json", status)
    print(json.dumps({k: status[k] for k in ("state", "detail", "steps", "stop") if k in status},
                     default=str))
    return 0 if status["state"] == "completed" else 1


def run(problem, driver, settings: ReconnectSettings, args, out: Path, tol: float,
        collapse: CollapseSettings) -> dict:
    import numpy as np

    from faraday_singularity.surgery import cut

    segs = problem.interface_segments()
    r_n, _, _ = neck_radius(segs[0]["r"], segs[0]["z"]) if len(segs) == 1 else (math.nan,) * 3
    if args.delta_ratio is not None:
        settings = dataclasses.replace(settings, delta=args.delta_ratio * r_n)
    run_case.write_json(out / "reconnect_settings.json", settings.to_dict())
    record = cut(problem, settings, out, driver=driver)
    geom = CutGeometry(**record["geometry"])
    h_tip = record["sizing"]["h_tip"]
    t_cut = record["t_cut"]
    t_p = record["pinch"]["t_p"]
    t_p = math.nan if t_p is None else float(t_p)  # None in JSON: no estimate
    z_cut = record["z_neck"]
    (out / "profiles").mkdir(exist_ok=True)
    table = (out / "steps.txt").open("w", buffering=1)
    table.write("# " + "\t".join(COLUMNS) + "\n")
    problem.start_diagnostics(out / "diagnostics.txt")
    stop = {"reason": "steps"}
    step = 0

    def row(dt, newton_tol, wall, **extra):
        segs = problem.interface_segments()
        obs = problem.cut_observables()
        m = driver.measure()
        ch = chain_record(segs) if len(segs) == 2 else {}
        t = problem.get_current_time(dimensional=False, as_float=True)
        values = {"step": step, "time": t, "tau": t - t_cut, "tau_p": t - t_p, "dt": dt,
                  "newton_tol": newton_tol,
                  **{k: obs[k] for k in ("volume", "kinetic_energy", "momentum_z",
                                         "divergence_sq")},
                  **ch, **{k: m[k] for k in ("axis_curvature", "r_j", "z_b", "Q_j", "V_j",
                                              "h_near", "U")},
                  "remesh_count": problem.remesh_count, "wall": wall, **extra}
        table.write("\t".join(f"{float(values.get(c, math.nan)):.12e}"
                              if c not in ("step", "remesh_count") else str(values[c])
                              for c in COLUMNS) + "\n")
        finite = math.isfinite(t_p)
        header = {"time": t, "tau": t - t_cut, "tau_p": t - t_p if finite else None,
                  "t_p": t_p if finite else None, "step": step}
        for c in segs:
            write_profile(out / "profiles" / f"postcut_{step:04d}_{'open' if c['open'] else 'bubble'}.txt",
                          c, header)
        return segs, m

    def stops(segs, m):
        chains = [Chain(np.column_stack([c["r"], c["z"]]), tuple(c["end_types"])) for c in segs]
        return post_cut_stops(chains, geom, settings, h_tip,
                              neck_elements=collapse.neck_stop_elements,
                              fold_ratio=collapse.fold_ratio, fold_stop=collapse.fold_stop,
                              fold_stop_elements=collapse.fold_stop_elements,
                              fold_search=FOLD_SEARCH) or driver._geometry_stop(m)

    reach = 100 * settings.delta
    segs, m = row(0.0, 0.0, 0.0)
    dt_prev = settings.dt0 / settings.dt_growth
    try:
        found = stops(segs, m)  # the state right after the cut, before any step
        if found is not None:
            return {"state": "completed", "steps": 0, "stop": found, "cut": "cut_diagnostics.json"}
        for step in range(1, args.steps + 1):
            t = problem.get_current_time(dimensional=False, as_float=True)
            if t - t_cut >= settings.tau_end:
                stop = {"reason": "tau-end"}
                step -= 1
                break
            U, h_near = near_cut(segs, z_cut, reach)
            h_cut = min(h_tip, h_near)
            dt = min(settings.dt_growth * dt_prev, settings.courant * h_cut / U)
            newton_tol = max(tol, collapse.newton_floor / dt)
            wall0 = time.time()
            problem.run(t + dt, timestep=dt, maxstep=dt, outstep=False, temporal_error=1.0,
                        newton_solver_tolerance=newton_tol)
            dt = problem.get_current_time(dimensional=False, as_float=True) - t
            dt_prev = dt
            segs, m = row(dt, newton_tol, time.time() - wall0, h_cut=h_cut, U_cut=U)
            found = stops(segs, m)
            if found is not None:
                stop = found
                break
    finally:
        table.close()
    return {"state": "completed", "steps": step, "stop": stop, "cut": "cut_diagnostics.json"}


if __name__ == "__main__":
    raise SystemExit(main())
