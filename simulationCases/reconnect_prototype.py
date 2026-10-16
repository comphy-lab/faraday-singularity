#!/usr/bin/env python3
"""Reconnection prototype: load a landing state, cut the neck, and take a few steps after it.

Example::

    .venv-pyoomph-f912ac10-py313/bin/python simulationCases/reconnect_prototype.py CASE.json \\
        --state RUN/states/landing_0.001.dump --outdir OUT --delta-ratio 1.05 --steps 20

A development tool beside the production runner, whose cut is a restart of
``simulationCases/run_faraday.py`` (``README.md``). The prototype cuts at ``--delta`` or at
``--delta-ratio`` times the neck radius of the state, with options that override the case's
``collapse.reconnect`` block, and then takes ``--steps`` steps under the same collapse driver in the
same post-cut mode as the production runner (``driver.CollapseDriver`` made with the reconnect
settings: the stops on both chains, the step rule that follows the tips of both chains and the
capillary time of the smallest tip, ``tau`` and ``tau_p`` in every output). It has no stepping loop
of its own. Its manifest has its own schema (``faraday-reconnect-prototype-v1``), so its outputs
feed no figure and no runner continues its states.

``CASE.json`` is the ``pyoomph-case-v1`` file of the run that wrote the state, or one derived from
it in the keys a restart may change and in its ``reconnect`` block
(``run_case.PROTOTYPE_DERIVED_KEYS``: the name, the description ``completion.value``, and the
collapse keys ``keep_stop_state``, ``landing``, ``neck_stop`` and ``reconnect``); the physics, mesh
and every other collapse setting must be the ones the state was computed with. The run that wrote
the state is the directory above ``states/``: its ``run-manifest.json`` must be a normalised one
with a clean source commit, and its case is compared with ``CASE.json``
(``run_case.restart_record``); the manifest of the prototype records that lineage. The state is a
driver dump with its JSON beside it, normally a landing state. The cut radius must lie between
``min_landing_ratio`` and ``max_landing_ratio`` times the neck radius of the state, so a study in
``delta`` needs a landing state for each ``delta``. A state written after a cut is refused: one cut
per run.

The cut (``faraday_singularity.surgery.cut``) writes ``cut_diagnostics.json`` and keeps the
``precut``/``postcut`` states under ``states/``; a refused or failed cut writes the partial record
there instead, with the stage reached (``failed_at``), and ``run_status.json`` names it. The run
then ends: the problem is not restored, since ``states/precut_<delta>.dump`` holds the state before
the cut. Otherwise the driver writes ``collapse.txt`` (with ``tau`` and ``tau_p``), ``chains.txt``
(both chains, the budgets and the step limits), ``events.jsonl``, the profiles of both chains and its
dumps. The run ends after ``--steps`` steps, at ``tau_end`` after the cut, or at a stop the mesh
cannot follow; ``run_status.json`` records how (``steps``: the steps after the cut).
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
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.provenance import (component_commit, create_run_manifest,  # noqa: E402
                                            pyoomph_revision)
from faraday_singularity.reconnect import (ReconnectSettings, neck_radius,  # noqa: E402
                                           split_collapse_block)


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
    for name in ("dt0", "dt_growth", "courant", "tau_end", "cap_window_factor"):
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
                 "curvature_sizing"):
        value = getattr(args, name)
        if value is not None:
            recon[name] = value
    return collapse, recon, period


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
    from faraday_singularity.surgery import CutFailed, CutRefused, cut

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
            if args.delta_ratio is not None:
                segs = problem.interface_segments()
                r_n = neck_radius(segs[0]["r"], segs[0]["z"])[0] if len(segs) == 1 else math.nan
                settings = dataclasses.replace(settings, delta=args.delta_ratio * r_n)
            run_case.write_json(out / "reconnect_settings.json", settings.to_dict())
            step0 = int(meta["step"])
            steps = dataclasses.replace(collapse, max_steps=step0 + args.steps)
            driver = CollapseDriver(problem, steps, out, reconnect=settings)
            try:
                driver.restore(meta)
                cut(problem, settings, out, driver=driver)
                problem.start_diagnostics(out / "diagnostics.txt")
                result = driver.run(tol)
                status.update(state="completed", steps=result["steps"] - step0,
                              stop=result["stop"], cut="cut_diagnostics.json",
                              tau=result["cut"]["tau"], tau_p=result["cut"]["tau_p"])
            except (CutRefused, CutFailed) as exc:
                # the run ends here; the problem is not restored (a failed remesh can leave it
                # half rebuilt, and a failing reload would hide this failure): the precut dump
                # holds the state before the cut
                status.update(state="refused" if isinstance(exc, CutRefused) else "failed",
                              detail=str(exc), cut_diagnostics="cut_diagnostics.json",
                              failed_at=(exc.record or {}).get("failed_at"))
                if exc.precut is not None:  # every failure, and refusals from the plan on
                    status["precut"] = exc.precut
            finally:
                driver.close()
                problem.close_diagnostics()
    except BaseException as exc:
        interrupted = isinstance(exc, run_case.Terminated) or not isinstance(exc, Exception)
        status.update(state="interrupted" if interrupted else "failed",
                      detail=f"{type(exc).__name__}: {exc}")
        status["ended"] = time.time()
        run_case.write_json(out / "run_status.json", status)
        raise
    status["ended"] = time.time()
    status["wall_seconds"] = status["ended"] - status["started"]
    run_case.write_json(out / "run_status.json", status)
    print(json.dumps({k: status[k] for k in ("state", "detail", "steps", "stop") if k in status},
                     default=str))
    return 0 if status["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
