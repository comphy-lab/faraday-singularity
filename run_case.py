#!/usr/bin/env python3
"""Run one axisymmetric Faraday case described by a ``pyoomph-case-v1`` JSON file.

Example::

    .venv-pyoomph-f912ac10-py313/bin/python run_case.py \
        verificationCases/free-decay/free-decay-h0.025-n100.json --outdir /path/to/run

Times in the case file are in forcing periods ``2 pi/Omega``. A case with a ``collapse`` block runs
the collapse driver (``src/faraday_singularity/driver.py``) up to ``completion.t_max`` or until it
stops at its conservative neck-radius or fold-gap cutoff or, if the case sets a finite
``collapse.post_birth_aspect``, once a born jet is unambiguous, with the fixed step of
``time_stepping`` as its largest step. A
Newton failure that the driver classifies as the approach to a pinch-off or to the singularity is
such a stop, and the run is ``completed``; the stop is recorded under ``collapse.stop`` in
``run_status.json``. Any other failure marks the run ``failed``. Unrecognised options are left for
pyoomph's own command-line parser (e.g. ``--omp 4``). Each output directory receives
``parameters.json`` (inputs, dimensionless groups, numerical settings and provenance),
``run_status.json``, ``diagnostics.txt`` (axis height and velocity, mode amplitude, volume, area,
kinetic energy, dissipation and remesh count after every step), pyoomph's per-step integrals and
free-surface snapshots (``liquid__interface/``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _shared_memory_btl() -> str | None:
    """Open MPI's shared-memory transport: ``vader`` before release 5, ``sm`` from 5 on."""
    exe = shutil.which("ompi_info")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Open MPI v(\d+)", out)
    if match is None:
        return None
    return "self,vader" if int(match.group(1)) < 5 else "self,sm"


# Runs are serial: the MPI-enabled pyoomph build only needs the self and shared-memory transports.
# Restricting them avoids probing network transports during singleton MPI initialisation.
if "OMPI_MCA_btl" not in os.environ and (_btl := _shared_memory_btl()) is not None:
    os.environ["OMPI_MCA_btl"] = _btl

from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.cases import validate_case  # noqa: E402
from faraday_singularity.mesh import MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.provenance import create_run_manifest, pyoomph_revision  # noqa: E402

MESH_KEYS = ("h_interface", "h_axis", "h_max", "grading", "interface_points")
CASE_FIELDS_USED = ("schema", "case_id", "geometry.kind", "time_stepping.scheme",
                    "time_stepping.steps_per_forcing_period", "output.cadence_unit",
                    "output.cadence", "output.write_states", "completion.kind",
                    "completion.unit", "completion.t_max", "completion.value",
                    "physics", "mesh.remeshing", "tolerances.newton", "collapse") + tuple(
                        "mesh." + key for key in MESH_KEYS)


def case_has_field(case: dict, field: str) -> bool:
    value = case
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def restart_record(state: Path, case: dict, case_path: Path, *, legacy: bool = False) -> dict:
    """Validate restart custody before output, preserving explicitly imported legacy lineage."""
    state = state.resolve()
    if not state.is_file() or not state.with_suffix(".json").is_file():
        raise SystemExit("restart needs both its state dump and driver metadata")
    parent = state.parent.parent
    manifest_path = parent / "run-manifest.json"
    case_hash = hashlib.sha256(case_path.read_bytes()).hexdigest()
    if legacy:
        if manifest_path.exists():
            raise SystemExit("use --restart-from for a run with a normalised manifest")
        recorded_path = parent / "parameters.json"
        if not recorded_path.is_file():
            raise SystemExit("legacy restart has no saved parameters")
        recorded = json.loads(recorded_path.read_text())
        source = recorded.get("provenance", {})
        if source.get("component_dirty") is not False or not re.fullmatch(
                r"[0-9a-f]{40}", str(source.get("component_commit", ""))):
            raise SystemExit("legacy restart has no clean recorded source")
        case_file = recorded.get("case_file")
        if not isinstance(case_file, str) or not case_file:
            raise SystemExit("legacy restart has no recorded case file")
        original = Path(case_file)
        if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != source.get("case_sha256"):
            raise SystemExit("legacy restart case does not match its recorded hash")
        if recorded.get("case", {}).get("physics") != case.get("physics") or recorded.get("case", {}).get("geometry") != case.get("geometry"):
            raise SystemExit("legacy restart changes physics or geometry")
        parent_format = "legacy-parameters-v1"
        parent_commit = source["component_commit"]
    else:
        if not manifest_path.is_file():
            raise SystemExit("restart has no parent run-manifest.json; legacy imports need --legacy-restart-from")
        recorded_path = manifest_path
        recorded = json.loads(manifest_path.read_text())
        if recorded.get("case_sha256") != case_hash:
            raise SystemExit("restart parent belongs to a different case")
        if recorded.get("schema") != "pyoomph-runner-manifest-v1" or not re.fullmatch(
                r"[0-9a-f]{40}", str(recorded.get("component_commit", ""))):
            raise SystemExit("restart parent needs a normalised manifest and a clean source commit")
        parent_format = "pyoomph-runner-manifest-v1"
        parent_commit = recorded["component_commit"]
    return {"path": str(state), "sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(state.with_suffix(".json").read_bytes()).hexdigest(),
            "parent_record": {"path": str(recorded_path),
                              "sha256": hashlib.sha256(recorded_path.read_bytes()).hexdigest(),
                              "format": parent_format, "component_commit": parent_commit}}


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    if case.get("schema") != "pyoomph-case-v1":
        raise SystemExit(f"{path}: not a pyoomph-case-v1 file")
    expected = {
        ("geometry", "kind"): "axisymmetric",
        ("time_stepping", "scheme"): "BDF2",
        ("output", "cadence_unit"): "forcing periods",
        ("completion", "unit"): "forcing periods",
    }
    # linear cases run to a fixed time; collapse cases run to an event (stop before pinch-off)
    expected[("completion", "kind")] = "event" if "collapse" in case else "time"
    for (block, key), value in expected.items():
        if case.get(block, {}).get(key) != value:
            raise SystemExit(f"{path}: {block}.{key} must be {value!r}")
    steps = case["time_stepping"].get("steps_per_forcing_period")
    if not isinstance(steps, int) or steps < 1:
        raise SystemExit(f"{path}: time_stepping.steps_per_forcing_period must be a positive integer")
    if "collapse" in case and not isinstance(case["completion"].get("t_max"), (int, float)):
        raise SystemExit(f"{path}: collapse cases need completion.t_max (forcing periods)")
    validate_case(case)
    return case


class Terminated(Exception):
    """Raised from the SIGTERM handler so that the run status records the interruption."""


def _raise_terminated(signum, _frame):
    raise Terminated(f"signal {signum}")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def provenance(case_path: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                                 check=False, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    info = {"case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "component_commit": commit,
            # None when the state cannot be determined; untracked files count as dirty
            "component_dirty": None if status is None else bool(status)}
    try:
        from importlib.metadata import distribution

        direct = distribution("pyoomph").read_text("direct_url.json")
        info["pyoomph"] = json.loads(direct) if direct else None
    except Exception as exc:  # provenance must not stop a run; record why it is missing
        info["pyoomph"] = f"unavailable: {type(exc).__name__}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--outdir", "--out", dest="outdir", type=Path, required=True)
    ap.add_argument("--periods", type=float, help="override the end time, in forcing periods")
    ap.add_argument("--mesh-output", action="store_true",
                    help="also write bulk VTU files at every output time")
    restarts = ap.add_mutually_exclusive_group()
    restarts.add_argument("--restart", "--restart-from", dest="restart", type=Path,
                    help="collapse cases: resume from a state dump written by the collapse driver")
    restarts.add_argument("--legacy-restart-from", type=Path,
                         help="explicitly import a legacy state with verified saved parameters")
    ap.add_argument("--check", action="store_true", help="validate without creating output")
    args, _ = ap.parse_known_args()

    case = load_case(args.case)
    if not isinstance(case.get("case_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", case["case_id"]):
        raise SystemExit("case_id must be a valid case identifier")
    tol = float(case["tolerances"]["newton"])
    write_states = case["output"]["write_states"]
    if not math.isfinite(tol) or tol <= 0 or not isinstance(write_states, bool):
        raise SystemExit("Newton tolerance must be positive and finite; write_states must be boolean")
    params = CaseParameters.from_dict(case["physics"])
    nd = nondimensionalise(params)
    mesh = MeshSettings(**{k: case["mesh"][k] for k in MESH_KEYS if k in case["mesh"]})
    remesh = bool(case["mesh"].get("remeshing", True))

    ts = case["time_stepping"]
    period = nd.forcing_period
    collapse = "collapse" in case
    if args.periods is not None:
        periods = args.periods
    else:
        periods = float(case["completion"]["t_max" if collapse else "value"])
    dt = period / ts["steps_per_forcing_period"]  # fixed step, or the largest step of a collapse
    dt_out = period * float(case["output"]["cadence"])
    t_end = periods * period
    settings = None
    if collapse:
        block = dict(case["collapse"])
        if "snapshot_dt_periods" in block:
            block["snapshot_dt"] = block.pop("snapshot_dt_periods") * period
        settings = CollapseSettings.from_dict(block, t_end=t_end, dt_max=dt)

    if not math.isfinite(periods) or periods <= 0 or not math.isfinite(dt_out) or dt_out <= 0:
        raise SystemExit("completion time and output cadence must be positive and finite")
    if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())):
        raise SystemExit(f"{args.outdir}: output directory is not empty")
    source_state = args.restart or args.legacy_restart_from
    parent_record = None
    restart_meta = None
    if source_state is not None:
        if not collapse:
            raise SystemExit("restarts require a collapse case")
        parent_record = restart_record(source_state, case, args.case,
                                       legacy=args.legacy_restart_from is not None)
        metadata_bytes = source_state.with_suffix(".json").read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != parent_record["metadata_sha256"]:
            raise SystemExit("restart metadata changed during admission")
        restart_meta = json.loads(metadata_bytes)
        saved_time = restart_meta.get("time")
        if isinstance(saved_time, bool) or not isinstance(saved_time, (int, float)) or not math.isfinite(saved_time):
            raise SystemExit("restart metadata must record a finite physical time")
        if saved_time >= t_end - 1e-14:
            raise SystemExit(f"restart time {saved_time:g} ({saved_time / period:g} forcing periods) "
                             f"must precede end time {t_end:g} ({periods:g} forcing periods), "
                             "with physical times in capillary units")
        parent_root = source_state.resolve().parent.parent
        if args.outdir.resolve().is_relative_to(parent_root):
            raise SystemExit("restart output cannot be inside its parent run")
        args.restart = source_state
    if args.check:
        print(json.dumps({"valid": True, "case_id": case["case_id"], "restart": parent_record}))
        return 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    recorded_provenance = provenance(args.case)
    runner_path = Path(sys.argv[0]).resolve()
    if not runner_path.is_file():
        runner_path = Path(__file__).resolve()
    commit = recorded_provenance["component_commit"] or "unknown"
    if recorded_provenance["component_dirty"] is not False:
        commit += "-dirty"
    manifest = {"schema": "pyoomph-runner-manifest-v1", "case_id": case["case_id"],
                "case_file": str(args.case.resolve()),
                "case_sha256": recorded_provenance["case_sha256"],
                "case_fields_used": [field for field in CASE_FIELDS_USED
                                     if case_has_field(case, field)
                                     and not (collapse and field == "completion.value")
                                     and not (not collapse and field == "completion.t_max")
                                     and not (args.periods is not None and field in {"completion.value", "completion.t_max"})],
                "component_commit": commit, "pyoomph": pyoomph_revision(),
                "argv": list(sys.argv),
                "runner": {"path": str(runner_path),
                           "sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest()},
                "units": "dimensionless; length 1/k; time in capillary units",
                "case_time_units": "completion and snapshot_dt_periods in forcing periods; other collapse time controls in capillary units",
                "effective_time_stepping": {"dt_max": dt, "t_end": t_end, "dt_out": dt_out}}
    if parent_record is not None:
        manifest["restart_from"] = parent_record
    create_run_manifest(args.outdir, manifest)
    with (args.outdir / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "start", "wall_time": time.time()}) + "\n")
    info = {
        "case_file": str(args.case.resolve()),
        "case": case,
        "physical": params.to_dict(),
        "dimensionless": nd.to_dict(),
        "mesh": asdict(mesh) | {"remeshing": remesh},
        "time_stepping": {"scheme": "BDF2", "dt": dt, "dt_out": dt_out, "t_end": t_end,
                          "periods": periods, "mode": "collapse" if collapse else "fixed-step"},
        "collapse": asdict(settings) if settings is not None else None,
        "provenance": recorded_provenance,
        "restart": str(args.restart.resolve()) if args.restart is not None else None,
        "argv": sys.argv,
    }
    write_json(args.outdir / "parameters.json", info)

    from faraday_singularity.problem import FaradayProblem

    signal.signal(signal.SIGTERM, _raise_terminated)
    status = {"state": "running", "started": time.time()}
    write_json(args.outdir / "run_status.json", status)
    problem = None
    try:
        problem = FaradayProblem(nd, mesh=mesh, remesh=remesh, mesh_output=args.mesh_output)
        problem.set_output_directory(str(args.outdir))
        problem.write_states = write_states
        with problem:
            problem.initialise()
            # pyoomph's initialisation resets SIGTERM to its default action, which would end the
            # run without recording it in the status
            signal.signal(signal.SIGTERM, _raise_terminated)
            if settings is None:
                problem.start_diagnostics(args.outdir / "diagnostics.txt")
                problem.run(t_end, timestep=dt, outstep=dt_out, newton_solver_tolerance=tol)
            else:
                driver = CollapseDriver(problem, settings, args.outdir)
                try:
                    if args.restart is not None:
                        from pyoomph.output.states import copy_state_file

                        local = args.outdir / "restart" / args.restart.name
                        local.parent.mkdir(exist_ok=True)
                        if hashlib.sha256(args.restart.read_bytes()).hexdigest() != parent_record["sha256"]:
                            raise RuntimeError("restart state changed after admission")
                        if hashlib.sha256(args.restart.with_suffix(".json").read_bytes()).hexdigest() != parent_record["metadata_sha256"]:
                            raise RuntimeError("restart metadata changed after admission")
                        copy_state_file(str(args.restart), str(local))
                        problem.load_state(str(local))
                        driver.restore(restart_meta)
                    problem.start_diagnostics(args.outdir / "diagnostics.txt")
                    status["collapse"] = driver.run(tol)
                finally:
                    driver.close()
        status["state"] = "completed"
    except BaseException as exc:  # record failures and interrupts, then re-raise
        interrupted = isinstance(exc, (Terminated, KeyboardInterrupt))
        state = "interrupted" if interrupted or not isinstance(exc, Exception) else "failed"
        status.update(state=state, detail=f"{type(exc).__name__}: {exc}", ended=time.time(),
                      remesh_count=getattr(problem, "remesh_count", None))
        write_json(args.outdir / "run_status.json", status)
        raise
    finally:
        if problem is not None:
            problem.close_diagnostics()
    status["ended"] = time.time()
    status["wall_seconds"] = status["ended"] - status["started"]
    status["remesh_count"] = problem.remesh_count
    write_json(args.outdir / "run_status.json", status)
    with (args.outdir / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "completed", "wall_time": time.time(),
                                 "collapse": status.get("collapse")}) + "\n")
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
