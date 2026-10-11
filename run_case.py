#!/usr/bin/env python3
"""Run one axisymmetric Faraday case described by a ``pyoomph-case-v1`` JSON file.

Example::

    .venv-pyoomph-f912ac10-py313/bin/python run_case.py \
        verificationCases/free-decay/free-decay-h0.025-n100.json --outdir /path/to/run

Times in the case file are in forcing periods ``2 pi/Omega``. Unrecognised options are left for
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

from faraday_singularity.mesh import MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402

MESH_KEYS = ("h_interface", "h_axis", "h_max", "grading", "interface_points")


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    if case.get("schema") != "pyoomph-case-v1":
        raise SystemExit(f"{path}: not a pyoomph-case-v1 file")
    expected = {
        ("geometry", "kind"): "axisymmetric",
        ("time_stepping", "scheme"): "BDF2",
        ("output", "cadence_unit"): "forcing periods",
        ("completion", "kind"): "time",
        ("completion", "unit"): "forcing periods",
    }
    for (block, key), value in expected.items():
        if case.get(block, {}).get(key) != value:
            raise SystemExit(f"{path}: {block}.{key} must be {value!r}")
    steps = case["time_stepping"].get("steps_per_forcing_period")
    if not isinstance(steps, int) or steps < 1:
        raise SystemExit(f"{path}: time_stepping.steps_per_forcing_period must be a positive integer")
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
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--periods", type=float, help="override the end time, in forcing periods")
    ap.add_argument("--mesh-output", action="store_true",
                    help="also write bulk VTU files at every output time")
    args, _ = ap.parse_known_args()

    case = load_case(args.case)
    params = CaseParameters.from_dict(case["physics"])
    nd = nondimensionalise(params)
    mesh = MeshSettings(**{k: case["mesh"][k] for k in MESH_KEYS if k in case["mesh"]})
    remesh = bool(case["mesh"].get("remeshing", True))

    ts = case["time_stepping"]
    period = nd.forcing_period
    periods = args.periods if args.periods is not None else float(case["completion"]["value"])
    dt = period / ts["steps_per_forcing_period"]
    dt_out = period * float(case["output"]["cadence"])
    t_end = periods * period

    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise SystemExit(f"{args.outdir}: output directory is not empty")
    args.outdir.mkdir(parents=True, exist_ok=True)
    info = {
        "case_file": str(args.case.resolve()),
        "case": case,
        "physical": params.to_dict(),
        "dimensionless": nd.to_dict(),
        "mesh": asdict(mesh) | {"remeshing": remesh},
        "time_stepping": {"scheme": "BDF2", "dt": dt, "dt_out": dt_out, "t_end": t_end,
                          "periods": periods},
        "provenance": provenance(args.case),
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
        problem.write_states = bool(case["output"]["write_states"])
        with problem:
            problem.initialise()
            problem.start_diagnostics(args.outdir / "diagnostics.txt")
            problem.run(t_end, timestep=dt, outstep=dt_out,
                        newton_solver_tolerance=float(case["tolerances"]["newton"]))
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
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
