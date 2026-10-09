#!/usr/bin/env python3
"""Run one axisymmetric Faraday case described by a ``pyoomph-case-v1`` JSON file.

Example::

    .venv-pyoomph-f912ac10-py313/bin/python run_case.py \
        verificationCases/free-decay/free-decay-h0.05.json --outdir /path/to/run

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
    if case.get("geometry", {}).get("kind") != "axisymmetric":
        raise SystemExit(f"{path}: geometry.kind must be axisymmetric")
    return case


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def provenance(case_path: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              check=False).stdout.strip()

    info = {"case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "component_commit": git("rev-parse", "HEAD") or None,
            "component_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
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
    dt = period / int(ts["steps_per_forcing_period"])
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

    status = {"state": "running", "started": time.time()}
    write_json(args.outdir / "run_status.json", status)
    problem = FaradayProblem(nd, mesh=mesh, remesh=remesh, mesh_output=args.mesh_output)
    try:
        problem.set_output_directory(str(args.outdir))
        problem.write_states = bool(case["output"]["write_states"])
        with problem:
            problem.initialise()
            problem.start_diagnostics(args.outdir / "diagnostics.txt")
            problem.run(t_end, timestep=dt, outstep=dt_out,
                        newton_solver_tolerance=float(case["tolerances"]["newton"]))
        status["state"] = "completed"
    except Exception as exc:  # record the failure, then re-raise for a non-zero exit
        status.update(state="failed", detail=f"{type(exc).__name__}: {exc}", ended=time.time(),
                      remesh_count=problem.remesh_count)
        write_json(args.outdir / "run_status.json", status)
        raise
    finally:
        problem.close_diagnostics()
    status["ended"] = time.time()
    status["wall_seconds"] = status["ended"] - status["started"]
    status["remesh_count"] = problem.remesh_count
    write_json(args.outdir / "run_status.json", status)
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
