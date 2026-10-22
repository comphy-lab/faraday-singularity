"""Restarts of a run with an active gas (gate G6c of the neck gas): a synthetic necked cavity
(``test_reconnect_fem.neck_interface``, closed by an imposed flow) with a two-way gas from its
first step and a landing state on the way, dumped at every step. The gas's side file of a dump
reads back into the gas and writes again to the same bytes, so that a restart restores the gas to
the bit; a restart in a fresh process from a rotating dump or from the landing dump (kept within
its step, ``resume_at``) then continues the uninterrupted run's rows of ``collapse.txt`` (the wall
time aside) and ``gas.txt`` to round-off (a relative 1e-10 and 1e-8): pyoomph's restart of this release,
gas or none, restores the finite-element state to round-off only (about 1e-16 at the first step
after it, 1e-12 a few steps on, the same without the gas), not to the byte. A dump of the gas has
its side file, whose rotation follows the dump's, and a restart refuses a side file of other bytes
or none."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testCases"))
pytest.importorskip("pyoomph")

import test_reconnect_fem as fem  # noqa: E402

RUN = """
import json, math, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.argv = sys.argv[:3] + sys.argv[3:4]
import numpy as np
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.mesh import FaradayCylinderMesh, MeshSettings
from faraday_singularity.neckgas.coupler import GasSettings
from faraday_singularity.parameters import CaseParameters, nondimensionalise
from faraday_singularity.problem import FaradayProblem
from faraday_singularity.reconnect import neck_radius
R_N = %(R_N)r
%(neck_interface)s
%(impose_closing_flow)s

class NeckMesh(FaradayCylinderMesh):
    def _interface_points(self, segs=None):
        if self.is_first_time():
            return neck_interface(self.radius)
        return super()._interface_points(segs)


class NeckProblem(FaradayProblem):
    mesh_class = NeckMesh


out, spec = Path(sys.argv[2]), json.loads(sys.argv[3])
sys.argv = sys.argv[:1]
nd = nondimensionalise(CaseParameters.from_dict(spec["physics"]))
settings = CollapseSettings(t_end=10.0, dt_max=0.01, **spec["collapse"])
gas = GasSettings.from_dict(spec["gas"])
pb = NeckProblem(nd, mesh=MeshSettings(**spec["mesh"]), gas=True)
pb.set_output_directory(str(out))
with pb:
    pb.initialise()
    driver = CollapseDriver(pb, settings, out, gas=gas, landing=tuple(spec["landing"]))
    if spec.get("restart") is None:
        impose_closing_flow(pb)
    else:
        state = Path(spec["restart"])
        pb.load_state(str(state))
        gas_file = state.with_suffix(".gas.npz")
        driver.restore(json.loads(state.with_suffix(".json").read_text()),
                       gas_file=gas_file if gas_file.is_file() else None)
    pb.start_diagnostics(out / "diagnostics.txt")
    result = driver.run(1e-8)
    driver.close()
    pb.close_diagnostics()
(out / "result.json").write_text(json.dumps(result, default=float))
"""

SPEC = {"physics": fem.PHYSICS, "mesh": fem.MESH,
        "collapse": {"max_steps": 10, "state_dt": 0.0, "state_keep": 50, "snapshot_dt": 0.0,
                     "neck_stop": 1e-4},
        "landing": [0.0153],  # lands at step 5 of the imposed closing
        "gas": {"z_hi": 0.72, "neck_window": [0.5, 0.7], "r_on": 0.05, "dz_cap": 0.01,
                "dz_max_lo": 0.01, "dz_max_hi": 0.01, "half_width": 10.0}}  # air


def _script() -> str:
    return RUN % {"R_N": fem.R_N, "neck_interface": inspect.getsource(fem.neck_interface),
                  "impose_closing_flow": inspect.getsource(fem.impose_closing_flow)}


def _run(cwd: Path, spec: dict, check=True):
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", _script(), str(ROOT / "src"), "run",
                           json.dumps(spec)], capture_output=True, text=True, timeout=1800,
                          cwd=cwd, env=env, check=False)
    if check:
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


def _rows(path: Path, drop=("wall", "mass_error", "energy_error")):
    """The rows of a table without the wall time and the budgets' errors (round-off noise, at
    1e-15, which other tests bound)."""
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    ks = {head.index(d) for d in drop if d in head}
    return [lines[0]] + ["\t".join(x for j, x in enumerate(r.split("\t")) if j not in ks)
                         for r in lines[1:]]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gasrestart")
    _run(tmp / "base", SPEC)
    return tmp


def _step_of(dump: Path) -> int:
    return int(json.loads(dump.with_suffix(".json").read_text())["step"])


def _cells(row: str) -> list[float]:
    """The numbers of a row (the variant's name and the neck's source aside)."""
    out = []
    for x in row.split("\t")[1:]:
        try:
            out.append(float(x))
        except ValueError:
            continue
    return out


def _compare_after(a: Path, b: Path, step: int):
    """b restarted at ``step`` from a's dump: every row of b after ``step`` is a's row of that
    step to a relative 1e-10 in ``collapse.txt`` and 1e-8 in ``gas.txt`` (whose differences of
    nearly equal pressures, ``dp_th``, amplify the round-off of the restart)."""
    for name, tol in (("collapse.txt", 1e-10), ("gas.txt", 1e-8)):
        ra, rb = _rows(a / name), _rows(b / name)
        assert ra[0] == rb[0]
        by_step = {}
        for r in ra[1:]:
            by_step.setdefault(int(r.split("\t")[0]), []).append(r)
        later = [r for r in rb[1:] if int(r.split("\t")[0]) > step]
        assert len(later) >= 3, name
        for r in later:
            k = int(r.split("\t")[0])
            mine = np.array(_cells(r))
            ref = min((np.array(_cells(x)) for x in by_step[k]),
                      key=lambda y: float(np.nansum(np.abs(y - mine))))
            scale = np.maximum(np.abs(ref), 1e-12)
            assert np.nanmax(np.abs(mine - ref) / scale) <= tol, (name, k)
    assert sorted(p.name for p in (b / "states").glob("*.gas.npz") if _step_of(
        p.with_name(p.name.replace(".gas.npz", ".dump"))) > step)


def _cells(row: str) -> list[float]:
    """The numbers of a row (the variant's name and the neck's source aside)."""
    out = []
    for x in row.split("\t")[1:]:
        try:
            out.append(float(x))
        except ValueError:
            continue
    return out


def test_a_gas_file_reads_back_to_the_same_bytes(base):
    sys.path.insert(0, str(ROOT / "src"))
    from faraday_singularity.neckgas.coupler import (GasSettings, NeckGas, read_state_file,
                                                     write_state_file)

    run = base / "base" / "run"
    files = sorted((run / "states").glob("*.gas.npz"))
    assert len(files) >= 5
    for f in files:
        gas = NeckGas(GasSettings.from_dict(SPEC["gas"]))
        gas.load_state_dict(*read_state_file(f))
        again = base / "again.gas.npz"
        write_state_file(again, *gas.state_dict())
        assert again.read_bytes() == f.read_bytes(), f.name


def test_the_run_dumps_its_gas_and_lands(base):
    run = base / "base" / "run"
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "gas-activated" in kinds and "gas-started" in kinds and "landing" in kinds
    landing = run / "states" / "landing_0.0153.dump"
    meta = json.loads(landing.with_suffix(".json").read_text())
    assert meta["resume_at"] == "remesh-check" and meta["gas"]["file"] == "landing_0.0153.gas.npz"
    for p in (run / "states").glob("state_*.dump"):
        m = json.loads(p.with_suffix(".json").read_text())
        assert m["gas"]["status"] in ("pending", "active", "dormant")
        assert p.with_suffix(".gas.npz").is_file() == bool(m["gas"]["file"])
    rows = (run / "gas.txt").read_text().splitlines()[1:]
    assert len(rows) >= 5


@pytest.mark.parametrize("which", ["rotating", "landing"])
def test_g6c_a_restart_continues_the_run_to_the_byte(base, which):
    run = base / "base" / "run"
    if which == "rotating":
        dumps = sorted((run / "states").glob("state_*.dump"))
        dump = next(p for p in dumps if _step_of(p) >= 4)
    else:
        dump = run / "states" / "landing_0.0153.dump"
    assert json.loads(dump.with_suffix(".json").read_text())["gas"]["file"]
    cwd = base / f"restart-{which}"
    _run(cwd, dict(SPEC, restart=str(dump)))
    _compare_after(run, cwd / "run", _step_of(dump))


def test_a_restart_refuses_a_gas_file_of_other_bytes_or_none(base):
    run = base / "base" / "run"
    dump = next(p for p in sorted((run / "states").glob("state_*.dump")) if _step_of(p) >= 4)
    copy = run / "states" / "tampered.dump"  # beside it: its mesh files are found from there
    for suffix in (".dump", ".json", ".gas.npz"):
        shutil.copyfile(dump.with_suffix(suffix), copy.with_suffix(suffix))
    data = bytearray(copy.with_suffix(".gas.npz").read_bytes())
    data[-10] ^= 1
    copy.with_suffix(".gas.npz").write_bytes(bytes(data))
    proc = _run(base / "tampered-run", dict(SPEC, restart=str(copy)), check=False)
    assert proc.returncode != 0 and "does not match its record" in proc.stderr, proc.stderr[-2000:]
    copy.with_suffix(".gas.npz").unlink()
    proc = _run(base / "missing-run", dict(SPEC, restart=str(copy)), check=False)
    assert proc.returncode != 0 and "is needed to continue it" in proc.stderr, proc.stderr[-2000:]
