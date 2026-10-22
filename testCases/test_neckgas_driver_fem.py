"""The gas in the collapse driver, finite-element runs of the production runner on a short trough
(the trough on the axis is a duct of the gas between ``z_lo`` and ``z_hi``; the gas is a dense,
slow test gas so that it steps a few times per step of the liquid and acts on it).

* G6b: a block that stays dormant gives the trajectory without the block to round-off only (the
  pinned field and its zero traction change the compiled code, as a reconnect block does): the
  same rows, events and files, every number within a relative 1e-8 (but the shortest edge near the
  axis, whose nodes gmsh places in a rebuild, within 1e-5) and ``z_axis`` within 1e-10;
* an active one-way gas leaves the trajectory as the dormant block's, to the byte, while it writes
  its rows and profiles;
* an active two-way gas writes its rows (the start, then a row per step), its profiles beside the
  interface profiles, closes its budgets, sets the traction on the free surface and changes the
  trajectory.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = {
    "schema": "pyoomph-case-v1", "case_id": "trough",
    "geometry": {"kind": "axisymmetric"},
    "physics": {"fluid": {"Oh": 0.05, "Bo": 0.0}, "container": {"mode": 1, "depth": 2.0},
                "forcing": {"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
                "initial": {"amplitude": -0.2}},
    "tolerances": {"newton": 1e-8},
    "mesh": {"h_interface": 0.05, "h_axis": 0.05, "h_max": 0.4, "grading": 0.3,
             "interface_points": 201},
    "time_stepping": {"scheme": "BDF2", "steps_per_forcing_period": 160},
    "collapse": {"max_steps": 8, "refine_ratio": 0.5, "remesh_backoff": 2, "state_dt": 0.0,
                 "state_keep": 3, "snapshot_dt": 0.03},
    "output": {"cadence": 0.25, "cadence_unit": "forcing periods", "write_states": False},
    "completion": {"kind": "event", "t_max": 1.0, "unit": "forcing periods"},
}
GAS = {"z_hi": 2.04, "neck_window": [1.85, 2.0], "density_ratio": 1.0, "p0": 1.0, "h_bz": 0.02,
       "dz_cap": 0.02, "dz_max_lo": 0.02, "dz_max_hi": 0.02, "half_width": 2.0, "r_on": 10.0}


def case_with(gas=None, **collapse):
    c = copy.deepcopy(CASE)
    if gas is not None:
        c["collapse"]["gas"] = gas
    c["collapse"].update(collapse)
    return c


def run(tmp_path: Path, name: str, case: dict, *extra) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(case))
    out = tmp_path / name
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases" / "run_faraday.py"),
                           str(path), "--out", str(out), *extra], capture_output=True, text=True,
                          timeout=900, cwd=tmp_path, env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return out


def table(path: Path):
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return head, lines[1:]


def numbers(path: Path, drop=("wall",)):
    head, rows = table(path)
    keep = [k for k, h in enumerate(head) if h not in drop]
    return [head[k] for k in keep], np.array([[float(r.split("\t")[k]) for k in keep]
                                              for r in rows])


def gas_rows(out: Path) -> list[dict]:
    head, rows = table(out / "gas.txt")
    return [dict(zip(head, r.split("\t"))) for r in rows]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    pytest.importorskip("pyoomph")
    tmp = tmp_path_factory.mktemp("gasdriver")
    return {"none": run(tmp, "none", case_with()),
            "dormant": run(tmp, "dormant", case_with(dict(GAS, r_on=1e-9))),
            "oneway": run(tmp, "oneway", case_with(dict(GAS, coupling="one-way", variants=[
                {"name": "A"}, {"name": "S", "separated": True}]))),
            "twoway": run(tmp, "twoway", case_with(dict(GAS)))}


def _events(out):
    return [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]


def test_g6b_a_dormant_block_changes_the_trajectory_at_round_off_only(runs):
    a, b = runs["none"], runs["dormant"]
    ha, xa = numbers(a / "collapse.txt")
    hb, xb = numbers(b / "collapse.txt")
    assert ha == hb and xa.shape == xb.shape
    scale = np.maximum(np.abs(xa), 1e-300)
    rel = np.where(np.abs(xa) > 1e-6, np.abs(xa - xb) / scale, 0.0)
    mesh = [ha.index("h_near")]  # the shortest edge near the axis: gmsh's nodes of a rebuild
    assert np.nanmax(np.delete(rel, mesh, axis=1)) <= 1e-8
    assert np.nanmax(rel[:, mesh]) <= 1e-5
    za = ha.index("z_axis")
    assert np.max(np.abs(xa[:, za] - xb[:, za])) <= 1e-10
    assert [e["event"] for e in _events(a)] == [e["event"] for e in _events(b)]
    assert not (b / "gas.txt").read_text().splitlines()[1:]  # dormant: no row
    assert json.loads((b / "run_status.json").read_text())["collapse"]["gas"]["status"] == \
        "dormant"
    names = sorted(p.name for p in (a / "profiles").glob("*.txt"))
    assert names == sorted(p.name for p in (b / "profiles").glob("*.txt"))
    assert not list((b / "gas_profiles").glob("*.npz"))


def test_an_active_one_way_gas_leaves_the_trajectory_of_the_dormant_block(runs):
    a, b = runs["dormant"], runs["oneway"]
    ha, xa = numbers(a / "collapse.txt")
    hb, xb = numbers(b / "collapse.txt")
    assert ha == hb and np.array_equal(xa, xb, equal_nan=True)
    assert (a / "diagnostics.txt").read_text() == (b / "diagnostics.txt").read_text()
    for p in sorted((a / "profiles").glob("*.txt")):
        assert p.read_text() == (b / "profiles" / p.name).read_text(), p.name
    rows = gas_rows(b)
    assert {r["variant"] for r in rows} == {"A", "S"} and all(r["applied"] == "0" for r in rows)
    steps = sorted({int(r["step"]) for r in rows})
    assert steps == list(range(0, 9)), steps  # the start at step 0, then every step


def test_an_active_two_way_gas_acts_on_the_liquid(runs):
    a, b = runs["dormant"], runs["twoway"]
    events = [e["event"] for e in _events(b)]
    assert "gas-activated" in events and "gas-started" in events
    rows = gas_rows(b)
    assert [int(r["step"]) for r in rows] == list(range(0, 9))
    assert all(r["applied"] == "1" for r in rows)
    assert all(abs(float(r["mass_error"])) <= 1e-12 and abs(float(r["energy_error"])) <= 1e-10
               for r in rows)
    assert int(rows[-1]["n_sub"]) >= 1 and float(rows[-1]["W_liquid"]) != 0.0
    ha, xa = numbers(a / "collapse.txt")
    hb, xb = numbers(b / "collapse.txt")
    za = ha.index("z_axis")
    assert np.max(np.abs(xa[:, za] - xb[:, za])) > 1e-8  # the gas pushes on the surface
    profiles = sorted(p.stem.split("_")[1] for p in (b / "profiles").glob("profile_*.txt"))
    gas = sorted(p.stem.split("_")[1] for p in (b / "gas_profiles").glob("gas_*.npz"))
    assert gas and set(gas) <= set(profiles)
    g = np.load(b / "gas_profiles" / f"gas_{gas[-1]}.npz")
    head = json.loads(str(g["header"]))
    assert head["variants"] == ["A"] and g["A_p"].shape == g["A_zc"].shape
    status = json.loads((b / "run_status.json").read_text())["collapse"]["gas"]
    assert status["status"] == "active" and status["coupling"] == "two-way"
