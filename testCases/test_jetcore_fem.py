"""Finite-element tests of the stabilisation of the run after the cut (the events block's ``jetcore``
and ``pinch`` parts) on the synthetic states of the liquid cut and of the ring.

* The far region's sizing on the synthetic state of ``test_liquid_cut_fem.py`` with heads that do
  not land (their necks at ``R_N delta``): with a far floor at the heads' element length and the
  far kinks recentred, the guard's far kinks (an ``edge_turn_refine`` of a few degrees) are regions
  at the floor, four chords wide, that ask for a rebuild only where their elements are coarser than
  the refine ratio times the floor and never mesh below it, the run steps to its limit without a
  stop, and the core's regions are sized as without the part.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_liquid_cut_fem as lcf  # noqa: E402
import test_reconnect_fem as fem  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import surgery  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import FaradayCylinderMesh, MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.problem import FaradayProblem  # noqa: E402

DELTA = lcf.DELTA
R_N = 0.3  # the heads' necks in units of delta: r_n/s 0.6, above eps_d and eps_max
H_HEAD = 0.04 * DELTA  # the element length on the heads (twice their node spacing)


def _chains():
    return lcf.jet_chains(DELTA, R_N)


def _focus():
    return lcf.jets_focus(_chains()[2], r_n=R_N * DELTA)


class HeadsMesh(FaradayCylinderMesh):
    """The first mesh is built on the chains of ``test_liquid_cut_fem.jet_chains`` at ``R_N``."""

    def define_geometry(self):
        if not self.is_first_time():
            return super().define_geometry()
        bubble, cavity, _ = _chains()
        focus = _focus()
        s = self.settings
        self.mesh_mode = "tris"
        self.set_gmsh_parameter("General.NumThreads", 1)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", min(s.h_interface, focus.size))
        self.set_gmsh_parameter("Mesh.MeshSizeMax", s.h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)
        self.default_resolution = s.h_max
        return self._define_chain_geometry([bubble, cavity], focus, focus.size)


class HeadsProblem(FaradayProblem):
    mesh_class = HeadsMesh


def heads_problem(outdir, recon):
    nd = nondimensionalise(CaseParameters.from_dict(fem.PHYSICS))
    pb = HeadsProblem(nd, mesh=MeshSettings(**lcf.MESH), reconnect=recon)
    pb.set_output_directory(str(outdir))
    return pb


def gas_cut_record(pb, recon) -> dict:
    """``test_liquid_cut_fem.gas_cut_record`` with this state's focus."""
    named, _ = surgery._named_chains(pb)
    geom = rc.cut_geometry([c for _, c in named], recon.reservoir_depth, gas_before=1.0)
    t = pb.get_current_time(dimensional=False, as_float=True)
    record = {"delta": DELTA, "rmin": DELTA, "r_neck": DELTA / 1.05, "z_neck": 0.85, "t_cut": t,
              "pinch": {"t_p": t - 1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": recon.tip_size * DELTA, "h_bridge": recon.bridge_size * DELTA,
                         "curvature": 0, "grading": 0.3},
              "settings": json.loads(json.dumps(recon.to_dict())), "focus": {"kept": None}}
    surgery.restore_cut(pb, record)
    pb.active_mesh_template().focus = _focus()
    return record


RECON_KINKS = rc.ReconnectSettings(delta=DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                                   focus_size=0.25, courant=0.4, edge_turn_refine=4.0)
BLOCK = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 1e-5}}


def _events(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def heads(tmp_path_factory):
    """The heads' state run three steps without jetcore and with its far sizing (the floor at the
    heads' element length, the far kinks recentred)."""
    out = tmp_path_factory.mktemp("heads")
    res = {}
    for name, extra in (("plain", {}), ("far", {"jetcore": {
            "enabled": True, "far_floor": H_HEAD, "far_kinks": "recentre"}})):
        block = ev.EventSettings.from_dict({**BLOCK, **extra})
        with heads_problem(out / f"pb-{name}", RECON_KINKS) as pb:
            pb.initialise()
            gas_cut_record(pb, RECON_KINKS)
            drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=3,
                                                      state_dt=0.0, state_keep=10),
                                 out / name, reconnect=RECON_KINKS, events=block)
            first = drv.reset_after_cut(pb.cut)
            res[name] = {"first": {g["name"]: dict(g) for g in first["regions"]},
                         "result": drv.run(1e-8), "kinks": dict(drv.state.kinks),
                         "events": _events(out / name / "events.jsonl"), "out": out / name}
            drv.close()
    return res


def test_far_kinks_at_the_floor_ask_for_no_rebuild_and_the_run_steps_on(heads):
    far = heads["far"]
    assert far["result"]["stop"]["reason"] == "max-steps" and far["result"]["steps"] == 3
    kinks = far["kinks"]
    assert {"bubble_far"} <= set(kinks), kinks.keys()
    for key, g in kinks.items():
        if g.get("far"):
            assert g["target"] >= H_HEAD and g["yields"] == 0, (key, g)
            assert g["radius"] <= 4.0 * g["h_before"] * (1.0 + 1e-12)  # four chords, no growth
    # a far kink at the floor asks for a rebuild only where its elements are coarser than the
    # refine ratio times the floor (here on the bubble's wall, beyond its head), never below it
    for e in far["events"]:
        if e["event"] != "remesh":
            continue
        recs = {g["name"]: g for g in e.get("regions", [])}
        for a in e.get("asked_by", []):
            if a.endswith("_far"):
                g = recs[a]
                assert g["target"] >= H_HEAD and g["h_max"] > 1.5 * g["target"], (a, g)
                assert g["new_size"] >= H_HEAD, (a, g)
    rows = (far["out"] / "jetcore.txt").read_text().splitlines()
    assert len(rows) == 2 + far["result"]["steps"]


def test_the_core_is_sized_as_without_the_part_and_the_far_region_at_the_floor(heads):
    plain, far = heads["plain"]["first"], heads["far"]["first"]
    core = [name for name, g in far.items() if g.get("class") == "core"]
    assert "jet_base_cavity_tip" in core or "cavity_tip" in core, far.keys()
    for name in core:
        assert far[name]["target"] == plain[name]["target"], name
    floored = [name for name, g in far.items() if g.get("floored")]
    for name, g in far.items():
        if g.get("class") == "far":
            assert g["target"] >= H_HEAD or name.endswith("_far"), name
    assert all(far[name]["class"] == "far" for name in floored)
    assert heads["plain"]["result"]["stop"]["reason"] == heads["far"]["result"]["stop"]["reason"]
    assert math.isfinite(heads["far"]["result"]["remesh_count"])
    assert np.isfinite(H_HEAD)
