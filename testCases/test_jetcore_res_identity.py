"""The core of the upward jet meshed finer (``jetcore.core_elements_per_scale``) against 3e00e92, the
commit before it: the source of this commit and that of 3e00e92, each in a fresh process with MKL on
one thread. Runs without the key (null, its default) are 3e00e92's to the byte:

* the stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery;
* the production events block (every part enabled as the runs of the delta study set it, jetcore
  with its far sizing and far Courant number) and variants of it: the block as its records hold it,
  its hash and the keys a restart changes, so that every leg and dump recorded under 3e00e92 is
  continued as it was;
* the finite-element run after the cut of the synthetic necked cavity under the production block,
  and a restart in a new process from one of its dumps: every output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with ``ring.enabled`` and jetcore's far
  sizing: the ring cut made by the driver and the steps of the ring stage;
* the far-head cut of ``test_jetcore_fem.py``'s heads' state made by the driver under jetcore's far
  sizing, and the steps after it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_events_fem as fem_pins  # noqa: E402
from test_events_fem import _reference_src  # noqa: E402
from test_events_gasneck import BATTERY, _run  # noqa: E402
from test_far_head_first_identity import FAR_HEAD_RUN  # noqa: E402
from test_jetcore_identity import RING_RUN_JETCORE, _run_fem, _spec  # noqa: E402
from test_ring_identity import _drop_wall_column  # noqa: E402

BASE = "3e00e92"  # the commit before the core's elements per scale

# the events block of the runs of the delta study (every key written out)
PRODUCTION = {
    "cascade_count": 3, "cascade_min_interval": 0.25, "cascade_ratio": 0.7, "max_events": 1000,
    "gap": {"kappa_g": 0.05},
    "liquid": {"box_radius": 4.0, "cap": "hemispheroid", "cap_fill": "sample", "cap_height": 1.0,
               "enabled": True, "eps_d": 0.2, "eps_max": 0.5, "max_head_volume": 20.0,
               "projection": True, "resolution": 4.0, "restart": None, "sigma_KE": 0.0,
               "sigma_Pz": 0.0, "sigma_V": 0.0, "track_depth": 0.05},
    "post": {"bubble_bottom_size": None, "bubble_bottom_sizing": "kept", "dt_min": None,
             "h_floor": None, "keep_event_dumps": "all", "log_box_noise": False,
             "minimum_dt": None, "neck_elements_per_scale": None, "neck_floor": 7.8125e-07,
             "open_neck_stop": False, "prune_gmsh": True, "rise_tol": None},
    "ring": {"enabled": True, "gap": 1.0, "rho": 0.5, "waist": 0.5, "cut_elements": 4.0,
             "fill": "blend", "projection": True, "restart": None, "box_radius": 4.0,
             "sigma_V": 1.13e-13, "sigma_Pz": 3.24e-10, "sigma_KE": 2.02e-07, "size": None,
             "axis_ratio": 0.5, "after_removal": "stop"},
    "jetcore": {"enabled": True, "core_radius": 2.0, "wall_radius": 10.0, "far_courant": 1.6,
                "far_floor": 1.5625e-05, "far_kinks": "recentre", "far_yield_turn": 75.0,
                "far_refine_ratio": 2.0, "ring_kink_floor": True, "edge_turn_snapshots": "core",
                "prep_gap": 4.0, "prep_rho": 0.125, "profile_dt": 2.5e-07, "box_gate": False,
                "box_gate_sigmas": 5.0},
    "pinch": {"enabled": True, "r_c": 0.2, "window": 0.05, "min_samples": 8, "kappa": 3.0,
              "far_head": True},
    "groove": {"enabled": True, "mouth": 1.5, "cap_height": 1.0, "cut_elements": None,
               "fill": "blend", "projection": True, "restart": None, "box_radius": 4.0,
               "sigma_V": 1.13e-13, "sigma_Pz": 3.24e-10, "sigma_KE": 2.02e-07, "max_fill": 0.01,
               "sheets": "log", "sheet_contact": 1.0}}

BLOCKS = """
import json, sys
from pathlib import Path
src, tests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, src)
import faraday_singularity
assert Path(faraday_singularity.__file__).resolve().is_relative_to(Path(src).resolve())
from faraday_singularity import events as ev
PRODUCTION = %r
blocks = {"production": PRODUCTION,
          "jetcore-defaults": {**PRODUCTION, "jetcore": {"enabled": True}},
          "no-jetcore": {k: v for k, v in PRODUCTION.items() if k not in ("jetcore", "pinch")},
          "empty": {}}
res = {}
for name, b in blocks.items():
    s = ev.EventSettings.from_dict(json.loads(json.dumps(b)))
    res[name] = {"dict": s.to_dict(), "sha256": s.sha256(),
                 "changed_from_none": s.changed_keys(None),
                 "changed_from_production": s.changed_keys(PRODUCTION),
                 "sizing": sorted(set(s.changed_keys(None)) & set(ev.SIZING_KEYS)
                                  - {"jetcore.core_elements_per_scale"})}
out.write_text(json.dumps(res, sort_keys=True))
""" % (PRODUCTION,)

# the far-head cut under jetcore's far sizing (the pin of test_far_head_first_identity with it)
FAR_HEAD_FAR = FAR_HEAD_RUN.replace(
    "block = ev.EventSettings.from_dict({**J.FAR_HEAD_BLOCK, **GROOVE})",
    "block = ev.EventSettings.from_dict({**J.FAR_HEAD_BLOCK, **GROOVE, 'jetcore': {"
    "'enabled': True, 'far_floor': J.H_HEAD, 'far_kinks': 'recentre', 'far_refine_ratio': 2.0,"
    " 'far_courant': 1.6, 'edge_turn_snapshots': 'core'}})")
assert FAR_HEAD_FAR != FAR_HEAD_RUN

# the ring run with jetcore's far sizing of the ring (the column's floor on its kinks)
RING_RUN_FAR = RING_RUN_JETCORE.replace(
    '{"jetcore": {"enabled": True}}',
    '{"jetcore": {"enabled": True, "far_kinks": "recentre", "ring_kink_floor": True, '
    '"far_floor": 1e-4, "far_refine_ratio": 2.0}}')
assert RING_RUN_FAR != RING_RUN_JETCORE


def test_the_stops_are_3e00e92s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_the_production_block_its_hash_and_its_changed_keys_are_3e00e92s(tmp_path):
    """The block of the runs of the delta study, and with jetcore at its defaults, without jetcore
    and empty: as their records hold them, their hashes, the keys they change against a state
    without a block and against the production block, and the sizing keys among them, as 3e00e92
    reads them."""
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BLOCKS, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert "core_elements_per_scale" not in json.dumps(out["head"])


def test_a_finite_element_run_under_the_production_block_and_its_restart_are_3e00e92s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    spec = _spec(**PRODUCTION)
    runs = {name: _run_fem(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["remesh_count"] >= 2, result
    assert {"budget.txt", "jetcore.txt"} <= set(names)
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, tmp_path / f"{name}-restart", r)
    fem_pins._same_outputs(again["ref"], again["head"])


def test_a_ring_run_with_jetcores_far_sizing_is_3e00e92s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_FAR, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 4, result["stop"]
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    assert {"ring.txt", "jetcore.txt", "events/event_001.json"} <= set(names)


def test_the_far_head_cut_under_jetcores_far_sizing_is_3e00e92s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    script = FAR_HEAD_FAR % '{"groove": {"enabled": True}}'
    runs = {name: _run(script, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["events"]["made"] == 1 and result["events"]["by_kind"] == {"far-head": 1}
    assert {"events.txt", "jetcore.txt", "events/event_001.json"} <= set(names)
