"""Contract tests of the plumbing of the events block (``collapse.reconnect_events``): the runner's
admission of a case with it (cut, post-cut continuation, the class keys fixed after an event, the
state before an event, refusals before any output), the dumps and legs of a run after the cut with
it, and the frozen sizing of the bubble's bottom. No finite-element problem is built."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_runner import CASE, ROOT
from test_runner_manifest import RECON, _parent_run, _prototype, _write_meta

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import run_case  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity.driver import CollapseDriver  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

LIQUID = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False}}


def _cut_case(**collapse):
    case = json.loads(json.dumps(CASE))
    case["collapse"]["reconnect"] = RECON.to_dict()
    case["collapse"].update(collapse)
    return case


def _write(tmp_path, case, name):
    path = tmp_path / name
    path.write_text(json.dumps(case))
    return path


def _admit(state, case, path):
    """``restart_admission`` as the runner calls it, the case's blocks parsed as it parses them."""
    block, events_block = ev.split_events_block(case["collapse"])
    _, recon, _, _ = rc.split_collapse_block(block)
    events = None if events_block is None else ev.EventSettings.from_dict(events_block)
    return run_case.restart_admission(state, case, path,
                                      None if recon is None else ReconnectSettings.from_dict(recon),
                                      events=events)


def _check(case_path, out, state=None):
    cmd = [sys.executable, str(ROOT / "simulationCases/run_faraday.py"), str(case_path), "--out",
           str(out), "--check"] + (["--restart-from", str(state)] if state is not None else [])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)


# --- the runner -----------------------------------------------------------------------------------


def test_a_state_written_after_a_cut_before_the_block_continues_with_or_without_it(tmp_path):
    """A post-cut dump of a run without the block (as every run before it) continues under the
    case of its cut, and under one that adds the block: a post-cut continuation, whose lineage
    names the block among the keys that differ (``POST_CUT_DERIVED_KEYS``)."""
    cut_case = _cut_case()
    state, _ = _parent_run(tmp_path, case=cut_case)
    _write_meta(state, cut={"delta": RECON.delta, "t_cut": 0.5 - 0.5e-4,
                            "settings": RECON.to_dict()})
    path = tmp_path / "parent-case.json"
    record, mode = _admit(state, cut_case, path)
    assert mode == run_case.POST_CUT and "derived_keys" not in record["parent_record"]
    child = _cut_case(reconnect_events=LIQUID)
    child["case_id"] = "sigterm-events"
    child_path = _write(tmp_path, child, "events.json")
    record, mode = _admit(state, child, child_path)
    assert mode == run_case.POST_CUT
    assert record["parent_record"]["derived_keys"] == ["case_id", "collapse.reconnect_events"]
    proc = _check(child_path, tmp_path / "out", state)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1])["restart_mode"] == "post-cut"
    assert not (tmp_path / "out").exists()
    assert run_case.POST_CUT_DERIVED_KEYS == run_case.DERIVED_KEYS + ("collapse.reconnect_events",)
    assert run_case.CUT_EVENTS_DERIVED_KEYS == run_case.CUT_DERIVED_KEYS + (
        "collapse.reconnect_events",)


def test_a_state_written_with_the_block_needs_it(tmp_path):
    """A dump written with the block cannot be continued by a case without it, which would fall
    back to the stops of the cut alone."""
    with_block = _cut_case(reconnect_events=LIQUID)
    state, _ = _parent_run(tmp_path, case=with_block)
    s = ev.EventSettings.from_dict(LIQUID)
    _write_meta(state, cut={"delta": RECON.delta, "settings": RECON.to_dict()},
                events_settings=s.to_dict(), events=[], legs=[])
    plain = _cut_case()
    plain["case_id"] = "sigterm-plain"
    path = _write(tmp_path, plain, "plain.json")
    with pytest.raises(SystemExit, match="needs a case with that block"):
        _admit(state, plain, path)
    assert _admit(state, with_block, tmp_path / "parent-case.json")[1] == run_case.POST_CUT


def test_the_keys_of_a_class_are_fixed_after_its_first_event_but_from_the_state_before_one(
        tmp_path):
    """Once a liquid event was made, a continuation may not change the liquid keys, but the keys of
    the leg (post) it may; from the state before an event (precut_e<k>) the class keys may change,
    applying from that event on."""
    block = LIQUID
    parent = _cut_case(reconnect_events=block)
    state, _ = _parent_run(tmp_path, case=parent)
    s = ev.EventSettings.from_dict(block)
    made = {"cut": {"delta": RECON.delta, "settings": RECON.to_dict()},
            "events_settings": s.to_dict(), "events": [{"k": 1, "kind": "liquid-neck"}], "legs": []}
    _write_meta(state, **made)
    changed = _cut_case(reconnect_events={"liquid": {"enabled": True, "eps_d": 0.1},
                                          "post": {"open_neck_stop": False}})
    changed["case_id"] = "sigterm-eps"
    path = _write(tmp_path, changed, "eps.json")
    with pytest.raises(SystemExit, match="liquid.eps_d"):
        _admit(state, changed, path)
    proc = _check(path, tmp_path / "out", state)
    assert proc.returncode != 0 and "liquid.eps_d" in proc.stderr
    assert not (tmp_path / "out").exists()
    leg = _cut_case(reconnect_events={"liquid": {"enabled": True},
                                      "post": {"open_neck_stop": False, "h_floor": 1e-7}})
    leg["case_id"] = "sigterm-leg"
    assert _admit(state, leg, _write(tmp_path, leg, "leg.json"))[1] == run_case.POST_CUT
    # from the state before event 2, the liquid keys apply anew
    _write_meta(state, **made, cut_stage=ev.PRECUT_EVENT, event={"k": 2, "kind": "liquid-neck"})
    record, mode = _admit(state, changed, path)
    assert mode == run_case.POST_CUT
    assert record["parent_record"]["derived_keys"] == ["case_id",
                                                       "collapse.reconnect_events.liquid.eps_d"]
    # but it must say which event it precedes
    _write_meta(state, **made, cut_stage=ev.PRECUT_EVENT)
    with pytest.raises(SystemExit, match="records no event number"):
        _admit(state, changed, path)


def test_the_block_applies_after_a_cut_only(tmp_path):
    """The block is refused, before any output, in a case without a reconnect block, in a run that
    starts from t = 0 or restarts ordinarily before any cut; a cut restart takes it with the
    reconnect block it adds."""
    no_recon = json.loads(json.dumps(CASE))
    no_recon["collapse"]["reconnect_events"] = {}
    out = tmp_path / "out"
    proc = _check(_write(tmp_path, no_recon, "no-recon.json"), out)
    assert proc.returncode != 0 and "needs a collapse.reconnect block" in proc.stderr
    fresh = _cut_case(reconnect_events={})
    proc = _check(_write(tmp_path, fresh, "fresh.json"), out)
    assert proc.returncode != 0 and "only a cut or post-cut restart takes it" in proc.stderr
    # an ordinary restart: the parent's own reconnect block, before any cut
    parent = _cut_case()
    state, _ = _parent_run(tmp_path / "p", case=parent)
    child = _cut_case(reconnect_events={}, landing=[1e-3])
    child["case_id"] = "sigterm-ordinary"
    path = _write(tmp_path, child, "ordinary.json")
    with pytest.raises(SystemExit, match="applies after a cut"):
        _admit(state, child, path)
    # a cut restart from a landing state takes it, with the reconnect block it adds
    landing_parent, _ = _parent_run(tmp_path / "l")
    _write_meta(landing_parent, landing={"delta": RECON.delta, "r_neck": 1e-3})
    cut = _cut_case(reconnect_events=LIQUID)
    cut["case_id"] = "sigterm-cut"
    cut_path = _write(tmp_path, cut, "cut.json")
    record, mode = _admit(landing_parent, cut, cut_path)
    assert mode == run_case.CUT
    assert record["parent_record"]["derived_keys"] == ["case_id", "collapse.reconnect",
                                                       "collapse.reconnect_events"]
    proc = _check(cut_path, out, landing_parent)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1])["restart_mode"] == "cut"
    assert not out.exists()
    # the prototype takes none
    proc = _prototype(cut_path, "--state", landing_parent, "--outdir", out, "--delta-ratio",
                      "1.05", "--check")
    assert proc.returncode != 0 and "takes no collapse.reconnect_events" in proc.stderr
    assert not out.exists()


@pytest.mark.parametrize("collapse, message", [
    ({"reconnect_events": {"post": {"courant_mode": "ale"}}}, "post: unknown key"),
    ({"reconnect_events": {"liquid": {"enabled": True}}}, "open_neck_stop false"),
    ({"reconnect_events": []}, "must be an object"),
    ({"reconnect_events": {"post": {"minimum_dt": 9e-13}}}, "at most half the step-size floor"),
    ({"fold_stopp": 1e-4}, "collapse: .*fold_stopp"),
])
def test_the_runner_refuses_an_invalid_block_or_collapse_key_before_output(tmp_path, collapse,
                                                                           message):
    """An unknown or invalid key of the block, and an unknown key of the collapse block (a
    ``TypeError`` of the driver's settings), are refused with a message before any output."""
    case = _cut_case(**collapse)
    out = tmp_path / "out"
    proc = _check(_write(tmp_path, case, "bad.json"), out)
    assert proc.returncode != 0 and "Traceback" not in proc.stderr, proc.stderr[-2000:]
    assert __import__("re").search(message, proc.stderr), proc.stderr[-2000:]
    assert not out.exists()


# --- the driver: dumps and legs -------------------------------------------------------------------


def _events_driver(tmp_path, pb, recon, events=None, leg_info=None, **kw):
    """``test_bookkeeping._cut_driver`` with the events block."""
    from test_bookkeeping import CUT_GRADING, _settings
    from faraday_singularity.reconnect import Chain, cut_geometry

    drv = CollapseDriver(pb, _settings(**{"t_end": 1.0, "dt_max": 1.0, "max_steps": 3, **kw}),
                         tmp_path, reconnect=recon, events=events, leg_info=leg_info)
    chains = [Chain(np.column_stack([c["r"], c["z"]]), c["end_types"])
              for c in pb.interface_segments()]
    geom = cut_geometry(chains, 3.0, gas_before=1.0)
    pb.cut = {"delta": recon.delta, "rmin": recon.delta, "r_neck": 0.95e-3, "z_neck": 4.685,
              "t_cut": 0.0, "pinch": {"t_p": -1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": recon.tip_size * recon.delta,
                         "h_bridge": recon.bridge_size * recon.delta,
                         "curvature": recon.curvature_sizing, "grading": CUT_GRADING},
              "settings": json.loads(json.dumps(recon.to_dict()))}
    drv.reset_after_cut(pb.cut)
    return drv


def _events(outdir: Path) -> list[dict]:
    return [json.loads(x) for x in (outdir / "events.jsonl").read_text().splitlines()]


def test_dumps_after_a_cut_carry_the_block_its_events_and_legs_only_with_it(tmp_path):
    from test_bookkeeping import CutProblem

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    plain = _events_driver(tmp_path / "plain", CutProblem(), recon, state_dt=0.0)
    plain.run(1e-9)
    plain.close()
    s = ev.EventSettings.from_dict({"post": {"h_floor": 7.8125e-7}})
    drv = _events_driver(tmp_path / "block", CutProblem(), recon, events=s,
                         leg_info={"commit": "c" * 40}, state_dt=0.0)
    drv.run(1e-9)
    drv.close()
    a = json.loads(max((tmp_path / "plain" / "states").glob("state_*.json")).read_text())
    b = json.loads(max((tmp_path / "block" / "states").glob("state_*.json")).read_text())
    assert not {"events_settings", "events", "legs"} & set(a)
    assert set(b) - set(a) == {"events_settings", "events", "legs"}
    assert b["events_settings"] == s.to_dict() and b["events"] == []
    leg = b["legs"][0]
    assert len(b["legs"]) == 1 and leg["start"] == "cut" and leg["leg"] == 0
    assert leg["commit"] == "c" * 40 and leg["settings_sha256"] == s.sha256()
    assert leg["tau"] == 0.0 and leg["step"] == 0
    assert [e["event"] for e in _events(tmp_path / "block")].count("leg") == 1
    assert "leg" not in [e["event"] for e in _events(tmp_path / "plain")]


def test_a_restart_with_the_block_starts_a_leg_and_mirrors_the_runner_refusals(tmp_path,
                                                                             monkeypatch):
    from types import SimpleNamespace

    from test_bookkeeping import CutProblem, _settings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", SimpleNamespace(
        restore_cut=lambda problem, record: setattr(problem, "cut", record)))
    plain = _events_driver(tmp_path / "plain", CutProblem(), recon, state_dt=0.0)
    plain.run(1e-9)
    plain.close()
    legacy = json.loads(max((tmp_path / "plain" / "states").glob("state_*.json")).read_text())
    s = ev.EventSettings.from_dict({"post": {"h_floor": 7.8125e-7}})
    # a state written before the block: the block is added, at the first leg the state records
    pb = CutProblem()
    drv = CollapseDriver(pb, _settings(), tmp_path / "a", reconnect=recon, events=s,
                         leg_info={"commit": "d" * 40})
    drv.restore(legacy)
    drv.close()
    leg = drv.state.legs[-1]
    assert leg["start"] == "restart" and leg["added"] and leg["changed"] == ["post.h_floor"]
    assert leg["commit"] == "d" * 40 and pb.events == []
    # a state written with the block, under a driver without it
    with_block = _events_driver(tmp_path / "block", CutProblem(), recon, events=s, state_dt=0.0)
    with_block.run(1e-9)
    with_block.close()
    meta = json.loads(max((tmp_path / "block" / "states").glob("state_*.json")).read_text())
    with pytest.raises(RuntimeError, match="only by a driver with that block"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "b", reconnect=recon).restore(meta)
    # the leg keys change: a new leg that names them
    later = ev.EventSettings.from_dict({"post": {"h_floor": 5e-7, "dt_min": 1e-14}})
    drv = CollapseDriver(CutProblem(), _settings(), tmp_path / "c", reconnect=recon, events=later)
    drv.restore(meta)
    drv.close()
    assert [x["leg"] for x in drv.state.legs] == [0, 1]
    assert drv.state.legs[1]["changed"] == ["post.dt_min", "post.h_floor"]
    assert not drv.state.legs[1]["added"]
    # the class keys after an event of the class, but from the state before an event
    meta_made = dict(meta, events=[{"k": 1, "kind": "groove-fill"}])
    other = ev.EventSettings.from_dict({"gap": {"kappa_g": 0.1}})
    with pytest.raises(ValueError, match="gap.kappa_g"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "d", reconnect=recon,
                       events=other).restore(meta_made)
    drv = CollapseDriver(CutProblem(), _settings(), tmp_path / "e", reconnect=recon, events=other)
    drv.restore(dict(meta_made, cut_stage=ev.PRECUT_EVENT, event={"k": 2}))
    drv.close()
    assert drv.state.legs[-1]["event_keys_changed_at"] == 2
    assert drv.state.legs[-1]["from_event_state"] == 2
    assert drv.pb.events == [{"k": 1, "kind": "groove-fill"}]
    # and no block without a cut
    with pytest.raises(ValueError, match="needs the reconnect settings"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "f", events=s)


# --- the bubble's bottom, frozen ------------------------------------------------------------------


def test_a_frozen_bubble_bottom_is_never_coarser_than_its_size_whatever_the_landing():
    """``post.bubble_bottom_sizing`` ``frozen`` replaces the landing's resolution at the bubble's
    bottom (``focus.kept``) by a fixed size: at the kept size it is the kept rule itself; a size
    coarser than the landing's coarsens the bottom to it (over the region the landing resolved),
    caps the base of its jet there, and a jet narrower than it still asks for its own size."""
    from test_reconnect_geometry import _post_cut_segs, _regions

    s = rc.ReconnectSettings(delta=1e-3)
    segs = _post_cut_segs(bottom=(2e-3, 5e-3))
    z_bb, z_bt = float(segs[0]["z"][0]), float(segs[0]["z"][-1])
    kept = rc.kept_region({"size": 4e-5, "radius": 0.02, "grading": 0.2}, 6.25e-5, z_bb, z_bt)
    cut = {"delta": 1e-3, "rmin": 1e-3, "focus": {"kept": kept}}
    base = {"bubble_bottom": {"r": 5e-3, "z": 4.49, "R_apex": 3.2e-4, "R_pred": 3.2e-4}}
    legacy = _regions(segs, cut, s, h_floor=6.25e-6, apexes=base)
    same = _regions(segs, cut, s, h_floor=6.25e-6, apexes=base, bubble_bottom_sizing="frozen",
                    bubble_bottom_size=4e-5)
    assert same == legacy  # frozen at the kept size: the kept rule
    assert legacy == _regions(segs, cut, s, h_floor=6.25e-6, apexes=base,
                              bubble_bottom_sizing="kept")
    frozen = _regions(segs, cut, s, h_floor=6.25e-6, apexes=base, bubble_bottom_sizing="frozen",
                      bubble_bottom_size=1.6e-4)
    g, h = frozen["bubble_bottom"], legacy["bubble_bottom"]
    assert g["target"] == 1.6e-4 and g["kept"] == 1.6e-4 and h["target"] == 4e-5
    assert g["radius"] == h["radius"] and g["z"] == h["z"] == z_bb
    # the base of the jet there: capped at the bottom's size, so finer than 3.2e-4/16 only below it
    assert frozen["jet_base_bubble_bottom"]["target"] == pytest.approx(3.2e-4 / 16)
    assert "jet_base_bubble_bottom" not in _regions(
        segs, cut, s, h_floor=6.25e-6, apexes={"bubble_bottom": dict(base["bubble_bottom"],
                                                                     R_apex=3.2e-3, R_pred=3.2e-3)},
        bubble_bottom_sizing="frozen", bubble_bottom_size=1.6e-4)
    # a narrow jet asks for its own size, below the frozen one
    narrow = _regions(_post_cut_segs(bottom=(2e-4, 1e-4), n=40001), cut, s, h_floor=6.25e-6,
                      bubble_bottom_sizing="frozen", bubble_bottom_size=1.6e-4)["bubble_bottom"]
    assert narrow["target"] == pytest.approx(max(narrow["ell"] / 16, 6.25e-6))
    assert narrow["target"] < 1.6e-4
    # without a landing focus: frozen caps the region of the jet; no jet, no region
    alone = _regions(segs, {"delta": 1e-3, "rmin": 1e-3}, s, h_floor=6.25e-6,
                     bubble_bottom_sizing="frozen", bubble_bottom_size=1e-4)["bubble_bottom"]
    assert alone["target"] == 1e-4 and alone["radius"] < 0.5 * (z_bt - z_bb) + 1e-12
    sphere = _post_cut_segs()
    assert "bubble_bottom" not in _regions(sphere, {"delta": 1e-3, "rmin": 1e-3}, s,
                                           bubble_bottom_sizing="frozen",
                                           bubble_bottom_size=1e-4)
    with pytest.raises(ValueError, match="unknown bubble_bottom_sizing"):
        _regions(segs, cut, s, bubble_bottom_sizing="own")


def test_the_driver_sizes_the_bubble_bottom_by_the_block(tmp_path):
    """The driver passes the block's sizing of the bubble's bottom to its regions after the cut."""
    from test_bookkeeping import CutSurfaceProblem
    from faraday_singularity.mesh import Focus

    recon = ReconnectSettings(delta=1e-2, dt0=1e-6)
    out = {}
    for name, events in (("kept", None), ("frozen", ev.EventSettings.from_dict(
            {"post": {"bubble_bottom_sizing": "frozen", "bubble_bottom_size": 3e-3}}))):
        size = recon.focus_size * recon.delta
        pb = CutSurfaceProblem(recon.tip_size * recon.delta, size, bottom=(2e-4, 1e-3))
        pb.template.focus = Focus(size=size, radius=0.2, grading=0.2)
        drv = _events_driver(tmp_path / name, pb, recon, events=events, max_steps=0)
        drv.pb.cut["focus"] = {"kept": {"name": "bubble_bottom", "size": 1e-3, "radius": 0.02}}
        regions, _ = drv._feature_regions({"z_tip": 4.69, "kappa": 1 / 0.08, "r_j": 0.0,
                                           "rise": 0.0, "bump": False, "flat_bump": False},
                                          drv._bubble_features()[1])
        out[name] = {g["name"]: g for g in regions}
        drv.close()
    assert out["kept"]["bubble_bottom"]["kept"] == 1e-3
    assert out["frozen"]["bubble_bottom"]["kept"] == 3e-3
    assert out["frozen"]["bubble_bottom"]["target"] <= 3e-3
    assert math.isclose(out["frozen"]["bubble_bottom"]["radius"],
                        out["kept"]["bubble_bottom"]["radius"])
