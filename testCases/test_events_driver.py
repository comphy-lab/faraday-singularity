"""Contract tests of the collapse driver after a cut with the events block, on stand-in problems
whose two chains are prescribed (``test_bookkeeping.CutProblem``): the stops of the events branch,
an event without a handler as a resumable stop with the dump of the state before it, and a run
that goes on from that dump. No finite-element problem is built."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_bookkeeping import CutProblem, _settings
from test_events_plumbing import _admit, _events, _events_driver
from test_events_stops import bubble_with_head
from test_runner import CASE, ROOT
from test_runner_manifest import RECON

sys.path.insert(0, str(ROOT / "src"))
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity.driver import CollapseDriver  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

LIQUID = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 2e-6}}


class HeadProblem(CutProblem):
    """``CutProblem`` whose bubble is ``test_events_stops.bubble_with_head`` at delta = 1e-3: a jet
    hanging from its top into it, whose neck closes from ``r_n(t)`` delta, its head 0.5 delta
    wide."""

    def __init__(self, r_n=lambda t: 0.3 if t <= 0.0 else 0.09, **kw):
        super().__init__(**kw)
        self.r_n = r_n
        self._cache = None

    def interface_segments(self):
        r_n = self.r_n(self.t)
        if self._cache is None or self._cache[0] != r_n:
            chain, _ = bubble_with_head(1e-3, r_n=r_n)
            self._cache = (r_n, chain)
        segs = super().interface_segments()
        P = self._cache[1].points
        segs[0] = self._chain("bubble", P[:, 0].copy(), P[:, 1].copy(), ("axis", "axis"))
        return segs


def test_without_the_block_the_neck_of_a_jet_below_delta_stops_the_run_at_once(tmp_path):
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    out = _events_driver(tmp_path, HeadProblem(), recon).run(1e-9)
    assert out["stop"]["reason"] == "secondary-pinch-approach" and out["steps"] == 0
    assert out["stop"]["neck_side"] == "liquid" and "precut" not in out["stop"]


def test_an_event_without_a_handler_stops_the_run_resumably(tmp_path, monkeypatch):
    """With liquid events the neck of the hanging jet is followed until it lands (after the first
    step, at 0.09 delta with a head 0.5 delta wide); the event has no handler yet: the run stops on
    it (``event-unhandled``, event 1, kind ``liquid-neck``) and keeps the state before it,
    ``precut_e001`` (``cut_stage`` ``precut-event``), besides the end state. A continuation from
    that dump is admitted after the cut, may change the keys of the event's class, and meets the
    event again at once while it has no handler."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    events = ev.EventSettings.from_dict(LIQUID)
    run = tmp_path / "run"
    out = _events_driver(run, HeadProblem(), recon, events=events, state_keep=10).run(1e-9)
    stop = out["stop"]
    assert stop["reason"] == ev.EVENT_UNHANDLED and out["steps"] == 1
    assert (stop["kind"], stop["class"], stop["handler"], stop["k"]) == (
        "liquid-neck", "L", "liquid-cut", 1)
    assert stop["chain"] == "bubble" and stop["ratio"] == pytest.approx(0.9, rel=1e-3)
    precut, end = Path(stop["precut"]), Path(stop["end_state"])
    assert precut.name == "precut_e001.dump" and precut.is_file() and end.is_file()
    meta = json.loads(precut.with_suffix(".json").read_text())
    assert meta["cut_stage"] == ev.PRECUT_EVENT and meta["event"]["k"] == 1
    assert meta["event"]["kind"] == "liquid-neck" and meta["event"]["index"] == stop["index"]
    assert meta["events"] == [] and meta["events_settings"] == events.to_dict()
    assert meta["step"] == 1 and meta["resume_at"] == "remesh-check"
    assert [e["event"] for e in _events(run)][-1] == "stop"
    # the runner admits a continuation from it (the parent's records, as run_case writes them)
    case = json.loads(json.dumps(CASE))
    case["collapse"].update(reconnect=dict(RECON.to_dict(), delta=1e-3, dt0=1e-6),
                            reconnect_events=LIQUID)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    (run / "run-manifest.json").write_text(json.dumps(
        {"schema": "pyoomph-runner-manifest-v1", "component_commit": "a" * 40,
         "case_sha256": digest}))
    (run / "parameters.json").write_text(json.dumps(
        {"case": case, "provenance": {"case_sha256": digest}}))
    assert _admit(precut, case, case_path)[1] == "post-cut"
    eps = json.loads(json.dumps(case))
    eps["case_id"] = "sigterm-eps"
    eps["collapse"]["reconnect_events"] = {"liquid": {"enabled": True, "eps_d": 0.1},
                                           "post": {"open_neck_stop": False, "neck_floor": 2e-6}}
    eps_path = tmp_path / "eps.json"
    eps_path.write_text(json.dumps(eps))
    record, mode = _admit(precut, eps, eps_path)
    assert mode == "post-cut" and record["parent_record"]["derived_keys"] == [
        "case_id", "collapse.reconnect_events.liquid.eps_d"]
    # the continuation: the same event, again without a handler, at once
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", SimpleNamespace(
        restore_cut=lambda problem, record: setattr(problem, "cut", record)))
    pb = HeadProblem()
    pb.t = meta["time"]
    again = CollapseDriver(pb, _settings(t_end=1.0, dt_max=1.0, max_steps=3), tmp_path / "again",
                           reconnect=recon, events=events)
    again.restore(meta)
    res = again.run(1e-9)
    again.close()
    assert res["stop"]["reason"] == ev.EVENT_UNHANDLED and res["steps"] == 1
    assert res["stop"]["k"] == 1 and Path(res["stop"]["precut"]).name == "precut_e001.dump"
    leg = again.state.legs[-1]
    assert leg["start"] == "restart" and leg["from_event_state"] == 1 and leg["changed"] == []


def test_a_restart_meets_a_dumped_mesh_no_step_ran_on_as_fresh(tmp_path, monkeypatch):
    """A tracked neck of 0.02 delta, below 4 element lengths at it, not landed (eps_d 0.02): on the
    mesh of the cut, which no step has run on, the run stops at once (``liquid-neck-unresolved``,
    a check of a fresh mesh) and dumps that mesh with ``mesh_fresh``. A continuation from the dump
    meets it as fresh again and stops at once, instead of stepping on it until a rebuild; one that
    changes a key sizing the mesh (``events.SIZING_KEYS``) takes it as stepped on, so that the
    rule of its rebuilds meets the new sizes."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    block = {"liquid": {"enabled": True, "eps_d": 0.02},
             "post": {"open_neck_stop": False, "neck_floor": 2e-6}}
    events = ev.EventSettings.from_dict(block)
    thin = dict(r_n=lambda t: 0.02)
    out = _events_driver(tmp_path / "run", HeadProblem(**thin), recon, events=events).run(1e-9)
    stop = out["stop"]
    assert stop["reason"] == "liquid-neck-unresolved" and stop["fresh"] and out["steps"] == 0
    meta = json.loads(Path(stop["end_state"]).with_suffix(".json").read_text())
    assert meta["mesh_fresh"] is True
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", SimpleNamespace(
        restore_cut=lambda problem, record: setattr(problem, "cut", record)))

    def again(name, block):
        pb = HeadProblem(**thin)
        pb.t = meta["time"]
        drv = CollapseDriver(pb, _settings(t_end=1.0, dt_max=1.0, max_steps=3), tmp_path / name,
                             reconnect=recon, events=ev.EventSettings.from_dict(block))
        drv.restore(meta)
        res = drv.run(1e-9)
        drv.close()
        return drv, res

    drv, res = again("same", block)
    assert res["stop"]["reason"] == "liquid-neck-unresolved" and res["steps"] == 0
    assert res["stop"]["fresh"]
    finer = json.loads(json.dumps(block))
    finer["post"]["neck_floor"] = 1e-6
    drv, res = again("finer", finer)
    assert drv.state.legs[-1]["changed"] == ["post.neck_floor"] and res["steps"] >= 1
    # a mesh a step has run on stays stepped on
    later = _events_driver(tmp_path / "later", HeadProblem(), recon,
                           events=ev.EventSettings.from_dict(LIQUID), state_dt=0.0)
    later.run(1e-9)
    later.close()
    dumps = sorted((tmp_path / "later" / "states").glob("state_*.json"))
    assert [json.loads(d.read_text())["mesh_fresh"] for d in dumps][-1] is False


def test_the_bubble_pierced_by_its_jet_is_a_resumable_stop(tmp_path):
    """The bubble's two tips within delta of each other (the downward jet piercing the bubble):
    without the block a stop (``bubble-jets-approach``); with it an event whose handler, the ring,
    does not exist yet (``event-unhandled``, kind ``piercing``), the state before it kept."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    short = {"a": 1e-3, "b": 4e-4}  # a bubble 8e-4 long
    plain = _events_driver(tmp_path / "plain", CutProblem(**short), recon).run(1e-9)
    assert plain["stop"]["reason"] == "bubble-jets-approach"
    out = _events_driver(tmp_path / "block", CutProblem(**short), recon,
                         events=ev.EventSettings()).run(1e-9)
    stop = out["stop"]
    assert stop["reason"] == ev.EVENT_UNHANDLED and stop["kind"] == "piercing"
    assert stop["class"] == "W1" and stop["handler"] == "ring" and stop["k"] == 1
    assert stop["bubble_length"] == pytest.approx(8e-4, rel=1e-6)
    assert Path(stop["precut"]).name == "precut_e001.dump" and out["steps"] == 0
    assert stop["tau"] == 0.0


def test_a_closing_groove_is_met_at_kappa_g_delta_not_at_the_absolute_fold_stop(tmp_path):
    """A groove 0.3 delta wide around a jet (no neck yet closing): with the collapse's absolute
    fold_stop of 1e-3 the cut's stops end the run on it (a fold), with the block's kappa_g delta =
    5e-5 the run goes on."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)

    class Groove(HeadProblem):
        def interface_segments(self):
            if self._cache is None:
                chain, _ = bubble_with_head(1e-3, r_n=0.42, r_w=0.8)
                self._cache = (0.42, chain)
            return super().interface_segments()

    plain = _events_driver(tmp_path / "plain", Groove(r_n=lambda t: 0.42), recon,
                           fold_stop=1e-3).run(1e-9)
    assert plain["stop"]["reason"] in ("self-contact-approach", "annular-contact-approach")
    assert plain["stop"]["trigger"] == "fold-gap" and plain["steps"] == 0
    out = _events_driver(tmp_path / "block", Groove(r_n=lambda t: 0.42), recon, fold_stop=1e-3,
                         events=ev.EventSettings()).run(1e-9)
    assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3
    rows = np.loadtxt(tmp_path / "block" / "chains.txt", ndmin=2)
    assert len(rows) == 4


def test_the_limits_after_the_cut_are_the_blocks_and_before_it_the_collapses(tmp_path):
    """With the block the driver's floors after the cut are post's (h_floor, rise_tol, dt_min, the
    open chain's neck stop through neck_stop_elements h_floor), each the collapse's own when null;
    before the cut, and without the block, the collapse's. A step below post.dt_min is the
    dt-floor stop."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    post = {"h_floor": 1e-7, "rise_tol": 1e-8, "dt_min": 1e-5, "neck_floor": 2e-7}
    events = ev.EventSettings.from_dict({"post": post})
    pb = CutProblem()
    drv = CollapseDriver(pb, _settings(neck_stop=1e-4, h_floor=1e-5), tmp_path / "pre",
                         reconnect=recon, events=events)
    assert (drv._h_floor(), drv._rise_tol(), drv._dt_min(), drv._fold_stop()) == (
        1e-5, 1e-6, 1e-12, 1e-3)  # no cut yet
    assert drv._neck_limit() == max(1e-4, 3 * 1e-5)
    drv.close()
    drv = _events_driver(tmp_path / "block", CutProblem(), recon, events=events, neck_stop=1e-4,
                         h_floor=1e-5)
    assert (drv._h_floor(), drv._rise_tol(), drv._dt_min(), drv._focus_floor()) == (
        1e-7, 1e-8, 1e-5, 1e-7)
    assert drv._fold_stop() == pytest.approx(0.05 * 1e-3) and drv._neck_limit() == 1e-4
    regions = {g["name"]: g for g in drv.measure()["regions"]}
    assert all(g["floor"] in (1e-7, recon.focus_size * 1e-3) for g in regions.values())
    # the leg records them, resolved, and in units of delta (lengths) and delta^2 (times)
    limits = drv.state.legs[0]["limits"]
    assert limits == {"delta": 1e-3, "h_floor": 1e-7, "neck_floor": 2e-7, "rise_tol": 1e-8,
                      "fold_stop": 0.05 * 1e-3, "neck_limit": 1e-4, "dt_min": 1e-5,
                      "minimum_dt": 1e-12, "in_delta": limits["in_delta"]}  # 1e-8 capped
    assert limits["in_delta"] == {"h_floor": 1e-7 / 1e-3, "neck_floor": 2e-7 / 1e-3,
                                  "rise_tol": 1e-8 / 1e-3, "fold_stop": 0.05 * 1e-3 / 1e-3,
                                  "neck_limit": 1e-4 / 1e-3, "dt_min": 1e-5 / 1e-3**2,
                                  "minimum_dt": 1e-12 / 1e-3**2}
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"] == {**out["stop"], "reason": "dt-floor"} and out["steps"] == 0
    plain = _events_driver(tmp_path / "plain", CutProblem(), recon, neck_stop=1e-4, h_floor=1e-5)
    assert (plain._h_floor(), plain._rise_tol(), plain._dt_min(), plain._fold_stop()) == (
        1e-5, 1e-6, 1e-12, 1e-3)
    assert plain.run(1e-9)["stop"]["reason"] == "max-steps"
    plain.close()
    nulls = _events_driver(tmp_path / "nulls", CutProblem(), recon,
                           events=ev.EventSettings.from_dict({"gap": {"kappa_g": None}}),
                           neck_stop=1e-4, h_floor=1e-5)
    assert (nulls._h_floor(), nulls._rise_tol(), nulls._dt_min(), nulls._fold_stop()) == (
        1e-5, 1e-6, 1e-12, 1e-3)
    nulls.close()


def test_with_liquid_events_the_open_chains_neck_stop_is_off_after_the_cut(tmp_path):
    """The open chain's neck stop (collapse.neck_stop, here its default of 1e-3) would end the run
    on the neck at the root of the jet rising from the cavity's tip, which liquid events follow:
    post.open_neck_stop false drops it after the cut, and the neck is tracked instead."""
    from test_events_stops import cavity_with_head

    class RisingJet(CutProblem):
        def interface_segments(self):
            segs = super().interface_segments()
            P = cavity_with_head(1e-3, r_n=0.3).points
            segs[1] = self._chain("open", P[:, 0].copy(), P[:, 1].copy(), ("axis", "fixed"))
            return segs

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    plain = _events_driver(tmp_path / "plain", RisingJet(), recon).run(1e-9)
    assert plain["stop"]["reason"] == "pinch-off-approach" and plain["steps"] == 0
    kept = _events_driver(tmp_path / "kept", RisingJet(), recon,
                          events=ev.EventSettings()).run(1e-9)
    assert kept["stop"]["reason"] == "pinch-off-approach"  # the default keeps it
    out = _events_driver(tmp_path / "off", RisingJet(), recon,
                         events=ev.EventSettings.from_dict(LIQUID)).run(1e-9)
    assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3


def test_oomph_libs_smallest_step_is_lowered_after_the_cut_with_the_block_only(tmp_path,
                                                                              monkeypatch):
    """At the cut and at every restart after it the driver with the block sets oomph-lib's smallest
    step (``DTSF_minimum_dt``) to post.minimum_dt, or 1e-3 times its step-size floor, and records it
    in the leg; before the cut and without the block it leaves it alone."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    plain = CutProblem()
    _events_driver(tmp_path / "plain", plain, recon).close()
    assert not hasattr(plain, "DTSF_minimum_dt")
    pb = CutProblem()
    drv = CollapseDriver(pb, _settings(), tmp_path / "pre", reconnect=recon,
                         events=ev.EventSettings())
    assert not hasattr(pb, "DTSF_minimum_dt")  # no cut yet
    drv.close()
    pb = CutProblem()
    drv = _events_driver(tmp_path / "floor", pb, recon,
                         events=ev.EventSettings.from_dict({"post": {"dt_min": 1e-14}}))
    # exact: the product the driver sets (approx's absolute tolerance of 1e-12 would pass any)
    assert pb.DTSF_minimum_dt == 1e-3 * 1e-14 and drv.state.legs[0]["minimum_dt"] == \
        pb.DTSF_minimum_dt
    assert drv.state.legs[0]["minimum_dt_ulps"] is None  # the cut at t = 0
    drv.close()
    pb = CutProblem()
    drv = _events_driver(tmp_path / "default", pb, recon, events=ev.EventSettings(), state_dt=0.0)
    assert pb.DTSF_minimum_dt == 1e-3 * 1e-12  # 1e-3 times the collapse's dt_min
    drv.run(1e-9)
    drv.close()
    meta = json.loads(max((tmp_path / "default" / "states").glob("state_*.json")).read_text())
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", SimpleNamespace(
        restore_cut=lambda problem, record: setattr(problem, "cut", record)))
    pb = CutProblem()
    pb.t = meta["time"]
    events = ev.EventSettings.from_dict({"post": {"minimum_dt": 2e-18}})
    drv = CollapseDriver(pb, _settings(), tmp_path / "restart", reconnect=recon, events=events)
    drv.restore(meta)
    drv.close()
    assert pb.DTSF_minimum_dt == 2e-18 and drv.state.legs[-1]["minimum_dt"] == 2e-18
    assert drv.state.legs[-1]["changed"] == ["post.minimum_dt"]
    assert drv.state.legs[-1]["minimum_dt_ulps"] == 2e-18 / math.ulp(meta["time"])


def test_oomph_libs_smallest_step_must_lie_at_most_half_the_floor_after_the_cut(tmp_path):
    """A post.minimum_dt above half the driver's step-size floor after the cut (here the
    collapse's own, post.dt_min null) is refused where the driver is made: a step failing at that
    floor would not be halved, and would end the run as a failed one."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    events = ev.EventSettings.from_dict({"post": {"minimum_dt": 9e-13}})
    with pytest.raises(ValueError, match="at most half the step-size floor"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "a", reconnect=recon, events=events)
    drv = CollapseDriver(CutProblem(), _settings(dt_min=2e-12), tmp_path / "b", reconnect=recon,
                         events=events)
    drv.close()


# --- the readers of the limits after the cut ------------------------------------------------------

# a liquid event block whose post keys all differ from the collapse's own, so that each reader
# shows which value it took
WIRED = {"liquid": {"enabled": True},
         "gap": {"kappa_g": 0.07},
         "post": {"open_neck_stop": False, "h_floor": 3e-7, "rise_tol": 2e-8, "neck_floor": 4e-7,
                  "neck_elements_per_scale": 24.0, "dt_min": 1e-9}}


def _spy(monkeypatch, calls: list, *names):
    """Record every call of the driver's own names (``faraday_singularity.driver``) as
    ``(name, args, kw)``, the call itself unchanged."""
    import faraday_singularity.driver as module

    for name in names:
        real = getattr(module, name)

        def spy(*args, _name=name, _real=real, **kw):
            calls.append((_name, args, kw))
            return _real(*args, **kw)

        monkeypatch.setattr(module, name, spy)


def test_after_the_cut_every_reader_of_a_limit_takes_the_blocks_value(tmp_path, monkeypatch):
    """Each place the driver reads a limit or floor after the cut takes the block's value with the
    block and the collapse's own without it: the rise of a jet (the open chain's jet base, the
    bubble's tips and the bases of their jets), the fold stop of the grooves at the jets' bases
    (``kappa_g delta``), the regions of the features (``h_floor``, ``neck_floor``,
    ``neck_elements_per_scale``), the guard and the kink regions of the regions (``h_floor``), and
    the stops (the guard's floor and freshness, the check of the curved elements on a fresh
    mesh)."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    names = ("jet_base", "tip_jet", "groove_gap", "feature_regions", "interface_guard",
             "kink_regions", "post_cut_stops")
    for block in (WIRED, None):
        calls: list = []
        events = None if block is None else ev.EventSettings.from_dict(block)
        drv = _events_driver(tmp_path / str(block is None), HeadProblem(r_n=lambda t: 0.3), recon,
                             events=events, h_floor=1e-5, rise_tol=1e-6)
        _spy(monkeypatch, calls, *names)
        drv._post_cut_cache = None
        drv.measure()
        drv._post_cut_stop()
        monkeypatch.undo()
        got = {}
        for name, args, kw in calls:
            got.setdefault(name, []).append((args, kw))
        assert set(got) == set(names), set(names) - set(got)
        rise = 2e-8 if block else 1e-6
        assert [a[2] for a, _ in got["jet_base"]] == [rise] * len(got["jet_base"])
        assert [k["rise_tol"] for _, k in got["tip_jet"]] == [rise, rise]
        assert {k["stop"] for _, k in got["groove_gap"]} == {0.07 * 1e-3 if block else 1e-3}
        (_, regions), = got["feature_regions"]
        assert regions["h_floor"] == (3e-7 if block else 1e-5)
        assert {k: regions.get(k) for k in ("neck_floor", "neck_elements_per_scale")} == (
            {"neck_floor": 4e-7, "neck_elements_per_scale": 24.0} if block
            else {"neck_floor": None, "neck_elements_per_scale": None})
        assert {k["h_floor"] for _, k in got["interface_guard"]} == {3e-7 if block else 1e-5}
        assert {k["h_floor"] for _, k in got["kink_regions"]} == {3e-7 if block else 1e-5}
        (_, stops), = got["post_cut_stops"]
        assert stops["guard"] == {"fresh": True, "h_floor": 3e-7 if block else 1e-5,
                                  "refine_ratio": 1.5}
        if block:
            assert stops["events"] is drv.events and stops["invalid"] == drv._jacobian_stop
        else:
            assert "events" not in stops and "invalid" not in stops
        drv.close()


def test_after_a_step_the_stops_take_the_guard_of_a_stepped_mesh_and_no_jacobian_check(
        tmp_path, monkeypatch):
    """With the block the stops of a state are told whether its mesh is fresh, and only a fresh
    one has its curved elements checked: the call after a step that rebuilt nothing has
    ``fresh`` false and no ``invalid``."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    drv = _events_driver(tmp_path, HeadProblem(r_n=lambda t: 0.3), recon,
                         events=ev.EventSettings.from_dict(WIRED), max_steps=1, h_floor=1e-5)
    calls: list = []
    _spy(monkeypatch, calls, "post_cut_stops")
    out = drv.run(1e-9)
    drv.close()
    assert out["steps"] == 1 and out["stop"]["reason"] == "max-steps"
    seen = [(k["guard"]["fresh"], k["invalid"] is not None) for _, _, k in calls]
    assert seen[0] == (True, True) and (False, False) in seen
    assert all(fresh == checked for fresh, checked in seen)


def test_the_open_chains_neck_stop_after_the_cut_has_the_floor_of_the_block(tmp_path):
    """``max(neck_stop, neck_stop_elements h_floor)`` with the block's ``post.h_floor``: a neck of
    the open chain of 3e-4 (the root of a jet rising from the cavity's tip) lies below that limit
    with the collapse's floor of 2e-4 (6e-4), and above it with the block's of 1e-6 (the neck
    stop, 1e-5): the run then meets the neck as an event instead."""
    from test_events_stops import cavity_with_head

    class RisingJet(CutProblem):
        def interface_segments(self):
            segs = super().interface_segments()
            P = cavity_with_head(1e-3, r_n=0.3).points
            segs[1] = self._chain("open", P[:, 0].copy(), P[:, 1].copy(), ("axis", "fixed"))
            return segs

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    kw = {"neck_stop": 1e-5, "h_floor": 2e-4}
    for name, events, limit in (
            ("plain", None, 6e-4), ("collapse-floor", ev.EventSettings(), 6e-4),
            ("block-floor", ev.EventSettings.from_dict({"post": {"h_floor": 1e-6}}), 1e-5)):
        drv = _events_driver(tmp_path / name, RisingJet(), recon, events=events, **kw)
        assert drv._neck_limit() == pytest.approx(limit, rel=1e-12, abs=0.0)
        out = drv.run(1e-9)
        drv.close()
        stop = out["stop"]
        if limit > 3e-4:
            assert stop["reason"] == "pinch-off-approach" and stop["neck_limit"] == drv._neck_limit()
        else:
            assert stop["reason"] == ev.EVENT_UNHANDLED and stop["kind"] == "liquid-neck"
            assert stop["chain"] == "open" and stop["class"] == "L-unhandled"


def test_the_grooves_at_the_jets_bases_are_measured_against_kappa_g_delta(tmp_path):
    """The groove around a jet's base is read with the fold limit after the cut (``groove_gap``'s
    ``stop``, its ``groove_limit``): ``kappa_g delta`` with the block, the collapse's absolute
    ``fold_stop`` without it."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)

    class Groove(HeadProblem):
        def interface_segments(self):
            if self._cache is None:
                chain, _ = bubble_with_head(1e-3, r_n=0.42, r_w=0.8)
                self._cache = (0.42, chain)
            return super().interface_segments()

    for name, events, limit in (("plain", None, 1e-3), ("block", ev.EventSettings(), 5e-5)):
        drv = _events_driver(tmp_path / name, Groove(r_n=lambda t: 0.42), recon, events=events,
                             fold_stop=1e-3, max_steps=1)
        apex = drv._post_cut_state()["apexes"]["bubble_top"]
        drv.run(1e-9)
        drv.close()
        assert apex["groove_limit"] == pytest.approx(limit, rel=1e-12, abs=0.0)
        assert apex["groove_gap"] == pytest.approx(0.37e-3, rel=0.01)


def test_with_the_block_a_fresh_mesh_whose_curved_elements_fail_stops_the_run(tmp_path):
    """The check of the curved elements (``CollapseDriver._jacobian_stop``) is among the stops of
    an invalid state in the events branch: on the cut's mesh, which no step has run on, a check
    that fails stops the run at once (``interface-unresolved``, trigger ``curved-jacobian``),
    before any event."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    bad = {"reason": "interface-unresolved", "trigger": "curved-jacobian", "scaled_jacobian": 0.1}
    for name, events in (("plain", None), ("block", ev.EventSettings())):
        drv = _events_driver(tmp_path / name, CutProblem(), recon, events=events)
        calls = []
        drv._jacobian_stop = lambda: calls.append(1) or dict(bad)
        out = drv.run(1e-9)
        drv.close()
        assert out["steps"] == 0 and calls == [1]
        assert out["stop"]["reason"] == "interface-unresolved"
        assert out["stop"]["trigger"] == "curved-jacobian"


def test_the_guard_and_the_kink_regions_have_the_blocks_floor(tmp_path):
    """An element of the jet's root bent by 40 degrees asks the guard for a refinement: a kink
    region floored at ``h_floor``, and the guard's stop once its chord lies within 1.5 floors.
    With the block's ``post.h_floor`` of delta/512 the run goes on, the kink region at that
    floor; with the block's ``post.h_floor`` null, the collapse's delta/128, the guard of the
    fresh mesh stops it at once (``feature-below-floor``)."""
    from test_reconnect_geometry import _bent

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)

    class Bent(HeadProblem):
        def interface_segments(self):
            segs = super().interface_segments()
            chain, apex = bubble_with_head(1e-3, r_n=0.3)
            P = _bent(chain.points, (apex + 200) // 2, turn_deg=40.0)
            segs[0] = self._chain("bubble", P[:, 0].copy(), P[:, 1].copy(), ("axis", "axis"))
            return segs

    block = json.loads(json.dumps(WIRED))
    block["post"].update(h_floor=1e-3 / 512, dt_min=1e-12)
    collapse = json.loads(json.dumps(block))
    collapse["post"]["h_floor"] = None
    for name, floor in (("collapse", None), ("block", 1e-3 / 512)):
        events = ev.EventSettings.from_dict(block if floor else collapse)
        drv = _events_driver(tmp_path / name, Bent(r_n=lambda t: 0.3), recon, events=events,
                             h_floor=1e-3 / 128, max_steps=1)
        regions = {g["name"]: g for g in drv.measure()["regions"]}
        if floor is None:  # the element lies within 1.5 floors already: no refinement, a stop
            assert "kink_bubble" not in regions
        else:
            assert regions["kink_bubble"]["floor"] == floor
            assert regions["kink_bubble"]["target"] >= floor
        out = drv.run(1e-9)
        drv.close()
        stop = out["stop"]
        if floor is None:
            assert (stop["reason"], stop["trigger"]) == ("interface-unresolved",
                                                         "feature-below-floor")
            assert out["steps"] == 0 and stop["fresh"]
        else:
            assert stop["reason"] == "max-steps" and out["steps"] == 1
