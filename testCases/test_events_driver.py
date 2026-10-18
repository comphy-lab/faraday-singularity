"""Contract tests of the collapse driver after a cut with the events block, on stand-in problems
whose two chains are prescribed (``test_bookkeeping.CutProblem``): the stops of the events branch,
an event without a handler as a resumable stop with the dump of the state before it, and a run
that goes on from that dump. No finite-element problem is built."""

from __future__ import annotations

import hashlib
import json
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

LIQUID = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False}}


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
                                           "post": {"open_neck_stop": False}}
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
