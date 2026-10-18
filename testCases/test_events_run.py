"""Contract tests of the events a run after the cut makes with the events block
(``collapse.reconnect_events``): the tip of an event, the monitor of a cascade of events at a tip
and the ghosts of the removed heads (``events.py``), and the collapse driver that makes the events
within its run, on stand-in problems whose chains are prescribed (``test_bookkeeping.
CutProblem``) and a stand-in of the liquid cut. No finite-element problem is built."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import events as ev  # noqa: E402

DELTA = 1e-3


# --- the tip of an event, and the cascade at a tip ------------------------------------------------


def test_the_tip_of_an_event_is_the_end_on_the_axis_its_head_holds():
    assert ev.event_tip({"chain": "open", "side": "before"}) == "cavity_tip"
    assert ev.event_tip({"chain": "bubble", "side": "after"}) == "bubble_top"
    assert ev.event_tip({"chain": "bubble", "side": "before"}) == "bubble_bottom"
    assert ev.FAR_TIPS == {"bubble_top": "bubble_bottom", "bubble_bottom": "bubble_top",
                           "cavity_tip": None}


def _monitor(**kw):
    s = ev.EventSettings.from_dict(kw)
    return ev.CascadeMonitor.for_run(s, DELTA, T=2e-6)


def _sequence(monitor, tip, intervals, volumes, t0=0.0):
    """Events at ``tip`` at the cumulative ``intervals`` (in delta^2) with heads of the
    ``volumes``: the verdict of each."""
    t, out = t0, []
    for k, (dt, V) in enumerate(zip(intervals, volumes), start=1):
        t += dt * DELTA**2
        out.append(monitor.add(tip, k, t, V))
    return out


def test_events_at_a_steady_rate_are_a_stationary_cascade():
    """Events 4 delta^2 apart with heads of one volume: the first at its tip, then undetermined
    until three pairs of ratios (the fifth event), then stationary; the interval in delta^2 and
    as events per T."""
    got = _sequence(_monitor(), "cavity_tip", [1.0, 4.0, 4.0, 4.0, 4.0, 4.0], [1.0] * 6)
    assert [g["verdict"] for g in got] == ["first", "undetermined", "undetermined",
                                           "undetermined", "stationary", "stationary"]
    assert [g["generation"] for g in got] == [1, 2, 3, 4, 5, 6]
    assert math.isnan(got[0]["interval"])
    assert got[1]["interval_over_delta2"] == pytest.approx(4.0)
    assert got[1]["events_per_T"] == pytest.approx(2e-6 / (4.0 * DELTA**2))
    assert got[2]["interval_ratio"] == pytest.approx(1.0) and got[2]["volume_ratio"] == 1.0


def test_events_that_shrink_in_interval_and_volume_are_a_cascade_that_stops():
    """Intervals and heads each 0.6 of the one before, below the ratio 0.7 of the test: the three
    pairs of ratios of the fifth event make it a shrinking cascade; an interval below
    cascade_min_interval delta^2 makes one at once, from the second event at a tip."""
    shrink = [4.0 * 0.6**j for j in range(5)]
    got = _sequence(_monitor(), "bubble_top", shrink, [0.6**j for j in range(5)])
    assert [g["verdict"] for g in got] == ["first", "undetermined", "undetermined",
                                           "undetermined", "shrink"]
    assert got[-1]["shrinking"] == 3 and got[-1]["interval_ratio"] == pytest.approx(0.6)
    got = _sequence(_monitor(), "bubble_top", [1.0, 0.2], [1.0, 1.0])
    assert got[1]["verdict"] == "shrink" and got[1]["interval_over_delta2"] == pytest.approx(0.2)
    # a shorter interval with heads that do not shrink is no cascade
    got = _sequence(_monitor(), "bubble_top", shrink, [1.0] * 5)
    assert got[-1]["verdict"] == "stationary" and got[-1]["shrinking"] == 0
    # nor with the test's keys set apart: cascade_min_interval 0 never stops on the interval
    got = _sequence(_monitor(cascade_min_interval=0.0), "bubble_top", [1.0, 0.01], [1.0, 1.0])
    assert got[1]["verdict"] == "undetermined"


def test_events_that_grow_in_interval_and_volume_are_a_growing_cascade():
    grow = [1.0 * 1.6**j for j in range(5)]
    got = _sequence(_monitor(), "cavity_tip", grow, [1.6**j for j in range(5)])
    assert got[-1]["verdict"] == "grow" and got[-1]["growing"] == 3
    got = _sequence(_monitor(cascade_count=1), "cavity_tip", grow[:3], [1.0, 1.6, 2.56])
    assert [g["verdict"] for g in got] == ["first", "undetermined", "grow"]


def test_the_tips_are_followed_apart_and_a_run_rebuilds_its_monitor_from_its_records():
    """The events of each tip are their own sequence; a monitor made from the records of the
    events a run made (a restart) gives the same verdicts, which a check of the next event
    continues."""
    m = _monitor()
    _sequence(m, "cavity_tip", [1.0, 4.0, 4.0], [1.0, 1.0, 1.0])
    first = m.check("bubble_top", 1e-5, 1.0)
    assert first["verdict"] == "first" and first["generation"] == 1
    records = [{"k": k, "kind": "liquid-neck", "outcome": "made", "tip": e["tip"], "t": e["t"],
                "trigger": {"head_volume": e["V"]}}
               for k, e in enumerate(({"tip": "cavity_tip", "t": x["t"], "V": x["V"]}
                                      for x in m.history["cavity_tip"]), start=1)]
    again = ev.CascadeMonitor.for_run(ev.EventSettings(), DELTA, T=2e-6, records=records)
    assert again.verdicts() == m.verdicts()
    nxt = m.history["cavity_tip"][-1]["t"] + 4.0 * DELTA**2
    assert again.check("cavity_tip", nxt, 1.0) == m.check("cavity_tip", nxt, 1.0)
    assert again.check("cavity_tip", nxt, 1.0)["generation"] == 4


# --- the ghosts of the removed heads --------------------------------------------------------------


def _ghost(**kw):
    """A head hanging from the bubble's top, cut at a neck at z = 1: it ended at 0.9 and the cap's
    tip, the stump's, is at 0.99; it moves down at 50, its rear retracting at 20 more."""
    return {"t0": 0.0, "U": -50.0, "direction": -1.0, "v_ret": 20.0, "z_front0": 0.9,
            "z_rear0": 0.99, "tip": "bubble_top", "far_tip": "bubble_bottom", "step0": 10,
            "recontacts": [], **kw}


def test_a_ghost_moves_rigidly_and_its_rear_retracts_towards_its_head():
    g = _ghost()
    assert ev.ghost_place(g, 0.0) == (0.9, 0.99)
    front, rear = ev.ghost_place(g, 1e-3)
    assert front == pytest.approx(0.9 - 0.05) and rear == pytest.approx(0.99 - 0.07)


def test_a_ghost_flags_the_stump_that_catches_it_and_the_far_tip_it_reaches_once_each():
    """Rear: the stump's tip past the ghost's rear; front: the ghost's front at the bubble's
    bottom; neither before two steps after the cut, each once; the head of the cavity's tip rises
    into the open gas and reaches no tip."""
    g = _ghost()
    tips = {"bubble_top": 0.99 - 0.2, "bubble_bottom": 0.5, "cavity_tip": 1.2}
    assert ev.ghost_recontacts(g, 1e-3, 11, tips) == []  # one step after the cut: not yet
    (rear,) = ev.ghost_recontacts(g, 1e-3, 12, tips)  # the stump's tip is below the rear
    assert rear["kind"] == "rear" and rear["tip"] == "bubble_top"
    assert rear["z_ghost"] == pytest.approx(0.92) and rear["z_tip"] == pytest.approx(0.79)
    g["recontacts"].append(rear)
    assert ev.ghost_recontacts(g, 1e-3, 13, tips) == []
    (front,) = ev.ghost_recontacts(g, 8e-3, 14, tips)  # 0.9 - 0.4 = 0.5: the bubble's bottom
    assert front["kind"] == "front" and front["tip"] == "bubble_bottom"
    up = _ghost(U=50.0, direction=1.0, z_front0=1.3, z_rear0=1.21, tip="cavity_tip",
                far_tip=None)
    assert ev.ghost_recontacts(up, 1.0, 100, {"cavity_tip": 0.21}) == []
    assert [x["kind"] for x in ev.ghost_recontacts(up, 1.0, 100, {"cavity_tip": 1e3})] == ["rear"]


# --- the events the driver makes within a run, on stand-ins ---------------------------------------

LIQUID = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 2e-6}}


class RegrowingHead:
    """A stand-in problem (``test_bookkeeping.CutProblem``) whose bubble is
    ``test_events_stops.bubble_with_head`` at delta = 1e-3, a jet hanging from its top whose neck
    is 0.3 delta wide until ``period`` after its last cut and 0.09 delta from then on, landed
    (eps_d 0.2 of a head 0.5 delta wide). The stand-in's cut (:func:`fake_surgery`) restores the
    neck, but keeps the head."""

    def __new__(cls, period=5e-6, **kw):
        from test_bookkeeping import CutProblem
        from test_events_stops import bubble_with_head

        class Problem(CutProblem):
            def __init__(self):
                super().__init__(**kw)
                self.period, self.last_cut, self._heads = period, 0.0, {}

            def r_n(self):
                return 0.3 if self.t - self.last_cut < self.period else 0.09

            def interface_segments(self):
                r_n = self.r_n()
                if r_n not in self._heads:
                    self._heads[r_n] = bubble_with_head(1e-3, r_n=r_n)[0]
                segs = super().interface_segments()
                P = self._heads[r_n].points
                segs[0] = self._chain("bubble", P[:, 0].copy(), P[:, 1].copy(), ("axis", "axis"))
                return segs

        return Problem()


def fake_surgery(*, refuse=(), refuse_late=(), unclosed=(), fail=(), keep_neck=False,
                 ghost=None):
    """A stand-in of ``surgery`` for the driver's liquid cuts on stand-in problems: the state
    before the cut kept as the real cut keeps it, the neck restored (but with ``keep_neck``), a
    rebuild counted, a record of the real one's form handed to ``on_made``, the state after it kept
    and the record written; or, for the event numbers in ``refuse`` (at admission) and
    ``refuse_late`` (once the state before it is kept), a refusal, in ``unclosed`` hard budgets that
    do not close, in ``fail`` a failure after the rebuild. ``ghost`` sets fields of the ghost."""
    import json as _json

    class CutRefused(RuntimeError):
        def __init__(self, message, record=None, precut=None):
            super().__init__(message)
            self.record, self.precut = record, precut

    class CutFailed(RuntimeError):
        def __init__(self, message, precut=None, record=None):
            super().__init__(message)
            self.precut, self.record = precut, record

    class BudgetUnclosed(CutFailed):
        pass

    calls = []

    def liquid_cut(problem, events, reconnect, outdir, request, *, k, dt_prev,
                   elements_per_scale, h_floor, driver=None, on_made=None, log=print):
        calls.append({"k": k, "dt_prev": dt_prev, "elements_per_scale": elements_per_scale,
                      "h_floor": h_floor, "kind": request["kind"]})
        t = problem.get_current_time()
        neck = ev.event_neck(request)
        record = {"schema": "faraday-event/2", "k": k, "kind": request["kind"],
                  "class": request["class"]}
        if k in refuse:
            raise CutRefused("refused for the test", {**record, "outcome": "refused",
                                                      "failed_at": "admission",
                                                      "error": "CutRefused: for the test"})
        precut = str(driver.keep_state(Path(outdir) / "states" / f"precut_e{k:03d}.dump",
                                       cut_stage=ev.PRECUT_EVENT,
                                       event={**{x: v for x, v in request.items()
                                                 if x != "reason"}, "k": k}))
        if k in refuse_late:
            raise CutRefused("refused late", {**record, "outcome": "refused",
                                              "failed_at": "caps"}, precut)
        problem.remesh_count += 1  # the rebuild of the cut
        if k in unclosed:
            raise BudgetUnclosed("unclosed for the test", precut,
                                 {**record, "outcome": "unclosed", "failed_at": "postcut",
                                  "closures": {"hard": ["F1"]}})
        if k in fail:
            raise CutFailed("failed for the test", precut, {**record, "outcome": "failed"})
        if not keep_neck:
            problem.last_cut = t
        d, V = float(neck["direction"]), float(neck["head_volume"])
        tip = ev.event_tip(neck)
        z_new = neck["z_n"] + d * neck["r_n"]
        record.update(t=t, tau=t - problem.cut["t_cut"], tau_p=t - problem.cut["pinch"]["t_p"],
                      chain=neck["chain"], tip=tip,
                      trigger={**neck, "eps_d": events.liquid.eps_d, "Re_n": 15.0,
                               "We_n": 27.0, "r_n_over_s": neck["r_n"] / neck["s"]},
                      cap={"eta": 1.0, "z_tip_old": neck["z_tip"], "z_tip_new": z_new},
                      removed={"V": V, "Pz": 50.0 * d * V, "KE": 1250.0 * V},
                      ghost={"V": V, "Pz": 50.0 * d * V, "KE": 1250.0 * V, "U": 50.0 * d,
                             "a": (0.75 * V / math.pi) ** (1 / 3), "v_ret": 20.0,
                             "direction": d, "t0": t, "z_front0": neck["z_tip"],
                             "z_rear0": z_new, "du_stump": 1.0, "status": "free",
                             **(ghost or {})},
                      closures={"relative": {f"F{j}": 1e-3 * j for j in range(1, 7)},
                                "F3_phi": {"phi_relative_error": 1e-4}, "flags": [], "hard": []},
                      transfer={"predicted_unlocated": 3, "reported_unlocated": 3},
                      dt={"history": dt_prev}, dumps={"precut": precut}, outcome="made",
                      wall_seconds=0.5)
        before = [e for e in problem.events if e.get("kind") in ev.EVENT_CLASSES["liquid"]]
        record["removed_total"] = {"events": len(before) + 1, **{
            x: sum(e["removed"][x] for e in before) + record["removed"][x]
            for x in ("V", "Pz", "KE")}}
        problem.events = list(problem.events) + [record]
        on_made(record)
        record["dumps"]["postcut"] = str(driver.keep_state(
            Path(outdir) / "states" / f"postcut_e{k:03d}.dump", cut_stage="postcut-event",
            event_k=k))
        path = Path(outdir) / "events" / f"event_{k:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        full = dict(record, ghost=dict(record["ghost"], polygon=[[0.0, 1.0], [1e-4, 1.0]]))
        path.write_text(_json.dumps(full))
        return record

    return SimpleNamespace(liquid_cut=liquid_cut, CutRefused=CutRefused, CutFailed=CutFailed,
                           BudgetUnclosed=BudgetUnclosed, calls=calls,
                           restore_cut=lambda problem, record: setattr(problem, "cut", record))


def _rows(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:]]


def _run(tmp_path, monkeypatch, problem, *, block=None, surgery=None, **kw):
    """A run after a cut at delta = 1e-3 of the stand-in ``problem`` with the events block (the
    liquid one by default) and a stand-in ``surgery``; returns the driver, its result and the
    stand-in."""
    from test_events_plumbing import _events_driver

    from faraday_singularity.reconnect import ReconnectSettings

    surgery = surgery or fake_surgery()
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", surgery)
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    events = ev.EventSettings.from_dict(LIQUID if block is None else block)
    drv = _events_driver(tmp_path, problem, recon, events=events,
                         leg_info={"commit": "c" * 40}, **{"max_steps": 12, **kw})
    out = drv.run(1e-9)
    drv.close()
    return drv, out, surgery


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_the_driver_makes_each_event_and_goes_on_from_the_fresh_mesh(tmp_path, monkeypatch):
    """The neck of the hanging jet lands 5e-6 after its last cut: the run makes each event
    (the stand-in's liquid cut, with the run's last step as the history's and the floors after
    the cut), re-measures the fresh mesh, checks its stops as those of a mesh freshly built and
    goes on to its step limit. Each record gets the driver's bookkeeping (the jet's base before
    the cut, the event's generation at its tip, step, leg, times in delta units, the cascade's
    verdict, the commit, the ghost's start, the jet after it); events.txt holds a row per event,
    events.jsonl a liquid-cut line, profiles/ a profile; the state after each cut is dumped with
    the bookkeeping reset, its mesh fresh, no remesh check pending."""
    drv, out, surgery = _run(tmp_path, monkeypatch, RegrowingHead())
    assert out["stop"]["reason"] == "max-steps" and out["steps"] == 12
    made = drv.pb.events
    assert len(made) >= 2 and [e["k"] for e in made] == list(range(1, len(made) + 1))
    assert [c["k"] for c in surgery.calls] == [e["k"] for e in made]
    for c in surgery.calls:
        assert c["elements_per_scale"] == 8.0 and c["h_floor"] == drv._h_floor()
        assert c["dt_prev"] > 0.0
    times = [e["t"] for e in made]
    assert all(b - a == pytest.approx(5e-6, rel=0.5) for a, b in itertools.pairwise(times))
    for j, e in enumerate(made, start=1):
        assert (e["tip"], e["chain"], e["generation"]) == ("bubble_top", "bubble", j)
        assert e["leg"] == 0 and e["commit"] == "c" * 40
        assert e["tau_over_delta2"] == pytest.approx(e["tau"] / 1e-6)
        assert e["t_prime_over_T"] == pytest.approx(e["tau_p"] / (-1e-6 - 0.0))
        assert e["cascade"]["generation"] == j
        assert e["cascade"]["verdict"] == ("first" if j == 1 else "undetermined")
        trig = e["trigger"]
        assert trig["r_j"] > 0.0 and trig["r_j_over_r_n"] == pytest.approx(trig["r_j"]
                                                                           / trig["r_n"])
        assert trig["neck_to_base_over_r_j"] > 0.0
        assert e["dt"]["before"] == e["dt"]["history"] > 0.0
        g = e["ghost"]
        assert (g["tip"], g["far_tip"], g["step0"]) == ("bubble_top", "bubble_bottom", e["step"])
        assert e["after"]["jet_born"] is True and e["after"]["r_j"] > 0.0
        meta = json.loads(Path(e["dumps"]["postcut"]).with_suffix(".json").read_text())
        assert meta["mesh_fresh"] is True and "resume_at" not in meta
        assert math.isnan(meta["ell_prev"]) and [x["k"] for x in meta["events"]] == list(
            range(1, j + 1))
    rows = _rows(tmp_path / "events.txt")
    assert [int(r["k"]) for r in rows] == [e["k"] for e in made]
    assert rows[0]["kind"] == "liquid-neck" and rows[0]["tip"] == "bubble_top"
    assert float(rows[1]["interval_over_delta2"]) == pytest.approx(
        (times[1] - times[0]) / 1e-6)
    assert float(rows[0]["F3"]) == pytest.approx(3e-3) and rows[0]["cascade"] == "first"
    lines = _jsonl(tmp_path / "events.jsonl")
    cuts = [x for x in lines if x["event"] == "liquid-cut"]
    assert [x["k"] for x in cuts] == [e["k"] for e in made]
    assert all(x["record"] == f"events/event_{x['k']:03d}.json" for x in cuts)
    shots = [json.loads(p.read_text().splitlines()[0][2:])
             for p in (tmp_path / "profiles").glob("profile_*[0-9].txt")]
    assert sum(s["reason"] == "liquid-cut" for s in shots) == len(made)
    summary = out["events"]
    assert summary["made"] == len(made) and summary["by_tip"] == {"bubble_top": len(made)}
    assert summary["removed_total"]["V"] == pytest.approx(sum(e["removed"]["V"] for e in made))
    assert summary["last"]["k"] == len(made) and summary["stop"] is None
    assert summary["cascade"] == {"bubble_top": made[-1]["cascade"]["verdict"]}
    assert out["window_end"] is None


def test_the_fresh_mesh_of_an_event_is_checked_as_a_freshly_built_one(tmp_path, monkeypatch):
    """After each event the stops of the fresh mesh are those of a mesh freshly built (the guard's
    limits there and the check of its curved elements), before the run steps from it."""
    import faraday_singularity.driver as module

    calls = []
    real = module.post_cut_stops

    def spy(*args, **kw):
        calls.append((kw["guard"]["fresh"], kw["invalid"] is not None))
        return real(*args, **kw)

    monkeypatch.setattr(module, "post_cut_stops", spy)
    drv, out, _ = _run(tmp_path, monkeypatch, RegrowingHead(), max_steps=6)
    assert drv.pb.events and out["stop"]["reason"] == "max-steps"
    assert calls.count((True, True)) >= 1 + len(drv.pb.events)


def test_the_run_stops_resumably_on_the_event_after_the_most_it_may_make(tmp_path, monkeypatch):
    """With max_events 1 the second landing stops the run (``max-events``, event 2) with the dump
    of the state before it, from which a run with a larger max_events goes on."""
    block = {**LIQUID, "max_events": 1}
    _, out, surgery = _run(tmp_path, monkeypatch, RegrowingHead(), block=block)
    stop = out["stop"]
    assert stop["reason"] == ev.EVENT_LIMIT and stop["k"] == 2 and stop["max_events"] == 1
    assert [c["k"] for c in surgery.calls] == [1]
    precut = Path(stop["precut"])
    assert precut.name == "precut_e002.dump" and Path(stop["end_state"]).is_file()
    meta = json.loads(precut.with_suffix(".json").read_text())
    assert meta["cut_stage"] == ev.PRECUT_EVENT and meta["event"]["k"] == 2
    assert [e["k"] for e in meta["events"]] == [1]
    assert out["events"]["stop"]["reason"] == ev.EVENT_LIMIT
    assert _jsonl(tmp_path / "events.jsonl")[-2]["event"] == ev.EVENT_LIMIT


def test_events_at_one_tip_that_come_ever_faster_stop_the_run_as_a_cascade(tmp_path,
                                                                           monkeypatch):
    """A cut that leaves the neck landed: the fresh mesh lands it again at once, an interval of
    zero at the tip, below cascade_min_interval delta^2: a shrinking cascade, the run stops
    resumably (``event-cascade``, with the verdict) before making it."""
    surgery = fake_surgery(keep_neck=True)
    drv, out, _ = _run(tmp_path, monkeypatch, RegrowingHead(), surgery=surgery)
    stop = out["stop"]
    assert stop["reason"] == ev.EVENT_CASCADE and stop["k"] == 2
    assert stop["cascade"]["verdict"] == "shrink" and stop["cascade"]["interval"] == 0.0
    assert [c["k"] for c in surgery.calls] == [1] and len(drv.pb.events) == 1
    assert Path(stop["precut"]).name == "precut_e002.dump"
    assert out["events"]["stop"]["reason"] == ev.EVENT_CASCADE


def test_a_refused_cut_stops_the_run_resumably_and_changes_nothing(tmp_path, monkeypatch):
    """A cut refused at admission: the run stops (``liquid-cut-refused``, with the stage and the
    error of the cut's record), keeps the state before the event and ends on it; one refused once
    the state before it is kept names that dump instead of keeping another."""
    drv, out, _ = _run(tmp_path / "a", monkeypatch, RegrowingHead(),
                       surgery=fake_surgery(refuse=(1,)))
    stop = out["stop"]
    assert stop["reason"] == ev.LIQUID_CUT_REFUSED and stop["k"] == 1
    assert stop["failed_at"] == "admission" and "for the test" in stop["error"]
    assert stop["record"] == "events/event_001.json" and drv.pb.events == []
    assert Path(stop["precut"]).name == "precut_e001.dump" and Path(stop["end_state"]).is_file()
    drv, out, _ = _run(tmp_path / "b", monkeypatch, RegrowingHead(),
                       surgery=fake_surgery(refuse_late=(1,)))
    stop = out["stop"]
    assert stop["reason"] == ev.LIQUID_CUT_REFUSED and stop["failed_at"] == "caps"
    kept = sorted(p.name for p in (tmp_path / "b" / "states").glob("precut_e*.dump"))
    assert kept == ["precut_e001.dump"] and Path(stop["precut"]).name == kept[0]


def test_a_cut_whose_budgets_do_not_close_ends_the_run_on_the_state_before_it(tmp_path,
                                                                             monkeypatch):
    """Hard budgets that do not close (``event-unclosed``): the problem holds a cut that is not
    accepted, so the run ends on the state before the event, which the cut kept (``end_state``),
    and dumps nothing more; any other failure of the cut fails the run."""
    _, out, _ = _run(tmp_path / "a", monkeypatch, RegrowingHead(),
                     surgery=fake_surgery(unclosed=(1,)))
    stop = out["stop"]
    assert stop["reason"] == ev.EVENT_UNCLOSED and stop["hard"] == ["F1"]
    assert stop["end_state"] == stop["precut"] and Path(stop["precut"]).is_file()
    dumps = sorted(p.name for p in (tmp_path / "a" / "states").glob("*.dump"))
    assert dumps == ["precut_e001.dump"] + [p for p in dumps if p.startswith("state_")]
    steps = [json.loads(p.read_text())["step"]
             for p in (tmp_path / "a" / "states").glob("state_*.json")]
    assert steps and max(steps) < out["steps"]  # no dump of the state the cut left
    shots = [json.loads(p.read_text().splitlines()[0][2:])["reason"]
             for p in (tmp_path / "a" / "profiles").glob("profile_*[0-9].txt")]
    assert "stop" not in shots
    surgery = fake_surgery(fail=(1,))
    with pytest.raises(RuntimeError, match="failed for the test"):
        _run(tmp_path / "b", monkeypatch, RegrowingHead(), surgery=surgery)


def test_a_restart_from_the_state_after_an_event_goes_on_with_its_cascade(tmp_path,
                                                                        monkeypatch):
    """From the dump of the state after the first event, a restart meets the next landing as the
    event after it (k = 2, the second generation at its tip, its interval from the first)."""
    from test_bookkeeping import _settings

    from faraday_singularity.driver import CollapseDriver
    from faraday_singularity.reconnect import ReconnectSettings

    drv, _, _ = _run(tmp_path / "a", monkeypatch, RegrowingHead(), max_steps=8)
    first = drv.pb.events[0]
    meta = json.loads(Path(first["dumps"]["postcut"]).with_suffix(".json").read_text())
    pb = RegrowingHead()
    pb.t, pb.last_cut = meta["time"], first["t"]
    surgery = fake_surgery()
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", surgery)
    again = CollapseDriver(pb, _settings(t_end=1.0, dt_max=1.0, max_steps=meta["step"] + 8),
                           tmp_path / "b", reconnect=ReconnectSettings(delta=1e-3, dt0=1e-6),
                           events=ev.EventSettings.from_dict(LIQUID))
    again.restore(meta)
    again.run(1e-9)
    again.close()
    assert [e["k"] for e in pb.events][:2] == [1, 2] and pb.events[1]["generation"] == 2
    assert pb.events[1]["cascade"]["interval"] == pytest.approx(pb.events[1]["t"] - first["t"])


def test_the_ghost_of_each_removed_head_is_followed_and_its_recontacts_flagged(tmp_path,
                                                                             monkeypatch):
    """The ghost of each removed head is followed after its cut: a ghost at rest whose rear lies
    beyond the stump's tip (here the stand-in keeps the head, so the stump's tip is the old head's)
    is caught by the stump two steps after the cut, one moving down at 2e4 reaches the bubble's
    bottom a few steps later. Each recontact is a flag (the run goes on to its step limit): in the
    ghost of the record the dumps keep and of its file, the polygon kept, an event
    ``ghost-recontact`` and a row of ``drops.txt``, after the row of the ghost made."""
    resting = fake_surgery(ghost={"U": 0.0, "v_ret": 0.0})
    drv, out, _ = _run(tmp_path / "rest", monkeypatch, RegrowingHead(period=1e3),
                       surgery=resting, max_steps=6)
    assert drv.pb.events == [] and out["events"]["ghost_recontacts"] == {"rear": 0, "front": 0}
    drv, out, _ = _run(tmp_path / "rear", monkeypatch, RegrowingHead(), surgery=resting)
    assert out["stop"]["reason"] == "max-steps" and drv.pb.events
    first = drv.pb.events[0]
    g = first["ghost"]
    assert g["status"] == "recontact" and [x["kind"] for x in g["recontacts"]] == ["rear"]
    hit = g["recontacts"][0]
    assert hit["step"] == first["step"] + ev.GHOST_SKIP_STEPS and hit["tip"] == "bubble_top"
    assert hit["tau"] == pytest.approx(hit["t"]) and hit["z_ghost"] == g["z_rear0"]
    lines = [x for x in _jsonl(tmp_path / "rear" / "events.jsonl")
             if x["event"] == "ghost-recontact"]
    assert [(x["k"], x["recontact"]) for x in lines][:1] == [(1, "rear")]
    rows = _rows(tmp_path / "rear" / "drops.txt")
    assert [(r["k"], r["status"]) for r in rows][:2] == [("1", "free"), ("1", "rear")]
    assert float(rows[0]["z_rear"]) == pytest.approx(g["z_rear0"])
    file = json.loads((tmp_path / "rear" / "events" / "event_001.json").read_text())
    assert file["ghost"]["recontacts"] == g["recontacts"] and file["ghost"]["polygon"]
    end = json.loads(Path(out["stop"]["end_state"]).with_suffix(".json").read_text())
    assert end["events"][0]["ghost"]["recontacts"] == g["recontacts"]
    followed = [e for e in drv.pb.events if e["step"] + ev.GHOST_SKIP_STEPS <= out["steps"]]
    assert out["events"]["ghost_recontacts"]["rear"] == len(followed) >= 1
    falling = fake_surgery(ghost={"U": -2e4, "v_ret": 0.0})
    drv, out, _ = _run(tmp_path / "front", monkeypatch, RegrowingHead(period=4e-6),
                       surgery=falling, max_steps=20)
    g = drv.pb.events[0]["ghost"]
    assert [x["kind"] for x in g["recontacts"]] == ["front"]
    hit = g["recontacts"][0]
    assert hit["tip"] == "bubble_bottom" and hit["z_ghost"] <= hit["z_tip"]
    assert hit["step"] > drv.pb.events[0]["step"] + ev.GHOST_SKIP_STEPS
    assert out["stop"]["reason"] == "max-steps"
