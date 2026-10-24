"""The collapse driver's meeting of a closing gas groove on stand-in problems
(``test_bookkeeping.CutProblem``; ``groovesurgery.groove_fill`` stood in for), without a
finite-element problem.

* without ``groove.enabled`` a groove (``A-groove``) and a fold of the ring (``RF``) stay events
  without a handler, as at 89f7f49, and no fill is asked for;
* with it the request reaches the fill with the run's last step, the elements per scale, the
  stops' fold limits and the core of jetcore (None without the part); the fill made: the
  bookkeeping of :meth:`CollapseDriver.reset_after_groove` (the chain's kink regions and the apex
  history of the jet the groove straddled dropped, no remesh check pending, an event
  ``groove-fill``), a row of ``events.txt`` (the chain filled, the fill a negative removal, G1-G6 in
  ``F1``-``F6``), the totals of ``budget.txt`` net of the fill, the steps after it;
* a refusal is the resumable stop ``groove-fill-refused`` and hard budgets that do not close
  ``event-unclosed``, each with the state before the event;
* with ``pinch.far_head`` too, a groove off every head is filled, not cut; a groove on a tracked
  far head (the bubble's head of ``test_jetcore.HeadProblem``) is met by the far-head cut of that
  head first (the stand-in cut of ``test_events_run.fake_surgery``), the fill never asked for, an
  event ``far-head`` with the groove's kind and class; without ``pinch.far_head`` the groove on the
  head is filled as before, and a fill failing off every head stops as before;
* a gas neck (``G``) at the pocket of a curled head of the bubble (``test_curled_head``, its rim
  curling up past its root neck) is met by the far-head cut of that root neck.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from test_bookkeeping import CutProblem
from test_events_plumbing import _events, _events_driver
from test_runner import ROOT
from test_ring_driver import RingProblem, made_ring_cut

sys.path.insert(0, str(ROOT / "src"))
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

RECON = ReconnectSettings(delta=1e-3, dt0=1e-6)
GROOVE_ON = ev.EventSettings.from_dict({"groove": {"enabled": True}})


def groove_request(chain="bubble", cls="A-groove", jet=None):
    out = {"reason": ev.EVENT_REQUEST, "kind": "groove" if cls == "A-groove" else "ring-fold",
           "class": cls, "handler": ev.GROOVE_HANDLER if cls == "A-groove" else None,
           "chain": chain, "contact": "fold", "fold_gap": 3e-5, "fold_r": 7e-3, "fold_z": 4.4,
           "fold_arc": 2e-4, "fold_h_elem": 1e-5, "fold_limit": 5e-5, "fold_pair": [100, 140],
           "kappa_g": 0.05}
    if jet is not None:
        out["jet"] = jet
    return out


def fake_fill(calls, fail=None, hard=("G1",)):
    """A stand-in of ``groovesurgery.groove_fill``: the precut dump kept, then a refusal of
    ``fail`` (its record written as the real fill writes it; ``BudgetUnclosed`` after a stand-in
    rebuild, the focus changed and a rebuild counted, with the closures ``hard`` beyond their
    limits) or a fill made (the record appended, ``on_made`` called, the postcut dump kept)."""
    from faraday_singularity import surgery

    def groove_fill(pb, events, reconnect, outdir, request, *, k, dt_prev, elements_per_scale,
                    h_floor, fold, core=None, driver=None, on_made=None, log=print):
        calls.append({"kind": request["kind"], "class": request["class"], "k": k,
                      "dt_prev": dt_prev, "eps": elements_per_scale, "fold": dict(fold),
                      "core": core})
        precut = str(driver.keep_state(Path(outdir) / "states" / f"precut_e{k:03d}.dump",
                                       cut_stage=ev.PRECUT_EVENT, event=dict(request, k=k)))
        if fail is not None:
            unclosed = fail == "BudgetUnclosed"
            if unclosed:  # the fill's rebuild, not accepted
                pb.remesh_count += 1
                pb.active_mesh_template().focus = "the fill's focus"
            record = {"k": k, "kind": "groove-fill", "outcome": "unclosed" if unclosed else
                      "refused", "failed_at": "postcut" if unclosed else "geometry",
                      "error": "refused for the test",
                      "closures": {"hard": list(hard), **{x: 5.8e-9 for x in hard},
                                   "limits": {x: 2.28e-9 for x in hard}}}
            path = Path(outdir) / "events" / f"event_{k:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
            err = getattr(surgery, fail)("refused for the test")
            err.precut, err.record = precut, record
            raise err
        t = pb.t
        rec = {"schema": ev.EVENT_SCHEMA, "k": k, "kind": "groove-fill",
               "class": request["class"], "handler": ev.GROOVE_HANDLER,
               "chain": request["chain"], "tip": None, "t": t, "tau": t, "tau_p": t + 1e-6,
               "request": {x: v for x, v in request.items() if x != "reason"},
               "outcome": "made", "filled": {"V": 2e-12, "Pz": -3e-9, "KE": 4e-6},
               "geometry": {"mouth": 7.5e-5},
               "closures": {"relative": {f"G{i}": 1e-3 * i for i in range(1, 7)},
                            "G3_phi": {"phi_relative_error": 2e-4}},
               "transfer": {"predicted_unlocated": 7, "reported_unlocated": 7},
               "dt": {"history": dt_prev}, "wall_seconds": 1.0}
        pb.events = list(pb.events) + [rec]
        on_made(rec)
        driver.keep_state(Path(outdir) / "states" / f"postcut_e{k:03d}.dump",
                          cut_stage="postcut-event", event_k=k)
        return rec

    return groove_fill


def _table(path):
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return head, [dict(zip(head, line.split("\t"))) for line in lines[1:]]


def _run(run, pb, events, request, monkeypatch, calls, fail=None, **kw):
    from faraday_singularity import groovesurgery

    monkeypatch.setattr(groovesurgery, "groove_fill", fake_fill(calls, fail))
    drv = _events_driver(run, pb, RECON, events=events, state_dt=0.0, state_keep=10, **kw)
    orig = drv._post_cut_stop

    def stop():  # the groove, until the fill is asked for
        return dict(request) if not calls else orig()

    drv._post_cut_stop = stop
    return drv


def test_without_the_part_a_groove_and_a_fold_of_the_ring_stay_without_a_handler(tmp_path,
                                                                               monkeypatch):
    for cls, chain in (("A-groove", "bubble"), ("RF", "ring")):
        calls = []
        drv = _run(tmp_path / cls, CutProblem(), ev.EventSettings(), groove_request(chain, cls),
                   monkeypatch, calls)
        out = drv.run(1e-9)
        drv.close()
        assert out["stop"]["reason"] == ev.EVENT_UNHANDLED and calls == []
        assert out["stop"]["handler"] == (ev.GROOVE_HANDLER if cls == "A-groove" else None)


def test_a_groove_is_filled_and_the_run_goes_on(tmp_path, monkeypatch):
    calls = []
    pb = CutProblem()
    run = tmp_path / "run"
    drv = _run(run, pb, GROOVE_ON, groove_request("bubble", jet="bubble_top"), monkeypatch,
               calls)
    kink = {"name": "kink_bubble", "chain": "bubble", "r": 0.005, "z": 4.4, "radius": 1e-3,
            "target": 1e-4, "h_before": 1e-3, "t_set": 0.0, "released": False}
    drv.state.kinks = {"bubble": dict(kink), "bubble_far": dict(kink, name="kink_bubble_far")}
    drv.state.apex_history = {"bubble_top": [0.0, 1e-4], "cavity_tip": [0.0, 2e-4]}
    drv._checks_pending = True
    out = drv.run(1e-9)
    drv.close()
    assert len(calls) == 1
    c = calls[0]
    assert (c["kind"], c["class"], c["k"]) == ("groove", "A-groove", 1) and c["dt_prev"] > 0.0
    assert c["fold"] == {"ratio": drv.s.fold_ratio, "stop": 0.05 * 1e-3,
                         "stop_elements": drv.s.fold_stop_elements}
    assert c["core"] is None and c["eps"] == drv.s.elements_per_scale
    assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3
    assert out["events"]["made"] == 1 and out["events"]["by_kind"] == {"groove-fill": 1}
    st = drv.state
    assert "bubble" not in st.kinks and "bubble_far" not in st.kinks
    assert "bubble_top" not in st.apex_history
    events = _events(run)
    made = [e for e in events if e["event"] == "groove-fill"]
    assert len(made) == 1 and made[0]["k"] == 1 and made[0]["chain"] == "bubble"
    head, rows = _table(run / "events.txt")
    row = rows[0]
    assert (row["kind"], row["class"], row["chain"]) == ("groove-fill", "A-groove", "bubble")
    assert float(row["V_rem"]) == -2e-12 and float(row["Pz_rem"]) == 3e-9
    assert float(row["F1"]) == pytest.approx(1e-3) and float(row["F6"]) == pytest.approx(6e-3)
    assert float(row["phi_relative_error"]) == pytest.approx(2e-4)
    _, budget = _table(run / "budget.txt")
    last = budget[-1]
    assert float(last["V_removed"]) == pytest.approx(-2e-12) and int(last["events"]) == 1
    assert float(last["volume_total"]) == pytest.approx(1.0 - 2e-12)
    dump = run / "states" / "postcut_e001.json"
    meta = json.loads(dump.read_text())
    assert "resume_at" not in meta and meta["events"][0]["kind"] == "groove-fill"
    assert ev.events_done(meta["events"])["gap"] == 1


def test_a_refused_or_unclosed_fill_is_a_resumable_stop(tmp_path, monkeypatch):
    for fail, reason in (("CutRefused", ev.GROOVE_FILL_REFUSED),
                         ("BudgetUnclosed", ev.EVENT_UNCLOSED)):
        calls = []
        drv = _run(tmp_path / fail, CutProblem(), GROOVE_ON, groove_request("open"), monkeypatch,
                   calls, fail=fail)
        out = drv.run(1e-9)
        drv.close()
        assert len(calls) == 1 and out["stop"]["reason"] == reason and out["steps"] == 0
        assert Path(out["stop"]["precut"]).name == "precut_e001.dump"
        assert out["events"]["made"] == 0
        assert out["events"]["stop"]["reason"] == reason
        if reason == ev.EVENT_UNCLOSED:
            assert out["stop"]["hard"] == ["G1"]


def test_a_fold_of_the_ring_is_filled_in_the_ring_stage(tmp_path, monkeypatch):
    calls = []
    pb = RingProblem()
    block = ev.EventSettings.from_dict({"ring": {"enabled": True}, "groove": {"enabled": True}})
    run = tmp_path / "run"
    drv = _run(run, pb, block, groove_request("ring", "RF"), monkeypatch, calls)
    made_ring_cut(drv, pb, k=1)
    drv.state.kinks = {"ring": {"name": "kink_ring", "chain": "ring", "r": 0.01, "z": 4.4,
                                "radius": 1e-3, "target": 1e-4, "h_before": 1e-3, "t_set": 0.0,
                                "released": False}}
    out = drv.run(1e-9)
    drv.close()
    assert [c["k"] for c in calls] == [2] and calls[0]["class"] == "RF"
    assert out["stop"]["reason"] == "max-steps"
    assert "ring" not in drv.state.kinks
    _, rows = _table(run / "events.txt")
    assert [r["kind"] for r in rows] == ["groove-fill"] and rows[0]["chain"] == "ring"
    _, rings = _table(run / "ring.txt")
    # the ring's own fill and the groove's, the cumulative fills of the stage
    assert float(rings[-1]["V_filled"]) == pytest.approx(1e-9 + 2e-12)


def test_with_the_far_head_cut_too_a_groove_is_filled(tmp_path, monkeypatch):
    """With ``pinch.far_head`` (and jetcore) a groove has a handler the driver makes, so the
    far-head route of an event without one is not taken (the far head is only looked up for the
    fall-back of a fill that fails); the fill gets jetcore's core mask."""
    calls, routed = [], []
    block = ev.EventSettings.from_dict({
        "liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 1e-6},
        "jetcore": {"enabled": True}, "pinch": {"enabled": True, "far_head": True},
        "groove": {"enabled": True}})
    drv = _run(tmp_path / "run", CutProblem(), block, groove_request("open"), monkeypatch, calls)
    orig = drv._far_head
    drv._far_head = lambda request: routed.append(request) or orig(request)
    drv.run(1e-9)
    drv.close()
    # the far head is looked up once, on the state before the fill, for its fall-back only
    assert len(calls) == 1 and [x["class"] for x in routed] == ["A-groove"]
    assert drv.pb.events[-1]["kind"] == "groove-fill"
    core = calls[0]["core"]
    assert core is not None and len(core) == len(CutProblem().interface_segments()[1]["r"])
    assert math.isfinite(float(np.sum(core)))


def test_the_event_table_reads_a_fill_as_a_negative_removal_with_its_closures():
    sys.path.insert(0, str(ROOT / "postProcess"))
    import event_table

    rec = {"kind": "groove-fill", "filled": {"V": 2e-12, "Pz": -3e-9, "KE": 4e-6},
           "closures": {"relative": {f"G{i}": 1e-3 * i for i in range(1, 7)},
                        "G3_phi": {"phi_relative_error": 2e-4}}}
    assert event_table._removal(rec) == {"V": -2e-12, "Pz": 3e-9, "KE": -4e-6}
    fields, phi = event_table._closure_fields(rec["closures"], "groove-fill")
    assert fields == {f"F{i}": pytest.approx(1e-3 * i) for i in range(1, 7)}
    assert phi == pytest.approx(2e-4)


# --- a groove, or a gas neck, on a far head: the far-head cut first -------------------------------

FAR_BLOCK = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 2e-6},
             "jetcore": {"enabled": True}, "pinch": {"far_head": True},
             "groove": {"enabled": True}}


def _far_run(tmp_path, monkeypatch, *, request_at, pb=None, fail=None, block=FAR_BLOCK):
    """A run of a stand-in with a tracked head on its bubble (``test_jetcore.HeadProblem`` by
    default) whose stops request what ``request_at(drv, neck)`` returns until the driver meets it;
    the fill stood in for (``fake_fill``, failing with ``fail``), the liquid cut by
    ``test_events_run.fake_surgery``'s, which makes it, and the cut's requests recorded."""
    from test_events_run import fake_surgery
    from test_jetcore import HeadProblem

    from faraday_singularity import groovesurgery, surgery

    fills, requests = [], []
    cut = fake_surgery()

    def liquid_cut(pb, events, reconnect, outdir, request, **kw):
        requests.append(json.loads(json.dumps(request)))
        return cut.liquid_cut(pb, events, reconnect, outdir, request, **kw)

    monkeypatch.setattr(groovesurgery, "groove_fill", fake_fill(fills, fail, hard=("G3_box",)))
    monkeypatch.setattr(surgery, "liquid_cut", liquid_cut)
    pb = HeadProblem.make(r_n=0.3) if pb is None else pb
    events = ev.EventSettings.from_dict(block)
    drv = _events_driver(tmp_path, pb, RECON, events=events, state_dt=0.0, state_keep=10)
    named = drv._named_post_cut_chains()
    neck = next(x for x in ev.liquid_necks(named, events.liquid)
                if x["chain"] == "bubble" and x["status"] == "head")
    request = request_at(drv, neck)
    orig = drv._post_cut_stop
    drv._post_cut_stop = lambda: dict(request) if not (fills or requests) else orig()
    out = drv.run(1e-9)
    drv.close()
    return {"drv": drv, "out": out, "fills": fills, "cuts": cut.calls, "requests": requests,
            "neck": neck, "run": tmp_path}


def _groove_on_head(drv, neck):
    i = neck["head"][1] - 2
    return dict(groove_request("bubble"), fold_pair=[i - 2, i])


def _groove_off_head(drv, neck):
    return dict(groove_request("bubble"), fold_pair=[2, 4])  # the bubble's bottom, on no head


def test_a_groove_on_a_far_head_is_cut_first_and_never_filled(tmp_path, monkeypatch):
    """Decision 22: the head is a nuisance, its cut's volume and momentum hard checks; a groove on
    it is met by the far-head cut of its neck as the event, with no fill asked for."""
    res = _far_run(tmp_path, monkeypatch, request_at=_groove_on_head)
    out = res["out"]
    assert res["fills"] == [] and [c["kind"] for c in res["cuts"]] == ["far-head"]
    req, = res["requests"]
    assert (req["kind"], req["class"]) == ("far-head", "F-head")
    assert req["index"] == res["neck"]["index"] and req["far_event"]["class"] == "A-groove"
    assert "fallback" not in req
    assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3
    assert out["events"]["made"] == 1 and out["events"]["by_kind"] == {"far-head": 1}
    far = [e for e in _events(res["run"]) if e["event"] == "far-head"]
    assert len(far) == 1 and far[0]["far_class"] == "A-groove" and far[0]["far_kind"] == "groove"
    assert not (res["run"] / "events" / "event_001_fill.json").exists()


def test_a_groove_off_every_head_is_filled_and_its_failure_stops(tmp_path, monkeypatch):
    """With ``pinch.far_head`` a groove on no tracked head is filled, and a fill that fails there
    is its resumable stop, no cut; without ``pinch.far_head`` a groove on the head is filled."""
    res = _far_run(tmp_path / "made", monkeypatch, request_at=_groove_off_head)
    assert len(res["fills"]) == 1 and res["cuts"] == []
    assert res["out"]["events"]["by_kind"] == {"groove-fill": 1}
    for fail, reason in (("BudgetUnclosed", ev.EVENT_UNCLOSED),
                         ("CutRefused", ev.GROOVE_FILL_REFUSED)):
        res = _far_run(tmp_path / fail, monkeypatch, request_at=_groove_off_head, fail=fail)
        out = res["out"]
        assert len(res["fills"]) == 1 and res["cuts"] == [] and out["steps"] == 0
        assert out["stop"]["reason"] == reason and out["events"]["made"] == 0
        assert Path(out["stop"]["precut"]).name == "precut_e001.dump"
    without = {k: v for k, v in FAR_BLOCK.items() if k != "pinch"}
    res = _far_run(tmp_path / "plain", monkeypatch, request_at=_groove_on_head, block=without)
    assert len(res["fills"]) == 1 and res["cuts"] == []


def test_a_gas_neck_at_the_pocket_of_a_curled_head_is_the_cut_of_its_root_neck(tmp_path,
                                                                                monkeypatch):
    from test_bookkeeping import CutProblem
    from test_curled_head import _lip_node, bubble_with_curled_head

    P = np.asarray(bubble_with_curled_head()[0].points)

    class Curled(CutProblem):
        def interface_segments(self):
            segs = super().interface_segments()
            segs[0] = self._chain("bubble", P[:, 0].copy(), P[:, 1].copy(), ("axis", "axis"))
            return segs

    def gas_neck(drv, neck):
        assert neck.get("curled") is True
        k = _lip_node(P, which="bubble")
        return {"reason": ev.EVENT_REQUEST, "kind": "gas-neck", "class": "G",
                "handler": ev.HANDLERS["G"], "chain": "bubble", "index": k}

    res = _far_run(tmp_path, monkeypatch, request_at=gas_neck, pb=Curled())
    assert res["fills"] == [] and [c["kind"] for c in res["cuts"]] == ["far-head"]
    req, = res["requests"]
    assert req["index"] == res["neck"]["index"] and req["curled"] is True
    assert req["far_event"]["class"] == "G"
    assert res["out"]["events"]["by_kind"] == {"far-head": 1}
