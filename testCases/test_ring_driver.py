"""The collapse driver in the ring stage on stand-in problems (``test_bookkeeping.CutProblem``):
the bookkeeping of a ring surgery made, the rows, profiles and dumps of the stage, and the
admission of a restart, without a finite-element problem.

* the bookkeeping of the ring cut (``CollapseDriver._on_ring_made``, ``reset_after_ring``): the
  apex histories and the kink region of the bubble dropped, the far tips of the ghosts of the
  bubble's heads retired, no remesh check left pending, ``ring.txt`` opened, a profile and an
  event ``ring-cut``;
* the steps after it: ``chains.txt`` with its columns, the bubble's keys NaN but its volume, the
  ring's; ``ring.txt`` a row per step; the totals of ``budget.txt`` net of the fill; the profiles
  with the ring's beside the open chain's; the dumps holding the record of the stage;
* a restart: in the ring stage under a block whose ring is not enabled refused, by the runner and
  the driver, from a rotating dump and from the state before an event alike; under the block with
  it, the stage restored.
"""

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
from test_events_plumbing import _admit, _events_driver
from test_runner import CASE, ROOT
from test_runner_manifest import RECON

sys.path.insert(0, str(ROOT / "src"))
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity.driver import RING_COLUMNS, CollapseDriver  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

RING_ON = ev.EventSettings.from_dict({"ring": {"enabled": True}})


class RingProblem(CutProblem):
    """``CutProblem`` that, once ``ringed``, has the open chain alone and a ring below it, an
    ellipse in the liquid about (0.01, 4.4), the ring's nodes moving down at ``w``."""

    def __init__(self, w=-0.5, **kw):
        super().__init__(**kw)
        self.ringed = False
        self.w = w
        self.ring = None

    def interface_segments(self):
        segs = super().interface_segments()
        return segs[1:] if self.ringed else segs

    def interface_arrays(self):
        c = next(c for c in self.interface_segments() if c["open"])
        return c["r"], c["z"], c["u_r"], c["u_z"]

    def interface_loops(self):
        if not self.ringed:
            return []
        t = np.linspace(0.0, 2.0 * math.pi, 200, endpoint=False)
        r, z = 0.01 + 4e-3 * np.cos(t), 4.4 + 2e-3 * np.sin(t)
        return [{"open": False, "end_types": ("loop", "loop"), "r": r, "z": z,
                 "u_r": np.zeros_like(r), "u_z": np.full_like(r, self.w),
                 "vertex": np.arange(len(r)) % 2 == 0}]


def ring_record(V_b):
    return {"k_cut": 1, "t_cut_ring": 0.0, "V_b": V_b, "V0": 0.3 * V_b, "rho": 1e-4, "r_w": 5e-5,
            "z_mid": 4.45, "removed": None,
            "sizing": {"h_ring": 6.25e-5, "h_col": 5e-6, "grading": 0.3,
                       "frozen": [{"name": "ring_bubble_bottom", "size": 2e-5, "r": 0.0,
                                   "z": 4.28, "radius": 0.01}],
                       "ring_cut": {"name": "ring_cut", "size": 5e-6, "r": 0.0, "z": 4.45,
                                    "radius": 2e-3}},
            "fill": {"V": 1e-9, "Pz": -2e-7, "KE": 3e-6}, "settings_sha256": RING_ON.sha256()}


def made_ring_cut(drv, pb, k=2):
    """The ring cut of event ``k`` as ``ringsurgery.ring_cut`` hands it to the driver: the
    problem in the ring stage, the record appended to its events, ``on_made`` called."""
    pb.ringed = True
    pb.ring = ring_record(float(pb.cut["geometry"]["bubble_volume"]))
    t = pb.t
    rec = {"schema": ev.EVENT_SCHEMA, "k": k, "kind": "piercing", "class": "W1",
           "handler": "ring", "chain": "bubble", "tip": None, "t": t, "tau": t, "tau_p": t + 1e-6,
           "outcome": "made", "filled": dict(pb.ring["fill"]), "dt": {"history": 1e-6}}
    pb.events = list(pb.events) + [rec]
    drv._on_ring_made(rec)
    return rec


def _table(path):
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return head, [dict(zip(head, line.split("\t"))) for line in lines[1:]]


def test_the_ring_cut_retires_the_bubbles_bookkeeping_and_the_stage_writes_its_rows(tmp_path):
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    pb = RingProblem()
    run = tmp_path / "run"
    drv = _events_driver(run, pb, recon, events=RING_ON, state_dt=0.0, state_keep=10)
    # a liquid cut made before, whose ghost moves towards the bubble's bottom
    pb.events = [{"k": 1, "kind": "liquid-neck", "outcome": "made", "tip": "bubble_top",
                  "t": 0.0, "removed": {"V": 2e-9, "Pz": 1e-7, "KE": 1e-6},
                  "ghost": {"far_tip": "bubble_bottom", "status": "free"}}]
    drv.state.apex_history = {"bubble_top": [0.0, 1e-4], "cavity_tip": [0.0, 2e-4]}
    drv.state.kinks = {"bubble": {"name": "kink_bubble"}}
    drv._checks_pending = True
    rec = made_ring_cut(drv, pb)
    st = drv.state
    assert not {"bubble_top", "bubble_bottom"} & set(st.apex_history)
    assert "bubble" not in st.kinks and drv._checks_pending is False
    ghost = pb.events[0]["ghost"]
    assert ghost["far_tip"] is None and ghost["status_note"] == "far tip gone (ring)"
    assert rec["step"] == 0 and rec["leg"] == 0 and rec["dt"]["before"] is not None
    assert (run / "ring.txt").is_file()
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "ring-cut" and events[-1]["k"] == 2
    # the stage's steps
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3
    head, chains = _table(run / "chains.txt")
    from faraday_singularity.driver import CHAIN_COLUMNS

    assert head == list(CHAIN_COLUMNS)
    after = [row for row in chains if int(row["step"]) >= 0][-3:]
    for row in after:
        assert math.isnan(float(row["z_bubble_top"])) and math.isnan(float(row["bridge"]))
        assert float(row["bubble_volume"]) > 0.0 and float(row["courant_chain"]) == 1.0
    head, rings = _table(run / "ring.txt")
    assert head == list(RING_COLUMNS) and len(rings) >= 3
    V = float(rings[-1]["ring_volume"])
    assert V == pytest.approx(2 * math.pi * 0.01 * math.pi * 4e-3 * 2e-3, rel=1e-3)
    assert float(rings[-1]["ring_volume_over_V_b"]) == pytest.approx(V / pb.ring["V_b"])
    assert float(rings[-1]["ring_u_max"]) == 0.5 and float(rings[-1]["V_filled"]) == 1e-9
    assert float(rings[-1]["ring_axis_limit"]) == pytest.approx(max(0.5 * 5e-5, 3 * 5e-6))
    _, budget = _table(run / "budget.txt")
    last = budget[-1]
    assert float(last["V_removed"]) == pytest.approx(2e-9 - 1e-9)
    assert float(last["Pz_removed"]) == pytest.approx(1e-7 + 2e-7)
    assert float(last["volume_total"]) == pytest.approx(1.0 + 1e-9) and int(last["events"]) == 2
    profiles = sorted((run / "profiles").glob("profile_*_ring.txt"))
    assert profiles and not list((run / "profiles").glob("profile_00000[3-9]_bubble.txt"))
    P = np.loadtxt(profiles[-1])
    assert np.array_equal(P[0], P[-1]) and len(P) == 201
    # the regions: the ring's frozen, at their sizes, beside the open chain's
    regions = {g["name"]: g for g in drv.measure()["regions"]}
    assert regions["ring_cut"]["target"] == 5e-6 and regions["ring_cut"]["kept"] == 5e-6
    assert math.isnan(regions["ring_cut"]["h_max"])
    assert "bubble_top" not in regions and "cavity_tip" in regions
    dump = max((run / "states").glob("state_*.json"))
    meta = json.loads(dump.read_text())
    assert meta["ring"] == pb.ring and "resume_at" not in meta


def test_a_restart_in_the_ring_stage_needs_the_ring_enabled(tmp_path, monkeypatch):
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    pb = RingProblem()
    run = tmp_path / "run"
    drv = _events_driver(run, pb, recon, events=RING_ON, state_dt=0.0, state_keep=10)
    made_ring_cut(drv, pb, k=1)
    drv.run(1e-9)
    drv.close()
    state = max((run / "states").glob("state_*.dump"))
    meta = json.loads(state.with_suffix(".json").read_text())
    assert meta["ring"]["k_cut"] == 1
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", SimpleNamespace(
        restore_cut=lambda problem, record: setattr(problem, "cut", record)))
    off = ev.EventSettings.from_dict({})
    for block in (off, ev.EventSettings.from_dict({"ring": {"enabled": False, "gap": 0.5}})):
        again = CollapseDriver(RingProblem(), _settings(t_end=1.0, dt_max=1.0, max_steps=3),
                               tmp_path / "again", reconnect=recon, events=block)
        with pytest.raises(ValueError, match="ring.enabled"):
            again.restore(meta)
        again.close()
    # the state before an event of the stage, too, which the class rule alone would admit
    precut = dict(meta, cut_stage=ev.PRECUT_EVENT, event={"k": 2, "kind": "ring-axis"})
    again = CollapseDriver(RingProblem(), _settings(t_end=1.0, dt_max=1.0, max_steps=3),
                           tmp_path / "precut", reconnect=recon, events=off)
    with pytest.raises(ValueError, match="ring.enabled"):
        again.restore(precut)
    again.close()
    # with it the stage comes back: the record, the template's ring sizing, ring.txt
    pb = RingProblem()
    pb.ringed = True
    pb.t = meta["time"]
    again = CollapseDriver(pb, _settings(t_end=1.0, dt_max=1.0, max_steps=4), tmp_path / "on",
                           reconnect=recon, events=RING_ON)
    again.restore(meta)
    assert pb.ring == meta["ring"] and pb.template.ring_stage is True
    assert pb.template.ring_sizing.h_ring == 6.25e-5
    res = again.run(1e-9)
    again.close()
    assert res["stop"]["reason"] == "max-steps" and (tmp_path / "on" / "ring.txt").is_file()
    # the runner refuses the same before any output
    case = json.loads(json.dumps(CASE))
    case["collapse"].update(reconnect=dict(RECON.to_dict(), delta=1e-3, dt0=1e-6),
                            reconnect_events={"ring": {"enabled": True}})
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    (run / "run-manifest.json").write_text(json.dumps(
        {"schema": "pyoomph-runner-manifest-v1", "component_commit": "a" * 40,
         "case_sha256": digest}))
    (run / "parameters.json").write_text(json.dumps(
        {"case": case, "provenance": {"case_sha256": digest}}))
    assert _admit(state, case, case_path)[1] == "post-cut"
    plain = json.loads(json.dumps(case))
    plain["collapse"]["reconnect_events"] = {}
    plain_path = tmp_path / "plain.json"
    plain_path.write_text(json.dumps(plain))
    with pytest.raises(SystemExit, match="ring.enabled"):
        _admit(state, plain, plain_path)


# --- the routing of the ring's handlers (ringsurgery stood in for) --------------------------------


class PiercedProblem(RingProblem):
    """``RingProblem`` whose bubble is 8e-4 long (delta 1e-3): pierced at the start; its ring, once
    cut, empties at ``removed`` (no loop)."""

    def __init__(self, **kw):
        super().__init__(a=1e-3, b=4e-4, **kw)
        self.removed = False

    def interface_loops(self):
        return [] if self.removed else super().interface_loops()


def fake_cut(calls, fail=None):
    from faraday_singularity import surgery

    def ring_cut(pb, events, reconnect, outdir, request, *, k, dt_prev, elements_per_scale,
                 h_floor, driver=None, on_made=None, log=print):
        calls.append(("cut", request["kind"], request["class"], k, dt_prev))
        precut = str(driver.keep_state(Path(outdir) / "states" / f"precut_e{k:03d}.dump",
                                       cut_stage=ev.PRECUT_EVENT, event=dict(request, k=k)))
        if fail is not None:
            err = fail("refused for the test")
            err.precut, err.record = precut, {"outcome": "refused", "failed_at": "geometry",
                                              "error": "refused for the test"}
            raise err
        made = made_ring_cut(driver, pb, k=k)
        driver.keep_state(Path(outdir) / "states" / f"postcut_e{k:03d}.dump",
                          cut_stage="postcut-event", event_k=k)
        return made

    def ring_removal(pb, events, reconnect, outdir, request, *, k, dt_prev, elements_per_scale,
                     h_floor, driver=None, on_made=None, log=print):
        calls.append(("removal", request["kind"], request["class"], k, dt_prev))
        pb.removed = True
        pb.ring = dict(pb.ring, removed={"k": k, "t": pb.t, "tau": pb.t, "V": 1e-9})
        rec = {"schema": ev.EVENT_SCHEMA, "k": k, "kind": "ring-collapse", "class": "W2",
               "handler": "ring-removal", "chain": "ring", "t": pb.t, "tau": pb.t,
               "tau_p": pb.t + 1e-6, "outcome": "made", "filled": {"V": 1e-9, "Pz": 0.0,
                                                                   "KE": 0.0},
               "dt": {"history": dt_prev}}
        pb.events = list(pb.events) + [rec]
        on_made(rec)
        return rec

    return ring_cut, ring_removal, surgery


def test_the_piercing_is_met_by_the_ring_cut_and_its_refusals_are_resumable_stops(tmp_path,
                                                                               monkeypatch):
    from faraday_singularity import ringsurgery

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    for name, fail in (("made", None), ("refused", "CutRefused"), ("unclosed", "BudgetUnclosed")):
        calls = []
        cut, removal, _ = fake_cut(calls, None if fail is None else getattr(surgery_mod(), fail))
        monkeypatch.setattr(ringsurgery, "ring_cut", cut)
        monkeypatch.setattr(ringsurgery, "ring_removal", removal)
        run = tmp_path / name
        pb = PiercedProblem()
        out = _events_driver(run, pb, recon, events=RING_ON, state_dt=0.0).run(1e-9)
        assert calls and calls[0][:4] == ("cut", "piercing", "W1", 1), (name, calls)
        if fail is None:
            assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3
            assert out["events"]["made"] == 1 and out["events"]["ring"]["cut"] == 1
            row = (run / "events.txt").read_text().splitlines()[1].split("\t")
            head = (run / "events.txt").read_text().splitlines()[0][2:].split("\t")
            assert row[head.index("chain")] == "ring" and row[head.index("kind")] == "piercing"
            assert float(row[head.index("V_rem")]) == -1e-9
        else:
            reason = ev.RING_CUT_REFUSED if fail == "CutRefused" else ev.EVENT_UNCLOSED
            assert out["stop"]["reason"] == reason and out["steps"] == 0
            assert Path(out["stop"]["precut"]).name == "precut_e001.dump"
            assert out["events"]["made"] == 0


def surgery_mod():
    from faraday_singularity import surgery

    return surgery


def test_the_rings_removal_ends_the_window_or_the_run_goes_on(tmp_path, monkeypatch):
    from faraday_singularity import ringsurgery

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    for mode in ("stop", "continue"):
        calls = []
        cut, removal, _ = fake_cut(calls)
        monkeypatch.setattr(ringsurgery, "ring_cut", cut)
        monkeypatch.setattr(ringsurgery, "ring_removal", removal)
        block = ev.EventSettings.from_dict({"ring": {"enabled": True, "after_removal": mode}})
        pb = PiercedProblem()
        drv = _events_driver(tmp_path / mode, pb, recon, events=block, state_dt=0.0,
                             max_steps=4)
        # the ring cut at the start; then the bubble's volume at the gas cut far above the ring's,
        # so that the ring lies below bubble_stop of it after the first step
        orig = drv._post_cut_stop

        def stop():
            if pb.ring is not None and pb.t > 0.0:
                pb.cut["geometry"]["bubble_volume"] = 1e6
            return orig()

        drv._post_cut_stop = stop
        out = drv.run(1e-9)
        drv.close()
        kinds = [c[0] for c in calls]
        assert kinds == ["cut", "removal"], calls
        assert calls[1][1:3] == ("ring-collapse", "W2")
        if mode == "stop":
            assert out["stop"]["reason"] == "bubble-vanishing" and out["stop"]["window_end"]
            assert out["stop"]["ring_removed"] == 2 and out["window_end"] is not None
        else:
            assert out["stop"]["reason"] == "max-steps" and out["steps"] == 4
            assert out["events"]["ring"]["removed"] == 2
            rows = (tmp_path / mode / "ring.txt").read_text().splitlines()
            assert rows[-1].split("\t")[RING_COLUMNS.index("ring_volume")] == "nan"


def test_without_the_ring_the_piercing_stays_an_event_without_a_handler(tmp_path, monkeypatch):
    from faraday_singularity import ringsurgery

    calls = []
    cut, removal, _ = fake_cut(calls)
    monkeypatch.setattr(ringsurgery, "ring_cut", cut)
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    out = _events_driver(tmp_path, PiercedProblem(), recon, events=ev.EventSettings()).run(1e-9)
    assert out["stop"]["reason"] == ev.EVENT_UNHANDLED and out["stop"]["handler"] == "ring"
    assert calls == []


def test_in_the_ring_stage_a_landed_liquid_neck_goes_to_the_liquid_cut(tmp_path, monkeypatch):
    """In the ring stage the driver makes the liquid cut of a landed neck of the open chain
    (``surgery.liquid_cut``, stood in for here by a refusal) as before the ring, not the resumable
    stop without a handler of the first delivery: the request reaches the cut with the run's last
    step and the elements per scale, and its refusal is the cut's own resumable stop
    (``liquid-cut-refused``, with the state before it)."""
    from faraday_singularity import surgery

    calls = []

    def liquid_cut(pb, events, reconnect, outdir, request, *, k, dt_prev, elements_per_scale,
                   h_floor, driver=None, on_made=None, log=print):
        calls.append((request["kind"], request["class"], k, pb.ring is not None))
        precut = str(driver.keep_state(Path(outdir) / "states" / f"precut_e{k:03d}.dump",
                                       cut_stage=ev.PRECUT_EVENT, event=dict(request, k=k)))
        err = surgery.CutRefused("refused for the test")
        err.precut, err.record = precut, {"outcome": "refused", "failed_at": "caps",
                                          "error": "refused for the test"}
        raise err

    monkeypatch.setattr(surgery, "liquid_cut", liquid_cut)
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    pb = RingProblem()
    block = ev.EventSettings.from_dict({"liquid": {"enabled": True}, "ring": {"enabled": True},
                                        "post": {"open_neck_stop": False, "neck_floor": 1e-6}})
    run = tmp_path / "run"
    drv = _events_driver(run, pb, recon, events=block, state_dt=0.0, state_keep=10)
    made_ring_cut(drv, pb, k=1)
    request = {"reason": ev.EVENT_REQUEST, "kind": "liquid-neck", "class": "L",
               "handler": ev.HANDLERS["L"], "chain": "open", "index": 3, "r_n": 1e-5,
               "z_n": 4.7, "s": 1e-4, "side": "before", "direction": 1.0, "status": "head",
               "head_volume": 3e-12}
    orig = drv._post_cut_stop
    drv._post_cut_stop = lambda: dict(request) if pb.ring is not None else orig()
    out = drv.run(1e-9)
    drv.close()
    assert calls == [("liquid-neck", "L", 2, True)]
    assert out["stop"]["reason"] == ev.LIQUID_CUT_REFUSED and out["stop"].get("stage") is None
    assert Path(out["stop"]["precut"]).name == "precut_e002.dump" and out["steps"] == 0
