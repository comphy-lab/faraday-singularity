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
