"""Contract tests of the stabilisation of the run after the cut, without a finite-element problem.

* The stops of a state after the cut are evaluated once per state (``CollapseDriver._post_cut_stop``,
  keyed by ``_stop_key``): the check at the loop's top of the state the step has just checked is
  not made again, and each caller is given its own copy; the step timing of the runner's
  ``--step-timing`` (``step_timing.txt``) reads clocks only and changes no other output.
* The parts ``jetcore`` and ``pinch`` of the events block: left out of the block and its hash at
  their defaults, validated, the keys of jetcore of the leg (its sizing keys among
  ``events.SIZING_KEYS``) and those of pinch of the class ``liquid``; a key whose behaviour the
  driver does not make yet is refused (``events.PENDING_KEYS``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from test_events_legacy import _same_outputs  # noqa: E402
from test_events_plumbing import LIQUID, _events_driver  # noqa: E402

from faraday_singularity import driver as drv_mod  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity.driver import STEP_TIMING_COLUMNS  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402


# --- C1: the stops once per state, and the step timing --------------------------------------------


def _counted(monkeypatch):
    """``post_cut_stops`` of the driver wrapped to count its calls."""
    calls = []
    real = drv_mod.post_cut_stops

    def counting(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(drv_mod, "post_cut_stops", counting)
    return calls


def test_the_stops_of_a_state_after_the_cut_are_evaluated_once(tmp_path, monkeypatch):
    """A stand-in run after a cut with the block, three steps and a rebuild: the stops are read
    once for the state after the cut, once for each step's state and once for the rebuilt mesh,
    not again at the loop's top; without the cache (a key that never repeats) they are read twice
    for each step. A copy is handed to each caller, so that a caller's change of it does not
    reach the next."""
    from test_bookkeeping import CutProblem

    block = ev.EventSettings.from_dict(LIQUID)
    counts = {}
    for name in ("cached", "uncached"):
        calls = _counted(monkeypatch)
        if name == "uncached":
            monkeypatch.setattr(drv_mod.CollapseDriver, "_stop_key", lambda self: object())
        drv = _events_driver(tmp_path / name, CutProblem(),
                             ReconnectSettings(delta=1e-3, dt0=1e-3), block, max_steps=3)
        out = drv.run(1e-9)
        assert out["stop"]["reason"] == "max-steps" and out["steps"] == 3
        counts[name] = (len(calls), out["remesh_count"])
        if name == "cached":
            kept = drv
        else:
            drv.close()
    (n_cached, rc), (n_plain, rc_plain) = counts["cached"], counts["uncached"]
    assert rc == rc_plain == 1
    assert n_cached == 1 + 3 + rc and n_plain == 1 + 2 * 3 + rc
    monkeypatch.undo()
    drv = kept
    calls = _counted(monkeypatch)
    # a copy per caller
    first = drv._post_cut_stop()
    n = len(calls)
    assert n == 0  # the state the run ended on, read already
    if first is not None:
        first["changed"] = True
    again = drv._post_cut_stop()
    assert len(calls) == n and (again is None or "changed" not in again)
    drv.close()


def test_the_cache_of_the_stops_follows_the_state(tmp_path, monkeypatch):
    """A new rebuild, step or event of the same time is a new state: its stops are read again."""
    from test_bookkeeping import CutProblem

    calls = _counted(monkeypatch)
    pb = CutProblem()
    drv = _events_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-3),
                         ev.EventSettings.from_dict(LIQUID))
    drv._post_cut_stop()
    n = len(calls)
    drv._post_cut_stop()
    assert len(calls) == n
    pb.remesh_count += 1
    drv._post_cut_stop()
    assert len(calls) == n + 1
    drv.state.step += 1
    drv._post_cut_stop()
    assert len(calls) == n + 2
    pb.events = [{"k": 1, "kind": "liquid-neck"}]
    drv._post_cut_stop()
    assert len(calls) == n + 3
    drv.close()


def test_the_step_timing_writes_its_rows_and_changes_no_other_output(tmp_path, monkeypatch):
    """Two stand-in runs after a cut with the block, one with ``step_timing``: every file but
    ``step_timing.txt`` the same (the wall times aside), and that file a row per step with every
    part not negative and the total their sum."""
    from test_bookkeeping import CutProblem

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    block = ev.EventSettings.from_dict(LIQUID)
    runs = {}
    for name, timing in (("plain", False), ("timed", True)):
        (tmp_path / name).mkdir()
        monkeypatch.chdir(tmp_path / name)  # the paths the runs record the same
        out = Path("run")
        out.mkdir()
        pb = CutProblem()
        drv = _events_driver(out, pb, recon, block, state_dt=0.0, state_keep=10, max_steps=4)
        if timing:  # as the driver's keyword sets it
            drv._timing = open(out / "step_timing.txt", "w", buffering=1)
            drv._timing.write("# " + "\t".join(STEP_TIMING_COLUMNS) + "\n")
        result = drv.run(1e-9)
        drv.close()
        (out / "result.json").write_text(json.dumps(result, default=float, sort_keys=True))
        runs[name] = tmp_path / name / "run"
    rows = np.loadtxt(runs["timed"] / "step_timing.txt", ndmin=2)
    (runs["timed"] / "step_timing.txt").unlink()
    _same_outputs(runs["plain"], runs["timed"])
    assert rows.shape == (4, len(STEP_TIMING_COLUMNS))
    cols = {c: rows[:, k] for k, c in enumerate(STEP_TIMING_COLUMNS)}
    assert list(cols["step"]) == [1, 2, 3, 4]
    parts = [c for c in STEP_TIMING_COLUMNS if c not in ("step", "time", "dt", "total")]
    assert all((cols[c] >= 0.0).all() for c in parts)
    assert np.allclose(sum(cols[c] for c in parts), cols["total"], rtol=1e-9, atol=1e-12)


def test_the_driver_keyword_opens_the_step_timing_and_the_runner_passes_it_on_its_flag(tmp_path):
    """``CollapseDriver(step_timing=True)`` writes the header of ``step_timing.txt``, and the
    runner makes the driver with ``step_timing`` only with ``--step-timing``."""
    from test_bookkeeping import CutProblem, _settings

    drv = drv_mod.CollapseDriver(CutProblem(), _settings(), tmp_path, step_timing=True)
    drv.close()
    head = (tmp_path / "step_timing.txt").read_text().splitlines()
    assert head == ["# " + "\t".join(STEP_TIMING_COLUMNS)]
    text = (ROOT / "run_case.py").read_text()
    assert '"--step-timing"' in text
    assert '**({"step_timing": True} if args.step_timing else {})' in text
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"), "--help"],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0 and "--step-timing" in proc.stdout


# --- C2: the optional parts jetcore and pinch of the events block ---------------------------------


@pytest.fixture
def made(monkeypatch):
    """Every key of jetcore and pinch admitted, as once the driver makes them all."""
    monkeypatch.setattr(ev, "PENDING_KEYS", {})


def test_the_parts_at_their_defaults_leave_every_block_and_its_hash_as_it_was():
    """``jetcore`` and ``pinch`` at their defaults are left out of the block, its hash and its
    record, and a record without them reads as their defaults."""
    for block in ({}, LIQUID, {"ring": {"enabled": True}}):
        s = ev.EventSettings.from_dict(block)
        assert "jetcore" not in s.to_dict() and "pinch" not in s.to_dict()
        assert s == ev.EventSettings.from_dict({**block, "jetcore": {}, "pinch": {}})
        assert s.sha256() == ev.EventSettings.from_dict(
            {**block, "jetcore": {"enabled": False}}).sha256()
        assert s.changed_keys(s.to_dict()) == []
    assert ev.EventSettings.OPTIONAL_PARTS == ("ring", "jetcore", "pinch")


def test_keys_not_made_yet_are_refused():
    """A key of either part whose behaviour the driver does not make yet is refused away from its
    default (``events.PENDING_KEYS``)."""
    pending = ev.PENDING_KEYS
    for name, part in (("jetcore", ev.JetCoreSettings), ("pinch", ev.PinchSettings)):
        for key in pending.get(name, ()):
            if key == "enabled":
                with pytest.raises(ValueError, match="not made yet"):
                    part(enabled=True)


def test_the_parts_away_from_their_defaults_are_written_and_hashed(made):
    jet = {"enabled": True, "far_floor": 1.5625e-5, "far_courant": 0.8, "far_kinks": "recentre",
           "far_refine_ratio": 2.0, "prep_gap": 4.0, "profile_dt": 2.5e-7}
    s = ev.EventSettings.from_dict({**LIQUID, "jetcore": jet,
                                    "pinch": {"enabled": True, "r_c": 0.1}})
    d = s.to_dict()
    assert d["jetcore"]["far_floor"] == 1.5625e-5 and d["pinch"]["r_c"] == 0.1
    assert ev.EventSettings.from_dict(json.loads(json.dumps(d))) == s
    assert s.sha256() != ev.EventSettings.from_dict(LIQUID).sha256()
    assert set(s.changed_keys(ev.EventSettings.from_dict(LIQUID).to_dict())) == {
        *(f"jetcore.{k}" for k in jet), "pinch.enabled", "pinch.r_c"}


@pytest.mark.parametrize("bad", [
    {"jetcore": {"far_floor": 1e-5}},  # a key without enabled
    {"jetcore": {"enabled": True, "core_radius": 0.0}},
    {"jetcore": {"enabled": True, "wall_radius": 1.0}},
    {"jetcore": {"enabled": True, "far_courant": -1.0}},
    {"jetcore": {"enabled": True, "far_refine_ratio": 1.0}},
    {"jetcore": {"enabled": True, "far_yield_turn": 180.0}},
    {"jetcore": {"enabled": True, "prep_rho": 0.9}},
    {"jetcore": {"enabled": True, "far_kinks": "shrink"}},
    {"jetcore": {"enabled": True, "edge_turn_snapshots": "far"}},
    {"jetcore": {"enabled": True, "box_gate": True}},  # no calibrated floors
    {"jetcore": {"enabled": True, "prep_gap": 1.0}, "ring": {"enabled": True, "gap": 1.0}},
    {"jetcore": {"enabled": True, "speed": 2}},
    {"pinch": {"enabled": True}, "liquid": {"enabled": False}},
    {"pinch": {"far_head": True}},  # without jetcore
    {"pinch": {"min_samples": 2}},
    {"pinch": {"r_c": 0.0}},
    {"pinch": {"enabled": 1}},
])
def test_the_parts_refuse_what_they_cannot_take(made, bad):
    block = json.loads(json.dumps(LIQUID))
    for name, part in bad.items():
        block[name] = {**block.get(name, {}), **part}
    with pytest.raises(ValueError):
        ev.EventSettings.from_dict(block)


def test_the_gate_takes_the_calibrated_floors(made):
    block = {"liquid": {**LIQUID["liquid"], "sigma_V": 1e-13, "sigma_Pz": 3e-10},
             "post": LIQUID["post"], "jetcore": {"enabled": True, "box_gate": True}}
    assert ev.EventSettings.from_dict(block).jetcore.box_gate


def test_jetcore_keys_are_of_the_leg_and_pinch_keys_of_the_class_liquid(made):
    """Every key of jetcore changes at any restart; a key of pinch is refused once a liquid event
    has been made, but from the state before an event; the sizing keys of jetcore make a dumped
    fresh mesh stepped (``events.SIZING_KEYS``)."""
    old = ev.EventSettings.from_dict(LIQUID).to_dict()
    new = ev.EventSettings.from_dict({**LIQUID, "jetcore": {"enabled": True, "far_floor": 1e-5},
                                      "pinch": {"enabled": True}})
    done = {"liquid": 1, "gap": 0, "ring": 0}
    assert new.refused_changes(old, done, None) == ["pinch.enabled"]
    assert new.refused_changes(old, done, 3) == []
    assert new.refused_changes(old, {"liquid": 0}, None) == []
    only_jet = ev.EventSettings.from_dict({**LIQUID, "jetcore": {"enabled": True,
                                                                 "far_courant": 0.8}})
    assert only_jet.refused_changes(old, done, None) == []
    assert ev.EventSettings.class_of("pinch.r_c") == "liquid"
    assert ev.EventSettings.class_of("jetcore.far_floor") is None
    assert ev.EventSettings.class_of("ring.after_removal") is None
    assert ev.EventSettings.class_of("ring.rho") == "ring"
    sizing = set(new.changed_keys(old)) & set(ev.SIZING_KEYS)
    assert sizing == {"jetcore.enabled", "jetcore.far_floor"}
    assert not set(only_jet.changed_keys(ev.EventSettings.from_dict(
        {**LIQUID, "jetcore": {"enabled": True}}).to_dict())) & set(ev.SIZING_KEYS)


def test_the_runner_admits_a_jetcore_change_and_refuses_a_pinch_change_after_a_liquid_event(made):
    import run_case

    old = ev.EventSettings.from_dict(LIQUID).to_dict()
    meta = {"events_settings": old, "events": [{"k": 1, "kind": "liquid-neck"}]}
    run_case._check_events(meta, ev.EventSettings.from_dict(
        {**LIQUID, "jetcore": {"enabled": True, "far_floor": 1e-5}}))
    with pytest.raises(SystemExit, match="pinch.enabled"):
        run_case._check_events(meta, ev.EventSettings.from_dict({**LIQUID,
                                                                 "pinch": {"enabled": True}}))
    run_case._check_events({**meta, "cut_stage": ev.PRECUT_EVENT, "event": {"k": 2}},
                           ev.EventSettings.from_dict({**LIQUID, "pinch": {"enabled": True}}))
