"""Contract tests of the stabilisation of the run after the cut, without a finite-element problem.

* The stops of a state after the cut are evaluated once per state (``CollapseDriver._post_cut_stop``,
  keyed by ``_stop_key``): the check at the loop's top of the state the step has just checked is
  not made again, and each caller is given its own copy; the step timing of the runner's
  ``--step-timing`` (``step_timing.txt``) reads clocks only and changes no other output.
* The parts ``jetcore`` and ``pinch`` of the events block: left out of the block and its hash at
  their defaults, validated, the keys of jetcore of the leg (its sizing keys among
  ``events.SIZING_KEYS``) and those of pinch of the class ``liquid``; a key whose behaviour the
  driver does not make yet is refused (``events.PENDING_KEYS``).
* The core of the upward jet (``jetcore.core_mask``: the base, the stem below the jet's tracked
  neck, the walls) and the split Courant limit (``jetcore.courant_split``), the limit of every node
  to the bit at one Courant number; stand-in runs with jetcore at its defaults take the same steps
  and write ``jetcore.txt``, and with a far Courant number of its own the far nodes take longer.
* The far region's sizing: the regions' classes (``jetcore.region_class``) and the guard's kinks
  split by the core (``jetcore.split_refine``), the far kinks recentred, floored and yielding
  below the floor only past the turn (``jetcore.far_kink_regions``), the far targets floored with
  the core's unchanged, the edge-turn profiles of the core.
* The refinement before the piercing (``jetcore.prep_regions``, ``prep_mask``): its regions once
  the bubble is shorter than ``prep_gap delta``, their size and their nodes stepped as the core's.
"""

from __future__ import annotations

import json
import math
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


# --- C3: the core of the upward jet and the split Courant limit -----------------------------------


def _jet_open_chain(delta=1e-3, r_n=0.3):
    """The open chain of ``test_events_stops.cavity_with_head`` (a jet rising from the trough of a
    groove, its head necked to ``r_n delta``) and its jet's base as the driver reads it."""
    from test_events_stops import cavity_with_head

    from faraday_singularity.collapse import Profile, jet_base
    from faraday_singularity.reconnect import jet_base_apex

    c = cavity_with_head(delta, r_n=r_n)
    r, z = c.points[:, 0], c.points[:, 1]
    jb = jet_base(Profile(r, z), 1.5, 1e-6)
    a = jet_base_apex(r, z, jb.index)
    return c, {"index": a["apex"], "r": jb.r_j, "z": jb.z_b, "R_apex": a["R_apex"]}


def test_the_core_is_the_base_of_the_jet_its_stem_below_the_neck_and_the_walls():
    """On a rising jet with a necked head: the nodes within ``R_K`` of the base are core but those
    above the jet's tracked neck (its head), the walls beyond the base within ``wall_radius L`` are
    core, the rest far; ``R_K`` is the larger of ``core_radius L`` and ``apex_extent R_apex``.
    Without a base of a jet the whole open chain is core."""
    from faraday_singularity.jetcore import core_mask, upward_neck

    delta = 1e-3
    c, apex = _jet_open_chain(delta)
    P = c.points
    kw = dict(delta=delta, core_radius=2.0, wall_radius=10.0, apex_extent=8.0, track_depth=0.05)
    core = core_mask(P, c.end_types, apex, **kw)
    L = max(apex["r"], delta)
    assert core["L"] == L and not core["whole"]
    assert core["R_K"] == max(2.0 * L, 8.0 * apex["R_apex"])
    neck = upward_neck(P, c.end_types, apex["index"], 0.05)
    assert neck is not None and 0 < neck["index"] < apex["index"]
    assert core["z_cap"] == neck["z"]
    m, idx = core["mask"], np.arange(len(P))
    d = np.hypot(P[:, 0] - apex["r"], P[:, 1] - apex["z"])
    assert not m[idx < neck["index"]].any()  # the head
    assert m[(idx >= neck["index"]) & (d <= core["R_K"])].all()
    walls = (idx > apex["index"]) & (d <= 10.0 * L)
    assert m[walls].all() and not m[(idx > apex["index"]) & (d > max(10.0 * L, core["R_K"]))].any()
    wider = core_mask(P, c.end_types, apex, **dict(kw, core_radius=4.0, wall_radius=20.0))
    assert wider["mask"].sum() > m.sum() and (wider["mask"] | ~m).all()
    whole = core_mask(P, c.end_types, None, **kw)
    assert whole["whole"] and whole["mask"].all()


def _random_segs(seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for k, n in enumerate((51, 77)):
        s = np.sort(rng.uniform(0.0, 1.0, n))
        out.append({"open": k == 0, "end_types": ("axis", "fixed"), "r": s,
                    "z": np.sin(3.0 * s), "u_r": rng.normal(0.0, 1.0, n),
                    "u_z": rng.normal(0.0, 3.0, n) * (rng.uniform(size=n) > 0.1)})
    return out


def test_the_split_limit_at_one_courant_number_is_the_limit_of_every_node_to_the_bit():
    from faraday_singularity.jetcore import courant_split
    from faraday_singularity.reconnect import courant_limit

    segs = _random_segs()
    rng = np.random.default_rng(5)
    masks = [rng.uniform(size=len(segs[0]["r"])) > 0.5, None]
    for c in (0.2, 0.4, 0.37):
        split = courant_split(segs, masks, core=c, far=c)
        assert split["dt"] == c * courant_limit(segs)["h_over_u"]
    split = courant_split(segs, masks, core=0.4, far=0.8)
    assert split["dt"] == min(split["dt_core"], split["dt_far"])
    assert split["dt"] >= 0.4 * courant_limit(segs)["h_over_u"]
    assert split["core"]["nodes"] + split["far"]["nodes"] == sum(len(c["r"]) for c in segs)
    assert split["far"]["nodes"] >= len(segs[1]["r"])
    allcore = courant_split(segs, [np.ones(len(c["r"]), bool) for c in segs], core=0.4, far=9.0)
    assert allcore["dt"] == 0.4 * courant_limit(segs)["h_over_u"] and allcore["binds"] == "core"


def _moving_problem():
    """``CutProblem`` with the bubble moving fast (its nodes set the Courant limit, far nodes:
    the open chain has no jet, so it is wholly core) and the open chain slowly."""
    from test_bookkeeping import CutProblem

    def speed(chain, r, z):
        return np.full_like(r, 50.0 if chain == "bubble" else 1.0)

    return CutProblem(speed=speed)


def _block(**jetcore):
    return ev.EventSettings.from_dict({**LIQUID, "jetcore": {"enabled": True, **jetcore}})


def test_jetcore_at_its_defaults_leaves_the_steps_and_writes_its_rows(tmp_path):
    """A stand-in run after the cut with jetcore at its defaults takes the steps of the run without
    it, to the bit, and writes a row of ``jetcore.txt`` per row of ``chains.txt``: the far bubble
    binds, at the core's Courant number, the core is the whole open chain (no jet)."""
    from faraday_singularity.jetcore import COLUMNS

    recon = ReconnectSettings(delta=1e-3, dt0=1e-9, courant=0.4, focus_courant=False)
    rows = {}
    for name, block in (("plain", ev.EventSettings.from_dict(LIQUID)), ("jet", _block())):
        drv = _events_driver(tmp_path / name, _moving_problem(), recon, block, max_steps=5)
        drv.run(1e-9)
        drv.close()
        rows[name] = (tmp_path / name / "chains.txt").read_text()
    assert rows["plain"] == rows["jet"]
    text = (tmp_path / "jet" / "jetcore.txt").read_text().splitlines()
    assert text[0] == "# " + "\t".join(COLUMNS) and len(text) == len(rows["jet"].splitlines())
    last = dict(zip(COLUMNS, text[-1].split("\t")))
    assert last["binds"] == "far" and float(last["far_courant"]) == 0.4
    assert last["limiter"] in ("courant-far", "growth", "dt_growth")
    assert not (tmp_path / "plain" / "jetcore.txt").exists()


def test_a_far_courant_number_of_its_own_lets_the_far_nodes_take_longer_steps(tmp_path):
    """With ``far_courant`` twice the core's the far bubble's limit doubles: the step the run
    takes is the split limit, ``min(0.4 min_K, 0.8 min_F)``, which the far part sets here."""
    from faraday_singularity.jetcore import COLUMNS

    recon = ReconnectSettings(delta=1e-3, dt0=1e-9, courant=0.4, focus_courant=False,
                              dt_growth=10.0)
    out = {}
    for name, block in (("plain", _block()), ("fast", _block(far_courant=0.8))):
        drv = _events_driver(tmp_path / name, _moving_problem(), recon, block, max_steps=6,
                             growth=10.0)
        drv.run(1e-9)
        drv.close()
        lines = (tmp_path / name / "jetcore.txt").read_text().splitlines()[1:]
        out[name] = [dict(zip(COLUMNS, x.split("\t"))) for x in lines]
    a, b = out["plain"][-1], out["fast"][-1]
    assert float(b["dt_far"]) == pytest.approx(2.0 * float(a["dt_far"]), rel=1e-12)
    assert float(b["dt_courant"]) == min(float(b["dt_core"]), float(b["dt_far"]))
    assert float(b["dt_courant"]) > float(a["dt_courant"])
    assert b["limiter"].startswith("courant-") and float(b["dt"]) == pytest.approx(
        float(out["fast"][-2]["dt_courant"]), rel=1e-12)
    legs = [json.loads(x) for x in (tmp_path / "fast" / "events.jsonl").read_text().splitlines()]
    leg = next(e for e in legs if e["event"] == "leg")
    assert leg["limits"]["jetcore"] == {"core_courant": 0.4, "far_courant": 0.8,
                                        "core_radius": 2.0, "wall_radius": 10.0}


# --- C4: the far region's sizing ------------------------------------------------------------------


def _readings(turns, L, mids):
    """A reading of a chain's quadratic elements (``reconnect.edge_reading``'s ``edges``) with the
    given turns, chords and midside nodes."""
    turns = np.asarray(turns, dtype=float)
    return {"edges": {"turn": turns, "L": np.full(len(turns), float(L)),
                      "mid": np.asarray(mids, dtype=float), "f": np.full(len(turns), 0.5),
                      "index": 2 * np.arange(len(turns)) + 1}}


def test_a_far_kink_is_recentred_floored_and_yields_below_the_floor_only_past_the_turn():
    """The far kink of a chain: about its worst far element at each request, four chords wide (no
    growth), at ``max(chord/4, far_floor)``; past ``far_yield_turn`` it halves its last target
    once the mesh has met it (a yield), never below ``h_floor``; released after its hold."""
    from faraday_singularity.jetcore import far_kink_regions

    h_F, h_floor = 1e-4, 1e-6
    kw = dict(hold=lambda L: 1.0, h_far=h_F, h_floor=h_floor, yield_turn=75.0)
    rd = {"bubble": _readings([40.0, 50.0, 10.0], 2e-4, [[0.1, 1.0], [0.2, 1.0], [0.3, 1.0]])}
    g = far_kink_regions({}, rd, {"bubble": np.array([0, 1])}, t=0.0, sizes={}, **kw)
    k = g["bubble_far"]
    assert (k["name"], k["r"], k["radius"], k["target"]) == ("kink_bubble_far", 0.2, 8e-4, h_F)
    assert k["yields"] == 0 and k["far"]
    # the worst element moves: the region follows it, no wider
    rd2 = {"bubble": _readings([60.0, 10.0, 10.0], 2e-4, [[0.5, 1.0], [0.2, 1.0], [0.3, 1.0]])}
    g2 = far_kink_regions(g, rd2, {"bubble": np.array([0])}, t=0.1, sizes={}, **kw)
    assert g2["bubble_far"]["r"] == 0.5 and g2["bubble_far"]["radius"] == 8e-4
    # past the turn: halved only once the mesh has the last target
    hot = {"bubble": _readings([80.0], 2e-4, [[0.5, 1.0]])}
    y0 = far_kink_regions(g2, hot, {"bubble": np.array([0])}, t=0.2,
                          sizes={"kink_bubble_far": 2 * h_F}, **kw)["bubble_far"]
    assert y0["target"] == h_F and y0["yields"] == 0
    y1 = far_kink_regions({"bubble_far": y0}, hot, {"bubble": np.array([0])}, t=0.3,
                          sizes={"kink_bubble_far": h_F}, **kw)["bubble_far"]
    assert y1["target"] == 0.5 * h_F and y1["yields"] == 1 and y1["yield_t"] == 0.3
    y2 = far_kink_regions({"bubble_far": y1}, hot, {"bubble": np.array([0])}, t=0.4,
                          sizes={"kink_bubble_far": h_F}, **kw)["bubble_far"]
    assert y2["target"] == 0.5 * h_F and y2["yields"] == 1  # not met yet: kept
    low = dict(y1, target=1.5 * h_floor)
    y3 = far_kink_regions({"bubble_far": low}, hot, {"bubble": np.array([0])}, t=0.5,
                          sizes={"kink_bubble_far": 1.5 * h_floor}, **kw)["bubble_far"]
    assert y3["target"] == h_floor
    # a floor of the chain's own (the ring's column) and the release
    ring = far_kink_regions({}, {"ring": _readings([40.0], 2e-4, [[0.1, 0.5]])},
                            {"ring": np.array([0])}, t=0.0, sizes={}, floors={"ring": 3e-4},
                            **kw)
    assert ring["ring_far"]["target"] == 3e-4
    gone = far_kink_regions(ring, {}, {}, t=2.0, sizes={}, **kw)["ring_far"]
    assert gone["released"] and gone["target"] == gone["h_before"] == 2e-4


def test_the_regions_and_kinks_are_split_by_the_core():
    from faraday_singularity.jetcore import CORE, FAR, NECK, PREP, region_class, split_refine

    P = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11)])
    core = {"mask": np.arange(11) >= 5}
    for name, r, cls in (("jet_base_cavity_tip", 0.0, CORE), ("cavity_tip", 0.0, FAR),
                         ("cavity_tip", 0.8, CORE), ("kink_open", 0.6, CORE),
                         ("kink_open", 0.1, FAR), ("neck_open", 0.1, NECK),
                         ("pierce_tip", 0.0, PREP), ("bubble_top", 0.9, FAR),
                         ("kink_bubble_far", 0.9, FAR), ("kink_ring", 0.9, FAR)):
        assert region_class({"name": name, "r": r, "z": 0.0}, core, P) == cls, name
    assert region_class({"name": "ring_cut", "r": 0, "z": 0, "frozen": True}, core, P) == "frozen"
    readings = {"open": {"edges": {"index": np.array([1, 3, 5, 7, 9])}}, "bubble": {}}
    near, far = split_refine({"open": np.array([0, 3, 4]), "bubble": np.array([2])}, readings,
                             core)
    assert list(near["open"]) == [3, 4] and list(far["open"]) == [0]
    assert list(far["bubble"]) == [2] and "bubble" not in near


def test_the_far_targets_are_floored_and_refined_at_the_far_ratio(tmp_path):
    """A stand-in run after the cut (the open chain without a jet, wholly core; the bubble far):
    with ``far_floor`` above the bubble's targets they are floored (class ``far``, ``floored``),
    the open chain's regions as without it; the remesh events record each region's class."""
    from test_bookkeeping import CutProblem

    recon = ReconnectSettings(delta=1e-3, dt0=1e-9, courant=0.4, focus_courant=False)
    got = {}
    for name, block in (("plain", _block()), ("floor", _block(far_floor=2e-4)),
                        ("ratio", _block(far_floor=2e-4, far_refine_ratio=50.0))):
        drv = _events_driver(tmp_path / name, CutProblem(), recon, block, max_steps=2)
        m = drv.measure()
        got[name] = {g["name"]: g for g in m["regions"]}
        drv.close()
    plain, floor = got["plain"], got["floor"]
    assert set(plain) == set(floor)
    for name, g in floor.items():
        if g["class"] == "core":
            assert g["target"] == plain[name]["target"], name
        elif g["class"] == "far":
            assert g["target"] == max(plain[name]["target"], 2e-4), name
            assert g.get("floored", False) == (plain[name]["target"] < 2e-4), name
    assert any(g.get("floored") for g in floor.values())
    assert {g["class"] for g in floor.values()} >= {"core", "far"}


def test_without_recentred_kinks_or_a_floor_the_kinks_are_the_rules_and_the_snapshots_may_be_cores(
        tmp_path):
    """jetcore at its defaults keeps the guard's kink regions as without it; with
    ``edge_turn_snapshots`` ``core`` the turn that writes a profile at every step is read on the
    core's elements only."""
    from test_bookkeeping import CutProblem

    recon = ReconnectSettings(delta=1e-3, dt0=1e-9)
    drv = _events_driver(tmp_path / "a", CutProblem(), recon,
                         _block(edge_turn_snapshots="core"), max_steps=1)
    pc = drv._post_cut_state()
    every = max(rd["turn"] for rd in pc["edges"].values() if np.isfinite(rd["turn"]))
    e = pc["edges"]["open"]["edges"]
    mask = pc["core"]["mask"]
    assert drv._edge_turn_max() == pytest.approx(float(np.nanmax(e["turn"][mask[e["index"]]])))
    assert drv._edge_turn_max() <= every
    drv.close()


def test_a_far_region_at_the_floor_asks_for_no_rebuild_and_one_above_it_at_the_far_ratio(
        tmp_path):
    """The rule's test of a far region (``class`` far) against ``far_refine_ratio``: at the floor,
    its longest element at the floor, no rebuild; coarser than the far ratio times its target, a
    rebuild; between the core's ratio and the far ratio, none."""
    from test_bookkeeping import CutProblem

    recon = ReconnectSettings(delta=1e-3, dt0=1e-9)
    drv = _events_driver(tmp_path, CutProblem(), recon, _block(far_floor=2e-4,
                                                               far_refine_ratio=2.0))
    asked = []

    def rule(h_max, cls="far"):
        g = {"name": "kink_bubble_far", "target": 2e-4, "h_max": h_max, "h_min": math.nan,
             "class": cls, "r": 0.0, "z": 4.5, "radius": 1e-3, "size": 2e-4}
        drv.pb.force_remesh = lambda: asked.append(h_max)
        drv.state.last_remesh_step = drv.state.refine_blocked_until = -10
        return drv._maybe_remesh_features({"regions": [g], "time": drv._t(), "ell": 1e-3,
                                           "U": 1.0, "h_near": 1e-3, "r_j": 0.0, "z_axis": 4.69,
                                           "axis_curvature": 0.0})

    assert rule(2e-4) is False and rule(3.5e-4) is False  # at the floor; within the far ratio
    assert rule(3.5e-4, cls="core") is not False  # the core's ratio, 1.5
    assert rule(4.5e-4) is not False and asked == [3.5e-4, 4.5e-4]
    drv.close()


# --- C5: the refinement before the piercing -------------------------------------------------------


def test_the_pre_pierce_regions_come_once_the_bubble_is_short_and_are_stepped_as_the_core():
    """``jetcore.prep_regions``: none while the bubble is at least ``prep_gap delta`` long; below
    it, ``pierce_tip`` and ``pierce_bottom`` at the bubble's two ends on the axis, of the radius of
    the downward jet's shoulder ``s_t``, at ``prep_rho s_t / 2.25``, floored at ``h_floor``; their
    nodes are those of ``jetcore.prep_mask``."""
    from test_events_stops import bubble_with_head

    from faraday_singularity.jetcore import prep_mask, prep_regions
    from faraday_singularity.ring import jet_shoulder

    P = bubble_with_head(1e-3, r_n=0.3)[0].points
    seg = {"r": P[:, 0], "z": P[:, 1]}
    length = float(P[-1, 1] - P[0, 1])
    kw = dict(delta=1e-3, prep_rho=0.125, h_floor=1e-7)
    assert prep_regions(seg, prep_gap=0.99 * length / 1e-3, **kw) == []
    assert prep_regions(seg, prep_gap=None, **kw) == [] and prep_mask(seg, []) is None
    got = prep_regions(seg, prep_gap=1.01 * length / 1e-3, **kw)
    s_t = float(P[jet_shoulder(P), 0])
    assert [g["name"] for g in got] == ["pierce_tip", "pierce_bottom"]
    assert [g["z"] for g in got] == [float(P[-1, 1]), float(P[0, 1])]
    for g in got:
        assert g["radius"] == s_t and g["target"] == pytest.approx(0.125 * s_t / 2.25)
    floored = prep_regions(seg, prep_gap=1.01 * length / 1e-3, delta=1e-3, prep_rho=0.125,
                           h_floor=1.0)
    assert all(g["target"] == 1.0 for g in floored)
    m = prep_mask(seg, got)
    d = np.minimum(np.hypot(P[:, 0], P[:, 1] - P[-1, 1]), np.hypot(P[:, 0], P[:, 1] - P[0, 1]))
    assert (m == (d <= s_t)).all() and m[0] and m[-1]


def test_before_the_piercing_the_driver_adds_the_regions_and_steps_their_nodes_at_the_cores_number(
        tmp_path):
    """A stand-in run after the cut whose bubble (an ellipse 2b long) is shorter than ``prep_gap
    delta``: the measured regions hold ``pierce_tip`` and ``pierce_bottom`` (class ``prep``, no far
    floor on them), the split Courant limit counts their nodes as core, and ``jetcore.txt`` the
    regions and ``s_t``; with the bubble longer, none."""
    from test_bookkeeping import CutProblem

    from faraday_singularity.jetcore import COLUMNS

    recon = ReconnectSettings(delta=1e-3, dt0=1e-9, courant=0.4, focus_courant=False)
    block = _block(prep_gap=4.0, far_floor=1.0, far_courant=0.8)
    short = _events_driver(tmp_path / "short", CutProblem(a=0.01, b=1.5e-3), recon, block,
                           max_steps=1)
    m = short.measure()
    prep = {g["name"]: g for g in m["regions"] if g["name"].startswith("pierce_")}
    assert set(prep) == {"pierce_tip", "pierce_bottom"}
    assert all(g["class"] == "prep" and not g.get("floored") and g["target"] < 1.0
               for g in prep.values())
    core = short._post_cut_state()["core"]
    n_open = len(short._post_cut_state()["segs"][1]["r"])
    assert core["split"]["core"]["nodes"] > n_open  # the open chain and the bubble's ends
    short.run(1e-9)
    short.close()
    row = dict(zip(COLUMNS, (tmp_path / "short" / "jetcore.txt").read_text().splitlines()[-1]
                   .split("\t")))
    assert row["prep"] == "2" and float(row["s_t"]) > 0.0
    long = _events_driver(tmp_path / "long", CutProblem(), recon, block, max_steps=1)
    assert not any(g["name"].startswith("pierce_") for g in long.measure()["regions"])
    assert long._post_cut_state()["core"]["split"]["core"]["nodes"] == n_open
    long.close()
