"""Contract tests of the core of the upward jet meshed finer (``jetcore.core_elements_per_scale``),
without a finite-element problem.

* The key: null by default, positive and finite or null, needing ``jetcore.enabled``; left out of
  the block and its hash where it is null (``EventSettings.LATE_KEYS``), so that every block
  recorded before it, and its hash, is unchanged; a key of the leg (any post-cut restart may change
  it, after events of every class too) that sizes the mesh (``events.SIZING_KEYS``); refused below
  the run's elements per scale after the cut, by the driver and by the runner before any output;
  the block with it refused where the block is (a run from the start, an ordinary restart); a
  post-cut continuation that changes it admitted by the runner, its lineage naming it.
* The core as a rebuild selects it again (``jetcore.core_spec``, ``core_select``): the mask of
  ``jetcore.core_mask`` exactly, with and without the capping neck and for a whole open chain; a
  focus region with it (``mesh.FocusRegion.core``) holds the core's nodes of its ball on the open
  chain alone, and a region without it writes the record it wrote before.
* The twins (``jetcore.core_twins``) of the regions whose ball holds core nodes, at ``q`` times
  their target (floored at ``h_floor``, flagged), none for the ring's frozen regions, the
  pre-pierce regions, the guard's own kinks or a region off the core; the guard's turn on the core
  at ``q`` times its limit (``jetcore.core_turns``); the spacing a region reads restricted to its
  nodes (``reconnect.region_spacing``'s ``select``).
* Stand-in runs after the cut on an open chain with a jet: with the key every region is the
  region without it (its target and ball), each region holding core nodes has its twin at half
  its target, the spacing of a twin reads the core and that of its region the rest, the rule asks
  for a rebuild from the twins and the focus it builds carries the core's selection; the leg
  records the key; a dumped fresh mesh restarted with the key changed is stepped on, so that the
  rule of its rebuilds meets the new sizes.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from test_bookkeeping import CutProblem, _settings  # noqa: E402
from test_events_plumbing import (LIQUID, _admit, _check, _cut_case, _events_driver,  # noqa: E402
                                  _write)
from test_runner_manifest import RECON, _parent_run, _write_meta  # noqa: E402

import run_case  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import jetcore as J  # noqa: E402
from faraday_singularity.driver import CollapseDriver  # noqa: E402
from faraday_singularity.mesh import Focus, FocusRegion  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings, region_spacing  # noqa: E402

KEY = "jetcore.core_elements_per_scale"
# the production part of the delta triple's events block (case r1), as a block of the stand-ins
JET = {"enabled": True, "core_radius": 2.0, "wall_radius": 10.0, "far_courant": 1.6,
       "far_floor": 1.5625e-5, "far_kinks": "recentre", "far_yield_turn": 75.0,
       "far_refine_ratio": 2.0, "ring_kink_floor": True, "edge_turn_snapshots": "core",
       "prep_gap": 4.0, "prep_rho": 0.125, "profile_dt": 2.5e-7}


def _block(**jet) -> ev.EventSettings:
    return ev.EventSettings.from_dict({**LIQUID, "jetcore": {**JET, **jet}})


# --- the key --------------------------------------------------------------------------------------


def test_the_key_is_left_out_of_the_block_and_its_hash_where_it_is_null():
    """Null, the part's block and hash are those of a part without the key, as every block recorded
    before it; set, it is written and hashed, and a record without it reads as null."""
    plain, null = _block(), _block(core_elements_per_scale=None)
    assert plain == null and plain.to_dict() == null.to_dict()
    assert "core_elements_per_scale" not in plain.to_dict()["jetcore"]
    assert plain.sha256() == null.sha256()
    fine = _block(core_elements_per_scale=16)
    assert fine.to_dict()["jetcore"]["core_elements_per_scale"] == 16
    assert fine.sha256() != plain.sha256()
    assert ev.EventSettings.from_dict(json.loads(json.dumps(fine.to_dict()))) == fine
    assert fine.changed_keys(plain.to_dict()) == [KEY]
    assert plain.changed_keys(fine.to_dict()) == [KEY]
    assert plain.changed_keys(plain.to_dict()) == []
    assert ev.EventSettings.LATE_KEYS == {"jetcore": ("core_elements_per_scale",)}
    # without jetcore the part is left out whole, as before
    assert "jetcore" not in ev.EventSettings.from_dict(LIQUID).to_dict()


@pytest.mark.parametrize("bad", [0, -8, math.inf, True, "16"])
def test_the_key_is_positive_and_finite_or_null(bad):
    with pytest.raises(ValueError, match="core_elements_per_scale"):
        _block(core_elements_per_scale=bad)
    with pytest.raises(ValueError, match="need jetcore.enabled"):
        ev.EventSettings.from_dict({**LIQUID, "jetcore": {"core_elements_per_scale": 16}})


def test_the_ratio_refines_the_core_and_is_refused_below_the_runs_number():
    assert J.core_ratio(None, 8.0) is None and J.core_ratio(8, 8.0) is None
    assert J.core_ratio(16, 8.0) == 0.5 and J.core_ratio(24.0, 8) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="at least the run's elements per scale"):
        J.core_ratio(4, 8.0)


def test_the_key_is_of_the_leg_and_sizes_the_mesh():
    """A key of the leg: no class, never refused, after events of every class and from any state;
    a sizing key, so that a dumped fresh mesh restarted under it is stepped on."""
    assert KEY in ev.SIZING_KEYS and ev.EventSettings.class_of(KEY) is None
    old = _block().to_dict()
    new = _block(core_elements_per_scale=16)
    done = {"liquid": 3, "gap": 1, "ring": 2}
    assert new.refused_changes(old, done, None) == []
    assert new.refused_changes(old, done, 4) == []
    assert set(new.changed_keys(old)) & set(ev.SIZING_KEYS) == {KEY}


def test_the_driver_refuses_a_core_coarser_than_the_run(tmp_path):
    """Below the run's elements per scale after the cut (the reconnect block's or the collapse's)
    the driver is not made; at it, or above, it is."""
    recon = ReconnectSettings(delta=1e-3, dt0=1e-9, elements_per_scale=8)
    with pytest.raises(ValueError, match="at least the run's elements per scale"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "a", reconnect=recon,
                       events=_block(core_elements_per_scale=4))
    for eps in (8, 16):
        CollapseDriver(CutProblem(), _settings(), tmp_path / f"b{eps}", reconnect=recon,
                       events=_block(core_elements_per_scale=eps)).close()
    collapse = ReconnectSettings(delta=1e-3, dt0=1e-9)  # the collapse's own number
    with pytest.raises(ValueError, match="at least the run's elements per scale"):
        CollapseDriver(CutProblem(), _settings(elements_per_scale=20), tmp_path / "c",
                       reconnect=collapse, events=_block(core_elements_per_scale=16))


# --- the runner's admission -----------------------------------------------------------------------


def _parent_with(tmp_path, block: dict, events: list[dict]):
    parent = _cut_case(reconnect_events=block)
    state, _ = _parent_run(tmp_path, case=parent)
    _write_meta(state, cut={"delta": RECON.delta, "t_cut": 0.5 - 0.5e-4,
                            "settings": RECON.to_dict()},
                events_settings=ev.EventSettings.from_dict(block).to_dict(), events=events,
                legs=[])
    return parent, state


def test_a_post_cut_continuation_may_change_the_key_after_events_of_every_class(tmp_path):
    """A post-cut continuation of a dump of a run with the block (a liquid cut, a fill and a ring
    cut made) under its case with the key added: admitted by the runner (``--check``, no output),
    its lineage naming the key; a change of a class key in the same continuation is refused."""
    block = {**LIQUID, "jetcore": JET}
    made = [{"k": 1, "kind": "liquid-neck"}, {"k": 2, "kind": "groove-fill"},
            {"k": 3, "kind": "piercing"}]
    parent, state = _parent_with(tmp_path, block, made)
    child = json.loads(json.dumps(parent))
    child["case_id"] = "sigterm-core"
    child["collapse"]["reconnect_events"]["jetcore"]["core_elements_per_scale"] = 16
    path = _write(tmp_path, child, "core.json")
    record, mode = _admit(state, child, path)
    assert mode == run_case.POST_CUT
    assert record["parent_record"]["derived_keys"] == [
        "case_id", "collapse.reconnect_events.jetcore.core_elements_per_scale"]
    proc = _check(path, tmp_path / "out", state)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1])["restart_mode"] == "post-cut"
    assert not (tmp_path / "out").exists()
    both = json.loads(json.dumps(child))
    both["collapse"]["reconnect_events"]["liquid"]["eps_d"] = 0.1
    with pytest.raises(SystemExit, match="liquid.eps_d"):
        _admit(state, both, _write(tmp_path, both, "both.json"))


def test_the_block_with_the_key_is_refused_where_the_block_is_and_below_the_runs_number(
        tmp_path):
    """Where the block is refused (a run from the start, an ordinary restart before any cut) it is
    with the key too; and a core coarser than the run (the key below the collapse's elements per
    scale, the reconnect block setting none) is refused before any output."""
    block = {**LIQUID, "jetcore": {**JET, "core_elements_per_scale": 16}}
    out = tmp_path / "out"
    proc = _check(_write(tmp_path, _cut_case(reconnect_events=block), "fresh.json"), out)
    assert proc.returncode != 0 and "only a cut or post-cut restart takes it" in proc.stderr
    state, _ = _parent_run(tmp_path / "p", case=_cut_case())
    child = _cut_case(reconnect_events=block, landing=[1e-3])
    child["case_id"] = "sigterm-ordinary"
    with pytest.raises(SystemExit, match="applies after a cut"):
        _admit(state, child, _write(tmp_path, child, "ordinary.json"))
    coarse = _cut_case(reconnect_events={**LIQUID, "jetcore": {**JET,
                                                               "core_elements_per_scale": 2}})
    proc = _check(_write(tmp_path, coarse, "coarse.json"), out)
    assert proc.returncode != 0 and "Traceback" not in proc.stderr, proc.stderr[-2000:]
    assert "at least the run's elements per scale" in proc.stderr
    assert not out.exists()


# --- the core selected again, the focus region ----------------------------------------------------


def _jet_chain(delta=1e-3, r_n=0.3):
    from test_jetcore import _jet_open_chain

    return _jet_open_chain(delta, r_n=r_n)


def _core(points, end_types, apex, delta=1e-3):
    return J.core_mask(points, end_types, apex, delta=delta, core_radius=2.0, wall_radius=10.0,
                       apex_extent=8.0, track_depth=0.05)


def test_the_core_selected_again_is_the_core_mask_exactly():
    """With the capping neck (the jet's head excluded), without it (a head too shallow to track),
    and for a whole open chain (no base): :func:`jetcore.core_select` of :func:`core_spec` is
    :func:`jetcore.core_mask`'s mask, node for node."""
    for r_n in (0.3, 0.49):
        c, apex = _jet_chain(r_n=r_n)
        core = _core(c.points, c.end_types, apex)
        assert not core["whole"]
        spec = J.core_spec(core, c.points, 10.0)
        assert np.array_equal(J.core_select(c.points, spec), core["mask"]), r_n
        assert spec[2] == core["R_K"] and spec[3] == 10.0 * core["L"]
        assert math.isfinite(spec[6]) == (core["neck"] is not None)
    c, apex = _jet_chain()
    assert _core(c.points, c.end_types, apex)["neck"] is not None
    whole = _core(c.points, c.end_types, None)
    assert whole["whole"] and all(math.isnan(x) for x in J.core_spec(whole, c.points, 10.0))
    assert J.core_select(c.points, J.core_spec(whole, c.points, 10.0)).all()


def test_a_focus_region_with_the_core_holds_its_nodes_of_the_open_chain_alone():
    c, apex = _jet_chain()
    core = _core(c.points, c.end_types, apex)
    spec = J.core_spec(core, c.points, 10.0)
    A = (core["A_r"], core["A_z"])
    plain = FocusRegion("neck_open", size=1e-5, r=A[0], z=A[1], radius=3e-3)
    twin = FocusRegion("core_neck_open", size=5e-6, r=A[0], z=A[1], radius=3e-3, core=spec)
    ball = plain.selects(c.points)
    assert np.array_equal(plain.selects(c.points, False), ball)  # no core: as before
    assert np.array_equal(twin.selects(c.points), ball & core["mask"])
    assert ball.sum() > twin.selects(c.points).sum() > 0
    assert not twin.selects(c.points, False).any()  # a chain not open, a loop: none of it
    # the record of a region without it is the one it wrote before; with it, it round-trips
    assert plain.to_dict() == {k: v for k, v in asdict(plain).items() if k != "core"}
    assert "core" not in plain.to_dict() and twin.to_dict()["core"] == spec
    f = Focus(size=5e-6, radius=3e-3, grading=0.2, regions=(plain, twin))
    again = Focus(**json.loads(json.dumps(f.to_dict())))
    assert again == f and again.region("core_neck_open").core == spec
    assert "core" not in json.dumps(Focus(size=1e-5, radius=3e-3, grading=0.2,
                                          regions=(plain,)).to_dict())


# --- the twins, the guard's turns, the spacing ----------------------------------------------------


def test_the_twins_halve_the_regions_on_the_cores_nodes_and_nothing_else():
    """Each region whose ball holds core nodes has a twin of its ball at ``q`` times its target,
    class core, its attractors the core's (``core``), floored at ``h_floor`` and flagged where the
    floor clips; ``kept`` scaled; none for a region off the core, the ring's frozen regions, the
    pre-pierce regions or the guard's own kinks (a far kink has one); the regions themselves are
    left as they were."""
    c, apex = _jet_chain()
    P = c.points
    core = _core(P, c.end_types, apex)
    spec = J.core_spec(core, P, 10.0)
    A = (core["A_r"], core["A_z"])

    def region(name, target, r=A[0], z=A[1], radius=1e-3, **kw):
        return {"name": name, "target": target, "r": r, "z": z, "radius": radius,
                "ell": math.inf, "floor": 1e-7, **kw}

    regions = [region("jet_base_cavity_tip", 2e-5),
               region("neck_open", 3.5e-5), region("cavity_tip", 6.25e-5, radius=0.02),
               region("bubble_bottom", 5e-5, r=0.0, z=4.0, kept=5e-5),  # off the core
               region("kink_open", 4e-6),  # the guard's own kink: none
               region("kink_open_far", 1.5625e-5),  # a far kink: one
               region("pierce_tip", 1e-5, **{"class": J.PREP}),
               region("ring_cut", 1e-5, frozen=True),
               region("tiny", 1.5e-6), region("kept_one", 4e-5, kept=3e-5)]
    before = json.loads(json.dumps(regions))
    twins = J.core_twins(regions, P, core, spec, ratio=0.5, h_floor=1e-6)
    assert regions == before
    by = {g["twin_of"]: g for g in twins}
    assert set(by) == {"jet_base_cavity_tip", "neck_open", "cavity_tip", "kink_open_far",
                       "tiny", "kept_one"}
    for g in twins:
        orig = next(x for x in regions if x["name"] == g["twin_of"])
        assert g["name"] == "core_" + orig["name"] and g["class"] == J.CORE
        assert (g["r"], g["z"], g["radius"]) == (orig["r"], orig["z"], orig["radius"])
        assert g["target"] == max(0.5 * orig["target"], 1e-6) and g["core"] == spec
        assert g.get("core_floored", False) == (0.5 * orig["target"] < 1e-6)
        inside = np.hypot(P[:, 0] - orig["r"], P[:, 1] - orig["z"]) <= orig["radius"]
        assert g["core_nodes"] == int((inside & core["mask"]).sum()) > 0
    assert by["tiny"]["core_floored"] and by["tiny"]["target"] == 1e-6
    assert by["kept_one"]["kept"] == 1.5e-5 and "kept" not in by["neck_open"]
    assert J.core_kink("kink_open") and J.core_kink("kink_ring")
    assert not J.core_kink("kink_open_far") and not J.core_kink("neck_open")


def test_the_guard_reads_the_cores_turns_at_q_times_its_limit():
    """The elements of the open chain whose midside node is core and whose turn exceeds the
    scaled limit, longer than the rule can refine; none off the core."""
    mask = np.array([True] * 6 + [False] * 5)
    e = {"index": np.array([1, 3, 5, 7, 9]), "turn": np.array([20.0, 10.0, 16.0, 40.0, 20.0]),
         "L": np.array([1e-5, 1e-5, 1e-9, 1e-5, 1e-5])}
    got = J.core_turns({"open": {"edges": e}}, {"mask": mask}, turn=15.0, min_length=1e-8)
    assert list(got) == [0]  # 2: below the length; 3, 4: off the core; 1: below the turn


def test_the_spacing_of_a_region_reads_the_nodes_it_selects():
    c, apex = _jet_chain()
    core = _core(c.points, c.end_types, apex)
    seg = {"open": True, "r": c.points[:, 0], "z": c.points[:, 1]}
    g = {"name": "x", "r": core["A_r"], "z": core["A_z"], "radius": 5e-3}
    every = region_spacing([seg], g, [], None, 1e-5)
    mask = core["mask"]
    on = region_spacing([seg], g, [], None, 1e-5, select=lambda c: mask)
    off = region_spacing([seg], g, [], None, 1e-5, select=lambda c: ~mask)
    assert region_spacing([seg], g, [], None, 1e-5, select=None) == every
    assert max(on[0], off[0]) == every[0] and min(on[1], off[1]) == every[1]


# --- stand-in runs after the cut ------------------------------------------------------------------


class JetProblem(CutProblem):
    """``test_bookkeeping.CutProblem`` whose open chain is ``test_events_stops.cavity_with_head`` at
    delta 1e-3: a jet rising from the trough of a groove, its head necked to 0.3 delta."""

    def __init__(self, **kw):
        from test_events_stops import cavity_with_head

        super().__init__(**kw)
        self._open = cavity_with_head(1e-3).points

    def interface_segments(self):
        segs = super().interface_segments()
        P = self._open
        segs[1] = self._chain("open", P[:, 0].copy(), P[:, 1].copy(), ("axis", "fixed"))
        return segs

    def interface_arrays(self):
        c = self.interface_segments()[1]
        return c["r"], c["z"], c["u_r"], c["u_z"]


RECON_8 = ReconnectSettings(delta=1e-3, dt0=1e-9, courant=0.4, focus_courant=False,
                            elements_per_scale=8)


def _regions(tmp_path, block, problem=JetProblem):
    drv = _events_driver(tmp_path, problem(), RECON_8, block, max_steps=1)
    m = drv.measure()
    return drv, m, {g["name"]: g for g in m["regions"]}


def test_with_the_key_every_region_is_as_without_it_and_its_twin_halves_it(tmp_path):
    """On the open chain with a jet (its base, the stem to the neck and the walls core): every
    region of the run without the key is a region of the run with it, the same in all but its
    spacing, which reads the nodes off the core where it has a twin; each region holding core nodes
    has its twin at half its target, reading the core's nodes alone; the regions without a twin
    hold none."""
    drv0, m0, plain = _regions(tmp_path / "plain", _block())
    drv1, m1, fine = _regions(tmp_path / "fine", _block(core_elements_per_scale=16))
    core = drv1._post_cut_state()["core"]
    assert not core["whole"] and core["neck"] is not None
    assert 0 < core["mask"].sum() < len(core["mask"])
    twins = {n: g for n, g in fine.items() if "twin_of" in g}
    assert {n for n in fine if n not in twins} == set(plain)
    for name, g in plain.items():
        f = fine[name]
        for k in ("target", "r", "z", "radius", "class"):
            assert f[k] == g[k], (name, k)
    assert {g["twin_of"] for g in twins.values()} >= {"jet_base_cavity_tip", "neck_open",
                                                      "cavity_tip", "cavity_tip_cut"}
    o = next(c for c in drv1._post_cut_state()["segs"] if c["open"])
    P = np.column_stack([o["r"], o["z"]])
    for name, t in twins.items():
        g = plain[t["twin_of"]]
        assert name == "core_" + t["twin_of"] and t["target"] == 0.5 * g["target"], name
        assert t["class"] == "core" and t["core"] == J.core_spec(core, P, JET["wall_radius"])
    for name, g in plain.items():
        if name not in {t["twin_of"] for t in twins.values()}:
            inside = np.hypot(P[:, 0] - g["r"], P[:, 1] - g["z"]) <= g["radius"]
            assert not (inside & core["mask"]).any(), name
    # the spacing: a twin's on the core alone, its region's off it, the others' every node
    sel = drv1._twin_selection(m1["regions"], core)
    seg = {"open": True, "r": o["r"], "z": o["z"]}
    assert np.array_equal(sel(twins["core_neck_open"])(seg), core["mask"])
    assert np.array_equal(sel(fine["neck_open"])(seg), ~core["mask"])
    assert not sel(twins["core_neck_open"])({"open": False, "r": np.zeros(4)}).any()
    assert drv0._twin_selection(m0["regions"], core) is None
    drv0.close()
    drv1.close()


def test_the_rule_rebuilds_from_the_twins_and_the_focus_carries_the_cores_selection(tmp_path):
    """The twins' spacing is their regions', twice their targets: the rule asks for a rebuild and
    the focus it builds holds each twin at its target with the core's selection, each other region
    as without the key; the remesh record names each twin's region and its core nodes; the leg
    records the key and the ratio."""
    drv, m, fine = _regions(tmp_path, _block(core_elements_per_scale=16))
    built = []
    drv.pb.force_remesh = lambda: built.append(drv.pb.active_mesh_template().focus)
    drv.state.last_remesh_step = drv.state.refine_blocked_until = -10
    assert drv._maybe_remesh_features(m) is not False
    focus = built[-1]
    for g in m["regions"]:
        f = focus.region(g["name"])
        assert f is not None, g["name"]
        if "twin_of" in g:  # at its target, refined by at most max_refine_factor at once
            assert f.core == tuple(g["core"]), g["name"]
            bound = g["h_max"] / drv.s.max_refine_factor if math.isfinite(g["h_max"]) else 0.0
            assert f.size <= max(g["target"], bound) * (1 + 1e-12), g["name"]
        else:
            assert f.core is None
    rec = next(e for e in _read_events(tmp_path) if e["event"] == "remesh")
    asked = set(rec["asked_by"])
    assert any(a.startswith("core_") for a in asked), asked
    twins = [g for g in rec["regions"] if "twin_of" in g]
    assert twins and all(g["class"] == "core" and g["core_nodes"] > 0 for g in twins)
    limits = drv._leg_limits()["jetcore"]
    assert limits["core_elements_per_scale"] == 16 and limits["core_ratio"] == 0.5
    drv.close()
    plain, _, _ = _regions(tmp_path / "plain", _block())
    assert "core_elements_per_scale" not in plain._leg_limits()["jetcore"]
    plain.close()


def _read_events(outdir: Path) -> list[dict]:
    return [json.loads(x) for x in (outdir / "events.jsonl").read_text().splitlines()]


def test_the_guards_turn_on_the_core_asks_for_a_core_kink_at_q_times_its_limit(tmp_path):
    """The guard's limit set between the core's worst turn and twice it: without the key no core
    element asks for a refinement (no ``kink_open``), with it (``q`` 1/2) the core's worst does,
    a quarter of its chord; its elements above the floor (``post.h_floor`` 1e-7 here), which the
    rule can refine."""
    post = {**LIQUID["post"], "h_floor": 1e-7}

    def block(**jet):
        return ev.EventSettings.from_dict({**LIQUID, "post": post, "jetcore": {**JET, **jet}})

    drv = _events_driver(tmp_path / "read", JetProblem(), RECON_8, block(), max_steps=1)
    pc = drv._post_cut_state()
    e = pc["edges"]["open"]["edges"]
    mask = pc["core"]["mask"][np.clip(e["index"], 0, len(pc["core"]["mask"]) - 1)]
    turn = np.where(mask, e["turn"], -1.0)
    k = int(np.nanargmax(turn))
    t, chord = float(turn[k]), float(e["L"][k])
    drv.close()
    assert 0.0 < t < RECON_8.edge_turn_refine  # the core turns less than the guard's limit
    recon = ReconnectSettings(delta=1e-3, dt0=1e-9, courant=0.4, focus_courant=False,
                              elements_per_scale=8, edge_turn_refine=1.5 * t)
    kinks = {}
    for name, b in (("plain", block()), ("fine", block(core_elements_per_scale=16))):
        drv = _events_driver(tmp_path / name, JetProblem(), recon, b, max_steps=1)
        m = drv.measure()
        kinks[name] = {g["name"]: g for g in m["regions"] if g["name"].startswith("kink_")}
        drv.close()
    assert "kink_open" not in kinks["plain"] and "kink_open" in kinks["fine"]
    assert kinks["fine"]["kink_open"]["target"] == pytest.approx(0.25 * chord, rel=1e-9)
    assert "core_kink_open" not in kinks["fine"]  # the guard's own kink has no twin


def test_a_dumped_fresh_mesh_restarted_with_the_key_is_stepped_on(tmp_path, monkeypatch):
    """As ``test_events_driver``'s restart of a fresh mesh: a tracked neck below the fresh mesh's
    resolution stops at once; a continuation under the same block stops at once again, one that
    sets the key takes the mesh as stepped on and steps, the rule of its rebuilds meeting the new
    sizes, the leg naming the key."""
    from test_events_driver import HeadProblem

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    block = {"liquid": {"enabled": True, "eps_d": 0.02},
             "post": {"open_neck_stop": False, "neck_floor": 2e-6}, "jetcore": {"enabled": True}}
    thin = dict(r_n=lambda t: 0.02)
    out = _events_driver(tmp_path / "run", HeadProblem(**thin), recon,
                         events=ev.EventSettings.from_dict(block)).run(1e-9)
    stop = out["stop"]
    assert stop["reason"] == "liquid-neck-unresolved" and stop["fresh"] and out["steps"] == 0
    meta = json.loads(Path(stop["end_state"]).with_suffix(".json").read_text())
    assert meta["mesh_fresh"] is True
    monkeypatch.setitem(sys.modules, "faraday_singularity.surgery", SimpleNamespace(
        restore_cut=lambda problem, record: setattr(problem, "cut", record)))

    def again(name, blk):
        pb = HeadProblem(**thin)
        pb.t = meta["time"]
        drv = CollapseDriver(pb, _settings(t_end=1.0, dt_max=1.0, max_steps=3), tmp_path / name,
                             reconnect=recon, events=ev.EventSettings.from_dict(blk))
        drv.restore(meta)
        res = drv.run(1e-9)
        drv.close()
        return drv, res

    drv, res = again("same", block)
    assert res["stop"]["reason"] == "liquid-neck-unresolved" and res["steps"] == 0
    finer = json.loads(json.dumps(block))
    finer["jetcore"]["core_elements_per_scale"] = 16
    drv, res = again("finer", finer)
    assert drv.state.legs[-1]["changed"] == [KEY] and res["steps"] >= 1
