"""The repair of a jet's tip folded along the axis, without a finite-element problem
(``axisrepair.axis_fold``; ``driver.CollapseDriver._axis_fold_of`` and ``_axis_repair`` on
stand-ins, ``axisrepair.axis_repair`` stood in for): r3 of the delta triple at e923ead stopped at
step 7338 on the downward jet's tip, the bubble's top end, its last element running 5.5e-6 up the
axis to the axis node, a vertex at r -1.6e-7, the element before it turning by 102.8 degrees.

* the fold: from an end on the axis, the nodes beside it within ``AXIS_FOLD_TOL`` of the spacing of
  the axis or across it; the first node off the axis kept; the new end where the line through it
  and the node before it meets the axis, at most its radius from its height; the sliver and its
  volume; none for a tip that meets the axis square, at either end; r3's numbers;
* the driver: a guard stop on the bubble before the piercing, with jetcore's recentred far kinks,
  at an element of the folded tail, is the repair's; nothing else is (the open chain, the ring
  stage, other triggers, an element away from the tail, no recentred kinks);
* the repair made: an event ``axis-repair``, its record, the run's stops read on the new state; a
  repair whose hard closures miss rolled back to the snapshot before it and the resumable stop
  ``axis-repair-unclosed``; a fold again at the same end within ``AXIS_REPAIR_RECUR_STEPS`` steps
  left as the stop it is, with ``axis_fold_recurs``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import axisrepair as ar  # noqa: E402
from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity.driver import CollapseDriver  # noqa: E402
from faraday_singularity.reconnect import revolved_volume  # noqa: E402

DELTA = 2.5e-4
H = 2.75e-6  # r3's spacing at the tip


def tip(n=40, fold=True, end_z=4.6770486, z0=4.6770430):
    """A flat tip meeting the axis at ``z0`` (spacing H), as the bubble's last end (r falling to
    zero); with ``fold`` its last two nodes run up the axis, the middle one at r < 0, as r3's."""
    r = np.arange(n, 0, -1) * H
    z = z0 + 1e-7 * (r / H) ** 0.5  # gently rising away from the axis
    P = np.column_stack([r, z])
    if fold:
        tail = np.array([[2.76e-7, z0 + 1e-7], [-1.61e-7, z0 + 2.8e-6], [0.0, end_z]])
    else:
        tail = np.array([[0.0, z0]])
    return np.vstack([P, tail])


def test_the_fold_of_a_tip_along_the_axis():
    P = tip()
    f = ar.axis_fold(P, "last")
    n = len(P)
    assert f["removed"] == [n - 1, n - 2, n - 3] and f["kept"] == n - 4 and f["k"] == 3
    assert f["h"] == pytest.approx(H, rel=0.2) and f["tol"] == pytest.approx(0.25 * f["h"])
    assert f["r_min"] == pytest.approx(-1.61e-7)
    a, b = P[n - 4], P[n - 5]
    z_line = a[1] + (a[1] - b[1]) * (0.0 - a[0]) / (a[0] - b[0])
    assert f["new_end"][0] == 0.0 and f["new_end"][1] == pytest.approx(z_line)
    assert np.allclose(f["points"][:-1], P[:n - 3]) and np.allclose(f["points"][-1], f["new_end"])
    assert np.allclose(f["old_end"], P[-1])
    assert f["V_sliver"] == pytest.approx(abs(revolved_volume(f["sliver"])))
    # the tail ends where it first reaches the axis, between the node at r 2.76e-7 and the one at
    # r -1.61e-7: the sliver and the old chain cut there are valid sections
    import shapely

    c = f["crossing"]
    assert c[0] == 0.0 and P[n - 3][1] < c[1] < P[n - 2][1]
    assert shapely.Polygon(f["sliver"]).is_valid and np.allclose(f["sliver"][-1], c)
    assert np.allclose(f["old_points"][:-1], P[:n - 2]) and np.allclose(f["old_points"][-1], c)
    assert 0.0 < f["V_sliver"] < ar.AXIS_REPAIR_BOUND * DELTA**3  # r3's: about 1e-6 delta^3
    # the first end, mirrored
    Q = P[::-1].copy()
    g = ar.axis_fold(Q, "first")
    assert g["removed"] == [0, 1, 2] and g["kept"] == 3
    assert np.allclose(g["points"][1:], Q[3:]) and np.allclose(g["points"][0], g["new_end"])
    assert g["V_sliver"] == pytest.approx(f["V_sliver"])
    # a tip that meets the axis square: no fold, at either end
    S = tip(fold=False)
    assert ar.axis_fold(S, "last") is None and ar.axis_fold(S[::-1], "first") is None
    assert ar.axis_fold(P, "first") is None  # the other end of this chain meets nothing
    # a steep surface: the new end at most the kept node's radius from its height
    steep = P.copy()
    steep[n - 5] = steep[n - 4] + np.array([1e-7, -5e-5])
    h = ar.axis_fold(steep, "last")
    assert abs(h["new_end"][1] - steep[n - 4][1]) <= steep[n - 4][0] + 1e-12


def _segs(P):
    open_chain = np.column_stack([np.linspace(0.0, 1e-3, 21), np.full(21, 4.7)])
    return [{"open": False, "end_types": ("axis", "axis"), "r": np.r_[0.0, P[:, 0]],
             "z": np.r_[4.6, P[:, 1]], "u_r": np.zeros(len(P) + 1),
             "u_z": np.full(len(P) + 1, -1.08e4)},
            {"open": True, "end_types": ("axis", "fixed"), "r": open_chain[:, 0],
             "z": open_chain[:, 1], "u_r": np.zeros(21), "u_z": np.zeros(21)}]


class Pb:
    def __init__(self, P):
        self.P = P
        self.snapshots, self.restored = [], []
        self.remesh_count = 0

    def interface_segments(self):
        return _segs(self.P)

    def _snapshot_state(self):
        self.snapshots.append("snap")
        return "snap"

    def _restore_state(self, s):
        self.restored.append(s)
        self.remesh_count += 1

    def invalidate_cached_mesh_data(self):
        pass


def _fake(tmp_path, P, *, jet="recentre", ring=None, step=7338):
    events = ev.EventSettings.from_dict({"groove": {"enabled": True, "sigma_V": 1.13e-13,
                                                    "sigma_Pz": 3.24e-10, "sigma_KE": 2.02e-7}})
    fake = SimpleNamespace(
        pb=Pb(P), events=events, reconnect=SimpleNamespace(delta=DELTA), outdir=tmp_path,
        state=SimpleNamespace(step=step, last_remesh_step=None),
        _jetcore=lambda: (SimpleNamespace(far_kinks=jet) if jet is not None else None),
        _fresh_mesh=lambda: False, _ring=lambda: ring, _t=lambda: 10.9547, _taus=lambda t: {"tau": 7.6e-7, "tau_p": 6.2e-7},
        measure=lambda: {"m": "after"}, _keep_landings=lambda m: None, _snapshot=lambda m, r: None,
        _stop=lambda m: None, _post_cut_cache=None, _axis_repairs=[], logged=[])
    fake._event = lambda name, m, **kw: fake.logged.append((name, kw))
    fake._axis_fold_of = lambda st: CollapseDriver._axis_fold_of(fake, st)
    return fake


def _guard(P, **kw):
    n = len(P) + 1  # the chain read with its first end prepended (_segs)
    out = {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "bubble",
           "turn": 102.8, "limit": 90.0, "fresh": True, "r": 2.05e-6, "z": 4.6770415,
           "L": 4.36e-6, "index": n - 4}
    out.update(kw)
    return out


def test_the_driver_meets_a_guard_stop_on_the_folded_tail_and_nothing_else(tmp_path):
    P = tip()
    fake = _fake(tmp_path, P)
    got = CollapseDriver._axis_fold_of(fake, _guard(P))
    assert got is not None and got[0] == "last" and got[1]["k"] == 3
    for trig in ("edge-midside", "feature-below-floor"):
        assert CollapseDriver._axis_fold_of(fake, _guard(P, trigger=trig)) is not None
    for stop in (_guard(P, chain="open"), _guard(P, chain="ring"), _guard(P, index=10),
                 _guard(P, trigger="curved-jacobian"), dict(_guard(P), reason="event-request"),
                 _guard(P, index=None), None):
        assert CollapseDriver._axis_fold_of(fake, stop) is None, stop
    assert CollapseDriver._axis_fold_of(_fake(tmp_path, P, jet=None), _guard(P)) is None
    assert CollapseDriver._axis_fold_of(_fake(tmp_path, P, jet="grow"), _guard(P)) is None
    assert CollapseDriver._axis_fold_of(_fake(tmp_path, P, ring={"removed": None}),
                                        _guard(P)) is None
    assert CollapseDriver._axis_fold_of(_fake(tmp_path, tip(fold=False)), _guard(P)) is None


def test_the_repair_made_rolled_back_and_not_repeated(tmp_path, monkeypatch):
    P = tip()
    calls = []

    def made(pb, events, reconnect, outdir, chain, fold, *, velocity, base_check=None,
             log=print):
        calls.append((chain, fold["end"], velocity.shape))
        return {"kind": "axis-repair", "outcome": "made",
                "fold": {"V_sliver": fold["V_sliver"],
                         "V_sliver_over_delta3": fold["V_sliver"] / DELTA**3,
                         "new_end": fold["new_end"].tolist()},
                "closures": {"hard": [], "flags": []}, "base_guard": {"ok": True}}
    monkeypatch.setattr(ar, "axis_repair", made)
    fake = _fake(tmp_path / "made", P)
    (tmp_path / "made").mkdir()
    m, stop = CollapseDriver._axis_repair(fake, {"m": "before"}, _guard(P))
    assert m == {"m": "after"} and stop is None and calls == [("bubble", "last", (len(P) + 1, 2))]
    assert fake.state.last_remesh_step == 7338 and fake._axis_repairs[0]["end"] == "last"
    assert [x[0] for x in fake.logged] == ["axis-repair"]
    rec = json.loads((tmp_path / "made" / "events" / "axis_repair_001.json").read_text())
    assert rec["n"] == 1 and rec["step"] == 7338 and rec["guard"]["index"] == len(P) - 3
    # the fold again at the same end within the steps of a recurrence: left as the stop it is
    fake.state.step = 7338 + ar.AXIS_REPAIR_RECUR_STEPS - 1
    m, stop = CollapseDriver._axis_repair(fake, {"m": "x"}, _guard(P))
    assert stop["reason"] == "interface-unresolved" and stop["axis_fold_recurs"]["repairs"] == 1
    assert len(calls) == 1
    fake.state.step = 7338 + ar.AXIS_REPAIR_RECUR_STEPS
    CollapseDriver._axis_repair(fake, {"m": "x"}, _guard(P))
    assert len(calls) == 2
    # hard closures that miss: rolled back, a resumable stop
    from faraday_singularity.surgery import BudgetUnclosed

    def unclosed(*a, **kw):
        e = BudgetUnclosed("miss")
        e.record = {"kind": "axis-repair", "outcome": "unclosed",
                    "closures": {"hard": ["A_V"], "flags": ["A_V"]}}
        raise e
    monkeypatch.setattr(ar, "axis_repair", unclosed)
    fake2 = _fake(tmp_path / "miss", P)
    (tmp_path / "miss").mkdir()
    m, stop = CollapseDriver._axis_repair(fake2, {"m": "before"}, _guard(P))
    assert stop["reason"] == "axis-repair-unclosed" and stop["hard"] == ["A_V"]
    assert fake2.pb.restored == ["snap"] and fake2._axis_repairs == [] and fake2.logged == []
    assert stop["record"] == "events/axis_repair_001.json" and m == {"m": "after"}
    assert math.isfinite(stop["tau"])


def test_a_curved_element_beside_the_bubbles_tip_is_put_straight_and_nothing_else_is():
    """``CollapseDriver._tip_straighten``: a fresh mesh's ``curved-jacobian`` at an element whose
    nearest surface is the bubble's chain, before the piercing, with jetcore's recentred far
    kinks: the surface's curved edges about it put straight (``straighten_interface_edges``) when
    the volume between their curves and chords is within ``AXIS_REPAIR_BOUND delta^3`` and the
    bound read again is at least ``jacobian_min`` (event ``tip-straighten``, no stop); the stop
    stays with the reading otherwise, and as it is near the open chain, in the ring stage, or
    without the recentred far kinks (r3, fix5 leg 1 at aa48432, step 7339: 0.054 against 0.2)."""
    P = tip(fold=False)
    bad = {"reason": "interface-unresolved", "trigger": "curved-jacobian",
           "jacobian_bound": 0.0537, "limit": 0.2, "r": 8.04e-6, "z": 4.6770412,
           "on_interface": True, "elements_below": 1, "fresh": True}

    def fake(*, dV=2.05e-16, after=0.805, ring=None, jet="recentre"):
        calls = []
        pb = Pb(P)
        pb.straighten_interface_edges = lambda r, z, radius, apply=True: (
            calls.append((r, z, radius, apply)) or
            {"edges": [{"r": 8.4e-6, "z": 4.6770414, "offset": 0.232, "chord": 4.86e-6,
                        "dV": dV}], "dV": dV, "dV_abs": abs(dV), "read": 3})
        pb.curved_jacobian_bound = lambda limit: {"min": after}
        f = SimpleNamespace(pb=pb, reconnect=SimpleNamespace(delta=DELTA, jacobian_min=0.2),
                            state=SimpleNamespace(step=7339), _tip_straightens=[], logged=[],
                            _post_cut_cache="x", _fresh_mesh=lambda: False,
                            _ring=lambda: ring, _t=lambda: 10.95,
                            _jetcore=lambda: (SimpleNamespace(far_kinks=jet) if jet else None))
        f._event = lambda name, m, **kw: f.logged.append((name, kw))
        return f, calls

    f, calls = fake()
    assert CollapseDriver._tip_straighten(f, dict(bad)) is None
    assert [c[3] for c in calls] == [False, True] and calls[0][:2] == (8.04e-6, 4.6770412)
    assert f._tip_straightens[0]["jacobian_bound_after"] == 0.805 and f._post_cut_cache is None
    assert f.logged[0][0] == "tip-straighten" and f.logged[0][1]["bound_dV"] == pytest.approx(
        ar.AXIS_REPAIR_BOUND * DELTA**3)
    # a volume beyond the bound: read only, the stop with the reading
    f, calls = fake(dV=-1e-14)
    out = CollapseDriver._tip_straighten(f, dict(bad))
    assert out["reason"] == "interface-unresolved" and out["tip_straighten"]["dV_abs"] == 1e-14
    assert [c[3] for c in calls] == [False] and not f._tip_straightens
    # put straight, but the bound still below: the stop with the reading
    f, calls = fake(after=0.1)
    out = CollapseDriver._tip_straighten(f, dict(bad))
    assert out["tip_straighten"]["jacobian_bound_after"] == 0.1 and not f.logged
    assert out["tip_straighten"]["rolled_back"] and f.pb.restored == ["snap"]
    assert f._stepped_rc == f.pb.remesh_count
    # near the open chain, in the ring stage, without the recentred kinks, off the surface: as it is
    for kw, stop in (({}, dict(bad, r=5e-4, z=4.7)), ({"ring": {"removed": None}}, dict(bad)),
                     ({"jet": "grow"}, dict(bad)), ({"jet": None}, dict(bad)),
                     ({}, dict(bad, on_interface=False))):
        f, calls = fake(**kw)
        assert CollapseDriver._tip_straighten(f, stop) == stop and not calls, (kw, stop)


    # The bubble may be nearest while the repair radius also reaches the upward chain.
    f, calls = fake()
    segments = f.pb.interface_segments()
    next(c for c in segments if c["open"])["z"][0] = bad["z"]
    f.pb.interface_segments = lambda: segments
    assert CollapseDriver._tip_straighten(f, dict(bad)) == bad and not calls


def test_the_axis_repair_volume_bound_is_checked_before_a_rebuild(tmp_path, monkeypatch):
    from faraday_singularity import surgery
    from faraday_singularity.reconnect import ReconnectSettings
    P = tip()
    f = ar.axis_fold(P, "last")
    f["V_sliver"] = 2 * ar.AXIS_REPAIR_BOUND * DELTA**3
    pb = Pb(P)
    pb.get_current_time = lambda **kw: 0.0
    fake = _fake(tmp_path, P)
    with pytest.raises(surgery.BudgetUnclosed) as error:
        ar.axis_repair(pb, fake.events, ReconnectSettings(delta=DELTA), tmp_path,
                       "bubble", f, velocity=np.zeros((len(P)+1, 2)))
    assert error.value.record["closures"]["hard"] == ["A_V"]
    assert pb.remesh_count == 0

def test_a_fold_whose_walk_exhausts_the_search_is_not_repaired():
    P = tip()
    P[-18:, 0] = 0.0
    assert ar.axis_fold(P, "last") is None


def test_the_slivers_own_quadrature_resolves_its_volume_at_small_scales(monkeypatch):
    from faraday_singularity import surgery
    def constant(pb, r, z, names):
        return {"velocity_y": np.full(len(r), -7.0)}, np.ones(len(r), dtype=bool)
    monkeypatch.setattr(surgery, "sample_fields", constant)
    for scale in (1.0, 1e-3, 1e-6):
        polygon = np.array([[0, 0], [2, 0], [1, 1], [0, 2]], dtype=float) * scale
        polygon[:, 1] += 4.67
        q = ar.sliver_quadrature(None, polygon)
        assert q["points"] >= 7 and q["found"] == q["points"]
        assert q["V_quad_over_V_sliver"] == pytest.approx(1.0, rel=1e-9)
        assert q["Pz"] == pytest.approx(-7 * q["V_sliver"], rel=1e-9)


def test_an_over_bound_axis_repair_does_not_snapshot_or_restore_the_driver(tmp_path):
    P = tip()
    fake = _fake(tmp_path, P)
    fold = ar.axis_fold(P, "last")
    fold["V_sliver"] = 2 * ar.AXIS_REPAIR_BOUND * DELTA**3
    fake._axis_fold_of = lambda stop: ("last", fold)
    fake.pb.get_current_time = lambda **kw: 10.9547
    before = {"m": "input"}
    after, stop = CollapseDriver._axis_repair(fake, before, _guard(P))
    assert after is before and stop["reason"] == "axis-repair-unclosed"
    assert fake.pb.snapshots == [] and fake.pb.restored == [] and fake.pb.remesh_count == 0
