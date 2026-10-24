"""The upward jet's cut-born head folding onto the core, without a finite-element problem
(``events.far_head_request``'s ``head_fold``; the driver's base check, ``driver.CollapseDriver.
_head_fold_check``, on stand-ins): a head sitting just above the jet's base, its rim closing on the
cavity's wall beside the base, as r3 of the delta triple met it at tau/delta^2 4.60 (the root neck
1.06 neck radii from the base's apex, the fold between the rim and the wall).

* a closing groove whose one node lies on the head and whose other lies in the core (the wall, or
  the stem below the neck) is the cut of the head at its root neck, ``head_fold`` naming the two
  nodes; the head's own nodes off the core, its neck at the core's edge;
* none for a gas neck in the core, for a fold with both nodes in the core or both off the head, for
  a fold whose core node lies on the head (the head reaching into the core), without a core, or on
  the bubble (every node far there, the cut a far-head cut as before, no ``head_fold``);
* a fold with both nodes on the head is the far-head cut as before, no ``head_fold``;
* the driver's check of the base: the cut passed ``base_check``, which reads the base before and
  after; a base unchanged makes the cut, a base changed beyond ``HEAD_FOLD_BASE_TOL`` is the hard
  closure ``F_base`` and the resumable stop ``event-unclosed``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from test_curled_head import _arc, _hermite, sarc_at  # noqa: E402
from test_events_stops import R_C, Z_APEX, _nodes  # noqa: E402

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402

DELTA = 2.5e-4
LIQUID = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                     "post": {"open_neck_stop": False, "neck_floor": 2e-6}})


def folded_head_chain(delta=DELTA, m=3000):
    """The open chain, in units of delta with the base's apex at ``z = 0``: the head's cap from its
    tip on the axis (``z`` 1.47) out to its rim (``r`` 0.70 at ``z`` 0.77), the rim's underside down
    to 0.51 and up to a bump at ``r`` 0.34, down to the root neck (``r`` 0.23 at ``z`` 0.23), the
    stem down to the base's apex (``r`` 0.33 at 0, through the groove's bottom), the cavity's wall
    up and out past the rim (``r`` 0.78 at the rim's height) to ``r`` 3, then to the
    container's wall. Returns the chain, the root neck's node, the base's apex node, a node of the
    rim and the wall's node nearest it."""
    th = np.linspace(0.0, 0.5 * math.pi, m)
    cap = np.column_stack([0.70 * np.sin(th), 0.77 + 0.70 * np.cos(th)])  # tip at 1.47
    under = _arc((0.55, 0.77), 0.15, 0.0, -math.pi / 2, m)  # (0.70, 0.77) down to (0.55, 0.62)
    under2 = np.column_stack([np.linspace(0.55, 0.42, m),
                              0.62 - 0.11 * np.sin(np.linspace(0.0, math.pi, m))])
    bump = _arc((0.38, 0.62), 0.04, 0.0, math.pi, m)  # over the bump to (0.34, 0.62)
    stem = _hermite((0.34, 0.62), (0.23, 0.23), m)  # down to the root neck
    stem2 = _hermite((0.23, 0.23), (0.33, 0.0), m)
    groove = _arc((0.33, 0.06), 0.06, -math.pi / 2, 0.0, m)  # the groove's bottom, out to the wall
    rw = np.linspace(0.39, 3.0, m)
    wall = np.column_stack([rw, 0.06 + (rw - 0.39) * (0.77 - 0.06) / (0.78 - 0.39)])
    pieces = [cap, under[1:], under2[1:], bump[1:], stem[1:], stem2[1:], groove[1:], wall[1:]]
    local = np.vstack(pieces)
    z_b = Z_APEX
    feat = np.column_stack([delta * local[:, 0], z_b + delta * local[:, 1]])
    far = np.column_stack([np.linspace(feat[-1, 0], R_C, 400)[1:],
                           np.linspace(feat[-1, 1], feat[-1, 1] + 0.05, 400)[1:]])
    dense = np.vstack([feat, far])
    rev = dense[::-1]
    sarc, nodes = _nodes(rev, sarc_at(rev, len(far)), delta, delta / 128, 1e-3)
    P = np.column_stack([np.interp(nodes, sarc, rev[:, 0]), np.interp(nodes, sarc, rev[:, 1])])
    P = P[::-1].copy()
    P[0, 0] = 0.0

    def node(r, z):
        return int(np.argmin(np.hypot(P[:, 0] - delta * r, P[:, 1] - (z_b + delta * z))))

    near = np.flatnonzero(np.abs(P[:, 1] - (z_b + 0.23 * delta)) < 0.05 * delta)
    near = near[P[near, 0] < 0.26 * delta]
    i_n = int(near[np.argmin(P[near, 0])])
    i_a = int(np.argmin(np.where(np.arange(len(P)) > i_n, P[:, 1], np.inf)))
    rim = node(0.70, 0.74)
    wall_nodes = np.arange(i_a, len(P))
    wall = int(wall_nodes[np.argmin(np.hypot(P[wall_nodes, 0] - P[rim, 0],
                                             P[wall_nodes, 1] - P[rim, 1]))])
    return rc.Chain(P, ("axis", "fixed")), i_n, i_a, rim, wall


@pytest.fixture(scope="module")
def state():
    chain, i_n, i_a, rim, wall = folded_head_chain()
    necks = ev.liquid_necks([("open", chain)], LIQUID.liquid)
    head = next(x for x in necks if x["status"] == "head" and x["head"][0] == 0)
    n = len(chain.points)
    core = np.zeros(n, dtype=bool)
    core[head["index"]:] = True  # the core from the neck on: the stem, the base, the walls
    return {"chain": chain, "i_n": i_n, "i_a": i_a, "rim": rim, "wall": wall, "necks": necks,
            "head": head, "core": core, "n": n}


def groove(a, b, chain="open"):
    return {"reason": ev.EVENT_REQUEST, "kind": "groove", "class": "A-groove",
            "handler": ev.GROOVE_HANDLER, "chain": chain, "contact": "fold", "side": "gas",
            "fold_pair": [int(a), int(b)], "jet": "cavity_tip"}


def test_the_synthetic_head_sits_on_the_base_with_its_rim_at_the_wall(state):
    from faraday_singularity.selfcontact import check_simple_interface

    P = np.asarray(state["chain"].points)
    check_simple_interface(P)
    h = state["head"]
    assert abs(h["index"] - state["i_n"]) <= 2 and h["head"] == [0, h["index"]]
    assert h["r_n"] == pytest.approx(0.23 * DELTA, rel=0.03)
    assert h["s"] == pytest.approx(0.70 * DELTA, rel=0.02)
    d_base = math.hypot(*(P[h["index"]] - P[state["i_a"]]))
    assert d_base < 1.5 * h["r_n"]  # the root neck about one neck radius from the base's apex
    assert state["rim"] < h["index"] < state["i_a"] < state["wall"]
    gap = math.hypot(*(P[state["rim"]] - P[state["wall"]]))
    assert gap < 0.15 * DELTA


def test_a_fold_of_the_head_onto_the_core_is_the_cut_of_the_head(state):
    h, core = state["head"], state["core"]
    got = ev.far_head_request(groove(state["rim"], state["wall"]), state["necks"], core)
    assert (got["kind"], got["class"], got["handler"]) == ("far-head", "F-head", "liquid-cut")
    assert got["index"] == h["index"] and got["head"] == h["head"]
    assert got["head_fold"] == {"head_node": state["rim"], "core_node": state["wall"]}
    assert got["far_event"]["fold_pair"] == [state["rim"], state["wall"]]
    # the pair either way round, and onto the stem below the neck
    assert ev.far_head_request(groove(state["wall"], state["rim"]), state["necks"],
                               core)["head_fold"]["core_node"] == state["wall"]
    stem = h["index"] + 10
    assert core[stem]
    assert ev.far_head_request(groove(state["rim"], stem), state["necks"],
                               core)["head_fold"] == {"head_node": state["rim"],
                                                      "core_node": stem}


def test_no_cut_where_the_head_reaches_into_the_core_or_the_event_is_another(state):
    h, core, n = state["head"], state["core"], state["n"]
    rim, wall = state["rim"], state["wall"]
    # the head's own nodes in the core: no cut
    reach = core.copy()
    reach[h["index"] - 20] = True
    assert ev.far_head_request(groove(rim, wall), state["necks"], reach) is None
    # the rim node itself in the core: both nodes in the core
    both = core.copy()
    both[rim] = True
    assert ev.far_head_request(groove(rim, wall), state["necks"], both) is None
    # a gas neck in the core, on the wall
    gas = {"reason": ev.EVENT_REQUEST, "kind": "gas-neck", "class": "G",
           "handler": ev.HANDLERS["G"], "chain": "open", "index": wall}
    assert ev.far_head_request(gas, state["necks"], core) is None
    # both nodes off the head, one in the core: the wall far out and the wall at the base
    far_wall = n - 5
    assert ev.far_head_request(groove(far_wall, wall), state["necks"], core) is None
    # no core
    assert ev.far_head_request(groove(rim, wall), state["necks"], None) is None


def test_a_fold_within_the_head_is_the_far_head_cut_as_before(state):
    h, core = state["head"], state["core"]
    got = ev.far_head_request(groove(state["rim"], state["rim"] + 60), state["necks"], core)
    assert got["index"] == h["index"] and "head_fold" not in got


def test_on_the_bubble_a_fold_onto_the_head_is_the_far_head_cut_as_before():
    from test_curled_head import _neck, bubble_with_curled_head

    bub, j_root = bubble_with_curled_head()
    y, necks = _neck(bub, "bubble", j_root)
    k = y["head"][0] + 40
    got = ev.far_head_request(groove(k, y["head"][0] - 50, chain="bubble"), necks)
    assert got["index"] == y["index"] and "head_fold" not in got


# --- the driver's check of the base ---------------------------------------------------------------


def _guard_run(tmp_path, monkeypatch, *, shift):
    """A stand-in driver after a cut makes the cut of a head-fold request (the stand-in's liquid cut
    calls ``base_check`` after its "remesh", which moves the base of ``CutProblem``'s open chain by
    ``shift`` relative), returning the driver's result of meeting it."""
    from test_bookkeeping import CutProblem
    from test_events_plumbing import _events_driver
    from test_events_run import fake_surgery

    from faraday_singularity import surgery
    from faraday_singularity.reconnect import ReconnectSettings

    cut = fake_surgery()
    seen = {}
    P = np.asarray(folded_head_chain()[0].points)

    class Moved(CutProblem):
        """The open chain the folded head's, at an axial speed 1000 everywhere; ``scale`` its radii
        (the base moved by the cut's rebuild)."""
        scale = 1.0

        def interface_segments(self):
            segs = super().interface_segments()
            r, z = P[:, 0] * self.scale, P[:, 1].copy()
            segs[1] = self._chain("open", r, z, ("axis", "fixed"))
            return segs

        def axial_velocity_at(self, points):
            return [1000.0] * len(points), True

    pb = Moved()

    def liquid_cut(problem, events, reconnect, outdir, request, *, base_check=None, **kw):
        seen["base_check"] = base_check is not None
        problem.scale = 1.0 + shift  # the cut's rebuild moves the base, or not
        if base_check is not None:
            guard = base_check()
            seen["guard"] = guard
            if not guard["ok"]:
                k = kw["k"]
                precut = str(kw["driver"].keep_state(
                    Path(outdir) / "states" / f"precut_e{k:03d}.dump",
                    cut_stage=ev.PRECUT_EVENT, event=dict(request, k=k)))
                err = surgery.BudgetUnclosed("the cut of the head changed the base", precut)
                err.record = {"outcome": "unclosed", "failed_at": "base",
                              "closures": {"hard": ["F_base"]}, "base_guard": guard}
                raise err
        return cut.liquid_cut(problem, events, reconnect, outdir, request, **kw)

    monkeypatch.setattr(surgery, "liquid_cut", liquid_cut)
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    block = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                        "post": {"open_neck_stop": False, "neck_floor": 2e-6},
                                        "jetcore": {"enabled": True},
                                        "pinch": {"far_head": True}})
    drv = _events_driver(tmp_path, pb, recon, block, state_dt=0.0, state_keep=10)
    m = drv.measure()
    assert m["r_j"] > 0.0 and m["Q_j"] > 0.0, m  # the stand-in's jet stands
    named = drv._named_post_cut_chains()
    request = {"reason": ev.EVENT_REQUEST, "kind": "far-head", "class": "F-head",
               "handler": "liquid-cut", "chain": "open", "index": 3, "r_n": 1e-4, "z_n": 4.7,
               "s": 3e-4, "side": "before", "direction": 1.0, "status": "head", "head": [0, 3],
               "z_tip": 4.71, "head_volume": 1e-12, "far_event": groove(1, 40),
               "head_fold": {"head_node": 1, "core_node": 40}}
    assert named
    _, stop = drv._liquid_cut_event(m, request)
    drv.close()
    return stop, seen, m


def test_the_cut_of_a_folded_head_passes_the_base_check_and_is_made(tmp_path, monkeypatch):
    stop, seen, m = _guard_run(tmp_path, monkeypatch, shift=0.0)
    assert stop is None and seen["base_check"] is True
    g = seen["guard"]
    assert g["ok"] is True and g["tol"] == pytest.approx(1e-3)
    assert set(g["change"]) == {"r_j", "z_b", "V_j", "Q_j"}


def test_a_cut_that_moves_the_base_is_the_hard_closure_f_base_and_a_stop(tmp_path, monkeypatch):
    stop, seen, m = _guard_run(tmp_path, monkeypatch, shift=0.01)
    g = seen["guard"]
    assert g["ok"] is False and abs(g["change"]["r_j"]) > 1e-3
    assert stop["reason"] == ev.EVENT_UNCLOSED and stop["hard"] == ["F_base"]
    assert Path(stop["precut"]).name == "precut_e001.dump"
