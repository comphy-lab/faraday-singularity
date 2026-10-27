"""The closures and the transfer check of the ring surgeries without a finite-element problem
(``ringsurgery._ring_closures``, ``ringsurgery._transfer_check``):

* each hard closure (R1 the volume gained, R2 the momentum the transfer carried, R3 the cutoff
  identity against the larger of its sides, ``R3_box`` the box's momentum the projection changed
  against ``P_ref``; Q1-Q3 and ``Q3_box`` for the removal) beyond its floored limit is a hard miss,
  within it none, and the floors are the multiples of the box's noise of ``RING_NOISE_FLOORS``;
* an identity whose sides cancel far below ``P_ref`` misses its own limit and passes only on its
  floor; a side missing or not finite is a miss; with no projection the identity is zero;
* the transfer: nodes without a value only on the corner set and at most those of it the geometry
  makes new, the closure holding nodes, as ``surgery.transfer_problems`` checks a liquid cut's.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pyoomph")

from faraday_singularity import ringsurgery as rs  # noqa: E402
from faraday_singularity.surgery import INTERFACE  # noqa: E402

V, U = 2.0, 3.0
SCALES = {"V_fill": V, "U": U, "P_ref": V * U, "KE_ref": 0.5 * V * U**2}


def budgets(changes=(), phi=None):
    """A ring cut whose budgets close exactly: the liquid gains the filled volume ``V``, the
    transfer carries the filled region's momentum and energy into the box, the projection changes
    nothing and its identity holds; ``changes`` (part, stage, key, change) open them."""
    fill = {"V": V, "Pz": 0.5, "KE": 0.25}
    poly = {"before": {"V_liquid": 10.0, "A": 1.0}, "after": {"V_liquid": 10.0 + V, "A": 1.3}}
    box = {"before": {"Pz": 1.0, "KE": 1.0}, "after_transfer": {"Pz": 1.5, "KE": 1.25},
           "after_projection": {"Pz": 1.5, "KE": 1.25}}
    phi = {"phi_lhs": 0.4, "phi_rhs": 0.4, "phi_error": 0.0, "phi_relative_error": 0.0} \
        if phi is None else phi
    parts = {"poly": poly, "box": box, "phi": phi}
    for part, stage, key, change in changes:
        if part == "phi":
            parts["phi"] = dict(parts["phi"], **{key: parts["phi"][key] + change})
        else:
            parts[part][stage][key] += change
    return fill, parts["poly"], parts["box"], parts["phi"], SCALES, {"A_cl": 0.5, "A_rem": 0.2}


def test_the_closures_of_a_ring_cut_that_closes_have_no_flag_and_four_hard_closures():
    assert rs.HARD_RING_CLOSURES == ("R1", "R2", "R3", "R3_box")
    assert rs.RING_BUDGET_LIMITS["R3_box"] == 0.05  # the liquid cut's F3
    c = rs._ring_closures(*budgets())
    assert c["flags"] == [] and c["hard"] == []
    assert set(c["limits"]) == {"R1", "R2", "R3", "R3_box", "R4", "R5", "R6"}
    assert c["R3_projected"] is True
    q = rs._ring_closures(*budgets(), prefix="Q")
    assert set(q["limits"]) == {"Q1", "Q2", "Q3", "Q3_box", "Q4", "Q5", "Q6"}
    assert "Q3_box" in q and "Q3_phi" in q


@pytest.mark.parametrize("prefix", ["R", "Q"])
@pytest.mark.parametrize("name,changes", [
    ("1", [("poly", "after", "V_liquid", -0.1)]),  # five hundredths of V lost
    ("2", [("box", "after_transfer", "Pz", 0.6), ("box", "after_projection", "Pz", 0.6)]),
    ("3", [("phi", None, "phi_rhs", 0.02)]),  # the identity misses by five hundredths
    ("3_box", [("box", "after_projection", "Pz", 0.6)])])  # a tenth of P_ref
def test_each_hard_closure_beyond_its_floored_limit_is_a_hard_miss(prefix, name, changes):
    """Each hard closure, opened by a few times its limit, is the one hard miss; under a noise
    whose floor lies above the miss it passes, under one whose floor lies below it it misses."""
    target = prefix + name
    c = rs._ring_closures(*budgets(changes), prefix=prefix)
    assert c["hard"] == [target], c
    value = c[target]
    assert value > c["limits"][target]
    noise_name, factor = rs.RING_NOISE_FLOORS["R" + name]
    over = rs._ring_closures(*budgets(changes), sigma={noise_name: 1.01 * value / factor},
                             prefix=prefix)
    assert target not in over["flags"]
    under = rs._ring_closures(*budgets(changes), sigma={noise_name: 0.99 * value / factor},
                              prefix=prefix)
    assert under["hard"] == [target]


def test_the_identity_is_measured_against_the_larger_of_its_sides_not_against_p_ref():
    """Sides far below ``P_ref`` that miss each other by four fifths: a miss of the identity's own
    limit, not hidden by the scale of the column's impulse; a floor above it passes it, and the
    relative error is the identity's own."""
    phi = {"phi_lhs": 1.5e-6, "phi_rhs": 7.8e-6, "phi_error": -6.3e-6,
           "phi_relative_error": -6.3e-6 / 7.8e-6}
    c = rs._ring_closures(*budgets(phi=phi))
    assert c["hard"] == ["R3"]
    assert c["limits"]["R3"] == pytest.approx(1e-2 * 7.8e-6)
    assert c["relative"]["R3"] == pytest.approx(6.3e-6 / 7.8e-6)
    floored = rs._ring_closures(*budgets(phi=phi), sigma={"Pz": 1e-5})
    assert floored["hard"] == [] and floored["limits"]["R3"] == pytest.approx(5e-5)


def test_an_identity_with_a_side_missing_or_not_finite_misses_and_none_without_projection():
    for phi in ({}, {"phi_lhs": math.nan, "phi_rhs": 0.4}, {"phi_lhs": 0.4, "phi_rhs": math.inf}):
        c = rs._ring_closures(*budgets(phi=phi), sigma={"V": 1.0, "Pz": 1.0, "KE": 1.0})
        assert "R3" in c["hard"], phi
    fill, poly, box, _, scales, areas = budgets()
    c = rs._ring_closures(fill, poly, box, None, scales, areas)
    assert c["R3"] == 0.0 and c["hard"] == [] and c["R3_projected"] is False


# --- the transfer check ---------------------------------------------------------------------------


def transfer(no_value=None, iface=4, bulk=10, axis=3):
    return {"unlocated": {INTERFACE: [iface, 500], "liquid": [bulk, 9000],
                          "liquid/axis": [axis, 40]},
            "no_value": {} if no_value is None else no_value}


def check(tr, on_closure=True, corners=None, closure_nodes=6):
    predicted = np.zeros(20, dtype=bool)
    predicted[:4] = True
    closure = np.zeros(20, dtype=bool)
    if on_closure:
        closure[:closure_nodes] = True
    return rs._transfer_check(tr, {}, predicted, closure if on_closure is not None else None,
                              np.ones(10, dtype=bool), np.ones(3, dtype=bool),
                              {"nodes": 1, "new": 0} if corners is None else corners)


def test_the_transfer_check_refuses_nodes_without_a_value_off_the_new_corners():
    corner = INTERFACE + "/axis"
    tr = transfer()
    assert check(tr) == []
    assert tr["closure_nodes"] == 6 and tr["corners"] == {"nodes": 1, "new": 0}
    assert any("without a value" in p for p in check(transfer({corner: [1, 1]})))
    assert check(transfer({corner: [1, 2]}), corners={"nodes": 2, "new": 1}) == []
    assert any("without a value" in p
               for p in check(transfer({"liquid/interface": [1, 400]}),
                              corners={"nodes": 2, "new": 1}))


def test_the_transfer_check_needs_nodes_on_the_closure():
    assert any("no interface node on the planned closure" in p
               for p in check(transfer(iface=0), closure_nodes=0))
    tr = transfer(iface=0, axis=0)  # the removal: no closure, none predicted unlocated
    problems = rs._transfer_check(tr, {}, np.zeros(20, dtype=bool), None,
                                  np.ones(10, dtype=bool), np.zeros(3, dtype=bool),
                                  {"nodes": 1, "new": 0})
    assert problems == [] and tr["closure_nodes"] is None


def test_the_corners_the_geometry_makes_new_are_those_off_every_old_axis_end():
    from types import SimpleNamespace

    class Node:
        def __init__(self, r, z):
            self._x = (r, z)

        def x(self, i):
            return self._x[i]

    class Mesh:
        def __init__(self, nodes):
            self._nodes = nodes

        def nnode(self):
            return len(self._nodes)

        def nodes(self):
            return list(self._nodes)

    pb = SimpleNamespace(get_mesh=lambda name: Mesh([Node(0.0, 1.0), Node(0.0, 0.5)]))
    assert rs._corners(pb, [np.array([0.0, 1.0 + 1e-9])], 1e-3) == {"nodes": 2, "new": 1}
    assert rs._corners(pb, [np.array([0.0, 1.0]), np.array([0.0, 0.5])], 1e-3) == {
        "nodes": 2, "new": 0}
    assert rs._corners(pb, [], 1e-3) == {"nodes": 2, "new": 2}
