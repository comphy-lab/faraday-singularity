"""The closures of a groove's fill without a finite-element problem (``groovesurgery.
merge_impulse`` on ``ringsurgery._ring_closures`` with prefix ``G``):

* ``G3_box`` within its limit: the closures unchanged, no ``G3_merge``;
* ``G3_box`` beyond its limit (the disc's momentum moved by the projection, the impulse of the merge
  the fill makes) with the cutoff identity G3 holding and resolving it (its larger side at least
  ``MERGE_RESOLVED`` of ``G3_box`` and at the momentum floor or above): a flag, not hard;
* the identity not resolving it (its sides below half the change, or below the floor), missing
  its own limit, or a side not finite: ``G3_box`` stays hard, as do the other hard closures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

from test_ring_closures import budgets  # noqa: E402

from faraday_singularity import groovesurgery as gs  # noqa: E402
from faraday_singularity import ringsurgery as rs  # noqa: E402


def closures(changes=(), phi=None, sigma=None):
    return rs._ring_closures(*budgets(changes, phi), sigma=sigma, prefix="G")


def test_a_fill_whose_box_closes_is_unchanged():
    c = closures()
    assert gs.merge_impulse(c) is c and "G3_merge" not in c
    assert c["hard"] == []


def moved(dPz, lhs, rhs=None):
    """The projection moves ``dPz`` out of the disc (P_ref 6, the limit 0.3); the identity's sides
    ``lhs``, ``rhs`` (``lhs`` by default, holding exactly)."""
    rhs = lhs if rhs is None else rhs
    phi = {"phi_lhs": lhs, "phi_rhs": rhs, "phi_error": lhs - rhs,
           "phi_relative_error": (lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)}
    return [("box", "after_projection", "Pz", dPz)], phi


def test_the_impulse_of_the_merge_resolved_by_the_identity_is_a_flag():
    """r3's event 1 in units of its P_ref: 0.127 moved out of the disc, the identity's sides 0.116
    and 0.116 (a share 0.91), holding to 1.1e-3."""
    P_ref = budgets()[4]["P_ref"]
    changes, phi = moved(0.127 * P_ref, 0.1158 * P_ref, 0.1157 * P_ref)
    c = closures(changes, phi)
    assert c["hard"] == ["G3_box"] and "G3" not in c["flags"]
    out = gs.merge_impulse(c, floor_Pz=0.0)
    assert out["hard"] == [] and "G3_box" in out["flags"]
    m = out["G3_merge"]
    assert m["resolved"] is True and m["share"] == pytest.approx(0.1158 / 0.127)
    assert m["side"] == pytest.approx(0.1158 * P_ref) and m["floor"] == 0.0
    assert out["G3_box"] == c["G3_box"] and out["limits"] == c["limits"]
    # the floor of the closures, five sigma_Pz, below the side: still resolved
    assert gs.merge_impulse(c, floor_Pz=0.1 * P_ref)["hard"] == []


@pytest.mark.parametrize("case", ["share", "floor", "identity", "nan"])
def test_an_impulse_the_identity_does_not_resolve_stays_hard(case):
    P_ref = budgets()[4]["P_ref"]
    floor = 0.0
    if case == "share":  # the sides below half the change: most of it unseen by the identity
        changes, phi = moved(0.127 * P_ref, 0.05 * P_ref)
    elif case == "floor":  # sides at the noise of the box
        changes, phi = moved(0.127 * P_ref, 0.1 * P_ref)
        floor = 0.2 * P_ref
    elif case == "identity":  # the identity misses its own limit
        changes, phi = moved(0.127 * P_ref, 0.1 * P_ref, 0.12 * P_ref)
    else:
        changes, phi = moved(0.127 * P_ref, math.nan, 0.1 * P_ref)
    c = closures(changes, phi)
    out = gs.merge_impulse(c, floor_Pz=floor)
    assert "G3_box" in out["hard"], case
    assert out["G3_merge"]["resolved"] is False
    if case in ("identity", "nan"):
        assert "G3" in out["hard"]


def test_the_other_hard_closures_stay_hard_beside_a_resolved_impulse():
    P_ref = budgets()[4]["P_ref"]
    changes, phi = moved(0.127 * P_ref, 0.12 * P_ref)
    changes = changes + [("poly", "after", "V_liquid", -0.1)]  # the volume misses too
    out = gs.merge_impulse(closures(changes, phi))
    assert out["hard"] == ["G1"] and out["G3_merge"]["resolved"] is True
    assert set(out["flags"]) >= {"G1", "G3_box"}
