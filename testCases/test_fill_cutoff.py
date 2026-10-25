"""The fill's budgets weighted by the box's cutoff, without a finite-element problem
(``surgery.box_integrals`` and ``surgery._box_projection`` with ``cutoff``; ``groovesurgery.
unweighted_g2``): r1h's fill of the ring at step 7239 (event 4) missed G2, 4.0e-9 against 2.7e-9,
by one sample of the budget's polar grid at the box's edge (rho 3.95e-3 of R 4e-3) that the
rebuild moved across the downward jet's surface (u_z -2863); the fill's own momentum closed to
1.5e-11. The lead's decision (2026-10-25): every fill's box integrals weighted by the cutoff
``phi`` the identity G3 already uses, the unweighted G2 recorded beside it.

* ``box_integrals`` with ``cutoff``: each cell weighted by ``phi`` (one within half the radius,
  zero at the edge), the sums without it kept as ``unweighted``, for the box and each polygon;
  without ``cutoff`` the sums as before;
* a sample at the box's edge changing phase moves the weighted sums by ``phi`` of its cell's, the
  unweighted by the whole cell's;
* ``_box_projection`` with ``cutoff``: ``before_phi`` and ``after_phi``, whose change is the
  identity's left side ``phi_lhs``; without it the record as before;
* ``unweighted_g2``: the unweighted G2 against G2's limit, reported, ``beyond`` where it misses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import surgery  # noqa: E402
from faraday_singularity.groovesurgery import unweighted_g2  # noqa: E402

R = 4e-3
CENTRE = (3e-4, 4.6115)


def _fields(grid, *, flip=None, uz=-2000.0):
    """A velocity sampled on the grid: liquid everywhere but where ``flip`` (a mask) says gas,
    u_z ``uz`` and u_r a tenth of it."""
    n = len(grid["r"])
    found = np.ones(n, dtype=bool) if flip is None else ~flip
    f = {"velocity_x": np.full(n, 0.1 * uz), "velocity_y": np.full(n, uz),
         "pressure": np.linspace(0.0, 1.0, n)}
    return f, found


def test_the_box_integrals_weighted_by_the_cutoff_and_without_it(monkeypatch):
    grid = surgery.site_grid(*CENTRE, R, n=64)
    state = {}
    monkeypatch.setattr(surgery, "sample_fields",
                        lambda pb, r, z, names: _fields(grid, flip=state.get("flip")))
    square = np.array([[CENTRE[0] - 1e-4, CENTRE[1] - 1e-4], [CENTRE[0] + 1e-4, CENTRE[1] - 1e-4],
                       [CENTRE[0] + 1e-4, CENTRE[1] + 1e-4], [CENTRE[0] - 1e-4, CENTRE[1] + 1e-4]])
    plain = surgery.box_integrals(None, grid, {"sq": square})
    cut = surgery.box_integrals(None, grid, {"sq": square}, cutoff=True)
    assert "unweighted" not in plain and "weight" not in plain  # as before
    w, phi = grid["dV"], grid["phi"]
    assert plain["Pz"] == pytest.approx(np.sum(w * -2000.0), rel=1e-12)
    assert cut["Pz"] == pytest.approx(np.sum(w * phi * -2000.0), rel=1e-12)
    assert cut["weight"] == "cutoff" and cut["unweighted"] == {
        k: plain[k] for k in ("Pz", "KE", "V_grid", "points")}
    assert cut["KE"] < plain["KE"] and cut["V_grid"] < plain["V_grid"]
    # within half the radius the cutoff is one: the fill's polygon reads the same either way
    assert cut["sq"]["Pz"] == pytest.approx(plain["sq"]["Pz"], rel=1e-12)
    assert cut["sq"]["unweighted"] == plain["sq"]
    # one cell at the edge changes phase: the unweighted sum moves by the cell, the weighted by
    # phi of it
    k = int(np.argmax(grid["rho"] * (grid["dV"] > 0)))
    state["flip"] = np.zeros(len(grid["r"]), dtype=bool)
    state["flip"][k] = True
    moved = surgery.box_integrals(None, grid, cutoff=True)
    assert moved["unweighted"]["Pz"] - cut["unweighted"]["Pz"] == pytest.approx(
        2000.0 * w[k], rel=1e-9)
    assert moved["Pz"] - cut["Pz"] == pytest.approx(2000.0 * w[k] * phi[k], rel=1e-6)
    assert phi[k] < 1e-2


def test_the_projections_box_with_the_cutoff_and_without_it():
    grid = surgery.site_grid(*CENTRE, R, n=32)
    before, found = _fields(grid)
    after, _ = _fields(grid, uz=-1990.0)
    plain = surgery._box_projection(grid, before, after, found)
    cut = surgery._box_projection(grid, before, after, found, cutoff=True)
    assert set(cut) - set(plain) == {"before_phi", "after_phi"}
    assert {k: v for k, v in cut.items() if k in plain} == plain
    assert cut["after_phi"]["Pz"] - cut["before_phi"]["Pz"] == pytest.approx(
        plain["phi_lhs"], rel=1e-9)
    assert cut["before_phi"]["V_grid"] == pytest.approx(np.sum(grid["dV"] * grid["phi"]),
                                                        rel=1e-12)


def test_the_unweighted_g2_is_reported_against_g2s_limit():
    before = {"Pz": 1.0, "unweighted": {"Pz": 2.0}}
    transfer = {"Pz": 0.9, "unweighted": {"Pz": 1.9 + 4.0e-9}}
    fill = {"Pz_fill": -0.1, "unweighted": {"Pz": -0.1}}
    closures = {"G2": 1.5e-11, "limits": {"G2": 2.69e-9}}
    got = unweighted_g2(before, transfer, fill, closures)
    assert got["value"] == pytest.approx(4.0e-9, rel=1e-6) and got["beyond"]
    assert got["limit"] == 2.69e-9 and got["weighted"] == 1.5e-11
    transfer["unweighted"]["Pz"] = 1.9
    assert not unweighted_g2(before, transfer, fill, closures)["beyond"]
