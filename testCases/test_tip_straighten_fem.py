"""The straightening of the free surface's curved edges about a point
(``FaradayProblem.straighten_interface_edges``) on a finite-element problem, the synthetic pierced
state of ``ring_fem_state.py``: r3 of the delta triple (fix5 leg 1 at aa48432, step 7339) stopped on a
rebuilt mesh's curved element beside its repaired tip, the surface's edge there with its midside
node a quarter of its chord off the middle, the element's Bezier bound 0.054 against 0.2.

* an edge of the bubble's top whose midside node is pushed off its chord: read (``apply`` false)
  with its offset and its volume of revolution between curve and chord, nothing moved; applied,
  its midside node on the middle of its chord at every time level the time stepper reads, its
  velocity the mean of its vertices', the curved element's bound back up;
* edges farther than the radius, or within ``offset_min``: left as they are.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import ring_fem_state as S  # noqa: E402


def _edges(pb):
    out = []
    for e in pb.get_mesh("liquid/interface").elements():
        if e.nnode() == 3:
            out.append(tuple(e.node_pt(k) for k in range(3)))
    return out


def test_a_curved_edge_of_the_bubble_is_put_straight_and_its_volume_read(tmp_path):
    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.impose(pb)
        fields = pb.get_mesh("liquid").get_nodal_field_indices()
        ux = fields["velocity_x"]
        # an edge of the bubble's top, off the axis: the one nearest a point of its roof
        target = np.array([0.3 * S.DELTA, S.Z0 + 4.0 * S.DELTA])
        a, mid, b = min(_edges(pb), key=lambda e: math.hypot(
            0.5 * (e[0].x(0) + e[2].x(0)) - target[0], 0.5 * (e[0].x(1) + e[2].x(1)) - target[1]))
        ca, cb = np.array([a.x(0), a.x(1)]), np.array([b.x(0), b.x(1)])
        chord = cb - ca
        length = float(np.hypot(*chord))
        normal = np.array([-chord[1], chord[0]]) / length
        push = 0.25 * length
        levels = pb.timestepper.nprev_values() + 1
        for t in range(min(levels, mid.ntstorage())):
            for i in range(2):
                mid.set_x_at_t(t, i, 0.5 * (a.x_at_t(t, i) + b.x_at_t(t, i)) + push * normal[i])
        mid.set_value(ux, mid.value(ux) + 7.0)
        centre = 0.5 * (ca + cb)
        # read only
        read = pb.straighten_interface_edges(centre[0], centre[1], 2 * length, apply=False)
        assert len(read["edges"]) == 1 and read["edges"][0]["offset"] == pytest.approx(0.25)
        dV = 2 * math.pi * centre[0] * (2.0 / 3.0) * length * push
        assert read["dV"] == pytest.approx(dV, rel=1e-9) and read["dV_abs"] == pytest.approx(abs(dV))
        assert math.hypot(mid.x(0) - centre[0], mid.x(1) - centre[1]) == pytest.approx(push)
        # beyond the radius, or within offset_min: nothing
        assert pb.straighten_interface_edges(centre[0] + 50 * length, centre[1], length)["edges"] == []
        assert pb.straighten_interface_edges(centre[0], centre[1], 2 * length,
                                             offset_min=0.3)["edges"] == []
        assert math.hypot(mid.x(0) - centre[0], mid.x(1) - centre[1]) == pytest.approx(push)
        # applied: on the chord at every level, the velocity its vertices' mean
        done = pb.straighten_interface_edges(centre[0], centre[1], 2 * length)
        assert len(done["edges"]) == 1
        for t in range(min(levels, mid.ntstorage())):
            for i in range(2):
                assert mid.x_at_t(t, i) == pytest.approx(0.5 * (a.x_at_t(t, i) + b.x_at_t(t, i)),
                                                         abs=1e-15)
        assert mid.value(ux) == pytest.approx(0.5 * (a.value(ux) + b.value(ux)))
        # and nothing more to put straight there
        assert pb.straighten_interface_edges(centre[0], centre[1], 2 * length)["edges"] == []
