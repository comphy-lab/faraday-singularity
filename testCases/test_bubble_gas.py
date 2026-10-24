"""The polytropic bubble after the cut (``neckgas.bubble``) and the surface the driver reads for it
(``CollapseDriver._bubble_surface``), without a finite-element problem: the state the coupled
gas's partition hands over, the polytrope through jumps, the ring's cut and the removal, the work
along the polytrope against the lagged work of the traction, the isothermal gas, the bitwise state
of a dump, and the driver's choice of the bubble's chain, the ring's loop or none."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity.neckgas import bubble as bg  # noqa: E402

HANDOVER = {"z_n": 4.703, "m_b": 6.7e-8, "E_b": 0.163, "KE_b": 6.3e-4, "V_b": 2.11e-5,
            "V_gas_below": 2.11e-5, "p_pk": 3074.0, "p_b_mix": 3074.0, "p_bz": 2626.0,
            "m_above": 7.4e-6, "E_above": 1.0, "rho0": 1.2 / 998.0, "p0": 781.6358024691358,
            "gamma": 1.4}


def test_the_bubble_starts_on_the_handover_polytrope():
    g = bg.BubbleGas.from_handover("polytropic", HANDOVER, 2.0e-5, t=1.0, step=7)
    assert g.kappa == 1.4 and g.m == HANDOVER["m_b"]
    assert g.pressure(HANDOVER["V_gas_below"]) == pytest.approx(3074.0, rel=1e-14)
    assert g.C == 3074.0 * 2.11e-5 ** 1.4
    tr = g.transitions[0]
    assert tr["kind"] == "cut" and tr["V_before"] == 2.11e-5 and tr["V_after"] == 2.0e-5
    assert tr["p_after"] == pytest.approx(3074.0 * (2.11 / 2.0) ** 1.4, rel=1e-13)
    assert g.V_end == 2.0e-5 and g.stage == "bubble"
    with pytest.raises(ValueError):
        bg.BubbleGas.from_handover("adiabatic", HANDOVER, 2e-5, t=0, step=0)


def test_steps_jumps_and_work():
    g = bg.BubbleGas.from_handover("polytropic", HANDOVER, 2.0e-5, t=0.0, step=0)
    p0 = HANDOVER["p0"]
    T = g.before_step(0.0, 0, "bubble", 2.0e-5)  # the cut's own volume: no jump
    assert T == pytest.approx(g.pressure(2.0e-5) - p0, rel=1e-15) and g.jumps == 0
    row = g.after_step(1e-6, 1, 1.9e-5, nodes=12)
    assert row["V_start"] == 2.0e-5 and row["V_end"] == 1.9e-5 and row["nodes"] == 12
    assert row["W_traction"] == pytest.approx((g.pressure(2e-5) - p0) * (1.9e-5 - 2e-5), rel=1e-14)
    exact = quad(lambda V: g.pressure(V) - p0, 2e-5, 1.9e-5)[0]
    assert row["W_poly"] == pytest.approx(exact, rel=1e-10)
    assert row["E_end"] == pytest.approx(g.pressure(1.9e-5) * 1.9e-5 / 0.4, rel=1e-14)
    assert row["rho_over_rho0"] == pytest.approx(HANDOVER["m_b"] / 1.9e-5 / HANDOVER["rho0"])
    # a rebuild between steps: a jump along the polytrope, in the next row
    g.before_step(1e-6, 1, "bubble", 1.8999e-5)
    row = g.after_step(2e-6, 2, 1.85e-5)
    assert row["jumps"] == 1 and row["jump_V"] == pytest.approx(1.8999e-5 - 1.9e-5)
    assert row["jump_W"] == pytest.approx(quad(lambda V: g.pressure(V) - p0, 1.9e-5,
                                               1.8999e-5)[0], rel=1e-9)
    g.before_step(2e-6, 2, "bubble", 1.85e-5)
    assert g.after_step(3e-6, 3, 1.8e-5)["jump_V"] == 0.0


def test_the_ring_keeps_the_gas_on_the_same_polytrope_and_the_removal_ends_it():
    g = bg.BubbleGas.from_handover("polytropic", HANDOVER, 2.0e-5, t=0.0, step=0)
    g.before_step(0.0, 0, "bubble", 2.0e-5)
    g.after_step(1e-6, 1, 1.5e-5)
    T = g.before_step(1e-6, 1, "ring", 1.4e-5)  # the ring's cut closed the column
    tr = g.transitions[-1]
    assert tr["kind"] == "bubble-to-ring" and g.stage == "ring" and tr["m"] == g.m
    assert tr["p_after"] == pytest.approx(g.C / 1.4e-5 ** 1.4, rel=1e-14)
    assert T == pytest.approx(tr["p_after"] - HANDOVER["p0"], rel=1e-14)
    g.after_step(2e-6, 2, 1.3e-5)
    assert g.before_step(2e-6, 2, "gone", None) == 0.0
    assert g.stage == "gone" and g.transitions[-1]["kind"] == "removed"
    assert g.transitions[-1]["E_removed"] == pytest.approx(g.energy(1.3e-5))
    assert g.after_step(3e-6, 3, None) is None and g.before_step(3e-6, 3, "gone", None) == 0.0


def test_the_isothermal_gas():
    g = bg.BubbleGas.from_handover("isothermal", HANDOVER, 2.11e-5, t=0.0, step=0)
    assert g.kappa == 1.0 and g.pressure(1.055e-5) == pytest.approx(2 * 3074.0, rel=1e-14)
    assert bg.poly_work(g.C, 1.0, 2e-5, 1e-5) == pytest.approx(-g.C * math.log(2.0), rel=1e-14)
    assert math.isnan(g.energy(1e-5))


def test_the_state_round_trips_to_the_bit():
    g = bg.BubbleGas.from_handover("polytropic", HANDOVER, 2.0e-5, t=0.0, step=0)
    g.before_step(0.0, 0, "bubble", 2.0e-5)
    g.after_step(1e-6, 1, 1.9e-5)
    h = bg.BubbleGas.from_dict(json.loads(json.dumps(g.state_dict())))
    for x in (g, h):
        x.before_step(1e-6, 1, "bubble", 1.88e-5)
    assert json.dumps(g.after_step(2e-6, 2, 1.8e-5)) == json.dumps(h.after_step(2e-6, 2, 1.8e-5))
    assert g.state_dict() == h.state_dict()


def _bubble_chain(n=41, z0=0.3, z1=0.6, R=0.1):
    th = np.linspace(0.0, math.pi, n)
    zc, a = 0.5 * (z0 + z1), 0.5 * (z1 - z0)
    return {"open": False, "r": R * np.sin(th), "z": zc - a * np.cos(th)}


def test_the_driver_reads_the_bubble_the_ring_or_none():
    from faraday_singularity.driver import CollapseDriver
    from faraday_singularity.reconnect import revolved_volume

    open_chain = {"open": True, "r": np.linspace(0, 1, 5), "z": np.full(5, 1.0)}
    bub = _bubble_chain()
    loop = {"r": 0.2 + 0.05 * np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
            "z": 0.5 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False))}
    pb = types.SimpleNamespace(ring=None, interface_segments=lambda: [bub, open_chain],
                               interface_loops=lambda: [])
    drv = types.SimpleNamespace(pb=pb, _ring=lambda: pb.ring)
    read = CollapseDriver._bubble_surface
    stage, V, (r, z) = read(drv)
    assert stage == "bubble" and V == revolved_volume(np.column_stack([bub["r"], bub["z"]]))
    assert V == pytest.approx(4 / 3 * math.pi * 0.1 ** 2 * 0.15, rel=5e-3)
    pb.ring, pb.interface_loops = {"removed": None}, lambda: [loop]
    stage, V, _ = read(drv)
    assert stage == "ring" and V == pytest.approx(2 * math.pi ** 2 * 0.2 * 0.05 ** 2, rel=1e-2)
    pb.interface_loops = lambda: []
    assert read(drv) == (None, None, None)  # the ring stage without a loop, not removed: unread
    pb.ring = {"removed": {"V": 1e-6}}
    assert read(drv) == ("gone", None, None)
    pb.ring, pb.interface_segments = None, lambda: [open_chain]
    assert read(drv) == (None, None, None)
