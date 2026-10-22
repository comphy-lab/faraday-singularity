"""Contract tests of the gas geometry and the coupler (``neckgas.geometry``, ``neckgas.coupler``)
on synthetic chains: the block's keys, the duct checks, the exact volumes and rates of a linear
chain and their second-order error on a curved one (T8), the dormant, pending and active stages,
the sync after a rebuild, the traction table, the step limit, the budgets, the regrids (the neck
halving and the bottom rising into ``z_lo``), every stop, the hand-over partition and a save and
load in the middle of a run continuing to the bit (T9)."""

from __future__ import annotations

import copy
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity.neckgas import coupler as cp  # noqa: E402
from faraday_singularity.neckgas.geometry import (ChainGeom, GasGeometryError,  # noqa: E402
                                                  chain_neck, single_valued)

BLOCK = {"z_hi": 1.0, "neck_window": [0.45, 0.75]}


def _settings(**kw):
    return cp.GasSettings.from_dict({**BLOCK, **kw})


# --- the block ----------------------------------------------------------------------------------

def test_the_block_round_trips_and_refuses_what_it_does_not_know():
    s = _settings(coupling="one-way", variants=[{"name": "A"}, {"name": "S", "separated": True},
                                                {"name": "A-f1", "fric": 1.0}])
    again = cp.GasSettings.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again == s and again.sha256() == s.sha256()
    assert s.to_dict()["p0"] == cp.P0 and s.density_ratio == 1.2 / 998.0
    for bad in ({"unknown": 1}, {"coupling": "both"}, {"variants": [{"name": "A"}, {"name": "B"}]},
                {"variants": [{"name": "A", "drag": 1}]}, {"r_on": 0.0}, {"gamma": 1.0},
                {"neck_window": [0.8, 0.5]}, {"z_lo": 2.0}, {"stall_steps": 0},
                {"s_passive": [1.0]}):
        with pytest.raises(ValueError):
            _settings(**bad)
    with pytest.raises(ValueError, match="required"):
        cp.GasSettings.from_dict({"z_hi": 1.0})
    rest, block = cp.split_gas_block({"gas": dict(BLOCK), "neck_stop": 1e-4})
    assert rest == {"neck_stop": 1e-4} and block == BLOCK
    assert cp.split_gas_block({"neck_stop": 1e-4}) == ({"neck_stop": 1e-4}, None)


# --- the geometry (T8) --------------------------------------------------------------------------

def _cone(n, a=0.1, b=0.3, z0=0.2, z1=1.2):
    """A cone ``r = a + b (z - z0)`` from a flat disc of radius ``a`` at ``z0`` on the axis."""
    z = np.r_[z0, np.linspace(z0, z1, n)]
    r = np.r_[0.0, a + b * (np.linspace(z0, z1, n) - z0)]
    return r, z


def test_t8_volumes_and_rates_are_exact_on_a_linear_chain():
    r, z = _cone(41)
    ur = np.full_like(r, -0.7)
    uz = np.full_like(r, 0.2)
    g = ChainGeom(r, z, ur, uz, 0.3, 1.0)
    zq = np.linspace(0.31, 0.99, 17)
    S, Vc, Rc, dSdt = g.at(zq)
    a, b, z0 = 0.1, 0.3, 0.2
    rq = a + b * (zq - z0)
    exact_V = math.pi * (rq ** 3 - a ** 3) / (3 * b)
    assert np.allclose(S, math.pi * rq ** 2, rtol=1e-14, atol=0)
    assert np.allclose(Vc, exact_V, rtol=1e-13, atol=0)
    # int 2 pi r (u_r dz - u_z dr): the disc (dz = 0, dr = a), then the cone, uniform velocities
    exact_R = -2 * math.pi * 0.2 * a * a / 2 + 2 * math.pi * (
        -0.7 * (rq ** 2 - a ** 2) / (2 * b) - 0.2 * (rq ** 2 - a ** 2) / 2)
    assert np.allclose(Rc, exact_R, rtol=1e-13, atol=1e-15)
    assert np.allclose(dSdt, 2 * math.pi * rq * (-0.7 - 0.2 * b), rtol=1e-13, atol=0)


def test_t8_volumes_converge_at_second_order_on_a_curved_chain():
    def chain(n):
        z = np.linspace(0.0, 1.0, n)
        return np.sqrt(z), z  # a paraboloid z = r^2

    errs = []
    for n in (51, 101, 201):
        r, z = chain(n)
        g = ChainGeom(r, z, 0 * r, 0 * r, 0.2, 0.9)
        Vc = g.at(np.array([0.5]))[1][0]
        errs.append(abs(Vc - math.pi * 0.5 ** 2 / 2))
    rates = [math.log2(errs[k] / errs[k + 1]) for k in range(2)]
    assert all(1.8 < x < 2.2 for x in rates), rates


def test_the_duct_checks():
    r, z = _cone(41)
    assert single_valued(r, z, 0.3, 1.0) is None
    assert single_valued(r, z, 0.1, 1.0)["reason"] == "bottom-above-z_lo"
    z2 = z.copy()
    z2[20] = z2[18]
    assert single_valued(r, z2, 0.3, 1.0)["reason"] == "not-monotone"
    z3 = np.r_[z, 0.9]
    r3 = np.r_[r, 1.0]  # back below z_hi beyond the duct
    assert single_valued(r3, z3, 0.3, 0.95)["reason"] in ("re-entry", "returns-below-z_hi")
    with pytest.raises(GasGeometryError):
        ChainGeom(r, z2, 0 * r, 0 * r, 0.3, 1.0)


# --- synthetic necks ----------------------------------------------------------------------------

def neck_chain(t, *, r0=0.05, T=1e-2, z_n=0.6, B=0.4, z_b=0.3, w_b=0.0, n=601, z_top=1.2):
    """A cavity whose neck at ``z_n`` closes as ``r0 (1 - t/T)^(1/2)``: ``r = r_n + B (z - z_n)^2``
    above a flat bottom at ``z_b + w_b t`` on the axis, the walls moving with the neck."""
    rn = r0 * math.sqrt(max(1.0 - t / T, 1e-12))
    ds = -0.5 * r0 / T / math.sqrt(max(1.0 - t / T, 1e-12))
    zb = z_b + w_b * t
    z = np.r_[zb, np.linspace(zb + 1e-3, z_top, n - 1)]
    r = rn + B * (z - z_n) ** 2
    r[0] = 0.0
    ur = np.full_like(r, ds)
    ur[0] = 0.0
    uz = np.zeros_like(r)
    uz[0] = w_b
    return r, z, ur, uz


def _run(gas, times, step0=0, **kw):
    rows = []
    for k in range(len(times) - 1):
        gas.before_step(times[k], neck_chain(times[k], **kw), step=step0 + k, dt_prev=1e-5)
        rows += gas.after_step(times[k + 1], neck_chain(times[k + 1], **kw), step=step0 + k + 1)
        if gas.stop:
            break
    return rows


def _gas(**kw):
    """A gas on the synthetic neck with a coarse grid (the tests' cost)."""
    return cp.NeckGas(_settings(**{"r_on": 0.03, "z_lo": 0.35, "half_width": 5.0,
                                   "dz_cap": 5e-3, "dz_max_lo": 1e-2, "dz_max_hi": 1e-2, **kw}))


def test_dormant_then_pending_then_active():
    gas = _gas()
    t = np.linspace(0.0, 8e-3, 41)
    gas.before_step(0.0, neck_chain(0.0), step=0)
    assert gas.status == "dormant" and gas.after_step(t[1], neck_chain(t[1]), step=1) == []
    gas = _gas()
    rows = _run(gas, t)
    assert gas.status == "active" and gas.stop is None
    on = gas.t_on
    first = [r for r in rows if r["time"] == on]
    assert len(first) == 1 and first[0]["dt"] == 0.0 and first[0]["neck_source"] == "neck"
    r_on = neck_chain(on)[0]
    assert np.min(r_on[1:]) <= 0.03 < np.min(neck_chain(t[list(t).index(on) - 1])[0][1:])
    after = [r for r in rows if r["time"] > on]
    assert all(abs(r["mass_error"]) < 1e-12 and abs(r["energy_error"]) < 1e-10 for r in after)
    assert all(r["n_sub"] >= 1 and r["s"] > 0 for r in after)
    p = [r["p_th"] for r in after]
    assert p[-1] > gas.s.p0  # a closing neck pushes the throat gas above ambient


def test_the_traction_table_and_its_coupling():
    two = _gas()
    one = _gas(coupling="one-way", variants=[{"name": "A"}, {"name": "S", "separated": True}])
    t = np.linspace(0.0, 8e-3, 41)
    for gas in (two, one):
        _run(gas, t)
        gas.before_step(t[-1], neck_chain(t[-1]), step=99)
    z = np.array([0.0, 0.34, 0.35, 0.6, 0.999, 1.0, 1.5])
    T2 = two.traction(z)
    sol = two.variants[0].sol
    pb = sol.bz_state()[1] - sol.p0
    assert T2[0] == T2[1] == T2[2] == pb and T2[-1] == T2[-2] == 0.0
    zc, p = sol.zc, sol.prim(sol.U)[2]
    assert np.allclose(two.traction(zc), p - sol.p0, rtol=0, atol=1e-12)
    assert math.isfinite(two.dt_limit()) and two.dt_limit() > 0
    assert np.all(one.traction(z) == 0.0) and one.dt_limit() == math.inf
    assert np.any(one.would_be_traction(z) != 0.0)


def test_a_rebuild_is_absorbed_by_the_wall_step():
    gas = _gas()
    t = np.linspace(0.0, 8e-3, 41)
    _run(gas, t)
    r, z, ur, uz = neck_chain(t[-1])
    zz = np.r_[z[0], np.linspace(z[1], z[-1], 450)]  # the same surface on other nodes
    rr = np.interp(zz, z, r)
    rr[0] = 0.0
    m0 = gas.variants[0].sol.totals()
    out = gas.before_step(t[-1], (rr, zz, np.interp(zz, z, ur), np.interp(zz, z, uz)), step=40)
    m1 = gas.variants[0].sol.totals()
    assert out["synced"] and out["dV_remesh"] > 0.0
    assert m1["mass_cells"] == pytest.approx(m0["mass_cells"], rel=1e-14)
    again = gas.before_step(t[-1], (rr, zz, np.interp(zz, z, ur), np.interp(zz, z, uz)), step=40)
    assert not again["synced"]


def test_regrids_when_the_neck_halves_and_when_the_bottom_rises():
    gas = _gas(z_lo=None, h_bz=0.02)
    t = np.linspace(0.0, 9.8e-3, 124)
    rows = _run(gas, t, w_b=4.0, B=4.0)  # a short neck: no thread before it halves
    reasons = {x for g in gas.regrids for x in g["reasons"]}
    assert {"neck", "z_lo"} <= reasons, gas.regrids
    assert gas.stop is None
    assert all(abs(r["mass_error"]) < 1e-12 for r in rows)
    assert gas.variants[0].sol.cum["merged_mass"] > 0.0
    zl = [g["z_lo"] for g in gas.regrids]
    assert zl == sorted(zl)


@pytest.mark.parametrize("kw, reason", [
    ({"knudsen_max": 1e-12}, "gas-continuum"),
    ({"thread_length": 0.5}, "gas-thread"),
    ({"s_passive": [1e6, 0.0], "stall_steps": 3, "stall_acoustic": 1e-9}, "gas-stall"),
    ({"mass_tol": 1e-300}, "gas-budget"),
])
def test_the_stops(kw, reason):
    gas = _gas(**kw)
    _run(gas, np.linspace(0.0, 8e-3, 41))
    assert gas.stop is not None and gas.stop["reason"] == reason, gas.stop


def test_a_reopening_neck_stops_the_run():
    gas = _gas(reopen_steps=3)
    t = np.linspace(0.0, 7.6e-3, 39)
    _run(gas, t)
    assert gas.status == "active" and gas.stop is None
    back = t[::-1]  # the same states backwards: the neck opens again
    for k in range(5):
        tt = t[-1] + (k + 1) * 1e-4
        gas.before_step(gas.t, neck_chain(back[k]), step=40 + k)
        gas.after_step(tt, neck_chain(back[k + 1]), step=41 + k)
        if gas.stop:
            break
    assert gas.stop["reason"] == "gas-stall" and gas.stop["trigger"] == "reopen"


def test_a_chain_that_is_no_duct_stops():
    gas = _gas()
    t = np.linspace(0.0, 8e-3, 41)
    _run(gas, t)
    r, z, ur, uz = neck_chain(t[-1])
    z = z.copy()
    z[300] = z[298]
    with pytest.raises(GasGeometryError):
        gas.before_step(t[-1], (r, z, ur, uz), step=40)
    gas.after_step(t[-1] + 1e-4, (r, z, ur, uz), step=41)
    assert gas.stop["reason"] == "gas-geometry"


def test_the_hand_over_partition_conserves_the_gas():
    gas = _gas(coupling="one-way", variants=[{"name": "A"}, {"name": "S", "separated": True}])
    _run(gas, np.linspace(0.0, 8e-3, 41))
    out = gas.handover()
    assert set(out) == {"A", "S"}
    for name, h in out.items():
        sol = next(v.sol for v in gas.variants if v.spec.name == name)
        tot = sol.totals()
        assert h["m_b"] + h["m_above"] == pytest.approx(tot["mass_cells"] + tot["mass_bz"],
                                                        rel=1e-13)
        assert h["E_b"] + h["E_above"] == pytest.approx(tot["energy_cells"] + tot["energy_bz"],
                                                        rel=1e-13)
        assert h["V_gas_below"] == pytest.approx(h["V_b"], rel=1e-12)
        assert h["p_b_mix"] > 0 and h["C_kappa"]["1"] == pytest.approx(h["p_b_mix"] * h["V_b"],
                                                                       rel=1e-12)


def _roundtrip(gas):
    arrays, meta = gas.state_dict()
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    buf.seek(0)
    loaded = dict(np.load(buf))
    other = cp.NeckGas(gas.s)
    other.load_state_dict(loaded, json.loads(json.dumps(meta)))
    return other


def test_t9_a_save_and_load_in_the_middle_of_a_run_continues_to_the_bit():
    t = np.linspace(0.0, 9e-3, 61)
    a = _gas(coupling="one-way", variants=[{"name": "A"}, {"name": "S-f1", "separated": True,
                                                           "fric": 1.0}])
    _run(a, t[:31])
    b = _roundtrip(a)
    ra = _run(a, t[30:], step0=30)
    rb = _run(b, t[30:], step0=30)
    assert len(ra) == len(rb) > 0
    for x, y in zip(ra, rb):
        x = {k: v for k, v in x.items() if not k.startswith("_")}
        y = {k: v for k, v in y.items() if not k.startswith("_")}
        assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
    for va, vb in zip(a.variants, b.variants):
        assert np.array_equal(va.sol.U, vb.sol.U) and va.sol.cum == vb.sol.cum
    with pytest.raises(ValueError):
        cp.NeckGas(_settings(r_on=0.5)).load_state_dict(*a.state_dict())


def test_a_dormant_state_round_trips():
    gas = _gas()
    other = _roundtrip(gas)
    assert other.status == "dormant" and other.chain is None


def test_the_neck_before_one_forms_is_the_window_minimum():
    r, z = _cone(41)
    r_n, z_n, j, source = chain_neck(r, z, 0.3, 1.0, (0.45, 0.75))
    assert source == "window" and z_n > 0.45 and r_n == pytest.approx(r[j])
    rr, zz, _, _ = neck_chain(0.0)
    assert chain_neck(rr, zz, 0.35, 1.0, (0.45, 0.75))[3] == "neck"


def test_retire_stops_the_gas():
    gas = _gas()
    _run(gas, np.linspace(0.0, 8e-3, 41))
    gas.retire()
    assert gas.status == "retired" and np.all(gas.traction(np.array([0.5])) == 0)
    assert gas.after_step(1.0, neck_chain(8e-3), step=99) == []
    assert copy.copy(gas.handover()) == {}
