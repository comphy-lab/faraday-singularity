"""Contract tests of the quasi-1D gas solver (``neckgas.q1d``): the exact limits of the fast
verification tests (T1 a gas at rest stays at rest, T2 an adiabatic compression), the budgets added
for the coupling, the retry of a failed step, a conservative regrid with a rising lowest face (T7)
and a save and load in the middle of an advance (T9). The full verification (T1-T6) is
``verificationCases/neck-gas/verify_q1d.py``."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity.neckgas import q1d  # noqa: E402

G = 1.4


def _verify():
    spec = importlib.util.spec_from_file_location(
        "neckgas_verify", ROOT / "verificationCases" / "neck-gas" / "verify_q1d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_t1_rest_and_t2_adiabatic_are_exact():
    v = _verify()
    rest = v.t1_rest()
    assert rest["max_abs_u_over_c"] == 0.0 and rest["max_rel_dp"] == 0.0
    for case in v.t2_adiabatic():
        assert case["p_over_exact_minus_1"] < 1e-14
        assert case["max_abs_u"] < 1e-13


def _duct(N=60, zlo=0.0, zhi=1.0):
    """A converging-diverging duct with a piecewise-linear radius through 401 points, its exact
    volume below z (the frustum sum) and a gas state with flow."""
    zp = np.linspace(zlo, zhi, 401)
    rp = 0.1 + 0.4 * (zp - 0.5) ** 2

    def vcum(z):
        z = np.asarray(z, dtype=float)
        i = np.clip(np.searchsorted(zp, z) - 1, 0, len(zp) - 2)
        seg = math.pi * np.diff(zp) * (rp[:-1] ** 2 + rp[:-1] * rp[1:] + rp[1:] ** 2) / 3.0
        cum = np.r_[0.0, np.cumsum(seg)]
        w = z - zp[i]
        rq = rp[i] + w * (rp[i + 1] - rp[i]) / (zp[i + 1] - zp[i])
        return cum[i] + math.pi * w * (rp[i] ** 2 + rp[i] * rq + rq ** 2) / 3.0 + 0.05

    zf = np.linspace(zlo, zhi, N + 1)
    sol = q1d.Q1D(zf, G, 1.0, 1.0, bc_bottom="bz", bc_top="p0")
    rng = np.random.default_rng(3)
    V = np.diff(vcum(zf))
    sol.init_uniform(V, float(vcum(np.array([zlo]))[0]), u=0.3 * rng.standard_normal(N),
                     p=1.0 + 0.2 * rng.random(N), r=1.0 + 0.1 * rng.random(N))
    sol.mb = 0.05 * 1.1
    return sol, vcum


@pytest.mark.parametrize("z_new", [0.0, 0.1, 1.0 / 6.0, 0.23456])
def test_t7_a_rising_lowest_face_merges_into_the_bubble_zone_conservatively(z_new):
    sol, vcum = _duct()
    before = sol.totals()
    zf_new = np.r_[z_new, np.linspace(max(z_new, 0.3), 1.0, 41)[1:]] if z_new > 0 else \
        np.linspace(0.0, 1.0, 37)
    if z_new > 0:
        zf_new = np.unique(np.r_[z_new, np.linspace(z_new, 1.0, 45)])
    rec = sol.regrid(zf_new, vcum)
    after = sol.totals()
    m0 = before["mass_cells"] + before["mass_bz"]
    m1 = after["mass_cells"] + after["mass_bz"]
    E0 = before["energy_cells"] + before["energy_bz"]
    E1 = after["energy_cells"] + after["energy_bz"]
    assert abs(m1 - m0) <= 1e-14 * m0
    assert abs(E1 - E0) <= 1e-14 * E0
    P0, P1 = before["momentum_cells"], after["momentum_cells"] + rec["merged"]["momentum"]
    assert abs(P1 - P0) <= 1e-14 * (abs(P0) + m0)
    assert sol.zf[0] == z_new and sol.Vbz == pytest.approx(float(vcum(np.array([z_new]))[0]),
                                                           rel=1e-15)
    if z_new > 0:
        assert rec["merged"]["mass"] > 0 and sol.cum["merged_mass"] == rec["merged"]["mass"]
    else:
        assert rec["merged"]["mass"] == 0.0


def test_a_lower_lowest_face_is_refused():
    sol, vcum = _duct(zlo=0.1)
    with pytest.raises(ValueError):
        sol.regrid(np.linspace(0.05, 1.0, 20), vcum)


def _squeeze(sol, t, dt):
    """A wall motion that squeezes the duct uniformly: S(z, t) = S0(z) exp(-a t)."""
    a = 0.5
    V0 = sol._V0
    Vbz0 = sol._Vbz0
    Sf0 = sol._Sf0

    def wall(tt):
        f = math.exp(-a * tt)
        Vc = np.r_[Vbz0, Vbz0 + np.cumsum(V0)] * f
        return Sf0 * f, Vc, -a * Vc
    return sol.substep(t, dt, wall)


def _squeezed_duct():
    sol, vcum = _duct()
    sol._V0 = sol.V.copy()
    sol._Vbz0 = sol.Vbz
    sol._Sf0 = np.full(sol.N + 1, 0.2)
    return sol


def test_budgets_close_over_steps():
    sol = _squeezed_duct()
    sol.fric, sol.mu, sol.separated = 1.0, 1e-3, True
    t0 = sol.totals()
    t = 0.0
    for _ in range(200):
        t = _squeeze(sol, t, 0.5 * sol.max_dt())
    t1 = sol.totals()
    c = sol.cum
    m_err = (t1["mass_cells"] - t0["mass_cells"]) - (c["bottom_mass"] - c["top_mass"])
    E_err = (t1["energy_cells"] - t0["energy_cells"]) - (c["bottom_energy"] - c["top_energy"]
                                                         + c["wall_work"])
    assert abs(m_err) <= 1e-12 * t0["mass_cells"]
    assert abs(E_err) <= 1e-12 * t0["energy_cells"]
    total = (t1["mass_cells"] + t1["mass_bz"]) - (t0["mass_cells"] + t0["mass_bz"])
    assert abs(total + c["top_mass"]) <= 1e-12 * t0["mass_cells"]
    assert c["fric_loss"] > 0 and c["sep_loss"] >= 0 and c["wall_work"] > 0


def test_a_failed_step_is_retaken_at_first_order_then_halved(monkeypatch):
    sol = _squeezed_duct()
    calls = []
    real = q1d.Q1D.flux_step

    def flaky(self, dt, Sf, ue):
        calls.append((self.order, dt))
        if len(calls) <= 2:
            return False
        return real(self, dt, Sf, ue)
    monkeypatch.setattr(q1d.Q1D, "flux_step", flaky)
    dt = 0.5 * sol.max_dt()
    t = _squeeze(sol, 0.0, dt)
    assert [o for o, _ in calls] == [2, 1, 1]
    assert calls[0][1] == calls[1][1] == dt and calls[2][1] == 0.5 * dt
    assert t == 0.5 * dt and sol.cum["retries"] == 2 and sol.order == 2


def test_a_step_failing_at_every_attempt_raises(monkeypatch):
    sol = _squeezed_duct()
    monkeypatch.setattr(q1d.Q1D, "flux_step", lambda self, dt, Sf, ue: False)
    U = sol.U.copy()
    with pytest.raises(q1d.StepFailed):
        _squeeze(sol, 0.0, 1e-3)
    assert np.array_equal(sol.U, U)


def test_t9_save_and_load_in_the_middle_of_an_advance_is_bitwise():
    a = _squeezed_duct()
    t = 0.0
    for _ in range(50):
        t = _squeeze(a, t, 0.5 * a.max_dt())
    saved = copy.deepcopy(a.state_dict())
    b = q1d.Q1D(np.linspace(0, 1, 3), 1.0, 2.0, 3.0)  # another grid and gas, all overwritten
    b.load_state_dict(saved)
    b._V0, b._Vbz0, b._Sf0 = a._V0, a._Vbz0, a._Sf0
    ta = tb = t
    for _ in range(50):
        ta = _squeeze(a, ta, 0.5 * a.max_dt())
        tb = _squeeze(b, tb, 0.5 * b.max_dt())
    assert ta == tb
    for k in ("U", "V", "zf"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert (a.Vbz, a.mb, a.Kb) == (b.Vbz, b.mb, b.Kb) and a.cum == b.cum


def test_the_port_keeps_the_solver_of_its_source():
    """The arithmetic of the one-way solver is unchanged: a step of the port and of a copy of its
    methods without the budgets agree to the bit (here: against the same step with the budget
    keys removed, which no method reads)."""
    a = _squeezed_duct()
    b = _squeezed_duct()
    for k in q1d.BUDGET_KEYS:
        b.cum.pop(k)
    b.cum.update({k: 0.0 for k in q1d.BUDGET_KEYS})
    _squeeze(a, 0.0, 1e-3)
    _squeeze(b, 0.0, 1e-3)
    assert np.array_equal(a.U, b.U)
