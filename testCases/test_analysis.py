"""Contract tests for the collapse post-processing on synthetic data with known answers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "postProcess"))
sys.path.insert(0, str(ROOT / "modelComparisonCases"))

from collapse_analysis import analyse, clean, fit_power_law, growth, main, time_free  # noqa: E402
from inertial_similarity import alpha_of_beta, legendre_p, nu_of_beta  # noqa: E402


def _jet(alpha, rj_birth, rj_window=(3e-3, 3e-2), n=40, spacing="radius"):
    """``r_j = (t - t0)^alpha`` with ``t0 = 1``: birth time and the samples inside the window."""
    t_birth = 1.0 + rj_birth ** (1 / alpha)
    if spacing == "radius":  # adaptive steps follow the local scale: geometric in r_j
        t = 1.0 + np.geomspace(*rj_window, n) ** (1 / alpha)
    else:
        t = np.linspace(1.0 + rj_window[0] ** (1 / alpha), 1.0 + rj_window[1] ** (1 / alpha), n)
    return t_birth, t, (t - 1.0) ** alpha


@pytest.mark.parametrize("alpha", [0.5, 0.629, 2 / 3, 0.726])
@pytest.mark.parametrize("rj_birth", [1e-4, 1e-3, 2.5e-3])
@pytest.mark.parametrize("spacing", ["radius", "time"])
def test_power_law_origin_next_to_birth(alpha, rj_birth, spacing):
    # the origin lies 0.5-3 % of the span before birth, where the old grid stepped by 2.3 %
    t_birth, t, r = _jet(alpha, rj_birth, spacing=spacing)
    fit = fit_power_law(t, r, t_birth, span=t[-1] - t_birth)
    assert fit["alpha"] == pytest.approx(alpha, abs=1e-4)
    assert fit["t0_fit"] == pytest.approx(1.0, abs=1e-4 * (t[-1] - t_birth))
    assert fit["log_A"] == pytest.approx(0.0, abs=1e-3)
    assert fit["log_rms"] < 1e-5
    assert not fit["t0_at_bound"]


def test_power_law_origin_beyond_the_range_is_flagged():
    t_birth, t, r = _jet(0.629, 1e-3)
    span = 0.5 * (t_birth - 1.0)  # the true origin lies twice as far before birth
    fit = fit_power_law(t, r, t_birth, span=span)
    assert fit["t0_at_bound"]
    assert fit["t0_fit"] == pytest.approx(t_birth - span, rel=1e-12)


def test_power_law_without_admissible_origin():
    t = np.linspace(1.0, 2.0, 10)
    assert fit_power_law(t, t, t_first=1.5, span=0.1) == {}


@pytest.mark.parametrize("nu", [0.0, 0.5, 0.40966])
def test_time_free_slopes_of_exact_power_laws(nu):
    # line sink (q_j constant), inertio-capillary (nu = 1/2), cone at 38.4 deg; samples clustered
    # at small r_j, as adaptive steps place them
    r_j = np.geomspace(3e-3, 3e-2, 400) ** np.linspace(1.0, 1.15, 400)
    q_j = 2.0 * r_j**nu
    V_j = q_j / r_j
    out = time_free(r_j, math.pi * r_j * q_j, q_j**2 / r_j, V_j, Oh=0.04, bins=12, nu_cone=nu)
    assert out["slope_q_j"] == pytest.approx(nu, abs=1e-9)
    assert out["slope_Q_j"] == pytest.approx(1 + nu, abs=1e-9)
    assert out["slope_We_j"] == pytest.approx(2 * nu - 1, abs=1e-9)
    assert out["slope_identity_residual"] < 1e-9
    assert out["alpha_free"] == pytest.approx(1 / (2 - nu), abs=1e-9)
    assert all(row["slope_q_j"] == pytest.approx(nu, abs=1e-9) for row in out["local"])
    assert out["sensitivity"]["alpha_range"] == pytest.approx([1 / (2 - nu)] * 2, abs=1e-9)
    assert out["laws"]["cone"]["log_rms"] < 1e-9
    assert out["laws"]["line_sink"]["log_rms"] > 1e-3 or nu == 0.0
    if nu != 0.5:  # We_j is constant at nu = 1/2
        assert out["We_j"]["rises_as_r_j_falls"] == (nu < 0.5)
    assert out["Re_j"]["max"] == pytest.approx(q_j.max() / 0.04)
    assert sum(out["bin_counts"]) == r_j.size


def test_growth_from_the_smallest_radius_after_birth():
    # a base that first closes in on the axis and then opens as the jet rises, up to r_search
    r = np.array([0.3, 0.2, 0.1, 0.05, 0.08, 0.2, 0.5, 1.4, 1.5])
    assert growth(r, r_cap=1.35) == (3, 6, "r_search")
    assert growth(np.array([0.1, 0.2, 0.15, 0.3]), r_cap=1.35) == (0, 1, "decrease")
    assert growth(np.array([0.3, 0.2, 0.1]), r_cap=1.35) == (2, 2, "end of cycle")
    assert growth(np.array([1.4, 1.5]), r_cap=1.35)[1] < 0
    # a jet that rises from birth and falls back below its birth radius: the fall, here down to the
    # r_j = 0 of the row at which the jet has gone, does not move the reference
    r = np.array([0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.15, 0.05, 0.003, 0.0])
    assert growth(r, r_cap=1.35) == (0, 6, "decrease")
    # an uptick of 0.3 % while the base closes does not start the growth
    r = np.array([0.3, 0.2, 0.2006, 0.1, 0.05, 0.08, 0.2, 0.5, 1.4, 1.5])
    assert growth(r, r_cap=1.35) == (4, 7, "r_search")


def test_growth_tolerates_small_downticks_of_the_jet_radius():
    # a rebuild of the mesh moves the refined base by a little: r_j falls by 1e-4 of itself and
    # rises on; the growth goes on to r_search
    r = np.geomspace(0.01, 1.5, 40)
    r[20] = r[19] * (1 - 1e-4)
    m = int(np.argmax(r >= 1.35))
    assert growth(r, r_cap=1.35) == (0, m - 1, "r_search")
    # a fall by more than 1 % below the running maximum ends it there, also in small steps
    r = np.geomspace(0.01, 1.5, 40)
    r[20] = r[19] * 0.98
    assert growth(r, r_cap=1.35) == (0, 19, "decrease")
    r = np.concatenate([np.geomspace(0.01, 0.5, 20), 0.5 * 0.996 ** np.arange(1, 6)])
    assert growth(r, r_cap=1.35) == (0, 19, "decrease")
    # downticks within the tolerance at the end of the cycle stop at the largest r_j
    r = np.concatenate([np.geomspace(0.01, 0.5, 20), [0.4999, 0.4998]])
    assert growth(r, r_cap=1.35) == (0, 19, "end of cycle")


# --- a synthetic run directory -----------------------------------------------------------------

COLUMNS = ("step", "cycle", "time", "cavity", "jet_born", "r_j", "Q_j", "V_j", "We_j",
           "axis_curvature", "h_near", "h_focus", "volume", "beta_fit", "t0_est", "alpha_est",
           "t0_source")


def _wall_angle(beta, r_b, dbeta, r):
    """Semi-angle of the wall at ``r``: ``beta`` at the base, rising by ``dbeta`` in ``ln r`` to
    ``r = 2``."""
    return beta + dbeta * np.log(r / r_b) / math.log(2.0 / r_b)


def _jet_profile(beta, r_b, field, dbeta=0.0):
    """Free surface with a bump of height ``r_b`` on the axis, its base at ``r = r_b``, and a
    wall of semi-angle ``beta`` beyond (straight unless ``dbeta``), with velocities only on the
    wall: the gradient of ``r_p^nu P_nu(-cos theta)`` about its virtual apex (``"cone"``) or
    ``(-1/r, 0)``."""
    z_b = 5.0
    r1 = np.linspace(0.0, r_b, 60)[:-1]
    z1 = z_b + r_b * (1 - (r1 / r_b) ** 2) ** 2  # z''(0) = -4/r_b
    r2 = np.geomspace(r_b, 2.0, 3000)
    slope = 1.0 / np.tan(_wall_angle(beta, r_b, dbeta, r2))
    z2 = z_b + np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(r2))])
    r, z = np.concatenate([r1, r2]), np.concatenate([z1, z2])
    ur, uz = np.zeros_like(r), np.zeros_like(r)
    z_apex = z_b - r_b / math.tan(beta)
    nu = nu_of_beta(beta)

    def phi(x, y):
        rho = math.hypot(x, y - z_apex)
        return rho**nu * legendre_p(nu, -(y - z_apex) / rho)

    for i in range(r1.size, r.size):
        if field == "sink":
            ur[i] = -1.0 / r[i]
        else:
            h = 1e-6 * math.hypot(r[i], z[i] - z_apex)
            ur[i] = (phi(r[i] + h, z[i]) - phi(r[i] - h, z[i])) / (2 * h)
            uz[i] = (phi(r[i], z[i] + h) - phi(r[i], z[i] - h)) / (2 * h)
    return np.column_stack([r, z, ur, uz])


def _write_run(path, *, beta_deg, nu, r_b=1e-3, field="cone", dbeta_deg=0.0):
    """A collapse run with one cycle: a cavity, then a jet born at ``t = 1`` with ``r_j = r_b``,
    growing as ``r_j = (t - t0)^alpha`` with ``alpha = 1/(2 - nu)`` and ``q_j = 2 r_j^nu``."""
    alpha = 1 / (2 - nu)
    t_birth = 1.0
    t0 = t_birth - r_b ** (1 / alpha)
    rows = []
    for k, t in enumerate(np.linspace(0.0, t_birth, 200, endpoint=False)):
        kappa = 40.0 * math.exp(-(((t - 0.9) / 0.05) ** 2))  # largest at t = 0.9
        rows.append([k, 0, t, int(k > 0), 0, 0.0, math.nan, math.nan, math.nan, kappa, 1e-4,
                     1e-4, 1.0, math.radians(60.0), 0.95 if t > 0.5 else math.nan,
                     0.6 if t > 0.5 else math.nan, 1 if t > 0.5 else 0])
    for r in np.geomspace(r_b, 1.45, 300):
        q = 2.0 * r**nu
        rows.append([len(rows), 0, t0 + r ** (1 / alpha), 1, 1, r, math.pi * r * q, q / r,
                     q**2 / r, -1.0, 1e-4, 1e-4, 1.0, math.radians(70.0), math.nan, math.nan, 0])
    rows[200][2] = t_birth  # exactly, against rounding in t0 + r_b^(1/alpha)
    path.mkdir()
    np.savetxt(path / "collapse.txt", np.array(rows), delimiter="\t",
               header="\t".join(COLUMNS), comments="# ")
    events = [{"event": "cavity", "time": 0.005, "cycle": 0},
              {"event": "jet-born", "time": t_birth, "cycle": 0, "max_curvature": 40.0,
               "max_curvature_time": 0.9, "t0_apex": 0.95, "alpha_apex": 0.6, "t_apex_est": 0.99}]
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (path / "run_status.json").write_text(json.dumps({"state": "completed"}))
    (path / "parameters.json").write_text(json.dumps(
        {"case": {"case_id": "synthetic"}, "dimensionless": {"Gamma": 1.0, "Oh": 0.04, "a0": 0.5}}))
    (path / "collapse_settings.json").write_text(json.dumps({"r_search": 1.5, "rise_tol": 1e-6}))
    (path / "profiles").mkdir()
    header = json.dumps({"time": t_birth, "step": 200, "reason": "jet-born", "r_j": r_b})
    np.savetxt(path / "profiles" / "profile_000000.txt",
               _jet_profile(math.radians(beta_deg), r_b, field, math.radians(dbeta_deg)),
               header=header + "\nr z u_r u_z", comments="# ")


def test_analysis_of_a_cone_jet(tmp_path):
    beta = math.radians(38.4)
    nu = nu_of_beta(beta)
    _write_run(tmp_path / "run", beta_deg=38.4, nu=nu)
    summary = analyse(tmp_path / "run")
    json.dumps(clean(summary), allow_nan=False)
    (jet,) = summary["jets"]
    assert jet["status"] == "ok" and jet["birth_observed"]
    assert jet["growth"]["reference"] == "birth" and jet["growth"]["ended_by"] == "r_search"
    assert summary["max_V_j"] == jet["growth"]["max_V_j"] > 0
    # time-free: q_j ~ r_j^nu, alpha = 1/(2 - nu)
    tf = jet["time_free"]
    assert tf["slope_q_j"] == pytest.approx(nu, abs=1e-9)
    assert tf["alpha_free"] == pytest.approx(alpha_of_beta(beta), abs=1e-9)
    # the cone at inception: window from 10 r_b, the angle, the kinematics of phi = 0 on the cone
    inc = jet["inception"]
    assert inc["status"] == "ok" and inc["profile_reason"] == "jet-born"
    assert inc["r_j"] == pytest.approx(1e-3, rel=1e-2)  # base refined next to a slope kink
    assert inc["r_lo"] == pytest.approx(10 * inc["r_j"], rel=1e-12)
    assert inc["r_hi"] == pytest.approx(1.0)
    assert inc["probe"]["angle_deg"] == pytest.approx([38.4] * 33, abs=1e-6)
    assert inc["beta_deg"] == pytest.approx(38.4, abs=1e-6)
    assert inc["angle_spread_deg"] < 1e-6
    kin = inc["wall_kinematics"]
    for part in ("window", "inner_half", "outer_half"):
        assert kin[part]["cos_normal"] == pytest.approx(1.0, abs=1e-6)
        assert kin[part]["speed_slope"] == pytest.approx(nu - 1, abs=1e-6)
    assert kin["expected"]["cone"]["speed_slope"] == pytest.approx(nu - 1, abs=1e-12)
    assert jet["alpha_cone"] == pytest.approx(alpha_of_beta(beta), abs=1e-6)
    assert jet["beta_below_taylor"] is True
    assert jet["alpha_line_sink"] == 0.5 and jet["alpha_inertio_capillary"] == pytest.approx(2 / 3)
    # the per-step angle is kept only as a diagnostic
    assert jet["beta_fit_median_deg_diagnostic"] == pytest.approx(70.0)
    # singular time from the apex, inside [time of largest apex curvature, birth]
    st = jet["singular_time"]
    assert st["t0_apex"] == 0.95 and st["source"] == "jet-born" and st["in_bracket"]
    assert st["max_axis_curvature_time"] == 0.9
    # time fits: the joint fit finds the origin; fixed at birth the exponent is biased
    joint = jet["time_fits"]["joint"]
    assert joint["alpha"] == pytest.approx(alpha_of_beta(beta), abs=1e-4)
    assert joint["t0_fit"] == pytest.approx(1.0 - 1e-3 ** (1 / alpha_of_beta(beta)), abs=1e-7)
    assert not joint["t0_at_bound"]
    assert jet["time_fits"]["fixed_origin"]["reference"]["alpha"] < alpha_of_beta(beta) - 0.01


def test_cone_window_starts_above_the_resolution_at_inception(tmp_path):
    # the free-surface spacing h_near at birth, 5e-3, exceeds r_j = 1e-3 and the apex radius of
    # curvature r_j/4: it sets the inner edge of the window
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    tab[200, 10] = 5e-3
    _rewrite(run, header, tab)
    inc = analyse(run)["jets"][0]["inception"]
    assert inc["status"] == "ok" and inc["h_near"] == 5e-3
    assert inc["r_lo"] == pytest.approx(10 * 5e-3, rel=1e-12)
    assert inc["beta_deg"] == pytest.approx(38.4, abs=1e-6)


def test_distant_profile_cannot_establish_inception_cone(tmp_path):
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    for profile in (run / "profiles").glob("*.txt"):
        lines = profile.read_text().splitlines()
        metadata = json.loads(lines[0].lstrip("# "))
        metadata["time"] += 1.0
        metadata["reason"] = "snapshot"
        profile.write_text("# " + json.dumps(metadata) + "\n" + "\n".join(lines[1:]) + "\n")
    jet = analyse(run)["jets"][0]
    assert jet["inception"]["status"] == "profile too far from growth reference"
    assert jet["alpha_cone"] is None and jet["beta_below_taylor"] is None


def test_analysis_of_a_line_sink_jet(tmp_path):
    # q_j constant and a horizontal inflow on a 60 deg wall: every law is reported, none chosen
    _write_run(tmp_path / "run", beta_deg=60.0, nu=0.0, field="sink")
    (jet,) = analyse(tmp_path / "run")["jets"]
    assert jet["time_free"]["alpha_free"] == pytest.approx(0.5, abs=1e-9)
    assert jet["time_free"]["laws"]["line_sink"]["log_rms"] < 1e-9
    assert jet["time_free"]["We_j"]["rises_as_r_j_falls"]
    kin = jet["inception"]["wall_kinematics"]["window"]
    assert kin["cos_normal"] == pytest.approx(0.5, abs=1e-9)
    assert kin["speed_slope"] == pytest.approx(-1.0, abs=1e-9)
    assert jet["alpha_cone"] == pytest.approx(alpha_of_beta(math.radians(60.0)), abs=1e-6)
    assert jet["beta_below_taylor"] is False
    assert jet["time_free"]["laws"]["cone"]["log_rms"] > 0.1


def test_analysis_of_a_restart_without_a_growing_jet(tmp_path):
    run = tmp_path / "run"
    _write_run(run, beta_deg=45.0, nu=0.4)
    tab = np.loadtxt(run / "collapse.txt", ndmin=2)
    # cycle 0 begins born at the first row with a base closing in on the axis; cycle 1 has no jet
    shrink = tab[200:260].copy()
    shrink[:, 5] = np.geomspace(0.3, 0.01, 60)
    shrink[:, 6:8] *= -1  # downward flux through the closing base
    cavity = tab[:50].copy()
    cavity[:, 1], cavity[:, 2] = 1, shrink[-1, 2] + np.linspace(0.01, 0.5, 50)
    with (run / "collapse.txt").open() as fh:
        header = fh.readline()[2:].strip()
    np.savetxt(run / "collapse.txt", np.vstack([shrink, cavity]), delimiter="\t", header=header,
               comments="# ")
    summary = analyse(run)
    (jet,) = summary["jets"]
    assert jet["cycle"] == 0 and not jet["birth_observed"] and jet["t_birth"] is None
    assert jet["status"] == "r_j does not grow after birth"
    assert jet["inception"] == {"status": "r_j does not grow after birth"}
    assert jet["growth"]["reference"] == "r_j minimum"
    assert jet["growth"]["ended_by"] == "end of cycle"
    assert math.isnan(jet["growth"]["max_V_j"])  # no flux into a jet
    assert math.isnan(summary["max_V_j"])
    assert summary["t_birth"] is None
    (other,) = summary["cycles_without_jet"]
    assert other["cycle"] == 1
    assert other["max_axis_curvature"] == pytest.approx(cavity[:, 9].max())


def _table(run):
    """Header and rows of the run's ``collapse.txt``."""
    with (run / "collapse.txt").open() as fh:
        header = fh.readline()[2:].strip()
    return header, np.loadtxt(run / "collapse.txt", ndmin=2)


def _rewrite(run, header, rows, events=None):
    np.savetxt(run / "collapse.txt", rows, delimiter="\t", header=header, comments="# ")
    if events is not None:
        (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def _events(run):
    return [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("r_gone", [0.0, 5e-5])
@pytest.mark.parametrize("next_cycle", ["table", "event"])
def test_analysis_of_a_jet_that_falls_back_before_r_search(tmp_path, r_gone, next_cycle):
    # the jet rises from birth to r_j = 0.5 and falls back; the driver writes the step at which it
    # has gone (the axis is the lowest point again) with jet_born still set and r_j = 0, or the
    # radius of a flattened bump, and then starts a new cycle, which the run may not reach
    run = tmp_path / "run"
    beta = math.radians(38.4)
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(beta))
    header, tab = _table(run)
    tab = tab[(tab[:, 4] == 0) | (tab[:, 5] <= 0.5)]
    fall = np.repeat(tab[-1:], 20, axis=0)
    fall[:, 2] = tab[-1, 2] + np.linspace(0.01, 0.2, 20)
    fall[:, 5] = np.geomspace(0.4, 1e-4, 20)
    fall[:, 6:8] = [-1.0, -10.0]  # Q_j and V_j: flux out of the jet through the closing base
    gone = fall[-1].copy()
    gone[2] += 0.01
    gone[5], gone[6:9] = r_gone, math.nan
    if r_gone > 0:  # a flattened bump can carry a small flux, and V_j = Q_j/(pi r_j^2) is large
        gone[6] = 1e-6
        gone[7] = gone[6] / (math.pi * r_gone**2)
        gone[8] = gone[7] ** 2 * r_gone
    cavity = tab[:50].copy()
    cavity[:, 1], cavity[:, 2] = 1, gone[2] + np.linspace(0.01, 0.5, 50)
    rows = [tab, fall, gone[None, :]] + ([cavity] if next_cycle == "table" else [])
    events = _events(run) + [{"event": "cavity", "time": gone[2], "cycle": 1}]
    _rewrite(run, header, np.vstack(rows), events)
    summary = analyse(run)
    (jet,) = summary["jets"]
    assert jet["status"] == "ok"
    assert jet["growth"]["reference"] == "birth" and jet["growth"]["ended_by"] == "decrease"
    assert jet["growth"]["r_j_end"] == pytest.approx(tab[-1, 5])
    assert jet["min_r_j_after_birth"] == pytest.approx(1e-4)
    assert jet["time_free"]["alpha_free"] == pytest.approx(alpha_of_beta(beta), abs=1e-9)
    assert len(summary["cycles_without_jet"]) == (next_cycle == "table")
    # the row at which the jet has gone is no part of it, for the run-level speed too
    assert summary["max_V_j"] == pytest.approx(jet["growth"]["max_V_j"])


def test_run_level_jet_speed_is_that_of_the_first_jet(tmp_path):
    # as in collapse_table's summary: the first jet born within the record, not a faster jet of a
    # later cycle
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    later = tab.copy()
    later[:, 1], later[:, 2] = 1, tab[-1, 2] + 0.01 + tab[:, 2]
    later[:, 7] *= 10.0  # V_j
    _rewrite(run, header, np.vstack([tab, later]),
             _events(run) + [{"event": "cavity", "time": later[0, 2], "cycle": 1}])
    summary = analyse(run)
    first, second = summary["jets"]
    assert second["growth"]["max_V_j"] == pytest.approx(10 * first["growth"]["max_V_j"])
    assert summary["max_V_j"] == pytest.approx(first["growth"]["max_V_j"])


def test_rows_at_which_the_jet_has_gone_without_the_column_cycle(tmp_path):
    # tables without the column cycle keep jet_born set after the first birth: the row at which
    # the jet has gone and the new cavity after it, with r_j = 0, are no part of the jet
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    tab = tab[(tab[:, 4] == 0) | (tab[:, 5] <= 0.5)]
    fall = np.repeat(tab[-1:], 20, axis=0)
    fall[:, 2] = tab[-1, 2] + np.linspace(0.01, 0.2, 20)
    fall[:, 5] = np.geomspace(0.4, 1e-4, 20)
    fall[:, 6:8] = [-1.0, -10.0]
    cavity = tab[:50].copy()
    cavity[:, 2] = fall[-1, 2] + np.linspace(0.01, 0.5, 50)
    cavity[:, 4], cavity[:, 5], cavity[:, 6:9] = 1, 0.0, math.nan
    columns = header.split("\t")
    _rewrite(run, "\t".join(c for c in columns if c != "cycle"),
             np.delete(np.vstack([tab, fall, cavity]), columns.index("cycle"), axis=1))
    (jet,) = analyse(run)["jets"]
    assert jet["cycle"] is None and jet["status"] == "ok"
    assert jet["min_r_j_after_birth"] == pytest.approx(1e-4)


def _set_event(run, **fields):
    """Replace the fields of the ``jet-born`` event; a field given as None is removed."""
    events = _events(run)
    born = next(e for e in events if e["event"] == "jet-born")
    for k, v in fields.items():
        if v is None:
            born.pop(k, None)
        else:
            born[k] = v
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_singular_time_is_the_estimate_the_jet_born_event_carries(tmp_path):
    # the table holds apex-based estimates (t0 = 0.95) up to the row before birth
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    st = analyse(run)["jets"][0]["singular_time"]
    assert st["t0_apex"] == 0.95 and st["source"] == "jet-born" and st["in_bracket"]
    # an estimate after the reference instant lies outside the bracket
    _set_event(run, t0_apex=1.05)
    st = analyse(run)["jets"][0]["singular_time"]
    assert st["source"] == "jet-born" and st["bracket"] == [0.9, 1.0] and st["in_bracket"] is False
    # an event without an estimate (a run that predates the field, or a driver that discarded it on
    # every row without a trough apex): the table's last one
    for missing in (math.nan, None):
        _set_event(run, t0_apex=missing, alpha_apex=missing)
        st = analyse(run)["jets"][0]["singular_time"]
        assert st["t0_apex"] == 0.95 and st["alpha_apex"] == 0.6 and st["source"] == "table"
        assert st["in_bracket"]
    # a bump growing on the axis before birth ends the apex series but keeps its estimate
    header, tab = _table(run)
    bump = tab.copy()
    bump[197:200, 9], bump[197:200, 5] = -1.0, 2e-4  # z''(0) < 0; the base is off the axis
    bump[197:200, 14:17] = math.nan, math.nan, 0
    _rewrite(run, header, bump)
    st = analyse(run)["jets"][0]["singular_time"]
    assert st["t0_apex"] == 0.95 and st["source"] == "table"
    # a new apex series after it discards it
    bump[199, 9], bump[199, 5] = 100.0, 0.0
    _rewrite(run, header, bump)
    st = analyse(run)["jets"][0]["singular_time"]
    assert math.isnan(st["t0_apex"]) and st["source"] is None and st["in_bracket"] is None
    # unless that trough apex is broader than the cap of the local scale, which the driver does not
    # sample: h_axis elements_per_scale = 0.4
    params = json.loads((run / "parameters.json").read_text())
    (run / "parameters.json").write_text(json.dumps(params | {"mesh": {"h_axis": 0.05}}))
    bump[199, 9] = 2.0
    _rewrite(run, header, bump)
    summary = analyse(run)
    assert summary["options"]["ell_cap"] == pytest.approx(0.4)
    assert summary["jets"][0]["singular_time"]["t0_apex"] == 0.95
    # not from a row in another episode than the birth
    _rewrite(run, header + "\tepisode", np.column_stack([tab, tab[:, 2] >= 1.0]))
    assert math.isnan(analyse(run)["jets"][0]["singular_time"]["t0_apex"])
    _rewrite(run, header, tab)
    # an estimate before the largest trough curvature lies outside the bracket
    _set_event(run, t0_apex=0.5, alpha_apex=0.6)
    st = analyse(run)["jets"][0]["singular_time"]
    assert st["source"] == "jet-born" and st["bracket"] == [0.9, 1.0] and st["in_bracket"] is False


@pytest.mark.parametrize("birth_in_crest", [False, True])
@pytest.mark.parametrize("episodes", ["recorded", "reconstructed"])
def test_bracket_starts_at_the_trough_of_the_collapse(tmp_path, episodes, birth_in_crest):
    # crest, a sharper trough that does not collapse, crest, then the trough that launches the jet,
    # with a bump on the axis just before birth whose |z''(0)| exceeds both troughs; a run without
    # the column episode is segmented again from z_axis. A jet born in the crest after that trough
    # takes the trough before its episode.
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    t = tab[:, 2]
    episode = np.searchsorted([0.2, 0.5, 0.7] + [0.98] * birth_in_crest, t, side="right")
    tab[:, 9] = np.where((episode == 1) & (t < 0.4), 80.0 * np.exp(-((t - 0.35) / 0.05) ** 2),
                         tab[:, 9])
    tab[(t >= 0.97) & (t < 1.0), 9] = -100.0
    events = _events(run)
    if episodes == "recorded":
        header, column = header + "\tepisode", episode
        events += [{"event": "trough", "time": t0, "episode": k, "h_crest": 0.1,
                    "t_crest": t0 - 0.1, "partial": False} for t0, k in ((0.2, 1), (0.7, 3))]
    else:
        params = json.loads((run / "parameters.json").read_text())
        params["dimensionless"]["depth"] = 2.0
        (run / "parameters.json").write_text(json.dumps(params))
        header, column = header + "\tz_axis", 2.0 + np.where(episode % 2, -0.1, 0.1)
    _rewrite(run, header, np.column_stack([tab, column]), events)
    # the driver before this rule carried the bump's |z''(0)| in the event; it is not used
    _set_event(run, max_curvature=100.0, max_curvature_time=0.97)
    st = analyse(run)["jets"][0]["singular_time"]
    assert st["max_axis_curvature"] == pytest.approx(40.0, rel=1e-3)
    assert st["max_axis_curvature_time"] == pytest.approx(0.9, abs=5e-3)
    # without a trough on the axis in that episode there is no bracket, not an earlier trough
    tab[(episode == 3) & (tab[:, 9] > 0), 9] = -1.0
    _rewrite(run, header, np.column_stack([tab, column]))
    st = analyse(run)["jets"][0]["singular_time"]
    assert math.isnan(st["max_axis_curvature"]) and st["in_bracket"] is None


@pytest.mark.parametrize("trough_start, partial", [(None, False), (0.7, False), (0.92, False),
                                                    (0.7, True)])
def test_bracket_of_a_record_restarted_after_the_trough_maximum(tmp_path, trough_start, partial):
    # restarted at t = 0.95, after the largest trough curvature (40 at t = 0.9) and before birth:
    # the jet-born event carries that maximum across the restart, without episodes or within the
    # trough in progress at the restart, but not from before that trough or from a trough whose
    # start is unknown
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    tab = tab[tab[:, 2] >= 0.95 - 1e-9]
    events = [e for e in _events(run) if e["event"] == "jet-born"]
    # a driver that restored a dump marked as holding the trough maximum
    restart = {"event": "restart", "time": tab[0, 2], "max_curvature_reset": False}
    if trough_start is not None:
        header, tab = header + "\tepisode", np.column_stack([tab, np.full(len(tab), 3.0)])
        restart |= {"episode": 3, "episode_kind": -1, "episode_start": trough_start,
                    "episode_partial": partial, "h_crest": 0.1, "t_crest": 0.6}
    _rewrite(run, header, tab, [restart] + events)
    st = analyse(run)["jets"][0]["singular_time"]
    if trough_start == 0.92 or partial:
        assert st["max_axis_curvature_source"] == "table"
        assert st["max_axis_curvature_time"] == tab[0, 2]
    else:
        assert st["max_axis_curvature_source"] == "jet-born"
        assert st["max_axis_curvature"] == 40.0 and st["bracket"] == [0.9, 1.0]
        assert st["in_bracket"]


@pytest.mark.parametrize("reset_field", ["absent", "no-restart-event"])
def test_bracket_ignores_a_maximum_carried_across_a_legacy_restart(tmp_path, reset_field):
    # restarted at t = 0.95 inside the trough that began at 0.7, by a driver that predates
    # max_curvature_reset: it restored the dump's value as it was, here |z''(0)| = 100 of a bump at
    # t = 0.93, which an older driver kept over the cycle; the rows' maximum is kept
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    tab = tab[tab[:, 2] >= 0.95 - 1e-9]
    events = [e for e in _events(run) if e["event"] == "jet-born"]
    events[0] |= {"max_curvature": 100.0, "max_curvature_time": 0.93}
    if reset_field == "absent":
        header, tab = header + "\tepisode", np.column_stack([tab, np.full(len(tab), 3.0)])
        events.insert(0, {"event": "restart", "time": tab[0, 2], "episode": 3,
                          "episode_kind": -1, "episode_start": 0.7, "episode_partial": False,
                          "h_crest": 0.1, "t_crest": 0.6})
    _rewrite(run, header, tab, events)
    st = analyse(run)["jets"][0]["singular_time"]
    assert st["max_axis_curvature_source"] == "table"
    assert st["max_axis_curvature_time"] == tab[0, 2] and st["bracket"] == [tab[0, 2], 1.0]
    assert "max_curvature_reset" in st["max_axis_curvature_note"]


def test_cycle_without_a_jet_reports_its_largest_trough_curvature(tmp_path):
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    # a second cycle without a jet: a trough, a bump on the axis with a larger |z''(0)|, and a row
    # where the surface overturns next to the axis and z''(0) is not measured
    other = tab[:100].copy()
    other[:, 1], other[:, 2] = 1, tab[-1, 2] + np.linspace(0.01, 1.0, 100)
    other[:, 3], other[:, 4] = 1, 0
    other[:, 9] = 3.0 * np.exp(-((np.arange(100) - 40) / 10.0) ** 2)
    other[70, 9], other[80, 9] = -9.0, math.nan
    _rewrite(run, header, np.vstack([tab, other]))
    (cycle,) = analyse(run)["cycles_without_jet"]
    assert cycle["cycle"] == 1
    assert cycle["t_start"] == other[0, 2] and cycle["t_end"] == other[-1, 2]
    assert cycle["max_axis_curvature"] == 3.0 and cycle["max_axis_curvature_time"] == other[40, 2]


def test_a_jet_over_a_short_range_of_r_j_is_not_ok(tmp_path):
    # 0.13 decades of r_j: the slopes are fitted and reported, but the jet is not labelled ok
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    summary = analyse(run, rj_window=(0.01, 0.0135))
    (jet,) = summary["jets"]
    assert summary["options"]["min_decades"] == 0.5
    assert jet["time_free"]["decades"] < 0.5 and math.isfinite(jet["time_free"]["alpha_free"])
    assert jet["status"] == "r_j spans fewer than 0.5 decades"
    assert analyse(run, rj_window=(0.01, 0.0135), min_decades=0.1)["jets"][0]["status"] == "ok"


def test_a_jet_whose_fit_samples_fill_too_few_bins_is_not_ok(tmp_path):
    # fit samples at both ends of the window only: a decade of r_j, but 2 of 12 bins filled, so the
    # time-free test has no slopes
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    header, tab = _table(run)
    gap = (tab[:, 4] > 0) & (tab[:, 5] > 3.3e-3) & (tab[:, 5] < 2.7e-2)
    _rewrite(run, header, tab[~gap])
    (jet,) = analyse(run)["jets"]
    assert jet["fit_points"] >= 6 and jet["time_free"]["decades"] > 0.5
    assert jet["time_free"]["status"] == "fewer than 3 filled bins"
    assert jet["status"] == "fewer than 3 filled bins"


def test_analysis_command_line(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    _write_run(run, beta_deg=38.4, nu=nu_of_beta(math.radians(38.4)))
    out = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["collapse_analysis.py", str(run), "--out", str(out),
                                      "--rj-window", "0.01", "0.0135", "--min-decades", "0.1",
                                      "--no-plot"])
    assert main() == 0
    capsys.readouterr()
    summary = json.loads((out / "synthetic_summary.json").read_text())
    assert summary["options"]["min_decades"] == 0.1
    assert summary["options"]["rj_window"] == [0.01, 0.0135]
    assert summary["jets"][0]["status"] == "ok"
    assert not list(out.glob("*.png"))


def test_cone_angle_range_over_a_curved_wall(tmp_path):
    # the wall angle rises by about 0.9 deg across the window, within angle_tol = 2 deg: beta is
    # fitted over the whole window and alpha_cone_range spans the alphas of the probed angles
    run = tmp_path / "run"
    _write_run(run, beta_deg=40.0, nu=0.4, field="sink", dbeta_deg=1.5)
    inc = analyse(run)["jets"][0]["inception"]
    assert inc["status"] == "ok" and inc["r_hi"] == pytest.approx(1.0)
    expected = np.degrees(_wall_angle(math.radians(40.0), 1e-3, math.radians(1.5),
                                      np.array([inc["r_lo"], 1.0])))
    probed = np.array(inc["probe"]["angle_deg"])
    assert [probed.min(), probed.max()] == pytest.approx(expected, abs=0.01)
    assert inc["angle_spread_deg"] == pytest.approx(probed.max() - probed.min(), abs=1e-12)
    assert inc["alpha_cone_range"] == pytest.approx(
        [alpha_of_beta(math.radians(probed.min())), alpha_of_beta(math.radians(probed.max()))],
        abs=1e-12)
    assert inc["alpha_cone_range"][1] - inc["alpha_cone_range"][0] > 2e-3


def test_a_small_downtick_of_r_j_keeps_every_fit_sample(tmp_path):
    run = tmp_path / "run"
    nu = nu_of_beta(math.radians(38.4))
    _write_run(run, beta_deg=38.4, nu=nu)
    header, tab = _table(run)
    inside = np.flatnonzero((tab[:, 4] > 0) & (tab[:, 5] >= 3e-3) & (tab[:, 5] <= 3e-2))
    k = inside[inside.size // 2]
    tab[k, 5] = tab[k - 1, 5] * (1 - 1e-4)  # as after a rebuild of the mesh
    _rewrite(run, header, tab)
    summary = analyse(run)
    assert summary["options"]["growth_max_drop"] == 0.01
    (jet,) = summary["jets"]
    assert jet["status"] == "ok" and jet["growth"]["ended_by"] == "r_search"
    assert jet["fit_points"] == inside.size


def test_fit_samples_need_flux_into_the_jet(tmp_path):
    run = tmp_path / "run"
    nu = nu_of_beta(math.radians(38.4))
    _write_run(run, beta_deg=38.4, nu=nu)
    header, tab = _table(run)
    inside = np.flatnonzero((tab[:, 4] > 0) & (tab[:, 5] >= 3e-3) & (tab[:, 5] <= 3e-2))
    tab[inside[[3, 10, 20]], 6:8] *= -1  # flux out of the jet
    tab[inside[[5, 15]], 6:9] = math.nan  # flux not found
    _rewrite(run, header, tab)
    (jet,) = analyse(run)["jets"]
    assert jet["status"] == "ok" and jet["growth"]["ended_by"] == "r_search"
    assert jet["fit_points"] == inside.size - 5
    assert jet["excluded"] == {"flux_not_found": 2, "flux_not_positive": 3}
    assert jet["time_free"]["alpha_free"] == pytest.approx(1 / (2 - nu), abs=1e-9)
