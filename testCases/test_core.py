"""Software-contract tests for the parameters, the J0 polynomial and the linear comparators.

These tests check implementation contracts and classical limits; they are not verification of the
finite-element solver, which lives in ``verificationCases``.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0, jn_zeros

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "verificationCases" / "linear_theory"))
sys.path.insert(0, str(ROOT / "postProcess"))

import faraday_linear as fl  # noqa: E402
import initial_value as iv  # noqa: E402
from faraday_singularity.bessel import j0_polynomial, terms_for_radius  # noqa: E402
from faraday_singularity.parameters import (CaseParameters, Container, Fluid,  # noqa: E402
                                            Forcing, dimensional_reference, nondimensionalise)
from linear_fits import fit_damped_cosine, floquet_exponent  # noqa: E402

# 40-digit references (case C1: Oh = 0.04, Bo = 0, d = 2 pi, free-slip bottom, Omega = 2 omega0),
# from an mpmath evaluation of the closed-form dispersion relation and Hill determinant.
S_C1 = complex(-0.06845826023495448661653473, 0.9889116146345264107352514)
GAMMA_C_C1 = 0.277382155988667845740707


def test_parameters_groups():
    p = CaseParameters(fluid=Fluid(Oh=0.04, Bo=0.0), container=Container(mode=1, depth=2 * math.pi),
                       forcing=Forcing(Gamma=0.3, detuning=1.0))
    nd = nondimensionalise(p)
    assert nd.radius == pytest.approx(jn_zeros(1, 1)[0], rel=1e-14)
    assert nd.omega0 == pytest.approx(math.sqrt(math.tanh(2 * math.pi)), rel=1e-14)
    assert nd.Omega == pytest.approx(2 * nd.omega0, rel=1e-14)
    assert nd.forcing_period == pytest.approx(nd.mode_period / 2, rel=1e-14)
    ref = dimensional_reference(nd, density=1000.0, viscosity=1e-3, surface_tension=0.072)
    # Oh = mu sqrt(k/(rho sigma)) must be recovered from the k it implies
    assert 1e-3 * math.sqrt(ref["k"] / (1000.0 * 0.072)) == pytest.approx(0.04, rel=1e-12)


@pytest.mark.parametrize("bad", [dict(fluid=Fluid(Oh=0.0)), dict(forcing=Forcing(Gamma=-1.0)),
                                 dict(container=Container(mode=0))])
def test_parameters_reject_invalid(bad):
    with pytest.raises(ValueError):
        nondimensionalise(CaseParameters(**bad))


@pytest.mark.parametrize("x_max", [jn_zeros(1, 1)[0], jn_zeros(1, 2)[0]])
def test_j0_polynomial(x_max):
    x = np.linspace(0.0, x_max, 2001)
    err = np.max(np.abs(j0_polynomial(x, terms_for_radius(x_max)) - j0(x)))
    assert err < 1e-13


def test_lamb_limit_and_closed_form():
    for Oh in (1e-4, 1e-3):
        s = fl.normal_mode(Oh, 0.0, np.inf, "free-slip")
        assert s.real / (-2 * Oh) == pytest.approx(1.0, abs=5 * math.sqrt(Oh))
    s = np.array([0.3 + 1.1j, -0.05 + 2.0j, 1e-3 + 0.5j])
    assert np.allclose(fl.D_fun(s, 0.04, 2.0, "free-slip"), fl.D_free_slip_closed(s, 0.04, 2.0),
                       rtol=1e-12, atol=0)


def test_reference_values_C1():
    d = 2 * math.pi
    s = fl.normal_mode(0.04, 0.0, d, "free-slip")
    assert abs(s - S_C1) < 1e-13
    Omega = 2 * fl.omega0(0.0, d)
    assert fl.gamma_c(0.04, 0.0, d, "free-slip", Omega) == pytest.approx(GAMMA_C_C1, rel=1e-12)


def test_initial_value_response_matches_floquet_and_normal_mode():
    d, Oh = 2 * math.pi, 0.04
    Omega = 2 * fl.omega0(0.0, d)
    T = 2 * math.pi / Omega
    t = np.linspace(0.0, 30 * T, 30 * 64 + 1)
    s = fl.normal_mode(Oh, 0.0, d, "free-slip")
    # bulk vortical modes and the viscous memory make the fitted rate differ from the pure normal
    # mode by a few parts in 1e4 over this window
    fit = fit_damped_cosine(t, iv.response(t, Oh=Oh, Bo=0.0, d=d), t_start=4 * T,
                            omega_guess=s.imag, rate_guess=s.real)
    assert fit["s_real"] == pytest.approx(s.real, rel=2e-3)
    assert fit["s_imag"] == pytest.approx(s.imag, rel=5e-4)
    Gamma = 0.4
    fold = fl.fold_point(Omega, 20, Oh, 0.0, d, "free-slip", s)
    mu = fl.mu_KT(Gamma, Omega, 20, Oh, 0.0, d, "free-slip", fold=fold)
    Z = iv.response(t, Oh=Oh, Bo=0.0, d=d, Gamma=Gamma, Omega=Omega)
    assert floquet_exponent(t, Z, period=T, t_start=10 * T)["mu"] == pytest.approx(mu, abs=1e-5)


def test_bdf2_response_is_second_order():
    d, Oh = 2 * math.pi, 0.04
    T = 2 * math.pi / (2 * fl.omega0(0.0, d))
    t_end = 4 * T
    errs = []
    for n in (50, 100, 200):
        tb, Zb = iv.response_bdf2(T / n, 4 * n, Oh=Oh, Bo=0.0, d=d, N=32)
        Z = iv.response(tb, Oh=Oh, Bo=0.0, d=d, N=32)
        assert tb[-1] == pytest.approx(t_end, rel=1e-12)
        errs.append(np.max(np.abs(Zb - Z)))
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.15)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.15)


def _synthetic_run(tmp_path: Path, Gamma: float, a0: float, periods: float, n: int) -> Path:
    """A run directory whose amplitude is the exact BDF2 linear response over ``periods`` T_f."""
    import json

    d, Oh = 2 * math.pi, 0.04
    nd = nondimensionalise(CaseParameters(fluid=Fluid(Oh=Oh), container=Container(depth=d),
                                          forcing=Forcing(Gamma=Gamma)))
    dt = nd.forcing_period / n
    t, Z = iv.response_bdf2(dt, round(periods * n), Oh=Oh, Bo=0.0, d=d, Gamma=Gamma,
                            Omega=nd.Omega)
    run = tmp_path / f"G{Gamma}"
    run.mkdir()
    cols = ["time", "z_axis", "w_axis", "amp", "volume", "area", "kinetic_energy", "dissipation",
            "remesh_count"]
    data = np.zeros((t.size, len(cols)))
    data[:, 0], data[:, 1], data[:, 3] = t, d + a0 * Z, a0 * Z
    data[:, 4] = 1.0
    data[:, 5] = math.pi * nd.radius**2 * (1 + 1e-6)
    with (run / "diagnostics.txt").open("w") as fh:
        fh.write("# " + "\t".join(cols) + "\n")
        np.savetxt(fh, data, delimiter="\t")
    params = {"dimensionless": nd.to_dict() | {"a0": a0},
              "case": {"case_id": f"synthetic-G{Gamma}",
                       "time_stepping": {"steps_per_forcing_period": n},
                       "tolerances": {"newton": 1e-10}},
              "mesh": {"h_interface": 1.0, "h_axis": 1.0, "h_max": 1.0, "grading": 1.0},
              "time_stepping": {"dt": dt, "t_end": t[-1]}}
    (run / "parameters.json").write_text(json.dumps(params))
    (run / "run_status.json").write_text(json.dumps({"state": "completed", "remesh_count": 0}))
    return run


def test_comparison_pipeline_on_synthetic_runs(tmp_path):
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import compare_linear as cl

    out = tmp_path / "analysis"
    out.mkdir()
    free = cl.analyse(_synthetic_run(tmp_path, 0.0, -0.01, 12, 50), out, 4.0, None, False)
    assert abs(free["rate_rel_error_spatial"]) < 1e-9
    assert abs(free["frequency_rel_error_spatial"]) < 1e-9
    assert free["max_abs_diff_bdf2_over_a0"] < 1e-9
    assert free["fit_window"] == [free["fit_simulation"][k] for k in ("t_start", "t_end")]
    runs = [_synthetic_run(tmp_path, G, -0.001, 20, 50) for G in (0.2, 0.3)]
    forced = [cl.analyse(run, out, 4.0, None, False) for run in runs]
    for r in forced:
        assert abs(r["modes_mu_error_spatial"]) < 1e-8
        assert r["radius"] == pytest.approx(jn_zeros(1, 1)[0], rel=1e-14)
        assert r["newton_tolerance"] == 1e-10
        # the window the Floquet fits used, which all share: without --fit-end, the outputs from
        # 4 T_f to the run's end, cut to whole forcing periods
        window = [r["fit_simulation"]["t_start"], r["fit_simulation"]["t_end"]]
        assert r["fit_window"] == window
        for fit in ("fit_linear_response", "modes_fit_simulation", "modes_fit_linear_response",
                    "modes_fit_linear_response_bdf2"):
            assert [r[fit]["t_start"], r[fit]["t_end"]] == window
        T = 2 * math.pi / r["Omega"]
        assert 4 * T <= window[0] < 4 * T + r["dt"] * (1 + 1e-9)
        assert r["t_end"] - T < window[1] <= r["t_end"]
    (th,) = cl.thresholds(forced)
    assert th["problem"] == {"a0": -0.001, "radius": forced[0]["radius"]}
    assert th["fit_window"] == forced[0]["fit_window"]
    assert th["physics"]["Oh"] == 0.04
    # a --fit-end beyond the run's end fits only to that end
    clipped = cl.analyse(runs[0], out, 4.0, 40.0, False)
    assert clipped["fit_window"] == forced[0]["fit_window"]
    assert th["floquet_exact"] == pytest.approx(th["Gamma_c_exact"], rel=2e-3)


def test_thresholds_group_runs_fitted_over_one_window(tmp_path):
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import compare_linear as cl

    # runs stopped less than one forcing period apart (as analysed with --allow-incomplete) are
    # fitted over the same whole periods, so they form one threshold group
    out = tmp_path / "analysis"
    out.mkdir()
    forced = [cl.analyse(_synthetic_run(tmp_path, G, -0.001, periods, 50), out, 4.0, None, False)
              for G, periods in ((0.2, 20.1), (0.4, 20.9))]
    assert forced[0]["t_end"] != forced[1]["t_end"]
    assert forced[0]["fit_window"] == forced[1]["fit_window"]
    (th,) = cl.thresholds(forced)
    assert th["Gammas"] == [0.2, 0.4]
    assert th["fit_window"] == forced[0]["fit_window"]
    for name in ESTIMATORS:
        assert math.isfinite(th[name])
    assert th["floquet_exact"] == pytest.approx(GAMMA_C_C1, rel=1e-2)


def test_main_keeps_run_results_when_thresholds_fail(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import json

    import compare_linear as cl

    # two runs of one case at one Gamma, in two directories, whose amplitudes differ (as from two
    # solver revisions): that threshold group is ambiguous, but each run's comparison stands, and
    # so does the threshold of another group (here another a0) analysed in the same call
    runs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        runs.append(_synthetic_run(tmp_path / name, 0.3, -0.001, 12, 50))
    diag = runs[1] / "diagnostics.txt"
    header = diag.read_text().splitlines()[0]
    data = np.loadtxt(diag, comments="#")
    data[:, 3] *= np.exp(1e-3 * data[:, 0])
    with diag.open("w") as fh:
        fh.write(header + "\n")
        np.savetxt(fh, data, delimiter="\t")
    (tmp_path / "c").mkdir()
    runs += [_synthetic_run(tmp_path / "c", G, -0.002, 12, 50) for G in (0.2, 0.4)]

    out = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", ["compare_linear.py", *map(str, runs), "--out", str(out)])
    calls = []
    monkeypatch.setattr(cl, "plot", lambda results, stem: calls.append(
        ([r["case_id"] for r in results], stem)))
    assert cl.main() == 1
    case_ids = ["synthetic-G0.3", "synthetic-G0.3", "synthetic-G0.2", "synthetic-G0.4"]
    assert calls == [(case_ids, out / "linear_comparison")]
    summary = json.loads((out / "linear_comparison.json").read_text())
    assert [r["case_id"] for r in summary["runs"]] == case_ids
    # one linear-response file per run directory, holding that run's amplitude; a case analysed
    # from one directory keeps its own name
    files = [r["linear_response_file"] for r in summary["runs"]]
    assert len(set(files)) == len(files)
    assert files[2:] == ["synthetic-G0.2_linear_response.txt", "synthetic-G0.4_linear_response.txt"]
    for run, name in zip(runs, files):
        amp = cl.read_table(run / "diagnostics.txt")["amp"]
        np.testing.assert_allclose(np.loadtxt(out / name)[:, 1], amp, rtol=1e-12, atol=0)
    by_a0 = {e["problem"]["a0"]: e for e in summary["thresholds"]}
    assert len(by_a0) == len(summary["thresholds"]) == 2
    ambiguous, kept = by_a0[-0.001], by_a0[-0.002]
    assert ambiguous["Gammas"] == [0.3]
    assert all(ambiguous[name] is None for name in ESTIMATORS)
    # the runs share their case_id, so the error tells them apart by directory
    for run in runs[:2]:
        assert repr(str(run)) in ambiguous["error"]
    assert summary["thresholds_error"] == [ambiguous["error"]]
    assert "error" not in kept
    assert kept["Gammas"] == [0.2, 0.4]
    for name in ESTIMATORS:
        assert math.isfinite(kept[name])
    assert kept["floquet_exact"] == pytest.approx(GAMMA_C_C1, rel=1e-2)


MESH = {"h_interface": 0.05, "h_axis": 0.05, "h_max": 0.4, "grading": 0.3,
        "interface_points": 401, "remeshing": True}  # as run_case.py writes params["mesh"]


def _threshold_input(Gamma: float, mu: float, *, Oh: float = 0.04, Bo: float = 0.0,
                     depth: float = 2 * math.pi, Omega: float = 2.0, a0: float = -0.001,
                     radius: float | None = float(jn_zeros(1, 1)[0]), mesh: dict = MESH,
                     dt: float = 0.0625, Gamma_c: float = 0.3, case_id: str | None = None,
                     fit_window: tuple | None = (4 * math.pi, 20 * math.pi),
                     newton_tolerance: float | None = 1e-10) -> dict:
    """A forced-run result with the fields that ``thresholds`` reads; every estimator gives ``mu``.

    ``radius``, ``fit_window`` or ``newton_tolerance`` set to None omits the field, as in results
    stored before it was recorded.
    """
    out = {"case_id": case_id or f"G{Gamma:g}",
           "Gamma": Gamma, "Oh": Oh, "Bo": Bo, "depth": depth, "Omega": Omega, "a0": a0,
           "mesh": dict(mesh), "dt": dt, "Gamma_c": Gamma_c, "mu_floquet": mu,
           "fit_simulation": {"mu": mu}, "fit_linear_response": {"mu": mu},
           "modes_fit_simulation": {"mu": mu}, "modes_fit_linear_response": {"mu": mu}}
    if radius is not None:
        out["radius"] = radius
    if fit_window is not None:
        out["fit_window"] = list(fit_window)
    if newton_tolerance is not None:
        out["newton_tolerance"] = newton_tolerance
    return out


ESTIMATORS = ("single_simulation", "single_linear_response", "modes_simulation",
              "modes_linear_response", "floquet_exact")


@pytest.mark.parametrize("field, other", [("Oh", 0.08), ("Bo", 1.0), ("depth", 1.0),
                                          ("Omega", 2.1)])
def test_thresholds_do_not_mix_physics(field, other):
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import compare_linear as cl

    # same mesh and dt; mu = Gamma - 0.3 for one system and Gamma - 0.5 for the other. Mixed, the
    # first sign change would lie between Gamma = 0.3 (second system) and 0.4 (first system).
    first = [_threshold_input(G, G - 0.3) for G in (0.2, 0.4)]
    second = [_threshold_input(G, G - 0.5, Gamma_c=0.5, **{field: other}) for G in (0.3, 0.6)]
    entries = cl.thresholds(first + second)
    assert len(entries) == 2
    by_value = {e["physics"][field]: e for e in entries}
    for value, Gammas, Gamma_c in ((first[0][field], [0.2, 0.4], 0.3), (other, [0.3, 0.6], 0.5)):
        e = by_value[value]
        assert e["Gammas"] == Gammas
        assert e["Gamma_c_exact"] == Gamma_c
        for name in ESTIMATORS:
            assert e[name] == pytest.approx(Gamma_c, rel=1e-12)


@pytest.mark.parametrize("change", [
    {"a0": -0.002}, {"radius": float(jn_zeros(1, 2)[-1])}, {"radius": None},
    {"mesh": MESH | {"h_interface": 0.025}}, {"mesh": MESH | {"interface_points": 801}},
    {"mesh": MESH | {"remeshing": False}}, {"dt": 0.03125},
    {"fit_window": (4 * math.pi, 12 * math.pi)}, {"fit_window": None},
    {"newton_tolerance": 1e-6}, {"newton_tolerance": None},
], ids=["a0", "radius", "radius-missing", "h_interface", "interface_points", "remeshing", "dt",
        "fit_window", "fit_window-missing", "newton_tolerance", "newton_tolerance-missing"])
def test_thresholds_do_not_mix_problem_mesh_or_dt(change):
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import compare_linear as cl

    # same physics; one other part of the simulated problem, the fitting window (a run stopped
    # early, or run for fewer periods, fitted to its own end), the mesh, the time step or the
    # Newton tolerance differs.
    # mu = Gamma - 0.3 for one series and Gamma - 0.5 for the other: mixed, the first sign change
    # would lie between Gamma = 0.3 (second series) and 0.4 (first series).
    first = [_threshold_input(G, G - 0.3) for G in (0.2, 0.4)]
    second = [_threshold_input(G, G - 0.5, **change) for G in (0.3, 0.6)]
    forward = cl.thresholds(first + second)
    backward = cl.thresholds(second[::-1] + first[::-1])
    assert len(forward) == 2
    by_gammas = {tuple(e["Gammas"]): e for e in forward}
    assert by_gammas == {tuple(e["Gammas"]): e for e in backward}
    for runs, zero in ((first, 0.3), (second, 0.5)):
        e = by_gammas[tuple(r["Gamma"] for r in runs)]
        assert e["problem"] == {"a0": runs[0]["a0"], "radius": runs[0].get("radius")}
        assert e["fit_window"] == runs[0].get("fit_window")
        assert e["mesh"] == runs[0]["mesh"]
        assert e["dt"] == runs[0]["dt"]
        assert e["newton_tolerance"] == runs[0].get("newton_tolerance")
        # the positional form of the Phase 1a results
        m = runs[0]["mesh"]
        assert e["mesh_and_dt"] == [m["h_interface"], m["h_axis"], m["h_max"], m["grading"],
                                    runs[0]["dt"]]
        for name in ESTIMATORS:
            assert e[name] == pytest.approx(zero, rel=1e-12)
    a, b = forward
    fields = ("problem", "fit_window", "mesh", "dt", "newton_tolerance")
    assert [a[k] for k in fields] != [b[k] for k in fields]


def test_thresholds_reject_repeated_gamma():
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import compare_linear as cl

    # runs at one Gamma that differ in a field outside the key (a solver revision, say) give
    # different exponents; which of them brackets the zero would depend on the order of the inputs,
    # so that group gives no threshold. Another group (a finer mesh) keeps its threshold.
    runs = [_threshold_input(G, G - 0.3) for G in (0.2, 0.4)]
    rerun = _threshold_input(0.2, -0.05, case_id="G0.2-rerun")
    fine = [_threshold_input(G, G - 0.5, mesh=MESH | {"h_interface": 0.025}) for G in (0.3, 0.6)]
    for inputs in (runs + [rerun] + fine, fine + [rerun] + runs):
        entries = {tuple(e["Gammas"]): e for e in cl.thresholds(inputs)}
        assert len(entries) == 2
        e = entries[(0.2, 0.4)]
        assert re.fullmatch(r"repeated Gamma 0\.2 in threshold group physics=.*"
                            r"(?:'G0\.2', 'G0\.2-rerun'|'G0\.2-rerun', 'G0\.2').*", e["error"])
        for name in ESTIMATORS:
            assert e[name] is None
        assert "single_like_for_like_rel_error" not in e
        assert "modes_rel_error_vs_exact" not in e
        e = entries[(0.3, 0.6)]
        assert "error" not in e
        for name in ESTIMATORS:
            assert e[name] == pytest.approx(0.5, rel=1e-12)


def test_thresholds_collapse_exact_duplicates():
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import copy

    import compare_linear as cl

    # the same run passed twice is not ambiguous, also where an exponent is NaN
    runs = [_threshold_input(0.1, float("nan"))] + [_threshold_input(G, G - 0.3)
                                                   for G in (0.2, 0.4)]
    single = cl.thresholds(runs)
    assert single[0]["Gammas"] == [0.1, 0.2, 0.4]
    assert cl.thresholds(runs + copy.deepcopy(runs[:2])) == single
    assert cl.thresholds(runs[1:2] + runs) == single


def test_thresholds_skip_non_finite_endpoints():
    sys.path.insert(0, str(ROOT / "verificationCases"))
    import compare_linear as cl

    nan = float("nan")
    # below the multiplier collision mu_floquet is NaN; a fit without a subharmonic multiplier
    # also gives NaN. The bracket (0.1, 0.2) has a NaN end and must not be used.
    (e,) = cl.thresholds([_threshold_input(0.1, nan)]
                         + [_threshold_input(G, G - 0.25) for G in (0.2, 0.3, 0.4)])
    for name in ESTIMATORS:
        assert e[name] == pytest.approx(0.25, rel=1e-12)
    assert math.isfinite(e["single_like_for_like_rel_error"])
    assert math.isfinite(e["modes_rel_error_vs_exact"])
    # a sign change across a NaN has no finite bracket: no threshold rather than a non-finite one
    (e,) = cl.thresholds([_threshold_input(0.2, -0.05), _threshold_input(0.3, nan),
                          _threshold_input(0.4, 0.15)])
    for name in ESTIMATORS:
        assert e[name] is None
    assert "single_like_for_like_rel_error" not in e
    assert "modes_rel_error_vs_exact" not in e
