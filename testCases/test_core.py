"""Software-contract tests for the parameters, the J0 polynomial and the linear comparators.

These tests check implementation contracts and classical limits; they are not verification of the
finite-element solver, which lives in ``verificationCases``.
"""

from __future__ import annotations

import math
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


def _synthetic_run(tmp_path: Path, Gamma: float, a0: float, periods: int, n: int) -> Path:
    """A run directory whose amplitude is the exact BDF2 linear response."""
    import json

    d, Oh = 2 * math.pi, 0.04
    nd = nondimensionalise(CaseParameters(fluid=Fluid(Oh=Oh), container=Container(depth=d),
                                          forcing=Forcing(Gamma=Gamma)))
    dt = nd.forcing_period / n
    t, Z = iv.response_bdf2(dt, periods * n, Oh=Oh, Bo=0.0, d=d, Gamma=Gamma, Omega=nd.Omega)
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
                       "time_stepping": {"steps_per_forcing_period": n}},
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
    forced = [cl.analyse(_synthetic_run(tmp_path, G, -0.001, 20, 50), out, 4.0, None, False)
              for G in (0.2, 0.3)]
    for r in forced:
        assert abs(r["modes_mu_error_spatial"]) < 1e-8
    (th,) = cl.thresholds(forced)
    assert th["floquet_exact"] == pytest.approx(th["Gamma_c_exact"], rel=2e-3)
