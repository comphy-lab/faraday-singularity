"""Contract tests for the collapse driver's bookkeeping on synthetic data with known answers.

The driver loop runs against a stand-in problem whose free-surface measurement is prescribed as a
function of time, so stops, events and state dumps are checked without a finite-element solve.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "postProcess"))

import collapse_table

from faraday_singularity.driver import (CollapseDriver, CollapseSettings, classify_failure,
                                        limit_scale, oomph_error_message)
from faraday_singularity.episodes import EpisodeTracker, segment
from faraday_singularity.singular import singular_time


def _settings(**kw) -> CollapseSettings:
    return CollapseSettings(t_end=kw.pop("t_end", 1.0), dt_max=kw.pop("dt_max", 0.01), **kw)


# --- a stand-in problem and a driver with a prescribed measurement ---------------------------

MIN_DT_MESSAGE = "Tried to reduce dt to 7.27596e-13 which is less than the minimum dt (1e-12)."
# how oomph-lib reports it in the problem's log file
MIN_DT_LOG = f"""Processor 0:   TIMESTEP REJECTED


======================================================================
Oomph-lib ERROR

 at double oomph::Problem::adaptive_unsteady_newton_solve(const double&, const double&, const bool&)

 in problem.cc:11736

--------------------------------------------------------

{MIN_DT_MESSAGE}

======================================================================

"""


class FakeProblem:
    """The parts of ``FaradayProblem`` the driver uses; ``run`` raises ``error`` on call
    ``fail_at``, with ``log`` as the end of the log file."""

    def __init__(self, fail_at: int | None = None, depth: float = 2 * math.pi,
                 error: Exception | None = None, log: str = MIN_DT_LOG):
        self.t = 0.0
        self.calls = 0
        self.fail_at = fail_at
        self.error = RuntimeError("OomphException") if error is None else error
        self.log = log
        self.remesh_count = 0
        self.mesh_settings = SimpleNamespace(h_axis=0.05)
        self.nd = SimpleNamespace(radius=3.8317, depth=depth)
        self.template = SimpleNamespace(focus=None)
        self.saved: list[float] = []

    def get_current_time(self, dimensional=False, as_float=True):
        return self.t

    def run(self, t_end, **_kw):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            self.t = t_end  # a rejected step leaves the time advanced
            raise self.error
        self.t = t_end

    def oomph_log_tail(self):
        return self.log

    def observables(self):
        return {"volume": 1.0, "area": 1.0, "kinetic_energy": 0.0, "dissipation": 0.0}

    def active_mesh_template(self):
        return self.template

    def force_remesh(self):
        self.remesh_count += 1  # FaradayProblem.actions_after_remeshing

    def save_state(self, path, quiet=True):
        Path(path).write_text("state")
        self.saved.append(self.t)


class SurfaceProblem(FakeProblem):
    """A free surface ``z = surface(t, r)`` at rest, measured by the real driver.

    Its nodes are spaced ``h/2`` out to ``r = 0.5``, ``h`` the focus size of the last rebuild
    (``h_axis`` before any), so that ``h_near = h``. A rebuild asking for a focus finer than
    ``finest`` leaves the mesh as it was: a refinement that does not take effect.
    """

    def __init__(self, surface, finest: float = 0.0, **kw):
        super().__init__(**kw)
        self.surface = surface
        self.finest = finest
        self.h_mesh = self.mesh_settings.h_axis

    def force_remesh(self):
        super().force_remesh()
        if self.template.focus.size >= self.finest:
            self.h_mesh = self.template.focus.size

    def interface_arrays(self):
        inner = 0.5 * self.h_mesh * np.arange(math.ceil(1.0 / self.h_mesh) + 1)
        r = np.concatenate([inner, np.linspace(inner[-1], self.nd.radius, 100)[1:]])
        rest = np.zeros_like(r)
        return r, self.surface(self.t, r), rest, rest

    def axial_velocity_at(self, points):
        return [0.0] * len(points), True

    def save_state(self, path, quiet=True):
        Path(path).write_text(json.dumps({"time": self.t, "h_mesh": self.h_mesh}))
        self.saved.append(self.t)

    def load_state(self, path):
        dump = json.loads(Path(path).read_text())
        self.t, self.h_mesh = dump["time"], dump["h_mesh"]


def _sharp_then_flat_trough(t, r, t_flat=0.045):
    """A trough on the axis with apex radius of curvature 0.006 that flattens at ``t_flat`` to
    one of radius 2, above the cap of the local scale."""
    kappa = 1.0 / 0.006 if t < t_flat else 0.5
    return 2 * math.pi - 0.1 + 0.5 * kappa * r**2


class PrescribedDriver(CollapseDriver):
    """Measurement given by ``fields(t)``; everything else as in the real driver."""

    def __init__(self, pb, settings, outdir, fields, **kw):
        super().__init__(pb, settings, outdir, **kw)
        self.fields = fields

    def measure(self) -> dict:
        t = self._t()
        r = np.linspace(0.0, 1.0, 11)
        m = {"time": t, "axis_lowest": True, "z_axis": self.pb.nd.depth, "w_axis": 0.0,
             "axis_curvature": 1.0, "cavity": True, "jet_born": False, "flat_bump": False,
             "r_j": 0.0, "z_b": 0.0, "Q_j": math.nan, "V_j": math.nan, "We_j": math.nan,
             "flux_found": True, "r_neck": math.nan, "z_neck": math.nan, "ell": 0.4,
             "h_focus": 0.05, "h_near": 0.05, "U": 1.0, "beta_fit": math.nan,
             "beta_fit_points": 0, "beta_window_lo": 0.0, "beta_window_hi": 0.0,
             "angles": [math.nan] * len(self.s.cone_radii), "focus_radius": 1.0,
             "profile": (r, np.zeros_like(r), np.zeros_like(r), np.zeros_like(r))}
        m.update(self.fields(t))
        return m


def _read_events(outdir: Path) -> list[dict]:
    return [json.loads(line) for line in (outdir / "events.jsonl").read_text().splitlines()]


# --- A: a Newton failure near a pinch-off or the singularity ---------------------------------


def test_failure_with_a_narrow_neck_is_a_pinch_off():
    m = {"time": 2.0, "r_neck": 5e-4, "z_neck": 10.0, "ell": 5e-4}
    stop = classify_failure(m, _settings(), "RuntimeError: OomphException")
    assert stop["reason"] == "pinch-off-approach"
    assert stop["trigger"] == "newton-failure"
    assert stop["r_neck"] == 5e-4 and stop["z_neck"] == 10.0 and stop["time"] == 2.0
    assert stop["error"] == "RuntimeError: OomphException"


def test_failure_at_a_small_local_scale_is_the_singularity():
    s = _settings()
    for r_neck in (math.nan, 0.05):  # no neck, or one wider than the guard
        m = {"time": 2.0, "r_neck": r_neck, "ell": 2e-4, "axis_curvature": 5e3, "r_j": math.nan,
             "z_axis": 6.0, "jet_born": False}
        stop = classify_failure(m, s, "e")
        assert stop["reason"] == "singularity-approach"
        assert stop["trigger"] == "newton-failure"
        assert stop["ell"] == 2e-4 and stop["axis_curvature"] == 5e3


def test_other_failures_are_not_classified():
    s = _settings(neck_guard=1e-2, ell_guard=1e-3)
    assert classify_failure({"r_neck": 0.02, "ell": 0.02}, s, "e") is None
    assert classify_failure({"r_neck": math.nan, "ell": math.nan}, s, "e") is None
    assert classify_failure({"r_neck": 1e-2, "ell": 1e-3}, s, "e") is None  # guards are strict
    # the guards are settings
    assert classify_failure({"r_neck": 0.02, "ell": 0.02}, _settings(neck_guard=0.05), "e")
    # the scale guard applies to the measured scale, not to the growth-limited one
    assert classify_failure({"r_neck": math.nan, "ell": 6e-4, "ell_measured": 0.4}, s, "e") is None
    stop = classify_failure({"r_neck": math.nan, "ell": 6e-4, "ell_measured": 5e-4}, s, "e")
    assert stop["reason"] == "singularity-approach"
    assert stop["ell"] == 6e-4 and stop["ell_measured"] == 5e-4


def _sharp_apex_that_flattens(t, r):
    """A trough on the axis with apex radius of curvature 2e-4, below ``ell_guard``, that flattens
    at t = 0.01 to one of radius 2, above the cap of the local scale."""
    kappa = 1.0 / 2e-4 if t < 0.01 else 0.5
    return 2 * math.pi - 0.1 + 0.5 * kappa * r**2


@pytest.mark.parametrize("fail_at", [3, 9])
def test_failure_is_classified_by_the_measured_scale(tmp_path, fail_at):
    # oomph-lib gives up on the third solve, while the sharp apex stands, or on the ninth, four
    # steps after it has flattened: the growth-limited scale ell is then still below ell_guard, but
    # no feature that small is left, so the failure is not an expected end of the run
    s = _settings(t_end=0.4, dt_max=0.01, capillary=1e3, state_dt=1.0)
    drv = CollapseDriver(SurfaceProblem(_sharp_apex_that_flattens, fail_at=fail_at), s, tmp_path)
    if fail_at == 3:
        stop = drv.run(1e-9)["stop"]
        drv.close()
        assert stop["reason"] == "singularity-approach"
        assert stop["ell_measured"] == pytest.approx(2e-4, rel=1e-6)
        return
    with pytest.raises(RuntimeError):
        drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    assert tab["axis_curvature"][-1] == pytest.approx(0.5) and tab["ell"][-1] < s.ell_guard
    assert all(e["event"] != "stop" for e in _read_events(tmp_path))


def test_driver_stops_cleanly_on_newton_failure_near_a_neck(tmp_path):
    pb = FakeProblem(fail_at=40)
    s = _settings(t_end=1.0, dt_max=0.01, state_dt=0.1)
    # a neck closing linearly; it never reaches neck_stop, so only the failure can end the run
    drv = PrescribedDriver(pb, s, tmp_path, lambda t: {"r_neck": 0.009 - 1e-4 * t * 100,
                                                       "z_neck": 1.0})
    out = drv.run(1e-9)
    drv.close()
    stop = out["stop"]
    assert stop["reason"] == "pinch-off-approach" and stop["trigger"] == "newton-failure"
    assert stop["error"] == MIN_DT_MESSAGE
    assert out["steps"] == 39 and pb.t == pytest.approx(0.40)  # the rejected step advanced t
    tab = _rows(tmp_path)
    assert tab["step"][-1] == 39
    t_last = tab["time"][-1]  # the last measured state, to the table's 13 digits
    assert stop["time"] == pytest.approx(t_last, abs=1e-11)
    events = _read_events(tmp_path)
    assert events[-1]["event"] == "stop"
    assert events[-1]["time"] == pytest.approx(t_last, abs=1e-11)
    # no state is dumped from the rejected iterate
    assert pb.saved and all(t <= t_last + 1e-11 for t in pb.saved)
    profiles = sorted((tmp_path / "profiles").glob("profile_*.txt"))
    header = json.loads(profiles[-1].read_text().splitlines()[0][2:])
    assert header["reason"] == "stop" and header["time"] == pytest.approx(t_last, abs=1e-11)


def test_driver_reraises_unclassified_failures(tmp_path):
    drv = PrescribedDriver(FakeProblem(fail_at=3), _settings(), tmp_path, lambda t: {})
    with pytest.raises(RuntimeError):
        drv.run(1e-9)
    drv.close()


OTHER_LOG = MIN_DT_LOG.replace(MIN_DT_MESSAGE, "Linear solver failed")


def test_oomph_error_message_from_the_log():
    assert oomph_error_message(MIN_DT_LOG) == MIN_DT_MESSAGE
    assert oomph_error_message(MIN_DT_LOG + OTHER_LOG) == "Linear solver failed"  # the last one
    assert oomph_error_message("Processor 0:   TIMESTEP REJECTED\n") == ""


@pytest.mark.parametrize("error, log", [
    (RuntimeError("Gmsh could not mesh the geometry"), MIN_DT_LOG),  # mesh generation or remesh
    (NotImplementedError("OomphException"), MIN_DT_LOG),  # a subclass of RuntimeError
    (RecursionError("maximum recursion depth exceeded"), MIN_DT_LOG),
    (RuntimeError("OomphException"), OTHER_LOG),  # another oomph-lib error
    (RuntimeError("OomphException"), ""),  # no message in the log
], ids=["gmsh", "subclass", "recursion", "other-oomph-error", "no-message"])
def test_driver_reraises_other_failures_near_a_neck(tmp_path, error, log):
    pb = FakeProblem(fail_at=5, error=error, log=log)
    # a neck narrower than neck_guard and a local scale below ell_guard
    drv = PrescribedDriver(pb, _settings(), tmp_path, lambda t: {"r_neck": 2e-3, "ell": 1e-4})
    with pytest.raises(RuntimeError) as info:
        drv.run(1e-9)
    drv.close()
    assert info.value is error
    assert all(e["event"] != "stop" for e in _read_events(tmp_path))


# --- B: singular time from the log-derivative --------------------------------------------------


def _power_law(side: str, alpha: float, rng, t0=1.3, A=0.7, n=10):
    """Non-uniform samples of ``A |t - t0|^alpha``, approaching or leaving ``t0`` geometrically."""
    steps = np.cumsum(rng.uniform(0.05, 0.15, n))
    tau = 0.2 * np.exp(-steps) if side == "before" else 0.02 * np.exp(steps)
    t = t0 - tau if side == "before" else t0 + tau
    return t, A * tau**alpha, float(tau.min())


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("alpha", [0.5, 2 / 3, 1.0, 1.5])
def test_singular_time_of_exact_power_law(side, alpha):
    t, x, tau_min = _power_law(side, alpha, np.random.default_rng(1))
    t0, a = singular_time(t, x, min_samples=10)
    # second-order differences on steps of 5-15 % of the distance to t0
    assert a == pytest.approx(alpha, rel=3e-3)
    assert abs(t0 - 1.3) < 3e-3 * tau_min


@pytest.mark.parametrize("side", ["before", "after"])
def test_singular_time_with_noise(side):
    rng = np.random.default_rng(7)
    for _ in range(20):
        t, x, tau_min = _power_law(side, 2 / 3, rng)
        t0, a = singular_time(t, x * (1 + 1e-4 * rng.standard_normal(x.size)), min_samples=10)
        assert a == pytest.approx(2 / 3, rel=0.03)
        assert abs(t0 - 1.3) < 0.03 * tau_min


def test_singular_time_rejects_ill_conditioned_samples():
    rng = np.random.default_rng(3)
    t, x, _ = _power_law("before", 2 / 3, rng)
    nan = singular_time(t[:4], x[:4])  # too few samples
    assert all(math.isnan(v) for v in nan)
    assert math.isnan(singular_time(t, x, min_samples=11)[0])
    wiggle = x.copy()
    wiggle[5] = 1.01 * wiggle[4]  # not monotonic
    assert math.isnan(singular_time(t, wiggle)[0])
    assert math.isnan(singular_time(t[::-1], x)[0])  # times not increasing
    assert math.isnan(singular_time(t, -x)[0])  # not a length
    tt = np.linspace(0.0, 1.0, 10)
    assert math.isnan(singular_time(tt, np.exp(-tt))[0])  # exponential decay: no finite t0
    assert math.isnan(singular_time(tt, (2.0 - tt) ** -0.5)[0])  # a divergence, slope b < 0
    # samples straddling t0 (the length still decreasing at the last one)
    ts = 1.3 + np.array([-0.1, -0.08, -0.06, -0.045, -0.03, -0.02, -0.01, -0.004, 0.001, 0.002])
    assert math.isnan(singular_time(ts, np.abs(ts - 1.3) ** (2 / 3))[0])
    # a sound exponent outside the accepted range
    t, x, _ = _power_law("before", 1.5, rng)
    assert math.isnan(singular_time(t, x, alpha_range=(0.1, 1.2))[0])
    assert math.isfinite(singular_time(t, x)[0])


def _lorentzian_trough(t, kappa_max=5.0, t_max=0.5, width=0.25):
    """Apex curvature of a trough that deepens and recovers without collapsing."""
    return kappa_max / (1.0 + ((t - t_max) / width) ** 2)


@pytest.mark.parametrize("side", ["before", "after"])
def test_singular_time_needs_a_window_long_enough_to_extrapolate(side):
    # equal steps of 0.01: the window spans 0.09, so t0 may lie at most 0.18 beyond it
    tau = 0.15 + 0.01 * np.arange(10)[::-1] if side == "before" else 0.15 + 0.01 * np.arange(10)
    t = 1.3 - tau if side == "before" else 1.3 + tau
    t0, alpha = singular_time(t, tau ** (2 / 3), min_samples=10)
    assert t0 == pytest.approx(1.3, abs=1e-3) and alpha == pytest.approx(2 / 3, rel=1e-2)
    far = 0.2 + 0.01 * np.arange(10)[::-1] if side == "before" else 0.2 + 0.01 * np.arange(10)
    t = 1.3 - far if side == "before" else 1.3 + far
    assert math.isnan(singular_time(t, far ** (2 / 3), min_samples=10)[0])
    assert math.isfinite(singular_time(t, far ** (2 / 3), min_samples=10,
                                       max_extrapolation=2.5)[0])


def test_singular_time_of_a_trough_that_does_not_collapse_is_nan():
    # 1/kappa has a positive minimum; without the extrapolation rule 108 of these 160 windows
    # give a finite t0, with exponents between 2.1 and 4.8
    dt = 0.0125
    for t_last in np.arange(-0.5, 1.5, dt):
        t = t_last - dt * np.arange(9, -1, -1)
        assert math.isnan(singular_time(t, 1.0 / _lorentzian_trough(t), min_samples=10)[0])


def test_driver_gives_no_singular_time_for_a_trough_that_does_not_collapse(tmp_path):
    # 1/kappa >= 0.05 stays below the cap of 0.4 for most of the run
    drv = PrescribedDriver(FakeProblem(), _settings(t_end=1.0, dt_max=0.0125), tmp_path,
                           lambda t: {"axis_curvature": _lorentzian_trough(t, kappa_max=20.0),
                                      "U": 0.5})
    drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    assert np.allclose(tab["dt"][1:], 0.0125)
    assert np.all(tab["t0_source"] == 0) and np.all(np.isnan(tab["t0_est"]))


def _collapse_then_jet(t, t_birth=0.5, t0_apex=0.6, t0_jet=0.45, t_cavity=0.8, t0_next=0.95):
    """Apex radius closing on ``t0_apex``, then a jet growing from ``t0_jet``, then a new cavity."""
    if t < t_birth:
        return {"axis_curvature": 1.0 / (t0_apex - t) ** (2 / 3)}
    if t < t_cavity:
        return {"jet_born": True, "axis_lowest": False, "axis_curvature": -1.0,
                "r_j": 0.1 * (t - t0_jet) ** 0.6}
    return {"axis_curvature": 1.0 / (t0_next - t) ** (2 / 3)}


def _rows(outdir: Path) -> dict[str, np.ndarray]:
    path = outdir / "collapse.txt"
    with path.open() as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def test_driver_estimates_singular_time_before_and_after_birth(tmp_path):
    pb = FakeProblem()
    drv = PrescribedDriver(pb, _settings(t_end=0.94, dt_max=0.01), tmp_path, _collapse_then_jet)
    drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    t, src = tab["time"], tab["t0_source"]
    # the apex radius is sampled once below the cap h_axis elements_per_scale = 0.4 (t >= 0.35);
    # ten samples later t0 = 0.6 lies within twice their span of 0.09
    apex = (t > 0.435) & (t < 0.5)
    assert apex.sum() == 6 and np.all(src[apex] == 1)
    assert np.allclose(tab["t0_est"][apex], 0.6, atol=1e-3)
    assert np.allclose(tab["alpha_est"][apex], 2 / 3, rtol=1e-2)
    assert np.all(src[t < 0.435] == 0)
    # the jet radius from birth at 0.5; t0 = 0.45 lies within twice the span up to t = 0.72
    jet = (t > 0.585) & (t < 0.725)
    assert jet.sum() == 14 and np.all(src[jet] == 2)
    assert np.allclose(tab["t0_est"][jet], 0.45, atol=1e-3)
    assert np.allclose(tab["alpha_est"][jet], 0.6, rtol=1e-2)
    assert np.all(src[(t > 0.725) & (t < 0.8)] == 0)  # too far from t0 to extrapolate
    born = next(e for e in _read_events(tmp_path) if e["event"] == "jet-born")
    assert born["t0_apex"] == pytest.approx(0.6, abs=1e-3)
    # the jet goes, a new cavity starts a cycle and both series restart
    cavity = [e for e in _read_events(tmp_path) if e["event"] == "cavity"]
    assert cavity[-1]["cycle"] == 1
    after = t >= 0.8
    assert np.all(src[after][:10] == 0) and np.all(src[after][10:] == 1)
    assert np.allclose(tab["t0_est"][after][10:], 0.95, atol=1e-3)


def test_jet_birth_carries_no_stale_apex_estimate(tmp_path):
    # the apex series is interrupted (the axis is briefly not the lowest point) after an
    # estimate and the jet is born before ten new samples: its event carries no estimate
    def fields(t):
        if 0.47 <= t < 0.48:
            return {"axis_lowest": False, "axis_curvature": -1.0}
        return _collapse_then_jet(t)

    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.6, dt_max=0.01), tmp_path, fields)
    drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    assert np.any(tab["t0_source"][tab["time"] < 0.47] == 1)
    born = next(e for e in _read_events(tmp_path) if e["event"] == "jet-born")
    assert math.isnan(born["t0_apex"]) and math.isnan(born["alpha_apex"])


@pytest.mark.parametrize("bump_steps", [1, 2, 3])
def test_jet_birth_keeps_the_apex_estimate_across_a_growing_bump(tmp_path, bump_steps):
    # a bump on the axis (z''(0) < 0, the axis no longer the lowest point) grows for a few steps
    # before it stands high enough to count as a jet: it ends the apex series that led to the birth
    # but does not discard its estimate
    def fields(t):
        if 0.5 - 0.01 * bump_steps - 1e-9 <= t < 0.5:
            return {"axis_lowest": False, "axis_curvature": -1.0}
        return _collapse_then_jet(t)

    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.6, dt_max=0.01), tmp_path, fields)
    drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    assert np.all(tab["t0_source"][(tab["time"] > 0.5 - 0.01 * bump_steps - 1e-9)
                                   & (tab["time"] < 0.5)] == 0)
    born = next(e for e in _read_events(tmp_path) if e["event"] == "jet-born")
    assert born["time"] == pytest.approx(0.5)
    assert born["t0_apex"] == pytest.approx(0.6, abs=1e-3)
    assert born["alpha_apex"] == pytest.approx(2 / 3, rel=1e-2)
    assert born["t_apex_est"] == pytest.approx(0.5 - 0.01 * (bump_steps + 1))


def test_jet_radius_estimate_of_a_closing_base_lies_after_the_samples(tmp_path):
    # born at a finite radius, the base closes in on the axis: the t0 of r_j (source 2) is the time
    # at which it would reach the axis, after the samples, not the origin of a growth before them
    def fields(t):
        if t < 0.5:
            return {}  # a broad trough: 1/z''(0) = 1 is above the cap of 0.4
        return {"jet_born": True, "axis_lowest": False, "axis_curvature": -1.0,
                "r_j": 0.3 * ((0.7 - t) / 0.2) ** 0.6}

    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.695, dt_max=0.002), tmp_path, fields)
    drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    jet = tab["t0_source"] == 2
    assert jet.sum() >= 5 and np.all(tab["time"][jet] > 0.5)
    assert np.all(tab["t0_est"][jet] > tab["time"][jet])
    assert np.allclose(tab["t0_est"][jet], 0.7, atol=1e-4)
    assert np.allclose(tab["alpha_est"][jet], 0.6, rtol=1e-2)


# --- C: crest and trough episodes of the axis elevation ---------------------------------------


def _eta(t):
    """A decaying standing wave released as a crest of height 0.5, with a small fast wobble."""
    return 0.5 * np.exp(-0.05 * t) * np.cos(t) + 0.015 * np.sin(40 * t)


def test_episodes_of_a_standing_wave():
    t = np.linspace(0.0, 4 * math.pi, 4001)
    eta = _eta(t)
    index, kind, events = segment(t, eta, hys=0.02)
    assert [e["event"] for e in events] == ["crest", "trough", "crest", "trough", "crest"]
    assert [e["episode"] for e in events] == [0, 1, 2, 3, 4]
    assert events[0]["time"] == 0.0 and not events[0]["partial"]
    # each trough starts once eta < -hys, after the zero crossing near pi/2 + k pi
    for ev, k in zip(events[1::2], (0, 2), strict=True):
        assert math.pi / 2 + k * math.pi < ev["time"] < math.pi / 2 + k * math.pi + 0.15
    # its wave height is the largest eta over the crest episode before it
    first, second = events[1], events[3]
    assert first["h_crest"] == pytest.approx(eta[t < first["time"]].max())
    assert first["h_crest"] == pytest.approx(0.5, abs=0.02)
    crest = (t >= events[2]["time"]) & (t < second["time"])
    assert second["h_crest"] == pytest.approx(eta[crest].max())
    assert second["t_crest"] == pytest.approx(t[crest][np.argmax(eta[crest])])
    assert set(np.unique(kind)) == {1, -1}
    assert np.all(np.diff(index) >= 0) and index[-1] == 4


def test_hysteresis_ignores_wobbles_within_the_band():
    hys = 0.02
    eta = [0.1, 0.05, 0.0, -0.019, 0.019, -0.03, -0.01, 0.015, -0.025, -0.05, 0.019, 0.03]
    t = np.arange(len(eta), dtype=float)
    index, _, events = segment(t, eta, hys=hys)
    assert [(e["event"], e["time"]) for e in events] == [("crest", 0.0), ("trough", 5.0),
                                                        ("crest", 11.0)]
    assert list(index) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]
    assert events[1]["h_crest"] == 0.1 and events[1]["t_crest"] == 0.0


def test_a_record_starting_in_a_trough_has_no_wave_height():
    _, _, events = segment([0.0, 1.0, 2.0], [-0.3, 0.1, -0.2])
    assert events[0]["event"] == "trough" and math.isnan(events[0]["h_crest"])
    assert events[2]["h_crest"] == 0.1


def test_restart_without_episode_history_is_partial():
    t = np.linspace(0.0, 4 * math.pi, 4001)
    eta = _eta(t)
    i = np.searchsorted(t, 0.5)  # restart inside the first crest
    _, _, events = segment(t[i:], eta[i:], history_known=False)
    assert events[0]["event"] == "crest" and events[0]["partial"]
    assert events[1]["event"] == "trough" and math.isnan(events[1]["h_crest"])
    assert not events[1]["partial"] and math.isfinite(events[3]["h_crest"])
    # a record starting inside the band, part-way through a crest that rises again: the crest
    # began before the record, so the first trough's wave height is unknown, not 0.1
    eta_full = [0.0, 0.3, 0.1, 0.01, 0.1, -0.3]
    _, _, full = segment(np.arange(6.0), eta_full)
    assert full[-1]["event"] == "trough" and full[-1]["h_crest"] == 0.3
    _, _, events = segment(np.arange(3.0, 6.0), eta_full[3:], history_known=False)
    assert [(e["event"], e["partial"]) for e in events] == [("crest", True), ("trough", False)]
    assert math.isnan(events[1]["h_crest"])
    # with its history, a restarted tracker continues exactly
    full = EpisodeTracker()
    for ti, ei in zip(t[:i], eta[:i], strict=True):
        full.update(ti, ei)
    resumed = EpisodeTracker.from_dict(json.loads(json.dumps(full.to_dict())))
    ev_full = [full.update(ti, ei) for ti, ei in zip(t[i:], eta[i:], strict=True)]
    ev_resumed = [resumed.update(ti, ei) for ti, ei in zip(t[i:], eta[i:], strict=True)]
    assert ev_full == ev_resumed and resumed.to_dict() == full.to_dict()


def _wave_and_collapse(t):
    """Axis elevation of a standing wave; an apex radius closing on t = 0.345 and reopening after
    it; a shrinking local scale."""
    return {"z_axis": 2 * math.pi + float(_eta(8 * t)),
            "axis_curvature": 1.0 / abs(0.345 - t) ** (2 / 3), "ell": 0.05 + 0.3 * (1 - t)}


def test_driver_records_episodes_and_restarts_series_on_a_trough(tmp_path):
    pb = FakeProblem()
    drv = PrescribedDriver(pb, _settings(t_end=1.0, dt_max=0.01), tmp_path, _wave_and_collapse)
    drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    index, _, events = segment(tab["time"], tab["z_axis"] - pb.nd.depth)
    assert np.array_equal(tab["episode"], index)  # the same segmentation as in post-processing
    recorded = [e for e in _read_events(tmp_path) if e["event"] in ("crest", "trough")]
    assert [(e["event"], e["episode"]) for e in recorded] == \
        [(e["event"], e["episode"]) for e in events]
    # the table holds 13 significant digits
    assert np.allclose([e["time"] for e in recorded], [e["time"] for e in events], rtol=1e-11)
    troughs = [e for e in recorded if e["event"] == "trough"]
    assert len(troughs) >= 2 and troughs[0]["h_crest"] == pytest.approx(0.5, abs=0.02)
    k = int(np.argmin(np.abs(tab["time"] - troughs[0]["time"])))
    # the apex series restarts: no estimate for singular_samples - 1 steps
    assert tab["t0_source"][k - 1] == 1
    assert np.all(tab["t0_source"][k:k + 9] == 0) and tab["t0_source"][k + 9] == 1


def _wave_collapse_and_coarse_mesh(t):
    """As ``_wave_and_collapse``, with a free surface too coarse for the focus over a stretch, so
    that rebuilds are tried and backed off."""
    return _wave_and_collapse(t) | ({"h_near": 0.2} if 0.42 <= t < 0.75 else {})


def test_restart_continues_the_run_exactly(tmp_path):
    # scale_rate = 0.01: the shrink of ell = 0.05 + 0.3 (1 - t) sets the step from t = 0.17 on
    s = _settings(t_end=1.0, dt_max=0.01, state_dt=0.02, state_keep=1000, scale_rate=0.01)
    pb = FakeProblem()
    drv = PrescribedDriver(pb, s, tmp_path / "a", _wave_collapse_and_coarse_mesh)
    drv.run(1e-9)
    drv.close()
    full = _rows(tmp_path / "a")
    events = _read_events(tmp_path / "a")
    backoff = [e["step"] for e in events if e["event"] == "remesh-ineffective"]
    assert len(backoff) >= 3
    # a dump while a rebuild is backed off, with the surface still too coarse after it
    metas = [json.loads(f.read_text()) for f in sorted((tmp_path / "a" / "states").glob("*.json"))]
    meta = next(m for m in metas if m["time"] < 0.65
                and any(n < m["step"] < n + s.remesh_backoff for n in backoff))
    k = int(np.searchsorted(full["step"], meta["step"]))
    t_k = full["time"][k]
    assert full["dt"][k + 1] == pytest.approx(s.scale_rate * (0.05 + 0.3 * (1 - t_k)) / 0.3)
    pb2 = FakeProblem()
    pb2.t = meta["time"]  # load_state
    drv2 = PrescribedDriver(pb2, s, tmp_path / "b", _wave_collapse_and_coarse_mesh)
    drv2.restore(meta)
    drv2.run(1e-9)
    drv2.close()
    rest = _rows(tmp_path / "b")
    for name in ("step", "cycle", "episode", "time", "dt", "remesh_count", "t0_est", "alpha_est",
                 "t0_source"):
        first = 1 if name == "dt" else 0  # the restart row records no step
        assert np.array_equal(rest[name][first:], full[name][k + first:], equal_nan=True), name
    assert np.any(rest["t0_source"] > 0)
    after = [(e["event"], e["step"]) for e in events
             if e["event"].startswith("remesh") and e["step"] > meta["step"]]
    resumed = _read_events(tmp_path / "b")
    assert resumed[0]["event"] == "restart" and resumed[0]["episode_history"]
    assert [(e["event"], e["step"]) for e in resumed if e["event"].startswith("remesh")] == after
    assert after and pb2.remesh_count == pb.remesh_count


def test_restart_at_a_new_cycle_continues_the_singular_time(tmp_path):
    # a dump on every step, among them the step at which the jet has gone and a cycle begins
    s = _settings(t_end=0.94, dt_max=0.01, state_dt=0.0, state_keep=1000)
    drv = PrescribedDriver(FakeProblem(), s, tmp_path / "a", _collapse_then_jet)
    drv.run(1e-9)
    drv.close()
    full = _rows(tmp_path / "a")
    step = next(e["step"] for e in _read_events(tmp_path / "a")
                if e["event"] == "cavity" and e["cycle"] == 1)
    meta = json.loads((tmp_path / "a" / "states" / f"state_{step:08d}.json").read_text())
    assert meta["cycle"] == 1 and meta["apex_samples"] == []  # dumped after the series restarted
    pb2 = FakeProblem()
    pb2.t = meta["time"]  # load_state
    drv2 = PrescribedDriver(pb2, s, tmp_path / "b", _collapse_then_jet)
    drv2.restore(meta)
    drv2.run(1e-9)
    drv2.close()
    rest = _rows(tmp_path / "b")
    k = int(np.searchsorted(full["step"], step))
    # the dumped step's row was written before the cycle began; the rows after it continue
    for name in ("step", "cycle", "time", "dt", "t0_est", "alpha_est", "t0_source"):
        assert np.array_equal(rest[name][1:], full[name][k + 1:], equal_nan=True), name
    assert np.sum(rest["t0_source"] == 1) == 5


def test_restore_takes_the_hysteresis_of_the_new_settings(tmp_path):
    s = _settings(t_end=0.5, dt_max=0.01, state_dt=0.1, state_keep=20)
    drv = PrescribedDriver(FakeProblem(), s, tmp_path / "a", _wave_and_collapse)
    drv.run(1e-9)
    drv.close()
    meta = json.loads(sorted((tmp_path / "a" / "states").glob("*.json"))[-1].read_text())
    assert meta["episodes"]["hys"] == s.eta_hys
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, _settings(eta_hys=0.05), tmp_path / "b", _wave_and_collapse)
    drv.restore(meta)
    drv.close()
    assert drv.episodes.hys == 0.05
    assert drv.episodes.to_dict() == {**meta["episodes"], "hys": 0.05}


def test_restore_accepts_dumps_without_the_new_bookkeeping(tmp_path):
    old = {"step": 3575, "cycle": 0, "cavity_seen": True, "born": False, "neck_seen": True,
           "max_curvature": 2.5, "max_curvature_time": 10.9, "snapshots": 169, "time": 0.5,
           "remesh_count": 46, "focus": None, "dt_prev": 1e-4}
    pb = FakeProblem()
    pb.t = old["time"]
    drv = PrescribedDriver(pb, _settings(), tmp_path, _wave_and_collapse)
    drv.restore(old)
    st = drv.state
    assert st.step == 3575 and st.dt_prev == 1e-4 and pb.remesh_count == 46
    assert st.apex_samples == [] and st.jet_samples == [] and math.isnan(st.ell_prev)
    assert drv.episodes.index == -1 and not drv.episodes.history_known
    restart = _read_events(tmp_path)[0]
    assert restart["event"] == "restart" and not restart["episode_history"]
    drv.run(1e-9)
    drv.close()
    events = [e for e in _read_events(tmp_path) if e["event"] in ("crest", "trough")]
    assert events[0]["partial"] and events[0]["time"] == 0.5
    first_trough = next(e for e in events if e["event"] == "trough")
    assert math.isnan(first_trough["h_crest"])


# --- D: one row per trough episode -------------------------------------------------------------


def _write_run(run: Path, tab: dict[str, np.ndarray] | None, *, restart=None, stop=None,
               depth=2.0, state="completed") -> Path:
    """A run directory: ``collapse.txt`` from ``tab`` (unless None), parameters and status."""
    run.mkdir(parents=True, exist_ok=True)
    if tab is not None:
        cols = list(tab)
        with (run / "collapse.txt").open("w") as fh:
            fh.write("# " + "\t".join(cols) + "\n")
            np.savetxt(fh, np.column_stack([tab[c] for c in cols]), delimiter="\t")
    params = {"case": {"case_id": run.name}, "restart": restart, "collapse": None,
              "dimensionless": {"Gamma": 1.0, "Oh": 0.04, "a0": 0.5, "depth": depth,
                                "forcing_period": 3.14}}
    (run / "parameters.json").write_text(json.dumps(params))
    status = {"state": state, "collapse": {"stop": stop} if stop else None}
    (run / "run_status.json").write_text(json.dumps(status))
    return run


def _old_style_table(depth=2.0):
    """Two troughs; a jet is born in the second one, which ends at a closing neck."""
    t = np.linspace(0.0, 2.5 * 2 * math.pi, 2501)
    eta = 0.5 * np.cos(t) * (1 + 0.1 * t)
    kappa = -eta  # a trough has positive curvature on the axis
    born = (t > 3 * math.pi + 0.5) & (t < 3 * math.pi + 1.0)
    neck = t > 4 * math.pi + 2.0
    return {"step": np.arange(t.size), "time": t, "z_axis": depth + eta, "w_axis": -np.sin(t),
            "axis_curvature": kappa, "jet_born": born.astype(float),
            "r_j": np.where(born, 0.01 + 0.1 * (t - 3 * math.pi - 0.5), np.nan),
            "V_j": np.where(born, 3.0, np.nan),  # upward: flux into the jet
            "r_neck": np.where(neck, 0.1 - 0.01 * (t - 4 * math.pi - 2.0), np.nan)}


def test_per_episode_reconstructs_old_runs(tmp_path):
    tab = _old_style_table()
    run = _write_run(tmp_path / "old" / "out", tab,
                     stop={"reason": "pinch-off-approach", "trigger": "neck-radius"})
    rows = collapse_table.per_episode(run)
    assert [r["episode"] for r in rows] == [1, 3, 5]
    assert all(r["source"] == "reconstructed" and r["run"] == "old" for r in rows)
    t, eta = tab["time"], tab["z_axis"] - 2.0
    first, second, last = rows
    before = t < first["t_start"]
    assert first["h_crest"] == pytest.approx(eta[before].max())
    assert first["t_crest"] == pytest.approx(t[before][np.argmax(eta[before])])
    crest = (t > second["t_start"] - math.pi) & (t < second["t_start"])
    assert second["h_crest"] == pytest.approx(eta[crest].max())
    assert first["jet_born"] == 0 and math.isnan(first["max_V_j"])
    assert second["jet_born"] == 1 and second["max_V_j"] == pytest.approx(3.0)
    assert second["t_birth"] == pytest.approx(t[np.argmax(tab["jet_born"] > 0)])
    assert second["r_j_birth"] == pytest.approx(0.01, abs=1e-3)
    i0 = int(np.searchsorted(t, first["t_start"]))
    sel = slice(i0, i0 + int(np.argmax(eta[i0:] > 0.02)))  # up to the next crest
    assert first["min_z_axis"] == pytest.approx(tab["z_axis"][sel].min(), abs=1e-6)
    # the largest trough curvature (here -eta) is at the bottom of the trough
    i = int(np.argmin(tab["z_axis"][sel]))
    assert first["t_max_apex_curvature"] == pytest.approx(t[sel][i])
    assert first["max_apex_curvature"] == pytest.approx(-eta[sel][i])
    assert first["stop"] == "" and last["stop"] == "pinch-off-approach:neck-radius"
    assert math.isnan(first["min_r_neck"]) and last["min_r_neck"] < 0.1


def test_per_episode_marks_restarted_old_runs_partial(tmp_path):
    tab = {k: v[300:] for k, v in _old_style_table().items()}  # starts inside the first trough
    run = _write_run(tmp_path / "r1" / "out", tab, restart="state_00000300.dump", state="failed")
    rows = collapse_table.per_episode(run)
    assert rows[0]["partial"] == 1 and math.isnan(rows[0]["h_crest"])
    assert rows[1]["partial"] == 0 and math.isfinite(rows[1]["h_crest"])
    assert rows[-1]["stop"] == "failed"


def test_per_episode_keeps_the_jet_and_stop_in_the_crest_it_launches(tmp_path):
    # the jet born in the second trough rises into the crest after it, where its column necks
    # and the run stops
    full = _old_style_table()
    tab = {k: v[full["time"] < 3.5 * math.pi + 0.8] for k, v in full.items()}
    t = tab["time"]
    jet = t > 3 * math.pi + 0.5
    tab["jet_born"] = jet.astype(float)
    tab["r_j"] = np.where(jet, 0.01 + 0.01 * (t - 3 * math.pi - 0.5), np.nan)
    tab["V_j"] = np.where(jet, 3.0 + (t - 3 * math.pi - 0.5), np.nan)  # fastest at the end
    # the base first closes in on the axis, with fast flux out of the jet
    tab["V_j"][jet & (t < 3 * math.pi + 0.6)] = -25.0
    tab["r_neck"] = np.where(t > 3.5 * math.pi + 0.3, 0.05 - 0.05 * (t - 3.5 * math.pi), np.nan)
    run = _write_run(tmp_path / "jet" / "out", tab,
                     stop={"reason": "pinch-off-approach", "trigger": "neck-radius"})
    index, kind, _ = segment(t, tab["z_axis"] - 2.0)
    assert index[-1] == 4 and kind[-1] == 1  # the run ends in the crest after trough 3
    last = collapse_table.per_episode(run)[-1]
    assert last["episode"] == 3 and last["jet_born"] == 1
    assert last["stop"] == "pinch-off-approach:neck-radius"
    assert last["min_r_neck"] == pytest.approx(np.nanmin(tab["r_neck"]))
    assert last["max_V_j"] == pytest.approx(tab["V_j"][-1])
    assert last["min_z_axis"] == pytest.approx(tab["z_axis"][index == 3].min())


def test_jet_speeds_count_only_flux_into_the_jet(tmp_path):
    # the jet born in the second trough first closes its base in on the axis, with flux out of the
    # jet that grows fast as r_j -> 0 (V_j < 0, Q_j < 0); it is not a jet speed
    tab = _old_style_table()
    t, born = tab["time"], tab["jet_born"] > 0
    tab["V_j"][born & (t < 3 * math.pi + 0.6)] = -25.0
    tab["We_j"] = tab["V_j"] ** 2 * tab["r_j"]
    tab["cycle"], tab["cavity"] = np.zeros_like(t), np.ones_like(t)
    tab["remesh_count"], tab["volume"] = np.zeros_like(t), np.ones_like(t)
    run = _write_run(tmp_path / "old" / "out", tab)
    rows = collapse_table.per_episode(run)
    assert rows[1]["jet_born"] == 1 and rows[1]["max_V_j"] == pytest.approx(3.0)
    (cycle,) = collapse_table.per_cycle(run)
    assert cycle["max_V_j"] == pytest.approx(3.0)
    summary = collapse_table.summarise(run)
    assert summary["max_V_j"] == pytest.approx(3.0)
    assert summary["max_We_j"] == pytest.approx(np.nanmax((9.0 * tab["r_j"])[born]))
    # a record restarted while that base closes: the jet was born before it, in no episode of it
    i = int(np.argmax(born)) + 5
    rows = collapse_table.per_episode(_write_run(
        tmp_path / "restart" / "out", {k: v[i:] for k, v in tab.items()}, restart="state.dump"))
    assert rows[0]["jet_born"] == 0 and math.isnan(rows[0]["max_V_j"])
    # the jet goes (r_j = 0 on the row the driver writes then) and a later cycle launches a
    # thinner, faster jet: the summary describes the first jet and counts both
    two = {k: v.copy() for k, v in tab.items()}
    gone = int(np.flatnonzero(born)[-1]) + 1
    two["jet_born"][gone], two["r_j"][gone] = 1.0, 0.0
    two["cycle"][gone + 1:] = 1
    second = (t > 4.5 * math.pi + 0.3) & (t < 4.5 * math.pi + 0.8)
    two["jet_born"][second] = 1.0
    two["r_j"][second] = 0.005 + 0.01 * (t[second] - 4.5 * math.pi - 0.3)
    two["V_j"][second] = 30.0
    two["We_j"] = two["V_j"] ** 2 * two["r_j"]
    summary = collapse_table.summarise(_write_run(tmp_path / "two" / "out", two))
    assert summary["n_jets"] == 2 and summary["t_birth"] == pytest.approx(t[born][0])
    assert summary["max_V_j"] == pytest.approx(3.0)
    assert summary["min_r_j"] == pytest.approx(np.nanmin(tab["r_j"][born]))
    assert summary["max_We_j"] == pytest.approx(np.nanmax((9.0 * tab["r_j"])[born]))
    assert summary["max_w_axis"] == pytest.approx(np.nanmax(tab["w_axis"][born]))


def _collapses_in_one_trough():
    """Trough 1 launches a fast jet (cycle 0) that stands through crest 2 into trough 3, where it
    goes; in trough 3 a new cavity launches a slower jet (cycle 1) and, once that has gone, a third
    collapse launches none (cycle 2). The run ends in crest 4."""
    t = np.round(np.arange(0.0, 6.5, 0.01), 10)
    eta = np.where((t < 1.0) | ((t >= 2.5) & (t < 3.5)) | (t >= 6.0), 0.3, -0.3)
    cycle = np.searchsorted([4.0, 5.5], t, side="right")
    height, t_peak = np.array([50.0, 80.0, 120.0])[cycle], np.array([1.8, 4.6, 5.8])[cycle]
    born = ((t >= 2.0) & (t < 4.0)) | ((t >= 4.8) & (t < 5.5))
    kappa = np.where(born, -5.0, height * np.exp(-((t - t_peak) / 0.1) ** 2))
    kappa[(t >= 5.3) & (t < 5.5)] = 200.0  # a trough on the axis as the second jet subsides
    r_j = np.where(born, 0.01 + 0.1 * (t - np.where(cycle == 0, 2.0, 4.8)), 0.0)
    V_j = np.where(born, np.where(cycle == 0, 10.0, 4.0), np.nan)
    # the rows at which a jet has gone, written before the next cycle: r_j = 0, or a flattened bump
    r_j[t == 3.99], V_j[t == 3.99] = 0.0, np.nan
    r_j[t == 5.49], V_j[t == 5.49] = 0.05, 50.0
    return {"step": np.arange(t.size), "cycle": cycle.astype(float), "time": t,
            "z_axis": 2.0 + eta, "w_axis": np.ones_like(t), "axis_curvature": kappa,
            "cavity": (t >= 1.0).astype(float), "jet_born": born.astype(float), "r_j": r_j,
            "V_j": V_j, "We_j": V_j**2 * r_j, "r_neck": np.full_like(t, np.nan),
            "remesh_count": np.zeros_like(t), "volume": np.ones_like(t)}


def test_per_episode_splits_a_trough_by_collapse_cycle(tmp_path):
    tab = _collapses_in_one_trough()
    run = _write_run(tmp_path / "multi" / "out", tab, stop={"reason": "end-time"})
    rows = {(r["episode"], r["cycle"]): r for r in collapse_table.per_episode(run)}
    assert list(rows) == [(1, 0), (3, 0), (3, 1), (3, 2)]
    first, standing, second, third = rows.values()
    # each jet with the collapse before it, and the speed of that jet only
    assert first["jet_born"] == 1 and first["t_birth"] == pytest.approx(2.0)
    assert first["max_apex_curvature"] == pytest.approx(50.0)
    assert first["t_max_apex_curvature"] == pytest.approx(1.8)
    assert first["max_V_j"] == pytest.approx(10.0)
    assert second["jet_born"] == 1 and second["t_birth"] == pytest.approx(4.8)
    assert second["max_apex_curvature"] == pytest.approx(80.0)
    assert second["t_max_apex_curvature"] == pytest.approx(4.6)
    assert second["max_V_j"] == pytest.approx(4.0)  # not the faster jet still standing before it
    # the collapse after the second jet has its own row, without a jet
    assert third["jet_born"] == 0 and math.isnan(third["max_V_j"])
    assert third["max_apex_curvature"] == pytest.approx(120.0)
    assert third["t_max_apex_curvature"] == pytest.approx(5.8)
    assert third["stop"] == "end-time" and first["stop"] == second["stop"] == ""
    # the first jet still standing when the trough began: no birth
    assert standing["jet_born"] == 0 and math.isnan(standing["max_apex_curvature"])
    assert all(r["h_crest"] == pytest.approx(0.3) and r["partial"] == 0 for r in rows.values())
    cycles = collapse_table.per_cycle(run)
    assert [c["max_V_j"] for c in cycles[:2]] == pytest.approx([10.0, 4.0])
    assert math.isnan(cycles[2].get("max_V_j", math.nan))
    summary = collapse_table.summarise(run)
    assert summary["n_jets"] == 2 and summary["max_V_j"] == pytest.approx(10.0)
    assert summary["min_r_j"] == pytest.approx(0.01)


def test_per_episode_marks_a_trough_in_progress_at_a_restart_partial(tmp_path):
    s = _settings(t_end=1.0, dt_max=0.01, state_dt=0.1, state_keep=20)
    first = tmp_path / "a" / "out"
    drv = PrescribedDriver(FakeProblem(), s, first, _wave_and_collapse)
    drv.run(1e-9)
    drv.close()
    _write_run(first, None, depth=2 * math.pi, stop={"reason": "end-time"})
    full = {r["episode"]: r for r in collapse_table.per_episode(first)}
    metas = [json.loads(f.read_text()) for f in sorted((first / "states").glob("*.json"))]
    meta = next(m for m in metas if m["episodes"]["kind"] == -1)  # inside a trough
    pb = FakeProblem()
    pb.t = meta["time"]  # load_state
    out = tmp_path / "b" / "out"
    drv = PrescribedDriver(pb, s, out, _wave_and_collapse)
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    _write_run(out, None, restart="state.dump", depth=2 * math.pi, stop={"reason": "end-time"})
    rows = collapse_table.per_episode(out)
    k = meta["episodes"]["index"]
    assert rows[0]["episode"] == k and rows[0]["source"] == "recorded"
    assert rows[0]["partial"] == 1 and rows[0]["t_start"] < meta["time"]
    # the dumped tracker still knows the crest before the trough
    assert rows[0]["h_crest"] == full[k]["h_crest"] and math.isfinite(rows[0]["h_crest"])
    assert len(rows) > 1 and all(r["partial"] == 0 for r in rows[1:])


def test_per_episode_reads_recorded_episodes(tmp_path):
    pb = FakeProblem()
    out = tmp_path / "new" / "out"
    drv = PrescribedDriver(pb, _settings(t_end=1.0, dt_max=0.01), out, _wave_and_collapse)
    drv.run(1e-9)
    drv.close()
    _write_run(out, None, depth=pb.nd.depth, stop={"reason": "end-time"})
    recorded = collapse_table.per_episode(out)
    assert len(recorded) >= 2 and all(r["source"] == "recorded" for r in recorded)
    assert recorded[0]["h_crest"] == pytest.approx(0.5, abs=0.02)
    # the same rows as segmenting the record again
    tab = _rows(out)
    del tab["episode"]
    copy = _write_run(tmp_path / "copy" / "out", tab, depth=pb.nd.depth,
                      stop={"reason": "end-time"})
    again = collapse_table.per_episode(copy)
    assert len(again) == len(recorded)
    for a, b in zip(recorded, again, strict=True):
        assert a["source"] == "recorded" and b["source"] == "reconstructed"
        for k in ("episode", "partial", "jet_born", "stop"):
            assert a[k] == b[k], k
        for k in ("t_start", "h_crest", "t_crest", "min_z_axis", "max_apex_curvature",
                  "max_w_axis"):
            assert a[k] == pytest.approx(b[k], rel=1e-11, nan_ok=True), k


def _jet_through_the_crest(t):
    """A crest, then a trough whose jet rises through the next crest to 1.7 above the mean level
    and falls into the trough after it, where it goes; a smooth crest and a trough follow."""
    depth = 2 * math.pi
    if t < 0.1 or 0.6 <= t < 0.7:
        return {"z_axis": depth + (0.3 if t < 0.1 else 0.4), "axis_lowest": False,
                "axis_curvature": -1.0}
    if 0.2 <= t < 0.5:
        eta = -0.3 + 10.0 * (t - 0.2) if t < 0.4 else 1.7 - 20.0 * (t - 0.4)
        return {"z_axis": depth + eta, "axis_lowest": False, "axis_curvature": -5.0,
                "jet_born": True, "r_j": 0.05, "V_j": 2.0}
    return {"z_axis": depth - 0.3}


def _strip_crest_jet(run: Path):
    """The events as a driver that predates ``crest_jet`` wrote them."""
    events = [{k: v for k, v in e.items() if k != "crest_jet"} for e in _read_events(run)]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_a_crest_with_a_jet_gives_no_wave_height(tmp_path):
    s = _settings(t_end=0.8, dt_max=0.01, state_dt=0.0, state_keep=1000)
    out = tmp_path / "a" / "out"
    drv = PrescribedDriver(FakeProblem(), s, out, _jet_through_the_crest)
    drv.run(1e-9)
    drv.close()
    _write_run(out, None, depth=2 * math.pi, stop={"reason": "end-time"})
    troughs = [e for e in _read_events(out) if e["event"] == "trough"]
    assert [(e["episode"], e["crest_jet"]) for e in troughs] == [(1, False), (3, True), (5, False)]
    assert troughs[1]["h_crest"] == pytest.approx(1.7)  # the jet tip, not a wave
    flags = {(1, 0): 0, (3, 0): 1, (3, 1): 1, (5, 1): 0}
    rows = collapse_table.per_episode(out)
    assert {(r["episode"], r["cycle"]): r["crest_jet"] for r in rows} == flags
    # the same from the rows of each crest, for events that predate the flag and for a run
    # segmented again
    _strip_crest_jet(out)
    rows = collapse_table.per_episode(out)
    assert {(r["episode"], r["cycle"]): r["crest_jet"] for r in rows} == flags
    tab = _rows(out)
    del tab["episode"]
    again = collapse_table.per_episode(_write_run(tmp_path / "b" / "out", tab, depth=2 * math.pi))
    assert all(r["source"] == "reconstructed" for r in again)
    assert {(r["episode"], r["cycle"]): r["crest_jet"] for r in again} == flags
    # per cycle: the jet of cycle 0 stands in the mode period before the cavity of cycle 1
    assert [c["crest_jet"] for c in collapse_table.per_cycle(out)] == [0, 1]
    # a restart inside trough 3 carries the flag of its crest; without it, the crest began before
    # the record and the flag is unknown
    meta = json.loads((out / "states" / "state_00000055.json").read_text())
    assert meta["episodes"]["index"] == 3 and meta["episodes"]["crest_jet"] is True
    pb = FakeProblem()
    pb.t = meta["time"]
    rest = tmp_path / "c" / "out"
    drv = PrescribedDriver(pb, s, rest, _jet_through_the_crest)
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    _write_run(rest, None, restart="state.dump", depth=2 * math.pi)
    assert _read_events(rest)[0]["crest_jet"] is True
    first = collapse_table.per_episode(rest)[0]
    assert (first["episode"], first["partial"], first["crest_jet"]) == (3, 1, 1)
    _strip_crest_jet(rest)
    assert math.isnan(collapse_table.per_episode(rest)[0]["crest_jet"])
    # a restart inside the smooth crest 4: known from the dumped tracker, unknown from the rows
    meta = json.loads((out / "states" / "state_00000065.json").read_text())
    assert meta["episodes"]["index"] == 4
    pb = FakeProblem()
    pb.t = meta["time"]
    rest = tmp_path / "d" / "out"
    drv = PrescribedDriver(pb, s, rest, _jet_through_the_crest)
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    _write_run(rest, None, restart="state.dump", depth=2 * math.pi)
    (last,) = collapse_table.per_episode(rest)
    assert (last["episode"], last["partial"], last["crest_jet"]) == (5, 0, 0)
    _strip_crest_jet(rest)
    assert math.isnan(collapse_table.per_episode(rest)[0]["crest_jet"])
    # a jet that appears during a crest flags it
    tracker = EpisodeTracker()
    events = [tracker.update(ti, ei, ji) for ti, ei, ji in
              [(0.0, 0.3, False), (0.1, 0.5, True), (0.2, 0.3, False), (0.3, -0.3, False)]]
    assert events[-1]["event"] == "trough" and events[-1]["crest_jet"] is True
    # a tracker state that predates the flag, restored in crest 4: unknown, unless a jet is seen
    old = {k: v for k, v in meta["episodes"].items() if k not in ("crest_jet", "crest_has_jet")}
    assert EpisodeTracker.from_dict(old).update(0.7, -0.3)["crest_jet"] is None
    tracker = EpisodeTracker.from_dict(old)
    tracker.update(0.66, 0.5, True)
    assert tracker.update(0.7, -0.3)["crest_jet"] is True
    tracker.update(0.8, 0.4)  # a crest that begins after the restore is known
    assert tracker.update(0.9, -0.3)["crest_jet"] is False


def test_table_command_line(tmp_path, monkeypatch, capsys):
    tab = _old_style_table()
    tab["cycle"] = np.zeros_like(tab["time"])
    tab["cavity"] = np.ones_like(tab["time"])
    run = _write_run(tmp_path / "old" / "out", tab, stop={"reason": "end-time"})
    out = tmp_path / "episodes.csv"
    monkeypatch.setattr(sys, "argv", ["collapse_table.py", str(run), "--out", str(out),
                                      "--per-episode"])
    assert collapse_table.main() == 0
    lines = out.read_text().splitlines()
    assert lines[0].split(",") == list(collapse_table.EPISODE_FIELDS) and len(lines) == 4
    monkeypatch.setattr(sys, "argv", ["collapse_table.py", str(run), "--out", str(out),
                                      "--per-cycle"])
    assert collapse_table.main() == 0
    assert out.read_text().splitlines()[0].split(",") == list(collapse_table.CYCLE_FIELDS)
    capsys.readouterr()


@pytest.mark.parametrize("mode", [[], ["--per-cycle"], ["--per-episode"]])
def test_header_only_run_does_not_block_table_command(tmp_path, monkeypatch, mode):
    tab = _old_style_table()
    tab["cycle"] = np.zeros_like(tab["time"])
    tab["cavity"] = np.ones_like(tab["time"])
    tab["remesh_count"] = np.zeros_like(tab["time"])
    tab["volume"] = np.ones_like(tab["time"])
    tab["We_j"] = tab["V_j"] ** 2 * tab["r_j"]
    valid = _write_run(tmp_path / "valid", tab, stop={"reason": "end-time"})
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "collapse.txt").write_text((valid / "collapse.txt").read_text().splitlines()[0] + "\n")
    assert collapse_table.read_table(empty / "collapse.txt")["time"].size == 0
    output = tmp_path / "combined.csv"
    monkeypatch.setattr(sys, "argv", ["collapse_table.py", str(empty), str(valid),
                                      "--out", str(output), *mode])
    assert collapse_table.main() == 0
    assert len(output.read_text().splitlines()) > 1


def test_one_ulp_before_large_endpoint_is_end_time(tmp_path):
    pb = SurfaceProblem(lambda t, r: 2 * math.pi - 0.1 + 0.5 * r**2)
    pb.t = math.nextafter(65.0, -math.inf)
    driver = CollapseDriver(pb, _settings(t_end=65.0), tmp_path)
    try:
        result = driver.run(1e-9)
        assert result["stop"]["reason"] == "end-time"
        assert pb.calls == 0
    finally:
        driver.close()


def test_per_episode_curvature_with_a_sticky_jet_flag(tmp_path):
    # tables without the cycle column keep jet_born on after the first birth
    tab = _old_style_table()
    tab["jet_born"] = (tab["time"] > 3 * math.pi + 0.5).astype(float)
    rows = collapse_table.per_episode(_write_run(tmp_path / "old" / "out", tab))
    last = rows[-1]
    assert last["jet_born"] == 0 and math.isfinite(last["max_apex_curvature"])
    sel = tab["time"] >= last["t_start"]
    assert last["max_apex_curvature"] == pytest.approx(tab["axis_curvature"][sel].max())


def _troughs_then_bump(t):
    """A crest, a sharp trough that does not collapse, a crest, the trough that launches the jet
    and, just before birth, a bump on the axis with a larger ``|z''(0)|`` than either trough."""
    depth = 2 * math.pi
    if t < 0.2 or 0.4 <= t < 0.6:
        return {"z_axis": depth + 0.1, "axis_lowest": False, "axis_curvature": -1.0}
    if t < 0.4:
        return {"z_axis": depth - 0.1, "axis_curvature": 30.0}
    if t < 0.8:
        return {"z_axis": depth - 0.1, "axis_curvature": 5.0 + 10.0 * (t - 0.6)}
    return {"z_axis": depth - 0.1, "axis_lowest": False, "axis_curvature": -50.0,
            "jet_born": t >= 0.85, "r_j": 0.05}


def test_jet_birth_carries_the_largest_trough_curvature_of_its_episode(tmp_path):
    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.9, dt_max=0.01), tmp_path,
                           _troughs_then_bump)
    drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path)
    assert [e["episode"] for e in events if e["event"] == "trough"] == [1, 3]
    born = next(e for e in events if e["event"] == "jet-born")
    assert 0.79 < born["max_curvature_time"] < 0.8
    assert born["max_curvature"] == pytest.approx(5.0 + 10.0 * (born["max_curvature_time"] - 0.6))


def _sharp_trough_then_jet(t):
    """A trough on the axis, sharp up to t = 0.25, and a jet born at t = 0.45."""
    if t < 0.45:
        return {"z_axis": 2 * math.pi - 0.1, "axis_curvature": 3.0 if t < 0.25 else 1.0}
    return {"z_axis": 2 * math.pi - 0.1, "axis_lowest": False, "axis_curvature": -5.0,
            "jet_born": True, "r_j": 0.05}


@pytest.mark.parametrize("marked", [True, False])
def test_restore_keeps_the_trough_curvature_only_from_marked_dumps(tmp_path, marked):
    s = _settings(t_end=0.5, dt_max=0.01, state_dt=0.0, state_keep=1000)
    drv = PrescribedDriver(FakeProblem(), s, tmp_path / "a", _sharp_trough_then_jet)
    drv.run(1e-9)
    drv.close()
    full = next(e for e in _read_events(tmp_path / "a") if e["event"] == "jet-born")
    assert full["max_curvature"] == 3.0 and full["max_curvature_time"] == pytest.approx(0.01)
    # a dump inside the trough episode, before birth
    meta = json.loads((tmp_path / "a" / "states" / "state_00000030.json").read_text())
    assert meta["episodes"]["kind"] == -1 and not meta["born"]
    assert meta["max_curvature_kind"] == "trough-episode"
    if not marked:  # an older dump: its value may be |z''(0)| of a bump in an earlier episode
        del meta["max_curvature_kind"]
        meta["max_curvature"], meta["max_curvature_time"] = 4.2, 0.005
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, s, tmp_path / "b", _sharp_trough_then_jet)
    drv.restore(meta)
    status = drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path / "b")
    assert events[0]["event"] == "restart" and events[0]["max_curvature_reset"] is (not marked)
    born = next(e for e in events if e["event"] == "jet-born")
    if marked:  # as without the restart
        assert born["max_curvature"] == 3.0 and born["max_curvature_time"] == pytest.approx(0.01)
    else:  # only the trough measured after the restart
        assert born["max_curvature"] == 1.0 and born["max_curvature_time"] > meta["time"]
    assert status["max_axis_curvature_before_birth"] == born["max_curvature"]


def test_local_scale_grows_by_at_most_the_factor_per_step():
    # a bump vanishing at a new cycle: the measured scale jumps from 0.006 to the cap 0.4
    assert limit_scale(0.4, 0.006, 1.25) == pytest.approx(0.0075)
    ell, steps = 0.006, 0
    while ell < 0.4:
        ell, steps = limit_scale(0.4, ell, 1.25), steps + 1
    assert steps == 19 and ell == 0.4
    assert limit_scale(0.001, 0.006, 1.25) == 0.001  # shrinking is never limited
    assert limit_scale(0.4, math.nan, 1.25) == 0.4  # first step and old dumps


# steps of 0.01 throughout: the capillary and Courant limits are far above dt_max on a surface at
# rest, and the scale never shrinks
_SCALE_SETTINGS = dict(t_end=0.4, dt_max=0.01, capillary=1e3, state_dt=0.0, state_keep=1000)


def _restore(tmp_path, s, src: Path, step: int, out: str, drop=()):
    """A driver on a fresh surface, restored from the dump of ``step`` in run ``src``."""
    meta = json.loads((src / "states" / f"state_{step:08d}.json").read_text())
    for k in drop:
        del meta[k]
    pb = SurfaceProblem(_sharp_then_flat_trough, finest=0.002)
    pb.load_state(src / "states" / f"state_{step:08d}.dump")
    drv = CollapseDriver(pb, s, tmp_path / out)
    drv.restore(meta)
    return drv, pb


def test_driver_limits_the_growth_of_the_measured_scale(tmp_path):
    # the real measurement: the apex radius of curvature 0.006 sets the scale until the trough
    # flattens after step 4, when the measured scale jumps to the cap h_axis elements_per_scale
    s = _settings(**_SCALE_SETTINGS)
    drv = CollapseDriver(SurfaceProblem(_sharp_then_flat_trough, finest=0.002), s, tmp_path / "a")
    drv.run(1e-9)
    drv.close()
    full = _rows(tmp_path / "a")
    ell, n = full["ell"], full["step"]
    assert np.allclose(full["dt"][1:], 0.01) and n[-1] == 40
    expected = np.minimum(0.006 * s.scale_growth ** np.maximum(n - 4, 0), 0.4)
    assert np.allclose(ell, expected, rtol=1e-12)
    assert np.all(ell[1:] <= s.scale_growth * ell[:-1] * (1 + 1e-12))
    assert np.allclose(full["h_focus"], ell / s.elements_per_scale, rtol=1e-12)
    # a restart continues the growth from the dumped scale: at the jump and during the growth
    for step in (5, 10):
        drv, _ = _restore(tmp_path, s, tmp_path / "a", step, f"b{step}")
        restart = _read_events(tmp_path / f"b{step}")[0]
        assert restart["ell"] == pytest.approx(expected[step], rel=1e-12)
        drv.run(1e-9)
        drv.close()
        rest = _rows(tmp_path / f"b{step}")
        for name in ("step", "time", "dt", "ell", "h_focus", "h_near", "remesh_count"):
            first = 1 if name == "dt" else 0  # the restart row records no step
            assert np.array_equal(rest[name][first:], full[name][step + first:]), name
    # a dump without the previous scale restores an unlimited one
    drv, _ = _restore(tmp_path, s, tmp_path / "a", 5, "c", drop=("ell_prev",))
    drv.close()
    assert _read_events(tmp_path / "c")[0]["ell"] == pytest.approx(0.4)


def test_rebuilds_coarsen_the_focus_by_a_bounded_factor(tmp_path):
    # the third refinement, to a focus finer than 0.002, does not take effect and blocks refining
    # for remesh_backoff steps; the scale then grows from 0.006 to the cap 0.4
    s = _settings(**_SCALE_SETTINGS)
    drv = CollapseDriver(SurfaceProblem(_sharp_then_flat_trough, finest=0.002), s, tmp_path)
    drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path)
    (failed,) = [e["step"] for e in events if e["event"] == "remesh-ineffective"]
    remesh = [e for e in events if e["event"] == "remesh"]
    coarsen = [e for e in remesh if e["focus_size"] > e["h_near_before"]]
    assert len(coarsen) >= 3
    for e in coarsen:
        assert e["focus_size"] <= s.max_coarsen_factor * e["h_near_before"] * (1 + 1e-12)
    # coarsening follows the growing scale during the backoff of refinement
    assert coarsen[0]["step"] < failed + s.remesh_backoff
    assert not [e for e in remesh if failed < e["step"] <= failed + s.remesh_backoff
                and e["focus_size"] < e["h_near_before"]]
    tab = _rows(tmp_path)
    assert tab["h_near"][-1] >= s.coarsen_ratio * tab["h_focus"][-1]  # the focus has caught up


def test_restore_keeps_an_old_backoff_for_refinement_only(tmp_path):
    # a dump of a driver that backed off by moving the last rebuild into the future
    meta = {"step": 100, "cycle": 0, "cavity_seen": True, "born": False, "neck_seen": False,
            "max_curvature": 0.0, "max_curvature_time": math.nan, "snapshots": 3, "time": 0.5,
            "remesh_count": 7, "focus": None, "dt_prev": 0.01, "last_remesh_step": 110}
    pb = FakeProblem()
    pb.t = meta["time"]
    s = _settings()
    drv = PrescribedDriver(pb, s, tmp_path, lambda t: {})
    drv.restore(meta)
    drv.close()
    assert drv.state.last_remesh_step == 110 - s.remesh_backoff
    assert drv.state.refine_blocked_until == 110 + s.min_steps_between_remesh



# --- the optional stop once a born jet is unambiguous ----------------------------------------


def _jet_rising_after(t, t_birth=0.5, r_j=0.1, speed=1.0):
    """A trough until ``t_birth``, then a jet of base radius ``r_j`` born standing 0.11 r_j above its
    base, whose apex rises at ``speed``: its aspect is 0.11 + 10 (t - t_birth)."""
    if t < t_birth:
        return {"axis_curvature": 2.0}
    return {"jet_born": True, "axis_lowest": False, "axis_curvature": -1.0, "r_j": r_j,
            "z_b": 0.0, "z_axis": 0.11 * r_j + speed * (t - t_birth)}


def _birth_and_stop(tmp_path, **kw):
    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.7, dt_max=0.01, **kw), tmp_path,
                           _jet_rising_after)
    result = drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path)
    born = next(e for e in events if e["event"] == "jet-born")
    return result, born, events[-1]


def test_jet_stop_is_off_by_default(tmp_path):
    result, _, stop = _birth_and_stop(tmp_path)
    assert result["stop"] == {"reason": "end-time"} and stop["reason"] == "end-time"
    assert math.isinf(CollapseSettings(t_end=1.0, dt_max=0.01).post_birth_aspect)


def test_jet_stop_once_the_jet_stands_above_the_threshold(tmp_path):
    result, born, stop = _birth_and_stop(tmp_path, post_birth_aspect=0.5)
    # the aspect passes 0.5 at t - t_birth > 0.039: the fourth step of 0.01 after birth
    assert result["stop"]["reason"] == "jet-confirmed" and result["stop"]["trigger"] == "jet-aspect"
    assert result["steps"] - born["step"] == 4 == stop["step"] - result["stop"]["born_step"]
    assert result["stop"]["aspect"] > 0.5 and result["jet_born"]
    tab = _rows(tmp_path)
    aspect = (tab["z_axis"][-2:] - tab["z_b"][-2:]) / tab["r_j"][-2:]  # rows after birth
    assert aspect[-1] > 0.5 >= aspect[-2]


def test_jet_stop_waits_the_least_number_of_steps_after_birth(tmp_path):
    # a threshold the jet meets at birth: the stop still waits post_birth_steps accepted steps
    result, born, _ = _birth_and_stop(tmp_path, post_birth_aspect=0.05, post_birth_steps=6)
    assert result["stop"]["reason"] == "jet-confirmed"
    assert result["steps"] - born["step"] == 6


def test_neck_stop_takes_precedence_over_the_jet_stop(tmp_path):
    def fields(t):
        m = _jet_rising_after(t)
        if t >= 0.53:
            m["r_neck"], m["z_neck"] = 5e-4, 1.0
        return m

    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.7, dt_max=0.01, post_birth_aspect=0.1,
                                                    post_birth_steps=3), tmp_path, fields)
    result = drv.run(1e-9)
    drv.close()
    assert result["stop"]["reason"] == "pinch-off-approach"


@pytest.mark.parametrize("kw", [{"post_birth_aspect": 0.0}, {"post_birth_aspect": math.nan},
                                {"post_birth_aspect": True}, {"post_birth_steps": 0},
                                {"post_birth_steps": 1.5}])
def test_jet_stop_settings_are_validated(kw):
    with pytest.raises(ValueError):
        _settings(**kw)


def test_restore_keeps_the_birth_step(tmp_path):
    drv = PrescribedDriver(FakeProblem(), _settings(t_end=0.53, dt_max=0.01), tmp_path,
                           _jet_rising_after)
    drv.run(1e-9)
    drv.save_state()
    drv.close()
    born = next(e for e in _read_events(tmp_path) if e["event"] == "jet-born")
    meta = json.loads(sorted((tmp_path / "states").glob("*.json"))[-1].read_text())
    assert meta["born_step"] == born["step"]
    pb = FakeProblem()
    pb.t = meta["time"]
    out = tmp_path / "restored"
    again = PrescribedDriver(pb, _settings(t_end=0.7, dt_max=0.01, post_birth_aspect=0.5), out,
                             _jet_rising_after)
    again.restore(meta)
    result = again.run(1e-9)
    again.close()
    assert result["stop"]["reason"] == "jet-confirmed"
    assert result["stop"]["born_step"] == born["step"]


def _born_dump(step=50, time=0.55, **extra):
    return {"step": step, "cycle": 0, "cavity_seen": True, "born": True, "neck_seen": False,
            "max_curvature": 2.0, "max_curvature_time": 0.4, "snapshots": 3, "time": time,
            "remesh_count": 0, "focus": None, "dt_prev": 0.01, **extra}


def _restored(tmp_path, meta, **kw):
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, _settings(t_end=0.7, dt_max=0.01, **kw), tmp_path, _jet_rising_after)
    drv.restore(meta)
    result = drv.run(1e-9)
    drv.close()
    return result


def test_a_restart_that_already_qualifies_stops_before_advancing(tmp_path):
    # aspect 0.61 at t = 0.55, forty steps after birth: no further step is taken
    result = _restored(tmp_path, _born_dump(born_step=10), post_birth_aspect=0.5)
    assert result["stop"]["reason"] == "jet-confirmed" and result["steps"] == 50


def test_a_dump_without_the_birth_step_waits_from_the_restart(tmp_path):
    result = _restored(tmp_path, _born_dump(), post_birth_aspect=0.5, post_birth_steps=3)
    assert result["stop"]["reason"] == "jet-confirmed"
    assert result["steps"] == 53 and result["stop"]["born_step"] == 50


def test_jet_stop_is_checked_after_a_remesh(tmp_path):
    # the jet stands still at aspect 0.11 until a rebuild at t >= 0.53 re-reads it at 0.2
    pb = FakeProblem()

    def fields(t):
        if t < 0.5:
            return {"axis_curvature": 2.0}
        rebuilt = pb.remesh_count > 0
        return {"jet_born": True, "axis_lowest": False, "axis_curvature": -1.0, "r_j": 0.1,
                "z_b": 0.0, "z_axis": 0.02 if rebuilt else 0.011,
                "h_near": 0.1 if t >= 0.53 and not rebuilt else 0.05}

    drv = PrescribedDriver(pb, _settings(t_end=0.7, dt_max=0.01, post_birth_aspect=0.15), tmp_path,
                           fields)
    result = drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path)
    (remesh,) = [e for e in events if e["event"] == "remesh"]
    assert result["stop"]["reason"] == "jet-confirmed"
    assert result["steps"] == remesh["step"] and result["stop"]["aspect"] == pytest.approx(0.2)


# --- landing states for a study in the cut radius ------------------------------------------------


def _closing_neck(t):
    """A neck closing at a constant rate, by 0.15 to 1.2 % of its radius per step of 1e-3."""
    return {"r_neck": 2e-3 - 3e-3 * t, "z_neck": 1.0}


def _landing_settings():
    return _settings(t_end=1.0, dt_max=1e-3, neck_stop=1e-4, h_floor=1e-6, state_dt=0.1,
                     state_keep=100)


def test_landing_states_for_a_list_of_cut_radii(tmp_path):
    from faraday_singularity.reconnect import LANDING_RATIO, MAX_LANDING_RATIO

    deltas = (2.5e-4, 1e-3, 5e-4)
    drv = PrescribedDriver(FakeProblem(), _landing_settings(), tmp_path, _closing_neck,
                           landing=deltas)
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == "pinch-off-approach"
    landings = [e for e in _read_events(tmp_path) if e["event"].startswith("landing")]
    assert [e["event"] for e in landings] == ["landing"] * 3
    assert [e["delta"] for e in landings] == [1e-3, 5e-4, 2.5e-4]
    for e in landings:
        assert LANDING_RATIO <= e["delta_over_r_neck"] < MAX_LANDING_RATIO
        # the first state past delta / LANDING_RATIO: the step before had not reached it
        assert e["r_neck"] + 3e-6 > e["delta"] / LANDING_RATIO
        meta = json.loads(Path(e["path"]).with_suffix(".json").read_text())
        assert Path(e["path"]).name == f"landing_{e['delta']:.6g}.dump"
        assert meta["landing"]["delta"] == e["delta"] and meta["time"] == e["time"]
        assert e["delta"] in meta["landed"]
    # rotating dumps carry the radii kept so far
    rotating = [json.loads(f.read_text()) for f in sorted((tmp_path / "states").glob("state_*.json"))]
    assert rotating and all("landed" in m for m in rotating)


def test_landing_after_a_restart_and_a_missed_landing(tmp_path):
    s = _landing_settings()
    drv = PrescribedDriver(FakeProblem(), s, tmp_path / "a", _closing_neck, landing=(1e-3, 5e-4))
    drv.run(1e-9)
    drv.close()
    meta = json.loads((tmp_path / "a" / "states" / "landing_0.001.json").read_text())
    assert meta["landed"] == [1e-3]
    # resumed from the first landing state with one more radius: the kept one is not kept again
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, s, tmp_path / "b", _closing_neck, landing=(1e-3, 5e-4, 2.5e-4))
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    kept = sorted(f.name for f in (tmp_path / "b" / "states").glob("landing_*.dump"))
    assert kept == ["landing_0.00025.dump", "landing_0.0005.dump"]
    full, rest = _rows(tmp_path / "a"), _rows(tmp_path / "b")
    k = int(np.searchsorted(full["step"], meta["step"]))
    for name in ("step", "cycle", "time", "dt", "r_neck"):  # the run continues as it was
        assert np.array_equal(rest[name][1:], full[name][k + 1:], equal_nan=True), name
    # a radius the neck has already passed by more than MAX_LANDING_RATIO is no landing
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, s, tmp_path / "c", _closing_neck, landing=(1.2e-3,))
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    missed = [e for e in _read_events(tmp_path / "c") if e["event"].startswith("landing")]
    assert [e["event"] for e in missed] == ["landing-missed"]
    assert missed[0]["delta_over_r_neck"] > missed[0]["max_landing_ratio"]
    assert not list((tmp_path / "c" / "states").glob("landing_*"))


def test_dumps_without_landing_have_no_landing_record(tmp_path):
    drv = PrescribedDriver(FakeProblem(), _landing_settings(), tmp_path, _closing_neck)
    drv.run(1e-9)
    drv.close()
    metas = [json.loads(f.read_text()) for f in (tmp_path / "states").glob("*.json")]
    assert metas and all("landed" not in m for m in metas)
    # nor the other keys of a reconnection: the dumps of other runs keep their keys
    assert all("neck_samples" not in m and "resume_at" not in m for m in metas)
    assert not any(e["event"].startswith("landing") for e in _read_events(tmp_path))


def test_landing_radii_below_the_neck_stop_are_refused(tmp_path):
    """A run stops at the first state with r_n < max(neck_stop, neck_stop_elements h_floor); a cut
    radius below LANDING_RATIO times that can never land, and is refused before any output."""
    s = _settings(neck_stop=3e-4, h_floor=2e-5)
    with pytest.raises(ValueError, match=r"0\.00025, 0\.000125"):
        PrescribedDriver(FakeProblem(), s, tmp_path / "a", _closing_neck,
                         landing=(1e-3, 5e-4, 2.5e-4, 1.25e-4))
    assert not (tmp_path / "a").exists()
    s = _settings(neck_stop=1e-4, h_floor=1e-4)  # the limit is neck_stop_elements h_floor = 3e-4
    with pytest.raises(ValueError, match="neck_stop"):
        CollapseDriver(FakeProblem(), s, tmp_path / "b", landing=(3e-4,))
    CollapseDriver(FakeProblem(), s, tmp_path / "c", landing=(3.2e-4,)).close()


def test_radii_left_at_the_stop_are_recorded_and_landed_by_a_continuation(tmp_path):
    s = _settings(t_end=0.4, dt_max=1e-3, neck_stop=1e-4, h_floor=1e-6, state_dt=0.1,
                  state_keep=100)
    drv = PrescribedDriver(FakeProblem(), s, tmp_path / "a", _closing_neck, landing=(1e-3, 5e-4))
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == "end-time"
    landings = [e for e in _read_events(tmp_path / "a") if e["event"].startswith("landing")]
    assert [(e["event"], e["delta"]) for e in landings] == [("landing", 1e-3),
                                                           ("landing-missed", 5e-4)]
    assert landings[1]["reason"] == "stopped-first" and landings[1]["stop_reason"] == "end-time"
    assert out["landing"] == {"radii": [1e-3, 5e-4], "kept": [1e-3],
                              "missed": [{"delta": 5e-4, "reason": "stopped-first"}]}
    # the radius left at the stop is not recorded as landed: a continuation lands it
    assert drv.state.landed == [1e-3]
    meta = json.loads((tmp_path / "a" / "states" / "landing_0.001.json").read_text())
    assert meta["landed"] == [1e-3]
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, _landing_settings(), tmp_path / "b", _closing_neck,
                           landing=(1e-3, 5e-4))
    drv.restore(meta)
    out = drv.run(1e-9)
    drv.close()
    assert out["landing"]["kept"] == [5e-4] and out["landing"]["missed"] == []
    assert (tmp_path / "b" / "states" / "landing_0.0005.dump").is_file()


def test_a_rebuild_that_narrows_the_neck_below_the_stop_still_lands(tmp_path):
    """The smallest admitted radius, delta = 1.06 neck_limit: the neck is just wider than
    delta/1.05 after the step, and the rebuild of that step measures it below the neck limit, which
    stops the run. The state after the rebuild is offered for landing too."""
    s = _landing_settings()  # neck limit 1e-4
    delta = 1.06e-4
    pb = FakeProblem()

    def fields(t):
        r = 2e-3 - 3e-3 * t
        return {"r_neck": r - (2e-6 if pb.remesh_count else 0.0), "z_neck": 1.0,
                "h_near": 0.2 if r <= 1.015e-4 else 0.05}

    drv = PrescribedDriver(pb, s, tmp_path, fields, landing=(delta,))
    out = drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path)
    (remesh,) = [e for e in events if e["event"] == "remesh"]
    (landing,) = [e for e in events if e["event"] == "landing"]
    assert out["stop"]["reason"] == "pinch-off-approach" and landing["step"] == remesh["step"]
    assert landing["r_neck"] < s.neck_limit and 1.05 <= landing["delta_over_r_neck"] <= 1.1
    meta = json.loads(Path(landing["path"]).with_suffix(".json").read_text())
    assert "resume_at" not in meta and meta["remesh_count"] == 1  # kept after the rebuild


def _neck_with_rebuild_from(r_from: float, strict: bool = False):
    """``_closing_neck`` on a surface coarser than the focus wants once the neck is narrower than
    ``r_from`` (``<=``, or ``<`` with ``strict``): a refining rebuild is due from that step on, and
    does not take effect (``remesh-ineffective``), so it is tried again after the backoff."""
    def fields(t):
        f = _closing_neck(t)
        if f["r_neck"] < r_from or (not strict and f["r_neck"] == r_from):
            f["h_near"] = 0.2
        return f
    return fields


def test_restart_from_a_landing_state_makes_its_remesh_check_first(tmp_path):
    """The neck passes delta/1.05 at a step whose remesh check rebuilds the mesh. The landing state
    is kept before that check (``resume_at``), and a restart from it makes the check before its
    first step, as the parent did: both step from the rebuilt mesh, and the rebuilds after it fall
    on the same steps."""
    from faraday_singularity.reconnect import LANDING_RATIO

    s = _landing_settings()
    delta = 1e-3
    fields = _neck_with_rebuild_from(delta / LANDING_RATIO)
    pb = FakeProblem()
    drv = PrescribedDriver(pb, s, tmp_path / "a", fields, landing=(delta,))
    drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path / "a")
    (landing,) = [e for e in events if e["event"] == "landing"]
    remesh = [e["step"] for e in events if e["event"] == "remesh"]
    assert remesh[0] == landing["step"] and len(remesh) >= 3
    meta = json.loads(Path(landing["path"]).with_suffix(".json").read_text())
    assert meta["resume_at"] == "remesh-check" and meta["remesh_count"] == 0
    pb2 = FakeProblem()
    pb2.t = meta["time"]  # load_state
    drv2 = PrescribedDriver(pb2, s, tmp_path / "b", fields, landing=(delta,))
    drv2.restore(meta)
    drv2.run(1e-9)
    drv2.close()
    resumed = _read_events(tmp_path / "b")
    assert resumed[0]["event"] == "restart" and resumed[0]["resume_at"] == "remesh-check"
    assert [e["step"] for e in resumed if e["event"] == "remesh"] == remesh
    full, rest = _rows(tmp_path / "a"), _rows(tmp_path / "b")
    k = int(np.searchsorted(full["step"], meta["step"]))
    for name in ("step", "time", "dt", "remesh_count", "r_neck"):
        first = 1 if name == "dt" else 0  # the restart row records no step
        assert np.array_equal(rest[name][first:], full[name][k + first:], equal_nan=True), name
    assert pb2.remesh_count == pb.remesh_count
    # rotating dumps are written at the end of a step: a restart from one steps at once
    rotating = [json.loads(f.read_text()) for f in (tmp_path / "a" / "states").glob("state_*.json")]
    assert rotating and all("resume_at" not in m for m in rotating)


def test_restart_from_a_stop_state_with_a_lower_neck_stop_matches_a_run_with_it(tmp_path):
    """Following the neck further from the kept stop state (a derived case with a lower
    ``neck_stop``) gives the run that had the lower stop from the start, rebuilds included: the
    stop state was kept before the remesh check of its step."""
    hi, lo = (_settings(t_end=1.0, dt_max=1e-3, neck_stop=x, h_floor=1e-6, state_dt=0.1,
                        state_keep=100) for x in (5e-4, 1e-4))
    fields = _neck_with_rebuild_from(5e-4, strict=True)  # from the step that stops the parent
    drv = PrescribedDriver(FakeProblem(), hi, tmp_path / "a", fields, keep_stop_state=True)
    out = drv.run(1e-9)
    drv.close()
    stop = Path(out["stop"]["kept_state"])
    meta = json.loads(stop.with_suffix(".json").read_text())
    assert meta["resume_at"] == "remesh-check" and meta["neck_samples"][-1][0] == meta["time"]
    # the stop state records the pinch-time fit of its neck samples (a linearly closing neck here)
    assert set(meta["pinch"]) >= {"t_p", "alpha", "t_window", "r_window", "samples"}
    assert meta["pinch"]["t_window"][1] == meta["time"]
    ref = PrescribedDriver(FakeProblem(), lo, tmp_path / "ref", fields)
    ref.run(1e-9)
    ref.close()
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, lo, tmp_path / "b", fields)
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    full, rest = _rows(tmp_path / "ref"), _rows(tmp_path / "b")
    k = int(np.searchsorted(full["step"], meta["step"]))
    for name in ("step", "time", "dt", "remesh_count", "r_neck"):
        first = 1 if name == "dt" else 0
        assert np.array_equal(rest[name][first:], full[name][k + first:], equal_nan=True), name
    steps = [e["step"] for e in _read_events(tmp_path / "ref") if e["event"] == "remesh"]
    assert steps and steps[0] == meta["step"]
    assert [e["step"] for e in _read_events(tmp_path / "b") if e["event"] == "remesh"] == steps


def test_pinch_time_from_the_neck_samples_of_a_landing_state(tmp_path):
    """The landing state carries the latest neck samples; the pinch time fitted to them over the
    last factor PINCH_FIT_SPAN of the radius recovers that of a power-law neck."""
    from faraday_singularity.driver import NECK_SAMPLES, PINCH_FIT_SPAN

    t_p, A, alpha = 0.9, 4e-3, 0.56

    def fields(t):
        return {"r_neck": A * (t_p - t) ** alpha, "z_neck": 1.0}

    s = _landing_settings()
    drv = PrescribedDriver(FakeProblem(), s, tmp_path / "a", fields, landing=(5e-4,))
    drv.run(1e-9)
    drv.close()
    (landing,) = [e for e in _read_events(tmp_path / "a") if e["event"] == "landing"]
    meta = json.loads(Path(landing["path"]).with_suffix(".json").read_text())
    samples = meta["neck_samples"]
    assert len(samples) == NECK_SAMPLES and samples[-1] == [landing["time"], landing["r_neck"]]
    pb = FakeProblem()
    pb.t = meta["time"]
    drv = PrescribedDriver(pb, s, tmp_path / "b", fields)
    drv.restore(meta)
    fit = drv.pinch_time()
    drv.close()
    assert fit["t_p"] == pytest.approx(t_p, abs=1e-3 * (t_p - landing["time"]))
    assert fit["alpha"] == pytest.approx(alpha, rel=1e-2)
    assert fit["t_window"][1] == landing["time"] and fit["r_window"][1] == landing["r_neck"]
    assert fit["r_window"][0] <= PINCH_FIT_SPAN * fit["r_window"][1] and fit["samples"] >= 10
    assert fit["samples"] < NECK_SAMPLES  # the window, not the whole series
    # the landing state and its event record that fit
    assert meta["pinch"] == {**fit, "t_window": list(fit["t_window"]),
                             "r_window": list(fit["r_window"])}
    assert landing["t_p"] == fit["t_p"] and landing["alpha_p"] == fit["alpha"]
    # no neck samples: no pinch time
    fresh = PrescribedDriver(FakeProblem(), s, tmp_path / "c", fields)
    assert math.isnan(fresh.pinch_time()["t_p"])
    fresh.close()


# --- the post-cut mode: a run after a cut, on two prescribed chains -------------------------------


R_C = 3.8317059702075125


class CutProblem(FakeProblem):
    """The two chains after a cut, at rest unless ``speed(chain, r, z)`` gives an axial speed: an
    open chain from a cap on the axis (tip radius of curvature 1e-3, at z = 4.69) to the wall, and
    an elliptic bubble below it from z = 4.68 - 2b to 4.68 (tip radius a^2/b), whose width is
    multiplied by ``width(t)``."""

    def __init__(self, a=0.01, b=0.2, speed=None, width=None, **kw):
        super().__init__(**kw)
        self.a, self.b = a, b
        self.speed = speed or (lambda chain, r, z: np.zeros_like(r))
        self.width = width or (lambda t: 1.0)
        self.cut = None

    def _chain(self, name, r, z, ends):
        return {"open": name == "open", "end_types": ends, "r": r, "z": z,
                "u_r": np.zeros_like(r), "u_z": self.speed(name, r, z)}

    def interface_segments(self):
        th = np.linspace(0.0, math.pi, 801)
        br = self.a * self.width(self.t) * np.sin(th)
        br[[0, -1]] = 0.0
        bz = 4.68 - self.b - self.b * np.cos(th)
        phi = np.linspace(0.0, 0.5 * math.pi, 201)
        A, B = 2e-3, 4e-3  # tip radius A^2/B
        cr, cz = A * np.sin(phi), 4.69 + B * (1.0 - np.cos(phi))
        s = np.linspace(0.0, 1.0, 401)[1:]
        line = np.column_stack([A + (0.5 - A) * s, cz[-1] + (5.0 - cz[-1]) * s])
        s = np.linspace(0.0, 1.0, 101)[1:]
        wall = np.column_stack([0.5 + (R_C - 0.5) * s, 5.0 + s])
        P = np.vstack([np.column_stack([cr, cz]), line, wall])
        return [self._chain("bubble", br, bz, ("axis", "axis")),
                self._chain("open", P[:, 0], P[:, 1], ("axis", "fixed"))]

    def interface_arrays(self):
        c = self.interface_segments()[1]
        return c["r"], c["z"], c["u_r"], c["u_z"]

    def axial_velocity_at(self, points):
        return [0.0] * len(points), True

    def cut_observables(self):
        return {"volume": 1.0, "kinetic_energy": 0.0, "momentum_z": 0.0, "divergence_sq": 0.0}


def _cut_driver(tmp_path, pb, recon, **kw):
    """A driver in the post-cut mode, right after a cut at t = 0 with pinch time -1e-6."""
    from faraday_singularity.reconnect import Chain, cut_geometry

    drv = CollapseDriver(pb, _settings(**{"t_end": 1.0, "dt_max": 1.0, "max_steps": 3, **kw}),
                         tmp_path, reconnect=recon)
    chains = [Chain(np.column_stack([c["r"], c["z"]]), c["end_types"])
              for c in pb.interface_segments()]
    geom = cut_geometry(chains, 3.0, gas_before=1.0)
    pb.cut = {"delta": recon.delta, "rmin": recon.delta, "r_neck": 0.95e-3, "z_neck": 4.685,
              "t_cut": 0.0, "pinch": {"t_p": -1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": recon.tip_size * recon.delta},
              "settings": json.loads(json.dumps(recon.to_dict()))}
    drv.reset_after_cut(pb.cut)
    return drv


def _chains_rows(outdir: Path) -> dict[str, np.ndarray]:
    path = outdir / "chains.txt"
    cols = path.read_text().splitlines()[0][2:].split("\t")
    data = np.loadtxt(path, ndmin=2)
    return {c: data[:, k] for k, c in enumerate(cols)}


def test_after_a_cut_the_step_follows_the_capillary_time_of_the_sharpest_tip(tmp_path):
    """At rest, the step is set by the capillary time of the sharpest of the three tips, here the
    bubble's (radius 5e-4), which the open chain the driver measures does not see."""
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    drv = _cut_driver(tmp_path, CutProblem(), recon)
    out = drv.run(1e-9)
    drv.close()
    rows = _chains_rows(tmp_path)
    assert rows["R_tip_min"][0] == pytest.approx(5e-4, rel=1e-3)
    assert rows["R_bubble_bottom"][0] == pytest.approx(rows["R_bubble_top"][0], rel=1e-9)
    assert rows["R_cavity_tip"][0] == pytest.approx(1e-3, rel=1e-3)
    assert rows["dt_capillary"][0] == pytest.approx(0.2 * rows["R_tip_min"][0] ** 1.5, rel=1e-12)
    assert np.all(rows["dt"][1:] == pytest.approx(rows["dt_capillary"][:-1], rel=1e-12))
    assert math.isinf(rows["dt_courant"][0]) and out["stop"]["reason"] == "max-steps"
    # the ordinary rule alone, on the open chain, would allow the open cap's longer time
    assert 0.2 * 1e-3 ** 1.5 > 2 * rows["dt"][1]


def test_after_a_cut_the_step_follows_every_node_of_both_chains(tmp_path):
    """A fast bubble tip, outside the open chain the driver measures and far from the cut, sets the
    step through its Courant number."""
    from faraday_singularity.reconnect import ReconnectSettings

    def speed(chain, r, z):
        return np.where(z < 4.29, 100.0, 0.0) if chain == "bubble" else np.zeros_like(r)

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3, courant=0.1)
    drv = _cut_driver(tmp_path, CutProblem(speed=speed), recon)
    drv.run(1e-9)
    drv.close()
    rows = _chains_rows(tmp_path)
    assert np.all(rows["courant_chain"] == 1) and np.all(rows["U_courant"] == 100.0)
    assert rows["dt_courant"][0] == pytest.approx(0.1 * rows["h_courant"][0] / 100.0, rel=1e-12)
    assert rows["dt_courant"][0] < rows["dt_capillary"][0]
    assert np.all(rows["dt"][1:] == pytest.approx(rows["dt_courant"][:-1], rel=1e-12))


def test_after_a_cut_outputs_carry_tau_and_tau_p_and_end_at_tau_end(tmp_path):
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-6, dt_growth=2.0, tau_end=2.5e-6)
    pb = CutProblem()
    drv = _cut_driver(tmp_path, pb, recon, growth=2.0, max_steps=100)
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == "tau-end" and out["stop"]["tau"] == pytest.approx(2.5e-6)
    assert out["cut"]["t_cut"] == 0.0 and out["cut"]["t_p"] == -1e-6
    assert out["cut"]["tau_p"] == pytest.approx(out["cut"]["tau"] + 1e-6, abs=1e-18)
    rows = _chains_rows(tmp_path)
    assert list(rows["dt"]) == pytest.approx([0.0, 1e-6, 1.5e-6])  # the last one up to tau_end
    assert np.allclose(rows["tau"], rows["time"]) and np.allclose(rows["tau_p"], rows["time"] + 1e-6)
    table = _rows(tmp_path)
    assert np.allclose(table["tau"], table["time"]) and np.allclose(table["tau_p"],
                                                                   table["time"] + 1e-6)
    events = _read_events(tmp_path)
    assert [e["event"] for e in events][0] == "cut" and events[-1]["event"] == "stop"
    assert all("tau" in e and "tau_p" in e for e in events)
    for prof in sorted((tmp_path / "profiles").glob("profile_*[0-9].txt")):
        bubble = prof.with_name(prof.stem + "_bubble.txt")
        head, head_b = (json.loads(f.read_text().splitlines()[0][2:]) for f in (prof, bubble))
        assert head["chain"] == "open" and head_b["chain"] == "bubble"
        assert head["tau_p"] == pytest.approx(head["tau"] + 1e-6, abs=1e-18)
        assert np.loadtxt(bubble).shape == (801, 4)
    # no landing or stop states after a cut
    with pytest.raises(ValueError, match="one cut per run"):
        CollapseDriver(pb, _settings(), tmp_path / "x", reconnect=recon, landing=(5e-3,))


def test_after_a_cut_the_bubble_stops_are_checked(tmp_path):
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    pb = CutProblem(width=lambda t: 1.0 if t <= 0.0 else 0.05)  # drains after the first step
    drv = _cut_driver(tmp_path, pb, recon)
    out = drv.run(1e-9)
    drv.close()
    assert out["stop"]["reason"] == "bubble-vanishing" and out["steps"] == 1
    assert out["stop"]["tau"] > 0.0


def test_after_a_cut_no_newton_failure_ends_the_run_cleanly(tmp_path):
    """Before a cut this failure, with a neck narrower than neck_guard, would be a pinch-off
    approach; after one it is a failed run."""
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    drv = _cut_driver(tmp_path, CutProblem(fail_at=2), recon)
    with pytest.raises(RuntimeError, match="OomphException"):
        drv.run(1e-9)
    drv.close()


def test_a_post_cut_state_needs_the_reconnect_settings_of_its_cut(tmp_path):
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    pb = CutProblem()
    drv = _cut_driver(tmp_path / "a", pb, recon, state_dt=0.0)
    drv.run(1e-9)
    drv.close()
    meta = json.loads(sorted((tmp_path / "a" / "states").glob("state_*.json"))[-1].read_text())
    assert meta["cut"]["t_cut"] == 0.0 and meta["tau"] == meta["time"]
    with pytest.raises(RuntimeError, match="reconnect settings"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "b").restore(meta)
    other = ReconnectSettings(delta=1e-3, dt0=2e-3)
    with pytest.raises(ValueError, match="differ from those of the cut"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "c", reconnect=other).restore(meta)
    # nor does a driver without them run after a cut
    plain = CollapseDriver(pb, _settings(), tmp_path / "d")
    with pytest.raises(RuntimeError, match="needs the reconnect settings"):
        plain.reset_after_cut(pb.cut)
    with pytest.raises(RuntimeError, match="runs after a cut"):
        CollapseDriver(CutProblem(), _settings(), tmp_path / "e", reconnect=recon).run(1e-9)
