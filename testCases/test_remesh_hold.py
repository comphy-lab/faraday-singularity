"""Contract tests for the hold on coarsening after a refining rebuild of the focus.

The driver measures, with its real measurement, a stand-in free surface whose nodes near the axis
are spaced as the focus of the last rebuild asks. ``_spacing_dependent_apex`` gives an apex
curvature that depends on that spacing, as it can once rebuilds have distorted the apex: a fine
surface reads a broad apex that asks for a coarse focus, and a coarse one a sharp apex that asks for
a fine focus, at the same time. The other surfaces are paraboloids whose curvature depends on the
time only.
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

from faraday_singularity.driver import CollapseDriver, CollapseSettings, local_time

DEPTH = 2 * math.pi
SPEED = 7.0  # upward speed of every node, the speed U the driver measures
KAPPA_SHARP = 30.0
# the hold after a refinement asked for by an apex of curvature KAPPA_SHARP
TAU = min(1.0 / KAPPA_SHARP / SPEED, KAPPA_SHARP**-1.5)


def _settings(**kw) -> CollapseSettings:
    return CollapseSettings(t_end=kw.pop("t_end", 1.0), dt_max=kw.pop("dt_max", 0.01), **kw)


class RebuiltSurface:
    """The parts of ``FaradayProblem`` the driver uses, with the free surface ``surface(t, r, h)``.

    Its nodes are spaced ``h/2`` out to ``r = 0.5``, ``h`` the focus size of the last rebuild
    (``h_axis`` before any), so that ``h_near = h``, and move up at ``SPEED``. With ``auto_every``,
    element distortion asks for a rebuild after every ``auto_every``-th solve once a focus is set:
    pyoomph makes it inside the solve while ``do_call_remeshing_when_necessary`` is on, and
    otherwise only flags it for the driver (``take_distortion_flags``), as ``FaradayProblem`` does.
    """

    def __init__(self, surface, auto_every: int | None = None):
        self.t = 0.0
        self.surface = surface
        self.auto_every = auto_every
        self.solves = 0
        self.remesh_count = 0
        self.mesh_settings = SimpleNamespace(h_axis=0.05)
        self.nd = SimpleNamespace(radius=3.8317, depth=DEPTH)
        self.template = SimpleNamespace(focus=None)
        self.h_mesh = self.mesh_settings.h_axis
        self.do_call_remeshing_when_necessary = True
        self.flags: list[dict] = []
        self.pyoomph_rebuilds = 0

    def get_current_time(self, dimensional=False, as_float=True):
        return self.t

    def run(self, t_end, **_kw):
        self.t = t_end
        self.solves += 1
        if self.auto_every and self.solves % self.auto_every == 0 and self.template.focus:
            if self.do_call_remeshing_when_necessary:
                self.pyoomph_rebuilds += 1
                self.force_remesh()
            else:
                self.flags.append({"trigger": "expansion", "ratio": 1.8})

    def take_distortion_flags(self):
        flags, self.flags = self.flags, []
        return flags

    def force_remesh(self):
        self.remesh_count += 1  # FaradayProblem.actions_after_remeshing
        self.h_mesh = self.template.focus.size

    def active_mesh_template(self):
        return self.template

    def interface_arrays(self):
        inner = 0.5 * self.h_mesh * np.arange(math.ceil(1.0 / self.h_mesh) + 1)
        r = np.concatenate([inner, np.linspace(inner[-1], self.nd.radius, 100)[1:]])
        return (r, self.surface(self.t, r, self.h_mesh), np.zeros_like(r),
                np.full_like(r, SPEED))

    def axial_velocity_at(self, points):
        return [SPEED] * len(points), True

    def observables(self):
        return {"volume": 1.0, "area": 1.0, "kinetic_energy": 0.0, "dissipation": 0.0}

    def oomph_log_tail(self):
        return ""

    def save_state(self, path, quiet=True):
        Path(path).write_text(json.dumps({"time": self.t, "h_mesh": self.h_mesh}))

    def load_state(self, path):
        dump = json.loads(Path(path).read_text())
        self.t, self.h_mesh = dump["time"], dump["h_mesh"]


def _trough(kappa, r):
    return DEPTH - 0.1 + 0.5 * kappa * r**2


def _spacing_dependent_apex(t, r, h):
    """An apex of curvature 4 through nodes finer than 0.008, KAPPA_SHARP through coarser ones."""
    return _trough(4.0 if h < 8e-3 else KAPPA_SHARP, r)


def _flattening_trough(t, r, h, t_flat=2e-3):
    """A sharp apex that flattens at ``t_flat``, at every node spacing."""
    return _trough(KAPPA_SHARP if t < t_flat else 4.0, r)


def _sharpening_trough(t, r, h):
    """An apex whose curvature grows fourfold at t = 1e-3 and again at t = 2e-3."""
    return _trough(KAPPA_SHARP * 4 ** min(int(t / 1e-3), 2), r)


def _run(tmp_path, surface, out="a", auto_every=None, **kw):
    s = _settings(**kw)
    pb = RebuiltSurface(surface, auto_every=auto_every)
    drv = CollapseDriver(pb, s, tmp_path / out)
    status = drv.run(1e-9)
    drv.close()
    return s, pb, status


def _read_events(outdir: Path) -> list[dict]:
    return [json.loads(line) for line in (outdir / "events.jsonl").read_text().splitlines()]


def _rows(outdir: Path) -> dict[str, np.ndarray]:
    path = outdir / "collapse.txt"
    with path.open() as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def _rebuilds(outdir: Path) -> tuple[list[dict], list[dict]]:
    """The refining and the coarsening rebuilds of the driver's rule, in order (not those that
    element distortion asked for)."""
    remesh = [e for e in _read_events(outdir)
              if e["event"] == "remesh" and e.get("reason") != "distortion"]
    return ([e for e in remesh if e["focus_size"] < e["h_near_before"]],
            [e for e in remesh if e["focus_size"] > e["h_near_before"]])


def _wants_coarsening(tab, s) -> np.ndarray:
    return tab["h_near"] < s.coarsen_ratio * tab["h_focus"]


def test_local_time_is_the_shorter_of_advection_and_capillarity():
    assert local_time(0.0431, 7.706) == pytest.approx(0.0431 / 7.706, rel=1e-14)
    assert local_time(0.04, 1e-3) == pytest.approx(0.04**1.5, rel=1e-14)  # slow flow
    assert local_time(0.4, 0.0) == pytest.approx(0.4**1.5, rel=1e-14)  # no flow


@pytest.mark.parametrize("hold", [0.0, 1.0])
def test_refined_focus_is_not_coarsened_at_a_frozen_time(tmp_path, hold):
    # steps of at most 1e-9: 150 of them barely advance the time. The fine surface asks for a
    # focus about eight times coarser, the coarse one for refinement
    s, pb, status = _run(tmp_path, _spacing_dependent_apex, dt_max=1e-9, dt_min=1e-300,
                         max_steps=150, state_dt=1.0, coarsen_hold=hold)
    tab = _rows(tmp_path / "a")
    refine, coarsen = _rebuilds(tmp_path / "a")
    assert tab["time"][-1] <= 150 * s.dt_max * (1 + 1e-9)
    if hold == 0.0:  # without the hold the focus alternates while the time stands still
        assert len(coarsen) >= 10 and len(refine) >= len(coarsen) + 1
        assert all(a["step"] < b["step"] < c["step"]
                   for a, b, c in zip(refine[1:], coarsen, refine[2:]))
        # every sharp reading of a coarse rebuild cuts the step, until it no longer advances the
        # time or the steps run out
        assert tab["dt"][-1] < 1e-6 * s.dt_max
        assert status["stop"]["reason"] in ("dt-floor", "max-steps")
        return
    assert status["stop"]["reason"] == "max-steps"
    # the two rebuilds from h_axis reach a fine focus; it is then held, although the measured scale
    # asks for coarsening on almost every later step
    assert [e["step"] for e in refine] == [1, 2] and not coarsen
    assert np.sum(_wants_coarsening(tab, s)) > 130
    assert pb.remesh_count == 2
    for e in refine:
        assert e["coarsen_held_until"] == pytest.approx(e["time"] + TAU, rel=1e-9)


@pytest.mark.parametrize("auto_every", [None, 3])
def test_coarsening_follows_once_the_hold_has_passed(tmp_path, auto_every):
    # the apex flattens long before the hold of the last refinement has passed; the rebuilds that
    # element distortion asks for (every third solve), which the driver makes once a focus is set,
    # neither end nor extend that hold
    s, pb, status = _run(tmp_path, _flattening_trough, auto_every=auto_every, t_end=0.012,
                         dt_max=2e-4, state_dt=1.0)
    tab = _rows(tmp_path / "a")
    refine, coarsen = _rebuilds(tmp_path / "a")
    assert [e["step"] for e in refine] == [1, 2]
    held = refine[-1]["coarsen_held_until"]
    assert held == pytest.approx(refine[-1]["time"] + TAU, rel=1e-9)
    t = tab["time"]
    assert np.any(_wants_coarsening(tab, s) & (t > 2e-3) & (t < held - 1e-3))
    # coarsening starts on the first step at or after the end of the hold, then catches up
    assert len(coarsen) == 2
    assert coarsen[0]["time"] >= held and coarsen[0]["step"] == tab["step"][t >= held][0]
    assert coarsen[1]["step"] == coarsen[0]["step"] + s.min_steps_between_remesh
    assert not _wants_coarsening(tab, s)[-1]
    distortion = [e for e in _read_events(tmp_path / "a")
                  if e["event"] == "remesh" and e.get("reason") == "distortion"]
    assert status["distortion_remeshes"] == len(distortion) and status["pyoomph_remeshes"] == 0
    if auto_every:  # a focus is set from the first step on: every flag is the driver's rebuild
        assert len(distortion) == pb.solves // auto_every and pb.pyoomph_rebuilds == 0
        assert pb.remesh_count == len(refine) + len(coarsen) + len(distortion)
        assert all(e["trigger"] == "expansion" and e["flagged_substeps"] == 1
                   and e["focus_size"] <= e["focus_size_before"] for e in distortion)
    else:
        assert not distortion


def test_refinement_is_not_held(tmp_path):
    # the apex sharpens twice within the hold of the first refinement: both times the focus refines
    # at once, and each refinement holds coarsening until the later of the two ends
    _run(tmp_path, _sharpening_trough, t_end=2.5e-3, dt_max=2e-4, state_dt=1.0)
    refine, coarsen = _rebuilds(tmp_path / "a")
    assert not coarsen
    first_hold = refine[0]["coarsen_held_until"]
    sharpened = [e for e in refine if e["time"] > 1e-3]
    assert len(sharpened) >= 2 and all(e["time"] < first_hold for e in sharpened)
    assert sharpened[-1]["focus_size"] < refine[0]["focus_size"] / 10
    holds = [e["coarsen_held_until"] for e in refine]
    assert holds == sorted(holds)
    for e in refine:
        assert e["coarsen_held_until"] >= e["time"] + local_time(e["ell"], SPEED) * (1 - 1e-12)


class _PrescribedDriver(CollapseDriver):
    """Measurement given by ``fields(t)``: rebuilds leave it as it was, so none refines."""

    def __init__(self, pb, settings, outdir, fields):
        super().__init__(pb, settings, outdir)
        self.fields = fields

    def measure(self) -> dict:
        t = self._t()
        r = np.linspace(0.0, 1.0, 11)
        m = {"time": t, "axis_lowest": True, "z_axis": DEPTH, "w_axis": 0.0,
             "axis_curvature": 1.0, "cavity": True, "jet_born": False, "flat_bump": False,
             "r_j": 0.0, "z_b": 0.0, "Q_j": math.nan, "V_j": math.nan, "We_j": math.nan,
             "flux_found": True, "r_neck": math.nan, "z_neck": math.nan, "ell": 0.08,
             "h_focus": 0.01, "h_near": 0.05, "U": 1.0, "beta_fit": math.nan,
             "beta_fit_points": 0, "beta_window_lo": 0.0, "beta_window_hi": 0.0,
             "angles": [math.nan] * len(self.s.cone_radii), "focus_radius": 1.0,
             "profile": (r, np.zeros_like(r), np.zeros_like(r), np.zeros_like(r))}
        m.update(self.fields(t))
        return m


def test_ineffective_refinement_holds_coarsening_but_not_a_stop(tmp_path):
    # a refinement that does not take effect still holds coarsening, for 0.08^1.5 = 0.0226; the
    # surface then asks for coarsening while a neck closes, and the neck stops the run in the hold
    def fields(t):
        return {"h_near": 0.05 if t < 0.015 else 1e-3, "r_neck": 0.0205 - t, "z_neck": 1.0}

    pb = RebuiltSurface(_flattening_trough)
    drv = _PrescribedDriver(pb, _settings(t_end=1.0, dt_max=0.01, state_dt=1.0), tmp_path,
                            fields)
    status = drv.run(1e-9)
    drv.close()
    events = _read_events(tmp_path)
    (remesh,) = [e for e in events if e["event"] == "remesh"]
    assert [e["step"] for e in events if e["event"] == "remesh-ineffective"] == [remesh["step"]]
    assert remesh["coarsen_held_until"] == pytest.approx(remesh["time"] + 0.08**1.5, rel=1e-12)
    stop = status["stop"]
    assert stop["reason"] == "pinch-off-approach" and stop["trigger"] == "neck-radius"
    tab = _rows(tmp_path)
    assert tab["time"][-1] < remesh["coarsen_held_until"] and tab["h_near"][-1] == 1e-3


def _restart(tmp_path, s, src: Path, step: int, out: str, drop=()):
    """Resume from the dump of ``step`` in run ``src``, without the fields ``drop``, to the end."""
    meta = json.loads((src / "states" / f"state_{step:08d}.json").read_text())
    for k in drop:
        del meta[k]
    pb = RebuiltSurface(_spacing_dependent_apex)
    pb.load_state(src / "states" / f"state_{step:08d}.dump")
    drv = CollapseDriver(pb, s, tmp_path / out)
    drv.restore(meta)
    drv.run(1e-9)
    drv.close()
    return meta


def test_restart_keeps_the_hold(tmp_path):
    # the alternation of the stand-in surface proceeds one hold at a time; a dump on every step
    s, _, _ = _run(tmp_path, _spacing_dependent_apex, t_end=0.02, dt_max=2e-4, state_dt=0.0,
                   state_keep=10_000)
    full = _rows(tmp_path / "a")
    refine, coarsen = _rebuilds(tmp_path / "a")
    assert len(coarsen) >= 2
    for c in coarsen:  # each coarsening waits for the hold of the refinement before it
        before = [e for e in refine if e["step"] < c["step"]]
        assert c["time"] >= before[-1]["coarsen_held_until"]
    # a dump in the hold of the second refinement, at a step that asks for coarsening
    held = refine[1]["coarsen_held_until"]
    k = int(np.flatnonzero(_wants_coarsening(full, s) & (full["step"] > refine[1]["step"])
                           & (full["time"] < held))[0])
    step = int(full["step"][k])
    meta = _restart(tmp_path, s, tmp_path / "a", step, "b")
    assert meta["coarsen_held_until"] == held
    resumed = _read_events(tmp_path / "b")
    assert resumed[0]["event"] == "restart" and resumed[0]["coarsen_held_until"] == held
    rest = _rows(tmp_path / "b")
    for name in ("step", "time", "dt", "ell", "h_focus", "h_near", "remesh_count"):
        first = 1 if name == "dt" else 0  # the restart row records no step
        assert np.array_equal(rest[name][first:], full[name][k + first:]), name
    after = [(e["event"], e["step"], e["focus_size"]) for e in _read_events(tmp_path / "a")
             if e["event"] == "remesh" and e["step"] > step]
    assert [(e["event"], e["step"], e["focus_size"]) for e in resumed
            if e["event"] == "remesh"] == after
    # a dump of an older driver, without the hold and the fields added before it: the restart
    # holds nothing and coarsens on its first step
    _restart(tmp_path, s, tmp_path / "a", step, "c",
             drop=("coarsen_held_until", "refine_blocked_until", "max_curvature_kind"))
    old = _read_events(tmp_path / "c")
    assert old[0]["event"] == "restart" and old[0]["coarsen_held_until"] == -math.inf
    first = next(e for e in old if e["event"] == "remesh")
    assert first["step"] == step + 1 and first["focus_size"] > first["h_near_before"]
    assert first["time"] < held
