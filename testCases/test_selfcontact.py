"""Contract tests for folds and self-crossings of the free surface, and the stop before entrapment.

The driver runs on a stand-in free surface: a jet with a hemispherical tip on the axis, standing in
an air slot between it and the cavity wall, whose width closes in time. A surface that has crossed
itself must raise rather than end the run, as it must when a remesh is handed one (``mesh.py``).
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

from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.selfcontact import (
    InterfaceSelfIntersection,
    catmull_rom,
    check_simple_interface,
    chord_crossings,
    fold_gap,
)

DEPTH = 2 * math.pi
SPEED = 7.0
JET = 0.01  # jet radius
SLOT = 0.02  # slot depth
H = 1e-4  # node spacing near the slot


def slot_surface(gap, h=H, a=JET, depth=SLOT, R=3.8317):
    """Jet of radius ``a`` (hemispherical tip on the axis at ``DEPTH``) in a slot of width ``gap``.

    A negative ``gap`` puts the cavity wall inside the jet: the floor at the top of the slot then
    crosses the jet wall, as a fold that has closed through itself.
    """
    n_wall = round(0.03 / h)
    z_b = DEPTH - a - n_wall * h
    th = np.linspace(0.0, math.pi / 2, int(math.pi / 2 * a / h))
    r, z = [a * np.sin(th)], [DEPTH - a * (1 - np.cos(th))]
    # exact multiples of h: a node a round-off away from the next would make the uniform
    # Catmull-Rom spline loop there, and the slot would fail check_simple_interface
    wall = DEPTH - a - h * np.arange(1, n_wall)
    r.append(np.full_like(wall, a))
    z.append(wall)
    if gap > 0:
        ph = np.linspace(math.pi, 2 * math.pi, max(3, int(math.pi * gap / 2 / h)))
        r.append(a + gap / 2 * (1 + np.cos(ph)))
        z.append(z_b + gap / 2 * np.sin(ph))
    else:
        r.append(np.array([a, a + gap]))
        z.append(np.array([z_b, z_b]))
    z_top = z_b + depth + h / 3  # off the node heights of the jet wall
    outer = z_b + h * np.arange(1, round(depth / h) + 1)
    r.append(np.full_like(outer, a + gap))
    z.append(outer)
    # floor nodes off the jet wall, so that a crossing is never through a node, and graded as a
    # mesh is: on collinear nodes the uniform Catmull-Rom spline doubles back where the spacing
    # jumps more than fivefold
    floor, d = [a + gap + h / 2], h
    while floor[-1] + d < R:
        floor.append(floor[-1] + d)
        d = min(1.1 * d, 0.05) if floor[-1] > 0.05 else d
    floor = np.array(floor[:-1] + [R])
    r.append(floor)
    z.append(np.full_like(floor, z_top))
    r, z = np.concatenate(r), np.concatenate(z)
    keep = np.concatenate([[True], np.hypot(np.diff(r), np.diff(z)) > 1e-14])
    return r[keep], z[keep]


def v_groove(half_angle_deg, n=200, side=0.01):
    """Two straight walls of length ``side`` meeting at the bottom, symmetric about r = 0.1."""
    t = math.radians(half_angle_deg)
    s = np.linspace(0.0, side, n)
    left = (0.1 - s[::-1] * math.sin(t), -s[::-1] * math.cos(t))
    right = (0.1 + s[1:] * math.sin(t), -s[1:] * math.cos(t))
    return np.concatenate([left[0], right[0]]), -np.concatenate([left[1], right[1]])


# --- geometry -------------------------------------------------------------------------------

def test_chord_crossings_of_a_crossed_slot_only():
    r, z = slot_surface(2e-3)
    assert chord_crossings(r, z) == []
    r, z = slot_surface(-3e-4)
    hits = chord_crossings(r, z)
    assert len(hits) == 1
    i, j = hits[0]
    # the jet wall at r = JET is crossed by the floor at the top of the slot
    assert r[i] == pytest.approx(JET) and r[i + 1] == pytest.approx(JET)
    assert z[j] == z[j + 1] and r[j] < JET < r[j + 1]


def test_two_selected_chords_and_nonadjacent_touches_are_detected():
    points = np.array([[0., 0.], [2., 2.], [0., 2.], [2., 0.]])
    assert chord_crossings(points[:, 0], points[:, 1], [0, 2]) == [(0, 2)]
    for points in (np.array([[0., 0.], [1., 1.], [2., 0.], [1., 1.], [3., 1.]]),
                   np.array([[0., 0.], [2., 0.], [2., 1.], [1., 0.], [3., 0.]])):
        assert chord_crossings(points[:, 0], points[:, 1])
        with pytest.raises(InterfaceSelfIntersection):
            check_simple_interface(points)
    points = np.array([[0., 0.], [1., 0.], [2., 1.], [2., 0.], [3., 0.]])
    assert chord_crossings(points[:, 0], points[:, 1], [0, 3]) == []


def test_catmull_rom_is_the_uniform_spline_with_phantom_ends():
    rng = np.random.default_rng(1)
    P = np.cumsum(rng.uniform(0.1, 1.0, (7, 2)), axis=0)
    Q = catmull_rom(P[:, 0], P[:, 1], samples=2)
    assert np.allclose(Q[::2], P)  # it interpolates the nodes
    i = 3  # midpoint of an interior interval: (-P0 + 9 P1 + 9 P2 - P3)/16
    assert np.allclose(Q[2 * i + 1], (-P[i - 1] + 9 * P[i] + 9 * P[i + 1] - P[i + 2]) / 16)
    ghost = 2 * P[0] - P[1]  # first interval, with the phantom point
    assert np.allclose(Q[1], (-ghost + 9 * P[0] + 9 * P[1] - P[2]) / 16)


def test_simple_interface_check_refuses_crossings_of_polygon_and_spline():
    r, z = slot_surface(2e-3)
    check_simple_interface(np.column_stack([r, z]))
    r, z = slot_surface(-3e-4)
    with pytest.raises(InterfaceSelfIntersection, match="free surface crosses itself"):
        check_simple_interface(np.column_stack([r, z]))
    # a simple polygon whose uniform Catmull-Rom spline loops in its short middle interval
    loop = np.array([[0.1, 0.0], [2.0, 2.0], [2.1, 2.0], [4.0, 0.0]])
    assert chord_crossings(loop[:, 0], loop[:, 1]) == []
    with pytest.raises(InterfaceSelfIntersection, match="spline through the free surface"):
        check_simple_interface(loop)


def test_sampled_spline_guard_has_a_documented_resolution_limit():
    loop = np.array([[0., 0.], [1., 0.5], [1.19, 0.5], [2.19, 0.]])
    check_simple_interface(loop, samples=8)
    with pytest.raises(InterfaceSelfIntersection, match="spline through the free surface"):
        check_simple_interface(loop, samples=64)


def test_fold_gap_of_a_slot_is_its_width():
    for gap in (3e-3, 1e-3, 3e-4):
        r, z = slot_surface(gap)
        f = fold_gap(r, z, r_max=0.2, search=4e-3)
        assert not f.crossed
        assert gap * (1 - 1e-9) <= f.gap <= math.hypot(gap, H)
        assert JET <= f.r <= JET + gap
        assert f.h_elem == pytest.approx(2 * H, rel=0.05)
        assert f.arc > 4 * f.gap


def test_fold_gap_ignores_open_and_wide_shapes():
    r, z = slot_surface(5e-3)  # wider than the search radius
    assert fold_gap(r, z, r_max=0.2, search=4e-3).gap == math.inf
    # a groove of half-angle 28 degrees (the turn of 124 degrees seen before the fold closed) is
    # no fold; one of 10 degrees is
    for angle, folded in ((28.0, False), (20.0, False), (10.0, True)):
        r, z = v_groove(angle)
        f = fold_gap(r, z, search=0.05)
        assert math.isfinite(f.gap) == folded
    # a smooth rounded cavity on the axis
    th = np.linspace(0, math.pi / 2, 400)
    assert fold_gap(0.05 * np.sin(th), -0.05 * np.cos(th), search=0.05).gap == math.inf


def test_fold_crossing_outside_the_window_is_not_reported():
    r, z = slot_surface(-3e-4)
    assert fold_gap(r, z, r_max=0.2, search=4e-3).crossed
    assert not fold_gap(r, z, r_max=0.5 * JET, search=4e-3).crossed


def test_local_fold_search_reaches_a_coarse_slot_among_fine_nodes():
    r_slot, z_slot = slot_surface(5e-3, h=2e-3, R=0.2)
    r = np.concatenate([np.linspace(0., 0.05, 5001), r_slot + 0.1])
    z = np.concatenate([np.full(5001, DEPTH + 1.), z_slot])
    assert fold_gap(r, z, search=4e-3).gap == math.inf
    fold = fold_gap(r, z, stop=1e-3, stop_elements=3)
    assert math.isfinite(fold.gap)
    assert fold.gap < max(1e-3, 3 * fold.h_elem)


def test_coarse_fold_exceeding_its_limit_is_not_hidden_by_a_narrower_fold():
    fine_r, fine_z = slot_surface(2e-3, h=H, R=0.2)
    coarse_r, coarse_z = slot_surface(3e-3, h=2e-3, R=0.2)
    r = np.concatenate([fine_r, coarse_r + 0.3])
    z = np.concatenate([fine_z, coarse_z + 1.])
    nearest = fold_gap(r, z, search=4e-3)
    assert nearest.r < 0.3 and nearest.gap > max(1e-3, 3 * nearest.h_elem)
    limiting = fold_gap(r, z, stop=1e-3, stop_elements=3)
    assert limiting.r > 0.3 and limiting.gap < max(1e-3, 3 * limiting.h_elem)


@pytest.mark.parametrize("values", [
    {"fold_ratio": 1}, {"fold_ratio": 0.25}, {"fold_ratio": True},
    {"fold_ratio": "4"}, {"fold_ratio": math.inf},
    {"fold_stop": math.nan}, {"fold_stop": -1}, {"fold_stop": False},
    {"fold_stop_elements": "3"}, {"fold_stop_elements": -1},
    {"fold_stop": 0, "fold_stop_elements": 0},
])
def test_invalid_fold_settings_are_rejected_before_running(values):
    with pytest.raises(ValueError, match="fold"):
        CollapseSettings.from_dict(values, t_end=1., dt_max=1e-4)


# --- driver ---------------------------------------------------------------------------------

class SlotProblem:
    """The parts of ``FaradayProblem`` the driver uses; the free surface is ``slot_surface(gap(t))``."""

    def __init__(self, gap):
        self.t = 0.0
        self.gap = gap
        self.remesh_count = 0
        self.mesh_settings = SimpleNamespace(h_axis=0.05)
        self.nd = SimpleNamespace(radius=3.8317, depth=DEPTH)
        self.template = SimpleNamespace(focus=None)

    def get_current_time(self, dimensional=False, as_float=True):
        return self.t

    def run(self, t_end, **_kw):
        self.t = t_end

    def force_remesh(self):
        self.remesh_count += 1

    def active_mesh_template(self):
        return self.template

    def interface_arrays(self):
        r, z = slot_surface(self.gap(self.t))
        return r, z, np.zeros_like(r), np.full_like(r, SPEED)

    def axial_velocity_at(self, points):
        return [SPEED] * len(points), True

    def observables(self):
        return {"volume": 1.0, "area": 1.0, "kinetic_energy": 0.0, "dissipation": 0.0}

    def oomph_log_tail(self):
        return ""

    def save_state(self, path, quiet=True):
        Path(path).write_text(json.dumps({"time": self.t}))


class CoarsenedSlotProblem(SlotProblem):
    def __init__(self, gap):
        super().__init__(lambda _t: gap)
        self.h = H
        self.solves = self.saves = 0

    def run(self, t_end, **kwargs):
        self.solves += 1
        super().run(t_end, **kwargs)

    def interface_arrays(self):
        r, z = slot_surface(self.gap(self.t), h=self.h)
        return r, z, np.zeros_like(r), np.full_like(r, SPEED)

    def force_remesh(self):
        super().force_remesh()
        self.h = 4 * H

    def save_state(self, path, quiet=True):
        self.saves += 1
        super().save_state(path, quiet=quiet)


def _settings(**kw):
    kw.setdefault("coarsen_ratio", 0.0)  # the stand-in surface has a fixed spacing
    return CollapseSettings(t_end=kw.pop("t_end", 0.01), dt_max=kw.pop("dt_max", 1e-4),
                            state_dt=1.0, **kw)


def _rows(outdir: Path) -> dict[str, np.ndarray]:
    path = outdir / "collapse.txt"
    with path.open() as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def test_initial_geometry_is_checked_before_another_step_or_dump(tmp_path):
    pb = CoarsenedSlotProblem(5e-4)
    drv = CollapseDriver(pb, _settings(), tmp_path)
    try:
        status = drv.run(1e-9)
    finally:
        drv.close()
    assert status["stop"]["reason"] == "self-contact-approach"
    assert pb.solves == pb.saves == 0


def test_remeshed_geometry_is_checked_before_another_step_or_dump(tmp_path):
    pb = CoarsenedSlotProblem(1.5e-3)
    drv = CollapseDriver(pb, _settings(), tmp_path)
    def coarsen(_measurement):
        pb.force_remesh()
        return True
    drv._maybe_remesh = coarsen
    try:
        status = drv.run(1e-9)
    finally:
        drv.close()
    assert status["stop"]["reason"] == "self-contact-approach"
    assert pb.solves == 1 and pb.saves == 0
    assert status["stop"]["fold_h_elem"] > 2 * H


def test_driver_stops_right_before_a_closing_fold(tmp_path):
    g0, v = 3e-3, 1.0
    s = _settings()
    drv = CollapseDriver(SlotProblem(lambda t: g0 - v * t), s, tmp_path)
    status = drv.run(1e-9)
    drv.close()
    stop = status["stop"]
    assert stop["reason"] == "self-contact-approach" and stop["trigger"] == "fold-gap"
    assert stop["fold_limit"] == s.fold_stop  # 3 elements of 2e-4 are narrower than fold_stop
    tab = _rows(tmp_path)
    gaps = tab["fold_gap"]
    assert np.all(np.isfinite(gaps))  # the slot is within the search radius throughout
    assert gaps[-1] < s.fold_stop <= gaps[-2]  # the first row below the limit ends the run
    assert gaps[-1] > s.fold_stop - 2 * v * tab["dt"].max()
    # the measured gap is the width, up to a vertical offset of the wall nodes of at most H/2
    assert tab["time"][-1] == pytest.approx((g0 - gaps[-1]) / v, abs=H**2 / (8 * gaps[-1]) + 1e-9)
    assert JET <= tab["fold_r"][-1] <= JET + g0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "stop" and events[-1]["reason"] == "self-contact-approach"
    stops = [json.loads(p.read_text().splitlines()[0].lstrip("# "))
             for p in sorted((tmp_path / "profiles").glob("profile_*.txt"))]
    assert stops[-1]["reason"] == "stop"
    assert stops[-1]["time"] == pytest.approx(tab["time"][-1], rel=1e-11)


def test_driver_without_a_fold_writes_infinite_gaps(tmp_path):
    drv = CollapseDriver(SlotProblem(lambda t: 6e-3), _settings(t_end=5e-4), tmp_path)
    status = drv.run(1e-9)
    drv.close()
    assert status["stop"]["reason"] == "end-time"
    tab = _rows(tmp_path)
    assert np.all(np.isinf(tab["fold_gap"])) and np.all(np.isnan(tab["fold_r"]))


def test_driver_raises_on_a_surface_that_has_crossed_itself(tmp_path):
    # the slot jumps from open to crossed within one step, faster than the stop can see
    t_jump = 3e-4
    drv = CollapseDriver(SlotProblem(lambda t: 3e-3 if t < t_jump else -3e-4),
                         _settings(), tmp_path)
    with pytest.raises(InterfaceSelfIntersection, match="crosses itself"):
        drv.run(1e-9)
    drv.close()
    tab = _rows(tmp_path)
    assert tab["time"][-1] < t_jump and np.all(tab["fold_gap"] > 2e-3)  # only valid rows
