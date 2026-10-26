"""After a cut, failures that ended runs without an end state, on stand-ins (no finite elements):

* a step that fails (Newton at every step size, or pyoomph's net for an inverted element giving up,
  as r1 at b46713a did at step 7503) is rolled back to the state it started from, the snapshot the
  net took before it (``FaradayProblem._snapshot_state`` keeps the first of a step,
  ``CollapseDriver._post_cut_solve``), and is the resumable stop ``step-failed`` with that state as
  the end state and an event ``step-failed`` (``CollapseDriver._failed_step_stop``); without a
  snapshot (the net not armed, a problem without it) the failure is raised as before;
* in the ring stage, one retry of a failed step after its ring's sharpest element above the floor
  is rebuilt as the ring refine rebuilds (r1h at b36c9c1, step 7251: the root of the radial sheet,
  85.6 degrees in an element of 1.55e-5), ``CollapseDriver._failed_step_retry``;
* the ring's loop with a jump of its spacing beyond ``LOOP_STEP_RATIO`` beside a region refined to
  the floor is in order when it is simple (r1h at a150b74, step 7228: 3.6e-6 beside 6.2e-7 at the
  corner of a fill, a failed run); a walk that jumps across the loop and back crosses it and is
  refused as before.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from test_bookkeeping import CutProblem, _cut_driver  # noqa: E402

from faraday_singularity import ring  # noqa: E402


class NetProblem(CutProblem):
    """``CutProblem`` with what the driver reads of pyoomph's net for an inverted element: the
    snapshot of the state before each solve (the time only, here), restored on request, kept by
    the driver's collection as ``FaradayProblem._snapshot_state`` keeps it."""

    states_compression_level = 0
    min_newton_iterations = 1

    def __init__(self, **kw):
        super().__init__(**kw)
        self.restored = []

    def release_newton_tolerance(self):
        pass

    def _snapshot_state(self, to_memory=True):
        snap = {"t": self.t}
        kept = getattr(self, "_step_snapshots", None)
        if kept is not None and not kept and to_memory:
            kept.append(snap)
        return snap

    def _restore_state(self, snap):
        self.restored.append(snap)
        self.t = snap["t"]
        self.remesh_count += 1  # pyoomph load_state calls actions_after_remeshing

    def invalidate_cached_mesh_data(self):
        pass

    def run(self, t_end, **kw):
        self._snapshot_state()  # the net's, before the solve
        super().run(t_end, **kw)


def _events(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]


def test_a_failed_step_after_a_cut_is_a_resumable_stop_on_the_state_it_started_from(tmp_path):
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    pb = NetProblem(fail_at=2)
    drv = _cut_driver(tmp_path, pb, recon, max_steps=5)
    from faraday_singularity.events import EventSettings
    drv.events = EventSettings()
    drv._remesh_checks = lambda m: (m, None)  # keep the accepted mesh stepped-on
    out = drv.run(1e-9)
    drv.close()
    stop = out["stop"]
    assert stop["reason"] == "step-failed" and "OomphException" in stop["error"]
    assert out["steps"] == 1 and len(pb.restored) == 1
    t_start = pb.restored[0]["t"]
    assert pb.t == t_start and stop["dt_attempted"] > 0.0 and stop["net_rebuilds"] == 0
    end = Path(stop["end_state"])
    assert end.is_file() and end.name == "state_00000001.dump"
    meta = json.loads(end.with_suffix(".json").read_text())
    assert meta["time"] == t_start and meta["mesh_fresh"] is False
    failed = [e for e in _events(tmp_path) if e["event"] == "step-failed"]
    assert len(failed) == 1 and failed[0]["dt_attempted"] == stop["dt_attempted"]
    assert any(p.name.startswith("profile_000001_lastgood") for p in (tmp_path / "profiles").iterdir())


def test_without_the_nets_snapshot_a_failed_step_is_raised_as_before(tmp_path):
    from faraday_singularity.reconnect import ReconnectSettings

    recon = ReconnectSettings(delta=1e-3, dt0=1e-3)
    drv = _cut_driver(tmp_path, CutProblem(fail_at=2), recon)  # no net, no snapshot
    with pytest.raises(RuntimeError, match="OomphException"):
        drv.run(1e-9)
    drv.close()


def _graded_loop(jump=6.0, n_fine=40, h=6e-7):
    """A counter-clockwise loop (an ellipse) whose spacing jumps by ``jump`` from a run of fine
    chords to the coarse rest, as beside a region refined to the floor."""
    c, a, b = np.array([4e-4, 4.6115]), 6e-5, 3e-5
    fine = np.arange(n_fine) * h / a  # angle steps of the fine run
    coarse_step = jump * h / a
    rest = np.arange(fine[-1] + coarse_step, 2 * math.pi - coarse_step * 0.5, coarse_step)
    t = np.concatenate([fine, rest])
    return np.column_stack([c[0] + a * np.cos(t), c[1] + b * np.sin(t)])


def test_a_loop_whose_spacing_jumps_beside_a_refined_region_is_in_order():
    L = _graded_loop(jump=6.0)
    d = ring.loop_steps(L)
    beside = np.minimum(np.roll(d, 1), np.roll(d, -1))
    assert np.max(d / beside) > ring.LOOP_STEP_RATIO  # the ratio test alone refuses it
    ring.validate_loop_order(L)  # simple: in order
    swapped = L.copy()
    k = len(L) // 2
    swapped[[5, k]] = swapped[[k, 5]]  # a walk across the loop and back: it crosses itself
    with pytest.raises(RuntimeError, match="not geometrically ordered"):
        ring.validate_loop_order(swapped)


def test_a_failed_step_in_the_ring_stage_is_retried_once_after_a_refine_of_the_sharpest_element():
    """``CollapseDriver._failed_step_retry``: in the ring stage with recentred far kinks, the
    ring's sharpest element above the floor turning beyond ``edge_turn_refine`` rebuilt at the
    size its turn asks for (the ring refine's own rebuild, a reading with ``retry``) and the step
    made again; a second failure of the same step, no ring, or no such element: no retry."""
    from types import SimpleNamespace

    from faraday_singularity.driver import CollapseDriver

    turn = np.array([10.0, 85.6, 120.0, 40.0])
    L = np.array([2e-5, 1.55e-5, 1.0e-6, 3e-6])  # the 120 degrees at the floor: not refined
    mid = np.array([[1e-4, 4.6], [1.6e-4, 4.6116], [3e-4, 4.6115], [4e-4, 4.6114]])
    calls, logged = [], []

    def fake(chain="ring", turns=turn):
        f = SimpleNamespace(
            state=SimpleNamespace(step=7251), reconnect=SimpleNamespace(edge_turn_refine=30.0),
            s=SimpleNamespace(refine_ratio=1.5), _h_floor=lambda: 7.8125e-7,
            _refined_chain=lambda: chain,
            _post_cut_state=lambda: {"edges": {"ring": {"edges": {
                "turn": turns, "L": L, "mid": mid, "index": np.array([1, 3, 5, 7])}}}},
            _event=lambda name, m, **kw: logged.append((name, kw)))
        f._ring_refine_once = lambda m, reading, attempt: (calls.append(reading) or
                                                         ({"m": "refined"}, None))
        return f

    f = fake()
    out = CollapseDriver._failed_step_retry(f, {"m": 0}, {"reason": "step-failed", "error": "x"})
    assert out == ({"m": "refined"}, None) and len(calls) == 1
    assert calls[0]["index"] == 3 and calls[0]["turn"] == 85.6 and calls[0]["retry"] is True
    assert logged[0][0] == "step-retry"
    # once per step
    assert CollapseDriver._failed_step_retry(f, {"m": 0}, {"reason": "step-failed"}) is None
    f.state.step = 7252
    assert CollapseDriver._failed_step_retry(f, {"m": 0}, {"reason": "step-failed"}) is not None
    # no ring stage, or nothing above the floor that turns
    assert CollapseDriver._failed_step_retry(fake(chain=None), {"m": 0}, {}) is None
    flat = np.array([10.0, 20.0, 120.0, 25.0])
    assert CollapseDriver._failed_step_retry(fake(turns=flat), {"m": 0}, {}) is None


@pytest.mark.parametrize("outcome", ["none", "base", "error", "success"])
def test_the_run_loop_keeps_the_retry_stop_and_rolls_back_a_failed_retry(tmp_path, outcome):
    from faraday_singularity.reconnect import ReconnectSettings
    pb = NetProblem(fail_at=2)
    drv = _cut_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-3), max_steps=3)
    from faraday_singularity.events import EventSettings
    drv.events = EventSettings()
    drv._remesh_checks = lambda m: (m, None)  # keep the accepted mesh stepped-on
    calls = []
    def retry(m, stop):
        calls.append(pb.t)
        if outcome == "none":
            return None
        if outcome == "base":
            return m, {"reason": "interface-unresolved", "trigger": "ring-refine-base"}
        if outcome == "error":
            pb.t += 0.1  # a failed rebuild must not leak this rejected state
            raise RuntimeError("retry rebuild failed")
        pb.fail_at = None
        return m, None
    drv._failed_step_retry = retry
    result = drv.run(1e-9)
    drv.close()
    assert len(calls) == 1
    stop = result["stop"]
    if outcome == "base":
        assert stop["trigger"] == "ring-refine-base" and pb.calls == 2
    elif outcome == "success":
        assert stop["reason"] == "max-steps" and result["steps"] == 3
    else:
        assert stop["reason"] == "step-failed" and result["steps"] == 1
        if outcome == "error":
            assert "retry rebuild failed" in stop["retry_error"] and len(pb.restored) == 2
            assert pb.t == calls[0]
    meta = json.loads(Path(stop["end_state"]).with_suffix(".json").read_text())
    assert meta["mesh_fresh"] is False
    if outcome == "base":
        assert stop["failed_step"]["error"] == "RuntimeError: OomphException"


def test_the_retry_does_not_refine_a_ring_element_beyond_the_turn_cap():
    from types import SimpleNamespace
    from faraday_singularity.driver import CollapseDriver, RING_REFINE_TURN_MAX
    f = SimpleNamespace(state=SimpleNamespace(step=1),
        reconnect=SimpleNamespace(edge_turn_refine=30), s=SimpleNamespace(refine_ratio=1.5),
        _h_floor=lambda: 1e-6, _refined_chain=lambda: "ring",
        _post_cut_state=lambda: {"edges": {"ring": {"edges": {
            "turn": np.array([RING_REFINE_TURN_MAX + 1]), "L": np.array([1e-5])}}}})
    assert CollapseDriver._failed_step_retry(f, {}, {}) is None


def test_a_loop_with_a_repeated_interior_node_is_refused():
    L = _graded_loop()
    L = np.insert(L, 5, L[5], axis=0)
    with pytest.raises(RuntimeError, match="zero-length"):
        ring.validate_loop_order(L)



@pytest.mark.parametrize("kind", ["guard", "event"])
def test_a_retry_reenters_the_loop_handlers_for_a_guard_or_event(tmp_path, kind):
    from faraday_singularity.reconnect import ReconnectSettings
    from faraday_singularity.events import EventSettings, EVENT_REQUEST
    pb = NetProblem(fail_at=2)
    drv = _cut_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-3), max_steps=3)
    drv.events = EventSettings()
    drv._remesh_checks = lambda m: (m, None)
    handled = []
    original_stop = drv._stop
    request = ({"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "ring"}
               if kind == "guard" else {"reason": EVENT_REQUEST, "kind": "ring-fold"})
    def retry(m, stop):
        drv._stop = lambda m: request
        return m, request
    def handle(m, stop):
        handled.append(stop)
        drv._stop = original_stop
        pb.fail_at = None
        return m, None
    drv._failed_step_retry = retry
    drv._ring_refine_due = lambda stop: kind == "guard" and stop is request
    drv._ring_refine = handle
    drv._meet_event = handle
    result = drv.run(1e-9)
    drv.close()
    assert handled == [request]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 3


def test_a_retry_rebuild_failure_restores_its_focus_and_refinement_bookkeeping(tmp_path):
    import copy
    from faraday_singularity.events import EventSettings
    from faraday_singularity.mesh import Focus, FocusRegion
    from faraday_singularity.reconnect import ReconnectSettings
    pb = NetProblem(fail_at=2)
    focus = Focus(size=1e-4, radius=0.1, grading=0.3, regions=(
        FocusRegion("original", size=1e-4, r=0.0, z=4.6, radius=0.1),))
    pb.template.focus = focus
    drv = _cut_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-3), max_steps=3)
    drv.events = EventSettings()
    drv._remesh_checks = lambda m: (m, None)
    expected = {}
    def fail_rebuild(*args, **kwargs):
        assert pb.template.focus != expected["focus"]
        assert drv.state.far_yields > expected["far_yields"]
        raise RuntimeError("synthetic gmsh failure")
    pb.force_remesh = fail_rebuild
    drv._refined_chain = lambda: "ring"
    def retry(m, stop):
        expected.update(kinks=copy.deepcopy(drv.state.kinks), far_yields=drv.state.far_yields,
                        last_remesh_step=drv.state.last_remesh_step, focus=pb.template.focus)
        return drv._ring_refine_once(m, {"reason": "interface-unresolved", "trigger": "edge-turn",
            "chain": "ring", "index": 3, "turn": 85, "L": 1e-5, "r": 1e-4, "z": 4.6}, 1)
    drv._failed_step_retry = retry
    result = drv.run(1e-9)
    drv.close()
    assert "synthetic gmsh failure" in result["stop"]["retry_error"]
    assert drv.state.kinks == expected["kinks"]
    assert drv.state.far_yields == expected["far_yields"]
    assert drv.state.last_remesh_step == expected["last_remesh_step"]
    assert pb.active_mesh_template().focus == expected["focus"]
    meta = json.loads(Path(result["stop"]["end_state"]).with_suffix(".json").read_text())
    assert meta["focus"] == expected["focus"].to_dict()
    assert meta["kinks"] == expected["kinks"]
    assert meta.get("far_yields", 0) == expected["far_yields"]


@pytest.mark.parametrize("end", ["end-time", "dt-floor"])
def test_a_successful_retry_reports_the_actual_endpoint(tmp_path, end):
    from faraday_singularity.reconnect import ReconnectSettings
    pb = NetProblem(fail_at=2)
    drv = _cut_driver(tmp_path, pb, ReconnectSettings(delta=1e-3, dt0=1e-3),
                      max_steps=100, t_end=1e-5)
    drv._remesh_checks = lambda m: (m, None)
    def retry(m, stop):
        pb.fail_at = None
        if end == "dt-floor":
            drv._dt_min = lambda: math.inf
        return m, None
    drv._failed_step_retry = retry
    result = drv.run(1e-9)
    drv.close()
    assert result["stop"]["reason"] == end
    if end == "end-time":
        assert pb.t == drv.s.t_end
    else:
        assert result["stop"]["failed_step"]["error"] == "RuntimeError: OomphException"
