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
    out = drv.run(1e-9)
    drv.close()
    stop = out["stop"]
    assert stop["reason"] == "step-failed" and "OomphException" in stop["error"]
    assert out["steps"] == 1 and len(pb.restored) == 1
    t_start = pb.restored[0]["t"]
    assert pb.t == t_start and stop["dt_attempted"] > 0.0 and stop["net_rebuilds"] == 0
    end = Path(stop["end_state"])
    assert end.is_file() and end.name == "state_00000001.dump"
    assert json.loads(end.with_suffix(".json").read_text())["time"] == t_start
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
