"""After a cut, failures that ended runs without an end state, on stand-ins (no finite elements):

* a step that fails (Newton at every step size, or pyoomph's net for an inverted element giving up,
  as r1 at b46713a did at step 7503) is rolled back to the state it started from, the snapshot the
  net took before it (``FaradayProblem._snapshot_state`` keeps the first of a step,
  ``CollapseDriver._post_cut_solve``), and is the resumable stop ``step-failed`` with that state as
  the end state and an event ``step-failed`` (``CollapseDriver._failed_step_stop``); without a
  snapshot (the net not armed, a problem without it) the failure is raised as before;
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
