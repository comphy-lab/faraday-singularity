"""``FaradayProblem._snapshot_state`` on a finite-element problem (the synthetic pierced state of
``ring_fem_state.py``): pyoomph's snapshot as before, its first of a step kept while the driver
collects them (``_step_snapshots`` a list), none kept otherwise, and the kept one restoring the
state it was taken of (the rollback of ``CollapseDriver._failed_step_stop``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import ring_fem_state as S  # noqa: E402


def test_the_first_snapshot_of_a_step_is_kept_and_restores_its_state(tmp_path):
    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        assert getattr(pb, "_step_snapshots", None) is None
        plain = pb._snapshot_state()
        assert isinstance(plain, bytes) and getattr(pb, "_step_snapshots", None) is None
        pb._step_snapshots = []
        first = pb._snapshot_state()
        t0 = pb.get_current_time(dimensional=False, as_float=True)
        pb.set_current_time(t0 + 1e-3, dimensional=False)
        second = pb._snapshot_state()
        assert pb._step_snapshots == [first] and second is not first
        pb._restore_state(pb._step_snapshots[0])
        assert pb.get_current_time(dimensional=False, as_float=True) == t0
        pb._step_snapshots = None
        pb._snapshot_state()
        assert pb._step_snapshots is None
