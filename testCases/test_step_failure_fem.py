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


def test_a_restore_across_a_rebuild_keeps_the_templates_project_attributes(tmp_path):
    """``FaradayProblem._restore_state``: a snapshot taken on one mesh and restored after a rebuild
    reloads the stored mesh from a new sibling of the problem's original template, which copies that
    one's attributes. With the active template itself such a sibling, as in a run restarted from a
    dump, and the original's focus, cut and ring sizing not the active one's, pyoomph's restore
    alone leaves the focus None and the ring stage off (r1h at b36c9c1, its rollback at 7251); the
    project's puts the active template's back; a restore onto the same mesh makes no sibling."""
    from faraday_singularity.mesh import Focus, FocusRegion
    from faraday_singularity.problem import FaradayProblem

    pyoomph_restore = super(FaradayProblem, FaradayProblem)._restore_state

    def focus(size):
        return Focus(size=size, radius=0.2, grading=0.3,
                     regions=(FocusRegion("x", size=size / 2, r=0.0, z=0.95, radius=0.1),))

    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        root = pb.mesh_template
        # the active template a sibling of the original, as after a restart from a dump
        pb.force_remesh({root})
        pb._domains_to_remesh.clear()
        first = pb._snapshot_state()
        pb.force_remesh({root})
        pb._domains_to_remesh.clear()
        pb._restore_state(first)
        assert pb.active_mesh_template() is not root
        for restore, kept in ((lambda s: pyoomph_restore(pb, s), False), (pb._restore_state, True)):
            root.focus, root.ring_stage, root.ring_sizing = None, False, None
            active = pb.active_mesh_template()
            active.focus, active.ring_stage, active.ring_sizing = focus(0.004), True, "sizing"
            snap = pb._snapshot_state()
            pb.force_remesh({active})
            pb._domains_to_remesh.clear()
            assert pb.active_mesh_template() is active  # a rebuild keeps the template
            restore(snap)
            new = pb.active_mesh_template()
            assert new is not active and new.remesher.template is new
            if kept:
                assert new.focus == active.focus and new.ring_stage and new.ring_sizing == "sizing"
                assert new.cut_sizing is active.cut_sizing
            else:  # pyoomph's: the original's
                assert new.focus is None and not new.ring_stage and new.ring_sizing is None
        # a restore onto the mesh the snapshot was taken on: no sibling, nothing changed
        active = pb.active_mesh_template()
        same = pb._snapshot_state()
        pb._restore_state(same)
        assert pb.active_mesh_template() is active and active.focus.size == 0.004
