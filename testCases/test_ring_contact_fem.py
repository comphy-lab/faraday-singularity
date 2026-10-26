"""A real mesh refinement keeps one ring and closes the zero-removal budgets.

The phase and dispatch contracts are tested in test_ring_contact.py; this test
exercises the new audit around the existing ring rebuild on the synthetic pierced
state, with that state's existing ring budget settings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "testCases")]
pytest.importorskip("pyoomph")

import ring_fem_state as S
from test_ring_fem import state_vector
from test_ring_refine_fem import BLOCK
from faraday_singularity import driver as dv
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.events import EventSettings
from faraday_singularity.ring import loop_edges, loop_spline_contact


def test_audited_refinement_on_a_finite_element_ring(tmp_path, monkeypatch):
    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.gas_cut_record(pb)
        S.impose(pb)
        drv = CollapseDriver(pb, CollapseSettings(t_end=10, dt_max=.01, max_steps=0,
                                                  state_dt=0, state_keep=5),
                             tmp_path / "run", reconnect=S.RECON, events=EventSettings.from_dict(BLOCK))
        result = drv.run(1e-8)
        assert result["stop"]["reason"] == "max-steps" and result["events"]["made"] == 1
        loop = pb.interface_loops()[0]
        P = np.column_stack([loop["r"], loop["z"]])
        e = loop_edges(P)["edges"]
        k = int(np.argmax(e["turn"]))
        reading = {"reason": "interface-unresolved", "chain": "ring", "trigger": "liquid-kink",
                   "L": float(e["L"][k]), "r": float(e["mid"][k, 0]),
                   "z": float(e["mid"][k, 1]), "liquid_contact": {"synthetic_audit": True}}
        after, stop = drv._ring_liquid_refine(drv.measure(), reading, 1)
        record = json.loads(next((tmp_path / "run/events").glob("ring_contact_*.json")).read_text(),
                            parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"))
        assert record["closures"]["hard"] == [] and record["base"]["ok"]
        assert record["closures"]["V_added"] == record["closures"]["V_removed"] == 0
        assert record["before"]["loops"] == record["after"]["loops"] == 1
        assert stop is None, stop
        loop = pb.interface_loops()[0]
        assert loop_spline_contact(np.column_stack([loop["r"], loop["z"]])) is None
        assert after["time"] == drv.measure()["time"]
        # A rejected new-handler rebuild must restore the native state and its
        # fresh/stepped meaning even though reloading a mesh changes its count.
        real = dv.base_guard_checked

        def moved(*args):
            return dict(real(*args), ok=False)

        monkeypatch.setattr(dv, "base_guard_checked", moved)
        live = drv.state
        for attempt, fresh in ((2, False), (3, True)):
            drv._stepped_rc = None if fresh else pb.remesh_count
            vector = state_vector(pb).copy()
            focus = pb.active_mesh_template().focus
            old_kinks = set(drv.state.kinks)
            before_count = pb.remesh_count
            before_profiles, before_events = live.snapshots, len(live.events)
            restored, refused = drv._ring_liquid_refine(drv.measure(), reading, attempt)
            assert refused["trigger"] == "ring-contact-unclosed" and "C_base" in refused["hard"]
            assert drv.state is live and live.snapshots == before_profiles+1
            assert [e["event"] for e in live.events[before_events:]] == ["ring-refine", "ring-contact-refine"]
            assert pb.remesh_count > before_count and drv._fresh_mesh() is fresh
            assert pb.active_mesh_template().focus is focus and set(drv.state.kinks) == old_kinks
            np.testing.assert_array_equal(state_vector(pb), vector)
            rejected = tmp_path / "run/profiles" / f"profile_{before_profiles:06d}.txt"
            saved = rejected.read_bytes()
            drv._snapshot(restored, "stop")
            assert live.snapshots == before_profiles+2 and rejected.read_bytes() == saved
            drv._events.flush()
            log = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
            assert len(log) == len(live.events)
        drv.close()


@pytest.mark.parametrize("mode", ["spline", "step"])
def test_a_fem_contact_is_filled_through_the_run_loop(tmp_path, monkeypatch, mode):
    import test_groove_fem as GF
    from faraday_singularity import surgery
    from faraday_singularity.mesh import Focus, FocusRegion, RingSizing

    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.gas_cut_record(pb)
        template = pb.active_mesh_template()
        named = dict(surgery._named_chains(pb)[0])
        tip = np.array([GF.RING_C[0]+GF.RING_A+.03, GF.RING_C[1]])
        template.ring_stage = True
        template.ring_sizing = RingSizing(h_ring=.004, grading=.3)
        template.focus = Focus(size=GF.H_FINE, radius=.2, grading=.3, regions=(
            FocusRegion("cavity_tip", size=.004, r=0, z=.95, radius=.1),
            FocusRegion("finger", size=GF.H_FINE, r=float(tip[0])-.015, z=float(tip[1]), radius=.05)))
        geometry = GF.finger_loop()
        if mode == "step":
            from test_floor_fill import teardrop
            # A meshed gas wedge, with its sharp apex at the largest-r seam.
            geometry = 2*tip-teardrop(apex=tip, length=.04, h=GF.H_FINE, arc_points=121)[0]
        template.pending_ring_geometry = {"chains": [named["open"]], "loops": [geometry]}
        pb._ring_transfer = {"axis_bulk": True, "reach": .1}
        try:
            pb.force_remesh({template})
        finally:
            pb._ring_transfer = None
        pb._domains_to_remesh.clear()
        loop = pb.interface_loops()[0]
        P = np.column_stack([loop["r"], loop["z"]])
        pb.ring = GF.ring_record(pb, P)
        # The zero velocity field is solenoidal. Calibrate the geometric box
        # noise on the smooth state, before compressing one boundary interval.
        noise = GF.site_noise(pb, (float(tip[0])-.015, float(tip[1])), 4*S.DELTA)
        loop = pb.interface_loops(with_node_ids=True)[0]
        P = np.column_stack([loop["r"], loop["z"]])
        nodes = {(n.x(0), n.x(1)): n for n in pb.get_mesh("liquid/interface").nodes()}
        if mode == "spline":
            a = int(np.argmax(P[:, 0]))
            a -= a % 2
            b = (a+1) % len(P)
            mid = nodes[tuple(P[b])]
            compressed = P[a]+.05*(P[b]-P[a])
        else:
            # A graded remote arc is legal on the simple input. On a crossed
            # candidate the normal reader must reject its chord ratio.
            b = (len(P)//2) | 1
            mid = nodes[tuple(P[b])]
            compressed = P[b]  # grade only the discarded candidate, after its solve
        for level in range(mid.ntstorage()):
            for dim in range(2):
                mid.set_x_at_t(level, dim, float(compressed[dim]))
        pb.invalidate_cached_mesh_data()
        pb.assign_initial_values_impulsive(GF.DT)
        block = GF.block(noise).to_dict()
        block["jetcore"] = {"enabled": True, "far_kinks": "recentre", "ring_kink_floor": True}
        drv = CollapseDriver(pb, CollapseSettings(t_end=10, dt_max=.01,
                                                  max_steps=int(mode == "step"),
                                                  state_dt=0, state_keep=5),
                             tmp_path / "run", reconnect=S.RECON, events=EventSettings.from_dict(block))
        drv.state.dt_prev = GF.DT
        if mode == "spline":
            found = drv._stop(drv.measure())
            assert found["trigger"] == "spline-contact" and found["contact"] == "spline", found
            request = drv._floor_fill_request(found)
            assert request is not None and request["side"] == "gas"
        else:
            from faraday_singularity.ringcontact import step_contact, step_wedge
            from faraday_singularity.ring import LOOP_STEP_RATIO, loop_steps
            loop = pb.interface_loops(with_node_ids=True)[0]
            P = np.column_stack([loop["r"], loop["z"]])
            assert loop_spline_contact(P) is None
            # Move the second vertex before the seam through the opposite
            # wall. The meshed cusp lies next to the seam, not exactly on it.
            candidate = P.copy()
            candidate[-2] += -(P[-4]-P[-2])+3*(P[0]-P[-2])
            candidate[b] = P[b-1]+.14*(P[(b+1) % len(P)]-P[b-1])
            d = loop_steps(candidate)
            assert max(d/np.minimum(np.roll(d, 1), np.roll(d, -1))) > LOOP_STEP_RATIO
            hit = step_contact(candidate)
            assert hit is not None and step_wedge(P, hit, stop_elements=3)[0] is not None
            boundary = {(n.x(0), n.x(1)): n for n in pb.get_mesh("liquid/interface").nodes()}
            ordered = [boundary[tuple(x)] for x in P]
            native_meet = drv._meet_ring_step_contact
            native_restore = drv._restore_step_snapshot
            attempts, rollback, input_vectors = [], [], []
            def solve(*args, **kw):
                # Prescribed nodal candidate: no transient solve or physical
                # onset is claimed. Snapshot, restore, fill and its projection
                # are native FEM operations; the real-state replay tests onset.
                input_vectors.append(state_vector(pb).copy())
                pb._snapshot_state()
                if not attempts:
                    for node, point in zip(ordered, candidate):
                        for dim in range(2):
                            node.set_x(dim, float(point[dim]))
                    pb.invalidate_cached_mesh_data()
                    after = pb.interface_loops(validate=False, with_node_ids=True)[0]
                    assert after["node_ids"][0] != loop["node_ids"][0]
                    with pytest.raises(RuntimeError, match="not geometrically ordered"):
                        pb.interface_loops()
                    pb._distortion_flags.append({"synthetic_rejected_candidate": True})
                attempts.append(True)
            def restore():
                native_restore()
                np.testing.assert_array_equal(state_vector(pb), input_vectors[0])
            def meet(m, contact):
                result = native_meet(m, contact)
                rollback.append(True)
                assert not pb.take_distortion_flags()
                return result
            monkeypatch.setattr(pb, "run", solve)
            monkeypatch.setattr(drv, "_restore_step_snapshot", restore)
            monkeypatch.setattr(drv, "_meet_ring_step_contact", meet)
            # The imposed candidate alone drives this software test; ordinary
            # geometry guards are covered by the spline and refinement cases.
            monkeypatch.setattr(drv, "_stop", lambda m: ({"reason": "test-contact-met"} if rollback else None))
            monkeypatch.setattr(drv, "_remesh_checks", lambda m: (m, None))
            monkeypatch.setattr(drv, "_quality_floor_flags", lambda: [])
            monkeypatch.setattr(drv, "_distortion_remesh", lambda flags: pytest.fail(f"stale flags {flags}"))
        result = drv.run(1e-8)
        drv.close()
        record = json.loads((tmp_path / "run/events/event_001.json").read_text())
        assert result["stop"]["reason"] == ("max-steps" if mode == "spline" else "test-contact-met"), result["stop"]
        assert record["outcome"] == "made" and record["closures"]["hard"] == []
        assert record["base_guard"]["ok"] and "bulk_band" in record["transfer"]
        key = "spline_fills" if mode == "spline" else "step_contact_fills"
        assert result["events"][key] == 1 and len(pb.interface_loops()) == 1
        loop = pb.interface_loops()[0]
        assert loop_spline_contact(np.column_stack([loop["r"], loop["z"]])) is None
        if mode == "step":
            assert result["steps"] == 0 and len(attempts) == 1 and rollback == [True]
            log = [json.loads(x) for x in (tmp_path / "run/events.jsonl").read_text().splitlines()]
            rolled = [e for e in log if e["event"] == "ring-contact-step-rollback"]
            assert len(rolled) == 1
            assert {"synthetic_rejected_candidate": True} in rolled[0]["discarded_distortion"]
            assert not any(e["event"] == "step-failed" for e in log)
            assert record["request"]["step_fill"]["guard"]["pre_step_phase"] == "gas"
