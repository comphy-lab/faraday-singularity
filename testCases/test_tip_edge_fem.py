"""A controlled FEM tip edge: made, hard refusals and lossless rollback."""

from pathlib import Path
import sys
import json
import numpy as np
import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
pytest.importorskip("pyoomph")
import ring_fem_state as S  # noqa: E402
from test_ring_fem import state_vector  # noqa: E402
from faraday_singularity import driver as dv, tipcontact  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.events import EventSettings  # noqa: E402
from faraday_singularity.reconnect import p2_edges  # noqa: E402


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "addition",
        "momentum",
        "base",
        "coverage",
        "volume",
        "guard",
        "no-op",
        "event",
        "window",
    ],
)
def test_a_fem_tip_edge_is_audited_and_failed_checks_restore_all_levels(
    tmp_path, monkeypatch, failure
):
    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.impose(pb)
        record = S.gas_cut_record(pb)
        loop = next(c for c in pb.interface_segments() if not c["open"])
        P = np.column_stack([loop["r"], loop["z"]])
        i = len(P) - (6 if failure == "addition" else 4)
        mid = next(
            n for n in pb.get_mesh("liquid/interface").nodes() if (n.x(0), n.x(1)) == tuple(P[i])
        )
        a, b = P[i - 1], P[i + 1]
        chord = b - a
        L = np.linalg.norm(chord)
        normal = np.array([-chord[1], chord[0]]) / L
        for t in range(mid.ntstorage()):
            for k in range(2):
                mid.set_x_at_t(
                    t,
                    k,
                    float(
                        (a[k] + b[k]) / 2
                        + (-0.26 if failure == "addition" else 0.26) * L * normal[k]
                    ),
                )
        pb.invalidate_cached_mesh_data()
        block = EventSettings.from_dict(
            {
                "groove": {"enabled": True, "sigma_V": 1e-8, "sigma_Pz": 1e-6},
                "jetcore": {"enabled": True, "far_kinks": "recentre"},
            }
        )
        drv = CollapseDriver(
            pb,
            CollapseSettings(t_end=1, dt_max=1e-5),
            tmp_path / "run",
            reconnect=S.RECON,
            events=block,
        )
        drv.reset_after_cut(record)
        m = drv.measure()
        P = np.column_stack([loop["r"], loop["z"]])
        e = p2_edges(P)
        j = (i - 1) // 2
        stop = {
            "reason": "interface-unresolved",
            "trigger": "edge-turn",
            "chain": "bubble",
            "index": i,
            "r": mid.x(0),
            "z": mid.x(1),
            "L": float(e["L"][j]),
            "turn": float(e["turn"][j]),
            "fresh": True,
        }

        def history():
            fields = pb.get_mesh("liquid").get_nodal_field_indices()
            velocity = [fields[k] for k in ("velocity_x", "velocity_y")]
            nodes = sorted(pb.get_mesh("liquid/interface").nodes(), key=lambda n: (n.x(0), n.x(1)))
            return np.array(
                [
                    [*[n.x_at_t(t, k) for k in range(2)], *[n.value_at_t(t, k) for k in velocity]]
                    for n in nodes
                    for t in range(pb.timestepper.nprev_values() + 1)
                ]
            )

        initial_volume = pb.observables()["volume"]
        vector = state_vector(pb).copy()
        histories = history().copy()
        native_input = pb._snapshot_state()
        assert drv._fresh_mesh()
        later = (
            {"reason": "event-request", "kind": "test-later"}
            if failure == "event"
            else {"reason": "bubble-vanishing", "window_end": True}
            if failure == "window"
            else None
        )
        monkeypatch.setattr(drv, "_stop", lambda m: later)
        if failure == "base":
            original = dv.base_guard_checked
            monkeypatch.setattr(
                dv, "base_guard_checked", lambda *args: dict(original(*args), ok=False)
            )
        if failure in ("momentum", "coverage"):
            original = tipcontact.sliver_momentum

            def changed(*a):
                q = original(*a)
                if failure == "momentum":
                    q["Pz"] = 1.0
                else:
                    q["found"] -= 1
                return q

            monkeypatch.setattr(tipcontact, "sliver_momentum", changed)
        if failure == "volume":
            from faraday_singularity import surgery

            original = surgery.box_integrals
            calls = []

            def wrong_volume(*a, **kw):
                result = original(*a, **kw)
                if calls:
                    result["V_grid"] += 1
                calls.append(True)
                return result

            monkeypatch.setattr(surgery, "box_integrals", wrong_volume)
        if failure in ("guard", "no-op"):
            original = pb.straighten_interface_edges

            def another_bad_edge(*a, apply=True, **kw):
                result = original(*a, apply=apply, **kw)
                if apply:
                    # Leave the actual moved edge off-centre: the pure guard
                    # must refuse it, with no call to the mutating _stop path.
                    if failure == "no-op":
                        mid.set_x(0, float((a0[0] + b0[0]) / 2 + 0.26 * L * normal[0]))
                        mid.set_x(1, float((a0[1] + b0[1]) / 2 + 0.26 * L * normal[1]))
                    else:
                        # A different element develops a real guard stop after
                        # the admitted edge changed: the pure guard must see it.
                        j = i - 8
                        other = next(
                            n
                            for n in pb.get_mesh("liquid/interface").nodes()
                            if (n.x(0), n.x(1)) == tuple(P[j])
                        )
                        chord_other = P[j + 1] - P[j - 1]
                        normal_other = np.array([-chord_other[1], chord_other[0]])
                        target = 0.5 * (P[j - 1] + P[j + 1]) + 0.26 * normal_other
                        for dim in range(2):
                            other.set_x(dim, float(target[dim]))
                    pb.invalidate_cached_mesh_data()
                return result

            a0, b0 = a.copy(), b.copy()
            monkeypatch.setattr(pb, "straighten_interface_edges", another_bad_edge)
            monkeypatch.setattr(
                drv, "_stop", lambda m: pytest.fail("stop called inside rejected audit")
            )
        if failure in ("momentum", "coverage", "base", "volume"):
            monkeypatch.setattr(
                drv,
                "_tip_edge_guard",
                lambda: pytest.fail("hard closure must fail before guard read"),
            )
        after, found = drv._tip_edge_straighten(m, stop)
        rec = json.loads(next((tmp_path / "run/events").glob("tip_edge_*.json")).read_text())
        if failure in ("momentum", "base", "coverage", "volume", "guard", "no-op"):
            assert found["trigger"] == "tip-edge-unclosed"
            assert rec["outcome"] == "rolled-back" and rec["closures"]["hard"]
            np.testing.assert_array_equal(state_vector(pb), vector)
            np.testing.assert_array_equal(history(), histories)
            assert drv._fresh_mesh() and pb._jacobian_cache is None
            if failure == "no-op":
                assert "T_V_mesh" in rec["closures"]["hard"]
            if failure == "guard":
                assert "T_guard" in rec["closures"]["hard"]
            if failure != "volume":
                assert pb.observables()["volume"] == initial_volume
            assert found["guard"]["index"] == i and "tau" in found and "tau_p" in found
            assert not drv._tip_edge_balances
        else:
            assert found == later and rec["outcome"] == "made" and rec["closures"]["hard"] == []
            assert rec["base"]["ok"] and rec["quadrature"]["found"] == rec["quadrature"]["points"]
            assert len(drv._tip_edge_balances) == 1
            assert (rec["dV"] > 0) == (failure == "addition")
            assert rec["quadrature"]["KE"] >= 0
            assert abs(pb.observables()["volume"] - initial_volume - rec["dV"]) <= 0.01 * abs(
                rec["dV"]
            )
            for key in ("T_V_mesh", "T_V_plan", "T_P_mesh"):
                assert rec["closures"]["values"][key] <= rec["closures"]["limits"][key]
            assert drv._removed()["events"] == 0
            # An axis-fold recurrence stays stopped. Once already straight,
            # this edge no longer qualifies, so the stale guard is unchanged.
            # The restored bulged input below tests the five-step window.
            assert drv._tip_edge_straighten(after, dict(stop, axis_fold_recurs={"last_step": 0}))[
                1
            ]["axis_fold_recurs"]
            assert drv._tip_edge_straighten(after, stop)[1] == stop
            path = drv.keep_state(tmp_path / "run/states/after.dump")
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["tip_edge_balances"] == drv._tip_edge_balances
            kept = list(drv._tip_edge_balances)
            drv._tip_edge_balances = []
            drv.restore(meta)
            assert drv._tip_edge_balances == kept
            balance = drv._removed()
            assert balance["V"] == pytest.approx(-rec["dV"])
            assert balance["Pz"] == pytest.approx(-rec["closures"]["Pz_liquid_change"])
            if failure is None:
                from faraday_singularity.axisrepair import (
                    AXIS_REPAIR_BOUND,
                    AXIS_REPAIR_RECUR_STEPS,
                )

                pb._restore_state(native_input)
                pb.invalidate_cached_mesh_data()
                drv._stepped_rc = None
                drv._post_cut_cache = drv._stop_cache = None
                assert "tip_edge_recurs" in drv._tip_edge_straighten(drv.measure(), stop)[1]
                drv.state.step += AXIS_REPAIR_RECUR_STEPS
                drv._tip_edge_balances = [
                    dict(V=AXIS_REPAIR_BOUND * S.RECON.delta**3, Pz=0.0, KE=0.0, step=0)
                ]
                _, refused = drv._tip_edge_straighten(drv.measure(), stop)
                assert (
                    refused["tip_edge_volume"]["cumulative_abs"]
                    > refused["tip_edge_volume"]["bound"]
                )
        drv.close()


def _lower_an_unrelated_bulk_jacobian(pb):
    """Deform one interior P2 edge far from either tip, keeping its elements positive."""
    from faraday_singularity.reconnect import p2_jacobian_bounds

    bulk = pb.get_mesh("liquid")
    element = next(e for e in bulk.elements() if e.nnode() == 6
                   and all(e.node_pt(k).x(0) > 1.5 and 0.1 < e.node_pt(k).x(1) < 0.4
                           for k in range(6)))
    local = np.array([element.local_coordinate_of_node(k) for k in range(6)])
    k = next(k for k, point in enumerate(local) if np.any(np.isclose(point, 0.5)))
    node = element.node_pt(k)
    point = np.array([node.x(0), node.x(1)])
    values, cells, _, fields = bulk.to_numpy(False, True, 0, False)[:4]
    xy = np.asarray(values)[:, [fields["coordinate_x"], fields["coordinate_y"]]].copy()
    cells = np.asarray(cells)
    index = int(np.argmin(np.linalg.norm(xy - point, axis=1)))
    assert np.linalg.norm(xy[index] - point) < 1e-12
    affected = cells[np.any(cells == index, axis=1)]
    assert len(affected) == 2  # an interior edge, with no free-surface node moved
    length = max(np.linalg.norm(np.array([element.node_pt(j).x(0),
                                         element.node_pt(j).x(1)]) - point)
                 for j in range(6))
    for fraction in np.linspace(0, 0.5, 501):
        xy[index] = point + [0, fraction * length]
        bounds = p2_jacobian_bounds(xy, affected)
        if 0.1 < float(np.min(bounds)) < 0.19:
            break
    else:
        pytest.fail("the remote interior edge did not reach a positive low Jacobian bound")
    displacement = xy[index] - point
    for t in range(node.ntstorage()):
        for dim in range(2):
            node.set_x_at_t(t, dim, node.x_at_t(t, dim) + float(displacement[dim]))
    pb.invalidate_cached_mesh_data()
    pb._jacobian_cache = None
    bound = pb.curved_jacobian_bound(S.RECON.jacobian_min)
    assert 0 < bound["min"] < S.RECON.jacobian_min
    assert not bound["on_interface"] and bound["r"] > 1.5


def _weaken_the_changed_triangle(pb, point, chord):
    """Keep a valid dipped triangle whose boundary straightening would lower its bound."""
    from itertools import product
    from faraday_singularity.reconnect import p2_jacobian_bounds

    patch = tipcontact.edge_patch(pb, point, chord, np.zeros(2))
    local = np.linalg.inv(patch["inverse"])[:, 1:3]
    surface = {(n.x(0), n.x(1)) for n in pb.get_mesh("liquid/interface").nodes()}
    values, cells, _, fields = pb.get_mesh("liquid").to_numpy(False, True, 0, False)[:4]
    xy = np.asarray(values)[:, [fields["coordinate_x"], fields["coordinate_y"]]].copy()
    cells = np.asarray(cells)
    for k, node in enumerate(patch["nodes"]):
        if not np.any(np.isclose(local[k], 0.5)) or node.x(0) <= 0 \
                or (node.x(0), node.x(1)) in surface:
            continue
        length = np.linalg.norm(point - chord) / 0.15
        for dr, dz in product(np.linspace(-0.4, 0.4, 25), repeat=2):
            displacement = length * np.array([dr, dz])
            candidate = dict(patch)
            candidate["positions"] = patch["positions"].copy()
            candidate["planned_positions"] = patch["planned_positions"].copy()
            candidate["positions"][k] += displacement
            candidate["planned_positions"][k] += displacement
            before = tipcontact.patch_jacobian_bound(candidate, initial=True)
            after = tipcontact.patch_jacobian_bound(candidate, planned=True)
            if before >= S.RECON.jacobian_min and 0 < after < 0.19:
                index = int(np.argmin(np.linalg.norm(xy - patch["positions"][k], axis=1)))
                trial = xy.copy()
                trial[index] += displacement
                affected = cells[np.any(cells == index, axis=1)]
                if np.min(p2_jacobian_bounds(trial, affected)) <= 0:
                    continue
                for level in range(node.ntstorage()):
                    for dim in range(2):
                        node.set_x_at_t(level, dim, node.x_at_t(level, dim)
                                        + float(displacement[dim]))
                pb.invalidate_cached_mesh_data()
                pb._jacobian_cache = None
                assert pb.curved_jacobian_bound(S.RECON.jacobian_min)["min"] > 0
                return
    pytest.fail("no valid dipped triangle whose straightening lowers its own bound was found")


@pytest.mark.parametrize("failure", [None, "base", "jacobian"])
@pytest.mark.parametrize("unrelated_low_bound", [False, True])
def test_a_stepped_terminal_dimple_uses_the_same_audit_and_rollback(
    tmp_path, monkeypatch, failure, unrelated_low_bound
):
    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.impose(pb)
        record = S.gas_cut_record(pb)
        if unrelated_low_bound:
            _lower_an_unrelated_bulk_jacobian(pb)
        loop = next(c for c in pb.interface_segments() if not c["open"])
        P = np.column_stack([loop["r"], loop["z"]])
        point = P[-2]
        mid = next(n for n in pb.get_mesh("liquid/interface").nodes()
                   if (n.x(0), n.x(1)) == tuple(point))
        # Reproduce the saved cap's annular dip at the synthetic mesh's scale.
        a, b = P[-3], P[-1]
        chord = b - a
        normal = np.array([-chord[1], chord[0]])
        for t in range(mid.ntstorage()):
            for k in range(2):
                mid.set_x_at_t(t, k, float(0.5 * (a[k] + b[k]) + 0.15 * normal[k]))
        pb.invalidate_cached_mesh_data()
        if failure == "jacobian":
            _weaken_the_changed_triangle(pb, np.array([mid.x(0), mid.x(1)]), 0.5 * (a + b))
        block = EventSettings.from_dict({
            "groove": {"enabled": True, "sigma_V": 1e-8, "sigma_Pz": 1e-6},
            "jetcore": {"enabled": True, "far_kinks": "recentre"}})
        drv = CollapseDriver(pb, CollapseSettings(t_end=1, dt_max=1e-5), tmp_path / "run",
                             reconnect=S.RECON, events=block)
        drv.reset_after_cut(record)
        drv._stepped_rc = pb.remesh_count
        m = drv.measure()
        vector = state_vector(pb).copy()
        history = np.array([[mid.x_at_t(t, k) for k in range(2)]
                            for t in range(mid.ntstorage())])
        assert not drv._fresh_mesh()
        loop = next(c for c in pb.interface_segments() if not c["open"])
        P = np.column_stack([loop["r"], loop["z"]])
        assert tipcontact.terminal_dimple(P, S.RECON.edge_turn_refine, 0.1) is not None
        if failure == "base":
            original = dv.base_guard_checked
            monkeypatch.setattr(dv, "base_guard_checked",
                                lambda *args: dict(original(*args), ok=False))
        monkeypatch.setattr(drv, "_stop", lambda m: None)
        # A valid stepped dip remains unchanged until the native net has failed.
        assert drv._tip_edge_straighten(m, None) == (m, None)
        np.testing.assert_array_equal(state_vector(pb), vector)
        failed = {"reason": "step-failed", "error": "InvertedElementRemeshRequest: folded"}
        _, stop = drv._failed_step_retry(m, failed)
        rec = json.loads(next((tmp_path / "run/events").glob("tip_edge_*.json")).read_text())
        assert rec["guard"]["trigger"] == "tip-dimple" and not rec["guard"]["fresh"]
        assert rec["failed_step"] == failed
        if failure is not None:
            hard = "T_base" if failure == "base" else "T_guard"
            assert stop["trigger"] == "tip-edge-unclosed" and rec["closures"]["hard"] == [hard]
            if failure == "jacobian":
                assert rec["jacobian"]["before"] >= rec["jacobian"]["limit"]
                assert 0 < rec["jacobian"]["after"] < rec["jacobian"]["limit"]
            np.testing.assert_array_equal(state_vector(pb), vector)
            np.testing.assert_array_equal(history, [[mid.x_at_t(t, k) for k in range(2)]
                                                    for t in range(mid.ntstorage())])
            assert not drv._tip_edge_balances
        else:
            assert stop is None and rec["closures"]["hard"] == [] and rec["base"]["ok"]
            assert len(drv._tip_edge_balances) == 1
            if unrelated_low_bound:
                pb._jacobian_cache = None
                bound = pb.curved_jacobian_bound(S.RECON.jacobian_min)
                assert 0 < bound["min"] < S.RECON.jacobian_min
                assert not bound["on_interface"] and bound["r"] > 1.5
        assert not drv._fresh_mesh()
        assert drv._failed_step_retry(m, failed) is None  # at most once at this step
        drv.close()


@pytest.mark.parametrize("outcome", ["success", "second-failure", "snapshot-error", "stop-error"])
def test_the_run_loop_uses_real_tip_audits_and_keeps_native_state_with_its_booking(
    tmp_path, monkeypatch, outcome
):
    from dataclasses import replace
    from pyoomph.generic.problem import InvertedElementRemeshRequest

    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.impose(pb)
        record = S.gas_cut_record(pb)
        record["sizing"]["h_tip"] = S.H_FINE  # this fixture's actual tip mesh scale
        c = next(c for c in pb.interface_segments() if not c["open"])
        P = np.column_stack([c["r"], c["z"]])
        a, point, b = P[-3:]
        mid = next(n for n in pb.get_mesh("liquid/interface").nodes()
                   if (n.x(0), n.x(1)) == tuple(point))
        chord = b - a
        normal = np.array([-chord[1], chord[0]])
        for t in range(mid.ntstorage()):
            for k in range(2):
                mid.set_x_at_t(t, k, float(0.5 * (a[k] + b[k]) + 0.15 * normal[k]))
        pb.invalidate_cached_mesh_data()
        block = EventSettings.from_dict({
            "ring": {"enabled": True, "gap": 0.2},
            "groove": {"enabled": True, "sigma_V": 1e-8, "sigma_Pz": 1e-6},
            "jetcore": {"enabled": True, "far_kinks": "recentre"}})
        # Exclude the fixture's separate shallow head neck from this recovery case.
        recon = replace(S.RECON, newton_min_iterations=2, neck_depth=0.9)
        drv = CollapseDriver(pb, CollapseSettings(t_end=1, dt_max=1e-5, max_steps=1),
                             tmp_path / "run", reconnect=recon, events=block)
        drv.reset_after_cut(record)
        drv._stepped_rc = pb.remesh_count
        initial = state_vector(pb).copy()
        original_run, original_snapshot, original_stop = pb.run, drv._snapshot, drv._stop
        inputs, before_snapshot, stop_snapshot, injected = [], [], [], []

        def step(t_end, **kw):
            pb._snapshot_state()  # same native snapshot collection as the inversion net
            inputs.append(state_vector(pb).copy())
            if len(inputs) == 1 or outcome == "second-failure":
                raise InvertedElementRemeshRequest("mesh kept folding (injected test failure)")
            return original_run(t_end, **kw)

        def snapshot(m, reason):
            if reason == "stop":
                stop_snapshot.append((drv.state.snapshots, drv.state.last_snapshot_t,
                                      drv.state.last_snapshot_ell))
            if reason == "tip-edge-straighten":
                before_snapshot.append((drv.state.snapshots, drv.state.last_snapshot_t,
                                        drv.state.last_snapshot_ell))
            original_snapshot(m, reason)
            if reason == "tip-edge-straighten" and outcome == "snapshot-error":
                raise OSError("injected post-booking profile error")

        def stop(m):
            if outcome == "stop-error" and drv._tip_edge_balances and not injected:
                injected.append(True)
                raise RuntimeError("injected post-booking stop error")
            return original_stop(m)

        monkeypatch.setattr(pb, "run", step)
        monkeypatch.setattr(drv, "_snapshot", snapshot)
        monkeypatch.setattr(drv, "_stop", stop)
        result = drv.run(1e-8)
        end = Path(result["stop"]["end_state"])
        meta = json.loads(end.with_suffix(".json").read_text())
        audit = json.loads(next((tmp_path / "run/events").glob("tip_edge_*.json")).read_text())
        events = [json.loads(x) for x in (tmp_path / "run/events.jsonl").read_text().splitlines()]
        assert audit["inversion_site"].startswith("unknown")
        if outcome in ("snapshot-error", "stop-error"):
            assert result["stop"]["reason"] == "step-failed"
            assert "post-booking" in result["stop"]["retry_error"]
            np.testing.assert_array_equal(state_vector(pb), initial)
            assert not drv._tip_edge_balances and "tip_edge_balances" not in meta
            assert audit["outcome"] == "rolled-back" and "retry_error" in audit
            assert any(e["event"] == "tip-edge-rollback" for e in events)
            assert drv._removed()["V"] == drv._removed()["Pz"] == 0
            assert meta["snapshots"] == before_snapshot[0][0]
            assert stop_snapshot[0] == before_snapshot[0]
        else:
            assert len(inputs) == 2 and len(drv._tip_edge_balances) == 1
            assert not np.array_equal(inputs[0], inputs[1])
            assert meta["tip_edge_balances"] == drv._tip_edge_balances
            assert audit["outcome"] == "made" and audit["closures"]["hard"] == []
            if outcome == "second-failure":
                assert result["stop"]["reason"] == "step-failed" and result["steps"] == 0
                np.testing.assert_array_equal(state_vector(pb), inputs[1])
                assert result["stop"]["failed_step"]["error"].startswith("InvertedElementRemeshRequest:")
            else:
                assert result["steps"] == 1 and result["stop"]["reason"] == "max-steps"
        drv.close()
