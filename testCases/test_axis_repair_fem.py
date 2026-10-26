"""Axis repair on an imposed one-element FEM fold, with hard budgets and rollback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))
pytest.importorskip("pyoomph")

import ring_fem_state as S  # noqa: E402

from faraday_singularity import axisrepair as ar, events as ev, surgery  # noqa: E402
from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402


def impose_fold(pb):
    bubble = next(c for c in pb.interface_segments() if not c["open"])
    P = np.column_stack([bubble["r"], bubble["z"]])
    # The bubble's last end is the downward jet's tip; fold its final quadratic element.
    h = float(np.linalg.norm(P[-3] - P[-5]))
    z = float(P[-1, 1])
    targets = {tuple(P[-2]): (-0.02 * h, z + 0.5 * h), tuple(P[-1]): (0.0, z + h)}
    for n in pb.get_mesh("liquid/interface").nodes():
        key = (n.x(0), n.x(1))
        if key in targets:
            for level in range(n.ntstorage()):
                for k, value in enumerate(targets[key]):
                    n.set_x_at_t(level, k, value)
    pb.invalidate_cached_mesh_data()
    bubble = next(c for c in pb.interface_segments() if not c["open"])
    P = np.column_stack([bubble["r"], bubble["z"]])
    fold = ar.axis_fold(P, "last")
    assert fold is not None and fold["V_sliver"] < ar.AXIS_REPAIR_BOUND * S.DELTA**3
    return fold


def bubble_values(pb):
    c = next(c for c in pb.interface_segments() if not c["open"])
    return np.column_stack([c[k] for k in ("r", "z", "u_r", "u_z")])


@pytest.mark.parametrize("failure", [None, "momentum", "coverage", "transfer", "topology"])
def test_a_fold_is_repaired_with_checked_transfer_and_independent_budgets(
        tmp_path, monkeypatch, failure):
    with S.pierce_problem(tmp_path / "pb") as pb:
        pb.initialise()
        S.impose(pb)
        record = S.gas_cut_record(pb)
        surgery.restore_cut(pb, record)
        noise = S.rebuild_noise(pb, S.Z0 + S.GAP, 4 * S.DELTA, rebuilds=1)
        fold = impose_fold(pb)
        block = ev.EventSettings.from_dict({
            "groove": {"enabled": True, **{"sigma_" + k: 2 * v for k, v in noise.items()}},
            "jetcore": {"enabled": True, "far_kinks": "recentre"}})
        driver = CollapseDriver(pb, CollapseSettings(t_end=1, dt_max=1e-5), tmp_path / "run",
                                reconnect=S.RECON, events=block)
        driver.reset_after_cut(record)
        m = driver.measure()
        before = bubble_values(pb)
        driver._stepped_rc = pb.remesh_count  # a stepped-on input stays so after rollback
        if failure in ("momentum", "coverage"):
            original = ar.sliver_quadrature

            def measured(*args, **kwargs):
                result = original(*args, **kwargs)
                if failure == "momentum":
                    U = np.max(np.hypot(*before[fold["removed"] + [fold["kept"]], 2:].T))
                    result["Pz"] = 2 * ar.AXIS_REPAIR_BOUND * S.DELTA**3 * U
                else:
                    result["found"] -= 1
                return result

            monkeypatch.setattr(ar, "sliver_quadrature", measured)
        if failure == "transfer":
            original = surgery.transfer_warnings

            def reported(text):
                result = original(text)
                result["unlocated"]["liquid"] = [10000, 10000]
                return result

            monkeypatch.setattr(surgery, "transfer_warnings", reported)
        if failure == "topology":
            original = surgery._named_chains
            count = pb.remesh_count

            def renamed(problem):
                named, segs = original(problem)
                if problem.remesh_count != count:
                    named = [("unexpected" if name == "bubble" else name, chain)
                             for name, chain in named]
                return named, segs

            monkeypatch.setattr(surgery, "_named_chains", renamed)
        guard = {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "bubble",
                 "index": fold["kept"] + 1, "turn": 100, "L": fold["h"]}
        _, stop = driver._axis_repair(m, guard)
        driver.close()
        rec = json.loads((tmp_path / "run/events/axis_repair_001.json").read_text())
        if failure:
            unclosed = failure in ("momentum", "coverage")
            assert stop["reason"] == ("axis-repair-unclosed" if unclosed
                                      else "axis-repair-failed"), rec
            np.testing.assert_allclose(bubble_values(pb), before, rtol=1e-14, atol=1e-14)
            assert not driver._fresh_mesh()
            if failure == "momentum":
                assert "A_P" in rec["closures"]["hard"]
                assert "A_V" not in rec["closures"]["hard"]
            elif failure == "coverage":
                assert "A_P_coverage" in rec["closures"]["hard"]
            else:
                assert rec["transfer"]["problems"]
        else:
            assert stop is None or stop.get("kind") == "piercing", rec
            assert rec["outcome"] == "made" and rec["closures"]["hard"] == []
            assert rec["transfer"]["problems"] == []
            quad = rec["budgets"]["sliver_quadrature"]
            assert rec["closures"]["A_P"] == pytest.approx(abs(quad["Pz"]))
            assert quad["checked"] and quad["found"] == quad["points"]
            assert quad["V_quad_over_V_sliver"] == pytest.approx(1, rel=1e-10)
            for key in ("A_V", "A_P"):
                assert rec["closures"][key] <= rec["closures"]["limits"][key]
