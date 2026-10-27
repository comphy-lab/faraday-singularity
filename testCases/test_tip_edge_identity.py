"""Legacy pins against 6ef3c4d, plus the pre-merge stop battery at 4c98e0b."""

from test_events_fem import _reference_src, _same_outputs
from test_events_gasneck import BATTERY, _run
from test_ring_refine_identity import RING_RUN_STAGE
from test_ring_identity import _drop_wall_column
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_stops_without_a_tip_edge_are_6ef3c4ds(tmp_path):
    ref = _reference_src(tmp_path, "6ef3c4d")
    a = json.loads(_run(BATTERY, ref, tmp_path / "a").read_text())
    b = json.loads(_run(BATTERY, ROOT / "src", tmp_path / "b").read_text())
    assert a == b and len(a) > 40


def test_stops_without_a_tip_edge_match_4c98e0b_before_merge_152d39d(tmp_path):
    ref = _reference_src(tmp_path, "4c98e0b")
    a = json.loads(_run(BATTERY, ref, tmp_path / "a").read_text())
    b = json.loads(_run(BATTERY, ROOT / "src", tmp_path / "b").read_text())
    assert a == b and len(a) > 40


def test_a_production_ring_stage_without_a_tip_edge_is_6ef3c4ds(tmp_path):
    ref = _reference_src(tmp_path, "6ef3c4d")
    runs = [_run(RING_RUN_STAGE, s, tmp_path / n) for n, s in [("a", ref), ("b", ROOT / "src")]]
    for r in runs:
        _drop_wall_column(r / "events.txt")
    _same_outputs(*runs)


def test_a_groove_run_and_its_restart_without_a_tip_edge_are_6ef3c4ds(tmp_path):
    from test_jetcore_identity import _run_fem, _spec

    ref = _reference_src(tmp_path, "6ef3c4d")
    spec = _spec(groove={"enabled": True, "sheets": "log"})
    runs = {n: _run_fem(src, tmp_path / n, spec) for n, src in [("a", ref), ("b", ROOT / "src")]}
    _same_outputs(runs["a"], runs["b"])
    restarted = {}
    for n, src in [("a", ref), ("b", ROOT / "src")]:
        again = dict(spec, restart=str(tmp_path / n / "run/states/state_00000003.dump"))
        restarted[n] = _run_fem(src, tmp_path / (n + "-restart"), again)
    _same_outputs(restarted["a"], restarted["b"])


RECENTRED_GUARD = r"""
import sys, json
from pathlib import Path

sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
import ring_fem_state as S
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.events import EventSettings

out = Path(sys.argv[3])
with S.pierce_problem(out) as pb:
    pb.initialise()
    S.impose(pb)
    record = S.gas_cut_record(pb)
    c = next(c for c in pb.interface_segments() if not c["open"])
    P = np.column_stack([c["r"], c["z"]])
    i = (len(P) // 2) | 1
    mid = next(
        n for n in pb.get_mesh("liquid/interface").nodes() if (n.x(0), n.x(1)) == tuple(P[i])
    )
    chord = P[i + 1] - P[i - 1]
    normal = np.array([-chord[1], chord[0]])
    target = 0.5 * (P[i - 1] + P[i + 1]) + 0.26 * normal
    for t in range(mid.ntstorage()):
        for k in range(2):
            mid.set_x_at_t(t, k, float(target[k]))
    pb.invalidate_cached_mesh_data()
    block = EventSettings.from_dict(
        {"groove": {"enabled": True}, "jetcore": {"enabled": True, "far_kinks": "recentre"}}
    )
    drv = CollapseDriver(
        pb,
        CollapseSettings(t_end=1, dt_max=1e-5, max_steps=0),
        out,
        reconnect=S.RECON,
        events=block,
    )
    drv.reset_after_cut(record)
    result = drv.run(1e-8)
    drv.close()
    assert result["stop"]["trigger"] == "edge-turn" and result["stop"]["chain"] == "bubble", result[
        "stop"
    ]
    assert result["stop"]["index"] < len(P) - 6
    (out / "result.json").write_text(json.dumps(result, indent=1))
"""


def test_a_recentred_middle_bubble_guard_is_unchanged_from_6ef3c4d(tmp_path):
    ref = _reference_src(tmp_path, "6ef3c4d")
    runs = [
        _run(RECENTRED_GUARD, src, tmp_path / name)
        for name, src in (("a", ref), ("b", ROOT / "src"))
    ]
    _same_outputs(*runs)
