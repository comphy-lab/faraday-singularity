"""The fill of a closing gas groove against 89f7f49, the commit before it: the source of this commit
and that of 89f7f49, each in a fresh process with MKL on one thread.

Without the part ``groove`` every run is 89f7f49's to the byte:

* the stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery,
  the block's settings, their hash and the cut's keys;
* the finite-element run after the cut of the synthetic necked cavity with the block
  (``test_events_fem.py``: rebuilds by the rule at every step it may, a dump at every step) and a
  restart in a new process from one of its dumps: every output and every mesh gmsh wrote;
* the finite-element run of the synthetic pierced state with ``ring.enabled``
  (``test_ring_identity.py``): the ring cut made by the driver and the steps of the ring stage.

With ``groove.enabled`` and no groove met, the same finite-element runs have the same solution to
the byte: only the records hold the part (the block's settings and its hash).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_events_fem as fem_pins  # noqa: E402
from test_events_fem import _reference_src  # noqa: E402
from test_events_gasneck import BATTERY, _run  # noqa: E402
from test_jetcore_identity import _run_fem, _spec  # noqa: E402
from test_ring_identity import RING_RUN, _drop_wall_column  # noqa: E402

BASE = "89f7f49"  # the commit before the groove fill


def test_the_stops_are_89f7f49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def test_a_finite_element_run_with_the_block_and_its_restart_are_89f7f49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    spec = _spec()
    runs = {name: _run_fem(src, tmp_path / name, spec) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["remesh_count"] >= 2, result
    assert "budget.txt" in names
    again = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        r = dict(spec, restart=str(tmp_path / name / "run" / "states" / "state_00000003.dump"))
        again[name] = _run_fem(src, tmp_path / f"{name}-restart", r)
    fem_pins._same_outputs(again["ref"], again["head"])


def test_a_ring_run_is_89f7f49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = fem_pins._same_outputs(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 4, result["stop"]
    assert result["events"]["made"] == 1 and result["events"]["ring"]["cut"] == 1
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)


# --- with groove enabled and no groove met: the same solution ------------------------------------


def _without_groove(x):
    """A record without what the part ``groove`` adds to it by design: its settings (``groove``
    wherever the block is written) and the hash of the block."""
    if isinstance(x, dict):
        return {k: _without_groove(v) for k, v in x.items()
                if k not in ("groove", "settings_sha256", "wall_seconds")}
    if isinstance(x, list):
        return [_without_groove(v) for v in x]
    return x


def _same_solution(ref: Path, head: Path) -> list[str]:
    """``test_events_fem._same_outputs`` for a run of ``head`` with ``groove.enabled`` against one
    of ``ref`` without it: every file the same, the JSON records and the lines of
    ``events.jsonl`` but for :func:`_without_groove`, the meshes gmsh wrote by name and hash."""
    assert fem_pins._gmsh_files(ref) == fem_pins._gmsh_files(head)

    def files(run):
        return sorted(f.relative_to(run).as_posix() for f in run.rglob("*") if f.is_file()
                      and not f.relative_to(run).parts[0].startswith("_"))

    names = files(ref)
    assert files(head) == names
    for name in names:
        x, y = ref / name, head / name
        if name == "collapse.txt":
            hx, *rx = x.read_text().splitlines()
            hy, *ry = y.read_text().splitlines()
            wall = hx[2:].split("\t").index("wall")
            assert hx == hy and len(rx) == len(ry), name
            for u, v in zip(rx, ry):
                cu, cv = u.split("\t"), v.split("\t")
                assert cu[:wall] + cu[wall + 1:] == cv[:wall] + cv[wall + 1:], name
        elif name.endswith(".json"):
            assert _without_groove(json.loads(x.read_text())) == _without_groove(
                json.loads(y.read_text())), name
        elif name == "events.jsonl":
            a = [_without_groove(json.loads(u)) for u in x.read_text().splitlines()]
            b = [_without_groove(json.loads(u)) for u in y.read_text().splitlines()]
            assert a == b, name
        else:
            assert x.read_bytes() == y.read_bytes(), name
    return names


def test_with_groove_enabled_and_no_groove_the_finite_element_run_is_89f7f49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {"ref": _run_fem(ref, tmp_path / "ref", _spec()),
            "head": _run_fem(ROOT / "src", tmp_path / "head", _spec(groove={"enabled": True}))}
    _same_solution(runs["ref"], runs["head"])
    result = json.loads((runs["head"] / "result.json").read_text())
    assert result["steps"] == 6 and result["events"]["made"] == 0
    meta = json.loads(next((runs["head"] / "states").glob("state_*.json")).read_text())
    assert meta["events_settings"]["groove"]["enabled"] is True


RING_RUN_GROOVE = RING_RUN.replace(
    '''block = ev.EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6,
                                             "sigma_KE": 1e-6}})''',
    '''block = ev.EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6,
                                             "sigma_KE": 1e-6},
                                    **({"groove": {"enabled": True}}
                                       if hasattr(ev, "GrooveEventSettings") else {})})''')
assert RING_RUN_GROOVE != RING_RUN


def test_with_groove_enabled_a_ring_run_is_89f7f49s(tmp_path):
    ref = _reference_src(tmp_path, BASE)
    runs = {name: _run(RING_RUN_GROOVE, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    for run in runs.values():
        _drop_wall_column(run / "events.txt")
    names = _same_solution(runs["ref"], runs["head"])
    assert {"ring.txt", "events.txt", "budget.txt", "events/event_001.json"} <= set(names)
