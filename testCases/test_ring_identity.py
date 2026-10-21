"""The ring part of the events block against c4620f6, the commit before it: the source of this
commit and that of c4620f6, each in a fresh process with MKL on one thread.

* The stops of the events branch on every synthetic state of ``test_events_gasneck.py``'s battery,
  with the ring off: the same requests, to the byte and key order; the block's settings, their hash
  and the cut's keys the same.
* (a) The stand-in run of a bubble whose downward jet pierces it after two steps
  (``test_bookkeeping.CutProblem``), with the block and the ring absent: every file the same (the
  wall times aside), the piercing a resumable stop with ``precut_e001``.
* (b) The same with ``ring.enabled``: up to the request the same, the rows of the steps before it,
  the dump of the state before the event and its record, but for the block it records (which holds
  the enabled part, by design) and the request's fields, a superset of c4620f6's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_events_fem import _reference_src, _without_wall  # noqa: E402
from test_events_gasneck import BATTERY, PRELUDE, _run  # noqa: E402
from test_events_legacy import _same_outputs  # noqa: E402

RING_REFERENCE = "c4620f6"  # the commit before the ring part

PIERCING = PRELUDE + """
from test_bookkeeping import CUT_GRADING, CutProblem, _settings
from faraday_singularity.driver import CollapseDriver
from faraday_singularity.reconnect import Chain, ReconnectSettings, cut_geometry


class Piercing(CutProblem):
    \"\"\"``CutProblem`` whose bubble shortens from 1.2e-3 to 8e-4 (delta 1e-3) after its first step
    (which ends at t = 8.9e-7): the downward jet piercing it.\"\"\"

    def interface_segments(self):
        self.b = 6e-4 if self.t <= 1.2e-6 else 4e-4
        return super().interface_segments()


def run(name, events, **kw):
    pb = Piercing()
    recon = ReconnectSettings(delta=1e-3, dt0=1e-6)
    drv = CollapseDriver(pb, _settings(**{"t_end": 1.0, "dt_max": 1.0, "max_steps": 4, **kw}),
                         out / name, reconnect=recon, events=events)
    chains = [Chain(np.column_stack([c["r"], c["z"]]), c["end_types"])
              for c in pb.interface_segments()]
    geom = cut_geometry(chains, 3.0, gas_before=1.0)
    pb.cut = {"delta": recon.delta, "rmin": recon.delta, "r_neck": 0.95e-3, "z_neck": 4.685,
              "t_cut": 0.0, "pinch": {"t_p": -1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": recon.tip_size * recon.delta,
                         "h_bridge": recon.bridge_size * recon.delta,
                         "curvature": recon.curvature_sizing, "grading": CUT_GRADING},
              "settings": json.loads(json.dumps(recon.to_dict()))}
    drv.reset_after_cut(pb.cut)
    result = drv.run(1e-9)
    drv.close()
    (out / name / "result.json").write_text(json.dumps(result, default=float, sort_keys=True))


run("plain", ev.EventSettings(), state_dt=0.0, state_keep=10)
# with the ring where the source has it; c4620f6 runs the block without it
ring = ({"ring": {"enabled": True}} if hasattr(ev, "RingEventSettings") else {})
run("ring", ev.EventSettings.from_dict(ring), state_dt=0.0, state_keep=10)
"""


def test_the_stops_with_the_ring_off_are_c4620f6s(tmp_path):
    ref = _reference_src(tmp_path, RING_REFERENCE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"] == out["head"]
    assert len(out["head"]) > 40


def _rows(path: Path, before: int) -> list[list[str]]:
    """The rows of a table of the driver of the steps before ``before``, the wall column out."""
    head, *rows = path.read_text().splitlines()
    cols = head[2:].split("\t")
    step = cols.index("step")
    wall = cols.index("wall") if "wall" in cols else None
    out = [head]
    for u in rows:
        c = u.split("\t")
        if int(c[step]) < before:
            out.append("\t".join(c[:wall] + c[wall + 1:]) if wall is not None else u)
    return out


def test_a_piercing_with_the_block_is_c4620f6s_without_the_ring_and_up_to_it_with_it(tmp_path):
    ref = _reference_src(tmp_path, RING_REFERENCE)
    runs = {name: _run(PIERCING, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    # (a) the ring absent: every file the same
    _same_outputs(runs["ref"] / "plain", runs["head"] / "plain")
    old = json.loads((runs["ref"] / "plain" / "result.json").read_text())
    stop = old["stop"]
    assert (stop["reason"], stop["kind"], stop["class"], stop["k"]) == (
        "event-unhandled", "piercing", "W1", 1)
    assert old["steps"] == 2 and Path(stop["precut"]).name == "precut_e001.dump"
    # (b) the ring enabled: the same up to the request
    a, b = runs["ref"] / "ring", runs["head"] / "ring"
    old = json.loads((a / "result.json").read_text())
    new = json.loads((b / "result.json").read_text())
    assert old["stop"]["reason"] == "event-unhandled" and old["steps"] == new["steps"] == 2
    assert (new["stop"]["kind"], new["stop"]["class"], new["stop"]["handler"]) == (
        "piercing", "W1", "ring")
    assert new["stop"]["gap"] == 1.0 and new["stop"]["limit"] == old["stop"]["limit"]
    steps = old["steps"]
    for name in ("collapse.txt", "chains.txt", "budget.txt"):
        assert _rows(a / name, steps) == _rows(b / name, steps), name
    for name in ("precut_e001.dump",):
        assert (a / "states" / name).read_bytes() == (b / "states" / name).read_bytes()
    x = _without_wall(json.loads((a / "states" / "precut_e001.json").read_text()))
    y = _without_wall(json.loads((b / "states" / "precut_e001.json").read_text()))
    assert "ring" not in x["events_settings"] and y["events_settings"]["ring"]["enabled"] is True
    assert {k: v for k, v in y["event"].items() if k in x["event"]} == x["event"]
    for d in (x, y):
        d.pop("events_settings"), d.pop("event")
        for leg in d["legs"]:
            leg.pop("settings_sha256")
    assert x == y
