"""The gas-neck test after the cut with the events block against e330650, the commit whose test read
a gas neck's depth against ``chain_necks``' global shoulder alone: the source of this commit, which
also measures a gas neck against its adjacent shoulders (``reconnect.adjacent_shoulders``), and that
of e330650, each in a fresh process with MKL on one thread.

* The stops of the events branch (``reconnect.post_cut_stops`` with ``events``) on the synthetic
  states of ``test_events_stops.py`` and ``test_reconnect_geometry.py``, with liquid events on and
  off: the same request, to the byte and in the same key order, on every state but those with a
  stationary pair on the inner wall of a rim (``test_events_stops.bubble_with_rim``), where e330650
  raises a ``gas-neck`` event and this commit none; where both raise one, this commit's request is
  e330650's with the four adjacent fields appended. The settings of the block, its hash and the keys
  of the cut's settings are the same, so the admission of a restart cannot have changed.
* Stand-in runs after a cut with the block (``test_bookkeeping.CutProblem``): one that meets no gas
  neck writes the same files (the wall times aside); one whose bubble grows the rim's pair after its
  first step stops under e330650 on a ``gas-neck`` event, resumably, with the dump of the state
  before it, while under this commit it runs to its step limit, the rows, profiles and dumps of the
  steps before e330650's stop the same in both.

The module imports ``test_events_fem`` (the extraction of a reference commit), which needs pyoomph;
it takes under a minute.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_events_fem import _reference_src, _without_wall  # noqa: E402
from test_events_legacy import _same_outputs  # noqa: E402

GAS_REFERENCE = "e330650"  # the commit whose gas test read the global depth alone
ADDED = ["neck_shoulder_left", "neck_shoulder_right", "neck_shoulder_adjacent",
         "neck_depth_adjacent"]

PRELUDE = """
import json, math, sys
from pathlib import Path
src, tests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, src)
import faraday_singularity  # the source under test, before the tests put theirs on the path
assert Path(faraday_singularity.__file__).resolve().is_relative_to(Path(src).resolve())
from faraday_singularity import events as ev, reconnect as rc
sys.path.insert(1, tests)
import numpy as np
import test_events_stops as S
import test_reconnect_geometry as G
assert all(Path(m.__file__).resolve().is_relative_to(Path(src).resolve())
           for m in (ev, rc, S.ev, S.rc, G.rc))
"""

BATTERY = PRELUDE + """
LIQUID, OFF = S.LIQUID, ev.EventSettings()
results = {}


def state(name, chains, cut_chains=None, h_min=None, delta=1e-3, **kw):
    s = rc.ReconnectSettings(delta=delta)
    cut = rc.cut_geometry(cut_chains or chains, 3.0, gas_before=1.0)
    h = s.tip_size * s.delta if h_min is None else h_min
    for label, events in (("liquid", LIQUID), ("off", OFF)):
        try:
            got = rc.post_cut_stops(chains, cut, s, h, events=events, **kw)
        except Exception as exc:  # the same refusal is the same outcome
            got = {"raised": type(exc).__name__, "message": str(exc)}
        results[f"{name}/{label}"] = json.dumps(got, default=float)


cav = S.cavity()
for r_n in (0.3, 0.09, 0.095):
    state(f"head-{r_n}", [S.bubble_with_head(1e-3, r_n=r_n)[0], cav])
state("rising", [S.bubble_with_head(1e-3, r_n=0.095)[0], S.cavity_with_head(1e-3, r_n=0.05)])
for r_n in (0.09, 0.3):
    bubble, apex = S.bubble_with_head(1e-3, r_n=r_n)
    bent = rc.Chain(G._bent(bubble.points, (apex + 200) // 2, turn_deg=130.0), bubble.end_types)
    for fresh in (False, True):
        state(f"bent-{r_n}-{fresh}", [bent, cav], cut_chains=[bubble, cav],
              guard={"fresh": fresh, "h_floor": 1e-3 / 512, "refine_ratio": 1.5})
bubble, gcav = G._bubble(), G._cavity()
state("plain", [bubble, gcav], h_min=1e-5)
state("bridge", [bubble, rc.Chain(gcav.points - [0.0, 0.0095], ("axis", "fixed"))],
      cut_chains=[bubble, gcav], h_min=1e-5)
beside = rc.Chain(np.array([[0.0, 4.69], [0.0103, 4.6905], [0.0102, 4.64], [0.0105, 4.6],
                            [0.02, 4.6], [0.5, 5.0], [S.R_C, 6.0]]), ("axis", "fixed"))
state("beside", [bubble, beside], cut_chains=[bubble, gcav], h_min=1e-4)
small = rc.Chain(np.array([[0.0, 4.6], [1e-3, 4.60025], [0.0, 4.6005]]), ("axis", "axis"))
state("small", [small, gcav], h_min=1e-5)
state("thin", [rc.Chain(bubble.points * [0.05, 1.0], ("axis", "axis")), gcav],
      cut_chains=[bubble, gcav], h_min=1e-5)
state("waist", [G._bubble(waist=0.95), gcav], cut_chains=[bubble, gcav], h_min=1e-5)
jet_cav = G._cavity(z_tip=4.70)
for dent in (0.6, 0.9):
    state(f"jet-{dent}", [G._bubble_with_jet(dent=dent), jet_cav],
          cut_chains=[G._bubble_with_jet(), jet_cav], h_min=1e-6)
for A in (0.0, 3e-6, 1e-4):
    state(f"rim-{A}", [S.bubble_with_rim(1e-3, A=A), cav])
state("rim-small", [S.bubble_with_rim(1e-3), cav], neck_elements=8.0)
results["settings"] = json.dumps(ev.EventSettings().to_dict())
results["settings_liquid"] = json.dumps(LIQUID.to_dict())
results["sha256"] = json.dumps([ev.EventSettings().sha256(), LIQUID.sha256()])
results["reconnect_keys"] = json.dumps(sorted(rc.ReconnectSettings(delta=1e-3).to_dict()))
out.write_text(json.dumps(results))
"""

STANDINS = PRELUDE + """
from test_bookkeeping import CUT_GRADING, CutProblem, _settings
from faraday_singularity.driver import CollapseDriver
from faraday_singularity.reconnect import Chain, ReconnectSettings, cut_geometry


class RimProblem(CutProblem):
    \"\"\"``CutProblem`` whose bubble is ``test_events_stops.bubble_with_rim`` at delta = 1e-3, the
    pair on the rim's inner wall ``A(t)`` delta apart.\"\"\"

    def __init__(self, A, **kw):
        super().__init__(**kw)
        self.A = A
        self._cache = None

    def interface_segments(self):
        A = self.A(self.t)
        if self._cache is None or self._cache[0] != A:
            self._cache = (A, S.bubble_with_rim(1e-3, A=A))
        segs = super().interface_segments()
        P = self._cache[1].points
        segs[0] = self._chain("bubble", P[:, 0].copy(), P[:, 1].copy(), ("axis", "axis"))
        return segs


def run(name, pb, recon, events, **kw):
    \"\"\"``test_events_plumbing._events_driver``, run to its end.\"\"\"
    drv = CollapseDriver(pb, _settings(**{"t_end": 1.0, "dt_max": 1.0, "max_steps": 3, **kw}),
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


block = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                    "post": {"open_neck_stop": False, "neck_floor": 2e-6}})
run("plain", CutProblem(), ReconnectSettings(delta=1e-3, dt0=1e-3), block, state_dt=0.0,
    state_keep=10)
# the pair from the second step on (the first ends at t = 8.9e-7, the second near 2e-6)
run("rim", RimProblem(lambda t: 0.0 if t <= 1.2e-6 else 3e-6),
    ReconnectSettings(delta=1e-3, dt0=1e-6), block, state_dt=0.0, state_keep=10, max_steps=4)
"""


def _run(script: str, src: Path, cwd: Path) -> Path:
    """``script`` in a fresh process from ``cwd`` with the package at ``src``, MKL on one thread;
    its outputs at the relative ``run``, so that the paths they record are the same for every
    source."""
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", script, str(src), str(ROOT / "testCases"), "run"],
                          capture_output=True, text=True, timeout=900, cwd=cwd, env=env,
                          check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return cwd / "run"


def test_the_stops_with_the_block_are_e330650s_but_on_a_stationary_pair_of_a_gas_wall(tmp_path):
    """Every synthetic state gives e330650's request, to the byte and key order, but the rim's
    pair with liquid events on (with them off the jet's neck at the cut's second-pinch test comes
    first), on which e330650 raises a ``gas-neck`` event and this commit none; a gas neck both
    raise (the bubble's waist, the rim's pair narrower than ``neck_elements h_min``) is e330650's
    request with the adjacent fields appended. The block's settings, their hash and the cut's keys
    are e330650's."""
    ref = _reference_src(tmp_path, GAS_REFERENCE)
    out = {name: json.loads(_run(BATTERY, src, tmp_path / name).read_text())
           for name, src in (("ref", ref), ("head", ROOT / "src"))}
    assert out["ref"].keys() == out["head"].keys()
    differ = {k for k in out["ref"] if out["ref"][k] != out["head"][k]}
    gas = {k for k, v in out["head"].items() if '"kind": "gas-neck"' in v}
    assert differ - gas == {"rim-3e-06/liquid", "rim-0.0001/liquid"}, differ
    assert gas == {"waist/liquid", "waist/off", "rim-small/liquid"}, gas
    for k in sorted(differ - gas):
        assert json.loads(out["ref"][k])["kind"] == "gas-neck"
        assert json.loads(out["head"][k]) is None
    for k in sorted(gas):
        old, new = json.loads(out["ref"][k]), json.loads(out["head"][k])
        assert list(new) == list(old) + ADDED
        assert {key: new[key] for key in old} == old
    for k in ("settings", "settings_liquid", "sha256", "reconnect_keys"):
        assert out["ref"][k] == out["head"][k], k


def _step_of(path: Path) -> int | None:
    """The step of the state a profile (its ``# {json}`` header) or a dump (its JSON record,
    ``.json`` beside its ``.dump``) of the driver was written at, None for other files."""
    if path.suffix == ".txt" and path.parent.name == "profiles":
        return int(json.loads(path.read_text().splitlines()[0][2:])["step"])
    if path.parent.name == "states" and path.suffix in (".json", ".dump"):
        return int(json.loads(path.with_suffix(".json").read_text())["step"])
    return None


def test_a_stand_in_run_is_e330650s_up_to_the_gas_neck_the_adjacent_shoulders_reject(tmp_path):
    """Stand-in runs after a cut with liquid events on, in fresh processes under e330650 and this
    commit. ``plain`` meets no neck: every file the same (the wall times aside), to its step
    limit. ``rim``: the bubble's rim grows its pair after the first step; e330650 stops on it after
    the second (``event-unhandled``, a ``gas-neck`` event of class ``G`` at the pair's minimum,
    with ``precut_e001``), this commit runs on to its step limit (4) with no event. Up to
    e330650's stop both are the same: every row of ``collapse.txt`` of a step before its stop step
    (the wall column aside), and every profile (``profiles/``: those of the cut and of the start)
    and dump (``states/``: the rotating dump of the first step) e330650 wrote at a step before it,
    to the byte (the dumps' JSON records without their wall times)."""
    ref = _reference_src(tmp_path, GAS_REFERENCE)
    runs = {name: _run(STANDINS, src, tmp_path / name) for name, src in
            (("ref", ref), ("head", ROOT / "src"))}
    _same_outputs(runs["ref"] / "plain", runs["head"] / "plain")
    plain = json.loads((runs["head"] / "plain" / "result.json").read_text())
    assert plain["stop"]["reason"] == "max-steps"
    old = json.loads((runs["ref"] / "rim" / "result.json").read_text())
    new = json.loads((runs["head"] / "rim" / "result.json").read_text())
    stop = old["stop"]
    assert (stop["reason"], stop["kind"], stop["class"]) == ("event-unhandled", "gas-neck", "G")
    assert stop["chain"] == "bubble" and stop["neck_side"] == "gas" and stop["k"] == 1
    assert stop["neck_depth"] >= 0.25 and "neck_depth_adjacent" not in stop
    assert Path(stop["precut"]).name == "precut_e001.dump"
    assert (runs["ref"] / "rim" / "states" / "precut_e001.json").is_file()
    assert old["steps"] == 2
    assert new["stop"]["reason"] == "max-steps" and new["steps"] == 4
    assert new["events"]["made"] == 0 and new["events"]["stop"] is None
    assert not (runs["head"] / "rim" / "events").exists()
    # identity before the divergence
    a, b = runs["ref"] / "rim", runs["head"] / "rim"
    hx, *rx = (a / "collapse.txt").read_text().splitlines()
    hy, *ry = (b / "collapse.txt").read_text().splitlines()
    assert hx == hy
    cols = hx[2:].split("\t")
    wall, step = cols.index("wall"), cols.index("step")
    before = [u for u in rx if int(u.split("\t")[step]) < old["steps"]]
    assert {int(u.split("\t")[step]) for u in before} == {0, 1} and len(ry) > len(before)
    for u, v in zip(before, ry):
        cu, cv = u.split("\t"), v.split("\t")
        assert cu[:wall] + cu[wall + 1:] == cv[:wall] + cv[wall + 1:]
    compared = []
    for sub in ("profiles", "states"):
        for f in sorted((a / sub).iterdir()):
            k = _step_of(f)
            if k is None or k >= old["steps"]:
                continue
            g = b / sub / f.name
            assert g.is_file(), f.name
            if f.suffix == ".json":
                assert _without_wall(json.loads(f.read_text())) == _without_wall(
                    json.loads(g.read_text())), f.name
            else:
                assert f.read_bytes() == g.read_bytes(), f.name
            compared.append(f"{sub}/{f.name}")
    assert {"states/state_00000001.dump", "states/state_00000001.json"} <= set(compared)
    assert sum(name.startswith("profiles/") for name in compared) >= 4
