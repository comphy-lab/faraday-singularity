"""The runs after a cut without the events block (``collapse.reconnect_events``) on stand-in
problems whose two chains are prescribed (``test_bookkeeping.CutProblem``) are those of commit
1b65916, to the byte: the source of this commit and that of 1b65916 each drive the same stand-ins in
a fresh process, and write the same files.

The stand-ins go through the branches the block's work touched, which the finite-element run of
``test_events_fem.py`` (the synthetic necked cavity: no neck after its cut, no jet at the bubble's
bottom, a run to its step limit) does not reach: the bubble's bottom with a jet and the landing's
kept focus, the necks of both chains and their regions, the open chain's neck stop and its floor,
the groove at a jet's base, the step-size floor, and the stops after the cut (a second pinch-off on
the liquid side after a step, an annular contact, a pinch-off of the open chain, the bubble's tips
meeting, the dt floor), with the regions of the features measured right after the cut and the dumps
of a run.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import test_events_driver  # noqa: E402
import test_events_stops  # noqa: E402
from test_events_fem import _reference_src, _without_wall  # noqa: E402

STANDINS = """
import json, math, sys
from pathlib import Path
src, tests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, src)
import faraday_singularity  # the source under test, before the tests put theirs on the path
assert Path(faraday_singularity.__file__).resolve().is_relative_to(Path(src).resolve())
from faraday_singularity import driver as _driver, reconnect as rc
sys.path.insert(1, tests)
import numpy as np
from test_bookkeeping import CUT_GRADING, CutProblem, _settings
from test_reconnect_geometry import _post_cut_segs
from faraday_singularity.driver import CollapseDriver
from faraday_singularity.reconnect import Chain, ReconnectSettings, cut_geometry
assert all(Path(m.__file__).resolve().is_relative_to(Path(src).resolve())
           for m in (_driver, rc))
Z_APEX, R_C = %(Z_APEX)r, %(R_C)r
%(geometry)s


def run(name, pb, recon, focus=None, **kw):
    drv = CollapseDriver(pb, _settings(**{"t_end": 1.0, "dt_max": 1.0, "max_steps": 3, **kw}),
                         out / name, reconnect=recon)
    chains = [Chain(np.column_stack([c["r"], c["z"]]), c["end_types"])
              for c in pb.interface_segments()]
    geom = cut_geometry(chains, 3.0, gas_before=1.0)
    pb.cut = {"delta": recon.delta, "rmin": recon.delta, "r_neck": 0.95 * recon.delta,
              "z_neck": 4.685, "t_cut": 0.0, "pinch": {"t_p": -1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": recon.tip_size * recon.delta,
                         "h_bridge": recon.bridge_size * recon.delta,
                         "curvature": recon.curvature_sizing, "grading": CUT_GRADING},
              "settings": json.loads(json.dumps(recon.to_dict())),
              **({"focus": focus} if focus is not None else {})}
    drv.reset_after_cut(pb.cut)
    regions = drv.measure().get("regions", [])
    result = drv.run(1e-9)
    drv.close()
    for what, x in (("regions", regions), ("result", result)):
        (out / name / f"{what}.json").write_text(json.dumps(x, default=float, sort_keys=True))


class Prescribed(CutProblem):
    def __init__(self, segs, **kw):
        super().__init__(**kw)
        self.segs = segs

    def interface_segments(self):
        return self.segs(self.t)


class Groove(HeadProblem):
    def interface_segments(self):
        if self._cache is None:
            chain, _ = bubble_with_head(1e-3, r_n=0.42, r_w=0.8)
            self._cache = (0.42, chain)
        return super().interface_segments()


class RisingJet(CutProblem):
    def interface_segments(self):
        segs = super().interface_segments()
        P = cavity_with_head(1e-3, r_n=0.3).points
        segs[1] = self._chain("open", P[:, 0].copy(), P[:, 1].copy(), ("axis", "fixed"))
        return segs


d = ReconnectSettings(delta=1e-3, dt0=1e-6)
# the bubble's bottom rising into a jet, the landing's focus kept there, the top a hanging jet
bottom = _post_cut_segs(bottom=(2e-3, 5e-3))
z_bb, z_bt = float(bottom[0]["z"][0]), float(bottom[0]["z"][-1])
kept = rc.kept_region({"size": 4e-5, "radius": 0.02, "grading": 0.2}, 6.25e-5, z_bb, z_bt)
run("bottom", Prescribed(lambda t: _post_cut_segs(bottom=(2e-3 * (1.0 + 1e4 * t), 5e-3))), d,
    focus={"kept": kept}, state_dt=0.0, state_keep=10)
# a jet hanging into the bubble, its neck 0.3 delta_geometry, then 0.09: a second pinch-off on
# the liquid side after a step (the cut radius below the neck, the fold stop below its groove)
run("head", HeadProblem(), ReconnectSettings(delta=2e-4, dt0=1e-6), fold_stop=1e-5)
run("groove", Groove(r_n=lambda t: 0.42), d, fold_stop=1e-3)
run("rising", RisingJet(), d)
run("rising-floor", RisingJet(), d, neck_stop=1e-5, h_floor=2e-4)
run("floor", CutProblem(), d, dt_min=1.0)
run("tips", CutProblem(a=1e-3, b=4e-4), d)
run("plain", CutProblem(), ReconnectSettings(delta=1e-3, dt0=1e-3), state_dt=0.0, state_keep=10)
"""


def _script() -> str:
    geometry = "\n\n".join(inspect.getsource(f) for f in (
        test_events_stops._hermite, test_events_stops.head_feature, test_events_stops._nodes,
        test_events_stops.bubble_with_head, test_events_stops.cavity_with_head,
        test_events_driver.HeadProblem))
    return STANDINS % {"Z_APEX": test_events_stops.Z_APEX, "R_C": test_events_stops.R_C,
                       "geometry": geometry}


def _same_outputs(a: Path, b: Path) -> list[str]:
    """Every file of two run directories the same, each stand-in's in its own directory: to the
    byte, but the wall-time column of each ``collapse.txt`` and the wall times of the JSON
    records (``test_events_fem._same_outputs``, for runs in subdirectories)."""
    names = sorted(f.relative_to(a).as_posix() for f in a.rglob("*") if f.is_file())
    assert names == sorted(f.relative_to(b).as_posix() for f in b.rglob("*") if f.is_file())
    for name in names:
        x, y = a / name, b / name
        if Path(name).name == "collapse.txt":
            hx, *rx = x.read_text().splitlines()
            hy, *ry = y.read_text().splitlines()
            wall = hx[2:].split("\t").index("wall")
            assert hx == hy and len(rx) == len(ry), name
            for u, v in zip(rx, ry):
                cu, cv = u.split("\t"), v.split("\t")
                assert cu[:wall] + cu[wall + 1:] == cv[:wall] + cv[wall + 1:], name
        elif name.endswith(".json"):
            assert _without_wall(json.loads(x.read_text())) == _without_wall(
                json.loads(y.read_text())), name
        else:
            assert x.read_bytes() == y.read_bytes(), name
    return names


def _run(src: Path, cwd: Path) -> Path:
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", _script(), str(src), str(ROOT / "testCases"),
                           "run"], capture_output=True, text=True, timeout=600, cwd=cwd, env=env,
                          check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return cwd / "run"


def test_runs_after_a_cut_on_stand_ins_without_the_block_are_as_at_1b65916(tmp_path):
    """Without the block the stand-ins write what the source of 1b65916 writes, to the byte (the
    wall times aside): their regions after the cut, rows, chains, events, profiles, dumps and the
    stops they end on."""
    ref = _reference_src(tmp_path)
    runs = {name: _run(src, tmp_path / name) for name, src in (("ref", ref),
                                                               ("head", ROOT / "src"))}
    names = _same_outputs(runs["ref"], runs["head"])
    stops = {name: json.loads((runs["head"] / name / "result.json").read_text())["stop"]
             for name in ("bottom", "head", "groove", "rising", "rising-floor", "floor", "tips",
                          "plain")}
    assert stops["head"]["reason"] == "secondary-pinch-approach"
    assert stops["head"]["neck_side"] == "liquid"
    assert stops["groove"]["reason"] == "annular-contact-approach"
    assert stops["rising"]["reason"] == stops["rising-floor"]["reason"] == "pinch-off-approach"
    assert stops["rising-floor"]["neck_limit"] == pytest.approx(6e-4, rel=1e-12, abs=0.0)
    assert stops["floor"]["reason"] == "dt-floor"
    assert stops["tips"]["reason"] == "bubble-jets-approach"
    assert stops["plain"]["reason"] == stops["bottom"]["reason"] == "max-steps"
    result = json.loads((runs["head"] / "head" / "result.json").read_text())
    assert result["steps"] == 1  # the neck closed after a step
    regions = {name: {g["name"]: g for g in json.loads(
        (runs["head"] / name / "regions.json").read_text())} for name in stops}
    assert regions["bottom"]["bubble_bottom"]["kept"] == 4e-5  # the landing's focus, kept
    assert "kink_bubble" in regions["bottom"]
    assert "neck_bubble" in regions["head"] and "neck_bubble" in regions["groove"]
    assert "jet_base_bubble_top" in regions["groove"]
    assert {"neck_open", "jet_base_cavity_tip"} <= set(regions["rising"])
    assert any(n.startswith("bottom/states/state_") for n in names)
    assert any(n.startswith("plain/states/state_") for n in names)
