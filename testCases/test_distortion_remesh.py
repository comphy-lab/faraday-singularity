"""Contract tests for the rebuilds that element distortion asks for (pyoomph's ``RemeshWhen``).

Before the driver has set a focus, pyoomph makes the rebuild itself, inside the solve, as at commit
148340e; once a focus is set, it only flags the distortion and the driver rebuilds after the step,
at a refreshed focus, and logs it (``driver.CollapseDriver._distortion_remesh``). The stand-in
problem below checks the bookkeeping; the finite-element runs check, against 148340e in fresh
processes with MKL on one thread, that a run without a distortion rebuild in the refined phase is
unchanged to the byte and that one with such a rebuild gets one rebuild, the driver's, at the step
pyoomph asked for it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity.driver import CollapseDriver, CollapseSettings  # noqa: E402
from faraday_singularity.mesh import Focus  # noqa: E402

DEPTH = 2 * math.pi
KAPPA = 6.25  # apex curvature of the stand-in trough: ell = 0.16, a target focus of 0.02
FOCUS = Focus(size=0.02, radius=0.5, grading=0.2)
REFERENCE = "148340e"  # the commit whose distortion rebuilds were all pyoomph's own


# --- the bookkeeping, on a stand-in problem -------------------------------------------------------


class DistortedSurface:
    """The parts of ``FaradayProblem`` the driver uses: a trough at rest whose nodes near the axis
    are spaced ``h/2`` (``h_near = h``), ``h`` the focus size of the last rebuild compressed by
    ``squeeze`` at every solve. The solves in ``flag_at`` are flagged by element distortion,
    ``substeps`` times each (sub-steps of one driver step): pyoomph rebuilds at once while
    ``do_call_remeshing_when_necessary`` is on, and otherwise leaves the flags to the driver."""

    def __init__(self, focus=None, flag_at=(4,), substeps=1, squeeze=0.97, fail_at=None):
        self.t = 0.0
        self.solves = 0
        self.remesh_count = 0
        self.mesh_settings = SimpleNamespace(h_axis=0.05)
        self.nd = SimpleNamespace(radius=3.8317, depth=DEPTH)
        self.template = SimpleNamespace(focus=focus)
        self.h_mesh = focus.size if focus is not None else self.mesh_settings.h_axis
        self.flag_at, self.substeps, self.squeeze, self.fail_at = flag_at, substeps, squeeze, fail_at
        self.do_call_remeshing_when_necessary = True
        self.modes: list[bool] = []  # the flag at each solve
        self.flags: list[dict] = []
        self.pyoomph_rebuilds = 0

    def get_current_time(self, dimensional=False, as_float=True):
        return self.t

    def run(self, t_end, **_kw):
        self.solves += 1
        self.modes.append(self.do_call_remeshing_when_necessary)
        self.h_mesh *= self.squeeze
        if self.solves in self.flag_at:
            for k in range(self.substeps):
                if self.do_call_remeshing_when_necessary:
                    self.pyoomph_rebuilds += 1
                    self.remesh_count += 1
                    self.h_mesh = self.mesh_settings.h_axis if self.template.focus is None \
                        else self.template.focus.size
                else:
                    self.flags.append({"trigger": "shrinkage", "ratio": 0.59 - 0.01 * k})
        if self.fail_at is not None and self.solves >= self.fail_at:
            self.t = t_end
            raise RuntimeError("OomphException")
        self.t = t_end

    def take_distortion_flags(self):
        flags, self.flags = self.flags, []
        return flags

    def force_remesh(self):
        self.remesh_count += 1  # FaradayProblem.actions_after_remeshing
        self.h_mesh = self.template.focus.size

    def active_mesh_template(self):
        return self.template

    def interface_arrays(self):
        inner = 0.5 * self.h_mesh * np.arange(math.ceil(1.0 / self.h_mesh) + 1)
        r = np.concatenate([inner, np.linspace(inner[-1], self.nd.radius, 100)[1:]])
        z = DEPTH - 0.1 + 0.5 * KAPPA * r**2
        return r, z, np.zeros_like(r), np.zeros_like(r)

    def axial_velocity_at(self, points):
        return [0.0] * len(points), True

    def observables(self):
        return {"volume": 1.0, "area": 1.0, "kinetic_energy": 0.0, "dissipation": 0.0}

    def oomph_log_tail(self):
        from test_bookkeeping import MIN_DT_LOG

        return MIN_DT_LOG

    def save_state(self, path, quiet=True):
        Path(path).write_text("state")


def _h_near(h: float) -> float:
    """``h_near`` of the stand-in trough with nodes ``h/2`` apart: twice its first arc edge."""
    return 2.0 * math.hypot(h / 2, 0.5 * KAPPA * (h / 2) ** 2)


def _drive(tmp_path, pb, **kw):
    s = CollapseSettings(t_end=kw.pop("t_end", 1.0), dt_max=kw.pop("dt_max", 1e-3),
                         max_steps=kw.pop("max_steps", 8), state_dt=1.0, **kw)
    drv = CollapseDriver(pb, s, tmp_path)
    try:
        result = drv.run(1e-9)
    finally:
        drv.close()
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    with (tmp_path / "collapse.txt").open() as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(tmp_path / "collapse.txt", ndmin=2)
    return drv, result, events, {name: data[:, i] for i, name in enumerate(header)}


@pytest.mark.parametrize("substeps", [1, 2])
def test_a_refined_phase_flag_is_one_driver_rebuild_at_a_refreshed_focus(tmp_path, substeps):
    # the focus is set: pyoomph only flags (do_call_remeshing_when_necessary off at every solve);
    # the driver rebuilds once, whatever the number of flagged sub-steps, before the step's row
    pb = DistortedSurface(focus=FOCUS, substeps=substeps)
    drv, result, events, rows = _drive(tmp_path, pb)
    assert pb.modes == [False] * pb.solves and pb.pyoomph_rebuilds == 0
    (e,) = [x for x in events if x["event"] == "remesh"]
    assert e["reason"] == "distortion" and e["step"] == 4 and e["flagged_substeps"] == substeps
    assert e["trigger"] == "shrinkage" and e["ratio"] == 0.59  # the first flagged sub-step's
    # compressed below the focus: the rebuild does not coarsen it
    assert e["h_near_before"] == pytest.approx(_h_near(FOCUS.size * 0.97**4), rel=1e-12)
    assert e["focus_size_before"] == FOCUS.size and e["focus_radius_before"] == FOCUS.radius
    assert e["focus_size"] == e["h_near_before"]  # min(focus size, h_near)
    assert e["focus_radius"] == pytest.approx(20.0 / KAPPA, rel=1e-9)  # that of the scale now
    focus = pb.active_mesh_template().focus
    assert (focus.size, focus.radius, focus.grading) == (e["focus_size"], e["focus_radius"],
                                                         FOCUS.grading)
    assert e["h_near_after"] == pytest.approx(_h_near(e["focus_size"]), rel=1e-12)
    # the row of the step is measured on the rebuilt mesh (collapse.txt holds 12 digits)
    k = int(np.flatnonzero(rows["step"] == 4)[0])
    assert rows["h_near"][k] == pytest.approx(e["h_near_after"], rel=1e-11)
    assert rows["remesh_count"][k] == 1
    assert rows["remesh_count"][k - 1] == 0
    assert result["distortion_remeshes"] == 1 and result["pyoomph_remeshes"] == 0
    assert result["remesh_count"] == pb.remesh_count == 1
    # no hold started or ended, and the rule of the driver may still rebuild in the same step
    st = drv.state
    assert st.coarsen_held_until == -math.inf and st.refine_blocked_until == -10**9
    assert st.last_remesh_step == -10**9


def test_the_refreshed_focus_size_is_the_smaller_of_the_focus_and_h_near(tmp_path):
    # a mesh near the axis finer than the focus builds it is kept as fine; a coarser one does not
    # coarsen the focus
    focus = Focus(size=0.025, radius=0.5, grading=0.2)
    pb = DistortedSurface(focus=focus, squeeze=1.0)
    pb.h_mesh = 0.021
    _drive(tmp_path, pb)
    (e,) = [x for x in _events(tmp_path) if x["event"] == "remesh"]
    assert e["focus_size"] == e["h_near_before"] == pytest.approx(_h_near(0.021), rel=1e-12)
    assert e["focus_size"] < focus.size
    pb = DistortedSurface(focus=focus, squeeze=1.0)
    pb.h_mesh = 0.024
    (tmp_path / "b").mkdir()
    _, _, events, _ = _drive(tmp_path / "b", pb)
    (e,) = [x for x in events if x["event"] == "remesh"]
    assert e["focus_size"] == pytest.approx(_h_near(0.024), rel=1e-12)
    assert e["focus_size"] < focus.size
    # an h_near above the focus size leaves the size as it was
    pb = DistortedSurface(focus=focus, squeeze=1.0)
    pb.h_mesh = 0.026
    (tmp_path / "c").mkdir()
    _, _, events, _ = _drive(tmp_path / "c", pb)
    (e,) = [x for x in events if x["event"] == "remesh"]
    assert e["focus_size"] == focus.size and e["h_near_before"] > focus.size


def test_before_any_focus_pyoomph_rebuilds_inside_the_solve(tmp_path):
    # no focus yet: pyoomph's own rebuild, as at 148340e: no event, counted in the result
    pb = DistortedSurface(focus=None, flag_at=(2, 5), squeeze=1.0)
    _, result, events, rows = _drive(tmp_path, pb, refine_ratio=10.0)  # the rule never refines
    assert pb.modes == [True] * pb.solves and pb.pyoomph_rebuilds == 2
    assert not [x for x in events if x["event"] == "remesh"]
    assert result["pyoomph_remeshes"] == 2 and result["distortion_remeshes"] == 0
    assert list(rows["remesh_count"]) == [0, 0, 1, 1, 1, 2, 2, 2, 2]


def test_the_mode_follows_the_focus_from_step_to_step(tmp_path):
    # the driver's first refinement sets the focus: from the next step on, its rebuilds
    pb = DistortedSurface(focus=None, flag_at=(1, 6), squeeze=1.0)
    _, result, events, _ = _drive(tmp_path, pb, refine_ratio=0.5)
    first = next(x for x in events if x["event"] == "remesh")
    assert "reason" not in first and first["step"] == 1  # the driver's rule
    assert pb.modes[0] and not any(pb.modes[1:])
    assert pb.pyoomph_rebuilds == 1 and result["pyoomph_remeshes"] == 1
    distortion = [x for x in events if x.get("reason") == "distortion"]
    assert [x["step"] for x in distortion] == [6] and result["distortion_remeshes"] == 1


def test_a_newton_failure_after_a_flagged_substep_records_the_flag(tmp_path):
    # the flagged sub-step is accepted and a later one fails at every step size: the run ends
    # cleanly (a scale below ell_guard), and the stop says that the failing mesh was flagged
    pb = DistortedSurface(focus=FOCUS, flag_at=(3,), fail_at=3)
    _, result, events, _ = _drive(tmp_path, pb, ell_guard=1.0)
    stop = result["stop"]
    assert stop["reason"] == "singularity-approach" and stop["trigger"] == "newton-failure"
    assert stop["distortion_flagged_substeps"] == 1
    assert not [x for x in events if x["event"] == "remesh"]
    assert result["distortion_remeshes"] == 0


# --- the finite-element problem ---------------------------------------------------------------


DISTORTION_RUN = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.mesh import Focus, MeshSettings
from faraday_singularity.parameters import CaseParameters, nondimensionalise
from faraday_singularity.problem import FaradayProblem

out, physics, focus = Path(sys.argv[2]), json.loads(sys.argv[3]), json.loads(sys.argv[4])
nd = nondimensionalise(CaseParameters.from_dict(physics))
mesh = MeshSettings(h_interface=0.05, h_axis=0.05, h_max=0.4, grading=0.3, interface_points=201)
# the scale stays above the cap (no refinement); element distortion asks for a rebuild at step 53
settings = CollapseSettings(t_end=20.0, dt_max=0.1, max_steps=60, state_dt=0.3, state_keep=100,
                            snapshot_dt=0.5)
pb = FaradayProblem(nd, mesh=mesh)
pb.set_output_directory(str(out))
with pb:
    pb.initialise()
    if focus is not None:  # as if the driver had refined before the first step
        pb.active_mesh_template().focus = Focus(**focus)
    driver = CollapseDriver(pb, settings, out)
    pb.start_diagnostics(out / "diagnostics.txt")
    result = driver.run(1e-8)
    driver.close()
    pb.close_diagnostics()
    f = pb.active_mesh_template().focus
    result["final_focus"] = None if f is None else [f.size, f.radius, f.grading]
    result["pending_request"] = len(pb._domains_to_remesh)
(out / "result.json").write_text(json.dumps(result))
"""

PHYSICS = {"fluid": {"Oh": 0.05, "Bo": 0.0}, "container": {"mode": 1, "depth": 2.0},
           "forcing": {"Gamma": 0.3, "detuning": 1.0, "phase": 0.0},
           "initial": {"amplitude": 0.5}}
NEW_KEYS = ("distortion_remeshes", "pyoomph_remeshes")
INVOKED = re.compile(r"Remeshing invoked from liquid by an element (expanded|shrunken|lost quality)"
                     r" by a factor of (\S+)")


def _reference_src(tmp_path: Path) -> Path:
    try:
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", REFERENCE, "src"],
                                 capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {REFERENCE} is not available")
    ref = tmp_path / f"code-{REFERENCE}"
    ref.mkdir()
    subprocess.run(["tar", "-x", "-C", str(ref)], input=archive, check=True)
    return ref / "src"


def _run_pair(tmp_path: Path, focus) -> tuple[dict, dict]:
    """The small case under the reference and under this commit, each in a fresh process with MKL
    on one thread: the run directory and the remesh messages of pyoomph per run."""
    pytest.importorskip("pyoomph")
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    runs = {}
    for name, src in (("ref", _reference_src(tmp_path)), ("head", ROOT / "src")):
        out = tmp_path / name
        proc = subprocess.run([sys.executable, "-c", DISTORTION_RUN, str(src), str(out),
                               json.dumps(PHYSICS), json.dumps(focus)], capture_output=True,
                              text=True, timeout=900, cwd=tmp_path, env=env, check=False)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        runs[name] = {"out": out, "invoked": INVOKED.findall(proc.stdout),
                      "result": json.loads((out / "result.json").read_text())}
    return runs["ref"], runs["head"]


def _without_wall(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    wall = lines[0][2:].split("\t").index("wall")
    return [lines[0]] + ["\t".join(x for k, x in enumerate(line.split("\t")) if k != wall)
                         for line in lines[1:]]


def _events(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]


def test_a_run_without_a_refined_phase_distortion_rebuild_is_as_at_148340e(tmp_path):
    """Before any focus, pyoomph's own distortion rebuild (here one, at step 53) is made inside
    the solve as at 148340e, so every output is the same to the byte: rows (but the wall time),
    events, diagnostics, profiles, dumps and meshes; the result only gains its two counts."""
    ref, head = _run_pair(tmp_path, None)
    a, b = ref["out"], head["out"]
    names = {f.relative_to(a).as_posix() for f in a.rglob("*") if f.is_file()
             and (not f.relative_to(a).parts[0].startswith("_") or f.suffix == ".msh")}
    assert names == {f.relative_to(b).as_posix() for f in b.rglob("*") if f.is_file()
                     and (not f.relative_to(b).parts[0].startswith("_") or f.suffix == ".msh")}
    result = ref["result"]
    assert result["remesh_count"] >= 1 and result["final_focus"] is None  # no refinement
    assert not [e for e in _events(a) if e["event"].startswith("remesh")]
    rebuilt = np.loadtxt(a / "collapse.txt", ndmin=2)
    assert any(name.startswith("states/") and name.endswith(".dump") for name in names)
    assert rebuilt.shape[0] == 61 and len(ref["invoked"]) == result["remesh_count"]
    assert head["invoked"] == ref["invoked"]
    other = head["result"]
    assert other["pyoomph_remeshes"] == result["remesh_count"]
    assert other["distortion_remeshes"] == 0 and other["pending_request"] == 0
    for k in ("wall",) + NEW_KEYS:
        other.pop(k, None)
    assert json.dumps(other, sort_keys=True) == json.dumps(result, sort_keys=True)
    assert _without_wall(a / "collapse.txt") == _without_wall(b / "collapse.txt")
    for name in sorted(names - {"collapse.txt", "result.json"}):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_a_refined_phase_distortion_flag_is_rebuilt_once_by_the_driver(tmp_path):
    """With a focus set, the flag pyoomph raised at step 53 under 148340e (and rebuilt inside the
    solve, unlogged, at that focus) is now one rebuild by the driver: at the same step, before the
    step's row, at the refreshed focus, logged with pyoomph's trigger and ratio; one mesh is written
    for it, so pyoomph made none of its own, and the request is cleared."""
    focus = {"size": 0.05, "radius": 1.0, "grading": 0.3}
    ref, head = _run_pair(tmp_path, focus)
    a, b = ref["out"], head["out"]
    assert not [e for e in _events(a) if e["event"].startswith("remesh")]
    rows_a = np.loadtxt(a / "collapse.txt", ndmin=2)
    cols = (a / "collapse.txt").read_text().splitlines()[0][2:].split("\t")
    count = cols.index("remesh_count")
    (k,) = np.flatnonzero(np.diff(rows_a[:, count]) > 0) + 1  # pyoomph's rebuild, at row k
    step = int(rows_a[k, cols.index("step")])
    assert ref["result"]["remesh_count"] == 1 and len(ref["invoked"]) == 1
    (e,) = [x for x in _events(b) if x["event"].startswith("remesh")]
    kind, factor = ref["invoked"][0]
    assert head["invoked"] == ref["invoked"]  # pyoomph still detects it, at the same solve
    assert e["event"] == "remesh" and e["reason"] == "distortion" and e["step"] == step
    assert e["trigger"] == {"expanded": "expansion", "shrunken": "shrinkage",
                            "lost quality": "quality"}[kind]
    assert e["ratio"] == float(factor) and e["flagged_substeps"] == 1
    assert e["size_ratio_min"] <= e["ratio"] <= e["size_ratio_max"]
    assert e["focus_size_before"] == focus["size"] and e["focus_radius_before"] == focus["radius"]
    assert e["focus_size"] == min(focus["size"], e["h_near_before"])
    assert head["result"]["final_focus"] == [e["focus_size"], e["focus_radius"], focus["grading"]]
    result = head["result"]
    assert result["distortion_remeshes"] == 1 and result["pyoomph_remeshes"] == 0
    assert result["remesh_count"] == 1 and result["pending_request"] == 0
    assert sorted(p.name for p in (b / "_gmsh").glob("*_REMESH_*.msh")) == [
        "FaradayCylinderMesh_REMESH_000000.msh"]
    # the same trajectory up to the flagged step; that step's row is measured on the new mesh
    lines_a, lines_b = _without_wall(a / "collapse.txt"), _without_wall(b / "collapse.txt")
    assert lines_a[:k + 1] == lines_b[:k + 1]  # the header and the rows before the step
    rows_b = np.loadtxt(b / "collapse.txt", ndmin=2)
    assert rows_b[k, cols.index("h_near")] == pytest.approx(e["h_near_after"], rel=1e-11)
    assert rows_b[k, count] == 1 and rows_b[k - 1, count] == 0
    assert rows_b[k, cols.index("time")] == rows_a[k, cols.index("time")]
    assert rows_b[k, cols.index("time")] == pytest.approx(e["time"], rel=1e-11)


def test_after_a_cut_every_distortion_flag_is_the_drivers_rebuild(tmp_path):
    """After a cut the focus is the cut's: with thresholds that flag every checked step, each is
    rebuilt once by the driver, with the post-cut times in its event, the cut's sizing kept and the
    focus refreshed, one region per feature: the tips the cut made at the cut's focus size, which
    a rebuild for distortion keeps; pyoomph rebuilds nothing of its own. pyoomph does not check the
    first solve on a new mesh (its count of solves since the rebuild is still 0 then), so the steps
    right after a rebuild are not flagged: the cut's projection counts as that solve, the step after
    the driver's first rebuild does not."""
    pytest.importorskip("pyoomph")
    from pyoomph.equations.generic import RemeshingOptions

    from faraday_singularity import surgery
    from faraday_singularity.mesh import MeshSettings
    from faraday_singularity.parameters import CaseParameters, nondimensionalise
    from faraday_singularity.reconnect import ReconnectSettings
    from test_reconnect_fem import FAST_CUT, MESH, NeckProblem, impose_closing_flow
    from test_reconnect_fem import PHYSICS as NECK_PHYSICS

    class FlaggingNeckProblem(NeckProblem):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.remeshing_options = RemeshingOptions(max_expansion=1 + 1e-9,
                                                      min_expansion=1 - 1e-9)

    s = ReconnectSettings(**FAST_CUT)
    nd = nondimensionalise(CaseParameters.from_dict(NECK_PHYSICS))
    settings = CollapseSettings(t_end=10.0, dt_max=0.01, max_steps=3)
    pb = FlaggingNeckProblem(nd, mesh=MeshSettings(**MESH), reconnect=s)
    pb.set_output_directory(str(tmp_path))
    with pb:
        pb.initialise()
        impose_closing_flow(pb)
        driver = CollapseDriver(pb, settings, tmp_path, reconnect=s)
        rec = surgery.cut(pb, s, tmp_path, driver=driver, log=lambda *_: None)
        rebuilds = pb.remesh_count
        result = driver.run(1e-8)
        driver.close()
        template = pb.active_mesh_template()
        focus, sizing = template.focus, template.cut_sizing
        chains = len(pb.interface_segments())
        pending = len(pb._domains_to_remesh)
    events = _events(tmp_path)
    distortion = [e for e in events if e.get("reason") == "distortion"]
    assert result["stop"]["reason"] == "max-steps" and result["steps"] == 3
    assert [e["step"] for e in distortion] == [1, 3]
    assert result["distortion_remeshes"] == 2 and result["pyoomph_remeshes"] == 0
    remesh = [e for e in events if e["event"] == "remesh"]
    assert pb.remesh_count == rebuilds + len(remesh) and pending == 0
    t_cut = rec["t_cut"]
    h_cut = s.focus_size * rec["rmin"]
    for e in distortion:
        assert e["tau"] == pytest.approx(e["time"] - t_cut, abs=1e-15) and "tau_p" in e
        sizes = {g["name"]: g["new_size"] for g in e["regions"]}
        assert sizes["cavity_tip"] == sizes["bubble_top"] == pytest.approx(h_cut)
        assert e["focus_size"] == min(sizes.values())
    assert distortion[0]["focus_size_before"] == rec["focus"]["cut"]["size"] == h_cut
    last = remesh[-1]
    if last.get("reason") == "distortion":
        assert (focus.size, focus.radius) == (last["focus_size"], last["focus_radius"])
    assert sizing is not None and sizing.h_tip == rec["sizing"]["h_tip"] and chains == 2
