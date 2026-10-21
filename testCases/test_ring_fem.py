"""Finite-element tests of the ring of a pierced bubble: a closed loop of the free surface's own
boundary, on ``FaradayProblem``.

* F0, the pyoomph facts the ring stage relies on: a problem with equations on a boundary of its own
  for the ring refuses a mesh without it (so the ring lives on ``interface``); the private names
  the chart and the transfer use exist; pyoomph's own arclength chart still refuses a loop beside an
  open segment (so the project's chart, ``problem.RingTolerantArclength``, is still needed).
* F1, a ring meshed as a hole: the synthetic necked cavity of ``test_reconnect_fem.py`` after its
  gas cut, with an elliptic ring added in the liquid through the template's planned geometry: the
  liquid volume falls by the ring's volume of revolution (Pappus), the loop reads back closed,
  ordered and seamed on a vertex, the chains are the open chain and the bubble; three steps, an
  ordinary rebuild, three steps; a dump restarted in two new processes (the problem made as the
  runner makes it, no hook before ``load_state``, the stage put back after it,
  ``ringsurgery.restore_ring``) steps the same in both to the bit, and as the process that wrote it
  to within what a reload changes (the mesh read back from its file); the ring removed, three steps.

Each problem compiles in seconds; the module takes a few minutes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_reconnect_fem as fem  # noqa: E402

from faraday_singularity import ring, ringsurgery, surgery  # noqa: E402
from faraday_singularity.mesh import RingSizing  # noqa: E402
from faraday_singularity.problem import RingTolerantArclength  # noqa: E402
from faraday_singularity.reconnect import ReconnectSettings  # noqa: E402

C, A, B = (1.2, 0.4), 0.08, 0.05  # the ring of F1: an ellipse in the liquid
DT = 1e-5


def ellipse(n=96):
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([C[0] + A * np.cos(t), C[1] + B * np.sin(t)])


def impulsive(pb):
    pb.assign_initial_values_impulsive(DT)
    pb.timestepper.set_num_unsteady_steps_done(0)


def state_vector(pb) -> np.ndarray:
    """Every degree of freedom and the positions of the interface's nodes, chains and loop."""
    dofs = np.asarray(pb.get_current_dofs()[0], dtype=float)
    parts = [dofs] + [np.concatenate([c["r"], c["z"]]) for c in pb.interface_segments()]
    parts += [np.concatenate([x["r"], x["z"]]) for x in pb.interface_loops()]
    return np.concatenate(parts)


RESTART = """
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1] + "/src")
sys.path.insert(0, sys.argv[1] + "/testCases")
import test_reconnect_fem as fem
import test_ring_fem as t
from faraday_singularity import ringsurgery, surgery
from faraday_singularity.reconnect import ReconnectSettings

dump, out = Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads(dump.with_suffix(".json").read_text())
with fem.neck_problem(out, ReconnectSettings(**fem.FAST_CUT)) as pb:
    pb.initialise()
    pb.load_state(str(dump))
    surgery.restore_cut(pb, meta["cut"])
    ringsurgery.restore_ring(pb, meta["ring"])
    for k in range(2):
        fem._step(pb, t.DT)
    v = t.state_vector(pb)
    np.save(out / "state.npy", v)
    print("LOOPS", len(pb.interface_loops()), pb.active_mesh_template().ring_stage)
"""


def _restart(dump: Path, out: Path) -> np.ndarray:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", RESTART, str(ROOT), str(dump), str(out)],
                          capture_output=True, text=True, timeout=900, env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "LOOPS 1 True" in proc.stdout
    return np.load(out / "state.npy")


@pytest.fixture(scope="module")
def ringed(tmp_path_factory):
    out = tmp_path_factory.mktemp("ring")
    res = {"out": out}
    s = ReconnectSettings(**fem.FAST_CUT)
    with fem.neck_problem(out / "pb", s) as pb:
        pb.initialise()
        fem.impose_closing_flow(pb)
        surgery.cut(pb, s, out / "cut", log=lambda *_: None)
        # F0: pyoomph's own chart, on the cut's two open-chart chains, then with the ring
        res["volume_before"] = pb.cut_observables()["volume"]
        tmpl = pb.active_mesh_template()
        named, _ = surgery._named_chains(pb)
        tmpl.ring_stage = True
        tmpl.ring_sizing = RingSizing(h_ring=0.01, grading=0.3)
        tmpl.pending_ring_geometry = {"chains": [c for _, c in named], "loops": [ellipse()]}
        pb.force_remesh({tmpl})
        pb._domains_to_remesh.clear()
        impulsive(pb)
        res["volume_ring"] = pb.cut_observables()["volume"]
        res["segs"] = [(c["open"], tuple(c["end_types"])) for c in pb.interface_segments()]
        res["loops"] = pb.interface_loops()
        res["pending_after"] = tmpl.pending_ring_geometry
        # pyoomph's stock chart on the ring's interface: refused
        from pyoomph.meshes.meshdatacache import MeshDataCacheEntry, MeshDataCacheKey
        from pyoomph.meshes.zeta import AssignZetaCoordinatesByArclength

        iface = pb.get_mesh("liquid/interface")
        cache = MeshDataCacheEntry(iface, MeshDataCacheKey(nondimensional=True,
                                                           tesselate_tri=True))
        try:
            AssignZetaCoordinatesByArclength._compute_zetas(pb.interface_zeta, iface, cache, True)
            res["stock"] = None
        except RuntimeError as e:
            res["stock"] = str(e)
        _, zetas = pb.interface_zeta._compute_zetas(iface, cache, True)
        res["zetas"] = zetas
        for _ in range(3):
            fem._step(pb, DT)
        pb.force_remesh({tmpl})  # an ordinary rebuild in the ring stage
        res["rebuilt_loops"] = pb.interface_loops()
        for _ in range(3):
            fem._step(pb, DT)
        dump = out / "ring.dump"
        pb.save_state(str(dump))
        record = {"sizing": {"h_ring": 0.01, "grading": 0.3}, "removed": None}
        dump.with_suffix(".json").write_text(json.dumps({"cut": pb.cut, "ring": record}))
        res["dump"] = dump
        for _ in range(2):
            fem._step(pb, DT)
        res["continued"] = state_vector(pb)
        # the removal
        v = pb.cut_observables()["volume"]
        tmpl.pending_ring_geometry = {"chains": [c for _, c in surgery._named_chains(pb)[0]],
                                      "loops": []}
        pb.force_remesh({tmpl})
        pb._domains_to_remesh.clear()
        impulsive(pb)
        res["removed_dV"] = pb.cut_observables()["volume"] - v
        res["loops_after_removal"] = pb.interface_loops()
        for _ in range(3):
            fem._step(pb, DT)
        res["segs_after_removal"] = len(pb.interface_segments())
        res["t_end"] = pb.get_current_time(dimensional=False, as_float=True)
    return res


def test_pyoomph_refuses_a_boundary_of_the_rings_own_and_has_the_names_the_stage_uses(tmp_path):
    from pyoomph.equations.generic import DirichletBC
    from pyoomph.meshes import interpolator, meshdatacache, zeta

    class OwnBoundary(fem.NeckProblem):
        def define_problem(self):
            super().define_problem()
            self.add_equations((DirichletBC(velocity_x=0) @ "ring") @ "liquid")

    nd = fem.nondimensionalise(fem.CaseParameters.from_dict(fem.PHYSICS))
    pb = OwnBoundary(nd, mesh=fem.MeshSettings(**fem.MESH))
    pb.set_output_directory(str(tmp_path / "own"))
    with pytest.raises(Exception, match="ring"):
        pb.initialise()
    assert callable(zeta._find_closed_segments)
    src = inspect.getsource(meshdatacache.MeshDataCacheEntry)
    assert "interface_lines_segs" in src and "interface_lines_segs_ninter" in src
    src = inspect.getsource(interpolator)
    for name in ("try_to_use_zeta_on_boundary", "bulk_locate_boundaries",
                 "boundary_max_distances"):
        assert name in src, name
    assert issubclass(RingTolerantArclength, zeta.AssignZetaCoordinatesByArclength)


def test_the_stock_chart_refuses_the_ring_and_the_projects_charts_it_beyond_the_chains(ringed):
    assert ringed["stock"] is not None and "closed loop" in ringed["stock"]
    z = np.array([x for _, x in ringed["zetas"]])
    assert np.all(np.isfinite(z)) and len(z) == len(set(i for i, _ in ringed["zetas"]))


def test_a_ring_is_meshed_as_a_hole_and_read_back_as_a_loop(ringed):
    dV = ringed["volume_ring"] - ringed["volume_before"]
    pappus = 2 * math.pi * C[0] * math.pi * A * B
    assert dV == pytest.approx(-pappus, rel=1e-4)
    assert sorted(ringed["segs"]) == [(False, ("axis", "axis")), (True, ("axis", "fixed"))]
    assert ringed["pending_after"] is None  # consumed by the build
    for loops in (ringed["loops"], ringed["rebuilt_loops"]):
        assert len(loops) == 1
        L = np.column_stack([loops[0]["r"], loops[0]["z"]])
        ring.validate_loop_order(L)
        assert ring.signed_area(L) > 0.0 and len(L) % 2 == 0 and loops[0]["vertex"][0]
        assert np.all(loops[0]["vertex"][::2]) and not np.any(loops[0]["vertex"][1::2])
        assert L[0, 0] == pytest.approx(C[0] + A, rel=1e-4)  # its seam, the outermost vertex
        assert ring.loop_edges(L)["edges"] is not None
        assert rc_volume(L) == pytest.approx(pappus, rel=1e-3)


def rc_volume(L):
    from faraday_singularity.reconnect import revolved_volume

    return revolved_volume(L)


def test_a_dump_of_the_ring_stage_restarts_in_new_processes(ringed, tmp_path):
    a = _restart(ringed["dump"], tmp_path / "a")
    b = _restart(ringed["dump"], tmp_path / "b")
    assert a.shape == b.shape and np.array_equal(a, b)
    c = ringed["continued"]
    assert c.shape == a.shape
    assert np.max(np.abs(a - c)) <= 1e-10 * np.max(np.abs(c))
    assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()


def test_the_ring_is_removed_and_the_run_steps_on(ringed):
    pappus = 2 * math.pi * C[0] * math.pi * A * B
    assert ringed["removed_dV"] == pytest.approx(pappus, rel=2e-3)
    assert ringed["loops_after_removal"] == [] and ringed["segs_after_removal"] == 2
    assert ringed["t_end"] > 0.0


def test_restore_ring_needs_its_sizing():
    from types import SimpleNamespace

    template = SimpleNamespace(ring_stage=False, ring_sizing=None, pending_ring_geometry=1)
    pb = SimpleNamespace(active_mesh_template=lambda: template, ring=None)
    with pytest.raises(RuntimeError, match="ring sizing"):
        ringsurgery.restore_ring(pb, {"sizing": {}})
    ringsurgery.restore_ring(pb, {"sizing": {"h_ring": 1e-5, "grading": 0.3}})
    assert template.ring_stage and template.ring_sizing.h_ring == 1e-5
    assert template.pending_ring_geometry is None and pb.ring["sizing"]["h_ring"] == 1e-5
