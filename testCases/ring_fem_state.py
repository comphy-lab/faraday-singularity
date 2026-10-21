"""The synthetic pierced state of the finite-element tests of the ring (``test_ring_fem.py``): the
state a gas cut leaves once the jet its bridge drives down into the bubble is about to reach the
bubble's bottom. Not a test module (no ``test_`` prefix): ``test_ring_fem.py`` imports it, and its
scripts in new processes do.

The bubble (``test_ring_geometry.pierced_bubble``, at ``DELTA``): a flat plug at ``Z0`` out to a
rounded wall of radius ``4 DELTA``, a roof ``4 DELTA`` above, and a jet hanging from the roof, a
neck of ``0.6 a`` above a head of radius ``a = DELTA/6`` whose tip lies ``GAP`` above the plug; the
open chain above it a smooth trough from its tip on the axis out to a flat surface at the wall. The
first mesh is built on both chains, as a mesh after a cut, with the bubble's top sized in a region
of its own (``bubble_top``, finer than the cut's focus size, so that the ring cut freezes a copy of
it); a synthetic record of the gas cut, the bubble's volume at it ``V_B_FACTOR`` times the
bubble's. The velocity is imposed: the solenoidal flow of :func:`closing_flow` (the jet down at
``W_JET``, the plug up at ``W_PLUG``), or a radial inflow besides, which is not solenoidal. The
noise of a box in the ordinary rebuilds of the state (:func:`rebuild_noise`) sets the floors of
the ring's closures, as ``post.log_box_noise`` would for a run.

The state of F4 (:func:`head_chains`, :class:`PierceHeadProblem`) has the same bubble below an open
chain whose tip is the head of a jet rising from the cavity's trough, necked and landed as in
``test_liquid_cut_fem.py``, with the flow about that neck besides (``impose(..., z_neck=)``): the
liquid cut of the cavity's tip in the ring stage.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

import test_reconnect_fem as fem  # noqa: E402
from test_ring_geometry import pierced_bubble  # noqa: E402

from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import surgery  # noqa: E402
from faraday_singularity.mesh import (FaradayCylinderMesh, Focus, FocusRegion,  # noqa: E402
                                      MeshSettings)
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.problem import FaradayProblem  # noqa: E402

R_C = 3.8317059702075125
DELTA = 0.05  # the cut radius of the synthetic gas cut
Z0 = 0.55  # the plug
GAP = 0.8 * DELTA  # the jet's tip above the plug
A_JET = DELTA / 6  # the radius of the jet's head
H_FINE = A_JET / 12  # the spacing of the nodes about the jet and the plug
V_B_FACTOR = 30.0
W_JET, W_PLUG = -1.0, 0.1
RECON = rc.ReconnectSettings(delta=DELTA, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                             focus_size=0.25)
MESH = {"h_interface": 0.01, "h_axis": 0.01, "h_max": 0.25, "grading": 0.3,
        "interface_points": 401}


def chains():
    """The bubble and the open chain of the state."""
    bubble, _ = pierced_bubble(GAP, a=A_JET, R0=4 * DELTA, H=4 * DELTA, h=H_FINE, h_far=0.004,
                               z0=Z0)
    r = np.linspace(0.0, 0.6, 600)
    z = 0.95 + 0.15 * np.sin(0.5 * math.pi * r / 0.6) ** 2
    s = np.linspace(0.0, 1.0, 300)[1:]
    wall = np.column_stack([0.6 + (R_C - 0.6) * s, np.full(len(s), 1.1)])
    D = np.vstack([np.column_stack([r, z]), wall])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(D, axis=0).T))])
    u, x = [0.0], 0.0
    while True:  # graded as a mesh is: the spline through the nodes loops where spacing jumps
        k = min(int(np.searchsorted(arc, x)), len(D) - 1)
        x += min(0.02, 0.004 + 0.05 * max(0.0, D[k, 0] - 0.1))
        if x > arc[-1] - 0.01:
            break
        u.append(x)
    u = np.array(u + [arc[-1]])
    if len(u) % 2 == 0:
        u = np.sort(np.append(u, 0.5 * (u[-1] + u[-2])))
    O = np.column_stack([np.interp(u, arc, D[:, 0]), np.interp(u, arc, D[:, 1])])
    O[0, 0], O[-1] = 0.0, (R_C, 1.1)
    return bubble, rc.Chain(O, ("axis", "fixed"))


def focus() -> Focus:
    regions = (FocusRegion("bubble_top", size=H_FINE, r=0.0, z=Z0 + GAP, radius=4 * A_JET),
               FocusRegion("plug", size=H_FINE, r=0.0, z=Z0, radius=3 * A_JET),
               FocusRegion("cavity_tip", size=0.004, r=0.0, z=0.95, radius=0.1))
    return Focus(size=H_FINE, radius=0.1, grading=0.3, regions=regions)


class PierceMesh(FaradayCylinderMesh):
    """The first mesh is built on the chains of :func:`chains`, as a mesh after a cut."""

    def first_chains(self):
        return chains()

    def first_focus(self) -> Focus:
        return focus()

    def define_geometry(self):
        if not self.is_first_time():
            return super().define_geometry()
        bubble, cavity = self.first_chains()
        f = self.first_focus()
        s = self.settings
        self.mesh_mode = "tris"
        self.set_gmsh_parameter("General.NumThreads", 1)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", min(s.h_interface, f.size))
        self.set_gmsh_parameter("Mesh.MeshSizeMax", s.h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)
        self.default_resolution = s.h_max
        return self._define_chain_geometry([bubble, cavity], f, f.size)


class PierceProblem(FaradayProblem):
    mesh_class = PierceMesh


def pierce_problem(outdir) -> PierceProblem:
    nd = nondimensionalise(CaseParameters.from_dict(fem.PHYSICS))
    pb = PierceProblem(nd, mesh=MeshSettings(**MESH), reconnect=RECON)
    pb.set_output_directory(str(outdir))
    return pb


# --- the state of F4: the cavity's tip necked above the pierced bubble ----------------------------

R_NECK_OPEN = 0.09  # the open chain's neck in units of DELTA under a head of radius DELTA/2
EPS_NECK = 8.0  # elements per neck radius at the neck


def head_chains():
    """The bubble of :func:`chains` and, above it, the open chain of ``test_liquid_cut_fem.
    jet_chains``: a jet rising from the cavity's trough (at 0.9) whose head, of radius
    ``DELTA/2``, is necked to ``R_NECK_OPEN DELTA`` (``test_events_stops.head_feature``), landed at
    ``liquid.eps_d`` 0.2; and the neck's height."""
    import test_liquid_cut_fem as lcf

    bubble, _ = chains()
    _, cavity, (_, z_neck) = lcf.jet_chains(DELTA, R_NECK_OPEN)
    return bubble, cavity, z_neck


def head_focus() -> Focus:
    """The focus of :func:`focus` with the cavity's tip meshed as ``test_liquid_cut_fem`` meshes
    its jets: the neck at ``r_n / EPS_NECK`` within four neck radii, the head at ``DELTA / (2
    EPS_NECK)``; and the root of the bubble's downward jet, its thread of radius ``0.6 A_JET``
    filleted into the roof (``test_ring_geometry.pierced_bubble``), at ``H_FINE`` (``neck_bubble``,
    which the ring cut freezes): at the interface's own size there the elements are twice the
    thread's radius, and the curved ones of the ring cut's mesh may turn inside out."""
    r_n, z_neck = R_NECK_OPEN * DELTA, head_chains()[2]
    regions = (FocusRegion("bubble_top", size=H_FINE, r=0.0, z=Z0 + GAP, radius=4 * A_JET),
               FocusRegion("neck_bubble", size=H_FINE, r=0.6 * A_JET, z=Z0 + 4 * DELTA,
                           radius=4 * A_JET),
               FocusRegion("plug", size=H_FINE, r=0.0, z=Z0, radius=3 * A_JET),
               FocusRegion("neck_open", size=r_n / EPS_NECK, r=r_n, z=z_neck, radius=4 * r_n),
               FocusRegion("cavity_tip", size=0.5 * DELTA / EPS_NECK, r=0.0, z=z_neck,
                           radius=0.1))
    return Focus(size=min(g.size for g in regions), radius=0.1, grading=0.3, regions=regions)


class PierceHeadMesh(PierceMesh):
    """The first mesh is built on the chains of :func:`head_chains`, as a mesh after a cut."""

    def first_chains(self):
        bubble, cavity, _ = head_chains()
        return bubble, cavity

    def first_focus(self) -> Focus:
        return head_focus()


class PierceHeadProblem(FaradayProblem):
    mesh_class = PierceHeadMesh


def pierce_head_problem(outdir) -> PierceHeadProblem:
    nd = nondimensionalise(CaseParameters.from_dict(fem.PHYSICS))
    pb = PierceHeadProblem(nd, mesh=MeshSettings(**MESH), reconnect=RECON)
    pb.set_output_directory(str(outdir))
    return pb


def gas_cut_record(pb, factor: float = V_B_FACTOR) -> dict:
    """A record of a gas cut at ``DELTA`` made just before the current state, its bubble's volume
    ``factor`` times the bubble's now, with the cut's sizing put on the template and the focus of
    the state on it (the focus the problem's first mesh was built under)."""
    named, _ = surgery._named_chains(pb)
    geom = rc.cut_geometry([c for _, c in named], RECON.reservoir_depth, gas_before=1.0)
    geom.bubble_volume = factor * geom.bubble_volume
    t = pb.get_current_time(dimensional=False, as_float=True)
    record = {"delta": DELTA, "rmin": DELTA, "r_neck": DELTA / 1.05, "z_neck": 0.85, "t_cut": t,
              "pinch": {"t_p": t - 1e-6}, "geometry": geom.to_dict(),
              "sizing": {"h_tip": RECON.tip_size * DELTA, "h_bridge": RECON.bridge_size * DELTA,
                         "curvature": 0, "grading": 0.3},
              "settings": json.loads(json.dumps(RECON.to_dict())), "focus": {"kept": None}}
    surgery.restore_cut(pb, record)
    pb.active_mesh_template().focus = pb.active_mesh_template().first_focus()
    return record


def closing_flow(r, z, *, solenoidal: bool = True):
    """The flow from the stream function ``psi = r^2/2 f(z) g(r)``, ``f`` from ``W_PLUG`` below the
    gap to ``W_JET`` above it within a window of ``8 DELTA``, ``g = exp(-(r/(2 DELTA))^2)``:
    ``u_z = f (g + r g'/2)``, ``u_r = -r f' g / 2``, solenoidal; without ``solenoidal`` a radial
    inflow besides."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    zm, w, L, a = Z0 + 0.5 * GAP, 0.2 * GAP, 4 * DELTA, 2 * DELTA
    t = np.tanh((z - zm) / w)
    W = 0.5 * (W_JET + W_PLUG) + 0.5 * (W_JET - W_PLUG) * t
    dW = 0.5 * (W_JET - W_PLUG) * (1.0 - t * t) / w
    H = np.exp(-((z - zm) / L) ** 2)
    dH = -2.0 * (z - zm) / L**2 * H
    f, df = W * H, dW * H + W * dH
    g = np.exp(-(r / a) ** 2)
    dg = -2.0 * r / a**2 * g
    uz = f * (g + 0.5 * r * dg)
    ur = -0.5 * r * df * g
    if not solenoidal:
        ur = ur - 5.0 * r * g * H
    return ur, uz


def impose(pb, *, solenoidal: bool = True, z_neck: float | None = None):
    """The flow of :func:`closing_flow`; with ``z_neck`` (the state of F4) also the solenoidal flow
    of ``test_liquid_cut_fem.stream_velocity`` about the open chain's neck at that height, which
    closes it and carries the head away from its stump."""
    m = pb.get_mesh("liquid")
    fi = m.get_nodal_field_indices()
    nodes = list(m.nodes())
    r, z = np.array([n.x(0) for n in nodes]), np.array([n.x(1) for n in nodes])
    ur, uz = closing_flow(r, z, solenoidal=solenoidal)
    if z_neck is not None:
        import test_liquid_cut_fem as lcf

        a, b = lcf.stream_velocity(r, z, [z_neck])
        ur, uz = ur + a, uz + b
    for n, a, b in zip(nodes, ur, uz):
        n.set_value(fi["velocity_x"], float(a))
        n.set_value(fi["velocity_y"], float(b))
    pb.reapply_boundary_conditions()
    pb.assign_initial_values_impulsive()
    pb.invalidate_cached_mesh_data()


def rebuild_noise(pb, z: float, R: float, rebuilds: int = 3) -> dict:
    """The noise of the box of radius ``R`` about ``(0, z)`` in the ordinary rebuilds of the current
    state (``post.log_box_noise``'s reading: the liquid volume of the polylines, the axial momentum
    and the kinetic energy on the grid): the largest change one of ``rebuilds`` successive rebuilds
    makes, the problem left rebuilt."""
    def budgets():
        named, _ = surgery._named_chains(pb)
        poly = surgery.box_polyline_budgets([c for _, c in named], z, R, RECON.reservoir_depth)
        grid = surgery.box_integrals(pb, surgery.box_grid(z, R))
        return {"V": poly["V_liquid"], "Pz": grid["Pz"], "KE": grid["KE"]}

    out = {"V": 0.0, "Pz": 0.0, "KE": 0.0}
    before = budgets()
    for _ in range(rebuilds):
        pb.force_remesh({pb.active_mesh_template()})
        pb._domains_to_remesh.clear()
        pb.invalidate_cached_mesh_data()
        after = budgets()
        out = {k: max(out[k], abs(after[k] - before[k])) for k in out}
        before = after
    return out
