"""pyoomph problem: an axisymmetric Faraday wave in a vertically vibrated cylinder.

Adapted from the surfactant-free path of faraday-surfactant-fem (planar, vertical-only mesh motion)
to an axisymmetric container whose free surface can deform arbitrarily:

* coordinates ``(r, z) = (x, y)`` with pyoomph's axisymmetric measure ``2 pi r``;
* incompressible Navier-Stokes in the container frame with body force
  ``-(Bo + Gamma cos(Omega t + phase)) e_z`` (``parameters.py`` defines the scales);
* free surface with unit surface tension and the kinematic condition coupled to the mesh;
* axis ``r = 0`` (``AxisymmetryBC``), free-slip bottom ``z = 0``, free-slip side wall ``r = R_c``
  along which the contact line slides with the natural 90 degree angle;
* hyperelastic mesh smoothing on an unstructured triangular mesh that is rebuilt from the live free
  surface whenever elements are too distorted (``RemeshWhen``). pyoomph makes that rebuild inside
  the solve unless ``do_call_remeshing_when_necessary`` is off: then it only flags the distortion,
  which the collapse driver reads after the step (:meth:`FaradayProblem.take_distortion_flags`)
  and rebuilds itself at a focus it refreshes (``driver.py``).

With ``R_c = j_{1,n}`` and a free-slip wall, ``J0(r)`` is an exact mode of the linearised problem,
so the fundamental-mode amplitude is the Hankel coefficient

    a(t) = 2 int_0^Rc zeta J0(r) r dr / (R_c^2 J0(R_c)^2),

evaluated on the interface with ``n_z ds = dr`` and pyoomph's ``2 pi r`` measure.

A problem made with ``reconnect`` settings (a case's ``collapse.reconnect`` block) can be cut through
a closing neck (``surgery.py``). Only then does it carry the reconnection handler on the free surface
(inert until the cut calls it), the pressure-impulse projection as an auxiliary residual, and the
observables the cut reports (axial momentum, ``int (div u)^2`` of the P2 velocity, an order-``h``
discretisation diagnostic rather than a test of the projection, the change made by the projection, the
pressure on the free surface). It adds no field, degree of freedom or global parameter, so state dumps
of either kind of problem load into the other. Without ``reconnect`` the problem is the one above.

The free surface's chart (:class:`RingTolerantArclength`) also charts the ring of a pierced bubble
(``ring.py``, ``ringsurgery.py``), a closed loop of the free surface's own boundary that carries
the same equations, and :meth:`FaradayProblem.interface_loops` reads it apart from the chains,
which :meth:`FaradayProblem.interface_segments` give without it; without a loop the chart is
pyoomph's own, call for call, and nothing else changes.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from pyoomph import Problem
from pyoomph.equations.ALE import HyperelasticSmoothedMesh
from pyoomph.equations.generic import (AxisymmetryBC, DirichletBC, IntegralObservables,
                                       RemeshingOptions, RemeshWhen)
from pyoomph.equations.navier_stokes import NavierStokesEquations, NavierStokesFreeSurface
from pyoomph.expressions import (cos, div, dot, double_dot, evaluate_in_past, grad, transpose, var,
                                 vector)
from pyoomph.meshes.zeta import (AssignZetaCoordinatesByArclength,
                                 AssignZetaCoordinatesByEulerianCoordinate)
from pyoomph.output.generic import IntegralObservableOutput, TextFileOutput
from pyoomph.output.meshio import MeshFileOutput
from scipy.special import j0

from .bessel import j0_polynomial, terms_for_radius
from .mesh import FaradayCylinderMesh, MeshSettings
from .parameters import Dimensionless
from .reconnect import ReconnectSettings, orient_chain

# a midside node of the axis further than this many edge lengths off the middle of its edge has been
# misplaced by the transfer to a rebuilt mesh (FaradayProblem.repair_axis_midsides)
AXIS_MIDSIDE_LIMIT = 0.1
# the integrals of observables() depend on the current values and positions only: they are kept for
# the state they were evaluated on (FaradayProblem._integrals), which the diagnostics of a step and the
# driver's row of it both read
CACHED_INTEGRALS = {"liquid": ("volume", "kinetic_energy", "dissipation"),
                    "liquid/interface": ("area", "amp")}


class RingTolerantArclength(AssignZetaCoordinatesByArclength):
    """The arclength chart of the free surface (``interface``), which also charts a ring.

    The ring of a pierced bubble (``ring.py``, ``ringsurgery.ring_cut``) is a closed loop of the
    free surface's own boundary: pyoomph refuses a problem with equations on a boundary its mesh
    lacks, at ``initialise`` and at ``load_state`` alike, so a boundary of the ring's own could
    never load a state written before the ring, while on ``interface`` the ring carries the free
    surface's own equations and nothing new is compiled. pyoomph's arclength chart refuses a closed
    loop beside other segments of its boundary; this one charts the open segments as pyoomph does
    and each loop beyond their range (``segment_jump_offset`` beyond the largest zeta, by arclength
    counter-clockwise from its vertex node of largest ``r``, normalised to a unit range), with the
    period 0 of an open chart, so that pyoomph's test of an invertible chart holds. The loop's chart
    is never used for a transfer: its seam element runs backwards over the loop's range, so every
    rebuild with a loop on either mesh places the interface nodes by projection onto the old
    interface instead (``try_to_use_zeta_on_boundary`` off, set in the hook before the transfer,
    :meth:`_before_mesh_to_mesh_interpolation`), and while a rebuild of the ring stage that changes
    a topology is under way (``FaradayProblem._ring_transfer``: a ring surgery, or a liquid cut of
    the open chain in the ring stage, ``surgery.liquid_cut``, with the ring or after its removal)
    the interface is placed by that projection too and the axis nodes are located in the old bulk,
    as pyoomph's reconnection does for its axis. Without a loop and without such a rebuild it is
    pyoomph's own chart, call for call: the class only names the equation differently in the
    problem's generated-code tree, and the generated code is the same."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop_seen = False

    def _compute_zetas(self, mesh, cache, whole_interface):
        from pyoomph.meshes.zeta import _find_closed_segments

        segs, ninter = cache.get_interface_line_segments()
        closed = _find_closed_segments(cache, segs)
        if not any(closed) or len(segs) == 1:
            return super()._compute_zetas(mesh, cache, whole_interface)
        opened = [s for s, c in zip(segs, closed) if not c]
        saved = cache.interface_lines_segs, cache.interface_lines_segs_ninter
        cache.interface_lines_segs, cache.interface_lines_segs_ninter = opened, ninter
        try:
            _, out = super()._compute_zetas(mesh, cache, False)
        finally:
            cache.interface_lines_segs, cache.interface_lines_segs_ninter = saved
        pts = cache.get_coordinates()
        vertices = {int(e[0]) for e in cache.elem_indices} | {int(e[-1])
                                                             for e in cache.elem_indices}
        start = (max(z for _, z in out) if out else 0.0) + self.segment_jump_offset
        for seg, c in zip(segs, closed):
            if not c:
                continue
            loop = [int(i) for i in (seg[:-1] if seg[0] == seg[-1] else seg)]
            n = len(loop)
            area = sum(pts[0, loop[k]] * pts[1, loop[(k + 1) % n]]
                       - pts[0, loop[(k + 1) % n]] * pts[1, loop[k]] for k in range(n))
            if area < 0:
                loop = [loop[0]] + loop[1:][::-1]
            cand = [k for k, i in enumerate(loop) if i in vertices] or list(range(n))
            k0 = max(cand, key=lambda k: (float(pts[0, loop[k]]), float(pts[1, loop[k]])))
            loop = loop[k0:] + loop[:k0]
            d = [math.hypot(pts[0, loop[(k + 1) % n]] - pts[0, loop[k]],
                            pts[1, loop[(k + 1) % n]] - pts[1, loop[k]]) for k in range(n)]
            total, acc = sum(d), 0.0
            for k, i in enumerate(loop):
                out.append((i, start + acc / total))
                acc += d[k]
            start += 1.0 + self.segment_jump_offset
        self._loop_seen = True
        return 0.0, out

    def _before_mesh_to_mesh_interpolation(self, eqtree, interpolator):
        self._loop_seen = False
        out = super()._before_mesh_to_mesh_interpolation(eqtree, interpolator)
        try:
            problem = self.get_problem()
        except RuntimeError:  # an equation not bound to a problem
            problem = None
        transfer = getattr(problem, "_ring_transfer", None) if problem is not None else None
        # a loop on the old mesh or the new one, or a rebuild of the ring stage under way that
        # changes the topology of the open chain (a liquid cut there, a loop or none): no transfer
        # through the chart, whose ranges no plan has matched across the change
        if self._loop_seen or transfer:
            interpolator.try_to_use_zeta_on_boundary = False
        if transfer:
            if transfer.get("axis_bulk"):
                interpolator.bulk_locate_boundaries.add("axis")
            reach = float(transfer.get("reach") or 0.0)
            if reach > 0.0:
                interpolator.boundary_max_distances["interface/axis"] = reach
                interpolator.boundary_max_distances["axis/interface"] = reach
        return out


class FaradayProblem(Problem):
    mesh_class = FaradayCylinderMesh

    def __init__(self, nd: Dimensionless, *, mesh: MeshSettings, remesh: bool = True,
                 mesh_output: bool = False, reconnect: ReconnectSettings | None = None):
        super().__init__()
        self.nd = nd
        self.mesh_settings = mesh
        self.remesh = remesh
        self.mesh_output = mesh_output
        self.remesh_count = 0
        self._diag = None  # open per-step diagnostics file, set by start_diagnostics()
        self.mesh_template: FaradayCylinderMesh | None = None
        self.reconnect = reconnect
        self.recon = None  # the reconnection handler on the free surface, with ``reconnect``
        self.interface_zeta: AssignZetaCoordinatesByArclength | None = None
        self.cut: dict | None = None  # the record of a cut, once one has been made
        # the records of the events made after the cut (the events block, events.EventSettings)
        self.events: list[dict] = []
        # the ring of a pierced bubble, once its cut is made (ringsurgery.ring_cut, its record of
        # the stage, dumped with every state of it), and the transfer settings of the rebuild of a
        # ring surgery, or of a liquid cut in the ring stage, while it is under way
        # (RingTolerantArclength); None before the ring
        self.ring: dict | None = None
        self._ring_transfer: dict | None = None
        self._projecting = False  # set while the projection's auxiliary solve runs
        # the thresholds of RemeshWhen, pyoomph's defaults: an element shrunk below 0.6 or grown
        # beyond 1.75 times its size at the build, or of a fifth of its quality then
        self.remeshing_options = RemeshingOptions()
        # what RemeshWhen flagged on the accepted steps since the last take_distortion_flags(),
        # while pyoomph leaves the rebuild to the driver (do_call_remeshing_when_necessary off)
        self._distortion_flags: list[dict] = []
        # after a cut: what repair_axis_midsides() did at the last rebuild, and the nodes it has
        # repaired since the problem was made
        self.axis_repair: dict | None = None
        self.axis_repairs_total = 0
        # each Newton solve makes at least this many iterations (the collapse driver sets it around
        # a step after a cut: reconnect.newton_min_iterations); 1 leaves oomph-lib's test alone
        self.min_newton_iterations = 1
        self._newton_checks = 0
        self._held_tolerance: float | None = None
        # around a step after a cut (the collapse driver sets it): (newton_tolerance, newton_floor),
        # the tolerance of each Newton solve then max(newton_tolerance, newton_floor/dt) with dt the
        # step that solve makes, a sub-step of pyoomph's halving included; None leaves it alone
        self.newton_floor_rule: tuple[float, float] | None = None
        # (fingerprint of the state, {(mesh, name): value}) of the integrals last evaluated
        self._integral_cache: tuple | None = None

    def initial_elevation(self, r):
        return self.nd.a0 * j0(r)

    def define_problem(self):
        nd = self.nd
        self.set_coordinate_system("axisymmetric")
        self.mesh_template = self.mesh_class(nd.radius, nd.depth, self.initial_elevation,
                                             self.mesh_settings, reconnect=self.reconnect is not None)
        self.add_mesh(self.mesh_template)

        t = var("time")
        u = var("velocity")
        gravity = vector(0, -(nd.Bo + nd.Gamma * cos(nd.Omega * t + nd.phase)))
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=nd.Oh, gravity=gravity)
        eqs += HyperelasticSmoothedMesh()
        if self.remesh:
            eqs += RemeshWhen(self.remeshing_options)

        eqs += AxisymmetryBC() @ "axis"  # velocity_x = mesh_x = 0, also at the interface end
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "bottom"  # free slip, nodes slide along r
        eqs += DirichletBC(velocity_x=0, mesh_x=True) @ "wall"  # free slip, contact line slides

        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axis"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "wall"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "bottom"
        # pyoomph's arclength chart, which also charts the ring of a pierced bubble (inert without)
        self.interface_zeta = RingTolerantArclength(sort_along_axis="x+")
        eqs += self.interface_zeta @ "interface"

        strain = (grad(u) + transpose(grad(u))) / 2
        eqs += IntegralObservables(volume=1, kinetic_energy=dot(u, u) / 2,
                                   dissipation=2 * nd.Oh * double_dot(strain, strain))
        eqs += IntegralObservableOutput()
        if self.mesh_output:
            eqs += MeshFileOutput()

        r, z = var("coordinate_x"), var("coordinate_y")
        n = var("normal")
        norm = math.pi * nd.radius**2 * float(j0(nd.radius)) ** 2
        j0_r = j0_polynomial(r, terms_for_radius(nd.radius))
        surf = NavierStokesFreeSurface(surface_tension=1)
        surf += IntegralObservables(area=1, amp=(z - nd.depth) * j0_r * n[1] / norm)
        surf += IntegralObservableOutput()
        surf += TextFileOutput(sort_along_axis="x+")
        if self.reconnect is not None:
            from .surgery import DriverTriggeredReconnection, PressureImpulseProjection

            p = var("pressure")
            u_before = evaluate_in_past(u, 1)
            eqs += PressureImpulseProjection()
            eqs += IntegralObservables(momentum_z=u[1], divergence_sq=div(u) ** 2,
                                       projection_change=dot(u - u_before, u - u_before) / 2)
            self.recon = DriverTriggeredReconnection(
                reservoir_depth=self.reconnect.reservoir_depth,
                cap_window_factor=self.reconnect.cap_window_factor)
            surf += self.recon
            surf += IntegralObservables(pressure_sq=p**2)
            eqs += IntegralObservables(pressure_bottom=p) @ "bottom"
        eqs += surf @ "interface"

        self.add_equations(eqs @ "liquid")

    # --- per-step diagnostics -------------------------------------------------------------------

    def axis_state(self) -> tuple[float, float]:
        """Height and axial velocity of the free surface on the axis.

        After a cut three interface ends lie on the axis; this is the end of the open chain, the
        cavity's bottom or the jet growing from it."""
        if self.cut is not None:
            ch = next(c for c in self.interface_segments() if c["open"])
            return float(ch["z"][0]), float(ch["u_z"][0])
        node = next(iter(self.get_mesh("liquid/interface/axis").nodes()))
        iw = self.get_mesh("liquid").get_nodal_field_indices()["velocity_y"]
        return float(node.x(1)), float(node.value(iw))

    def _fingerprint(self) -> tuple:
        """What identifies the state an integral was evaluated on: the time, the rebuild, and the
        values of every degree of freedom and pinned value, the nodal positions included."""
        dofs, _ = self.get_current_dofs()
        h = hashlib.blake2b(np.ascontiguousarray(dofs).tobytes(), digest_size=16)
        h.update(np.ascontiguousarray(self.get_current_pinned_values(True)).tobytes())
        return (self.get_current_time(dimensional=False, as_float=True), self.remesh_count,
                len(dofs), h.hexdigest())

    def _integrals(self, mesh: str, names) -> dict[str, float]:
        """``evaluate_observable`` of ``names`` on ``mesh``; those of ``CACHED_INTEGRALS`` once per
        state (:meth:`_fingerprint`), so that a step's diagnostics, the driver's row of it and the
        budgets after a cut evaluate each integral once. The others (history or pressure) are
        always evaluated."""
        key = self._fingerprint()
        if self._integral_cache is None or self._integral_cache[0] != key:
            self._integral_cache = (key, {})
        cache = self._integral_cache[1]
        m = self.get_mesh(mesh)
        out = {}
        for name in names:
            if name not in CACHED_INTEGRALS.get(mesh, ()):
                out[name] = float(m.evaluate_observable(name))
                continue
            if (mesh, name) not in cache:
                cache[(mesh, name)] = float(m.evaluate_observable(name))
            out[name] = cache[(mesh, name)]
        return out

    def observables(self) -> dict[str, float]:
        out = self._integrals("liquid", ("volume", "kinetic_energy", "dissipation"))
        out.update(self._integrals("liquid/interface", ("area", "amp")))
        return out

    DIAG_COLUMNS = ("time", "z_axis", "w_axis", "amp", "volume", "area", "kinetic_energy",
                    "dissipation", "remesh_count")

    def start_diagnostics(self, path):
        self._diag = open(path, "w", buffering=1)
        self._diag.write("# " + "\t".join(self.DIAG_COLUMNS) + "\n")
        self.write_diagnostics()

    def write_diagnostics(self):
        if self._diag is None:
            return
        z0, w0 = self.axis_state()
        obs = self.observables()
        row = [self.get_current_time(dimensional=False, as_float=True), z0, w0, obs["amp"],
               obs["volume"], obs["area"], obs["kinetic_energy"], obs["dissipation"],
               self.remesh_count]
        self._diag.write("\t".join(f"{v:.16e}" if isinstance(v, float) else str(v)
                                   for v in row) + "\n")

    def close_diagnostics(self):
        if self._diag is not None:
            self._diag.close()
            self._diag = None

    def active_mesh_template(self) -> FaradayCylinderMesh:
        """The template the liquid mesh is remeshed from.

        After ``load_state`` pyoomph rebuilds the stored mesh from a sibling of the original template
        (``GmshTemplate._template_for_stored_mesh_file``), and later remeshing goes through that
        sibling, so run-time settings such as the focus must be set on it, not on the original.
        """
        sibling = getattr(self.get_mesh("liquid"), "_templatemesh", None)
        return sibling if isinstance(sibling, FaradayCylinderMesh) else self.mesh_template

    def interface_arrays(self):
        """Free-surface nodes ordered from the axis to the wall: ``r, z, u_r, u_z``.

        After a cut, the open chain (the cavity, from its tip on the axis to the wall)."""
        cache = self.get_cached_mesh_data("liquid/interface", nondimensional=True,
                                          tesselate_tri=False)
        xy = cache.get_coordinates()
        segs, _ = cache.get_interface_line_segments()
        if len(segs) != 1 and self.reconnect is not None:
            ch = next(c for c in self.interface_segments() if c["open"])
            return ch["r"], ch["z"], ch["u_r"], ch["u_z"]
        if len(segs) != 1:
            raise RuntimeError(f"free surface split into {len(segs)} segments")
        seg = list(segs[0])
        if xy[0, seg[0]] > xy[0, seg[-1]]:
            seg = seg[::-1]
        ur, uz = cache.get_data("velocity_x"), cache.get_data("velocity_y")
        return xy[0, seg], xy[1, seg], ur[seg], uz[seg]

    def interface_segments(self) -> list[dict]:
        """Every interface chain: ``open`` (reaches the wall), ``end_types`` and ``r, z, u_r, u_z``
        ordered from the axis to the wall (open chain) or upwards (a chain from axis to axis). A
        closed loop of the free surface, the ring of a pierced bubble, is no chain: it is set aside
        (:meth:`interface_loops` reads it), which no state without a ring has."""
        from pyoomph.meshes.zeta import _find_closed_segments

        cache = self.get_cached_mesh_data("liquid/interface", nondimensional=True,
                                          tesselate_tri=False)
        xy = cache.get_coordinates()
        segs, _ = cache.get_interface_line_segments()
        ur, uz = cache.get_data("velocity_x"), cache.get_data("velocity_y")
        tol = 1e-7 * max(1.0, self.nd.radius)
        out = []
        closed = _find_closed_segments(cache, segs) if len(segs) > 1 else [False] * len(segs)
        for seg, loop in zip(segs, closed):
            if loop:
                continue
            seg = list(seg)
            ch = orient_chain(np.column_stack([xy[0, seg], xy[1, seg]]), self.nd.radius, tol)
            first = (abs(xy[0, seg[0]] - ch.points[0, 0]) + abs(xy[1, seg[0]] - ch.points[0, 1])
                     <= abs(xy[0, seg[-1]] - ch.points[0, 0]) + abs(xy[1, seg[-1]] - ch.points[0, 1]))
            if not first:
                seg = seg[::-1]
            out.append({"open": ch.is_open, "end_types": ch.end_types, "r": xy[0, seg],
                        "z": xy[1, seg], "u_r": ur[seg], "u_z": uz[seg]})
        return sorted(out, key=lambda c: float(np.min(c["z"])))

    def interface_loops(self, *, validate: bool = True, with_node_ids: bool = False) -> list[dict]:
        """Every closed loop of the free surface (the ring of a pierced bubble, ``ring.py``):
        ``r, z, u_r, u_z`` of its nodes in the order of the walk around it, its start not
        repeated, checked to be ordered (``ring.validate_loop_order``), counter-clockwise in
        ``(r, z)`` and starting at its seam, the vertex node of largest ``r`` (ties to the larger
        ``z``), so that its closed polyline (``ring.closed``) has ``2N + 1`` nodes from a vertex;
        ``open`` False, ``end_types`` ``("loop", "loop")``. Empty without a loop.
        The candidate-step guard alone skips order validation to read a crossing.
        Optional ``node_ids`` are cache indices, comparable only on an unchanged mesh;
        they follow the same orientation and seam rotation as the coordinates."""
        from pyoomph.meshes.zeta import _find_closed_segments

        from .ring import orient_loop, rotate, seam_index, validate_loop_order

        cache = self.get_cached_mesh_data("liquid/interface", nondimensional=True,
                                          tesselate_tri=False)
        segs, _ = cache.get_interface_line_segments()
        if len(segs) < 2:
            return []
        xy = cache.get_coordinates()
        ur, uz = cache.get_data("velocity_x"), cache.get_data("velocity_y")
        vertices = {int(e[0]) for e in cache.elem_indices} | {int(e[-1])
                                                             for e in cache.elem_indices}
        out = []
        for seg, loop in zip(segs, _find_closed_segments(cache, segs)):
            if not loop:
                continue
            idx = np.array([int(i) for i in (seg[:-1] if seg[0] == seg[-1] else seg)])
            P = np.column_stack([xy[0, idx], xy[1, idx]])
            if validate:
                validate_loop_order(P)
            vertex = np.array([i in vertices for i in idx])
            P, idx, vertex = orient_loop(P, idx, vertex)
            P, idx, vertex = rotate(P, seam_index(P, vertex), idx, vertex)
            out.append({"open": False, "end_types": ("loop", "loop"), "r": P[:, 0].copy(),
                        "z": P[:, 1].copy(), "u_r": np.asarray(ur)[idx],
                        "u_z": np.asarray(uz)[idx], "vertex": vertex})
            if with_node_ids:
                out[-1]["node_ids"] = idx.copy()
        return out

    def cut_observables(self) -> dict[str, float]:
        """The budgets a cut reports: bulk integrals, on the free surface its area and the integral
        of the squared pressure, and the integral of the pressure over the bottom (the pressure
        impulse, right after the projection)."""
        out = self._integrals("liquid", ("volume", "kinetic_energy", "momentum_z",
                                         "divergence_sq", "projection_change"))
        out.update(self._integrals("liquid/interface", ("area", "pressure_sq")))
        out["pressure_bottom"] = float(self.get_mesh("liquid/bottom").evaluate_observable(
            "pressure_bottom"))
        return out

    def axial_velocity_at(self, points) -> tuple[list[float], bool]:
        """``u_z`` at ``(r, z)`` points inside the liquid; the flag is False if any is not found."""
        return self.axial_velocity_sets([points])[0]

    def axial_velocity_sets(self, sets) -> list[tuple[list[float], bool]]:
        """:meth:`axial_velocity_at` of each set of points, all located in one call: pyoomph builds
        a locator over the whole liquid mesh for each call, which dominates its cost."""
        sets = [list(pts) for pts in sets]
        flat = [[float(r), float(z)] for pts in sets for r, z in pts]
        if not flat:
            return [([], True) for _ in sets]
        bulk = self.get_mesh("liquid")
        col = 1 + bulk.get_nodal_field_indices()["velocity_y"]
        rows = bulk.evaluate_at_points(flat, lagrangian=False)
        out, k = [], 0
        for pts in sets:
            part = rows[k:k + len(pts)]
            k += len(pts)
            out.append(([float(row[col]) if row[0] == 1.0 else float("nan") for row in part],
                        all(row[0] == 1.0 for row in part)))
        return out

    def oomph_log_tail(self, size: int = 1 << 16) -> str:
        """The last ``size`` bytes of the problem's log file, or ``""`` without one.

        pyoomph raises an ``OomphLibError`` in Python only as ``RuntimeError("OomphException")``;
        its message goes to the console and to this log, whose stream is flushed first.
        """
        from pyoomph import _pyoomph_core

        _pyoomph_core._write_to_log_file("")  # writes nothing and flushes the log-file stream
        if self.logfile_name is None:
            return ""
        path = Path(self.get_output_directory(self.logfile_name))
        try:
            with path.open("rb") as fh:
                fh.seek(max(0, path.stat().st_size - size))
                return fh.read().decode(errors="replace")
        except OSError:
            return ""

    def distortion_reading(self) -> dict:
        """The element distortion of the liquid mesh as ``RemeshWhen`` reads it.

        ``trigger`` and ``ratio`` belong to the first element, in pyoomph's order, outside its
        limits (``expansion`` or ``shrinkage`` of its size against its size at the build, else a
        ``quality`` fallen below ``min_quality_decrease`` of its quality then): the element pyoomph
        names in its message, not necessarily the worst one. ``trigger`` is None when no element
        is outside them. The extremes over the mesh are ``size_ratio_min``, ``size_ratio_max`` and
        ``quality_ratio_min``. It only reads the mesh."""
        o = self.remeshing_options
        trigger, ratio = None, math.nan
        size_min, size_max, quality_min = math.inf, 0.0, math.inf
        for e in self.get_mesh("liquid").elements():
            initial = e.get_initial_cartesian_nondim_size()
            size = e.get_current_cartesian_nondim_size() / initial if initial > 0 else math.nan
            size_min, size_max = min(size_min, size), max(size_max, size)
            if trigger is None and o.max_expansion and size > o.max_expansion:
                trigger, ratio = "expansion", size
            elif trigger is None and o.min_expansion and size < o.min_expansion:
                trigger, ratio = "shrinkage", size
            initial = e.get_initial_quality_factor()
            if initial > 0:
                quality = e.get_quality_factor() / initial
                quality_min = min(quality_min, quality)
                if trigger is None and o.min_quality_decrease and quality < o.min_quality_decrease:
                    trigger, ratio = "quality", quality
        return {"trigger": trigger, "ratio": float(ratio), "size_ratio_min": float(size_min),
                "size_ratio_max": float(size_max), "quality_ratio_min": float(quality_min)}

    def quality_floor_reading(self, floor: float) -> dict | None:
        """A :meth:`distortion_reading` with trigger ``absolute-quality`` (``ratio``: the smallest
        such quality) when an element of the liquid mesh has fallen below the quality ``floor``
        (pyoomph's ``get_quality_factor``: the smallest Jacobian at its integration points over its
        mean) since its build, at which it had at least that; None otherwise. ``RemeshWhen`` asks
        only for a fifth of the quality at the build, which an element built at a mediocre quality
        loses late. An element built below the floor is not flagged here: rebuilding would make it
        again at every step, and a rebuilt mesh is checked on its own (``curved_jacobian_bound``).
        It only reads the mesh."""
        q_min = math.inf
        for e in self.get_mesh("liquid").elements():
            q = e.get_quality_factor()
            if q < floor and q < q_min and e.get_initial_quality_factor() >= floor:
                q_min = q
        if not q_min < floor:
            return None
        reading = self.distortion_reading()
        reading.update(trigger="absolute-quality", ratio=float(q_min), quality_floor=float(floor))
        return reading

    def curved_jacobian_bound(self, limit: float | None = None) -> dict | None:
        """The smallest lower bound of the scaled Jacobian over the elements of the liquid mesh
        (``reconnect.p2_jacobian_bounds``: the Bezier bound of ``det(dx/ds)`` over that of the
        straight-sided triangle; exact at the vertices), with the centroid of that element
        (``r``, ``z``), whether one of its nodes lies on the free surface (``on_interface``), the
        number of elements below ``limit`` (``below``) and of elements; None for a mesh of other
        than six-node triangles. Cached for the mesh and time it was read on."""
        key = (self.remesh_count, self.get_current_time(dimensional=False, as_float=True), limit)
        cached = getattr(self, "_jacobian_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        from scipy.spatial import cKDTree

        from .reconnect import p2_jacobian_bounds

        values, elements, types, fields = self.get_mesh("liquid").to_numpy(False, True, 0,
                                                                           False)[:4]
        elements = np.asarray(elements)
        if elements.ndim != 2 or elements.shape[1] != 6 or len(np.unique(types)) != 1:
            return None
        xy = np.asarray(values)[:, [fields["coordinate_x"], fields["coordinate_y"]]]
        bounds = p2_jacobian_bounds(xy, elements)
        k = int(np.nanargmin(bounds)) if np.isfinite(bounds).any() else int(np.argmin(bounds))
        centre = xy[elements[k, :3]].mean(axis=0)
        cache = self.get_cached_mesh_data("liquid/interface", nondimensional=True,
                                          tesselate_tri=False)
        surface = np.asarray(cache.get_coordinates())[:2].T
        d, _ = cKDTree(surface).query(xy[elements[k]])
        out = {"min": float(bounds[k]), "r": float(centre[0]), "z": float(centre[1]),
               "on_interface": bool(np.min(d) <= 1e-12 * max(1.0, self.nd.radius)),
               "below": int(np.sum(bounds < limit)) if limit is not None else -1,
               "elements": len(bounds)}
        self._jacobian_cache = (key, out)
        return out

    def arm_inverted_element_remesh(self) -> bool:
        """After a cut: pyoomph's net for an element that turns inside out within a step (as
        ``RemeshWhen(on_inverted_element=True)`` arms it): the process-wide detection of inverted
        elements on, and the liquid's template among those pyoomph rebuilds when a step stalls on
        one, restoring the state the step started from and retrying it. Returns whether it was
        armed (not on a problem that does not remesh)."""
        template = self.active_mesh_template()
        if not self.remesh or template is None or getattr(template, "remesher", None) is None:
            return False
        from pyoomph import _pyoomph_core

        _pyoomph_core.set_detect_inverted_elements(True)
        self._domains_remesh_on_inversion.add(template)
        return True

    def disarm_inverted_element_remesh(self):
        """Undo :meth:`arm_inverted_element_remesh` (the detection is process-wide)."""
        from pyoomph import _pyoomph_core

        _pyoomph_core.set_detect_inverted_elements(False)
        self._domains_remesh_on_inversion.clear()

    def take_distortion_flags(self) -> list[dict]:
        """The distortion ``RemeshWhen`` flagged on the accepted steps since the last call, one
        :meth:`distortion_reading` per flagged step (empty without), and clear the request.

        Only while ``do_call_remeshing_when_necessary`` is off does a flag reach this list, and the
        rebuild is then the caller's. pyoomph clears its request (``_domains_to_remesh``) at the
        start of every Newton solve, but neither its rebuild nor ``force_remesh`` clears it; it is
        cleared here, so that a flag already answered, by the caller or by pyoomph inside the solve,
        is not taken for a new one. Nothing reads it before the next solve otherwise."""
        flags, self._distortion_flags = self._distortion_flags, []
        if self.remeshing_necessary() and not self.do_call_remeshing_when_necessary and not flags:
            flags = [self.distortion_reading()]  # flagged outside a step in time
        self._domains_to_remesh.clear()
        return flags

    def repair_axis_midsides(self, limit: float = AXIS_MIDSIDE_LIMIT) -> dict:
        """Put back on its edge every midside node of the axis that lies more than ``limit`` edge
        lengths off the middle of its edge, at any time level.

        pyoomph's transfer of the solution to a rebuilt mesh interpolates the node positions, the
        current ones and their history, as well as the values. After a cut it has placed a midside
        node of the axis far from its edge: at the bubble's top, 2.8e-4 off the middle of an edge of
        6.9e-6, while the mesh gmsh wrote had it in place. That turned its curved element inside out
        (``curved-jacobian``). The axis is straight at every time, so such a node takes, at every
        time level, the mean of its edge's two vertices in position and in velocity, the values of
        the axis at a midside node. A node within ``limit`` is left alone, so that a mesh with no
        such node is unchanged. The levels read and set are the current one and the history the
        time stepper uses. Returns the nodes repaired, the axis edges read and the largest offset
        found, in edge lengths (``max_offset``), with where it was (``r``, ``z``)."""
        bulk = self.get_mesh("liquid")
        fields = bulk.get_nodal_field_indices()
        velocity = [fields[k] for k in ("velocity_x", "velocity_y") if k in fields]
        # the current level and the history the time stepper reads (two for BDF2); its further
        # storage holds other data (the predictor and pyoomph's continuation slots), not positions
        n_levels = self.timestepper.nprev_values() + 1
        out = {"repaired": 0, "edges": 0, "max_offset": 0.0, "r": math.nan, "z": math.nan}
        for e in self.get_mesh("liquid/axis").elements():
            if e.nnode() != 3:  # a straight edge has no midside node to misplace
                continue
            a, mid, b = (e.node_pt(k) for k in range(3))
            out["edges"] += 1
            levels = range(min(n_levels, mid.ntstorage()))
            offset = 0.0
            for t in levels:
                length = math.hypot(b.x_at_t(t, 0) - a.x_at_t(t, 0), b.x_at_t(t, 1) - a.x_at_t(t, 1))
                gap = math.hypot(mid.x_at_t(t, 0) - 0.5 * (a.x_at_t(t, 0) + b.x_at_t(t, 0)),
                                 mid.x_at_t(t, 1) - 0.5 * (a.x_at_t(t, 1) + b.x_at_t(t, 1)))
                if length > 0.0:
                    offset = max(offset, gap / length)
            if offset > out["max_offset"]:
                out.update(max_offset=offset, r=mid.x(0), z=mid.x(1))
            if offset <= limit:
                continue
            for t in levels:
                for i in range(2):
                    mid.set_x_at_t(t, i, 0.5 * (a.x_at_t(t, i) + b.x_at_t(t, i)))
                for k in velocity:
                    mid.set_value_at_t(t, k, 0.5 * (a.value_at_t(t, k) + b.value_at_t(t, k)))
            out["repaired"] += 1
        if out["repaired"]:
            self.reapply_boundary_conditions()
        return out

    def straighten_interface_edges(self, r: float, z: float, radius: float, *,
                                   offset_min: float = AXIS_MIDSIDE_LIMIT,
                                   apply: bool = True) -> dict:
        """Put on the middle of its chord the midside node of every quadratic element of the free
        surface whose midpoint lies within ``radius`` of ``(r, z)`` and whose midside node lies
        more than ``offset_min`` chords off that middle, at every time level the time stepper reads
        (position and velocity, as :meth:`repair_axis_midsides` does on the axis); with ``apply``
        false the edges are only read. Each edge's volume of revolution between its quadratic curve
        and its chord, ``2 pi r_m (2/3) |chord| d`` (``d`` the signed offset of the midside node
        from the chord, ``r_m`` the radius of its middle), is summed with its sign (``dV``) and in
        magnitude (``dV_abs``). Returns the edges straightened (``edges``: the midside node's
        place, its offset in chords, its volume) and those sums. A mesh with no such edge within
        ``radius`` is unchanged."""
        bulk = self.get_mesh("liquid")
        fields = bulk.get_nodal_field_indices()
        velocity = [fields[k] for k in ("velocity_x", "velocity_y") if k in fields]
        n_levels = self.timestepper.nprev_values() + 1
        out = {"edges": [], "dV": 0.0, "dV_abs": 0.0, "read": 0}
        for e in self.get_mesh("liquid/interface").elements():
            if e.nnode() != 3:
                continue
            a, mid, b = (e.node_pt(k) for k in range(3))
            ca = np.array([a.x(0), a.x(1)])
            cb = np.array([b.x(0), b.x(1)])
            cm = np.array([mid.x(0), mid.x(1)])
            if math.hypot(*(0.5 * (ca + cb) - np.array([r, z]))) > radius:
                continue
            out["read"] += 1
            chord = cb - ca
            length = float(math.hypot(*chord))
            if length <= 0.0:
                continue
            offset = float(math.hypot(*(cm - 0.5 * (ca + cb)))) / length
            if offset <= offset_min:
                continue
            normal = np.array([-chord[1], chord[0]]) / length
            d = float(np.dot(cm - 0.5 * (ca + cb), normal))
            dV = 2.0 * math.pi * float(0.5 * (ca[0] + cb[0])) * (2.0 / 3.0) * length * d
            out["edges"].append({"r": float(cm[0]), "z": float(cm[1]), "offset": offset,
                                 "chord": length, "dV": dV})
            out["dV"] += dV
            out["dV_abs"] += abs(dV)
            if not apply:
                continue
            levels = range(min(n_levels, mid.ntstorage()))
            for t in levels:
                for i in range(2):
                    mid.set_x_at_t(t, i, 0.5 * (a.x_at_t(t, i) + b.x_at_t(t, i)))
                for k in velocity:
                    mid.set_value_at_t(t, k, 0.5 * (a.value_at_t(t, k) + b.value_at_t(t, k)))
        if apply and out["edges"]:
            self.reapply_boundary_conditions()
            self.invalidate_cached_mesh_data()
            self._jacobian_cache = None
        return out

    # --- at least min_newton_iterations Newton iterations per solve ------------------------------

    def actions_before_newton_solve(self):
        self.release_newton_tolerance()  # a solve that failed while it was held
        rule = self.newton_floor_rule
        if rule is not None:
            # the round-off floor of the time-discrete residual grows like 1/dt for the step this
            # solve makes, which pyoomph's adaptive stepping halves below the driver's after a
            # rejection: a tolerance fixed at the driver's newton_floor/dt is then below the floor
            # at every halving (r1h at 7251 and r1 at 7525 at 1f509ac, demo6 at 7503: Newton
            # stalled at 2-10 times it, the stall doubling with each halving, down to the smallest
            # step); the driver's own step keeps its tolerance as before
            dt = float(self.timestepper.time_pt().dt(0))
            if dt > 0.0:
                self.newton_solver_tolerance = max(rule[0], rule[1] / dt)
        super().actions_before_newton_solve()
        self._newton_checks = 0

    def actions_before_newton_convergence_check(self):
        super().actions_before_newton_convergence_check()
        if self.min_newton_iterations <= 1:
            return
        # oomph-lib calls this before the residual it tests at the start of a solve and after each
        # Newton step, so call k + 1 precedes the test after step k: below min_newton_iterations
        # steps no residual passes the zero tolerance held, after them the tolerance comes back
        self._newton_checks += 1
        if self._newton_checks <= self.min_newton_iterations:
            if self._held_tolerance is None:
                self._held_tolerance = self.newton_solver_tolerance
            self.newton_solver_tolerance = 0.0
        else:
            self.release_newton_tolerance()

    def actions_after_newton_solve(self):
        self.release_newton_tolerance()
        super().actions_after_newton_solve()

    def _snapshot_state(self, to_memory: bool = True):
        """pyoomph's snapshot of the state (the one its net for an inverted element takes before a
        step), the first of a step also kept while the driver collects it (``_step_snapshots`` a
        list, ``CollapseDriver._post_cut_solve``): the state the step started from, which a step
        that fails is rolled back to (``CollapseDriver._failed_step_stop``). Nothing else changes."""
        snap = super()._snapshot_state(to_memory)
        kept = getattr(self, "_step_snapshots", None)
        if kept is not None and not kept and to_memory:
            kept.append(snap)
        return snap

    # the project's attributes of the mesh's template, which pyoomph's reloaded template (a sibling of
    # the problem's original one, copying that one's) does not carry: kept across an in-memory restore
    TEMPLATE_KEYS = ("focus", "cut_sizing", "high_order_threshold", "rebuild_spline", "ring_stage",
                     "ring_sizing")

    def _restore_state(self, snapshot):
        """pyoomph's restore of a snapshot (``load_state`` of it), the template's project attributes
        (:data:`TEMPLATE_KEYS`: the driver's focus, the cut's sizing, the ring stage's) put back on
        the template the restore leaves active, and pyoomph's net for an inverted element re-armed
        on it (not twice: the old template out of its set).

        A snapshot taken on one mesh and restored after a rebuild (the rollback of a failed step
        after the net's rebuilds, :meth:`CollapseDriver._failed_step_stop`; that of an axis repair)
        reloads the stored mesh from a new sibling of the problem's ORIGINAL template
        (``GmshTemplate._template_for_stored_mesh_file`` copies its ``__dict__``), and in a run
        restarted from a dump, every leg of a chain, the attributes the driver and the cut set live
        on the sibling the restart made, not on that original: the restored template had the focus
        None, no cut sizing and the ring stage off, and every rebuild after the rollback met the mesh
        without them (r1h at b36c9c1: the end state of its failed step at 7251, dumped after the
        rollback, held no focus, and the net's rebuilds from it left the open chain 237 nodes
        against 4873). A restore onto the same mesh makes no sibling and changes nothing."""
        old = self.active_mesh_template()
        kept = ({k: getattr(old, k) for k in self.TEMPLATE_KEYS if hasattr(old, k)}
                if old is not None else {})
        armed = old is not None and old in getattr(self, "_domains_remesh_on_inversion", set())
        super()._restore_state(snapshot)
        new = self.active_mesh_template()
        if new is not None and new is not old:
            for k, v in kept.items():
                setattr(new, k, v)
            if armed:
                self._domains_remesh_on_inversion.discard(old)
                self._domains_remesh_on_inversion.add(new)

    def release_newton_tolerance(self):
        """Give back the Newton tolerance that a solve held at zero for its first iterations."""
        if self._held_tolerance is not None:
            self.newton_solver_tolerance = self._held_tolerance
            self._held_tolerance = None

    def actions_after_remeshing(self):
        super().actions_after_remeshing()
        self.remesh_count += 1
        if self.cut is not None:  # after a cut (surgery.cut, surgery.restore_cut)
            self.axis_repair = self.repair_axis_midsides()
            self.axis_repairs_total += self.axis_repair["repaired"]

    def actions_after_transient_solve(self):
        super().actions_after_transient_solve()
        if not self._projecting:  # the projection's solve is no step in time
            self.write_diagnostics()
            # an accepted step whose last Newton solve RemeshWhen flagged, while the rebuild is left
            # to the driver: read the distortion now, since a later sub-step clears the flag
            if self._domains_to_remesh and not self.do_call_remeshing_when_necessary:
                self._distortion_flags.append(self.distortion_reading())
