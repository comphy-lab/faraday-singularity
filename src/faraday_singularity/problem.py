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
  surface whenever elements are too distorted (``RemeshWhen``).

With ``R_c = j_{1,n}`` and a free-slip wall, ``J0(r)`` is an exact mode of the linearised problem,
so the fundamental-mode amplitude is the Hankel coefficient

    a(t) = 2 int_0^Rc zeta J0(r) r dr / (R_c^2 J0(R_c)^2),

evaluated on the interface with ``n_z ds = dr`` and pyoomph's ``2 pi r`` measure.

A problem made with ``reconnect`` settings (a case's ``collapse.reconnect`` block) can be cut through
a closing neck (``surgery.py``). Only then does it carry the reconnection handler on the free surface
(inert until the cut calls it), the pressure-impulse projection as an auxiliary residual, and the
observables the cut reports (axial momentum, ``int (div u)^2``, the change made by the projection, the
pressure on the free surface). It adds no field, degree of freedom or global parameter, so state dumps
of either kind of problem load into the other. Without ``reconnect`` the problem is the one above.
"""

from __future__ import annotations

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
        self._projecting = False  # set while the projection's auxiliary solve runs

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
            eqs += RemeshWhen(RemeshingOptions())

        eqs += AxisymmetryBC() @ "axis"  # velocity_x = mesh_x = 0, also at the interface end
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "bottom"  # free slip, nodes slide along r
        eqs += DirichletBC(velocity_x=0, mesh_x=True) @ "wall"  # free slip, contact line slides

        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axis"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "wall"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "bottom"
        self.interface_zeta = AssignZetaCoordinatesByArclength(sort_along_axis="x+")
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

    def observables(self) -> dict[str, float]:
        bulk = self.get_mesh("liquid")
        surf = self.get_mesh("liquid/interface")
        out = {name: float(bulk.evaluate_observable(name))
               for name in ("volume", "kinetic_energy", "dissipation")}
        out.update({name: float(surf.evaluate_observable(name)) for name in ("area", "amp")})
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
        ordered from the axis to the wall (open chain) or upwards (a chain from axis to axis)."""
        cache = self.get_cached_mesh_data("liquid/interface", nondimensional=True,
                                          tesselate_tri=False)
        xy = cache.get_coordinates()
        segs, _ = cache.get_interface_line_segments()
        ur, uz = cache.get_data("velocity_x"), cache.get_data("velocity_y")
        tol = 1e-7 * max(1.0, self.nd.radius)
        out = []
        for seg in segs:
            seg = list(seg)
            ch = orient_chain(np.column_stack([xy[0, seg], xy[1, seg]]), self.nd.radius, tol)
            first = (abs(xy[0, seg[0]] - ch.points[0, 0]) + abs(xy[1, seg[0]] - ch.points[0, 1])
                     <= abs(xy[0, seg[-1]] - ch.points[0, 0]) + abs(xy[1, seg[-1]] - ch.points[0, 1]))
            if not first:
                seg = seg[::-1]
            out.append({"open": ch.is_open, "end_types": ch.end_types, "r": xy[0, seg],
                        "z": xy[1, seg], "u_r": ur[seg], "u_z": uz[seg]})
        return sorted(out, key=lambda c: float(np.min(c["z"])))

    def cut_observables(self) -> dict[str, float]:
        """The budgets a cut reports: bulk integrals, and on the free surface its area and the
        integral of the squared pressure (the pressure impulse, right after the projection)."""
        bulk = self.get_mesh("liquid")
        surf = self.get_mesh("liquid/interface")
        out = {name: float(bulk.evaluate_observable(name))
               for name in ("volume", "kinetic_energy", "momentum_z", "divergence_sq",
                            "projection_change")}
        out.update({name: float(surf.evaluate_observable(name)) for name in ("area", "pressure_sq")})
        return out

    def axial_velocity_at(self, points) -> tuple[list[float], bool]:
        """``u_z`` at ``(r, z)`` points inside the liquid; the flag is False if any is not found."""
        bulk = self.get_mesh("liquid")
        col = 1 + bulk.get_nodal_field_indices()["velocity_y"]
        rows = bulk.evaluate_at_points([[float(r), float(z)] for r, z in points], lagrangian=False)
        found = all(row[0] == 1.0 for row in rows)
        return [float(row[col]) if row[0] == 1.0 else float("nan") for row in rows], found

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

    def actions_after_remeshing(self):
        super().actions_after_remeshing()
        self.remesh_count += 1

    def actions_after_transient_solve(self):
        super().actions_after_transient_solve()
        if not self._projecting:  # the projection's solve is no step in time
            self.write_diagnostics()
