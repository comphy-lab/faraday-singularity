"""pyoomph problem: an axisymmetric Faraday wave in a vertically vibrated cylinder.

Adapted from the surfactant-free path of faraday-surfactant-fem (planar, vertical-only mesh motion)
to an axisymmetric container whose free surface can deform arbitrarily:

* coordinates ``(r, z) = (x, y)`` with pyoomph's axisymmetric measure ``2 pi r``;
* incompressible Navier-Stokes in the container frame with body force
  ``-(Bo + Gamma cos(Omega t)) e_z`` (``parameters.py`` defines the scales);
* free surface with unit surface tension and the kinematic condition coupled to the mesh;
* axis ``r = 0`` (``AxisymmetryBC``), free-slip bottom ``z = 0``, free-slip side wall ``r = R_c``
  along which the contact line slides with the natural 90 degree angle;
* hyperelastic mesh smoothing on an unstructured triangular mesh that is rebuilt from the live free
  surface whenever elements are too distorted (``RemeshWhen``).

With ``R_c = j_{1,n}`` and a free-slip wall, ``J0(r)`` is an exact mode of the linearised problem,
so the fundamental-mode amplitude is the Hankel coefficient

    a(t) = 2 int_0^Rc zeta J0(r) r dr / (R_c^2 J0(R_c)^2),

evaluated on the interface with ``n_z ds = dr`` and pyoomph's ``2 pi r`` measure.
"""

from __future__ import annotations

import math

from pyoomph import Problem
from pyoomph.equations.ALE import HyperelasticSmoothedMesh
from pyoomph.equations.generic import (AxisymmetryBC, DirichletBC, IntegralObservables,
                                       RemeshingOptions, RemeshWhen)
from pyoomph.equations.navier_stokes import NavierStokesEquations, NavierStokesFreeSurface
from pyoomph.expressions import cos, dot, double_dot, grad, transpose, var, vector
from pyoomph.meshes.zeta import (AssignZetaCoordinatesByArclength,
                                 AssignZetaCoordinatesByEulerianCoordinate)
from pyoomph.output.generic import IntegralObservableOutput, TextFileOutput
from pyoomph.output.meshio import MeshFileOutput
from scipy.special import j0

from .bessel import j0_polynomial, terms_for_radius
from .mesh import FaradayCylinderMesh, MeshSettings
from .parameters import Dimensionless


class FaradayProblem(Problem):
    def __init__(self, nd: Dimensionless, *, mesh: MeshSettings, remesh: bool = True,
                 mesh_output: bool = False):
        super().__init__()
        self.nd = nd
        self.mesh_settings = mesh
        self.remesh = remesh
        self.mesh_output = mesh_output
        self.remesh_count = 0
        self._diag = None  # open per-step diagnostics file, set by start_diagnostics()

    def initial_elevation(self, r):
        return self.nd.a0 * j0(r)

    def define_problem(self):
        nd = self.nd
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(FaradayCylinderMesh(nd.radius, nd.depth, self.initial_elevation,
                                          self.mesh_settings))

        t = var("time")
        u = var("velocity")
        gravity = vector(0, -(nd.Bo + nd.Gamma * cos(nd.Omega * t)))
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
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="x+") @ "interface"

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
        eqs += surf @ "interface"

        self.add_equations(eqs @ "liquid")

    # --- per-step diagnostics -------------------------------------------------------------------

    def axis_state(self) -> tuple[float, float]:
        """Height and axial velocity of the free surface on the axis."""
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

    def actions_after_remeshing(self):
        super().actions_after_remeshing()
        self.remesh_count += 1

    def actions_after_transient_solve(self):
        super().actions_after_transient_solve()
        self.write_diagnostics()
