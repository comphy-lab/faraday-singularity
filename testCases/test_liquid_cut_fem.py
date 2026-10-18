"""Finite-element tests of the cut of a liquid neck after the gas cut.

* pyoomph's own parking of a plan, which the cut's ``surgery.park_plan`` copies (on the necked
  cavity of ``test_reconnect_fem.py``);
* the identity of the box's cutoff in the projection, on the manufactured field of
  ``test_reconnect_fem.py`` (``u~ = u_s + grad(phi)``).

Each problem compiles in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

pytest.importorskip("pyoomph")

import test_reconnect_fem as fem  # noqa: E402

from faraday_singularity import reconnect as rc  # noqa: E402
from faraday_singularity import surgery  # noqa: E402

RECON = rc.ReconnectSettings(delta=0.05, dt0=1e-5, tip_size=0.25, bridge_size=0.25,
                             focus_size=0.25)


# --- pyoomph's parking, and the identity of the cutoff --------------------------------------------


def test_the_plan_is_parked_as_pyoomphs_own_hook_parks_one(tmp_path):
    """``surgery.park_plan`` parks a plan with the same names, scale and remesh request as
    pyoomph's reconnection handler parks its own (``after_newton_solve``), on the necked cavity of
    ``test_reconnect_fem.py``: the table and the cut radius as given."""
    with fem.neck_problem(tmp_path, RECON) as pb:
        pb.initialise()
        template = pb.get_mesh("liquid")._templatemesh
        assert pb.recon.request_pinch(pb, 1.05 * fem.R_N) == "plan"
        plan, theirs = template._pending_surgery_plans[surgery.INTERFACE]
        assert pb.recon._pending_on == (template, surgery.INTERFACE)
        assert template in pb._domains_to_remesh
        surgery.clear_plan(pb)
        assert not template.has_pending_surgery_plan(surgery.INTERFACE)
        table = np.asarray(theirs["old_zeta_table"])
        surgery.park_plan(pb, plan, table, rmin=0.02)
        got, ours = template._pending_surgery_plans[surgery.INTERFACE]
        assert got is plan and pb.recon._pending_on == (template, surgery.INTERFACE)
        assert template in pb._domains_to_remesh
        assert set(ours) == set(theirs)
        assert str(ours["spatial_scale"]) == str(theirs["spatial_scale"])
        for key in ("interface_name", "axis_name", "opposite_axis_name",
                    "opposite_interface_name", "distmin_nd"):
            assert ours[key] == theirs[key], key
        assert ours["interface_name"] == surgery.INTERFACE and ours["axis_name"] == "axis"
        assert np.array_equal(ours["old_zeta_table"], table) and ours["rmin_nd"] == 0.02
        surgery.clear_plan(pb)


def test_the_identity_of_the_cutoff_holds_in_the_projection_of_a_gradient(tmp_path):
    """On the manufactured field ``u~ = u_s + grad(phi)`` (``phi = 0`` on the free top), with the
    box inside the liquid: ``int (u+ - u~)_z phi_B dV = int Pi dphi_B/dz dV`` to the quadrature of
    the grid and the interpolation of the cutoff, both sides of the size of the impulse, and the
    error falls with the mesh; the momentum and energy the projection changes in the box are those
    of the grid's samples, and without the box nothing is sampled."""
    errors = []
    for N in (8, 16):
        pb = fem.ManufacturedProblem(N)
        pb.set_output_directory(str(tmp_path / f"m{N}"))
        with pb:
            pb.initialise()
            pb.set_initial_condition()
            t0 = pb.get_current_time(dimensional=False, as_float=True)
            s = rc.ReconnectSettings(delta=1.0, dt0=0.25, restore_pressure=False)
            out = surgery.project(pb, s, t0, box=(0.5, 0.35))
            box = out["box"]
            assert box["z"] == 0.5 and box["R"] == 0.35 and box["n"] == surgery.BOX_SAMPLES
            assert box["points"] == surgery.BOX_SAMPLES**2  # the half-disc lies in the liquid
            assert abs(box["phi_lhs"]) > 1e-2 and box["phi_lhs"] * box["phi_rhs"] > 0.0
            assert box["delta_Pz"] == pytest.approx(box["after"]["Pz"] - box["before"]["Pz"])
            assert box["delta_KE"] < 0.0  # the projection takes kinetic energy
            errors.append(abs(box["phi_relative_error"]))
            plain = surgery.project(pb, s, t0)
            assert "box" not in plain
    assert errors[1] < errors[0] and errors[1] < 1e-4, errors
