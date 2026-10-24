"""The gas traction on the finite-element free surface (``FaradayProblem(gas=True)``): the quick
form of gates G1 and G2 of ``verificationCases/neck-gas/forced_wave.py`` on a coarse mesh, in a
fresh process. A uniform traction ``T`` on the layer at rest leaves it at rest and raises its
pressure by ``T`` (the sign: ``T`` is a gas overpressure pushing into the liquid), and does not
change a free decay; a traction ``eps J0(r) sin(omega t)`` set before each step gives the
amplitude of the time scheme's own linear response with that one-step lag, and lags the response
without it by one step (``omega dt``, first order in ``dt``). A problem without gas has no field."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verificationCases" / "neck-gas" / "forced_wave.py"


def test_quick_g1_and_g2(tmp_path):
    pytest.importorskip("pyoomph")
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-I", str(SCRIPT), "gate", "--out",
                           str(tmp_path / "gate"), "--quick"], capture_output=True, text=True,
                          env=env, timeout=1800, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    res = json.loads((tmp_path / "gate" / "results.json").read_text())
    g1 = res["G1"]
    for T in ("10", "100"):
        assert g1[f"rest_T{T}"]["dp_over_T_minus_1_max"] <= 1e-10
        assert g1[f"rest_T{T}"]["u_max"] <= 1e-10
    assert g1["decay_T100_vs_T0"]["max_rel_difference"] <= 1e-10
    for row in res["G2"]:
        assert abs(row["scale_vs_lagged"] - 1.0) <= 1e-3, row
        assert abs(row["phase_vs_lagged"]) <= 1e-3, row
        assert abs(row["lag_phase"] - row["predicted_lag_phase"]) <= 1e-3, row
        assert 0.98 < row["lag_over_dt"] < 1.05, row  # one step: omega dt, not omega dt/2
    assert all(x["pass"] for x in res["G2_order"])


def test_a_problem_without_gas_has_no_traction_field(tmp_path):
    pytest.importorskip("pyoomph")
    script = """
import sys
sys.path.insert(0, sys.argv[1])
sys.argv = sys.argv[:1]
from faraday_singularity.mesh import MeshSettings
from faraday_singularity.parameters import CaseParameters, nondimensionalise
from faraday_singularity.problem import FaradayProblem
nd = nondimensionalise(CaseParameters.from_dict({"fluid": {"Oh": 0.05, "Bo": 0.0},
    "container": {"mode": 1, "depth": 2.0}, "forcing": {"Gamma": 0.0}, "initial": {"amplitude": 0.0}}))
pb = FaradayProblem(nd, mesh=MeshSettings(h_interface=0.2, h_axis=0.2, h_max=0.4, grading=0.3,
                                          interface_points=101))
pb.set_output_directory(sys.argv[0] + "-out")
with pb:
    pb.initialise()
    try:
        pb.set_gas_traction(1.0)
    except RuntimeError as exc:
        print("refused:", exc)
"""
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", script, str(ROOT / "src")], capture_output=True,
                          text=True, env=env, cwd=tmp_path, timeout=600, check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "refused: the problem was made without gas" in proc.stdout
