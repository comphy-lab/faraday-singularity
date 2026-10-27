"""Contract test for the run status that ``run_case.py`` records when a run is terminated.

The runner executes in a child process with stand-ins for the problem and the collapse driver, so no
finite-element problem is built. Like pyoomph's initialisation, the stand-in resets the SIGTERM
disposition to the default action, which would end the process without recording why.
"""

from __future__ import annotations

import json
import hashlib
import math
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASE = {
    "schema": "pyoomph-case-v1",
    "case_id": "sigterm",
    "geometry": {"kind": "axisymmetric"},
    "physics": {"fluid": {"Oh": 0.04, "Bo": 0.0}, "container": {"mode": 1, "depth": 2 * math.pi},
                "forcing": {"Gamma": 1.0, "detuning": 1.0, "phase": -math.pi / 2},
                "initial": {"amplitude": 0.5}},
    "tolerances": {"newton": 1e-9},
    "mesh": {"h_interface": 0.05, "h_axis": 0.05, "h_max": 0.2, "grading": 0.3,
             "interface_points": 801},
    "time_stepping": {"scheme": "BDF2", "steps_per_forcing_period": 100},
    "collapse": {},
    "output": {"cadence": 0.25, "cadence_unit": "forcing periods", "write_states": False},
    "completion": {"kind": "event", "t_max": 3, "unit": "forcing periods"},
}

CHILD = """
import os, signal, sys, time, types
sys.path.insert(0, sys.argv[1])
import run_case

class Problem:
    remesh_count = 0
    def __init__(self, nd, **kw): pass
    def set_output_directory(self, path): pass
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def initialise(self): signal.signal(signal.SIGTERM, signal.SIG_DFL)  # as pyoomph does
    def start_diagnostics(self, path): pass
    def close_diagnostics(self): pass

class Driver:
    def __init__(self, problem, settings, outdir): pass
    def run(self, tol):  # terminated while the collapse runs
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    def close(self): pass

sys.modules["faraday_singularity.problem"] = types.SimpleNamespace(FaradayProblem=Problem)
run_case.CollapseDriver = Driver
sys.argv = ["run_case.py", sys.argv[2], "--outdir", sys.argv[3]]
run_case.main()
"""


def test_sigterm_during_a_collapse_run_is_recorded(tmp_path):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), str(case), str(out)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode != -signal.SIGTERM, "killed by SIGTERM without recording it"
    assert "Terminated" in proc.stderr
    status = json.loads((out / "run_status.json").read_text())
    assert status["state"] == "interrupted" and status["detail"] == "Terminated: signal 15"
    manifest = json.loads((out / "run-manifest.json").read_text())
    assert manifest["schema"] == "pyoomph-runner-manifest-v1"
    assert manifest["case_sha256"] == hashlib.sha256(case.read_bytes()).hexdigest()
    assert manifest["case_id"] == CASE["case_id"]
    assert "completion.t_max" in manifest["case_fields_used"]
    assert "completion.value" not in manifest["case_fields_used"]
    # the entry point and run_case.py itself are both identified
    assert manifest["runner"]["run_case_sha256"] == hashlib.sha256(
        (ROOT / "run_case.py").read_bytes()).hexdigest()
