"""The polytropic bubble through the ring's cut, on the synthetic pierced state of
``ring_fem_state.py`` with the gas's traction field: a bubble of gas at 1.1 p0 on its polytrope,
the driver's ring cut of the piercing, and the steps of the ring stage. The ring keeps the bubble's
gas, all of it, on the same polytrope (mass conserved; ``bubble-gas-bubble-to-ring`` with the
volumes and pressures before and after); every step of the ring stage holds the traction
``p(V_ring) - p0`` on the ring's nodes and none on the open chain; the rows of ``bubble_gas.txt``
say ``ring``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("pyoomph")

RUN = """
import dataclasses, json, sys
from pathlib import Path
src, tests, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, src)
sys.path.insert(1, tests)
sys.argv = sys.argv[:1]
import numpy as np
from scipy.spatial import cKDTree
import ring_fem_state as S
import test_reconnect_fem as fem
from faraday_singularity import events as ev
from faraday_singularity.driver import CollapseDriver, CollapseSettings
from faraday_singularity.mesh import MeshSettings
from faraday_singularity.neckgas.bubble import BubbleGas
from faraday_singularity.neckgas.coupler import GasSettings
from faraday_singularity.parameters import CaseParameters, nondimensionalise
block = ev.EventSettings.from_dict({"ring": {"enabled": True, "sigma_V": 1e-6, "sigma_Pz": 1e-6,
                                             "sigma_KE": 1e-6}})
recon = dataclasses.replace(S.RECON, bubble_model="polytropic")
gas = GasSettings.from_dict({"z_hi": 10.0, "neck_window": [0.0, 1.0]})
nd = nondimensionalise(CaseParameters.from_dict(fem.PHYSICS))
pb = S.PierceProblem(nd, mesh=MeshSettings(**S.MESH), reconnect=recon, gas=True)
pb.set_output_directory("pb")
with pb:
    pb.initialise()
    S.gas_cut_record(pb)
    S.impose(pb)
    drv = CollapseDriver(pb, CollapseSettings(t_end=10.0, dt_max=0.01, state_dt=0.0, state_keep=50,
                                              max_steps=4), out, reconnect=recon, events=block,
                         gas=gas)
    drv.gas.retire()  # the neck's gas handed over: a bubble at 1.1 p0 stands in for it
    stage, V, _ = drv._bubble_surface()
    p = 1.1 * gas.p0
    drv._bubble = BubbleGas(model="polytropic", kappa=1.4, p0=gas.p0, rho0=gas.density_ratio,
                            m=gas.density_ratio * 1.1 ** (1 / 1.4) * V, C=p * V ** 1.4, V_ref=V,
                            p_ref=p)
    drv._bubble.V_end = V
    drv._open_bubble_rows()
    drv.reset_after_cut(pb.cut)
    result = drv.run(1e-8)
    drv.close()
    r, z, T = pb.gas_traction()
    by = {}
    for name, cs in (("ring", pb.interface_loops()), ("open", pb.interface_segments())):
        vals = []
        for c in cs:
            d, _ = cKDTree(np.column_stack([c["r"], c["z"]])).query(np.column_stack([r, z]))
            vals += [float(x) for x in T[d <= 1e-12]]
        by[name] = vals
    (out / "traction_end.json").write_text(json.dumps(by))
    (out / "bubble_start.json").write_text(json.dumps({"V": V, "C": p * V ** 1.4, "p0": gas.p0}))
(out / "result.json").write_text(json.dumps(result, default=float, sort_keys=True))
"""


def test_the_ring_keeps_the_bubble_gas(tmp_path):
    env = dict(os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "-c", RUN, str(ROOT / "src"), str(ROOT / "testCases"),
                           str(out)], capture_output=True, text=True, timeout=1800, cwd=tmp_path,
                          env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads((out / "result.json").read_text())
    assert result["events"]["ring"]["cut"] == 1, result["events"]
    start = json.loads((out / "bubble_start.json").read_text())
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    tr = next(e for e in events if e["event"] == "bubble-gas-bubble-to-ring")
    assert tr["V_after"] < tr["V_before"] and tr["m"] == pytest.approx(
        1.2 / 998.0 * 1.1 ** (1 / 1.4) * start["V"], rel=1e-14)
    assert tr["p_after"] == pytest.approx(start["C"] / tr["V_after"] ** 1.4, rel=1e-12)
    lines = (out / "bubble_gas.txt").read_text().splitlines()
    head = lines[0][2:].split("\t")
    rows = [dict(zip(head, r.split("\t"))) for r in lines[1:]]
    ring_rows = [r for r in rows if r["stage"] == "ring"]
    assert ring_rows and all(int(r["nodes"]) > 10 for r in ring_rows)
    for r in ring_rows:
        assert float(r["p_start"]) == pytest.approx(start["C"] / float(r["V_start"]) ** 1.4,
                                                    rel=1e-11)
    T = json.loads((out / "traction_end.json").read_text())
    last = float(ring_rows[-1]["T_start"])
    assert T["ring"] and all(x == pytest.approx(last, rel=1e-12) for x in T["ring"])
    assert T["open"] and all(x == 0.0 for x in T["open"])
    assert result["bubble_gas"]["stage"] == "ring"
