"""A cut with a polytropic bubble (``reconnect.bubble_model`` ``polytropic``): the synthetic necked
cavity's landing state of a two-way gas (``test_neckgas_restart_fem.py``) cut and stepped on.
The bubble takes the coupled gas's partition at the cut's neck (``gas_cut.json``, its record
``bubble``, event ``bubble-gas``); every step holds the traction ``p(V) - p0`` of the polytrope at
the volume of the bubble's chain on the bubble's nodes and none on the open chain; the rows of
``bubble_gas.txt`` follow the polytrope, keep the mass and close the work; the dumps after the cut
carry the bubble's state and a restart from one continues its rows to round-off; and the bubble's
pressure changes the run after the cut against the passive bubble of the same cut."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testCases"))
pytest.importorskip("pyoomph")

import test_neckgas_restart_fem as R  # noqa: E402
from test_neckgas_restart_fem import base  # noqa: E402,F401  (the landing run, module-scoped)

CUT = {"delta": 0.0153, "dt0": 1e-5, "tip_size": 0.25, "bridge_size": 0.25, "focus_size": 0.25}


def _rows(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return [dict(zip(head, r.split("\t"))) for r in lines[1:]]


@pytest.fixture(scope="module")
def cuts(base):  # noqa: F811
    landing = base / "base" / "run" / "states" / "landing_0.0153.dump"
    out = {}
    for model in ("polytropic", "constant-pressure"):
        spec = dict(R.SPEC, restart=str(landing), cut=dict(CUT, bubble_model=model),
                    traction_out=True)
        spec["collapse"] = dict(R.SPEC["collapse"], max_steps=9)
        R._run(base / f"cut-{model}", spec)
        out[model] = base / f"cut-{model}" / "run"
    return out


def test_the_bubble_takes_the_coupled_gas_and_holds_its_pressure(cuts):
    out = cuts["polytropic"]
    rec = json.loads((out / "gas_cut.json").read_text())
    b, h = rec["bubble"], rec["handover"]["A"]
    assert b["model"] == "polytropic" and b["kappa"] == 1.4 and b["m"] == h["m_b"]
    assert b["p_ref"] == h["p_b_mix"] and b["V_ref"] == h["V_gas_below"]
    kinds = [json.loads(x)["event"] for x in (out / "events.jsonl").read_text().splitlines()]
    assert kinds.index("gas-retired") < kinds.index("bubble-gas") < kinds.index("cut")
    rows = _rows(out / "bubble_gas.txt")
    assert len(rows) >= 3 and {r["stage"] for r in rows} == {"bubble"}
    C, p0 = b["p_ref"] * b["V_ref"] ** 1.4, h["p0"]
    for r in rows:
        V0, V1 = float(r["V_start"]), float(r["V_end"])
        assert float(r["p_start"]) == pytest.approx(C / V0 ** 1.4, rel=1e-11)
        assert float(r["T_start"]) == pytest.approx(C / V0 ** 1.4 - p0, rel=1e-11)
        assert float(r["m"]) == pytest.approx(b["m"], rel=1e-12) and int(r["nodes"]) > 10
    W_t, W_p = float(rows[-1]["W_traction"]), float(rows[-1]["W_poly"])
    assert abs(W_t - W_p) <= 0.05 * abs(W_p) + 1e-14  # the lag of the coupling
    T = json.loads((out / "traction_end.json").read_text())
    last = float(rows[-1]["T_start"])
    assert T["bubble"] and all(x == pytest.approx(last, rel=1e-12) for x in T["bubble"])
    assert T["open"] and all(x == 0.0 for x in T["open"])  # the cavity vents to the atmosphere
    status = json.loads((out / "result.json").read_text())
    assert status["bubble_gas"]["stage"] == "bubble" and status["gas"]["status"] == "retired"
    for p in (out / "states").glob("*.dump"):
        meta = json.loads(p.with_suffix(".json").read_text())
        if meta.get("cut") is not None:
            assert meta["bubble_gas"]["model"] == "polytropic" and meta["bubble_gas"]["C"] == C


def test_the_bubble_pressure_changes_the_run_after_the_cut(cuts):
    a, b = cuts["polytropic"], cuts["constant-pressure"]
    assert not (b / "bubble_gas.txt").exists()
    assert json.loads((b / "traction_end.json").read_text())["bubble"] and all(
        x == 0.0 for x in json.loads((b / "traction_end.json").read_text())["bubble"])
    za = np.loadtxt(a / "chains.txt", ndmin=2)
    zb = np.loadtxt(b / "chains.txt", ndmin=2)
    head = (a / "chains.txt").read_text().splitlines()[0][2:].split("\t")
    k = head.index("z_bubble_top")
    n = min(len(za), len(zb))
    assert np.max(np.abs(za[1:n, k] - zb[1:n, k])) > 0.0  # the bubble's pressure acts


def test_a_restart_after_the_cut_continues_the_bubble(cuts, base):  # noqa: F811
    out = cuts["polytropic"]
    dumps = sorted(p for p in (out / "states").glob("state_*.dump")
                   if json.loads(p.with_suffix(".json").read_text()).get("cut") is not None)
    dump = dumps[len(dumps) // 2]
    step = json.loads(dump.with_suffix(".json").read_text())["step"]
    spec = dict(R.SPEC, restart=str(dump), cut=dict(CUT, bubble_model="polytropic"),
                post_cut=True)
    spec["collapse"] = dict(R.SPEC["collapse"], max_steps=9)
    R._run(base / "cut-restart", spec)
    again = base / "cut-restart" / "run"
    a = {int(r["step"]): r for r in _rows(out / "bubble_gas.txt")}
    b = [r for r in _rows(again / "bubble_gas.txt") if int(r["step"]) > step]
    assert b
    for r in b:
        ref = a[int(r["step"])]
        for key in ("V_start", "p_start", "V_end", "W_traction", "W_poly"):
            x, y = float(r[key]), float(ref[key])
            assert abs(x - y) <= 1e-8 * max(abs(y), 1e-300), (key, r["step"])
