"""The ``collapse.gas`` block in the runner (``run_case.py``), without a finite-element problem:
it is read before the events block and is no key of the driver's settings, it is checked before
any output (an unknown key, a missing height, an invalid value), it is no derived key (a restart
that adds, drops or changes it is refused as a different case; one that keeps it is admitted), and
it is recorded only where present."""

import hashlib
import json
import subprocess
import sys

import pytest

from test_runner import CASE, ROOT

sys.path.insert(0, str(ROOT))
import run_case  # noqa: E402

GAS = {"z_hi": 5.0, "neck_window": [4.62, 4.76], "coupling": "two-way"}


def _case(tmp_path, gas=GAS, name="case.json", **collapse):
    case = json.loads(json.dumps(CASE))
    if gas is not None:
        case["collapse"]["gas"] = gas
    case["collapse"].update(collapse)
    path = tmp_path / name
    path.write_text(json.dumps(case))
    return case, path


def _check(path, out):
    return subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"),
                           str(path), "--out", str(out), "--check"],
                          capture_output=True, text=True, timeout=120)


def test_a_gas_block_is_checked_before_any_output(tmp_path):
    _, path = _case(tmp_path)
    out = tmp_path / "out"
    proc = _check(path, out)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["valid"] and not out.exists()
    for bad, message in (({**GAS, "drag": 1.0}, "unknown key"),
                         ({"z_hi": 5.0}, "neck_window is required"),
                         ({**GAS, "coupling": "both"}, "coupling must be one of"),
                         ({**GAS, "variants": [{"name": "A"}, {"name": "S"}]},
                          "two-way coupling applies one variant")):
        _, path = _case(tmp_path, gas=bad, name="bad.json")
        proc = _check(path, out)
        assert proc.returncode != 0 and message in proc.stderr, proc.stderr
        assert not out.exists()


def test_the_gas_block_is_no_key_of_the_driver_settings(tmp_path):
    from faraday_singularity.neckgas.coupler import split_gas_block

    rest, block = split_gas_block({"gas": GAS, "neck_stop": 1e-4})
    assert block == GAS and rest == {"neck_stop": 1e-4}
    settings = run_case.CollapseSettings.from_dict(rest, t_end=1.0, dt_max=0.1)
    assert "gas" not in run_case.asdict(settings)
    assert not any(k.startswith("collapse.gas") for k in run_case.DERIVED_KEYS)
    assert not any(k.startswith("collapse.gas") for k in run_case.CUT_EVENTS_DERIVED_KEYS)


def _parent(tmp_path, case):
    parent = tmp_path / "parent"
    states = parent / "states"
    states.mkdir(parents=True)
    state = states / "state.dump"
    state.write_bytes(b"state")
    state.with_suffix(".json").write_text(json.dumps({"time": 0.5}))
    case_file = tmp_path / "parent-case.json"
    case_file.write_text(json.dumps(case))
    digest = hashlib.sha256(case_file.read_bytes()).hexdigest()
    (parent / "run-manifest.json").write_text(json.dumps({
        "schema": "pyoomph-runner-manifest-v1", "component_commit": "a" * 40,
        "case_sha256": digest}))
    (parent / "parameters.json").write_text(json.dumps({
        "case": case, "provenance": {"case_sha256": digest, "component_commit": "a" * 40,
                                     "component_dirty": False}}))
    return state


@pytest.mark.parametrize("parent_gas, child_gas", [(None, GAS), (GAS, None),
                                                   (GAS, {**GAS, "r_on": 0.1})])
def test_a_restart_may_not_add_drop_or_change_the_gas_block(tmp_path, parent_gas, child_gas):
    parent, _ = _case(tmp_path, gas=parent_gas, name="p.json")
    state = _parent(tmp_path, parent)
    child, path = _case(tmp_path, gas=child_gas, name="child.json", neck_stop=1e-4)
    child["case_id"] = "sigterm-landing"
    path.write_text(json.dumps(child))
    with pytest.raises(SystemExit, match="different case: it differs in collapse.gas"):
        run_case.restart_record(state, child, path)


def test_a_restart_that_keeps_the_gas_block_is_admitted(tmp_path):
    parent, _ = _case(tmp_path, name="p.json")
    state = _parent(tmp_path, parent)
    child, path = _case(tmp_path, name="child.json", neck_stop=1e-4, landing=[1e-2])
    child["case_id"] = "sigterm-landing"
    path.write_text(json.dumps(child))
    lineage = run_case.restart_record(state, child, path)
    assert lineage["parent_record"]["derived_keys"] == ["case_id", "collapse.landing",
                                                       "collapse.neck_stop"]


def test_the_driver_takes_the_gas_only_with_its_problem(tmp_path):
    import types

    from faraday_singularity.driver import CollapseDriver, CollapseSettings
    from faraday_singularity.neckgas.coupler import GasSettings
    from faraday_singularity.reconnect import ReconnectSettings

    gas = GasSettings.from_dict(GAS)
    settings = CollapseSettings(t_end=1.0, dt_max=0.1)
    with pytest.raises(ValueError, match="needs a problem made with gas"):
        CollapseDriver(types.SimpleNamespace(gas=False), settings, tmp_path / "a", gas=gas)
    drv = CollapseDriver(types.SimpleNamespace(gas=True), settings, tmp_path / "c", gas=gas)
    assert drv.gas.status == "dormant" and (tmp_path / "c" / "gas.txt").read_text().startswith(
        "# step\ttime")
    drv.close()
    # a driver after a cut takes the block too: the gas restored from a landing state is retired
    # at the cut (reset_after_cut), and one restored after the cut is retired already
    drv = CollapseDriver(types.SimpleNamespace(gas=True), settings, tmp_path / "d", gas=gas,
                         reconnect=ReconnectSettings(delta=1e-3))
    assert drv.gas.status == "dormant"
    drv.close()
