"""Output custody and lineage checks that do not build a finite-element problem."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_runner import CASE, ROOT

sys.path.insert(0, str(ROOT))
import run_case


def test_invalid_physics_is_refused_before_output(tmp_path):
    case = json.loads(json.dumps(CASE))
    case["physics"]["fluid"]["Oh"] = float("nan")
    case["physics"]["forcing"]["Gamma"] = -1
    source = tmp_path / "case.json"
    source.write_text(json.dumps(case))
    with pytest.raises(SystemExit) as error:
        run_case.load_case(source)
    assert "fluid.Oh" in str(error.value) and "forcing.Gamma" in str(error.value)


def test_canonical_runner_check_from_foreign_directory_creates_nothing(tmp_path):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"),
                             str(case), "--out", str(out), "--check"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert not out.exists()


def test_check_preserves_occupied_output(tmp_path):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("preserve")
    result = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"),
                             str(case), "--out", str(out), "--check"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "not empty" in result.stderr
    assert sentinel.read_text() == "preserve"


def test_restart_refuses_unrecorded_or_foreign_case(tmp_path):
    states = tmp_path / "parent/states"
    states.mkdir(parents=True)
    state = states / "state.dump"
    state.write_bytes(b"state")
    state.with_suffix(".json").write_text("{}")
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    with pytest.raises(SystemExit, match="no parent run-manifest"):
        run_case.restart_record(state, CASE, case)
    (states.parent / "run-manifest.json").write_text(json.dumps({"case_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="different case"):
        run_case.restart_record(state, CASE, case)


@pytest.mark.parametrize("commit", ["unknown", "a" * 40 + "-dirty"])
def test_modern_restart_refuses_dirty_parent(tmp_path, commit):
    states = tmp_path / "parent/states"
    states.mkdir(parents=True)
    state = states / "state.dump"
    state.write_bytes(b"state")
    state.with_suffix(".json").write_text('{"time": 0}')
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    (states.parent / "run-manifest.json").write_text(json.dumps({
        "schema": "pyoomph-runner-manifest-v1", "component_commit": commit,
        "case_sha256": hashlib.sha256(case.read_bytes()).hexdigest()}))
    with pytest.raises(SystemExit, match="clean source"):
        run_case.restart_record(state, CASE, case)


def test_restart_past_endpoint_is_refused_before_output(tmp_path):
    states = tmp_path / "parent/states"
    states.mkdir(parents=True)
    state = states / "state.dump"
    state.write_bytes(b"state")
    state.with_suffix(".json").write_text('{"time": 99}')
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    (states.parent / "run-manifest.json").write_text(json.dumps({
        "schema": "pyoomph-runner-manifest-v1", "component_commit": "a" * 40,
        "case_sha256": hashlib.sha256(case.read_bytes()).hexdigest()}))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"),
                           str(case), "--out", str(out), "--restart-from", str(state), "--check"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "must precede end time" in proc.stderr
    assert not out.exists()


def test_legacy_import_preserves_explicit_lineage_and_refuses_changed_physics(tmp_path):
    states = tmp_path / "parent/states"
    states.mkdir(parents=True)
    state = states / "state.dump"
    state.write_bytes(b"state")
    state.with_suffix(".json").write_text("{}")
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE))
    saved = {"case": CASE, "case_file": str(case),
             "provenance": {"component_dirty": False, "component_commit": "a" * 40,
                            "case_sha256": hashlib.sha256(case.read_bytes()).hexdigest()}}
    record = states.parent / "parameters.json"
    record.write_text(json.dumps(saved))
    lineage = run_case.restart_record(state, CASE, case, legacy=True)
    assert lineage["parent_record"]["format"] == "legacy-parameters-v1"
    assert lineage["sha256"] == hashlib.sha256(b"state").hexdigest()
    changed = json.loads(json.dumps(CASE))
    changed["physics"]["fluid"]["Oh"] = 0.01
    with pytest.raises(SystemExit, match="changes physics"):
        run_case.restart_record(state, changed, case, legacy=True)
    saved["provenance"]["component_dirty"] = True
    record.write_text(json.dumps(saved))
    with pytest.raises(SystemExit, match="no clean recorded source"):
        run_case.restart_record(state, CASE, case, legacy=True)


# --- derived cases: a restart that changes only what the run keeps and where it stops ---------------


def _parent_run(tmp_path, case=CASE, commit="a" * 40, time=0.5):
    """A parent run directory with a manifest, saved parameters and one state."""
    parent = tmp_path / "parent"
    states = parent / "states"
    states.mkdir(parents=True)
    state = states / "state.dump"
    state.write_bytes(b"state")
    state.with_suffix(".json").write_text(json.dumps({"time": time}))
    case_file = tmp_path / "parent-case.json"
    case_file.write_text(json.dumps(case))
    digest = hashlib.sha256(case_file.read_bytes()).hexdigest()
    (parent / "run-manifest.json").write_text(json.dumps({
        "schema": "pyoomph-runner-manifest-v1", "component_commit": commit,
        "case_sha256": digest}))
    (parent / "parameters.json").write_text(json.dumps({
        "case": case, "provenance": {"case_sha256": digest, "component_commit": commit,
                                     "component_dirty": False}}))
    return state, digest


def _derived(**collapse):
    child = json.loads(json.dumps(CASE))
    child["case_id"] = "sigterm-landing"
    child["collapse"].update(collapse)
    return child


def test_restart_admits_a_case_derived_in_the_kept_states_and_stops(tmp_path):
    state, parent_hash = _parent_run(tmp_path)
    child = _derived(keep_stop_state=True, landing=[1e-3, 5e-4], neck_stop=1e-4,
                     reconnect={"delta": 1e-3})
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    lineage = run_case.restart_record(state, child, path)
    rec = lineage["parent_record"]
    assert rec["case_sha256"] == parent_hash
    assert rec["derived_keys"] == ["case_id", "collapse.keep_stop_state", "collapse.landing",
                                   "collapse.neck_stop", "collapse.reconnect"]
    # the same case: the record of an ordinary restart
    same = run_case.restart_record(state, CASE, tmp_path / "parent-case.json")
    assert set(same["parent_record"]) == {"path", "sha256", "format", "component_commit"}


@pytest.mark.parametrize("change, key", [(("physics", "fluid", "Oh"), "physics.fluid.Oh"),
                                         (("mesh", "h_interface"), "mesh.h_interface"),
                                         (("collapse", "h_floor"), "collapse.h_floor"),
                                         (("time_stepping", "steps_per_forcing_period"),
                                          "time_stepping.steps_per_forcing_period")])
def test_restart_refuses_a_derived_case_that_changes_what_is_computed(tmp_path, change, key):
    state, _ = _parent_run(tmp_path)
    child = _derived(keep_stop_state=True)
    block = child
    for k in change[:-1]:
        block = block[k]
    block[change[-1]] = 0.03 if change[-1] != "steps_per_forcing_period" else 200
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    with pytest.raises(SystemExit, match=f"different case: it differs in {key}"):
        run_case.restart_record(state, child, path)


def test_restart_of_a_derived_case_needs_the_recorded_parent_case(tmp_path):
    state, _ = _parent_run(tmp_path)
    child = _derived(keep_stop_state=True)
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    params = state.parent.parent / "parameters.json"
    saved = json.loads(params.read_text())
    saved["provenance"]["case_sha256"] = "0" * 64  # not the case of the manifest
    params.write_text(json.dumps(saved))
    with pytest.raises(SystemExit, match="does not record that case"):
        run_case.restart_record(state, child, path)


def _prototype(*args):
    return subprocess.run([sys.executable, str(ROOT / "simulationCases" / "reconnect_prototype.py"),
                           *map(str, args)], capture_output=True, text=True, timeout=300,
                          check=False)


def test_prototype_checks_the_lineage_of_its_state(tmp_path):
    state, parent_hash = _parent_run(tmp_path)
    out = tmp_path / "out"
    child = _derived(reconnect={"delta": 1e-3})
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    proc = _prototype(path, "--state", state, "--outdir", out, "--delta-ratio", "1.05", "--check")
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["state"]["parent_record"]["case_sha256"] == parent_hash
    assert report["state"]["parent_record"]["derived_keys"] == ["case_id", "collapse.reconnect"]
    # other physics than the state was computed with
    child["physics"]["forcing"]["Gamma"] = 1.2
    path.write_text(json.dumps(child))
    proc = _prototype(path, "--state", state, "--outdir", out, "--delta-ratio", "1.05", "--check")
    assert proc.returncode != 0 and "differs in physics.forcing.Gamma" in proc.stderr
    assert not out.exists()


@pytest.mark.parametrize("commit", ["a" * 40 + "-dirty", "unknown"])
def test_prototype_refuses_a_state_from_an_unclean_source(tmp_path, commit):
    state, _ = _parent_run(tmp_path, commit=commit)
    out = tmp_path / "out"
    proc = _prototype(tmp_path / "parent-case.json", "--state", state, "--outdir", out,
                      "--delta-ratio", "1.05")
    assert proc.returncode != 0 and "clean source commit" in proc.stderr
    assert not out.exists()


def test_prototype_refuses_a_state_without_a_parent_manifest(tmp_path):
    state, _ = _parent_run(tmp_path)
    (state.parent.parent / "run-manifest.json").unlink()
    out = tmp_path / "out"
    proc = _prototype(tmp_path / "parent-case.json", "--state", state, "--outdir", out,
                      "--delta-ratio", "1.05")
    assert proc.returncode != 0 and "no parent run-manifest" in proc.stderr
    assert not out.exists()
