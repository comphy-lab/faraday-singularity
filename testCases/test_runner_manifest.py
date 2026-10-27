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


@pytest.mark.parametrize("source, commit, dirty", [
    ("a" * 40, "a" * 40, False), ("b" * 40 + "-dirty", "b" * 40, True), ("unknown", None, None)])
def test_provenance_reads_the_source_as_the_manifest_does(tmp_path, monkeypatch, source, commit,
                                                          dirty):
    import run_case

    case = tmp_path / "case.json"
    case.write_text("{}")
    # the source is read once; the manifest's commit comes from that same reading, so a change
    # of the tree between the two records cannot make them disagree
    readings = iter([source, "c" * 40])
    monkeypatch.setattr(run_case, "component_commit", lambda root: next(readings))
    info = run_case.provenance(case)
    assert info["component_commit"] == commit and info["component_dirty"] is dirty
    assert run_case.manifest_commit(info) == source
