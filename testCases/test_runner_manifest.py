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
from faraday_singularity.reconnect import ReconnectSettings


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
    child = _derived(keep_stop_state=True, landing=[1e-3, 5e-4], neck_stop=1e-4)
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    lineage = run_case.restart_record(state, child, path)
    rec = lineage["parent_record"]
    assert rec["case_sha256"] == parent_hash
    assert rec["derived_keys"] == ["case_id", "collapse.keep_stop_state", "collapse.landing",
                                   "collapse.neck_stop"]
    # the same case: the record of an ordinary restart
    same = run_case.restart_record(state, CASE, tmp_path / "parent-case.json")
    assert set(same["parent_record"]) == {"path", "sha256", "format", "component_commit"}


@pytest.mark.parametrize("parent_block, child_block", [(None, {"delta": 1e-3}),
                                                       ({"delta": 1e-3}, None),
                                                       ({"delta": 1e-3}, {"delta": 5e-4})])
def test_restart_refuses_a_derived_case_that_changes_the_reconnect_block(tmp_path, parent_block,
                                                                         child_block):
    """A reconnect block compiles the projection and the handler into the problem, whose trajectory
    then departs from the ordinary one at round-off: a restart may not add, drop or change it.
    The prototype, which builds its own problem with it, may (``PROTOTYPE_DERIVED_KEYS``)."""
    parent = json.loads(json.dumps(CASE))
    if parent_block is not None:
        parent["collapse"]["reconnect"] = parent_block
    state, _ = _parent_run(tmp_path, case=parent)
    child = json.loads(json.dumps(parent))
    child["case_id"] = "sigterm-landing"
    child["collapse"].pop("reconnect", None)
    if child_block is not None:
        child["collapse"]["reconnect"] = child_block
    child["collapse"]["landing"] = [1e-2]
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    with pytest.raises(SystemExit, match="different case: it differs in collapse.reconnect"):
        run_case.restart_record(state, child, path)
    lineage = run_case.restart_record(state, child, path,
                                      derived_keys=run_case.PROTOTYPE_DERIVED_KEYS)
    assert any(k.startswith("collapse.reconnect") for k in lineage["parent_record"]["derived_keys"])


def test_restart_refuses_a_state_written_after_a_cut_without_its_case(tmp_path):
    """Only the runner continues a post-cut state, and only under the case of the cut: the
    prototype (and restart_record without post_cut) refuse it, and so does a case without the
    reconnect block of the cut."""
    state, _ = _parent_run(tmp_path)
    state.with_suffix(".json").write_text(json.dumps({"time": 0.5, "cut": {"delta": 1e-3}}))
    for derived in (run_case.DERIVED_KEYS, run_case.PROTOTYPE_DERIVED_KEYS):
        with pytest.raises(SystemExit, match="one cut per run"):
            run_case.restart_record(state, CASE, tmp_path / "parent-case.json",
                                    derived_keys=derived)
    with pytest.raises(SystemExit, match="needs the case of that cut"):
        run_case.restart_admission(state, CASE, tmp_path / "parent-case.json", None)
    out = tmp_path / "out"
    proc = _prototype(tmp_path / "parent-case.json", "--state", state, "--outdir", out,
                      "--delta-ratio", "1.05", "--check")
    assert proc.returncode != 0 and "one cut per run" in proc.stderr
    assert not out.exists()


RECON = ReconnectSettings(delta=1.05e-3, tau_end=1e-4)


def _write_meta(state, **meta):
    state.with_suffix(".json").write_text(json.dumps({"time": 0.5, **meta}))


def _child_case(tmp_path, reconnect=None, name="child.json", **collapse):
    child = _derived(**collapse)
    if reconnect is not None:
        child["collapse"]["reconnect"] = reconnect
    path = tmp_path / name
    path.write_text(json.dumps(child))
    return child, path


def test_restart_admission_cuts_a_landing_state_under_a_case_that_adds_the_reconnect_block(
        tmp_path):
    state, parent_hash = _parent_run(tmp_path)
    landing = {"delta": 1.05e-3, "r_neck": 1e-3, "z_neck": 4.7, "delta_over_r_neck": 1.05}
    _write_meta(state, landing=landing)
    child, path = _child_case(tmp_path, RECON.to_dict(), landing=[1.05e-3])
    record, mode = run_case.restart_admission(state, child, path, RECON)
    assert mode == run_case.CUT
    assert record["parent_record"]["case_sha256"] == parent_hash
    assert "collapse.reconnect" in record["parent_record"]["derived_keys"]
    # the stop state of a neck-radius stop is a landing state too
    _write_meta(state, stop={"reason": "pinch-off-approach", "trigger": "neck-radius",
                             "r_neck": 1e-3})
    assert run_case.restart_admission(state, child, path, RECON)[1] == run_case.CUT
    # any other dump is not
    _write_meta(state)
    with pytest.raises(SystemExit, match="a cut starts from a landing state"):
        run_case.restart_admission(state, child, path, RECON)
    # a landing state of another cut radius
    _write_meta(state, landing=dict(landing, r_neck=5e-4))
    with pytest.raises(SystemExit, match="2.1 times the neck radius"):
        run_case.restart_admission(state, child, path, RECON)
    # without the block, or with the parent's own block, the restart is an ordinary one
    _write_meta(state, landing=landing)
    plain, plain_path = _child_case(tmp_path, name="plain.json", landing=[1.05e-3])
    assert run_case.restart_admission(state, plain, plain_path, None)[1] == run_case.ORDINARY
    parent = json.loads(json.dumps(CASE))
    parent["collapse"]["reconnect"] = RECON.to_dict()
    state2, _ = _parent_run(tmp_path / "p2", case=parent)
    _write_meta(state2, landing=landing)
    same, same_path = _child_case(tmp_path, RECON.to_dict(), name="same.json", landing=[1e-3])
    assert run_case.restart_admission(state2, same, same_path, RECON)[1] == run_case.ORDINARY
    # a legacy import is never a cut
    with pytest.raises(SystemExit, match="only with --restart-from"):
        run_case.restart_admission(state, child, path, RECON, legacy=True)


def test_restart_admission_continues_a_post_cut_state_under_the_case_of_its_cut(tmp_path):
    cut_case = json.loads(json.dumps(CASE))
    cut_case["collapse"]["reconnect"] = RECON.to_dict()
    state, cut_hash = _parent_run(tmp_path, case=cut_case)
    _write_meta(state, cut={"delta": RECON.delta, "settings": RECON.to_dict()})
    path = tmp_path / "parent-case.json"
    record, mode = run_case.restart_admission(state, cut_case, path, RECON)
    assert mode == run_case.POST_CUT and set(record["parent_record"]) == {
        "path", "sha256", "format", "component_commit"}
    # derived in where it stops: still the cut's continuation
    child = json.loads(json.dumps(cut_case))
    child["case_id"] = "sigterm-more"
    child["collapse"]["neck_stop"] = 1e-4
    child_path = tmp_path / "more.json"
    child_path.write_text(json.dumps(child))
    record, mode = run_case.restart_admission(state, child, child_path, RECON)
    assert mode == run_case.POST_CUT and record["parent_record"]["derived_keys"] == [
        "case_id", "collapse.neck_stop"]
    # other cut settings than those the cut recorded
    other = ReconnectSettings(delta=RECON.delta, tau_end=2e-4)
    child["collapse"]["reconnect"] = other.to_dict()
    child_path.write_text(json.dumps(child))
    with pytest.raises(SystemExit, match="differs from the settings of the cut"):
        run_case.restart_admission(state, child, child_path, other)


def test_runner_check_reports_a_cut_restart_and_refuses_a_foreign_delta(tmp_path):
    state, _ = _parent_run(tmp_path)
    _write_meta(state, landing={"delta": 1.05e-3, "r_neck": 1e-3})
    _, path = _child_case(tmp_path, RECON.to_dict())
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"), str(path),
                           "--out", str(out), "--restart-from", str(state), "--check"],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["restart_mode"] == "cut" and not out.exists()
    _, path = _child_case(tmp_path, dict(RECON.to_dict(), delta=2e-3), name="far.json")
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"), str(path),
                           "--out", str(out), "--restart-from", str(state), "--check"],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode != 0 and "times the neck radius" in proc.stderr and not out.exists()


def test_runner_refuses_landing_radii_below_the_neck_stop_before_output(tmp_path):
    """The run stops at max(neck_stop, neck_stop_elements h_floor) = 3e-4 here; the cut radii 2.5e-4
    and 1.25e-4 would need the neck at 2.38e-4 and 1.19e-4, which the run never reaches."""
    case = json.loads(json.dumps(CASE))
    case["collapse"].update(neck_stop=3e-4, h_floor=2e-5,
                            landing=[1e-3, 5e-4, 2.5e-4, 1.25e-4])
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"), str(path),
                           "--out", str(out), "--check"], capture_output=True, text=True,
                          timeout=120, check=False)
    assert proc.returncode != 0 and "collapse.landing" in proc.stderr
    assert "0.00025, 0.000125" in proc.stderr and not out.exists()
    case["collapse"]["landing"] = [1e-3, 5e-4, 3.2e-4]  # above 1.05 * 3e-4: reached
    path.write_text(json.dumps(case))
    proc = subprocess.run([sys.executable, str(ROOT / "simulationCases/run_faraday.py"), str(path),
                           "--out", str(out), "--check"], capture_output=True, text=True,
                          timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]


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
