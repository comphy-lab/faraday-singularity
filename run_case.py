#!/usr/bin/env python3
"""Run one axisymmetric Faraday case described by a ``pyoomph-case-v1`` JSON file.

Example::

    .venv-pyoomph-f912ac10-py313/bin/python run_case.py \
        verificationCases/free-decay/free-decay-h0.025-n100.json --outdir /path/to/run

Times in the case file are in forcing periods ``2 pi/Omega``. A case with a ``collapse`` block runs
the collapse driver (``src/faraday_singularity/driver.py``) up to ``completion.t_max`` or until it
stops at its conservative neck-radius or fold-gap cutoff or, if the case sets a finite
``collapse.post_birth_aspect``, once a born jet is unambiguous, with the fixed step of
``time_stepping`` as its largest step. A
Newton failure that the driver classifies as the approach to a pinch-off or to the singularity is
such a stop, and the run is ``completed``; the stop is recorded under ``collapse.stop`` in
``run_status.json``. Any other failure marks the run ``failed``. Unrecognised options are left for
pyoomph's own command-line parser (e.g. ``--omp 4``). Each output directory receives
``parameters.json`` (inputs, dimensionless groups, numerical settings and provenance),
``run_status.json``, ``diagnostics.txt`` (axis height and velocity, mode amplitude, volume, area,
kinetic energy, dissipation and remesh count after every step), pyoomph's per-step integrals and
free-surface snapshots (``liquid__interface/``).

Three keys of the ``collapse`` block are not driver settings. ``keep_stop_state`` (default false)
keeps the state at a neck-radius stop as ``states/stop_<step>.dump``, outside the rotation of dumps:
the landing state of a reconnection. ``landing``, a list of cut radii, keeps for each radius
``delta`` the first state whose neck has closed to ``delta/1.05`` as ``states/landing_<delta>.dump``
(``driver.CollapseDriver``); a radius the run cannot reach before its neck-radius stop, below 1.05
times ``max(neck_stop, neck_stop_elements h_floor)``, is refused before any output. A ``reconnect``
block (``reconnect.ReconnectSettings``) builds the problem with the reconnection handler and the
pressure-impulse projection, both inert until a cut, but compiled into it: the computed trajectory
then differs from that of the ordinary case at round-off (deterministically; from the third step,
by about 1e-14 relative, in the tests), and the step control carries the difference on. Two runs of
one case in fresh processes also differ by about 1e-14 unless MKL, whose threaded solve is the
source, runs on one thread (``MKL_NUM_THREADS=1``; ``OMP_NUM_THREADS`` alone does not do it); with
it they have agreed to the byte. The block makes the cut only at a restart that adds or changes it
(below). Without these keys the problem, the settings and every output are those of an ordinary case: the
tests run the driver of this commit against that of commit ``9e0d8a1`` and compare every output.

A restart (``--restart-from``) continues the run of the same case. It may also continue it under a
derived case, one that differs from the parent's only in ``case_id``, the description
``completion.value``, the collapse keys ``keep_stop_state``, ``landing``, ``neck_stop``,
``state_dt`` and ``state_keep``, and ``collapse.reconnect.tau_end`` (``DERIVED_KEYS``: what the run
keeps and where it stops, not the equations it solves): the parent's case is read from its
``parameters.json``, whose recorded hash must be the manifest's, and the restart manifest records
the parent's case hash and the keys that differ. This is how landing states are made from an
existing run. The restart has one of three modes (``restart_admission``), recorded as
``restart_mode`` in the manifest and in ``parameters.json``:

* ``ordinary``: as above; a reconnect block, if any, must be the parent's own (``tau_end`` aside).
* ``cut``: the derived case also adds or changes the ``collapse.reconnect`` block
  (``CUT_DERIVED_KEYS``), and the state is a landing state from before any cut (a
  ``landing_<delta>`` dump, or the ``stop_<step>`` dump of a neck-radius stop) whose recorded neck
  radius puts ``delta`` within the cut's landing window; both are checked before any output. The
  ``precut_<delta>`` dump of a cut run is a landing state too: a restart from it under any case with
  a reconnect block, the cut's own included, is a cut. At its start the run cuts the neck
  (``surgery.cut``: ``cut_diagnostics.json``, the kept ``precut`` and ``postcut`` dumps) and
  continues under the collapse driver in its post-cut mode: the stops before a second change of
  topology on both chains, a step that follows the tips and jets of both chains and the capillary
  time of the smallest tip, ``tau = t - t_cut`` and ``tau_p = t - t_p`` in every output,
  ``chains.txt`` and the profiles of both chains, rotating dumps at least each time ``tau`` doubles
  and a dump of the state it ends on (``driver.py``). A refused or failed cut, whatever failed in
  it, ends the run as ``failed``, with the record and the stage reached in ``cut_diagnostics.json``
  and under ``cut`` in ``run_status.json``, which names the ``precut`` dump once it was written;
  nothing is loaded back into the problem.
* ``post-cut``: a state written after the cut (the ``postcut`` dump, or a rotating dump of a run
  after the cut, the one it ended on included), under the case of that cut or one derived from it
  in ``DERIVED_KEYS``; the case's reconnect settings must be those the cut recorded but for
  ``tau_end``, which it may raise (or remove) to go on after a run that ended there, and the state's
  time after the cut must precede it. The driver restores the cut (``surgery.restore_cut``) and
  continues in the post-cut mode.

In both modes after a cut the case's ``landing`` and ``keep_stop_state`` (inherited from the landing
run) are inactive, as ``parameters.json`` records (``inactive_after_cut``): one cut per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _shared_memory_btl() -> str | None:
    """Open MPI's shared-memory transport: ``vader`` before release 5, ``sm`` from 5 on."""
    exe = shutil.which("ompi_info")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Open MPI v(\d+)", out)
    if match is None:
        return None
    return "self,vader" if int(match.group(1)) < 5 else "self,sm"


# Runs are serial: the MPI-enabled pyoomph build only needs the self and shared-memory transports.
# Restricting them avoids probing network transports during singleton MPI initialisation.
if "OMPI_MCA_btl" not in os.environ and (_btl := _shared_memory_btl()) is not None:
    os.environ["OMPI_MCA_btl"] = _btl

from faraday_singularity.driver import CollapseDriver, CollapseSettings, check_landing  # noqa: E402
from faraday_singularity.cases import validate_case  # noqa: E402
from faraday_singularity.mesh import MeshSettings  # noqa: E402
from faraday_singularity.parameters import CaseParameters, nondimensionalise  # noqa: E402
from faraday_singularity.provenance import create_run_manifest, pyoomph_revision  # noqa: E402
from faraday_singularity.reconnect import (CONTINUATION_KEYS, ReconnectSettings,  # noqa: E402
                                           split_collapse_block, tau_end_reached)

MESH_KEYS = ("h_interface", "h_axis", "h_max", "grading", "interface_points")
CASE_FIELDS_USED = ("schema", "case_id", "geometry.kind", "time_stepping.scheme",
                    "time_stepping.steps_per_forcing_period", "output.cadence_unit",
                    "output.cadence", "output.write_states", "completion.kind",
                    "completion.unit", "completion.t_max", "completion.value",
                    "physics", "mesh.remeshing", "tolerances.newton", "collapse") + tuple(
                        "mesh." + key for key in MESH_KEYS)


def case_has_field(case: dict, field: str) -> bool:
    value = case
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


# Keys a case derived from a parent run may change at a restart: the name, a description the runner
# does not read, and the collapse keys that decide what the run keeps (the kept states, the cadence
# and number of rotating dumps, which write the state and do not change it) and where it stops, not
# the equations it solves. Of the reconnect block only tau_end, which decides where a run after its
# cut ends (reconnect.CONTINUATION_KEYS): adding, dropping or changing the block otherwise compiles
# or removes the projection and the handler, whose trajectory then departs from the ordinary one at
# round-off, from the third step in the tests (a cut, which changes it outright, is another matter).
DERIVED_KEYS = ("case_id", "completion.value", "collapse.keep_stop_state", "collapse.landing",
                "collapse.neck_stop", "collapse.state_dt", "collapse.state_keep") + tuple(
                    "collapse.reconnect." + k for k in CONTINUATION_KEYS)
# A restart that cuts the neck of a landing state at its start: its case adds (or changes) the
# reconnect block, which governs only what follows the cut. The reconnection prototype cuts the
# state it loads too, so its case may also add or change that block.
CUT_DERIVED_KEYS = DERIVED_KEYS + ("collapse.reconnect",)
PROTOTYPE_DERIVED_KEYS = CUT_DERIVED_KEYS
# Restart modes: continue a run before any cut; cut at the start, then continue after the cut;
# continue a run from a state written after its cut.
ORDINARY, CUT, POST_CUT = "ordinary", "cut", "post-cut"


def case_differences(a, b, prefix: str = "") -> list[str]:
    """Dotted keys at which two case objects differ (a key in one only counts once)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(name)
            else:
                out += case_differences(a[key], b[key], name)
        return out
    return [] if a == b else [prefix]


def derived_case_keys(parent_case: dict, case: dict, allowed=DERIVED_KEYS) -> list[str]:
    """The keys in which ``case`` differs from ``parent_case``; refuses any outside ``allowed``."""
    diffs = case_differences(parent_case, case)
    foreign = [k for k in diffs if not any(k == d or k.startswith(d + ".") for d in allowed)]
    if foreign:
        raise SystemExit("restart parent belongs to a different case: it differs in "
                         + ", ".join(foreign))
    return diffs


def recorded_parent_case(parent: Path, case_sha256: str) -> dict:
    """The case a parent run recorded in its ``parameters.json``, if that file names the hash."""
    path = parent / "parameters.json"
    try:
        recorded = json.loads(path.read_text())
    except (OSError, ValueError):
        recorded = None
    if not isinstance(recorded, dict) or not isinstance(recorded.get("case"), dict) or \
            recorded.get("provenance", {}).get("case_sha256") != case_sha256:
        raise SystemExit("restart parent belongs to a different case, and its parameters.json "
                         "does not record that case to compare with")
    return recorded["case"]


def restart_record(state: Path, case: dict, case_path: Path, *, legacy: bool = False,
                   derived_keys=DERIVED_KEYS, post_cut: bool = False) -> dict:
    """Validate restart custody before output, preserving explicitly imported legacy lineage.

    A parent with a normalised manifest may have run the same case or one from which ``case`` is
    derived in ``derived_keys``. A state written after a cut is refused unless ``post_cut``: only
    this runner continues one (:func:`restart_admission`), the prototype cuts states before any
    cut."""
    state = state.resolve()
    if not state.is_file() or not state.with_suffix(".json").is_file():
        raise SystemExit("restart needs both its state dump and driver metadata")
    try:
        meta = json.loads(state.with_suffix(".json").read_text())
    except ValueError:
        meta = None
    if isinstance(meta, dict) and meta.get("cut") is not None and not post_cut:
        raise SystemExit("the state was written after a cut: one cut per run, so this entry point "
                         "takes a state from before any cut; continue a post-cut state with "
                         "simulationCases/run_faraday.py and the case of the cut")
    parent = state.parent.parent
    manifest_path = parent / "run-manifest.json"
    case_hash = hashlib.sha256(case_path.read_bytes()).hexdigest()
    if legacy:
        if manifest_path.exists():
            raise SystemExit("use --restart-from for a run with a normalised manifest")
        recorded_path = parent / "parameters.json"
        if not recorded_path.is_file():
            raise SystemExit("legacy restart has no saved parameters")
        recorded = json.loads(recorded_path.read_text())
        source = recorded.get("provenance", {})
        if source.get("component_dirty") is not False or not re.fullmatch(
                r"[0-9a-f]{40}", str(source.get("component_commit", ""))):
            raise SystemExit("legacy restart has no clean recorded source")
        case_file = recorded.get("case_file")
        if not isinstance(case_file, str) or not case_file:
            raise SystemExit("legacy restart has no recorded case file")
        original = Path(case_file)
        if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != source.get("case_sha256"):
            raise SystemExit("legacy restart case does not match its recorded hash")
        if recorded.get("case", {}).get("physics") != case.get("physics") or recorded.get("case", {}).get("geometry") != case.get("geometry"):
            raise SystemExit("legacy restart changes physics or geometry")
        parent_format = "legacy-parameters-v1"
        parent_commit = source["component_commit"]
    else:
        if not manifest_path.is_file():
            raise SystemExit("restart has no parent run-manifest.json; legacy imports need --legacy-restart-from")
        recorded_path = manifest_path
        recorded = json.loads(manifest_path.read_text())
        parent_case_hash = str(recorded.get("case_sha256", ""))
        derived = None
        if parent_case_hash != case_hash:
            derived = derived_case_keys(recorded_parent_case(parent, parent_case_hash), case,
                                        derived_keys)
        if recorded.get("schema") != "pyoomph-runner-manifest-v1" or not re.fullmatch(
                r"[0-9a-f]{40}", str(recorded.get("component_commit", ""))):
            raise SystemExit("restart parent needs a normalised manifest and a clean source commit")
        parent_format = "pyoomph-runner-manifest-v1"
        parent_commit = recorded["component_commit"]
    out = {"path": str(state), "sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
           "metadata_sha256": hashlib.sha256(state.with_suffix(".json").read_bytes()).hexdigest(),
           "parent_record": {"path": str(recorded_path),
                             "sha256": hashlib.sha256(recorded_path.read_bytes()).hexdigest(),
                             "format": parent_format, "component_commit": parent_commit}}
    if not legacy and derived is not None:  # a restart of the same case records what it did
        out["parent_record"].update(case_sha256=parent_case_hash, derived_keys=derived)
    return out


def _derives_reconnect(record: dict) -> bool:
    """Whether the case adds, drops or changes the parent's reconnect block in more than the keys a
    continuation may change (``CONTINUATION_KEYS``): a cut."""
    keys = record["parent_record"].get("derived_keys", [])
    allowed = {"collapse.reconnect." + k for k in CONTINUATION_KEYS}
    return any((k == "collapse.reconnect" or k.startswith("collapse.reconnect."))
               and k not in allowed for k in keys)


def landing_neck(meta: dict) -> float | None:
    """The neck radius recorded with a landing state: a ``landing_<delta>`` dump, a ``stop_<step>``
    dump of a neck-radius stop, or the ``precut_<delta>`` dump of a cut (the state it started from);
    None for any other dump."""
    landing, stop, neck = meta.get("landing"), meta.get("stop"), meta.get("neck")
    if isinstance(landing, dict):
        return landing.get("r_neck")
    if isinstance(stop, dict) and stop.get("trigger") == "neck-radius":
        return stop.get("r_neck")
    if meta.get("cut_stage") == "precut" and isinstance(neck, dict):
        return neck.get("r_neck")
    return None


def _check_cut_landing(meta: dict, reconnect) -> None:
    """Refuse a cut of a state that is no landing state, or whose neck puts ``delta`` outside the
    cut's landing window."""
    r_n = landing_neck(meta)
    if not isinstance(r_n, (int, float)) or isinstance(r_n, bool) or not r_n > 0.0:
        raise SystemExit("a cut starts from a landing state, states/landing_<delta>.dump, the "
                         "states/stop_<step>.dump of a neck-radius stop or the "
                         "states/precut_<delta>.dump of a cut, with its neck radius")
    ratio = reconnect.delta / r_n
    if not reconnect.min_landing_ratio <= ratio <= reconnect.max_landing_ratio:
        raise SystemExit(
            f"collapse.reconnect.delta {reconnect.delta:.6g} is {ratio:.4g} times the neck radius "
            f"{r_n:.6g} of the landing state; a cut needs {reconnect.min_landing_ratio} to "
            f"{reconnect.max_landing_ratio}: restart from the landing state of this delta")


def _check_post_cut(meta: dict, reconnect) -> None:
    """Refuse to continue a state written after a cut under other reconnect settings than the cut's
    (but for ``CONTINUATION_KEYS``), or once its time after the cut has reached ``tau_end``."""
    cut = meta["cut"]
    if not reconnect.continues_cut(cut.get("settings")):
        raise SystemExit("the case's collapse.reconnect differs from the settings of the cut that "
                         "wrote the state; a continuation may change only "
                         + ", ".join(f"collapse.reconnect.{k}" for k in CONTINUATION_KEYS))
    t, t_cut = meta.get("time"), cut.get("t_cut") if isinstance(cut, dict) else None
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
           for x in (t, t_cut)) and tau_end_reached(t - t_cut, reconnect.tau_end):
        raise SystemExit(f"the state lies {t - t_cut:.6g} after its cut, at the case's "
                         f"collapse.reconnect.tau_end {reconnect.tau_end:.6g}, where its run "
                         "stops: raise tau_end (or remove it) to continue")


def restart_admission(state: Path, case: dict, case_path: Path, reconnect, *,
                      legacy: bool = False) -> tuple[dict, str]:
    """The custody record of a restart and its mode, checked before any output.

    * ``post-cut``: the state was written after a cut (the ``postcut`` dump or a rotating dump of a
      run after a cut). The case must hold the reconnect block of that cut (its settings equal
      those the cut recorded but for ``tau_end``, which the case may raise or remove) and be the
      parent's or derived from it in ``DERIVED_KEYS``; the state's time after the cut must precede
      the case's ``tau_end``.
    * ``cut``: a state from before any cut, under a case that adds or changes the reconnect block
      (``CUT_DERIVED_KEYS``): the run cuts the neck at its start. The state must be a landing
      state, whose recorded neck radius ``r_n`` puts ``delta/r_n`` within the cut's
      ``[min_landing_ratio, max_landing_ratio]``. The ``precut`` dump of a cut run is such a state:
      under any case with a reconnect block (the cut's own, to make the same cut again) a restart
      from it is a cut, never an ordinary restart.
    * ``ordinary``: any other restart, with ``DERIVED_KEYS``; with a reconnect block the parent had
      too, the run goes on before any cut.

    Legacy imports (``legacy``) are ordinary restarts of cases without a reconnect block."""
    try:
        meta = json.loads(state.resolve().with_suffix(".json").read_text())
    except (OSError, ValueError):
        meta = None
    meta = meta if isinstance(meta, dict) else {}
    if legacy:
        if reconnect is not None or meta.get("cut") is not None:
            raise SystemExit("a case with collapse.reconnect, or a state written after a cut, "
                             "restarts only with --restart-from from a normalised run")
        return restart_record(state, case, case_path, legacy=True), ORDINARY
    if meta.get("cut") is not None:
        if reconnect is None:
            raise SystemExit("the state was written after a cut; continuing it needs the case of "
                             "that cut, with its collapse.reconnect block")
        _check_post_cut(meta, reconnect)
        return restart_record(state, case, case_path, post_cut=True), POST_CUT
    if meta.get("cut_stage") == "precut":
        # the state a cut started from: cut it again, never continue it before a cut
        if reconnect is None:
            raise SystemExit("the state is the precut state of a cut, which only a cut continues: "
                             "restart it under a case with a collapse.reconnect block")
        record = restart_record(state, case, case_path, derived_keys=CUT_DERIVED_KEYS)
        _check_cut_landing(meta, reconnect)
        return record, CUT
    if reconnect is None:
        return restart_record(state, case, case_path), ORDINARY
    record = restart_record(state, case, case_path, derived_keys=CUT_DERIVED_KEYS)
    if not _derives_reconnect(record):  # the parent's own reconnect block: no cut
        return record, ORDINARY
    _check_cut_landing(meta, reconnect)
    return record, CUT


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    if case.get("schema") != "pyoomph-case-v1":
        raise SystemExit(f"{path}: not a pyoomph-case-v1 file")
    expected = {
        ("geometry", "kind"): "axisymmetric",
        ("time_stepping", "scheme"): "BDF2",
        ("output", "cadence_unit"): "forcing periods",
        ("completion", "unit"): "forcing periods",
    }
    # linear cases run to a fixed time; collapse cases run to an event (stop before pinch-off)
    expected[("completion", "kind")] = "event" if "collapse" in case else "time"
    for (block, key), value in expected.items():
        if case.get(block, {}).get(key) != value:
            raise SystemExit(f"{path}: {block}.{key} must be {value!r}")
    steps = case["time_stepping"].get("steps_per_forcing_period")
    if not isinstance(steps, int) or steps < 1:
        raise SystemExit(f"{path}: time_stepping.steps_per_forcing_period must be a positive integer")
    if "collapse" in case and not isinstance(case["completion"].get("t_max"), (int, float)):
        raise SystemExit(f"{path}: collapse cases need completion.t_max (forcing periods)")
    validate_case(case)
    return case


class Terminated(BaseException):
    """Raised from the SIGTERM handler so that the run status records the interruption. Like
    ``KeyboardInterrupt`` it is no ``Exception``, so that nothing that handles failures (a cut that
    records a failed outcome, say) takes it for one."""


def _raise_terminated(signum, _frame):
    raise Terminated(f"signal {signum}")


def cut_failure_status(error) -> dict:
    """The ``cut`` block of ``run_status.json`` for a refused or failed cut (``surgery.CutRefused``,
    ``surgery.CutFailed``): the outcome and the stage reached, from the record the exception
    carries, and the ``precut`` dump once it was written (refusals from the plan on, every
    failure)."""
    record = getattr(error, "record", None) or {}
    precut = getattr(error, "precut", None) or (record.get("precut_state") or {}).get("path")
    return {"outcome": record.get("outcome"), "failed_at": record.get("failed_at"),
            "diagnostics": "cut_diagnostics.json", "precut": precut}


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def provenance(case_path: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                                 check=False, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    info = {"case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "component_commit": commit,
            # None when the state cannot be determined; untracked files count as dirty
            "component_dirty": None if status is None else bool(status)}
    try:
        from importlib.metadata import distribution

        direct = distribution("pyoomph").read_text("direct_url.json")
        info["pyoomph"] = json.loads(direct) if direct else None
    except Exception as exc:  # provenance must not stop a run; record why it is missing
        info["pyoomph"] = f"unavailable: {type(exc).__name__}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--outdir", "--out", dest="outdir", type=Path, required=True)
    ap.add_argument("--periods", type=float, help="override the end time, in forcing periods")
    ap.add_argument("--mesh-output", action="store_true",
                    help="also write bulk VTU files at every output time")
    restarts = ap.add_mutually_exclusive_group()
    restarts.add_argument("--restart", "--restart-from", dest="restart", type=Path,
                    help="collapse cases: resume from a state dump written by the collapse driver")
    restarts.add_argument("--legacy-restart-from", type=Path,
                         help="explicitly import a legacy state with verified saved parameters")
    ap.add_argument("--check", action="store_true", help="validate without creating output")
    args, _ = ap.parse_known_args()

    case = load_case(args.case)
    if not isinstance(case.get("case_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", case["case_id"]):
        raise SystemExit("case_id must be a valid case identifier")
    tol = float(case["tolerances"]["newton"])
    write_states = case["output"]["write_states"]
    if not math.isfinite(tol) or tol <= 0 or not isinstance(write_states, bool):
        raise SystemExit("Newton tolerance must be positive and finite; write_states must be boolean")
    params = CaseParameters.from_dict(case["physics"])
    nd = nondimensionalise(params)
    mesh = MeshSettings(**{k: case["mesh"][k] for k in MESH_KEYS if k in case["mesh"]})
    remesh = bool(case["mesh"].get("remeshing", True))

    ts = case["time_stepping"]
    period = nd.forcing_period
    collapse = "collapse" in case
    if args.periods is not None:
        periods = args.periods
    else:
        periods = float(case["completion"]["t_max" if collapse else "value"])
    dt = period / ts["steps_per_forcing_period"]  # fixed step, or the largest step of a collapse
    dt_out = period * float(case["output"]["cadence"])
    t_end = periods * period
    settings = None
    reconnect = None
    keep_stop_state = False
    landing = ()
    if collapse:
        try:
            block, recon_block, keep_stop_state, landing = split_collapse_block(case["collapse"])
            if recon_block is not None:
                reconnect = ReconnectSettings.from_dict(recon_block)
        except ValueError as exc:
            raise SystemExit(f"{args.case}: {exc}")
        if "snapshot_dt_periods" in block:
            block["snapshot_dt"] = block.pop("snapshot_dt_periods") * period
        settings = CollapseSettings.from_dict(block, t_end=t_end, dt_max=dt)
        try:
            check_landing(landing, settings)
        except ValueError as exc:
            raise SystemExit(f"{args.case}: {exc}")

    if not math.isfinite(periods) or periods <= 0 or not math.isfinite(dt_out) or dt_out <= 0:
        raise SystemExit("completion time and output cadence must be positive and finite")
    if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())):
        raise SystemExit(f"{args.outdir}: output directory is not empty")
    source_state = args.restart or args.legacy_restart_from
    parent_record = None
    restart_meta = None
    restart_mode = None
    inactive = {}
    if source_state is not None:
        if not collapse:
            raise SystemExit("restarts require a collapse case")
        parent_record, restart_mode = restart_admission(
            source_state, case, args.case, reconnect, legacy=args.legacy_restart_from is not None)
        if restart_mode in (CUT, POST_CUT):
            # one cut per run: what the case keeps for a cut is the landing run's, not this run's
            inactive = {k: v for k, v in (("landing", list(landing)),
                                          ("keep_stop_state", keep_stop_state)) if v}
            landing, keep_stop_state = (), False
        metadata_bytes = source_state.with_suffix(".json").read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != parent_record["metadata_sha256"]:
            raise SystemExit("restart metadata changed during admission")
        restart_meta = json.loads(metadata_bytes)
        saved_time = restart_meta.get("time")
        if isinstance(saved_time, bool) or not isinstance(saved_time, (int, float)) or not math.isfinite(saved_time):
            raise SystemExit("restart metadata must record a finite physical time")
        if saved_time >= t_end - 1e-14:
            raise SystemExit(f"restart time {saved_time:g} ({saved_time / period:g} forcing periods) "
                             f"must precede end time {t_end:g} ({periods:g} forcing periods), "
                             "with physical times in capillary units")
        parent_root = source_state.resolve().parent.parent
        if args.outdir.resolve().is_relative_to(parent_root):
            raise SystemExit("restart output cannot be inside its parent run")
        args.restart = source_state
    if args.check:
        print(json.dumps({"valid": True, "case_id": case["case_id"], "restart": parent_record,
                          "restart_mode": restart_mode}))
        return 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    recorded_provenance = provenance(args.case)
    runner_path = Path(sys.argv[0]).resolve()
    if not runner_path.is_file():
        runner_path = Path(__file__).resolve()
    commit = recorded_provenance["component_commit"] or "unknown"
    if recorded_provenance["component_dirty"] is not False:
        commit += "-dirty"
    manifest = {"schema": "pyoomph-runner-manifest-v1", "case_id": case["case_id"],
                "case_file": str(args.case.resolve()),
                "case_sha256": recorded_provenance["case_sha256"],
                "case_fields_used": [field for field in CASE_FIELDS_USED
                                     if case_has_field(case, field)
                                     and not (collapse and field == "completion.value")
                                     and not (not collapse and field == "completion.t_max")
                                     and not (args.periods is not None and field in {"completion.value", "completion.t_max"})],
                "component_commit": commit, "pyoomph": pyoomph_revision(),
                "argv": list(sys.argv),
                "runner": {"path": str(runner_path),
                           "sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest()},
                "units": "dimensionless; length 1/k; time in capillary units",
                "case_time_units": "completion and snapshot_dt_periods in forcing periods; other collapse time controls in capillary units",
                "effective_time_stepping": {"dt_max": dt, "t_end": t_end, "dt_out": dt_out}}
    if parent_record is not None:
        manifest["restart_from"] = parent_record
        manifest["restart_mode"] = restart_mode
    if reconnect is not None:
        manifest["reconnect"] = reconnect.to_dict()
    create_run_manifest(args.outdir, manifest)
    with (args.outdir / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "start", "wall_time": time.time()}) + "\n")
    info = {
        "case_file": str(args.case.resolve()),
        "case": case,
        "physical": params.to_dict(),
        "dimensionless": nd.to_dict(),
        "mesh": asdict(mesh) | {"remeshing": remesh},
        "time_stepping": {"scheme": "BDF2", "dt": dt, "dt_out": dt_out, "t_end": t_end,
                          "periods": periods, "mode": "collapse" if collapse else "fixed-step"},
        "collapse": asdict(settings) if settings is not None else None,
        "provenance": recorded_provenance,
        "restart": str(args.restart.resolve()) if args.restart is not None else None,
        "restart_mode": restart_mode,
        "argv": sys.argv,
    }
    if inactive:
        info["inactive_after_cut"] = inactive
    if reconnect is not None:
        info["reconnect"] = reconnect.to_dict()
    if keep_stop_state:
        info["keep_stop_state"] = True
    if landing:
        info["landing"] = list(landing)
    write_json(args.outdir / "parameters.json", info)

    from faraday_singularity.problem import FaradayProblem

    signal.signal(signal.SIGTERM, _raise_terminated)
    status = {"state": "running", "started": time.time()}
    write_json(args.outdir / "run_status.json", status)
    problem = None
    cut_error = None
    try:
        extra = {"reconnect": reconnect} if reconnect is not None else {}
        problem = FaradayProblem(nd, mesh=mesh, remesh=remesh, mesh_output=args.mesh_output,
                                 **extra)
        problem.set_output_directory(str(args.outdir))
        problem.write_states = write_states
        with problem:
            problem.initialise()
            # pyoomph's initialisation resets SIGTERM to its default action, which would end the
            # run without recording it in the status
            signal.signal(signal.SIGTERM, _raise_terminated)
            if settings is None:
                problem.start_diagnostics(args.outdir / "diagnostics.txt")
                problem.run(t_end, timestep=dt, outstep=dt_out, newton_solver_tolerance=tol)
            else:
                driver = CollapseDriver(problem, settings, args.outdir,
                                        **({"keep_stop_state": True} if keep_stop_state else {}),
                                        **({"landing": landing} if landing else {}),
                                        **({"reconnect": reconnect}
                                           if restart_mode in (CUT, POST_CUT) else {}))
                try:
                    if args.restart is not None:
                        from pyoomph.output.states import copy_state_file

                        local = args.outdir / "restart" / args.restart.name
                        local.parent.mkdir(exist_ok=True)
                        if hashlib.sha256(args.restart.read_bytes()).hexdigest() != parent_record["sha256"]:
                            raise RuntimeError("restart state changed after admission")
                        if hashlib.sha256(args.restart.with_suffix(".json").read_bytes()).hexdigest() != parent_record["metadata_sha256"]:
                            raise RuntimeError("restart metadata changed after admission")
                        copy_state_file(str(args.restart), str(local))
                        problem.load_state(str(local))
                        driver.restore(restart_meta)
                    if restart_mode == CUT:
                        from faraday_singularity.surgery import CutFailed, CutRefused, cut

                        try:
                            record = cut(problem, reconnect, args.outdir, driver=driver)
                        except (CutRefused, CutFailed) as exc:
                            # the run ends here; after a failure the problem may be half rebuilt,
                            # so nothing is loaded back into it (states/precut_* holds the state)
                            cut_error = exc
                        else:
                            status["cut"] = {"outcome": "cut", "t_cut": record["t_cut"],
                                             "t_p": record["pinch"]["t_p"],
                                             "diagnostics": "cut_diagnostics.json",
                                             "precut": record["precut_state"]["path"],
                                             "postcut": record["postcut_state"]["path"]}
                    if cut_error is None:
                        problem.start_diagnostics(args.outdir / "diagnostics.txt")
                        status["collapse"] = driver.run(tol)
                finally:
                    driver.close()
        status["state"] = "completed"
    except BaseException as exc:  # record failures and interrupts, then re-raise
        interrupted = isinstance(exc, (Terminated, KeyboardInterrupt))
        state = "interrupted" if interrupted or not isinstance(exc, Exception) else "failed"
        status.update(state=state, detail=f"{type(exc).__name__}: {exc}", ended=time.time(),
                      remesh_count=getattr(problem, "remesh_count", None))
        write_json(args.outdir / "run_status.json", status)
        raise
    finally:
        if problem is not None:
            problem.close_diagnostics()
    if cut_error is not None:
        status.update(state="failed", detail=f"{type(cut_error).__name__}: {cut_error}",
                      ended=time.time(), remesh_count=getattr(problem, "remesh_count", None),
                      cut=cut_failure_status(cut_error))
        write_json(args.outdir / "run_status.json", status)
        print(json.dumps(status), file=sys.stderr)
        return 1
    status["ended"] = time.time()
    status["wall_seconds"] = status["ended"] - status["started"]
    status["remesh_count"] = problem.remesh_count
    write_json(args.outdir / "run_status.json", status)
    with (args.outdir / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "completed", "wall_time": time.time(),
                                 "collapse": status.get("collapse")}) + "\n")
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
