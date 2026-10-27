"""Provenance written into run manifests: the component commit and the pyoomph revision."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

MANIFEST = "run-manifest.json"


def _git(root: Path, *args: str) -> str:
    """Output of a read-only git command in ``root``, ignoring any inherited repository selection."""
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"}}
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   stderr=subprocess.DEVNULL, env=environment).strip()


def component_commit(root: Path) -> str:
    """Commit of the component source at ``root``.

    A source tree materialised from an archive carries its commit in a ``COMMIT`` file;
    otherwise git HEAD, marked ``-dirty`` when tracked files differ from it, and ``unknown``
    when neither is available or ``root`` is not the top of its own work tree.
    """
    commit_file = root / "COMMIT"
    if commit_file.is_file():
        return commit_file.read_text(encoding="utf-8").strip() or "unknown"
    try:
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root.resolve():
            return "unknown"
        commit = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def pyoomph_revision() -> dict[str, str | None]:
    """Version, source repository and commit of the installed pyoomph, from its installation record.

    pyoomph is pinned to a git revision (``pyproject.toml``, ``uv.lock``); the installer records
    that revision in the distribution's ``direct_url.json``. Nothing is guessed: without an
    installation record the version is ``None``, and without the git record the repository
    and commit are ``None``.
    """
    info: dict[str, str | None] = {"version": None, "url": None, "commit": None}
    try:
        dist = importlib.metadata.distribution("pyoomph")
    except importlib.metadata.PackageNotFoundError:
        return info
    info["version"] = dist.version
    record = dist.read_text("direct_url.json")
    if record:
        direct = json.loads(record)
        info["url"] = direct.get("url")
        info["commit"] = (direct.get("vcs_info") or {}).get("commit_id")
    return info


def create_run_manifest(out: Path, manifest: dict) -> Path:
    """Create ``out/run-manifest.json`` exclusively; an existing manifest is never replaced.

    Raises ``FileExistsError`` when the folder already has a manifest.
    """
    path = out / MANIFEST
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path
