"""The suite's thread environment (``conftest.py``): MKL, OpenMP and OpenBLAS on one thread in the
pytest process and in every process it starts, whatever the caller's shell set (a caller's 4
included), so that the tests comparing a run with its restart in a new process compare two
single-thread runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = '''
import json, os, subprocess, sys
names = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
child = subprocess.run([sys.executable, "-c",
                        "import json, os; print(json.dumps({k: os.environ.get(k) for k in %r}))"
                        % (names,)], capture_output=True, text=True, check=True)
print(json.dumps({"parent": {k: os.environ.get(k) for k in names},
                  "child": json.loads(child.stdout)}))
'''


def test_the_suite_runs_on_one_thread_whatever_the_caller_set(tmp_path):
    import json

    from conftest import THREAD_VARIABLES

    # this process, under whatever the shell set
    assert all(os.environ[k] == "1" for k in THREAD_VARIABLES)
    # a pytest session started with every variable at 4: its process and the one it starts
    test = tmp_path / "test_probe.py"
    test.write_text("import json, os, subprocess, sys\n\ndef test_probe():\n"
                    f"    names = {tuple(THREAD_VARIABLES)!r}\n"
                    "    here = {k: os.environ.get(k) for k in names}\n"
                    f"    out = subprocess.run([sys.executable, '-c', {PROBE!r}], "
                    "capture_output=True, text=True, check=True)\n"
                    "    print('PROBE', json.dumps(dict(json.loads(out.stdout), pytest=here)))\n")
    (tmp_path / "conftest.py").write_text((ROOT / "testCases" / "conftest.py").read_text())
    env = dict(os.environ, **{k: "4" for k in THREAD_VARIABLES})
    proc = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
                           str(test)], capture_output=True, text=True, cwd=tmp_path, env=env,
                          timeout=300, check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("PROBE"))
    got = json.loads(line.split(" ", 1)[1])
    one = {k: "1" for k in THREAD_VARIABLES}
    # the pytest process itself, a process it starts, and one that process starts
    assert got["pytest"] == got["parent"] == got["child"] == one
