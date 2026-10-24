"""The coupler on saved frames of a passive run against the one-way gas solve of those frames.

``FARADAY_NECKGAS_REPLAY`` names a directory with ``frames.npz`` (the interface of every frame,
``r, z, ur, uz`` concatenated with their lengths ``lens``, and ``t``, ``r_n``, ``z_n``),
``config.json`` (the one-way solve's settings ``cfg`` and material values) and the one-way results
of variants ``<name>.npz`` (``F-A``, ``F-S``, ``F-A-fric1``, ``F-S-fric1``: attached, separated, and
each with laminar friction with slip). The coupler, one-way with the four variants on the one set
of frames, started at the first frame with the given ``z_lo`` and the frames' necks, must give
every row of each to ``1e-10``: the in-loop advance of the gas over a step of the liquid is the
one-way solve's over a frame interval. Without the directory the test is skipped.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity.neckgas import coupler as cp  # noqa: E402

DATA = os.environ.get("FARADAY_NECKGAS_REPLAY")
VARIANTS = {"F-A": {"name": "A"}, "F-S": {"name": "S", "separated": True},
            "F-A-fric1": {"name": "A-f1", "fric": 1.0}, "F-S-fric1": {"name": "S-f1",
                                                                       "separated": True,
                                                                       "fric": 1.0}}
# the rows of the one-way solve and of the coupler, by their names there and here
KEYS = {"p_th": "p_th", "rho_th": "rho_th", "u_th": "u_th", "M_th": "M_th", "mdot_th": "mdot_th",
        "pmax10": "p_max10", "pmin10": "p_min10", "Mmax20": "M_max20", "m_below": "m_below",
        "V_below": "V_g", "p_bz": "p_bz", "m_bz": "m_bz", "p_pocket": "p_pk",
        "m_total": "m_total", "N": "N", "t": "time", "r_n": "r_n", "z_n": "z_n"}
TOL = 1e-10


@pytest.mark.skipif(not DATA, reason="FARADAY_NECKGAS_REPLAY not set")
def test_the_coupler_replays_the_one_way_solve():
    d = Path(DATA)
    fr = np.load(d / "frames.npz")
    cfg = json.loads((d / "config.json").read_text())
    c = cfg["cfg"]
    assert (cfg["D"], cfg["P0"], cfg["MU_G"], cfg["LAMBDA0"], cfg["GAMMA"]) == (
        cp.DENSITY_RATIO, cp.P0, cp.MU, cp.LAMBDA0, cp.GAMMA)
    s = cp.GasSettings.from_dict({
        "z_hi": c["z_hi"], "z_lo": c["z_lo"], "neck_window": [4.62, 4.76],
        "coupling": "one-way", "variants": list(VARIANTS.values()), "r_on": 1.0,
        "cells_per_neck": float(c["Nr"]), "half_width": c["hw"], "dz_max_lo": c["dz_max_lo"],
        "dz_max_hi": c["dz_max_hi"], "dz_cap": c["dz_cap"], "cfl": c["cfl"],
        "wall_frac": c["wall_frac"], "drift_regrid": None, "init": c["init"],
        "gamma": c["gamma"], "thread_length": 1e6})  # the passive neck is a thread by 30 r_n
    off = np.r_[0, np.cumsum(fr["lens"])]

    def chain(k):
        sl = slice(off[k], off[k + 1])
        return fr["r"][sl], fr["z"][sl], fr["ur"][sl], fr["uz"][sl]

    t, r_n, z_n = fr["t"], fr["r_n"], fr["z_n"]
    gas = cp.NeckGas(s)
    rows = []
    for k in range(len(t) - 1):
        gas.before_step(float(t[k]), chain(k), step=k, neck=(r_n[k], z_n[k]))
        rows += gas.after_step(float(t[k + 1]), chain(k + 1), step=k + 1,
                               neck=(r_n[k + 1], z_n[k + 1]))
        assert gas.stop is None, gas.stop
    worst = {}
    for case, spec in VARIANTS.items():
        ref = np.load(d / f"{case}.npz")
        mine = [r for r in rows if r["variant"] == spec["name"]]
        assert len(mine) == len(ref["t"]) == len(t)
        for a, b in KEYS.items():
            x = np.array([r[b] for r in mine], dtype=float)
            y = np.asarray(ref[a], dtype=float)
            scale = np.maximum(np.abs(y), 1e-3 * np.max(np.abs(y)))
            err = float(np.max(np.abs(x - y) / scale))
            worst[(case, a)] = err
            assert err <= TOL, (case, a, err)
        assert all(abs(r["mass_error"]) < 1e-12 for r in mine)
        assert all(abs(r["energy_error"]) < 1e-10 for r in mine)
    out = os.environ.get("FARADAY_NECKGAS_REPLAY_OUT")
    if out:
        a = [r for r in rows if r["variant"] == "A"]
        thread = {f"{x:.0e}": next((r["thread_over_r_n"] for r in a if r["r_n"] <= x), None)
                  for x in (1e-2, 3e-3, 1e-3, 3e-4, 2e-4, 1e-4)}
        Path(out).write_text(json.dumps({"worst_relative_error": {
            f"{k[0]}:{k[1]}": v for k, v in worst.items()}, "thread_over_r_n_A": thread,
            "regrids": gas.regrids, "rows": len(rows)}, indent=1, default=float))
