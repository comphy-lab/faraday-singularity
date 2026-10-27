#!/usr/bin/env python3
"""One row per collapse run: wave height, jet birth, jet speed and stop reason.

Usage::

    python postProcess/collapse_table.py RUN_DIR [RUN_DIR ...] --out TABLE.csv \
        [--per-cycle | --per-episode [--eta-hys 0.02]]

``h_crest`` is the height of the last smooth wave: the largest axis elevation ``z_axis - d`` in
the mode period (two forcing periods) before the jet is born, or before the end of the run when no
jet forms; the axis carries a crest half a mode period before the trough that collapses.
``h_trough`` is the depth of the trough below ``d`` just before the jet is born. ``r_j_birth`` is
the jet-base radius at birth, ``min_r_j`` its smallest value afterwards, ``max_V_j`` the largest
mean jet-base speed with flux into the jet (``V_j > 0``, as ``Q_j > 0``; a base closing in on the
axis carries flux out of it, ``V_j < 0``), ``max_We_j`` the largest ``We_j`` of the same rows and
``max_w_axis`` the largest upward speed of the free surface on the axis. All of them describe the
first jet, over the rows of its collapse cycle that carry a jet base (:func:`jet_rows`; a table
without the column ``cycle`` is one cycle). ``n_jets`` counts the cycles with a jet;
``--per-episode`` pairs the wave height and the jet speed of each. Runs that are still going are
marked ``running``.

With ``--per-cycle`` (runs that record the collapse cycle), each collapse cycle of a run under
sustained forcing becomes its own row: the crest height on the axis in the mode period before the
cycle's cavity forms, the largest apex curvature while the cavity collapses, and the jet-base radius
at birth and largest jet speed ``V_j > 0`` of that cycle's jet rows (blank when no jet is born).
One forced run therefore samples several wave heights, but only in rows with ``crest_jet`` 0:
``crest_jet`` is 1 when a jet stood on the axis (``jet_born``) in that mode period, as the jet of
the previous cycle does, and ``h_crest`` is then the height of its tip.

With ``--per-episode``, each trough episode of the axis elevation ``eta = z_axis - d`` becomes its
own row (``src/faraday_singularity/episodes.py``: a trough starts when ``eta`` falls below
``-eta_hys`` after a crest, which starts when ``eta`` rises above ``+eta_hys``). This table carries
the wave height ``h`` used for the dependence of the collapse on the wave: ``h_crest``, the crest
height on the axis above the mean level, the largest ``eta`` over the crest episode just before the
trough, at ``t_crest``, but only in rows with ``crest_jet`` 0. A jet launched from a trough rises
into the crest after it, and while it stands the axis elevation is the height of its tip:
``crest_jet`` is 1 when a jet stood on the axis (``jet_born``) at any row of the crest episode
before the trough, and ``h_crest`` is then not a wave height. It is taken from the trough event or,
for runs that predate it, from the rows of that crest, and is NaN when it cannot be known (no jet
in the rows of a crest that began before the record). Unlike a cycle, an episode does not need a
jet, so collapses that launch no jet are separated too. When the table records the collapse cycle,
each trough episode is split by cycle (column ``cycle``): a jet that has gone and a new cavity in
the same trough give one row each. Each row also gives the lowest apex height ``min_z_axis`` over
its rows of the trough, and the largest positive apex curvature (a trough on the axis) with its
time over the same rows before its jet is born. A jet launched from the trough rises into the crest
episode that follows it, so the remaining columns cover the trough and that crest, within the
cycle: whether and when a jet is born, with ``r_j`` at birth, the largest jet speed ``V_j > 0`` of
that jet over its rows with a jet base, from its birth on, the largest upward speed of the axis
``max_w_axis``, the narrowest neck, and the stop reason if the run ended in either episode. A cycle
whose jet was born in an earlier episode and still stands in the trough gives a row without a birth
(and ``crest_jet`` 1). A birth is a row where ``jet_born`` turns on; tables without the column
``cycle`` keep it on after the first birth, so later births in them are not seen, and every crest
after it has ``crest_jet`` 1; their rows cover whole episodes, the apex curvature too.
Runs that record the column ``episode`` are read as recorded (``source`` ``recorded``); older runs
are segmented again from ``z_axis`` and the depth in ``parameters.json`` with the same tracker and
hysteresis (``reconstructed``). A trough that began before the first row of a restarted run is
``partial``: its other columns cover only the rows after the restart. Its ``h_crest`` is still known
when the run records its episodes (the restart carries the crest before it), and is NaN when no
complete crest precedes the trough.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faraday_singularity.episodes import ETA_HYS, segment

FIELDS = ("case_id", "Gamma", "Oh", "a0", "state", "stop", "t_end", "jet_born", "n_jets",
          "t_birth", "h_crest", "h_trough", "r_j_birth", "min_r_j", "max_V_j", "max_w_axis",
          "max_We_j", "min_r_neck", "remeshes", "volume_drift")


def read_table(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        header = fh.readline().lstrip("#").strip().split("\t")
        if not any(line.strip() and not line.startswith("#") for line in fh):
            return {name: np.empty(0) for name in header}
    data = np.loadtxt(path, comments="#", ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def jet_rows(tab: dict, rows: np.ndarray, events: list[dict]) -> np.ndarray:
    """Rows of the cycle ``rows`` that carry a jet base.

    The driver writes the step at which the jet has gone, the axis being the lowest point again,
    with ``jet_born`` still set before it starts the next cycle. That row has ``r_j = 0``, or the
    radius of a bump that has flattened, so rows with ``r_j <= 0`` are dropped, and so is the last
    row of a cycle that a later cycle follows, in the table or as a ``cavity`` event.
    """
    born = rows[(tab["jet_born"][rows] > 0) & ~(tab["r_j"][rows] <= 0)]
    if born.size and "cycle" in tab and born[-1] == rows[-1]:
        c = tab["cycle"][rows[-1]]
        if np.any(tab["cycle"] > c) or any(e["event"] == "cavity" and e.get("cycle") == c + 1
                                           for e in events):
            born = born[:-1]
    return born


def summarise(run: Path) -> dict:
    params = json.loads((run / "parameters.json").read_text())
    status = json.loads((run / "run_status.json").read_text())
    nd = params["dimensionless"]
    tab = read_table(run / "collapse.txt")
    t, z = tab["time"], tab["z_axis"] - nd["depth"]
    born = tab["jet_born"] > 0
    i_end = int(np.argmax(born)) if born.any() else len(t) - 1
    window = (t >= t[i_end] - 2 * nd["forcing_period"]) & (np.arange(len(t)) <= i_end)
    row = {"case_id": params["case"]["case_id"], "Gamma": nd["Gamma"], "Oh": nd["Oh"],
           "a0": nd["a0"], "state": status.get("state"),
           "stop": ((status.get("collapse") or {}).get("stop") or {}).get("reason", ""),
           "t_end": float(t[-1]), "jet_born": bool(born.any()),
           "n_jets": (np.unique(tab["cycle"][born]).size if "cycle" in tab else int(born.any())),
           "h_crest": float(z[window].max()), "h_trough": float(-z[i_end]),
           "remeshes": int(tab["remesh_count"][-1]),
           "volume_drift": float(np.nanmax(np.abs(tab["volume"] / tab["volume"][0] - 1))),
           "min_r_neck": float(np.nanmin(tab["r_neck"])) if np.isfinite(tab["r_neck"]).any()
           else math.nan}
    if born.any():  # the first jet, over its cycle
        rows = np.arange(t.size)
        if "cycle" in tab:
            rows = rows[tab["cycle"] == tab["cycle"][i_end]]
        jet = jet_rows(tab, rows, read_events(run))
        up = jet[tab["V_j"][jet] > 0]  # flux into the jet
        row.update(t_birth=float(t[i_end]), r_j_birth=float(tab["r_j"][i_end]),
                   min_r_j=_nan_extreme(np.min, tab["r_j"][jet]),
                   max_V_j=_nan_extreme(np.max, tab["V_j"][up]),
                   max_w_axis=_nan_extreme(np.max, tab["w_axis"][jet]),
                   max_We_j=_nan_extreme(np.max, tab["We_j"][up]))
    return row


CYCLE_FIELDS = ("case_id", "Gamma", "Oh", "a0", "cycle", "t_cavity", "h_crest", "crest_jet",
                "max_apex_curvature", "t_birth", "r_j_birth", "max_V_j", "max_w_axis",
                "min_r_neck")


def per_cycle(run: Path) -> list[dict]:
    params = json.loads((run / "parameters.json").read_text())
    nd = params["dimensionless"]
    tab = read_table(run / "collapse.txt")
    if "cycle" not in tab:
        return []
    t, z = tab["time"], tab["z_axis"] - nd["depth"]
    events = read_events(run)
    rows = []
    for c in np.unique(tab["cycle"]).astype(int):
        sel = (tab["cycle"] == c) & (tab["cavity"] > 0)
        if not sel.any():
            continue
        t0 = float(t[sel][0])
        before = (t >= t0 - 2 * nd["forcing_period"]) & (t <= t0)
        born = sel & (tab["jet_born"] > 0)
        pre = sel & ~born
        row = {"case_id": params["case"]["case_id"], "Gamma": nd["Gamma"], "Oh": nd["Oh"],
               "a0": nd["a0"], "cycle": int(c), "t_cavity": t0, "h_crest": float(z[before].max()),
               "crest_jet": int(np.any(tab["jet_born"][before] > 0)),
               "max_apex_curvature": float(np.nanmax(tab["axis_curvature"][pre]))
               if pre.any() else math.nan,
               "min_r_neck": float(np.nanmin(tab["r_neck"][sel]))
               if np.isfinite(tab["r_neck"][sel]).any() else math.nan}
        if born.any():
            i = int(np.argmax(born))
            jet = jet_rows(tab, np.flatnonzero(tab["cycle"] == c), events)
            row.update(t_birth=float(t[i]), r_j_birth=float(tab["r_j"][i]),
                       max_V_j=_nan_extreme(np.max, tab["V_j"][jet[tab["V_j"][jet] > 0]]),
                       max_w_axis=_nan_extreme(np.max, tab["w_axis"][jet]))
        rows.append(row)
    return rows


EPISODE_FIELDS = ("case_id", "run", "Gamma", "Oh", "ka0", "episode", "cycle", "partial", "source",
                  "t_start", "h_crest", "t_crest", "crest_jet", "min_z_axis", "max_apex_curvature",
                  "t_max_apex_curvature", "jet_born", "t_birth", "r_j_birth", "max_V_j",
                  "max_w_axis", "min_r_neck", "stop")


def read_events(run: Path) -> list[dict]:
    path = run / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trough_episodes(params: dict, tab: dict[str, np.ndarray], events: list[dict],
                    eta_hys: float | None = None) -> tuple[np.ndarray, dict[int, dict], str]:
    """Episode index of every row and the trough episodes ``{index: start, h_crest, ...}``.

    Recorded episodes come from the column ``episode`` and the ``trough`` events, or the ``restart``
    event for a trough already in progress at a restart. Without that column the record is
    segmented again from ``z_axis - depth`` and ``jet_born``, with the run's ``eta_hys`` unless one
    is given. ``crest_jet`` is None where the events predate it.
    """
    if "episode" in tab:
        troughs = {}
        for e in events:
            if e["event"] == "trough":
                troughs[int(e["episode"])] = {"t_start": e["time"], "h_crest": e["h_crest"],
                                              "t_crest": e["t_crest"], "partial": e["partial"],
                                              "crest_jet": e.get("crest_jet")}
            elif e["event"] == "restart" and e.get("episode_kind") == -1:
                troughs.setdefault(int(e["episode"]), {
                    "t_start": e["episode_start"], "h_crest": e["h_crest"],
                    "t_crest": e["t_crest"], "partial": e["episode_partial"],
                    "crest_jet": e.get("crest_jet")})
        return tab["episode"].astype(int), troughs, "recorded"
    if eta_hys is None:
        eta_hys = (params.get("collapse") or {}).get("eta_hys", ETA_HYS)
    eta = tab["z_axis"] - params["dimensionless"]["depth"]
    jet = tab["jet_born"] > 0 if "jet_born" in tab else None
    index, _, evs = segment(tab["time"], eta, eta_hys, history_known=params.get("restart") is None,
                            jet=jet)
    troughs = {e["episode"]: {"t_start": e["time"], "h_crest": e["h_crest"],
                              "t_crest": e["t_crest"], "partial": e["partial"],
                              "crest_jet": e["crest_jet"]}
               for e in evs if e["event"] == "trough"}
    return index, troughs, "reconstructed"


def crest_jet(info: dict, tab: dict[str, np.ndarray], index: np.ndarray, k: int,
              restarted: bool) -> float:
    """1 if a jet stood on the axis in the crest before trough ``k``, else 0; NaN if unknown.

    As the trough event recorded it, or, for events that predate the flag, from the rows of that
    crest: unknown when none of them has a jet and the crest began before the record.
    """
    if info.get("crest_jet") is not None:
        return int(info["crest_jet"])
    crest = index == k - 1
    if np.any(tab["jet_born"][crest] > 0):
        return 1
    whole = crest.any() and (not restarted or index[0] < k - 1)
    return 0 if whole and math.isfinite(info["h_crest"]) else math.nan


def _nan_extreme(fn, values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(fn(values)) if values.size else math.nan


def per_episode(run: Path, eta_hys: float | None = None) -> list[dict]:
    params = json.loads((run / "parameters.json").read_text())
    status_path = run / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    nd = params["dimensionless"]
    tab = read_table(run / "collapse.txt")
    events = read_events(run)
    index, troughs, source = trough_episodes(params, tab, events, eta_hys)
    restarted = params.get("restart") is not None or any(e["event"] == "restart" for e in events)
    t = tab["time"]
    row_index = np.arange(t.size)
    born = tab["jet_born"] > 0
    birth = born & ~np.concatenate([[True], born[:-1]])  # a jet standing at row 0 is not born there
    split = "cycle" in tab
    cycle = tab["cycle"].astype(int) if split else np.zeros(t.size, dtype=int)
    stop = (status.get("collapse") or {}).get("stop") or {}
    end = stop.get("reason") or status.get("state", "")
    if stop.get("trigger"):
        end += ":" + stop["trigger"]
    name = run.parent.name if run.name == "out" else run.name
    rows = []
    for k, info in sorted(troughs.items()):
        trough = index == k
        if not trough.any():
            continue
        opening = cycle[np.argmax(trough)]  # the cycle in progress when the trough began
        jet_in_crest = crest_jet(info, tab, index, k, restarted)
        for c in np.unique(cycle[trough]):
            in_cycle = cycle == c
            sel = trough & in_cycle
            jet = (trough | (index == k + 1)) & in_cycle  # the trough and the crest it launches
            births = row_index[birth & jet]
            apex = sel & (tab["axis_curvature"] > 0)
            if split and births.size:  # the collapse before its jet
                apex &= row_index < births[0]
            partial = c == opening and (bool(info["partial"]) or info["t_start"] < t[0])
            row = {"case_id": params["case"]["case_id"], "run": name, "Gamma": nd["Gamma"],
                   "Oh": nd["Oh"], "ka0": nd["a0"], "episode": k,
                   "cycle": int(c) if split else "", "partial": int(partial),
                   "source": source, "t_start": info["t_start"], "h_crest": info["h_crest"],
                   "t_crest": info["t_crest"], "crest_jet": jet_in_crest,
                   "min_z_axis": float(tab["z_axis"][sel].min()),
                   "max_apex_curvature": math.nan, "t_max_apex_curvature": math.nan,
                   "jet_born": int(births.size > 0), "max_V_j": math.nan,
                   "max_w_axis": _nan_extreme(np.max, tab["w_axis"][jet]),
                   "min_r_neck": _nan_extreme(np.min, tab["r_neck"][jet]),
                   "stop": end if index[-1] in (k, k + 1) and cycle[-1] == c else ""}
            if apex.any():
                i = int(np.flatnonzero(apex)[np.argmax(tab["axis_curvature"][apex])])
                row.update(max_apex_curvature=float(tab["axis_curvature"][i]),
                           t_max_apex_curvature=float(t[i]))
            if births.size:
                i = int(births[0])
                # the jet born here: flux into it from its birth on, over its rows with a jet base
                launched = jet_rows(tab, row_index[in_cycle], events)
                launched = launched[jet[launched] & (launched >= i)
                                    & (tab["V_j"][launched] > 0)]
                row.update(t_birth=float(t[i]), r_j_birth=float(tab["r_j"][i]),
                           max_V_j=_nan_extreme(np.max, tab["V_j"][launched]))
            rows.append(row)
    return rows


def _write(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    for r in rows:
        print("  ".join(f"{k}={r[k]:.4g}" if isinstance(r.get(k), float) else
                        f"{k}={r.get(k, '')}" for k in fields))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--per-cycle", action="store_true", help="one row per collapse cycle")
    mode.add_argument("--per-episode", action="store_true",
                      help="one row per trough episode, with the wave height h_crest")
    ap.add_argument("--eta-hys", type=float, default=None,
                    help="hysteresis for runs without recorded episodes (default: the run's "
                         f"eta_hys, else {ETA_HYS})")
    args = ap.parse_args()
    # A failed startup or an in-progress restart can have written only its table header.
    args.runs = [run for run in args.runs if (run / "collapse.txt").exists()
                 and read_table(run / "collapse.txt").get("time", np.empty(0)).size]
    if args.per_cycle:
        rows = [r for run in args.runs if (run / "collapse.txt").exists() for r in per_cycle(run)]
        rows.sort(key=lambda r: (r["Oh"], r["Gamma"], r["h_crest"]))
        _write(args.out, rows, CYCLE_FIELDS)
        return 0
    if args.per_episode:
        rows = [r for run in args.runs if (run / "collapse.txt").exists()
                for r in per_episode(run, args.eta_hys)]
        rows.sort(key=lambda r: (r["Oh"], r["Gamma"], r["ka0"], r["case_id"], r["run"],
                                 r["episode"], -1 if r["cycle"] == "" else r["cycle"]))
        _write(args.out, rows, EPISODE_FIELDS)
        return 0
    rows = []
    for run in args.runs:
        if (run / "collapse.txt").exists() and (run / "run_status.json").exists():
            rows.append(summarise(run))
    rows.sort(key=lambda r: (r["Oh"], r["Gamma"], r["a0"]))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    for r in rows:
        print("  ".join(f"{k}={r[k]:.4g}" if isinstance(r.get(k), float) else f"{k}={r.get(k, '')}"
                        for k in FIELDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
