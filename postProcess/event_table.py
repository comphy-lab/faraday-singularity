#!/usr/bin/env python3
"""One row per event of the runs after a cut with the events block, and one row per run.

Usage::

    python postProcess/event_table.py RUN [RUN ...] --out EVENTS.csv [--runs RUNS.csv] \
        [--tip cavity_tip [bubble_top ...]] [--after-events 1] [--near-base 3] \
        [--stationary-alpha 1e-3] [--stationary-q 1e-3]

Each ``RUN`` is the directory of a run after a cut with the ``collapse.reconnect_events`` block, or
the directories of the legs of one run continued in several, in their order, joined by
``os.pathsep`` (``leg1:leg2:leg3``). A leg directory holds the record of each of its events
(``events/event_<k>.json``: made, refused, unclosed or failed, or the event the run stopped on before
a handler was made for it, ``unhandled``, ``cascade`` or ``max-events``, with the handler its class
is to have), the rows of both chains per step (``chains.txt``), the budget of its liquid with the
removals (``budget.txt``), the start of the leg (the event ``leg`` of ``events.jsonl``: its step and
the events made before it), ``run_status.json`` and ``parameters.json``. The legs of a run are read
as one continuous run: the records by the event's number ``k``, a later leg's replacing an earlier
leg's (a leg writes the records of the events made before it, as its dump holds them, into its own
directory too, and rewrites them as it changes them), the records of an earlier leg beyond the
events a later one starts with dropped (that leg made them, or stopped on them, after the state the
later leg restarted from); the rows of ``chains.txt`` and ``budget.txt`` of the legs before a leg
beyond the step it starts at dropped, and its own first row where they have one at that step, so
that every step has the row a continuous run writes; the stop, the state and the end of the window
are the last leg's.

**Events** (``--out``): what made the event and what it removed, in units of the cut radius
``delta`` and of ``T = t_p - t_cut`` (the time from the cut to the pinch time it recorded, read as
``tau - tau_p``): the event's time (``tau_over_delta2``, ``t_prime_over_T``), the neck at its
landing (``r_n_over_delta``, ``r_n_over_s`` with ``s`` the head's largest radius), the removed head
(``V_over_delta3``, ``V_over_s3``), the interval since the event before at its tip
(``interval_over_delta2``, ``interval_over_T``, ``events_per_T``) and the cascade's verdict there,
the base of the jet whose neck was cut (``r_j_over_delta`` at the event, ``r_j_over_r_n``, the
neck's distance from the base in neck radii and in jet radii, ``near_base`` where that is within
``--near-base`` jet radii, and the base after the cut over the base before it), the step before it
(``dt_over_delta2``), the neck's Reynolds and Weber numbers, the ghost's speed and the stump's
speed relative to it, the closures of the cut relative to the head's own (``F1``-``F6``, its
``flags`` and ``hard`` misses), the transfer, the ghost's recontacts (their times in ``delta^2``)
and the dumps of the event the run deleted.

**Runs** (``--runs``): the legs read, the events at each tip and the cascade's last verdict there,
the volume the events removed (``delta^3``), the drift of the liquid of the mesh over the rows of
``budget.txt`` and of its sum with the removals (each relative to its first value: the removals
account for the difference), the share of the removals in that sum of the axial momentum and of the
kinetic energy at its last row, the ghosts' recontacts, the refused and failed cuts, the events made
with a closure flagged, the end of the window, the stop with the kind and number of the event it
stopped on, and the test of the base of the jet of a tip (``--tip``, one row per run and tip; the
upward jet into the cavity, ``cavity_tip``, by default, the downward one ``bubble_top``): over the
rows of ``chains.txt`` after the step of the ``--after-events``-th event made at that tip (its
number and step ``base_after_k``, ``base_after_step``; 0: from the run's start) in which that jet
stands and ``t' = t - t_p`` is positive, the least-squares slope of ``ln r_j`` against ``ln t'``
(``alpha_r_j``), the slope of ``Q_j`` against ``t'`` (``dQ_j_dt``), the change of ``Q_j`` over
the window it gives relative to the median ``|Q_j|`` there (``dQ_j_rel``) and the slope of
``ln Q_j`` against ``ln t'`` where every ``Q_j`` is positive (``alpha_Q_j``). Each of
``alpha_r_j`` and ``dQ_j_rel`` counts as zero within its floor (``--stationary-alpha``,
``--stationary-q``, ``STATIONARY_ALPHA`` and ``STATIONARY_Q`` by default, recorded with the row as
``stationary_alpha``, ``stationary_q``), so that a base standing still is not given a sign by the
round-off of its fit: ``widens`` when both are positive, ``narrows`` when ``alpha_r_j`` is negative,
``stationary`` when both are zero, ``mixed`` otherwise, ``insufficient`` with fewer than
``MIN_ROWS`` rows and ``no-events`` before the event the window starts at. The base is also given
against the root neck of its chain (``r_j - r_n`` in ``delta``, at the first and the last row of the
window), with the median step in ``delta^2``. The test is a report: it decides nothing about the
run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

MIN_ROWS = 5  # rows of the base test at least
# the floors within which the base test counts the slope of ln r_j against ln t' and the relative
# change of Q_j over its window as zero (a base standing still)
STATIONARY_ALPHA = 1e-3
STATIONARY_Q = 1e-3
TIP_NECKS = {"cavity_tip": "r_neck_open", "bubble_top": "r_neck_bubble",
             "bubble_bottom": "r_neck_bubble"}  # the root neck of each tip's chain (chains.txt)
TIPS = ("cavity_tip", "bubble_top", "bubble_bottom")

EVENT_FIELDS = ("run", "case_id", "delta", "k", "outcome", "failed_at", "kind", "class", "handler",
                "chain", "tip", "generation", "step", "leg", "t", "tau_over_delta2",
                "t_prime_over_T", "r_n_over_delta", "s_over_delta", "r_n_over_s", "V_over_delta3",
                "V_over_s3", "interval_over_delta2", "interval_over_T", "events_per_T", "cascade",
                "r_j_over_delta", "r_j_over_r_n", "neck_to_base_over_r_n",
                "neck_to_base_over_r_j", "near_base", "r_j_after_over_r_j", "dt_over_delta2",
                "Re_n", "We_n", "U_ghost", "du_stump", "F1", "F2", "F3", "F4", "F5", "F6",
                "phi_relative_error", "flags", "hard", "predicted_unlocated",
                "reported_unlocated", "rear_recontact_tau_over_delta2",
                "front_recontact_tau_over_delta2", "dumps_dropped")
RUN_FIELDS = ("run", "case_id", "legs", "delta", "T", "state", "stop", "stop_kind", "stop_event",
              "window_end", "events", *(f"events_{tip}" for tip in TIPS),
              *(f"cascade_{tip}" for tip in TIPS), "V_removed_over_delta3", "volume_drift",
              "volume_total_drift", "Pz_removed_share", "KE_removed_share", "rear_recontacts",
              "front_recontacts", "refused", "unclosed", "failed", "flagged", "tip",
              "base_after_k", "base_after_step", "base_rows", "base_t_prime_lo",
              "base_t_prime_hi", "alpha_r_j", "dQ_j_dt", "dQ_j_rel", "alpha_Q_j",
              "base_verdict", "stationary_alpha", "stationary_q", "r_j_start_over_delta",
              "r_j_end_over_delta", "base_minus_neck_start_over_delta",
              "base_minus_neck_end_over_delta", "dt_over_delta2_median")


def read_table(path: Path) -> dict[str, list]:
    """A table of the run (tab-separated, its header a ``#`` line): each column a list, numbers as
    floats and text as it is; empty without the file."""
    if not path.is_file():
        return {}
    lines = path.read_text().splitlines()
    if not lines:
        return {}
    head = lines[0].lstrip("#").strip().split("\t")
    out: dict[str, list] = {name: [] for name in head}
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        for name, cell in zip(head, line.split("\t")):
            try:
                out[name].append(float(cell))
            except ValueError:
                out[name].append(cell)
    return out


def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x


def _ratio(a, b) -> float:
    a, b = _num(a), _num(b)
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0.0 else math.nan


def run_legs(run) -> list[Path]:
    """The leg directories of a run: one directory, several joined by ``os.pathsep`` in their order,
    or a sequence of them."""
    if isinstance(run, (str, os.PathLike)):
        return [Path(p) for p in os.fspath(run).split(os.pathsep) if p]
    return [Path(p) for p in run]


def run_name(run) -> str:
    """The name of a run: that of its first leg's directory, or of its parent for a ``runtime``
    directory."""
    first = run_legs(run)[0]
    return first.parent.name if first.name == "runtime" else first.name


def leg_start(leg: Path) -> dict:
    """Where a leg starts: the step of its event ``leg`` (``events.jsonl``) and the events made before
    it (``events_made``, None where the leg does not record it); the first row of its ``chains.txt``
    without the event."""
    path = leg / "events.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if isinstance(x, dict) and x.get("event") == "leg":
                return {"step": _num(x.get("step")), "events_made": x.get("events_made")}
    steps = read_table(leg / "chains.txt").get("step") or []
    return {"step": _num(steps[0]) if steps else math.nan, "events_made": None}


def leg_records(leg: Path) -> list[dict]:
    """The records of the events in one leg's directory (``events/event_<k>.json``), in their
    order."""
    out = [_json(p) for p in sorted((leg / "events").glob("event_*.json"))]
    return sorted((r for r in out if r), key=lambda r: int(r.get("k", 0)))


def event_records(run) -> list[dict]:
    """The records of the events of a run over its legs, in their order (module docstring): by the
    event's number, a later leg's record replacing an earlier leg's, and the records of an earlier
    leg beyond the events a later leg starts with (``events_made`` of its start; where it does not
    record it, those at or after the step it starts at) dropped."""
    by_k: dict[int, dict] = {}
    for i, leg in enumerate(run_legs(run)):
        if i > 0 and by_k:
            start = leg_start(leg)
            made = start["events_made"]
            if isinstance(made, int) and not isinstance(made, bool):
                by_k = {k: r for k, r in by_k.items() if k <= made}
            elif math.isfinite(start["step"]):
                by_k = {k: r for k, r in by_k.items() if not _num(r.get("step")) >= start["step"]}
        for rec in leg_records(leg):
            by_k[int(rec.get("k", 0))] = rec
    return [by_k[k] for k in sorted(by_k)]


def join_rows(run, name: str) -> dict[str, list]:
    """The rows of one table of a run (``chains.txt``, ``budget.txt``) over its legs, as one
    continuous run writes them: before each leg after the first, the rows of the legs before it
    beyond the step it starts at are dropped, and its own first row where they have one at that
    step (the restart's row of the state the leg starts from, with no step of its own)."""
    rows: list[dict] = []
    for leg in run_legs(run):
        table = read_table(leg / name)
        steps = table.get("step") or []
        new = [{key: col[i] for key, col in table.items() if i < len(col)}
               for i in range(len(steps))]
        if not new:
            continue
        if rows:
            s0 = _num(new[0].get("step"))
            rows = [r for r in rows if not _num(r.get("step")) > s0]
            if any(_num(r.get("step")) == s0 for r in rows):
                new = new[1:]
        rows += new
    names = list(dict.fromkeys(key for r in rows for key in r))
    return {key: [r.get(key, math.nan) for r in rows] for key in names}


def run_info(run, records: list[dict]) -> dict:
    """The case (its last leg's), the cut radius ``delta`` (the case's ``collapse.reconnect``, else a
    record's), ``T = t_p - t_cut`` (``tau - tau_p`` of ``chains.txt``, else of a record) and the
    joined rows of ``chains.txt`` of a run."""
    legs = run_legs(run)
    params = [_json(leg / "parameters.json") for leg in legs]
    case = next((p.get("case") for p in reversed(params) if p.get("case")), None) or {}
    delta = next((d for d in (_num((p.get("reconnect") or {}).get("delta")) for p in params)
                  if math.isfinite(d)), math.nan)
    if not math.isfinite(delta):
        delta = next((_num(r.get("delta")) for r in records if r.get("delta") is not None),
                     math.nan)
    T = math.nan
    chains = join_rows(legs, "chains.txt")
    if chains.get("tau") and chains.get("tau_p"):
        T = _num(chains["tau"][0]) - _num(chains["tau_p"][0])
    if not math.isfinite(T):
        T = next((_num(r["tau"]) - _num(r["tau_p"]) for r in records
                  if r.get("tau") is not None and r.get("tau_p") is not None), math.nan)
    return {"run": run_name(legs), "case_id": case.get("case_id", ""), "legs": len(legs),
            "delta": delta, "T": T, "chains": chains}


def event_rows(run, *, near_base: float = 3.0) -> list[dict]:
    """The rows of the events of one run (module docstring)."""
    records = event_records(run)
    info = run_info(run, records)
    d, T = info["delta"], info["T"]
    rows = []
    for rec in records:
        trig, removed = rec.get("trigger") or {}, rec.get("removed") or {}
        cascade, ghost = rec.get("cascade") or {}, rec.get("ghost") or {}
        closures = rec.get("closures") or {}
        rel, tr = closures.get("relative") or {}, rec.get("transfer") or {}
        r_n, s, r_j = _num(trig.get("r_n")), _num(trig.get("s")), _num(trig.get("r_j"))
        z_n, z_b = _num(trig.get("z_n")), _num(trig.get("z_b"))
        V = _num(removed.get("V"))
        gap = abs(z_n - z_b) if math.isfinite(z_n) and math.isfinite(z_b) else math.nan
        to_base = _ratio(gap, r_j)
        hits = {x.get("kind"): x for x in ghost.get("recontacts") or ()}
        row = {"run": info["run"], "case_id": info["case_id"], "delta": d,
               **{key: rec.get(key) for key in ("k", "outcome", "failed_at", "kind", "class",
                                                 "chain", "tip", "generation", "step", "leg",
                                                 "t")},
               "handler": rec.get("handler", (rec.get("request") or {}).get("handler")),
               "tau_over_delta2": _ratio(rec.get("tau"), d * d),
               "t_prime_over_T": _ratio(rec.get("tau_p"), T),
               "r_n_over_delta": _ratio(r_n, d), "s_over_delta": _ratio(s, d),
               "r_n_over_s": _ratio(r_n, s), "V_over_delta3": _ratio(V, d**3),
               "V_over_s3": _ratio(V, s**3),
               "interval_over_delta2": _ratio(cascade.get("interval"), d * d),
               "interval_over_T": _ratio(cascade.get("interval"), T),
               "events_per_T": _num(cascade.get("events_per_T")),
               "cascade": cascade.get("verdict", ""),
               "r_j_over_delta": _ratio(r_j, d), "r_j_over_r_n": _ratio(r_j, r_n),
               "neck_to_base_over_r_n": _ratio(gap, r_n), "neck_to_base_over_r_j": to_base,
               "near_base": (int(to_base <= near_base) if math.isfinite(to_base) else ""),
               "r_j_after_over_r_j": _ratio((rec.get("after") or {}).get("r_j"), r_j),
               "dt_over_delta2": _ratio((rec.get("dt") or {}).get("history"), d * d),
               "Re_n": _num(trig.get("Re_n")), "We_n": _num(trig.get("We_n")),
               "U_ghost": _num(ghost.get("U")), "du_stump": _num(ghost.get("du_stump")),
               **{f: _num(rel.get(f)) for f in ("F1", "F2", "F3", "F4", "F5", "F6")},
               "phi_relative_error": _num((closures.get("F3_phi") or {})
                                          .get("phi_relative_error")),
               "flags": " ".join(closures.get("flags") or ()),
               "hard": " ".join(closures.get("hard") or ()),
               "predicted_unlocated": tr.get("predicted_unlocated", ""),
               "reported_unlocated": tr.get("reported_unlocated", ""),
               "dumps_dropped": " ".join((rec.get("dumps") or {}).get("dropped") or ())}
        for kind in ("rear", "front"):
            row[f"{kind}_recontact_tau_over_delta2"] = _ratio((hits.get(kind) or {}).get("tau"),
                                                              d * d)
        rows.append(row)
    return rows


def _slope(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0.0:
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def _sign(x: float, floor: float) -> int:
    """The sign of ``x``, zero within ``floor`` of zero."""
    return 1 if x > floor else -1 if x < -floor else 0


def base_test(chains: dict, tip: str, after_step: float | None, delta: float, *,
              alpha_floor: float = STATIONARY_ALPHA, q_floor: float = STATIONARY_Q) -> dict:
    """The test of the base of the jet of ``tip`` over the rows after the step ``after_step``
    (module docstring; ``-inf`` from the run's start, None: the run has not made the event the
    window starts at), with the floors of a base standing still, ``alpha_floor`` of ``alpha_r_j``
    and ``q_floor`` of ``dQ_j_rel``."""
    out = {"tip": tip, "base_rows": 0, "base_t_prime_lo": math.nan, "base_t_prime_hi": math.nan,
           "alpha_r_j": math.nan, "dQ_j_dt": math.nan, "dQ_j_rel": math.nan,
           "alpha_Q_j": math.nan,
           "base_verdict": "no-events" if after_step is None else "insufficient",
           "stationary_alpha": alpha_floor, "stationary_q": q_floor,
           "r_j_start_over_delta": math.nan, "r_j_end_over_delta": math.nan,
           "base_minus_neck_start_over_delta": math.nan,
           "base_minus_neck_end_over_delta": math.nan, "dt_over_delta2_median": math.nan}
    names = (f"r_j_{tip}", f"Q_j_{tip}", f"jet_born_{tip}", "tau_p", "step")
    if after_step is None or not all(chains.get(n) for n in names):
        return out
    cols = {n: np.asarray([_num(x) for x in chains[n]]) for n in names}
    neck_name = TIP_NECKS.get(tip)
    neck = np.asarray([_num(x) for x in chains.get(neck_name, [])]) if neck_name in chains \
        else np.full(len(cols["step"]), math.nan)
    dt = np.asarray([_num(x) for x in chains.get("dt", [])]) if "dt" in chains \
        else np.full(len(cols["step"]), math.nan)
    r_j, Q, t_p = cols[f"r_j_{tip}"], cols[f"Q_j_{tip}"], cols["tau_p"]
    # the rows after the event's step: its own row is the state before its cut
    sel = ((cols["step"] > after_step) & (cols[f"jet_born_{tip}"] > 0) & np.isfinite(r_j)
           & (r_j > 0.0) & np.isfinite(Q) & np.isfinite(t_p) & (t_p > 0.0))
    idx = np.flatnonzero(sel)
    out["base_rows"] = len(idx)
    if len(idx) < MIN_ROWS:
        return out
    x = np.log(t_p[idx])
    out.update(base_t_prime_lo=float(t_p[idx[0]]), base_t_prime_hi=float(t_p[idx[-1]]),
               alpha_r_j=_slope(x, np.log(r_j[idx])), dQ_j_dt=_slope(t_p[idx], Q[idx]),
               alpha_Q_j=_slope(x, np.log(Q[idx])) if np.all(Q[idx] > 0.0) else math.nan,
               r_j_start_over_delta=float(r_j[idx[0]] / delta),
               r_j_end_over_delta=float(r_j[idx[-1]] / delta),
               base_minus_neck_start_over_delta=float((r_j[idx[0]] - neck[idx[0]]) / delta),
               base_minus_neck_end_over_delta=float((r_j[idx[-1]] - neck[idx[-1]]) / delta),
               dt_over_delta2_median=float(np.nanmedian(dt[idx]) / delta**2)
               if np.isfinite(dt[idx]).any() else math.nan)
    # the change of Q_j over the window, by the fit, relative to the median |Q_j| there
    scale = float(np.median(np.abs(Q[idx])))
    out["dQ_j_rel"] = (out["dQ_j_dt"] * (out["base_t_prime_hi"] - out["base_t_prime_lo"]) / scale
                       if scale > 0.0 else math.nan)
    a = _sign(out["alpha_r_j"], alpha_floor)
    q = _sign(out["dQ_j_rel"], q_floor)
    out["base_verdict"] = ("widens" if a > 0 and q > 0 else "narrows" if a < 0
                           else "stationary" if a == 0 and q == 0 else "mixed")
    return out


def run_row(run, *, tip: str = "cavity_tip", after_events: int = 1,
            alpha_floor: float = STATIONARY_ALPHA, q_floor: float = STATIONARY_Q) -> dict:
    """The row of one run (module docstring) and the test of the base of the jet of ``tip``, with
    the floors of a base standing still (``alpha_floor``, ``q_floor``)."""
    legs = run_legs(run)
    records = event_records(legs)
    info = run_info(legs, records)
    d = info["delta"]
    status = _json(legs[-1] / "run_status.json")
    stop = ((status.get("collapse") or {}).get("stop") or {})
    window = status.get("window_end") or (status.get("collapse") or {}).get("window_end")
    made = [r for r in records if r.get("outcome") == "made"]
    row = {"run": info["run"], "case_id": info["case_id"], "legs": info["legs"], "delta": d,
           "T": info["T"], "state": status.get("state", ""), "stop": stop.get("reason", ""),
           "stop_kind": stop.get("kind", ""), "stop_event": stop.get("k", ""),
           "window_end": (window or {}).get("reason", ""), "events": len(made)}
    for t in TIPS:
        at = [r for r in made if r.get("tip") == t]
        row[f"events_{t}"] = len(at)
        row[f"cascade_{t}"] = (at[-1].get("cascade") or {}).get("verdict", "") if at else ""
    row["V_removed_over_delta3"] = _ratio(sum(_num((r.get("removed") or {}).get("V"))
                                              for r in made) if made else 0.0, d**3)
    budget = join_rows(legs, "budget.txt")
    for key, name in (("volume", "volume_drift"), ("volume_total", "volume_total_drift")):
        col = [_num(x) for x in budget.get(key, [])]
        row[name] = (col[-1] - col[0]) / col[0] if len(col) >= 2 and col[0] else math.nan
    for removed, total, name in (("Pz_removed", "momentum_z_total", "Pz_removed_share"),
                                 ("KE_removed", "kinetic_energy_total", "KE_removed_share")):
        a, b = budget.get(removed) or [], budget.get(total) or []
        row[name] = _ratio(a[-1], b[-1]) if a and b else math.nan
    hits = [x for r in made for x in (r.get("ghost") or {}).get("recontacts") or ()]
    row["rear_recontacts"] = sum(x.get("kind") == "rear" for x in hits)
    row["front_recontacts"] = sum(x.get("kind") == "front" for x in hits)
    for outcome in ("refused", "unclosed", "failed"):
        row[outcome] = sum(r.get("outcome") == outcome for r in records)
    row["flagged"] = sum(bool((r.get("closures") or {}).get("flags")) for r in made)
    # the window starts after the after_events-th event made at the tip tested, not at any tip
    own = [r for r in made if r.get("tip") == tip]
    anchor = own[after_events - 1] if 0 < after_events <= len(own) else None
    row.update(base_after_k=anchor.get("k", "") if anchor else "",
               base_after_step=anchor.get("step", "") if anchor else "")
    after = -math.inf if after_events <= 0 else (_num(anchor.get("step")) if anchor else None)
    row.update(base_test(info["chains"], tip, after, d, alpha_floor=alpha_floor,
                         q_floor=q_floor))
    return row


def _write(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="a run directory, or the directories of the legs of one run in their "
                         f"order joined by {os.pathsep!r}")
    ap.add_argument("--out", type=Path, required=True, help="the table of the events")
    ap.add_argument("--runs", dest="runs_out", type=Path, help="the table of the runs")
    ap.add_argument("--tip", choices=TIPS, nargs="+", default=["cavity_tip"],
                    help="the tips whose jets' bases the runs' tests read (a row per run and tip)")
    ap.add_argument("--after-events", type=int, default=1,
                    help="the test's window starts after this event made at its tip (0: at the "
                         "run's start)")
    ap.add_argument("--near-base", type=float, default=3.0,
                    help="a neck within this many jet radii of the base is near it")
    ap.add_argument("--stationary-alpha", type=float, default=STATIONARY_ALPHA,
                    help="the slope of ln r_j against ln t' counts as zero within this")
    ap.add_argument("--stationary-q", type=float, default=STATIONARY_Q,
                    help="the relative change of Q_j over the window counts as zero within this")
    args = ap.parse_args(argv)
    runs = [legs for legs in (run_legs(run) for run in args.runs)
            if legs and all(leg.is_dir() for leg in legs)]
    events = [r for legs in runs for r in event_rows(legs, near_base=args.near_base)]
    _write(args.out, events, EVENT_FIELDS)
    rows = [run_row(legs, tip=tip, after_events=args.after_events,
                    alpha_floor=args.stationary_alpha, q_floor=args.stationary_q)
            for legs in runs for tip in args.tip]
    if args.runs_out is not None:
        _write(args.runs_out, rows, RUN_FIELDS)
    for r in rows:
        print(f"{r['run']}: {r['events']} events, stop {r['stop'] or '-'}, base of {r['tip']} "
              f"{r['base_verdict']} (alpha_r_j {r['alpha_r_j']:.3g}, {r['base_rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
