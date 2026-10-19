"""Contract tests of the table of the events of runs after a cut (``postProcess/event_table.py``) on
synthetic run directories with known answers, and on the outputs of a run of the collapse driver
on a stand-in problem with a stand-in liquid cut. No finite-element problem is built."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "postProcess"))

import event_table  # noqa: E402

DELTA = 1e-3
T = 2e-6  # t_p - t_cut
CHAIN_COLUMNS = ("step", "time", "tau", "tau_p", "dt", "r_neck_open", "r_neck_bubble",
                 "r_j_cavity_tip", "Q_j_cavity_tip", "V_j_cavity_tip", "jet_born_cavity_tip",
                 "r_j_bubble_top", "Q_j_bubble_top", "V_j_bubble_top", "jet_born_bubble_top")


def _record(k, t, tip="cavity_tip", *, V=5e-10, r_n=1e-4, s=5e-4, r_j=6e-4, z_n=4.7, z_b=4.6995,
            interval=None, verdict="first", outcome="made", recontacts=(), dropped=None):
    chain = "open" if tip == "cavity_tip" else "bubble"
    rec = {"schema": "faraday-event/2", "k": k, "kind": "liquid-neck", "class": "L",
           "outcome": outcome, "chain": chain, "tip": tip, "generation": 1, "step": 10 * k,
           "t": t, "tau": t, "tau_p": t - T, "delta": DELTA,
           "trigger": {"r_n": r_n, "s": s, "r_j": r_j, "z_n": z_n, "z_b": z_b, "Re_n": 15.0,
                       "We_n": 27.0},
           "removed": {"V": V, "Pz": 50.0 * V, "KE": 1250.0 * V},
           "cascade": {"verdict": verdict, "interval": interval,
                       "events_per_T": None if interval is None else T / interval},
           "ghost": {"U": 50.0, "du_stump": 1.0, "recontacts": list(recontacts)},
           "closures": {"relative": {f"F{j}": 1e-3 * j for j in range(1, 7)},
                        "F3_phi": {"phi_relative_error": 1e-4}, "flags": ["F4"], "hard": []},
           "transfer": {"predicted_unlocated": 3, "reported_unlocated": 4},
           "dt": {"history": 2.2e-9}, "after": {"r_j": 1.2 * r_j},
           "dumps": {} if dropped is None else {"dropped": dropped}}
    if outcome != "made":
        rec["failed_at"] = "caps"
    return rec


BUDGET_HEAD = ("step", "time", "volume", "V_removed", "volume_total")


def _write_run(run: Path, records, rows, *, budget=None, status=None, budget_head=BUDGET_HEAD):
    (run / "events").mkdir(parents=True)
    (run / "parameters.json").write_text(json.dumps({"case": {"case_id": "case-a"},
                                                     "reconnect": {"delta": DELTA}}))
    (run / "run_status.json").write_text(json.dumps(status or {
        "state": "completed", "collapse": {"stop": {"reason": "max-steps"}},
        "window_end": None}))
    for rec in records:
        (run / "events" / f"event_{rec['k']:03d}.json").write_text(json.dumps(rec))
    lines = ["# " + "\t".join(CHAIN_COLUMNS)]
    lines += ["\t".join(f"{r.get(c, math.nan):.12e}" for c in CHAIN_COLUMNS) for r in rows]
    (run / "chains.txt").write_text("\n".join(lines) + "\n")
    if budget is not None:
        (run / "budget.txt").write_text("\n".join(["# " + "\t".join(budget_head)] + [
            "\t".join(f"{x:.12e}" for x in b) for b in budget]) + "\n")


def _rows(n=40, *, alpha=0.5, q=1.0, t0=1e-6, t1=2e-5):
    """Rows of chains.txt whose upward jet's base grows as t'^alpha after the pinch time and whose
    flux grows as t'^q, its root neck at 0.9 of the base."""
    out = []
    for i, t in enumerate(np.linspace(t0, t1, n)):
        tp = t - T
        r = 6e-4 * (tp / 1e-5) ** alpha if tp > 0 else 6e-4
        Q = 1e-3 * (max(tp, 1e-9) / 1e-5) ** q
        out.append({"step": i, "time": t, "tau": t, "tau_p": tp, "dt": 2.2e-9,
                    "r_neck_open": 0.9 * r, "r_neck_bubble": 3e-4, "r_j_cavity_tip": r,
                    "Q_j_cavity_tip": Q, "V_j_cavity_tip": Q / (math.pi * r * r),
                    "jet_born_cavity_tip": 1.0, "r_j_bubble_top": 4e-4, "Q_j_bubble_top": 1e-4,
                    "V_j_bubble_top": 1e-4 / (math.pi * 1.6e-7), "jet_born_bubble_top": 1.0})
    return out


def test_the_events_of_a_run_in_units_of_the_cut_radius(tmp_path):
    """Each event in units of delta, delta^2, delta^3 and T: its time, the neck, the removed head,
    the interval at its tip, the jet's base and the neck's distance from it (near the base within
    --near-base jet radii), the closures, the transfer, the ghost's recontacts and the dumps
    dropped; refused events are rows too."""
    records = [_record(1, 3e-6), _record(2, 7e-6, interval=4e-6, verdict="undetermined",
                                         z_b=4.698,
                                         recontacts=[{"kind": "rear", "tau": 9e-6}],
                                         dropped=["postcut"]),
               _record(3, 9e-6, tip="bubble_top", outcome="refused"),
               {"schema": "faraday-event/2", "k": 4, "kind": "piercing", "class": "W1",
                "handler": "ring", "outcome": "unhandled", "reason": "event-unhandled",
                "chain": "bubble", "tip": None, "step": 50, "t": 1.1e-5, "tau": 1.1e-5,
                "tau_p": 1.1e-5 - T, "delta": DELTA, "trigger": {},
                "request": {"bubble_length": 8e-4}, "dumps": {"precut": "precut_e004.dump"}}]
    run = tmp_path / "run-a" / "runtime"
    _write_run(run, records, _rows())
    rows = event_table.event_rows(run)
    assert [r["k"] for r in rows] == [1, 2, 3, 4] and rows[0]["run"] == "run-a"
    first, second, refused, pierced = rows
    assert (pierced["outcome"], pierced["kind"], pierced["class"], pierced["handler"]) == (
        "unhandled", "piercing", "W1", "ring")
    assert pierced["tau_over_delta2"] == pytest.approx(11.0)
    assert math.isnan(pierced["r_n_over_delta"]) and pierced["near_base"] == ""
    assert first["handler"] is None  # a made cut's record names it under its request only
    assert first["tau_over_delta2"] == pytest.approx(3.0)
    assert first["t_prime_over_T"] == pytest.approx((3e-6 - T) / T)
    assert first["r_n_over_delta"] == pytest.approx(0.1) and first["r_n_over_s"] == pytest.approx(
        0.2)
    assert first["V_over_delta3"] == pytest.approx(0.5) and first["V_over_s3"] == pytest.approx(4.0)
    assert math.isnan(first["interval_over_delta2"]) and first["cascade"] == "first"
    assert second["interval_over_delta2"] == pytest.approx(4.0)
    assert second["interval_over_T"] == pytest.approx(2.0)
    assert second["events_per_T"] == pytest.approx(0.5)
    assert first["r_j_over_delta"] == pytest.approx(0.6)
    assert first["r_j_over_r_n"] == pytest.approx(6.0)
    assert first["neck_to_base_over_r_n"] == pytest.approx(5.0)  # 5e-4 above the base
    assert first["neck_to_base_over_r_j"] == pytest.approx(5e-4 / 6e-4)
    assert first["near_base"] == 1 and second["near_base"] == 0  # 2e-3: 3.3 jet radii
    assert first["r_j_after_over_r_j"] == pytest.approx(1.2)
    assert first["dt_over_delta2"] == pytest.approx(2.2e-3)
    assert first["F3"] == pytest.approx(3e-3) and first["flags"] == "F4" and first["hard"] == ""
    assert (first["predicted_unlocated"], first["reported_unlocated"]) == (3, 4)
    assert second["rear_recontact_tau_over_delta2"] == pytest.approx(9.0)
    assert math.isnan(second["front_recontact_tau_over_delta2"])
    assert second["dumps_dropped"] == "postcut" and first["dumps_dropped"] == ""
    assert refused["outcome"] == "refused" and refused["failed_at"] == "caps"
    near = event_table.event_rows(run, near_base=4.0)
    assert near[1]["near_base"] == 1


def test_the_base_of_a_jet_that_widens_after_the_events_and_of_one_that_narrows(tmp_path):
    """The run's test of the upward jet's base from its first event on: a base growing as t'^0.5
    with a growing flux widens (alpha_r_j 0.5), one shrinking as t'^-0.3 narrows; the base against
    its root neck, the median step, the events at each tip and the cascade's last verdict there,
    the removed volume, the drift of the liquid with and without the removals, the recontacts and
    the refused cuts; without the event the window starts at, no test."""
    records = [_record(1, 3e-6), _record(2, 7e-6, interval=4e-6, verdict="undetermined",
                                         recontacts=[{"kind": "rear", "tau": 9e-6}]),
               _record(3, 9e-6, tip="bubble_top", outcome="refused")]
    head = BUDGET_HEAD + ("Pz_removed", "momentum_z_total", "KE_removed", "kinetic_energy_total")
    budget = [(0, 0.0, 100.0, 0.0, 100.0, 0.0, 2.0, 0.0, 8.0),
              (1, 1e-5, 100.0 - 1e-9, 1e-9, 100.0 + 2e-12, 0.1, 2.5, 0.2, 8.0)]
    status = {"state": "completed", "window_end": None,
              "collapse": {"stop": {"reason": "event-unhandled", "k": 4, "kind": "piercing"}}}
    wide = tmp_path / "wide"
    _write_run(wide, records, _rows(alpha=0.5), budget=budget, budget_head=head, status=status)
    row = event_table.run_row(wide)
    assert (row["stop"], row["stop_kind"], row["stop_event"]) == ("event-unhandled", "piercing", 4)
    assert row["Pz_removed_share_liquid"] == pytest.approx(0.04)
    assert row["KE_removed_share_liquid"] == pytest.approx(0.025)
    assert row["flagged"] == 2  # the two made, each with F4 flagged; the refused one not counted
    assert row["base_verdict"] == "widens" and row["alpha_r_j"] == pytest.approx(0.5)
    assert row["dQ_j_dt"] > 0.0 and row["alpha_Q_j"] == pytest.approx(1.0)
    assert row["base_t_prime_lo"] > 0.0 and row["base_rows"] >= event_table.MIN_ROWS
    assert row["r_j_end_over_delta"] > row["r_j_start_over_delta"]
    assert row["base_minus_neck_end_over_delta"] == pytest.approx(0.1 * row["r_j_end_over_delta"])
    assert row["dt_over_delta2_median"] == pytest.approx(2.2e-3)
    assert (row["events"], row["events_cavity_tip"], row["events_bubble_top"]) == (2, 2, 0)
    assert row["cascade_cavity_tip"] == "undetermined" and row["cascade_bubble_top"] == ""
    assert row["V_removed_over_delta3"] == pytest.approx(1.0)
    assert row["volume_drift"] == pytest.approx(-1e-11)
    assert row["volume_total_drift"] == pytest.approx(2e-14)
    assert (row["rear_recontacts"], row["front_recontacts"], row["refused"]) == (1, 0, 1)
    assert row["state"] == "completed" and row["T"] == pytest.approx(T)
    narrow = tmp_path / "narrow"
    _write_run(narrow, records, _rows(alpha=-0.3, q=-0.5))
    row = event_table.run_row(narrow)
    assert row["base_verdict"] == "narrows" and row["alpha_r_j"] == pytest.approx(-0.3)
    assert row["stop"] == "max-steps" and row["stop_kind"] == "" and row["stop_event"] == ""
    assert math.isnan(row["Pz_removed_share_liquid"])  # a budget without the columns: none
    late = event_table.run_row(wide, after_events=3)
    assert late["base_verdict"] == "no-events" and late["base_rows"] == 0
    start = event_table.run_row(wide, after_events=0)
    assert start["base_rows"] > row["base_rows"] - 1 and start["base_verdict"] == "widens"
    few = tmp_path / "few"
    _write_run(few, records, _rows(n=4))
    assert event_table.run_row(few)["base_verdict"] == "insufficient"


def test_the_command_line_writes_both_tables(tmp_path, capsys):
    run = tmp_path / "run"
    _write_run(run, [_record(1, 3e-6), _record(2, 4e-6, tip="bubble_top")], _rows())
    out, runs = tmp_path / "events.csv", tmp_path / "runs.csv"
    assert event_table.main([str(run), "--out", str(out), "--runs", str(runs)]) == 0
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["k"] for r in rows] == ["1", "2"] and tuple(rows[0]) == event_table.EVENT_FIELDS
    with runs.open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["base_verdict"] == "widens" and tuple(row) == event_table.RUN_FIELDS
    assert (row["base_after_k"], row["base_after_step"]) == ("1", "10")
    assert "2 events" in capsys.readouterr().out
    # a row per run and tip: the upward jet and the downward one, whose base stands still
    assert event_table.main([str(run), "--out", str(out), "--runs", str(runs), "--tip",
                             "cavity_tip", "bubble_top"]) == 0
    with runs.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["run"], r["tip"]) for r in rows] == [("run", "cavity_tip"), ("run", "bubble_top")]
    assert rows[0]["base_verdict"] == "widens" and abs(float(rows[1]["alpha_r_j"])) < 1e-12
    assert rows[1]["base_after_k"] == "2" and rows[1]["base_verdict"] == "stationary"
    assert float(rows[1]["stationary_alpha"]) == event_table.STATIONARY_ALPHA
    # the legs of one run, joined by os.pathsep, are one run
    assert event_table.main([f"{run}{os.pathsep}{run}", "--out", str(out)]) == 0
    assert "2 events" in capsys.readouterr().out


def test_the_removals_at_a_tip_are_set_against_what_its_jet_carried(tmp_path):
    """The heads removed at each tip against what the base of that jet carried into it: the upward
    jet's heads move up and the downward jet's down, so that their momenta cancel in the whole
    liquid's budget, whose share (labelled so) the standing wave dominates; signed along each jet
    they add, against the integrals of ``Q_j V_j`` and ``Q_j V_j^2/2`` over the rows in which the
    jet stands (the trapezoid rule). Each event gives its head's share of the box of its cut."""
    up = dict(_record(1, 3e-6), removed={"V": 1e-9, "Pz": 5e-8, "KE": 2e-6},
              budgets={"grid": {"before": {"Pz": 1e-6, "KE": 4e-5}}})
    down = dict(_record(2, 5e-6, tip="bubble_top"), removed={"V": 2e-9, "Pz": -4e-8, "KE": 1e-6})
    head = BUDGET_HEAD + ("Pz_removed", "momentum_z_total", "KE_removed", "kinetic_energy_total")
    budget = [(0, 0.0, 100.0, 0.0, 100.0, 0.0, -1.03e-3, 0.0, 8.0),
              (1, 1e-5, 100.0, 3e-9, 100.0, 1e-8, -1.03e-3 + 1e-8, 3e-6, 8.0)]
    rows = _rows()
    run = tmp_path / "run"
    _write_run(run, [up, down], rows, budget=budget, budget_head=head)
    events = event_table.event_rows(run)
    assert events[0]["Pz_rem_box_share"] == pytest.approx(0.05)
    assert events[0]["KE_rem_box_share"] == pytest.approx(0.05)
    assert math.isnan(events[1]["Pz_rem_box_share"])  # a record without the box's budgets
    t = np.array([r["time"] for r in rows])
    for tip, Pz, KE, V in (("cavity_tip", 5e-8, 2e-6, 1.0), ("bubble_top", 4e-8, 1e-6, 2.0)):
        row = event_table.run_row(run, tip=tip)
        flux = np.array([r[f"Q_j_{tip}"] * r[f"V_j_{tip}"] for r in rows])
        energy = 0.5 * np.array([r[f"Q_j_{tip}"] * r[f"V_j_{tip}"] ** 2 for r in rows])
        P_jet = float(np.sum(0.5 * (flux[1:] + flux[:-1]) * np.diff(t)))
        E_jet = float(np.sum(0.5 * (energy[1:] + energy[:-1]) * np.diff(t)))
        assert row["Pz_removed_tip"] == pytest.approx(Pz) and row["KE_removed_tip"] == KE
        assert row["V_removed_tip_over_delta3"] == pytest.approx(V)
        assert row["Pz_jet"] == pytest.approx(P_jet) and row["KE_jet"] == pytest.approx(E_jet)
        assert row["Pz_removed_jet_share"] == pytest.approx(Pz / P_jet)
        assert row["KE_removed_jet_share"] == pytest.approx(KE / E_jet)
        assert row["Pz_removed_share_liquid"] == pytest.approx(1e-8 / (-1.03e-3 + 1e-8))
    # a tip without events removed nothing; one whose jet never stands carried nothing
    none = event_table.run_row(run, tip="bubble_bottom")
    assert (none["Pz_removed_tip"], none["KE_removed_tip"]) == (0.0, 0.0)
    assert math.isnan(none["Pz_jet"]) and math.isnan(none["Pz_removed_jet_share"])


def test_a_base_that_stands_still_is_stationary_whatever_the_round_off_of_its_fit(tmp_path):
    """A base of constant radius and flux: its fitted slopes are round-off, of either sign with the
    radius, and within the floors (``STATIONARY_ALPHA``, ``STATIONARY_Q``) the verdict is
    ``stationary``, not ``narrows`` or ``widens``; without floors the sign of the round-off would
    decide. A trend within the floors is stationary too, and widens below them."""
    seen = set()
    for r in (4e-4, 5.1e-4, 3.7e-4, 6.3e-4, 1.234e-4):
        rows = [dict(x, r_j_cavity_tip=r, r_neck_open=0.9 * r, Q_j_cavity_tip=1.7e-4)
                for x in _rows()]
        run = tmp_path / f"r{r:g}"
        _write_run(run, [_record(1, 3e-6)], rows)
        row = event_table.run_row(run)
        assert row["base_verdict"] == "stationary", r
        assert abs(row["alpha_r_j"]) < 1e-12 and abs(row["dQ_j_rel"]) < 1e-12
        bare = event_table.run_row(run, alpha_floor=0.0, q_floor=0.0)
        seen.add(bare["base_verdict"])
    assert seen <= {"narrows", "widens", "mixed", "stationary"} and seen != {"stationary"}
    slow = tmp_path / "slow"  # r_j ~ t'^5e-4, Q_j ~ t'^2e-4
    _write_run(slow, [_record(1, 3e-6)], _rows(alpha=5e-4, q=2e-4))
    row = event_table.run_row(slow)
    assert row["base_verdict"] == "stationary" and row["alpha_r_j"] == pytest.approx(5e-4)
    assert 0.0 < row["dQ_j_rel"] < event_table.STATIONARY_Q
    finer = event_table.run_row(slow, alpha_floor=1e-4, q_floor=1e-5)
    assert finer["base_verdict"] == "widens" and finer["stationary_alpha"] == 1e-4


def _jsonl_leg(leg: Path, step, events_made):
    """The event ``leg`` of a leg's events.jsonl: where it starts and the events made before it."""
    (leg / "events.jsonl").write_text(json.dumps({"event": "restart", "step": step}) + "\n"
                                      + json.dumps({"event": "leg", "step": step,
                                                    "events_made": events_made}) + "\n")


def test_the_legs_of_a_run_are_read_as_one_continuous_run(tmp_path):
    """A run in two legs: the first makes events 1 (the cavity's tip) and 2 (the bubble's top) and
    stops at its max_events on event 3 at step 20, the second starts there with the two events made,
    writes their records as its dump holds them (the first's ghost now with a recontact it flagged),
    makes event 3 and then 4. Read together, the records of each event are the latest leg's (the
    stop record of event 3 replaced by the event made), the rows of each step are those a continuous
    run writes (the second leg's first row, the state it starts from, dropped), the stop is the last
    leg's and every event and recontact counts; the second leg alone still has every event of its
    run. A leg restarted from the state before event 2 (an arm) drops the first leg's events from
    2 on and its rows after the step it starts at, and a leg that does not record its events at its
    start drops those at or after that step."""
    first, second = tmp_path / "leg1", tmp_path / "leg2"
    head = BUDGET_HEAD
    stopped = dict(_record(3, 6e-6), outcome="max-events", reason="max-events", step=20)
    _write_run(first, [_record(1, 3e-6), _record(2, 5e-6, tip="bubble_top", z_b=4.698),
                       stopped], _rows()[:21],
               budget=[(s, 1e-6 * s, 100.0, 0.0, 100.0) for s in range(21)], budget_head=head,
               status={"state": "completed", "window_end": None,
                       "collapse": {"stop": {"reason": "max-events", "k": 3}}})
    _jsonl_leg(first, 0, 0)
    hit = {"kind": "rear", "tau": 9e-6}
    copies = [dict(_record(1, 3e-6, recontacts=[hit]), copy={"leg": 1, "of_leg": 0}),
              dict(_record(2, 5e-6, tip="bubble_top", z_b=4.698), copy={"leg": 1, "of_leg": 0})]
    made = [dict(_record(3, 6e-6, interval=3e-6), step=20), dict(_record(4, 1e-5), step=30)]
    _write_run(second, copies + made, _rows()[20:],
               budget=[(s, 1e-6 * s, 100.0 - (1e-9 if s > 20 else 0.0), 0.0, 100.0)
                       for s in range(20, 40)], budget_head=head)
    _jsonl_leg(second, 20, 2)
    run = f"{first}{os.pathsep}{second}"
    assert event_table.run_legs(run) == [first, second]
    records = event_table.event_records(run)
    assert [(r["k"], r["outcome"]) for r in records] == [(1, "made"), (2, "made"), (3, "made"),
                                                         (4, "made")]
    assert records[0]["ghost"]["recontacts"] == [hit]
    chains = event_table.join_rows(run, "chains.txt")
    assert chains["step"] == [float(s) for s in range(40)]
    assert chains["time"][20] == pytest.approx(_rows()[20]["time"])
    budget = event_table.join_rows(run, "budget.txt")
    assert budget["step"] == [float(s) for s in range(40)] and budget["volume"][20] == 100.0
    row = event_table.run_row(run)
    assert (row["legs"], row["events"], row["events_cavity_tip"], row["events_bubble_top"]) == (
        2, 4, 3, 1)
    assert row["rear_recontacts"] == 1 and row["stop"] == "max-steps"
    assert row["volume_drift"] == pytest.approx(-1e-11)
    assert row["base_verdict"] == "widens" and row["base_after_k"] == 1
    rows = event_table.event_rows(run)
    assert [r["k"] for r in rows] == [1, 2, 3, 4]
    assert rows[0]["rear_recontact_tau_over_delta2"] == pytest.approx(9.0)
    alone = event_table.run_row(second)
    assert (alone["legs"], alone["events"], alone["rear_recontacts"]) == (1, 4, 1)
    assert alone["base_verdict"] == "widens" and alone["base_after_k"] == 1
    # an arm from the state before event 2, at step 12: event 2 made again there, later
    arm = tmp_path / "arm"
    _write_run(arm, [dict(_record(1, 3e-6), copy={"leg": 1, "of_leg": 0}),
                     dict(_record(2, 5.5e-6, tip="bubble_top"), step=12)], _rows()[12:30])
    _jsonl_leg(arm, 12, 1)
    both = f"{first}{os.pathsep}{arm}"
    assert [(r["k"], r["t"]) for r in event_table.event_records(both)] == [(1, 3e-6), (2, 5.5e-6)]
    assert event_table.join_rows(both, "chains.txt")["step"] == [float(s) for s in range(30)]
    # a leg without the record of its start: the events of the legs before it at or after the step
    # it starts at are dropped
    (second / "events.jsonl").unlink()
    (second / "events" / "event_003.json").unlink()
    records = event_table.event_records(run)
    assert [(r["k"], r["outcome"]) for r in records] == [(1, "made"), (2, "made"), (4, "made")]


def test_the_base_of_a_tip_is_tested_after_its_own_event_from_the_step_after_it(tmp_path):
    """The upward jet's base narrows until its own cut (event 2, at the cavity's tip, step 20) and
    widens after it, the bubble's top having been cut first (event 1, step 10): the test of the
    cavity's tip starts after its own event, and widens, where a window from the run's first event
    would hold the narrowing and say it narrows. The window starts at the rows after the event's
    step, whatever the twelve digits of the time of its own row: the row of the event's step, which
    the run writes before the cut, is never in it."""
    t_cut = 4.5603994017228565e-06  # its row prints as 4.560399401723e-06, above it
    rows = []
    for i, t in enumerate(np.linspace(3e-6, 2e-5, 40)):
        t = t_cut if i == 20 else float(t)
        tp = t - T
        r = 6e-4 * (tp / 1e-5) ** (-1.0 if i <= 20 else 0.1)
        rows.append({"step": i, "time": t, "tau": t, "tau_p": tp, "dt": 2.2e-9,
                     "r_neck_open": 0.9 * r, "r_neck_bubble": 3e-4, "r_j_cavity_tip": r,
                     "Q_j_cavity_tip": 1e-3 * (tp / 1e-5) ** (-2.0 if i <= 20 else 1.0),
                     "jet_born_cavity_tip": 1.0, "r_j_bubble_top": 4e-4, "Q_j_bubble_top": 1e-4,
                     "jet_born_bubble_top": 1.0})
    run = tmp_path / "run"
    _write_run(run, [dict(_record(1, rows[10]["time"], tip="bubble_top"), step=10),
                     dict(_record(2, t_cut), step=20)], rows)
    assert f"{t_cut:.12e}" == "4.560399401723e-06" and float(f"{t_cut:.12e}") > t_cut
    row = event_table.run_row(run)
    assert (row["base_after_k"], row["base_after_step"]) == (2, 20)
    assert row["base_verdict"] == "widens" and row["alpha_r_j"] == pytest.approx(0.1)
    assert row["base_rows"] == 19  # steps 21 to 39
    chains = event_table.join_rows(run, "chains.txt")
    assert event_table.base_test(chains, "cavity_tip", 10, DELTA)["base_verdict"] == "narrows"


def test_the_table_reads_the_outputs_of_a_run_of_the_driver(tmp_path, monkeypatch):
    """On the outputs of a run of the driver after a cut (a stand-in problem whose hanging jet's
    neck lands again and again, and a stand-in liquid cut): a row per event made, as its record
    and the run's events.txt have it, and the run's row from its run result's events."""
    from test_events_run import RegrowingHead, _run

    drv, out, _ = _run(tmp_path, monkeypatch, RegrowingHead())
    made = drv.pb.events
    (tmp_path / "run_status.json").write_text(json.dumps({"state": "completed",
                                                          "collapse": out}))
    (tmp_path / "parameters.json").write_text(json.dumps({"reconnect": {"delta": 1e-3}}))
    rows = event_table.event_rows(tmp_path)
    assert [r["k"] for r in rows] == [e["k"] for e in made]
    assert all(r["tip"] == "bubble_top" and r["outcome"] == "made" for r in rows)
    table = {int(float(r["k"])): r for r in _table(tmp_path / "events.txt")}
    for r in rows:
        assert r["r_j_over_r_n"] == pytest.approx(float(table[r["k"]]["r_j_over_r_n"]), rel=1e-9)
        assert r["tau_over_delta2"] == pytest.approx(float(table[r["k"]]["tau_over_delta2"]),
                                                     rel=1e-9)
    row = event_table.run_row(tmp_path, tip="bubble_top")
    assert row["events"] == out["events"]["made"] == len(made)
    assert row["events_bubble_top"] == len(made) and row["stop"] == "max-steps"
    assert row["T"] == pytest.approx(-1e-6)  # the stand-in's pinch time precedes its cut


def test_the_table_reads_the_legs_of_a_run_of_the_driver_together(tmp_path, monkeypatch):
    """On the two legs of a run of the driver (``test_events_run.two_legs``: event 1 made in leg 1,
    its ghost's rear recontact flagged in leg 2): read together, the event has the recontact and
    the run counts it, with the event made and its rows joined; leg 2 alone, which holds the
    record of the event made before it, has the event, the recontact and a base test after it."""
    from test_events_run import two_legs

    legs = two_legs(tmp_path, monkeypatch)
    for leg in ("leg1", "leg2"):
        (legs[leg] / "parameters.json").write_text(json.dumps({"reconnect": {"delta": 1e-3}}))
    (legs["leg2"] / "run_status.json").write_text(json.dumps({"state": "completed",
                                                              "collapse": legs["res"]}))
    run = f"{legs['leg1']}{os.pathsep}{legs['leg2']}"
    (row,) = event_table.event_rows(run)
    (hit,) = json.loads((legs["leg2"] / "events" / "event_001.json").read_text())["ghost"][
        "recontacts"]
    assert row["k"] == 1 and row["leg"] == 0 and hit["kind"] == "rear"
    assert row["rear_recontact_tau_over_delta2"] == pytest.approx(hit["tau"] / 1e-6)
    runs = event_table.run_row(run, tip="bubble_top")
    assert (runs["legs"], runs["events"], runs["rear_recontacts"]) == (2, 1, 1)
    steps = event_table.join_rows(run, "chains.txt")["step"]
    assert steps == sorted(set(steps)) and len(steps) == legs["res"]["steps"] - steps[0] + 1
    alone = event_table.run_row(legs["leg2"], tip="bubble_top")
    assert (alone["events"], alone["rear_recontacts"]) == (1, 1)
    assert alone["base_verdict"] != "no-events" and alone["base_after_k"] == 1
    assert event_table.run_row(legs["leg1"], tip="bubble_top")["rear_recontacts"] == 0


def _table(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:]]
