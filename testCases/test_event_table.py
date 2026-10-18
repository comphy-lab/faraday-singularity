"""Contract tests of the table of the events of runs after a cut (``postProcess/event_table.py``) on
synthetic run directories with known answers, and on the outputs of a run of the collapse driver
on a stand-in problem with a stand-in liquid cut. No finite-element problem is built."""

from __future__ import annotations

import csv
import json
import math
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
                 "r_j_cavity_tip", "Q_j_cavity_tip", "jet_born_cavity_tip", "r_j_bubble_top",
                 "Q_j_bubble_top", "jet_born_bubble_top")


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
    lines += ["\t".join(f"{r[c]:.12e}" for c in CHAIN_COLUMNS) for r in rows]
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
        out.append({"step": i, "time": t, "tau": t, "tau_p": tp, "dt": 2.2e-9,
                    "r_neck_open": 0.9 * r, "r_neck_bubble": 3e-4, "r_j_cavity_tip": r,
                    "Q_j_cavity_tip": 1e-3 * (max(tp, 1e-9) / 1e-5) ** q,
                    "jet_born_cavity_tip": 1.0, "r_j_bubble_top": 4e-4, "Q_j_bubble_top": 1e-4,
                    "jet_born_bubble_top": 1.0})
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
    assert row["Pz_removed_share"] == pytest.approx(0.04)
    assert row["KE_removed_share"] == pytest.approx(0.025)
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
    assert math.isnan(row["Pz_removed_share"])  # a budget without the columns: none
    late = event_table.run_row(wide, after_events=3)
    assert late["base_verdict"] == "no-events" and late["base_rows"] == 0
    start = event_table.run_row(wide, after_events=0)
    assert start["base_rows"] > row["base_rows"] - 1 and start["base_verdict"] == "widens"
    few = tmp_path / "few"
    _write_run(few, records, _rows(n=4))
    assert event_table.run_row(few)["base_verdict"] == "insufficient"


def test_the_command_line_writes_both_tables(tmp_path, capsys):
    run = tmp_path / "run"
    _write_run(run, [_record(1, 3e-6)], _rows())
    out, runs = tmp_path / "events.csv", tmp_path / "runs.csv"
    assert event_table.main([str(run), "--out", str(out), "--runs", str(runs)]) == 0
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["k"] for r in rows] == ["1"] and tuple(rows[0]) == event_table.EVENT_FIELDS
    with runs.open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["base_verdict"] == "widens" and tuple(row) == event_table.RUN_FIELDS
    assert "1 events" in capsys.readouterr().out
    # a row per run and tip: the upward jet and the downward one, whose base stands still
    assert event_table.main([str(run), "--out", str(out), "--runs", str(runs), "--tip",
                             "cavity_tip", "bubble_top"]) == 0
    with runs.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["run"], r["tip"]) for r in rows] == [("run", "cavity_tip"), ("run", "bubble_top")]
    assert rows[0]["base_verdict"] == "widens" and abs(float(rows[1]["alpha_r_j"])) < 1e-12


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


def _table(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0][2:].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:]]
