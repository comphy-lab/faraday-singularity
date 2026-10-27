#!/usr/bin/env python3
"""Tests of the similarity laws of jet birth in one collapse run, per collapse cycle.

Usage::

    python postProcess/collapse_analysis.py RUN_DIR --out ANALYSIS_DIR \
        [--rj-window 3e-3 3e-2] [--bins 12] [--inner-factor 10] [--outer-radius 1] \
        [--angle-tol 2] [--min-decades 0.5] [--zoom 0.6] [--no-plot]

Three similarity laws can govern the birth of the jet (``modelComparisonCases/
inertial_similarity.py``): the inertial cone, ``r_j ~ tau^alpha(beta)`` with
``alpha = 1/(2 - nu(beta))``, where the velocity potential vanishes on the cone in
``r_j << r << 1`` at inception; the line sink of a truncated cone, ``alpha = 1/2``; and the
inertio-capillary balance, ``alpha = 2/3``. The analysis reports the evidence for each; it does not
decide between them.

Each collapse cycle with a jet is analysed on its own (a table without the column ``cycle`` is one
cycle); for a cycle without a jet only the largest trough curvature is reported, the largest
positive ``z''(0)``: a bump or crest on the axis has ``z''(0) < 0``. The row at which the jet has
gone, which the driver writes with ``jet_born`` still set before the next cycle, is not part of the
jet. The **growth** of the jet runs from the smallest ``r_j`` after birth before ``r_j`` first
rises by more than ``GROWTH_MIN_RISE`` above its running minimum (the base may close in on the axis
first, with smaller upticks), at the reference instant ``t_ref`` (the birth itself when ``r_j``
grows from there), to the largest ``r_j`` before it first falls by more than ``GROWTH_MAX_DROP``
below its running maximum (smaller downticks come from rebuilds of the mesh, which move the refined
base) or reaches ``GROWTH_CAP r_search``, beyond which the search for the base pins it; its
``max_V_j`` counts only samples with flux into the jet. The run-level ``max_V_j`` is, as in
``collapse_table.py``'s summary, that of the first jet born within the record, over its rows with
a jet base and flux into the jet.
The fit samples are the samples of the growth with ``r_j`` in the stated window and a positive flux
``Q_j``. A jet whose fit samples fill fewer than ``MIN_BINS`` bins of ``ln r_j``, or span fewer
than ``min_decades`` decades of ``r_j``, is not ``ok`` and its status says so; its slopes and fits
are still reported where they exist.

* **Time-free scaling**, the primary test (arXiv:2607.08972, p. 3): ``ln Q_j``, ``ln We_j`` and
  ``ln q_j``, with ``q_j = V_j r_j``, are fitted against ``ln r_j`` through the means over equal
  intervals of ``ln r_j``, so that the small adaptive steps do not weight the small radii. As
  ``Q_j = pi r_j q_j`` and ``We_j = q_j^2/r_j``, the three slopes are one measurement; it implies
  ``alpha = 1/(2 - s_q)``, and ``s_q`` estimates ``nu`` for the cone (0 for the line sink, 1/2 for
  the inertio-capillary law). Reported with running slopes over three bins, the sensitivity to the
  number of bins and to trimming a quarter of the range at either end, the level and trend of
  ``We_j``, ``Re_j = q_j/Oh``, and the misfit of each law with its slope imposed.
* **Cone angle at inception**: on the free-surface profile nearest ``t_ref`` (the ``jet-born``
  profile when the jet grows from birth), ``beta`` is fitted as a straight cone over a fixed
  intermediate window from ``r_lo = inner_factor max(r_j, 1/|z''(0)|, h_near)`` to the largest
  radius ``r_hi <= outer_radius`` over which the tangent angle varies by at most ``angle_tol``.
  Without such a window (``r_lo >= outer_radius`` or ``r_hi < 2 r_lo``), or with fewer than
  four fit nodes or no rising cone slope (status ``no cone fit``), ``beta`` is not measured.
  The tangent angles at five radii across the window, their spread and the range of
  ``alpha(beta)`` it implies are reported. The median of the driver's per-step ``beta_fit``, from a
  window that moves with the local scale, is kept as a diagnostic only.
* **Wall kinematics** on the same profile and window, and on its halves in ``ln r``
  (``collapse.wall_kinematics``): the median ``|cos|`` between the surface velocity and the wall
  normal (1 on the cone, ``cos(beta)`` for the horizontal inflow of a line sink) and the slope of
  ``ln|u|`` against ``ln r`` (``nu - 1`` and ``-1``).
* **Regimes**: ``alpha_cone = alpha(beta)`` with ``beta_below_taylor``, which ``We_j`` needs to
  diverge in the cone family and the cone law needs as ``r_j -> 0`` (necessary, not sufficient),
  beside ``alpha_line_sink`` and ``alpha_inertio_capillary``.
* **Time fits**, secondary. The singular time is the driver's apex-based estimate (``t0_est``
  with ``t0_source == 1``) of the series that led to the birth, carried by the ``jet-born`` event
  or, when the event has none, the last one in the table unless a new apex series began after it
  (:func:`apex_estimate`; the driver's cap of the local scale is ``options.ell_cap``). It is
  checked against the bracket from the largest trough curvature before birth in the trough episode
  in progress at birth, a measured event, to ``t_ref``. A record restarted within that episode and
  cycle lacks the rows before the restart; the maximum the driver carried across it in the
  ``jet-born`` event is then used when it lies before the first row and within the episode,
  exceeds the maximum over the rows, and the ``restart`` event records ``max_curvature_reset``
  false (``max_axis_curvature_source``; a restart by an older driver is noted). ``alpha`` is
  fitted alone with the origin fixed at each of the three, and the width of the bracket is given
  in units of the smallest ``t - t_ref`` among the fit samples. Last comes the joint fit of
  ``r_j = A (t - t0_fit)^alpha`` (:func:`fit_power_law`): ``t0_fit`` is only the origin of the
  fitted law, a virtual origin before birth when the jet is born at a finite radius.

Every window, threshold and count is written to the summary, with the resolution ``h_near`` over
the fit samples. Figures: free-surface profiles near the axis around the first birth, and the time
series of the apex, the jet base and the resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "modelComparisonCases"))
sys.path.insert(0, str(HERE.parent / "src"))

from collapse_table import jet_rows, trough_episodes  # noqa: E402
from faraday_singularity.collapse import (Profile, axis_curvature, cone_angles,  # noqa: E402
                                          cone_fit, jet_base, wall_kinematics)
from inertial_similarity import (BETA_TAYLOR, INERTIO_CAPILLARY, LINE_SINK,  # noqa: E402
                                 alpha_of_beta, exponents)

GROWTH_CAP = 0.9  # r_j above this fraction of r_search is pinned by the search for the base
GROWTH_MIN_RISE = 0.1  # a rise of r_j by this fraction above its running minimum is the growth
GROWTH_MAX_DROP = 0.01  # a fall of r_j by this fraction below its running maximum ends it
MIN_SAMPLES = 6  # fewest fit samples for any fit
MIN_BINS = 3  # fewest filled bins of ln r_j for a slope


def read_table(path: Path) -> dict[str, np.ndarray]:
    """The columns of ``collapse.txt``, empty for a table with only its header."""
    with path.open() as fh:
        header = fh.readline().lstrip("#").strip().split("\t")
        if not any(line.strip() and not line.startswith("#") for line in fh):
            return {name: np.empty(0) for name in header}
    data = np.loadtxt(path, comments="#", ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def profile_headers(run: Path) -> list[tuple[dict, Path]]:
    out = []
    for f in sorted((run / "profiles").glob("profile_*.txt")):
        with f.open() as fh:
            out.append((json.loads(fh.readline()[2:]), f))
    return out


def read_profiles(run: Path) -> list[tuple[dict, np.ndarray]]:
    return [(header, np.loadtxt(f)) for header, f in profile_headers(run)]


def fit_power_law(t: np.ndarray, y: np.ndarray, t_first: float, span: float) -> dict:
    """Best ``y = A (t - t0_fit)^alpha`` with ``t0_fit = t_first - offset`` in ``[0, span]``.

    For each trial offset ``log y`` is fitted linearly in ``log(t - t0_fit)``, with the root-mean-
    square misfit in ``log y`` as the residual. The trial offsets are geometric from ``1e-7 span``
    to ``span``, dense next to ``t_first`` where the origin of a jet born at a small radius lies,
    with offset 0 added when the first sample is after ``t_first``; the best trial is polished by a
    bounded minimisation over ``log(offset)`` between its neighbours. ``t0_at_bound`` marks a best
    trial at either end of the range, where the residual may still fall outside it: ``t0_fit`` and
    ``alpha`` are then not an optimum. ``log_A`` is returned rather than ``A``, which can overflow.
    Returns ``{}`` when no trial origin lies before every sample.
    """
    t = np.asarray(t, dtype=float)
    ly = np.log(np.asarray(y, dtype=float))

    def misfit(offset: float) -> tuple[float, float, float]:
        x = t - (t_first - offset)
        if np.any(x <= 0):
            return math.inf, math.nan, math.nan
        lx = np.log(x)
        p, log_a = np.polyfit(lx, ly, 1)
        return float(np.sqrt(np.mean((ly - (p * lx + log_a)) ** 2))), float(p), float(log_a)

    offsets = span * np.geomspace(1e-7, 1.0, 600)
    if t[0] > t_first:
        offsets = np.concatenate([[0.0], offsets])
    res = np.array([misfit(o)[0] for o in offsets])
    valid = np.nonzero(np.isfinite(res))[0]
    if valid.size == 0:
        return {}
    k = int(np.argmin(res))
    best = float(offsets[k])
    lo = float(offsets[k - 1]) if k - 1 in valid and offsets[k - 1] > 0 else best
    hi = float(offsets[min(k + 1, len(offsets) - 1)])
    if lo > 0 and hi > lo:
        opt = minimize_scalar(lambda u: misfit(math.exp(u))[0], method="bounded",
                              bounds=(math.log(lo), math.log(hi)), options={"xatol": 1e-10})
        if opt.fun < res[k]:
            best = float(math.exp(opt.x))
    rms, alpha, log_a = misfit(best)
    return {"t0_fit": t_first - best, "offset": best, "alpha": alpha, "log_A": log_a,
            "log_rms": rms, "t0_at_bound": bool(k == valid[0] or k == len(offsets) - 1)}


def fit_exponent(t: np.ndarray, y: np.ndarray, t0: float) -> dict:
    """``y = A (t - t0)^alpha`` with the origin ``t0`` fixed, over the samples after it."""
    tau = np.asarray(t, dtype=float) - t0
    sel = tau > 0
    if not math.isfinite(t0) or sel.sum() < 3:
        return {}
    lx, ly = np.log(tau[sel]), np.log(np.asarray(y, dtype=float)[sel])
    p, log_a = np.polyfit(lx, ly, 1)
    rms = float(np.sqrt(np.mean((ly - (p * lx + log_a)) ** 2)))
    return {"t0": float(t0), "alpha": float(p), "log_A": float(log_a), "log_rms": rms,
            "points": int(sel.sum()), "min_tau": float(tau[sel].min())}


# --- time-free scaling ----------------------------------------------------------------------------

def log_bins(x: np.ndarray, ys: dict[str, np.ndarray], bins: int):
    """Means of ``ln x`` and of each ``ln y`` over ``bins`` equal intervals of ``ln x``.

    The intervals span the samples; empty ones are dropped. Returns ``(X, {name: Y}, counts)``.
    """
    lx = np.log(x)
    edges = np.linspace(lx.min(), lx.max(), bins + 1)
    idx = np.clip(np.searchsorted(edges, lx, side="right") - 1, 0, bins - 1)
    filled = [b for b in range(bins) if np.any(idx == b)]
    X = np.array([lx[idx == b].mean() for b in filled])
    Y = {k: np.array([np.log(v)[idx == b].mean() for b in filled]) for k, v in ys.items()}
    return X, Y, [int(np.sum(idx == b)) for b in filled]


def slope_alpha(s_q: float) -> float:
    """``alpha`` implied by the slope of ``ln q_j`` against ``ln r_j``: ``s_q = 2 - 1/alpha``."""
    return 1.0 / (2.0 - s_q) if s_q < 2.0 else math.nan


def time_free(r_j, Q_j, We_j, V_j, Oh: float, bins: int, nu_cone: float | None) -> dict:
    """Free slopes of ``ln Q_j``, ``ln We_j`` and ``ln q_j`` in ``ln r_j`` on equal-log bins."""
    q_j = V_j * r_j
    out = {"samples": int(r_j.size), "r_j_range": [float(r_j.min()), float(r_j.max())],
           "decades": float(math.log10(r_j.max() / r_j.min())), "bins": bins}

    def free_slope(sel, nb):
        if sel.sum() < MIN_SAMPLES:
            return None
        X, Y, _ = log_bins(r_j[sel], {"q_j": q_j[sel]}, nb)
        return float(np.polyfit(X, Y["q_j"], 1)[0]) if len(X) >= MIN_BINS else None

    X, Y, counts = log_bins(r_j, {"Q_j": Q_j, "We_j": We_j, "q_j": q_j}, bins)
    out["filled_bins"] = len(X)
    out["bin_counts"] = counts
    if len(X) < MIN_BINS:
        out["status"] = f"fewer than {MIN_BINS} filled bins"
        return out
    s = {k: float(np.polyfit(X, Y[k], 1)[0]) for k in Y}
    out.update(slope_Q_j=s["Q_j"], slope_We_j=s["We_j"], slope_q_j=s["q_j"],
               slope_identity_residual=max(abs(s["Q_j"] - 1 - s["q_j"]),
                                           abs(s["We_j"] - 2 * s["q_j"] + 1)),
               alpha_free=slope_alpha(s["q_j"]))
    out["local"] = [{"r_j": float(math.exp(X[k])),
                     "slope_q_j": float(np.polyfit(X[k - 1:k + 2], Y["q_j"][k - 1:k + 2], 1)[0])}
                    for k in range(1, len(X) - 1)]
    for row in out["local"]:
        row["alpha"] = slope_alpha(row["slope_q_j"])
    lr, lo, hi = np.log(r_j), np.log(r_j.min()), np.log(r_j.max())
    every = np.ones(r_j.size, dtype=bool)
    variants = {f"bins_{max(bins // 2, MIN_BINS)}": free_slope(every, max(bins // 2, MIN_BINS)),
                f"bins_{2 * bins}": free_slope(every, 2 * bins),
                "without_smallest_quarter": free_slope(lr >= lo + 0.25 * (hi - lo), bins),
                "without_largest_quarter": free_slope(lr <= hi - 0.25 * (hi - lo), bins)}
    sens = {k: (slope_alpha(v) if v is not None else None) for k, v in variants.items()}
    alphas = [a for a in [out["alpha_free"], *sens.values()] if a is not None and math.isfinite(a)]
    sens["alpha_range"] = [min(alphas), max(alphas)] if alphas else None
    out["sensitivity"] = sens
    order = np.argsort(r_j)
    out["We_j"] = {"min": float(We_j.min()), "median": float(np.median(We_j)),
                   "max": float(We_j.max()), "at_smallest_r_j": float(We_j[order[0]]),
                   "at_largest_r_j": float(We_j[order[-1]]),
                   "rises_as_r_j_falls": bool(s["We_j"] < 0)}
    out["Re_j"] = {"min": float(q_j.min() / Oh), "max": float(q_j.max() / Oh)}
    # each law with its slope of ln q_j imposed and only the prefactor fitted
    laws = {"free": s["q_j"], "line_sink": LINE_SINK["q_j_rj"],
            "inertio_capillary": INERTIO_CAPILLARY["q_j_rj"]}
    if nu_cone is not None:
        laws["cone"] = nu_cone
    out["laws"] = {}
    for name, slope in laws.items():
        resid = Y["q_j"] - slope * X
        out["laws"][name] = {"slope_q_j": float(slope),
                             "log_rms": float(np.std(resid))}  # prefactor = mean(resid)
    return out


# --- inception profile ----------------------------------------------------------------------------

def inception(run: Path, tab: dict, t_ref: float, settings: dict, inner_factor: float,
              outer_radius: float, angle_tol: float, max_offset: float = math.inf) -> dict:
    """Cone angle and wall kinematics on the profile nearest ``t_ref`` (see the module notes)."""
    headers = profile_headers(run)
    if not headers:
        return {"status": "no profiles"}
    times = np.array([h["time"] for h, _ in headers])
    exact = [i for i, (h, _) in enumerate(headers) if h.get("reason") == "jet-born"
             and abs(h["time"] - t_ref) <= 1e-9 * max(1.0, abs(t_ref))]
    i = exact[0] if exact else int(np.argmin(np.abs(times - t_ref)))
    header, path = headers[i]
    offset = float(header["time"] - t_ref)
    if not exact and abs(offset) > max_offset:
        return {"status": "profile too far from growth reference", "profile_offset": offset,
                "max_profile_offset": max_offset}
    data = np.loadtxt(path, ndmin=2)
    r, z, ur, uz = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    if r[0] > r[-1]:
        r, z, ur, uz = r[::-1], z[::-1], ur[::-1], uz[::-1]
    p = Profile(r.copy(), z.copy())
    kappa = axis_curvature(p)
    jb = jet_base(p, settings.get("r_search", 1.5), settings.get("rise_tol", 1e-6))
    row = int(np.argmin(np.abs(tab["time"] - header["time"])))
    h_near = float(tab["h_near"][row])
    apex_radius = 1.0 / abs(kappa) if math.isfinite(kappa) and kappa != 0 else math.nan
    r_j = jb.r_j if jb.index > 0 else 0.0
    scales = [x for x in (r_j, apex_radius, h_near) if math.isfinite(x)]
    r_lo = inner_factor * max(scales) if scales else math.nan
    out = {"profile": path.name, "profile_reason": header.get("reason"),
           "profile_time": float(header["time"]), "profile_offset": float(header["time"] - t_ref),
           "r_j": r_j, "axis_curvature": kappa, "h_near": h_near, "base_index": jb.index,
           "inner_factor": inner_factor, "outer_radius": outer_radius,
           "angle_tol_deg": angle_tol, "r_lo": r_lo, "max_profile_offset": max_offset}
    if not r_lo < outer_radius:
        out["status"] = "no intermediate window: r_lo >= outer_radius"
        return out
    probe = np.geomspace(r_lo, outer_radius, 33)
    ang = np.degrees(cone_angles(p, jb.index, probe))
    out["probe"] = {"r": probe.tolist(), "angle_deg": ang.tolist()}
    k_hi = 0
    for k in range(1, probe.size):
        seg = ang[:k + 1]
        if not np.all(np.isfinite(seg)) or seg.max() - seg.min() > angle_tol:
            break
        k_hi = k
    r_hi = float(probe[k_hi])
    out["r_hi"] = r_hi
    if not math.isfinite(ang[0]) or r_hi < 2 * r_lo:
        out["status"] = "no intermediate window: tangent angle not constant over a factor 2"
        return out
    beta, z0, points = cone_fit(p, jb.index, r_lo, r_hi)
    if not math.isfinite(beta):
        out.update(status="no cone fit: fewer than four nodes in the window or no rising slope",
                   r_hi=r_hi, cone_points=points)
        return out
    radii = np.geomspace(r_lo, r_hi, 5)
    angles = np.degrees(cone_angles(p, jb.index, radii))
    spread = ang[:k_hi + 1]
    out.update(status="ok", beta=beta, beta_deg=math.degrees(beta), z0=z0, cone_points=points,
               angles_at={f"{x:.4g}": float(a) for x, a in zip(radii, angles)},
               angle_spread_deg=float(spread.max() - spread.min()))
    lo_hi = [math.radians(spread.min()), math.radians(spread.max())]
    if all(0 < b < 0.5 * math.pi for b in lo_hi):
        out["alpha_cone_range"] = [alpha_of_beta(lo_hi[0]), alpha_of_beta(lo_hi[1])]
    mid = math.sqrt(r_lo * r_hi)
    kin = {}
    for name, (a, b) in {"window": (r_lo, r_hi), "inner_half": (r_lo, mid),
                         "outer_half": (mid, r_hi)}.items():
        w = wall_kinematics(p, ur, uz, jb.index, a, b)
        kin[name] = {"r": [a, b], "cos_normal": w.cos_normal, "speed_slope": w.speed_slope,
                     "points": w.points}
    expected = {"line_sink": {"cos_normal": math.cos(beta), "speed_slope": -1.0}}
    if 0 < beta < 0.5 * math.pi:
        expected["cone"] = {"cos_normal": 1.0, "speed_slope": exponents(beta)["nu"] - 1.0}
    kin["expected"] = expected
    out["wall_kinematics"] = kin
    return out


# --- per cycle ------------------------------------------------------------------------------------

def cycle_rows(tab: dict) -> list[np.ndarray]:
    """Row indices of each collapse cycle; a table without the column ``cycle`` is one cycle."""
    if "cycle" not in tab:
        return [np.arange(tab["time"].size)]
    return [np.nonzero(tab["cycle"] == c)[0] for c in np.unique(tab["cycle"])]


def growth(r_j: np.ndarray, r_cap: float) -> tuple[int, int, str]:
    """``(k_ref, k_end, ended_by)`` of the rise of the jet radius ``r_j`` after birth.

    The rise starts once the base has stopped closing in on the axis, at the smallest ``r_j``
    before ``r_j`` first exceeds its running minimum by more than ``GROWTH_MIN_RISE``, so a small
    uptick while the base closes does not start it. It lasts while ``r_j`` stays below ``r_cap``
    and above ``1 - GROWTH_MAX_DROP`` times its running maximum since the start, and ends at that
    maximum: a rebuild of the mesh that moves the refined base by a little does not end it, while a
    later fall of a jet that retracts does not move it; ``k_end < k_ref`` when the first sample is
    not below ``r_cap``. ``ended_by`` names what ended the scan.
    """
    ok = np.isfinite(r_j) & (r_j < r_cap)
    m = r_j.size if ok.all() else int(np.argmin(ok))  # first sample that is not below r_cap
    if m == 0:
        return 0, -1, "r_search"
    head = r_j[:m]
    rise = np.flatnonzero(head > (1.0 + GROWTH_MIN_RISE) * np.minimum.accumulate(head))
    k_ref = int(np.argmin(head[:rise[0]] if rise.size else head))
    k = k_end = k_ref
    while k + 1 < m and r_j[k + 1] >= (1.0 - GROWTH_MAX_DROP) * r_j[k_end]:
        k += 1
        if r_j[k] > r_j[k_end]:
            k_end = k
    if k + 1 == r_j.size:
        return k_ref, k_end, "end of cycle"
    if k + 1 < m:
        return k_ref, k_end, "decrease"
    return k_ref, k_end, "r_search" if np.isfinite(r_j[m]) else "undefined r_j"


def episodes(params: dict, tab: dict, events: list[dict]) -> tuple[np.ndarray | None, dict]:
    """Episode index of every row and the trough episodes ``{index: start, partial, ...}``.

    As recorded, or segmented again from ``z_axis`` for runs without the column ``episode``
    (``collapse_table.trough_episodes``); ``(None, {})`` when the table allows neither.
    """
    if "episode" not in tab and ("z_axis" not in tab or "depth" not in params["dimensionless"]):
        return None, {}
    index, troughs, _ = trough_episodes(params, tab, events)
    return index, troughs


def trough_curvature(tab: dict, rows: np.ndarray) -> tuple[float, float]:
    """Largest positive apex curvature ``z''(0)`` over ``rows`` in the cavity, and its time.

    Positive curvature is a trough on the axis; ``|z''(0)|`` would also take bumps and crests,
    ``z''(0) < 0``. NaNs when no row has a trough on the axis.
    """
    if "cavity" in tab:
        rows = rows[tab["cavity"][rows] > 0]
    rows = rows[tab["axis_curvature"][rows] > 0]
    if rows.size == 0:
        return math.nan, math.nan
    j = rows[int(np.argmax(tab["axis_curvature"][rows]))]
    return float(tab["axis_curvature"][j]), float(tab["time"][j])


def _number(x) -> float:
    return math.nan if x is None else float(x)


def apex_estimate(tab: dict, before: np.ndarray, i_birth: int,
                  ell_cap: float) -> tuple[float, float]:
    """``(t0, alpha)`` of the driver's last apex-based estimate over ``before``, from the table.

    ``before`` are the rows of the apex series that can lead to the birth at row ``i_birth``: of
    its cycle, from the start of the trough episode in progress at birth. The estimate is the one of
    the last row with ``t0_source`` 1, unless a new apex series begins after it and before birth: a
    row that the driver samples, a trough apex on the axis (``jet_born`` 0, ``r_j`` 0) with
    ``z''(0) > 1/ell_cap``, after one that it does not sample. Rows without a trough apex in between,
    such as a bump growing into the jet, only end the series. Without ``ell_cap`` any trough apex
    counts. NaNs when no estimate remains.
    """
    if "t0_source" not in tab or before.size == 0:
        return math.nan, math.nan
    est = before[tab["t0_source"][before] == 1]
    if est.size == 0:
        return math.nan, math.nan
    j = int(est[-1])
    between = np.arange(j, i_birth)
    threshold = 1.0 / ell_cap if math.isfinite(ell_cap) and ell_cap > 0 else 0.0
    sampled = ((tab["jet_born"][between] == 0) & (tab["r_j"][between] == 0)
               & (tab["axis_curvature"][between] > threshold))
    if np.any(sampled[1:] & ~sampled[:-1]):
        return math.nan, math.nan
    return float(tab["t0_est"][j]), float(tab["alpha_est"][j])


def singular_time(tab: dict, rows: np.ndarray, i_birth: int, event: dict | None,
                  episode: np.ndarray | None, troughs: dict, *, ell_cap: float = math.nan,
                  restart: dict | None = None) -> dict:
    """The apex-based singular time of the cycle and the largest trough curvature before birth.

    Both are taken over the rows before birth from the start of the trough episode in progress at
    birth (``episode`` of every row and the ``troughs``), or over the cycle without episodes. The
    singular time is the driver's estimate carried by the ``jet-born`` event. When the event has
    none (no event, a run that predates the field, or a driver that discarded the estimate at every
    row without a trough apex, including a bump growing into the jet), it comes from the table
    (:func:`apex_estimate`), which gives NaN exactly where a driver that keeps the estimate across
    such rows does.

    When the cycle began before the record (a restart), the event's largest trough curvature, which
    the driver carried across the restart, is taken instead of the rows' if it lies before the first
    row and, when the start of the trough episode is known, after it, and is larger, but only when
    the ``restart`` event records ``max_curvature_reset`` false: only then did the driver restore a
    dump that holds the largest trough curvature of the episode. A driver that predates that field
    restores any dump's value, which may be the largest ``|z''(0)|`` over the cycle, a bump included
    (``max_axis_curvature_note``). A trough whose start the record does not know (``partial``)
    keeps the rows' maximum.
    """
    ev = event or {}
    before = rows[rows < i_birth]
    t_start = -math.inf  # the window starts with the cycle
    if episode is not None:
        ep = int(episode[i_birth])
        start = max((k for k in troughs if k <= ep), default=ep)
        before = before[episode[before] >= start]
        info = troughs.get(start)
        t_start = info["t_start"] if info and not info["partial"] else math.nan
    t0, alpha, source = _number(ev.get("t0_apex")), _number(ev.get("alpha_apex")), "jet-born"
    if not math.isfinite(t0):
        (t0, alpha), source = apex_estimate(tab, before, i_birth, ell_cap), "table"
    k_max, t_k = trough_curvature(tab, before)
    origin = "table"
    k_ev, t_ev = _number(ev.get("max_curvature")), _number(ev.get("max_curvature_time"))
    out = {}
    if rows[0] == 0 and k_ev > 0 and t_start <= t_ev < tab["time"][0] and not k_ev <= k_max:
        if restart is not None and restart.get("max_curvature_reset") is False:
            k_max, t_k, origin = k_ev, t_ev, "jet-born"
        elif restart is None or "max_curvature_reset" not in restart:
            out["max_axis_curvature_note"] = ("restart by a driver that predates "
                                              "max_curvature_reset: the curvature it carried "
                                              "across the restart is not used")
    return {"t0_apex": t0, "alpha_apex": alpha, "source": source if math.isfinite(t0) else None,
            "max_axis_curvature": k_max, "max_axis_curvature_time": t_k,
            "max_axis_curvature_source": origin if math.isfinite(k_max) else None, **out}


def analyse_jet(run: Path, tab: dict, rows: np.ndarray, events: list[dict], settings: dict,
                opts: dict, Oh: float, episode: np.ndarray | None, troughs: dict,
                ell_cap: float = math.nan) -> dict | None:
    jet = rows[tab["jet_born"][rows] > 0]
    if jet.size == 0:
        return None
    i_birth = int(jet[0])
    born = jet_rows(tab, rows, events)
    t, r = tab["time"], tab["r_j"]
    out = {"cycle": int(tab["cycle"][i_birth]) if "cycle" in tab else None,
           "birth_observed": bool(i_birth > rows[0]), "t_birth": float(t[i_birth]),
           "r_j_birth": float(r[i_birth]),
           "min_r_j_after_birth": float(np.nanmin(r[born])) if born.size else math.nan}
    if not out["birth_observed"]:
        out["t_birth"] = None  # born before the first row of the table (a restart)
    if born.size == 0:
        out["status"] = "no jet base after birth"
        out["inception"] = {"status": out["status"]}
        return out
    r_cap = GROWTH_CAP * settings.get("r_search", 1.5)
    k_ref, k_end, ended_by = growth(r[born], r_cap)
    seg = born[k_ref:k_end + 1]
    out["growth"] = {"reference": "birth" if k_ref == 0 else "r_j minimum",
                     "t_ref": float(t[born[k_ref]]), "r_j_ref": float(r[born[k_ref]]),
                     "samples": int(seg.size), "ended_by": ended_by, "r_cap": r_cap}
    if seg.size:
        up = seg[np.isfinite(tab["Q_j"][seg]) & (tab["Q_j"][seg] > 0)]  # flux into the jet
        out["growth"].update(t_end=float(t[seg[-1]]), r_j_end=float(r[seg[-1]]),
                             max_V_j=float(tab["V_j"][up].max()) if up.size else math.nan)
    t_ref = float(t[born[k_ref]])
    lo, hi = opts["rj_window"]
    Q = tab["Q_j"][seg]
    inside = (r[seg] >= lo) & (r[seg] <= hi)
    sel = seg[inside & np.isfinite(Q) & (Q > 0)]
    out["rj_window"] = [lo, hi]
    out["fit_points"] = int(sel.size)
    out["excluded"] = {"flux_not_found": int(np.sum(inside & ~np.isfinite(Q))),
                       "flux_not_positive": int(np.sum(inside & np.isfinite(Q) & (Q <= 0)))}
    event = None
    if out["birth_observed"]:
        cands = [e for e in events if e["event"] == "jet-born"
                 and abs(e["time"] - t[i_birth]) <= 1e-9 * max(1.0, abs(t[i_birth]))]
        event = cands[0] if cands else None

    # cone angle and wall kinematics at inception
    if seg.size < 2:
        inc = {"status": "r_j does not grow after birth"}
    elif not out["birth_observed"] and k_ref == 0:
        inc = {"status": "the jet was born before the first row of the table"}
    else:
        elapsed = t[seg] - t_ref
        positive = elapsed[elapsed > 0]
        max_offset = float(positive.min()) if positive.size else 0.0
        inc = inception(run, tab, t_ref, settings, opts["inner_factor"], opts["outer_radius"],
                        opts["angle_tol"], max_offset=max_offset)
    out["inception"] = inc
    beta = inc.get("beta", math.nan)
    cone = None
    if math.isfinite(beta) and 0 < beta < 0.5 * math.pi:
        cone = exponents(beta)
    out["regimes"] = {"cone": cone, "line_sink": LINE_SINK, "inertio_capillary": INERTIO_CAPILLARY}
    out["alpha_cone"] = cone["alpha"] if cone else None
    out["beta_below_taylor"] = bool(beta < BETA_TAYLOR) if cone else None
    out["alpha_line_sink"] = LINE_SINK["alpha"]
    out["alpha_inertio_capillary"] = INERTIO_CAPILLARY["alpha"]

    restart = next((e for e in events if e["event"] == "restart"), None)
    st = singular_time(tab, rows, i_birth, event, episode, troughs, ell_cap=ell_cap,
                       restart=restart)
    st["bracket"] = [st["max_axis_curvature_time"], t_ref]
    st["in_bracket"] = (bool(st["bracket"][0] <= st["t0_apex"] <= t_ref)
                        if math.isfinite(st["t0_apex"]) and math.isfinite(st["bracket"][0])
                        else None)
    out["singular_time"] = st
    if seg.size < 2:
        out["status"] = "r_j does not grow after birth"
        return out
    if sel.size < MIN_SAMPLES:
        out["status"] = f"fewer than {MIN_SAMPLES} fit samples"
        return out
    out["status"] = "ok"
    h_near = tab["h_near"][sel]
    out["resolution"] = {"h_near_min": float(np.nanmin(h_near)),
                         "h_near_max": float(np.nanmax(h_near)),
                         "r_j_over_h_near_min": float(np.nanmin(r[sel] / h_near)),
                         "r_j_over_h_near_median": float(np.nanmedian(r[sel] / h_near))}
    out["time_free"] = time_free(r[sel], tab["Q_j"][sel], tab["We_j"][sel], tab["V_j"][sel], Oh,
                                 opts["bins"], cone["nu"] if cone else None)
    if "status" in out["time_free"]:  # the primary test failed
        out["status"] = out["time_free"]["status"]
    elif out["time_free"]["decades"] < opts["min_decades"]:
        out["status"] = f"r_j spans fewer than {opts['min_decades']:g} decades"
    beta_fit = tab["beta_fit"][sel]
    out["beta_fit_median_deg_diagnostic"] = (float(np.degrees(np.nanmedian(beta_fit)))
                                             if np.any(np.isfinite(beta_fit)) else None)

    # time fits, secondary
    ts, ys = t[sel], r[sel]
    fixed = {name: fit_exponent(ts, ys, t0) or None
             for name, t0 in (("singular_time", st["t0_apex"]),
                              ("max_axis_curvature", st["max_axis_curvature_time"]),
                              ("reference", t_ref))}
    tau = ts - t_ref
    min_tau = float(tau[tau > 0].min()) if np.any(tau > 0) else math.nan
    width = t_ref - st["max_axis_curvature_time"]
    out["time_fits"] = {"fixed_origin": fixed, "bracket_width": width, "min_tau": min_tau,
                        "bracket_over_min_tau": width / min_tau if min_tau > 0 else math.nan,
                        "joint": fit_power_law(ts, ys, t_ref, span=float(ts[-1] - t_ref) + 1e-12)}
    return out


def analyse(run: Path, rj_window: tuple[float, float] = (3e-3, 3e-2), bins: int = 12,
            inner_factor: float = 10.0, outer_radius: float = 1.0,
            angle_tol: float = 2.0, min_decades: float = 0.5) -> dict:
    tab = read_table(run / "collapse.txt")
    if tab["time"].size == 0:  # e.g. a restart that failed before its first step
        raise SystemExit(f"{run}: no rows in collapse.txt")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    status = json.loads((run / "run_status.json").read_text())
    params = json.loads((run / "parameters.json").read_text())
    settings_path = run / "collapse_settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    nd = params["dimensionless"]
    # the cap of the local scale, above which the driver does not sample the apex radius
    h_axis = (params.get("mesh") or {}).get("h_axis")
    ell_cap = h_axis * settings.get("elements_per_scale", 8.0) if h_axis else math.nan
    opts = {"rj_window": list(rj_window), "bins": bins, "inner_factor": inner_factor,
            "outer_radius": outer_radius, "angle_tol": angle_tol, "min_decades": min_decades}
    out = {"case_id": params["case"]["case_id"], "run": str(run), "Gamma": nd["Gamma"],
           "Oh": nd["Oh"], "a0": nd["a0"], "phase": nd.get("phase"), "status": status.get("state"),
           "stop": (status.get("collapse") or {}).get("stop"),
           "events": [e for e in events if e["event"] != "remesh"],
           "remeshes": sum(e["event"] == "remesh" for e in events),
           "min_h_focus": float(np.nanmin(tab["h_focus"])),
           "volume_drift": float(np.nanmax(np.abs(tab["volume"] / tab["volume"][0] - 1))),
           "options": {**opts, "growth_cap": GROWTH_CAP, "growth_min_rise": GROWTH_MIN_RISE,
                       "growth_max_drop": GROWTH_MAX_DROP, "min_samples": MIN_SAMPLES,
                       "min_bins": MIN_BINS, "r_search": settings.get("r_search", 1.5),
                       "ell_cap": ell_cap},
           "beta_taylor_deg": math.degrees(BETA_TAYLOR),
           "alpha_line_sink": LINE_SINK["alpha"],
           "alpha_inertio_capillary": INERTIO_CAPILLARY["alpha"]}
    born = tab["jet_born"] > 0
    if born.any():  # as collapse_table's summary: the first jet born within the record
        out["max_V_j"] = math.nan
        births = np.flatnonzero(born & ~np.concatenate([[True], born[:-1]]))
        if births.size:
            jet = jet_rows(tab, next(r for r in cycle_rows(tab) if births[0] in r), events)
            up = jet[np.isfinite(tab["Q_j"][jet]) & (tab["Q_j"][jet] > 0)]  # flux into the jet
            out["max_V_j"] = float(tab["V_j"][up].max()) if up.size else math.nan
    episode, troughs = episodes(params, tab, events)
    jets = [analyse_jet(run, tab, rows, events, settings, opts, nd["Oh"], episode, troughs,
                        ell_cap) for rows in cycle_rows(tab)]
    out["jets"] = [j for j in jets if j is not None]
    out["cycles_without_jet"] = []
    for rows in cycle_rows(tab):
        if np.any(tab["jet_born"][rows] > 0):
            continue
        k_max, t_k = trough_curvature(tab, rows)
        out["cycles_without_jet"].append(
            {"cycle": int(tab["cycle"][rows[0]]) if "cycle" in tab else None,
             "t_start": float(tab["time"][rows[0]]), "t_end": float(tab["time"][rows[-1]]),
             "max_axis_curvature": k_max, "max_axis_curvature_time": t_k})
    observed = [j["t_birth"] for j in out["jets"] if j["t_birth"] is not None]
    out["t_birth"] = observed[0] if observed else None
    return out


def clean(x):
    """``x`` with NaN and infinities as ``None`` and numpy scalars as Python numbers, for JSON."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def plot(run: Path, out_dir: Path, summary: dict, zoom: float) -> None:
    name = summary["case_id"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "serif"
    if shutil.which("latex"):
        matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
        matplotlib.rcParams["text.usetex"] = True
    else:
        matplotlib.rcParams["mathtext.fontset"] = "cm"

    def style(ax):
        ax.tick_params(which="both", direction="out", width=3, labelsize=26, pad=8)
        ax.tick_params(which="major", length=10)
        for sp in ax.spines.values():
            sp.set_linewidth(3)

    profiles = read_profiles(run)
    t_birth = summary.get("t_birth")
    times = np.array([h["time"] for h, _ in profiles])
    if t_birth is not None:
        keep = np.nonzero(np.abs(times - t_birth) <= 0.5)[0]
    else:
        keep = np.arange(max(0, len(profiles) - 30), len(profiles))
    fig, ax = plt.subplots(figsize=(12, 12))
    if keep.size:
        lo, hi = times[keep].min(), times[keep].max()
        z_ref = []
        for k in keep:
            h, data = profiles[k]
            c = plt.cm.viridis((h["time"] - lo) / max(hi - lo, 1e-12))
            ax.plot(data[:, 0], data[:, 1], "-", color=c, lw=2)
            ax.plot(-data[:, 0], data[:, 1], "-", color=c, lw=2)
            z_ref.append(data[0, 1])
        zc = float(np.median(z_ref))
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(zc - zoom, zc + zoom)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$kr$", fontsize=36, labelpad=12)
    ax.set_ylabel(r"$kz$", fontsize=36, labelpad=12)
    style(ax)
    fig.savefig(out_dir / f"{name}_profiles.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}_profiles.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    tab = read_table(run / "collapse.txt")
    t = tab["time"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True)
    axes[0, 0].plot(t, tab["z_axis"], "k-", lw=2.5)
    axes[0, 0].set_ylabel(r"$kz_{\mathrm{axis}}$", fontsize=32)
    axes[0, 1].semilogy(t, np.abs(tab["axis_curvature"]), "k-", lw=2.5)
    axes[0, 1].set_ylabel(r"$|z''(0)|/k$", fontsize=32)
    axes[1, 0].semilogy(t, tab["r_j"], "k-", lw=2.5, label=r"$kr_j$")
    axes[1, 0].semilogy(t, tab["h_focus"], "--", color="tab:blue", lw=2, label=r"$kh_{\mathrm{focus}}$")
    axes[1, 0].legend(fontsize=24, frameon=False)
    axes[1, 1].semilogy(t, np.abs(tab["V_j"]), "k-", lw=2.5, label=r"$|V_j|$")
    axes[1, 1].semilogy(t, np.abs(tab["w_axis"]), ":", color="tab:red", lw=2, label=r"$|w_{\mathrm{axis}}|$")
    axes[1, 1].legend(fontsize=24, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel(r"$t\sqrt{\sigma k^3/\rho}$", fontsize=32)
    for ax in axes.flat:
        style(ax)
        if t_birth is not None:
            ax.axvline(t_birth, color="0.6", lw=1.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_series.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}_series.png", bbox_inches="tight", dpi=110)
    plt.close(fig)




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rj-window", type=float, nargs=2, default=(3e-3, 3e-2),
                    help="range of r_j for the fits")
    ap.add_argument("--bins", type=int, default=12, help="equal-log bins of r_j")
    ap.add_argument("--inner-factor", type=float, default=10.0,
                    help="r_lo of the cone window in units of max(r_j, 1/|z''(0)|, h_near)")
    ap.add_argument("--outer-radius", type=float, default=1.0,
                    help="largest r_hi of the cone window")
    ap.add_argument("--angle-tol", type=float, default=2.0,
                    help="largest variation of the tangent angle over the cone window, in deg")
    ap.add_argument("--min-decades", type=float, default=0.5,
                    help="smallest span of r_j over the fit samples, in decades, for status ok")
    ap.add_argument("--zoom", type=float, default=0.6, help="half-width of the profile window")
    ap.add_argument("--no-plot", action="store_true", help="write the summary only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = clean(analyse(args.run, tuple(args.rj_window), args.bins, args.inner_factor,
                            args.outer_radius, args.angle_tol, args.min_decades))
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.out / f"{summary['case_id']}_summary.json").write_text(text)
    print(text, end="")
    if not args.no_plot:
        plot(args.run, args.out, summary, args.zoom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
