"""Time stepping, resolution control and observables for a collapsing cavity and its jet.

Each driver step advances the solution by ``dt``, chosen from the local scale of the collapse, in
one BDF2 step unless a failed Newton solve makes pyoomph halve it (below):

* the local scale ``ell`` is the smallest of the apex radius of curvature ``1/|z''(0)|``, the
  off-axis base radius ``r_j`` once a resolved bump develops in a cavity, and the narrowest neck, capped at
  ``h_axis elements_per_scale``, where focus refinement begins, and it grows by at most
  ``scale_growth`` per step (:func:`limit_scale`);
* the target focus element size is ``h_focus = ell/elements_per_scale``, clipped to
  ``[h_floor, h_axis]``, which the mesh follows in rebuilds of bounded refinement and coarsening
  (below);
* the step is ``dt = min(dt_max, courant h_focus/U, capillary sqrt(ell^3), scale_rate ell/|dell/dt|,
  growth dt_prev)`` with ``U`` the largest free-surface speed in the focus region and the jet speed,
  ``sqrt(ell^3)`` the capillary time of the collapsing feature, and the last term keeping the
  decrease of ``ell`` per step to about ``scale_rate``: a geometric scale (a bump vanishing at the
  apex, a neck closing) can shrink much faster than any material speed.

Under sustained forcing collapses repeat. Whenever the axis becomes the lowest point again after a
jet was born, a new cycle starts (event ``cavity``, counter ``cycle``) and the jet must be born
again. A jet whose bump has flattened back (rise below ``flat_aspect r_j``) counts as gone. The Newton
tolerance is at least ``newton_floor/dt``: the residual of the time-discrete equations has a
round-off floor that grows like ``1/dt``, so a fixed tolerance cannot be met at the smallest steps.
A cavity exists once the axis is the lowest point of the free surface near it (``cavity_seen``);
only after that can the axis rise above an off-axis minimum and the jet be born: the axis must stand
above the base by more than ``jet_aspect r_j`` with ``r_j < jet_max_radius``, so a broad rebound of
the trough does not count. Once born, the jet stays born. Before a cavity
has formed, for example while a crest stands on the axis, ``r_j`` and the jet quantities are NaN.

When the free-surface spacing near the collapse is more than ``refine_ratio`` times the target
(or below ``coarsen_ratio`` times it), the mesh is rebuilt with the focus attractors at the new
size, refining by at most ``max_refine_factor`` and coarsening by at most ``max_coarsen_factor``
per rebuild, so a growing scale coarsens the focus in bounded steps. A rebuild changes the nodes
through which the apex curvature, and so the local scale, is measured, and ``ell`` may grow by
``scale_growth`` per step however short the step. Coarsening on the reading of a freshly refined
surface could then rebuild it coarse, read it sharp and refine it again, over and over at a frozen
time, each sharp reading cutting the step. After every refining rebuild, whether or not it refined,
the focus is therefore not coarsened for ``coarsen_hold`` local times (:func:`local_time` of the
scale and speed that asked for the rebuild) of physical time; refinement and the stops are never
held. After a refining rebuild that did not refine, refinement also waits ``remesh_backoff`` steps.
A step whose Newton solve fails is rejected, the state restored and the step halved by pyoomph's
adaptive stepping, which the driver uses with ``dt`` as the largest step: the driver step then
reaches ``t + dt`` in several accepted sub-steps, and only its end is measured. Element distortion
still triggers pyoomph's own remeshing, at the focus the driver last set, which a hold keeps from
coarsening; those rebuilds neither start nor end a hold. The run stops right before a pinch-off,
when a neck becomes narrower than ``max(neck_stop, neck_stop_elements h_floor)``, or at ``t_end``.

Close to a pinch-off or to the singular collapse the Newton solve can fail at every step size:
oomph-lib halves the step down to ``Problem::Minimum_dt`` (``1e-12``) and then gives up with an
``OomphLibError``, seen in Python only as ``RuntimeError: OomphException``; its message, ``Tried to
reduce dt to ... which is less than the minimum dt``, goes to the problem's log file. It leaves the
rejected iterate and the advanced time in place, so nothing is measured, dumped or written from it.
Of the failures of the solve, only this one, recognised by the exact exception and the message in
the log, can end a run: :func:`classify_failure` decides from the last state the driver measured
whether it is an expected end. A neck narrower than ``neck_guard`` stops the run as a
``pinch-off-approach``, otherwise a measured local scale (before its growth is limited) below
``ell_guard`` as a ``singularity-approach``, both with ``trigger`` ``newton-failure`` and the
oomph-lib message as ``error``. Every other
exception from the solve is raised, whatever the state: other oomph-lib errors, failures of mesh
generation or remeshing, and subclasses of ``RuntimeError``. The stop event and the ``stop`` profile
describe that measured state, the end of the last completed driver step. After a halving the
failure can come on a later sub-step of the failing driver step, so sub-steps accepted before it
are not measured, and the recorded state may precede them.

Per step ``collapse.txt`` records the cycle and episode, the apex (height, velocity, curvature),
the jet base (``r_j``, ``z_b``, flux ``Q_j``, mean speed ``V_j``, ``We_j = V_j^2 r_j``), the
narrowest neck, cone angles, the resolution, the global budgets and the power-law origin estimate.
``events.jsonl`` records cavities, jet birth (with the largest trough curvature ``z''(0) > 0``
before it in its cycle and trough episode; a bump or crest on the axis has ``z''(0) < 0``), the
first neck, crest and trough episodes, remeshes, restarts and the stop. Free-surface profiles with
velocities are written to ``profiles/`` at least every ``snapshot_dt`` and whenever ``ell`` changes
by ``snapshot_decades``.

The origin ``t0`` of a power law ``A |t - t0|^alpha`` is estimated during the run
(:func:`singular.singular_time`) from the latest ``singular_samples`` samples of a length: before
the jet is born, while the axis is the lowest point and a trough (``z''(0) > 0``), the apex radius
of curvature ``1/z''(0)``, which vanishes at the singular time; after birth, while the bump stands,
the jet-base radius ``r_j``. While the base closes in on the axis, the ``t0`` of ``r_j`` is the time
at which it would reach the axis, after the samples; while the jet grows, it is the origin of that
growth, a virtual one before birth, not a singular time, when the jet is born at a finite radius.
Either length is sampled only while it is below the cap of the local scale,
``h_axis elements_per_scale``, where it can set the local scale and with it the step; the apex
radius of a broad trough above the cap gives a well-fitted but meaningless ``t0``. A sample that
does not qualify restarts that series. Columns ``t0_est`` and ``alpha_est`` hold the estimate of
``x = A |t - t0|^alpha``, ``t0_source`` its origin (0 none, 1 apex radius, 2 jet radius, whose
``t0_est`` lies after the row's time while the base closes and before it while the jet grows); the
jet-born event carries the last apex-based estimate of the series that led to the birth, and when
it was made (``t_apex_est``). Rows without a trough apex between that series and the birth, such as
a bump growing on the axis into the jet, end the series but keep its estimate; a new apex series
discards it, so the estimate is NaN when the series restarted and gave no estimate since. Both
series restart with every cycle and every trough episode.

Cycles advance only once a jet has been born, so collapses that launch no jet share a cycle. The
axis elevation ``eta = z_axis - d`` (the mean level is the depth ``d``) is therefore also split into
crest and trough episodes with the hysteresis ``eta_hys`` (``episodes.py``): column ``episode``,
events ``crest`` and ``trough``. A trough event carries the wave height ``h_crest``, the largest
``eta`` over the crest episode just before it, its time ``t_crest``, and ``crest_jet``, whether a
jet stood on the axis during that crest: ``z_axis`` is then the jet tip, so ``h_crest`` is a wave
height only when ``crest_jet`` is false.

State dumps (``states/``) hold the pyoomph state and, in a JSON file beside it, the driver's
bookkeeping, so that a restart continues the cycles, episodes, singular-time series, step
limiters and the hold on coarsening. Dumps written before a field existed restore it to the value
of a fresh run (no hold on coarsening); the episode
in progress at such a restart is ``partial`` and its wave height unknown. The largest trough
curvature is restored only from dumps marked ``max_curvature_kind`` ``trough-episode``: older
dumps may hold the largest ``|z''(0)|`` over the cycle, a bump or crest included. Without the mark
it restarts from zero (``max_curvature_reset`` in the ``restart`` event), so the curvature that the
next ``jet-born`` event and the run status report covers only the rows after the restart.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .collapse import (Profile, axis_curvature, cone_angles, cone_fit, gauss_disc_points,
                       jet_base, necks)
from .episodes import ETA_HYS, EpisodeTracker
from .mesh import Focus
from .singular import singular_time


@dataclass(frozen=True)
class CollapseSettings:
    t_end: float
    dt_max: float
    dt_min: float = 1e-12
    courant: float = 0.2
    capillary: float = 0.2
    growth: float = 1.25
    scale_rate: float = 0.1  # largest relative change of the local scale per step
    scale_growth: float = 1.25  # largest growth factor of the local scale per step
    elements_per_scale: float = 8.0
    h_floor: float = 1e-5
    focus_extent: float = 20.0
    focus_grading: float = 0.2
    refine_ratio: float = 1.5
    coarsen_ratio: float = 0.3
    max_refine_factor: float = 4.0  # largest refinement of the focus size in one rebuild
    max_coarsen_factor: float = 2.0  # largest coarsening of the focus size in one rebuild
    min_steps_between_remesh: int = 1
    remesh_backoff: int = 20  # steps a refinement waits after a rebuild that did not refine
    coarsen_hold: float = 1.0  # local times a refined focus is not coarsened (local_time)
    r_search: float = 1.5
    rise_tol: float = 1e-6
    jet_aspect: float = 0.1  # a jet stands above its base by more than jet_aspect * r_j
    flat_aspect: float = 0.01  # a bump lower than flat_aspect * r_j has vanished: a cavity again
    newton_floor: float = 5e-10  # Newton tolerance is at least newton_floor/dt (round-off floor)
    jet_max_radius: float = 0.3  # and is narrower than this; a broad rebound is not a jet
    neck_stop: float = 1e-3
    neck_stop_elements: float = 3.0
    neck_guard: float = 1e-2  # a Newton failure with a neck narrower than this is a pinch-off
    ell_guard: float = 1e-3  # and with a local scale below this the approach to the singularity
    snapshot_dt: float = 0.25
    snapshot_decades: float = 0.05
    probe_points: int = 16
    cone_radii: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
    cone_window: tuple[float, float] = (4.0, 40.0)  # in units of the local scale
    max_steps: int = 1_000_000
    state_dt: float = 0.1  # time between state dumps (the last state_keep are kept)
    state_keep: int = 4
    singular_samples: int = 10  # samples in the running fit of the singular time
    singular_alpha_range: tuple[float, float] = (0.1, 5.0)  # exponents outside are rejected
    singular_residual: float = 0.05  # largest rms misfit of x/(dx/dt), relative to its range
    singular_extrapolation: float = 2.0  # largest distance of t0 from the samples, in window spans
    eta_hys: float = ETA_HYS  # axis elevation that starts a crest (+) or trough (-) episode

    @classmethod
    def from_dict(cls, data: dict, *, t_end: float, dt_max: float) -> "CollapseSettings":
        kw = dict(data)
        for key in ("cone_radii", "cone_window", "singular_alpha_range"):
            if key in kw:
                kw[key] = tuple(kw[key])
        return cls(t_end=t_end, dt_max=dt_max, **kw)


MIN_DT_FAILURE = "less than the minimum dt"  # oomph-lib's message when it gives up on a step
# what a dump's max_curvature holds: the largest z''(0) > 0 of the trough episode and cycle
MAX_CURVATURE_KIND = "trough-episode"


def oomph_error_message(log: str) -> str:
    """The message of the last oomph-lib error in the text of a log file, or ``""``.

    oomph-lib writes an ``OomphLibError`` as a block that opens with ``Oomph-lib ERROR`` and where
    it was raised, then a dashed rule, the message and a closing rule of ``=``.
    """
    i = log.rfind("Oomph-lib ERROR")
    if i < 0:
        return ""
    lines = [line.strip() for line in log[i:].splitlines()]
    rule = next((j for j, line in enumerate(lines) if line.startswith("-----")), None)
    if rule is None:
        return ""
    message = []
    for line in lines[rule + 1:]:
        if line.startswith("====="):
            break
        if line:
            message.append(line)
    return " ".join(message)


def limit_scale(ell: float, ell_prev: float, factor: float) -> float:
    """The local scale ``ell``, allowed to grow by at most ``factor`` on the previous step's.

    A measured scale can jump up within one step: the apex curvature crossing zero, a bump
    vanishing as a new cycle starts, or a jet base found afresh. Following the jump at once
    coarsens the focus while the feature that set the old scale is resolved only on the fine mesh;
    the coarse rebuild and the refinements after it then reconstruct the surface near the axis from
    too few nodes, which can leave a corner there. A shrinking scale is never limited.
    """
    if not (math.isfinite(ell_prev) and ell_prev > 0.0 and factor > 0.0):
        return ell
    return min(ell, factor * ell_prev)


def local_time(ell: float, U: float) -> float:
    """The time in which a feature of size ``ell`` can change: the shorter of its advective time
    ``ell/U`` and its capillary time ``sqrt(ell^3)`` (density and surface tension are 1).

    After a refining rebuild the focus is not coarsened for ``coarsen_hold`` of these, taken from
    the scale and speed that asked for the rebuild. The hold runs in physical time: steps that
    hardly advance it, however many, do not end it. The capillary time bounds it, at most
    ``sqrt(ell_cap^3)``, when the flow near the axis is slow.
    """
    return min(ell / U if U > 0.0 else math.inf, ell**1.5)


def classify_failure(m: dict, s: CollapseSettings, error: str) -> dict | None:
    """Stop record for a step whose Newton solve failed at every step size, or None to re-raise.

    ``m`` is the last state the driver measured, at the end of its last completed step, which can
    precede sub-steps accepted within the failing step. A finite neck narrower than ``neck_guard``
    makes the failure the approach to a pinch-off; otherwise a measured local scale
    ``ell_measured`` below ``ell_guard`` makes it the approach to the singular collapse (or to the
    vanishing of a feature of that size). The guard does not apply to ``ell``, whose growth is
    limited: after a small feature has gone, ``ell`` stays below the guard for several steps
    although no feature of that size is left. Anything else is not an expected end of the run.
    """
    r_neck = m.get("r_neck", math.nan)
    scale = m.get("ell_measured", m.get("ell", math.nan))
    if math.isfinite(r_neck) and r_neck < s.neck_guard:
        return {"reason": "pinch-off-approach", "trigger": "newton-failure", "r_neck": r_neck,
                "z_neck": m.get("z_neck", math.nan), "neck_guard": s.neck_guard,
                "time": m.get("time", math.nan), "error": error}
    if math.isfinite(scale) and scale < s.ell_guard:
        return {"reason": "singularity-approach", "trigger": "newton-failure",
                "ell": m.get("ell", math.nan), "ell_measured": scale,
                "axis_curvature": m.get("axis_curvature", math.nan),
                "r_j": m.get("r_j", math.nan), "z_axis": m.get("z_axis", math.nan),
                "z_b": m.get("z_b", math.nan), "V_j": m.get("V_j", math.nan),
                "jet_born": bool(m.get("jet_born", False)), "ell_guard": s.ell_guard,
                "time": m.get("time", math.nan), "error": error}
    return None


COLUMNS = ("step", "cycle", "episode", "time", "dt", "z_axis", "w_axis", "axis_curvature", "cavity",
           "jet_born", "r_j", "z_b",
           "Q_j", "V_j", "We_j", "flux_found", "r_neck", "z_neck", "ell", "h_focus", "h_near",
           "U", "beta_fit", "beta_fit_points", "beta_window_lo", "beta_window_hi",
           "volume", "area", "kinetic_energy", "dissipation", "remesh_count", "wall",
           "t0_est", "alpha_est", "t0_source")


@dataclass
class _State:
    step: int = 0
    dt_prev: float | None = None
    last_remesh_step: int = -10**9
    refine_blocked_until: int = -10**9  # first step at which a refining rebuild may follow
    coarsen_held_until: float = -math.inf  # time before which the focus is not coarsened
    last_snapshot_t: float = -math.inf
    last_snapshot_ell: float = math.nan
    snapshots: int = 0
    cavity_seen: bool = False
    born: bool = False
    cycle: int = 0
    ell_prev: float = math.nan
    t_prev: float = math.nan
    last_state_t: float = -math.inf
    states: list = field(default_factory=list)
    neck_seen: bool = False
    max_curvature: float = 0.0
    max_curvature_time: float = math.nan
    events: list = field(default_factory=list)
    apex_samples: list = field(default_factory=list)  # [t, 1/z''(0)] of the trough before birth
    jet_samples: list = field(default_factory=list)  # [t, r_j] of the standing jet
    apex_t0: float = math.nan  # last apex-based estimate of the cycle, and when it was made
    apex_alpha: float = math.nan
    apex_t_est: float = math.nan
    last_singular_t: float = -math.inf  # time of the latest sample offered to the series


class CollapseDriver:
    def __init__(self, problem, settings: CollapseSettings, outdir: Path):
        self.pb = problem
        self.s = settings
        self.outdir = Path(outdir)
        (self.outdir / "profiles").mkdir(parents=True, exist_ok=True)
        self.state = _State()
        self.episodes = EpisodeTracker(hys=settings.eta_hys)
        self._table = open(self.outdir / "collapse.txt", "w", buffering=1)
        cols = list(COLUMNS) + [f"beta_r{r:g}" for r in settings.cone_radii]
        self._table.write("# " + "\t".join(cols) + "\n")
        self._events = open(self.outdir / "events.jsonl", "w", buffering=1)
        (self.outdir / "collapse_settings.json").write_text(
            json.dumps(asdict(settings), indent=2) + "\n")

    def close(self):
        self._table.close()
        self._events.close()

    # --- measurement --------------------------------------------------------------------------

    def measure(self) -> dict:
        s, pb = self.s, self.pb
        r, z, ur, uz = pb.interface_arrays()
        prof = Profile(r, z)
        kappa = axis_curvature(prof)
        jb = jet_base(prof, s.r_search, s.rise_tol)
        rise = float(z[0]) - jb.z_b
        flat_bump = jb.index > 0 and rise <= s.flat_aspect * jb.r_j
        cavity = self.state.cavity_seen or jb.index == 0
        born = (self.state.born or (cavity and jb.jet_born and jb.r_j < s.jet_max_radius
                                    and rise > max(s.rise_tol, s.jet_aspect * jb.r_j)))
        r_j = jb.r_j if cavity else math.nan
        nk = necks(prof)
        r_neck, z_neck = (nk[0][0], nk[0][1]) if nk else (math.nan, math.nan)

        scales = []
        if math.isfinite(kappa) and kappa != 0.0:
            scales.append(1.0 / abs(kappa))
        # Resolve an emerging off-axis minimum before its rise reaches the jet-birth threshold.
        # A nearly vanished bump remains excluded so a noise minimum does not set the scale.
        if cavity and jb.jet_born and r_j > 0.0 and not flat_bump:
            scales.append(r_j)
        if nk:
            scales.append(r_neck)
        h_axis = pb.mesh_settings.h_axis
        # scales above the one at which the focus starts refining do not matter for resolution or
        # stepping; capping also keeps 1/|z''(0)| finite while the apex curvature changes sign
        ell_cap = h_axis * s.elements_per_scale
        ell_measured = min(min(scales) if scales else ell_cap, ell_cap)
        ell = limit_scale(ell_measured, self.state.ell_prev, s.scale_growth)
        h_focus = float(np.clip(ell / s.elements_per_scale, s.h_floor, h_axis))

        Q = V = We = math.nan
        found = True
        if born and r_j > 0.0:
            rr, ww = gauss_disc_points(r_j, s.probe_points)
            w, found = pb.axial_velocity_at([(x, jb.z_b) for x in rr])
            if found:
                Q = float(np.dot(ww, w))
                V = Q / (math.pi * r_j**2)
                We = V**2 * r_j

        near = r <= max(2.0 * ell, 4.0 * h_focus)
        ds = np.hypot(np.diff(r), np.diff(z))
        near_edges = near[:-1] | near[1:]
        h_near = 2.0 * float(ds[near_edges].min()) if near_edges.any() else math.inf

        focus_r = min(pb.nd.radius, s.focus_extent * ell)
        in_focus = r <= focus_r
        U = max(float(np.max(np.hypot(ur[in_focus], uz[in_focus]))) if in_focus.any() else 0.0,
                abs(V) if math.isfinite(V) else 0.0, 1e-3)

        ref = max(r_j if born else 0.0,
                  1.0 / abs(kappa) if math.isfinite(kappa) and kappa != 0 else 0.0)
        lo, hi = s.cone_window[0] * ref, s.cone_window[1] * ref
        start = jb.index if cavity else 0
        beta, _, npts = cone_fit(prof, start, lo, hi)
        angles = cone_angles(prof, start, s.cone_radii)

        flat = jb.index > 0 and rise <= s.flat_aspect * max(jb.r_j, h_focus)
        return {"time": self._t(), "axis_lowest": jb.index == 0 or flat, "flat_bump": flat_bump,
                "z_axis": float(z[0]), "w_axis": float(uz[0]), "axis_curvature": kappa,
                "cavity": cavity, "jet_born": born, "r_j": r_j,
                "z_b": jb.z_b if cavity else math.nan, "Q_j": Q, "V_j": V,
                "We_j": We, "flux_found": found, "r_neck": r_neck, "z_neck": z_neck, "ell": ell,
                "ell_measured": ell_measured, "h_focus": h_focus, "h_near": h_near, "U": U, "beta_fit": beta,
                "beta_fit_points": npts, "beta_window_lo": lo, "beta_window_hi": hi,
                "angles": angles, "focus_radius": focus_r, "profile": (r, z, ur, uz)}

    # --- bookkeeping --------------------------------------------------------------------------

    def _event(self, kind: str, m: dict, **extra):
        rec = {"event": kind, "time": m["time"], "step": self.state.step,
               "r_j": m["r_j"], "z_axis": m["z_axis"], "ell": m["ell"],
               "axis_curvature": m["axis_curvature"], **extra}
        self.state.events.append(rec)
        self._events.write(json.dumps(rec) + "\n")

    def _t(self) -> float:
        return self.pb.get_current_time(dimensional=False, as_float=True)

    def _write_row(self, m: dict, dt: float, wall: float, est: tuple[float, float, int]):
        obs = self.pb.observables()
        row = [self.state.step, self.state.cycle, self.episodes.index, m["time"], dt, m["z_axis"],
               m["w_axis"], m["axis_curvature"], int(m["cavity"]), int(m["jet_born"]), m["r_j"],
               m["z_b"], m["Q_j"], m["V_j"], m["We_j"],
               int(m["flux_found"]), m["r_neck"], m["z_neck"], m["ell"], m["h_focus"], m["h_near"],
               m["U"], m["beta_fit"], m["beta_fit_points"], m["beta_window_lo"],
               m["beta_window_hi"], obs["volume"], obs["area"], obs["kinetic_energy"],
               obs["dissipation"], self.pb.remesh_count, wall, *est] + list(m["angles"])
        self._table.write("\t".join(f"{v:.12e}" if isinstance(v, float) else str(v)
                                    for v in row) + "\n")

    def _snapshot(self, m: dict, reason: str):
        st = self.state
        r, z, ur, uz = m["profile"]
        header = {"time": m["time"], "step": st.step, "reason": reason, "r_j": m["r_j"],
                  "z_b": m["z_b"], "z_axis": m["z_axis"], "ell": m["ell"],
                  "remesh_count": self.pb.remesh_count}
        path = self.outdir / "profiles" / f"profile_{st.snapshots:06d}.txt"
        np.savetxt(path, np.column_stack([r, z, ur, uz]), header=json.dumps(header) +
                   "\nr z u_r u_z", comments="# ")
        st.snapshots += 1
        st.last_snapshot_t = m["time"]
        st.last_snapshot_ell = m["ell"]

    def _maybe_snapshot(self, m: dict, forced: str | None = None):
        st = self.state
        if forced is not None:
            self._snapshot(m, forced)
        elif self._t() - st.last_snapshot_t >= self.s.snapshot_dt:
            self._snapshot(m, "time")
        elif (math.isfinite(st.last_snapshot_ell) and m["ell"] > 0
              and abs(math.log10(m["ell"] / st.last_snapshot_ell)) >= self.s.snapshot_decades):
            self._snapshot(m, "scale")

    def _maybe_remesh(self, m: dict) -> bool:
        s, st = self.s, self.state
        if st.step - st.last_remesh_step < s.min_steps_between_remesh:
            return False
        target = m["h_focus"]
        refining = m["h_near"] > s.refine_ratio * target
        if refining:
            if st.step < st.refine_blocked_until:
                return False
            size = max(target, m["h_near"] / s.max_refine_factor)
        elif m["h_near"] < s.coarsen_ratio * target:
            if m["time"] < st.coarsen_held_until:  # held in physical time, not in steps
                return False
            size = min(target, s.max_coarsen_factor * m["h_near"])
        else:
            return False
        self.pb.active_mesh_template().focus = Focus(size=size, radius=m["focus_radius"],
                                                     grading=s.focus_grading)
        self.pb.force_remesh()
        st.last_remesh_step = st.step
        hold = {}
        if refining:
            # the next readings of the scale come from new nodes, also when the rebuild did not
            # refine: do not coarsen on them before the feature that asked for the rebuild can
            # have changed
            held = m["time"] + s.coarsen_hold * local_time(m["ell"], m["U"])
            st.coarsen_held_until = max(st.coarsen_held_until, held)
            hold["coarsen_held_until"] = st.coarsen_held_until
        after = self.measure()
        self._event("remesh", m, focus_size=size, h_near_before=m["h_near"],
                    h_near_after=after["h_near"], **hold)
        if refining and after["h_near"] > 0.9 * m["h_near"]:
            # the rebuild did not refine: do not try again on every step; a growing scale may
            # coarsen the focus once the hold has passed
            st.refine_blocked_until = st.step + s.remesh_backoff + s.min_steps_between_remesh
            self._event("remesh-ineffective", after, focus_size=size)
        return True

    def _singular(self, m: dict) -> tuple[float, float, int]:
        """Add ``m`` to the singular-time samples; return ``(t0_est, alpha_est, t0_source)``."""
        s, st = self.s, self.state
        kappa, t = m["axis_curvature"], m["time"]
        ell_cap = self.pb.mesh_settings.h_axis * s.elements_per_scale  # as in measure()
        apex = (not m["jet_born"] and m["axis_lowest"] and math.isfinite(kappa)
                and kappa > 1.0 / ell_cap)
        jet = (m["jet_born"] and not m["flat_bump"] and math.isfinite(m["r_j"])
               and 0.0 < m["r_j"] < ell_cap)
        # a restart offers the dumped sample again, also when a new cycle had cleared the series
        fresh = t > st.last_singular_t
        st.last_singular_t = max(st.last_singular_t, t)
        for samples, ok, x in ((st.apex_samples, apex, 1.0 / kappa if apex else math.nan),
                               (st.jet_samples, jet, m["r_j"])):
            if not ok:
                samples.clear()
            elif fresh and (not samples or t > samples[-1][0]):  # dumps without last_singular_t
                if samples is st.apex_samples and not samples:
                    # a new apex series begins: the estimate of the previous one is stale; rows
                    # that only end a series (a bump growing into the jet) keep its estimate
                    st.apex_t0 = st.apex_alpha = st.apex_t_est = math.nan
                samples.append([t, x])
                del samples[:-s.singular_samples]
        source, samples = (2, st.jet_samples) if jet else (1, st.apex_samples) if apex else (0, [])
        t0 = alpha = math.nan
        if len(samples) >= s.singular_samples:
            ts, xs = np.asarray(samples).T
            t0, alpha = singular_time(ts, xs, min_samples=s.singular_samples,
                                      alpha_range=s.singular_alpha_range,
                                      max_residual=s.singular_residual,
                                      max_extrapolation=s.singular_extrapolation)
        if not math.isfinite(t0):
            return math.nan, math.nan, 0
        if source == 1:
            st.apex_t0, st.apex_alpha, st.apex_t_est = t0, alpha, t
        return t0, alpha, source

    def _track(self, m: dict):
        """Advance the crest and trough episodes.

        A new trough restarts the singular-time series and the largest trough curvature.
        """
        ev = self.episodes.update(m["time"], m["z_axis"] - self.pb.nd.depth, m["jet_born"])
        if ev is None:
            return
        kind = ev.pop("event")
        if kind == "trough":
            self._reset_singular()
            self.state.max_curvature, self.state.max_curvature_time = 0.0, math.nan
        self._event(kind, m, **ev)

    def _reset_singular(self):
        st = self.state
        st.apex_samples.clear()
        st.jet_samples.clear()
        st.apex_t0 = st.apex_alpha = st.apex_t_est = math.nan

    # --- restarts -----------------------------------------------------------------------------

    _STATE_FIELDS = ("step", "cycle", "cavity_seen", "born", "neck_seen", "max_curvature",
                     "max_curvature_time", "snapshots")
    # bookkeeping added later: older dumps lack it and restore to the defaults of a fresh run
    _OPTIONAL_FIELDS = ("apex_samples", "jet_samples", "apex_t0", "apex_alpha", "apex_t_est",
                        "ell_prev", "t_prev", "last_remesh_step", "last_singular_t",
                        "refine_blocked_until", "coarsen_held_until")

    def save_state(self):
        """Dump the pyoomph state and the driver state; keep the last ``state_keep`` dumps."""
        st = self.state
        path = self.outdir / "states" / f"state_{st.step:08d}.dump"
        path.parent.mkdir(exist_ok=True)
        self.pb.save_state(str(path), quiet=True)
        focus = self.pb.active_mesh_template().focus
        meta = {k: getattr(st, k) for k in self._STATE_FIELDS + self._OPTIONAL_FIELDS}
        meta.update(time=self._t(), remesh_count=self.pb.remesh_count,
                    focus=None if focus is None else asdict(focus), dt_prev=st.dt_prev,
                    episodes=self.episodes.to_dict(), max_curvature_kind=MAX_CURVATURE_KIND)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        st.states.append(path)
        st.last_state_t = self._t()
        while len(st.states) > self.s.state_keep:
            old = st.states.pop(0)
            for f in (old, old.with_suffix(".json")):
                f.unlink(missing_ok=True)

    def restore(self, meta: dict):
        """Resume the driver bookkeeping from a state dump's metadata."""
        st = self.state
        for k in self._STATE_FIELDS:
            setattr(st, k, meta[k])
        for k in self._OPTIONAL_FIELDS:
            if k in meta:
                setattr(st, k, meta[k])
        reset = meta.get("max_curvature_kind") != MAX_CURVATURE_KIND
        if reset:  # possibly |z''(0)| of a bump, or of an earlier trough episode
            st.max_curvature, st.max_curvature_time = 0.0, math.nan
        if "refine_blocked_until" not in meta and st.last_remesh_step > st.step:
            # an older driver backed off by moving the last rebuild into the future, which also
            # blocked coarsening: keep only the wait of the refinement
            st.refine_blocked_until = st.last_remesh_step + self.s.min_steps_between_remesh
            st.last_remesh_step -= self.s.remesh_backoff
        st.dt_prev = meta.get("dt_prev")
        st.last_state_t = meta["time"]
        self.pb.remesh_count = meta["remesh_count"]
        if meta.get("focus"):
            self.pb.active_mesh_template().focus = Focus(**meta["focus"])
        if "episodes" in meta:
            self.episodes = EpisodeTracker.from_dict(meta["episodes"])
            self.episodes.hys = self.s.eta_hys
        else:  # the episode in progress began at a time this dump does not record
            self.episodes = EpisodeTracker(hys=self.s.eta_hys, history_known=False)
        ep = self.episodes
        self._event("restart", self.measure(), from_time=meta["time"],
                    episode_history="episodes" in meta, episode=ep.index, episode_kind=ep.kind,
                    episode_start=ep.start, episode_partial=ep.partial, h_crest=ep.h_crest,
                    t_crest=ep.t_crest, crest_jet=ep.crest_jet, max_curvature_reset=reset,
                    coarsen_held_until=st.coarsen_held_until)

    # --- main loop ----------------------------------------------------------------------------

    def run(self, newton_tolerance: float) -> dict:
        s, st, pb = self.s, self.state, self.pb
        m = self.measure()
        self._track(m)
        self._write_row(m, 0.0, 0.0, self._singular(m))
        self._snapshot(m, "start")
        stop = {"reason": "end-time"}
        while True:
            t = self._t()
            if s.t_end - t <= max(1e-14, 8 * math.ulp(s.t_end)):
                break
            if st.step >= s.max_steps:
                stop = {"reason": "max-steps"}
                break
            dt = min(s.dt_max, s.courant * m["h_focus"] / m["U"],
                     s.capillary * math.sqrt(m["ell"] ** 3), s.t_end - t)
            if math.isfinite(st.ell_prev) and st.t_prev < t:
                rate = (st.ell_prev - m["ell"]) / (t - st.t_prev)  # only a shrinking scale limits
                if rate > 0:
                    dt = min(dt, s.scale_rate * m["ell"] / rate)
            if st.dt_prev is not None:
                dt = min(dt, s.growth * st.dt_prev)
            if dt < s.dt_min:
                stop = {"reason": "dt-floor", "dt": dt}
                break
            st.ell_prev, st.t_prev = m["ell"], t
            wall0 = time.time()
            # temporal_error engages pyoomph's adaptive stepping: without an error estimator every
            # converged step is accepted, while a failed Newton solve is rejected and halved
            tol = max(newton_tolerance, s.newton_floor / dt)
            try:
                pb.run(t + dt, timestep=dt, maxstep=dt, outstep=False, temporal_error=1.0,
                       newton_solver_tolerance=tol)
            except RuntimeError as exc:
                # the problem now holds a rejected iterate: stop on the last measured state ``m``,
                # but only if oomph-lib gave up below its smallest step
                stop = None
                if type(exc) is RuntimeError and exc.args == ("OomphException",):
                    error = oomph_error_message(pb.oomph_log_tail())
                    if MIN_DT_FAILURE in error:
                        stop = classify_failure(m, s, error)
                if stop is None:
                    raise
                stop["dt_attempted"] = dt
                break
            st.step += 1
            dt = self._t() - t
            st.dt_prev = dt
            m = self.measure()
            self._track(m)
            self._write_row(m, dt, time.time() - wall0, self._singular(m))

            if st.born and m["axis_lowest"]:  # the jet has gone; a new cavity on the axis
                st.born = False
                st.cycle += 1
                st.max_curvature = 0.0
                st.max_curvature_time = math.nan
                self._reset_singular()
                m = self.measure()
                self._event("cavity", m, cycle=st.cycle)
                self._maybe_snapshot(m, "cavity")
            if m["cavity"] and not st.cavity_seen:
                st.cavity_seen = True
                self._event("cavity", m, cycle=st.cycle)
                self._maybe_snapshot(m, "cavity")
            if m["jet_born"] and not st.born:
                st.born = True
                self._event("jet-born", m, cycle=st.cycle, max_curvature=st.max_curvature,
                            max_curvature_time=st.max_curvature_time, t0_apex=st.apex_t0,
                            alpha_apex=st.apex_alpha, t_apex_est=st.apex_t_est)
                self._maybe_snapshot(m, "jet-born")
            # a trough apex only: a bump or crest on the axis has z''(0) < 0
            if st.cavity_seen and not st.born and m["axis_curvature"] > st.max_curvature:
                st.max_curvature = m["axis_curvature"]
                st.max_curvature_time = self._t()
            if math.isfinite(m["r_neck"]) and not st.neck_seen:
                st.neck_seen = True
                self._event("neck", m, r_neck=m["r_neck"], z_neck=m["z_neck"])
            neck_limit = max(s.neck_stop, s.neck_stop_elements * s.h_floor)
            if math.isfinite(m["r_neck"]) and m["r_neck"] < neck_limit:
                stop = {"reason": "pinch-off-approach", "trigger": "neck-radius",
                        "r_neck": m["r_neck"], "z_neck": m["z_neck"], "neck_limit": neck_limit}
                break
            self._maybe_snapshot(m)
            if self._maybe_remesh(m):
                m = self.measure()
            if self._t() - st.last_state_t >= s.state_dt:
                self.save_state()
        self._event("stop", m, **stop)
        self._snapshot(m, "stop")
        return {"stop": stop, "steps": st.step, "remesh_count": pb.remesh_count,
                "jet_born": st.born, "max_axis_curvature_before_birth": st.max_curvature,
                "max_axis_curvature_time": st.max_curvature_time, "snapshots": st.snapshots}
