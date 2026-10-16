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
when a neck becomes narrower than ``max(neck_stop, neck_stop_elements h_floor)``, at the
conservative self-contact cutoff, when a fold in the focus region (two nodes closer than
``1/fold_ratio`` of the arc length between them, :func:`selfcontact.fold_gap`; the air slot that
can close at the jet base) is narrower than ``max(fold_stop, fold_stop_elements h_elem)``, with
``h_elem`` the local element length there (``self-contact-approach``), or at ``t_end``. The pair
with smallest gap relative to its own limit governs this conservative cutoff; a static sharp
corner can also meet it, so the event alone does not establish physical entrapment. A run that
only has to decide whether a jet is born, such as one of a bisection between jet and pocket, can
also stop once the jet is unambiguous (``jet-confirmed``): at the first step, at least
``post_birth_steps`` accepted steps after birth, at which the jet stands more than
``post_birth_aspect r_j`` above its base. The default ``post_birth_aspect`` of ``inf`` never
stops on the jet. It is checked wherever the geometry stops are, and after them. A surface
that has already crossed itself is no end state: measuring it raises
:class:`~selfcontact.InterfaceSelfIntersection`, and so does a remesh handed one (``mesh.py``).

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
narrowest neck, cone angles, the resolution, the global budgets, the power-law origin estimate and
the fold closest to its local limit (``fold_gap``, ``inf`` without one within ``FOLD_SEARCH`` stop limits, at
``fold_r``, ``fold_z``).
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

Kept dumps lie outside the rotation of ``state_keep``: with ``keep_stop_state`` the state at a
neck-radius stop (``stop_<step>.dump``, the landing state of a reconnection); with ``landing``, a
list of cut radii, for each radius ``delta`` the first measured state at which the neck has closed
to ``delta/LANDING_RATIO`` (``landing_<delta>.dump``, event ``landing``), so that one run gives the
landing states of a study in ``delta``; and the ``precut`` and ``postcut`` states of a cut
(``surgery.py``). Every measured state is offered for landing, the stop state and the state after a
rebuild included. A radius the run cannot land before its neck-radius stop,
``delta < LANDING_RATIO max(neck_stop, neck_stop_elements h_floor)``, is refused when the driver is
made (:func:`check_landing`). Event ``landing-missed`` records a radius left without a landing state:
reason ``neck-passed`` for a state already narrower than ``delta/MAX_LANDING_RATIO`` (as after a
restart; a cut there is refused), ``stopped-first`` at the stop for each radius the run ended before
(these stay out of ``landed``, so that a continuation can still land them). The run's result then
lists the radii kept and missed (``landing``).

Landing and stop dumps are kept within their step: after its stop checks, before its snapshot and
remesh checks. Their JSON records ``resume_at`` ``remesh-check``, and a restart from one makes that
remesh check before its first step, as the run that wrote it did after keeping it, so that both step
from the same mesh (the restart's ``start`` profile stands for the snapshot). Rotating dumps are
written at the end of a step, and a restart from one steps at once. Kept dumps, and every dump of a
run that keeps landing states, also hold the latest ``NECK_SAMPLES`` samples ``(t, r_n)`` of the
closing neck, from which :meth:`CollapseDriver.pinch_time` fits the pinch time ``t_p`` (a power law
``r_n = A (t_p - t)^alpha`` with a free exponent, as for the singular time). The fit, with its
window and exponent, is recorded in the landing and stop dumps (``pinch``) and in the ``landing``
event, and the cut records it again in its ``precut`` dump and its record.

A driver made with ``reconnect`` settings continues a run after a cut (``surgery.cut``, made with
this driver from a landing state, or restored from a state written after one): the post-cut mode.
After the cut (:meth:`CollapseDriver.reset_after_cut`) the measured surface is the open chain, whose
tip on the axis is a fresh cavity bottom: the scale, step, neck and singular-time bookkeeping
restart, a new cycle begins, and the first step is the cut's ``dt0``. The jet measurement, the
neck and fold stops and the step rule above apply to the open chain as to any surface. In addition:

* the stops before a second change of topology, on both chains (``reconnect.post_cut_stops``), and
  ``tau-end`` once ``tau = t - t_cut`` reaches the cut's ``tau_end``;
* the step is also at most ``courant h/|u|`` over every node of both chains
  (``reconnect.courant_limit``: the tips and jets of either chain wherever they go), at most
  ``capillary R^(3/2)`` with ``R`` the smallest radius of curvature of the three tips on the axis,
  and grows by at most the cut's ``dt_growth``;
* no Newton failure ends the run cleanly: after a cut it is a failed run, whatever the state;
* ``collapse.txt`` gets the columns ``tau`` and ``tau_p = t - t_p`` (``t_p`` the pinch time the cut
  recorded, NaN without one), ``chains.txt`` a row per step with both chains
  (``reconnect.post_cut_record``: the three tips, bridge, bubble, necks, budgets) and the step
  limits of the state (``dt_courant``, ``dt_capillary``), the profiles a ``_bubble`` file beside
  each one of the open chain, and every event, profile header and dump after the cut ``tau`` and
  ``tau_p``;
* landing states and the stop state are not kept (one cut per run).

Dumps written after a cut carry its record, which :meth:`CollapseDriver.restore` restores
(``surgery.restore_cut``), only into a driver made with the same reconnect settings; the run then
continues in the post-cut mode.
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
from .reconnect import (LANDING_RATIO, MAX_LANDING_RATIO, Chain, CutGeometry, ReconnectSettings,
                        courant_limit, post_cut_record, post_cut_stops)
from .selfcontact import InterfaceSelfIntersection, fold_gap
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
    fold_ratio: float = 4.0  # a fold pair is closer than 1/fold_ratio of the arc between them
    fold_stop: float = 1e-3  # as neck_stop, for the gap of a closing fold
    fold_stop_elements: float = 3.0  # in local element lengths at the fold
    neck_guard: float = 1e-2  # a Newton failure with a neck narrower than this is a pinch-off
    ell_guard: float = 1e-3  # and with a local scale below this the approach to the singularity
    post_birth_aspect: float = math.inf  # stop once the born jet stands this many r_j above its base
    post_birth_steps: int = 1  # accepted steps after birth before post_birth_aspect can stop a run
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

    def __post_init__(self):
        for name in ("fold_ratio", "fold_stop", "fold_stop_elements"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if self.fold_ratio <= 1:
            raise ValueError("fold_ratio must be greater than one")
        if self.fold_stop < 0 or self.fold_stop_elements < 0:
            raise ValueError("fold stop limits must be non-negative")
        if self.fold_stop == 0 and self.fold_stop_elements == 0:
            raise ValueError("at least one fold stop limit must be positive")
        aspect = self.post_birth_aspect
        if isinstance(aspect, bool) or not isinstance(aspect, (int, float)) or not aspect > 0:
            raise ValueError("post_birth_aspect must be positive (inf: no post-birth stop)")
        steps = self.post_birth_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("post_birth_steps must be an integer of at least one")

    @classmethod
    def from_dict(cls, data: dict, *, t_end: float, dt_max: float) -> "CollapseSettings":
        kw = dict(data)
        for key in ("cone_radii", "cone_window", "singular_alpha_range"):
            if key in kw:
                kw[key] = tuple(kw[key])
        return cls(t_end=t_end, dt_max=dt_max, **kw)

    @property
    def neck_limit(self) -> float:
        """The neck radius below which a run stops (``pinch-off-approach``, ``neck-radius``)."""
        return max(self.neck_stop, self.neck_stop_elements * self.h_floor)


MIN_DT_FAILURE = "less than the minimum dt"  # oomph-lib's message when it gives up on a step
FOLD_SEARCH = 4.0  # folds are looked for within this many stop limits
# what a dump's max_curvature holds: the largest z''(0) > 0 of the trough episode and cycle
MAX_CURVATURE_KIND = "trough-episode"
# a dump kept within a step, after its stop checks and before its snapshot and remesh checks
RESUME_REMESH_CHECK = "remesh-check"
NECK_SAMPLES = 100  # (t, r_n) samples of the closing neck the driver keeps for the pinch time
PINCH_FIT_SPAN = 1.5  # the pinch time is fitted to the latest samples with r_n <= this * the last


def check_landing(landing, settings: CollapseSettings) -> None:
    """Refuse cut radii of ``landing`` that the run cannot land before its neck-radius stop.

    A landing state for ``delta`` is the first measured state with ``r_n <= delta/LANDING_RATIO``,
    and the run stops at the first with ``r_n < settings.neck_limit``. Since the stop state is
    itself offered for landing, every ``delta >= LANDING_RATIO * neck_limit`` is reached (unless the
    run ends otherwise first, which ``landing-missed`` records); a smaller one is not."""
    limit = settings.neck_limit
    short = [float(d) for d in landing if float(d) < LANDING_RATIO * limit]
    if short:
        raise ValueError(
            "collapse.landing: the run stops at the neck radius "
            f"max(neck_stop, neck_stop_elements * h_floor) = {limit:.6g} before the neck reaches "
            f"delta/{LANDING_RATIO} for the cut radii {', '.join(f'{d:.6g}' for d in short)}; "
            f"lower collapse.neck_stop below {min(short) / LANDING_RATIO:.6g} or drop those radii")


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
           "t0_est", "alpha_est", "t0_source", "fold_gap", "fold_r", "fold_z")
POST_CUT_COLUMNS = ("tau", "tau_p")  # appended to collapse.txt after a cut
CHAIN_COLUMNS = ("step", "time", "tau", "tau_p", "dt", "z_bubble_bottom", "w_bubble_bottom",
                 "R_bubble_bottom", "z_bubble_top", "w_bubble_top", "R_bubble_top", "z_cavity_tip",
                 "w_cavity_tip", "R_cavity_tip", "bridge", "bubble_volume", "bubble_r_max",
                 "bubble_z_r_max", "r_neck_bubble", "z_neck_bubble", "r_neck_open", "z_neck_open",
                 "R_tip_min", "volume", "kinetic_energy", "momentum_z", "divergence_sq",
                 "h_courant", "U_courant", "courant_chain", "dt_courant", "dt_capillary",
                 "remesh_count")


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
    born_step: int = -1  # step at which the standing jet was born
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
    landed: list = field(default_factory=list)  # cut radii of ``landing`` already kept or passed
    neck_samples: list = field(default_factory=list)  # [t, r_n] of the closing neck, the latest


class CollapseDriver:
    def __init__(self, problem, settings: CollapseSettings, outdir: Path, *,
                 keep_stop_state: bool = False, landing=(),
                 reconnect: ReconnectSettings | None = None):
        self.pb = problem
        self.s = settings
        self.keep_stop_state = keep_stop_state
        self.landing = tuple(sorted((float(x) for x in landing), reverse=True))
        check_landing(self.landing, settings)
        # the post-cut mode: the run continues after a cut made with these settings
        self.reconnect = reconnect
        if reconnect is not None and (self.landing or keep_stop_state):
            raise ValueError("a run after a cut keeps no landing or stop states (one cut per run)")
        self._post_cut_cache: tuple | None = None
        # the measured state has had its stop checks but not yet its snapshot and remesh checks
        self._checks_pending = False
        self.outdir = Path(outdir)
        (self.outdir / "profiles").mkdir(parents=True, exist_ok=True)
        self.state = _State()
        self.episodes = EpisodeTracker(hys=settings.eta_hys)
        self._table = open(self.outdir / "collapse.txt", "w", buffering=1)
        cols = list(COLUMNS) + [f"beta_r{r:g}" for r in settings.cone_radii]
        if reconnect is not None:
            cols += list(POST_CUT_COLUMNS)
        self._table.write("# " + "\t".join(cols) + "\n")
        self._chains = None
        if reconnect is not None:
            self._chains = open(self.outdir / "chains.txt", "w", buffering=1)
            self._chains.write("# " + "\t".join(CHAIN_COLUMNS) + "\n")
        self._events = open(self.outdir / "events.jsonl", "w", buffering=1)
        (self.outdir / "collapse_settings.json").write_text(
            json.dumps(asdict(settings), indent=2) + "\n")

    def close(self):
        self._table.close()
        self._events.close()
        if self._chains is not None:
            self._chains.close()

    # --- after a cut --------------------------------------------------------------------------

    def _cut_record(self) -> dict | None:
        """The record of the cut this run continues after, in the post-cut mode only."""
        return None if self.reconnect is None else getattr(self.pb, "cut", None)

    def _cut_times(self) -> tuple[float, float]:
        """``(t_cut, t_p)``: the time of the cut and the pinch time it recorded (NaN without)."""
        cut = self._cut_record()
        if cut is None:
            return math.nan, math.nan
        t_p = (cut.get("pinch") or {}).get("t_p")
        return float(cut["t_cut"]), math.nan if t_p is None else float(t_p)

    def _taus(self, t: float) -> dict:
        t_cut, t_p = self._cut_times()
        return {"tau": t - t_cut, "tau_p": t - t_p}

    def _post_cut_state(self) -> dict:
        """Both chains of the current state, their record (``reconnect.post_cut_record``) and the
        step limits they set, computed once per state."""
        key = (self._t(), self.pb.remesh_count, self.state.step)
        if self._post_cut_cache is not None and self._post_cut_cache[0] == key:
            return self._post_cut_cache[1]
        segs = self.pb.interface_segments()
        rec = post_cut_record(segs) if len(segs) == 2 else {}
        cl = courant_limit(segs)
        R = rec.get("R_tip_min", math.nan)
        state = {"segs": segs, "record": rec, "courant": cl,
                 "dt_courant": self.reconnect.courant * cl["h_over_u"],
                 "dt_capillary": self.s.capillary * R**1.5 if math.isfinite(R) else math.inf}
        self._post_cut_cache = (key, state)
        return state

    def _post_cut_stop(self) -> dict | None:
        """The stops before a second change of topology, on both chains (post-cut mode)."""
        cut = self._cut_record()
        if cut is None:
            return None
        segs = self._post_cut_state()["segs"]
        chains = [Chain(np.column_stack([c["r"], c["z"]]), tuple(c["end_types"])) for c in segs]
        s = self.s
        stop = post_cut_stops(chains, CutGeometry(**cut["geometry"]), self.reconnect,
                              cut["sizing"]["h_tip"], neck_elements=s.neck_stop_elements,
                              fold_ratio=s.fold_ratio, fold_stop=s.fold_stop,
                              fold_stop_elements=s.fold_stop_elements, fold_search=FOLD_SEARCH)
        if stop is not None:
            stop.update(self._taus(self._t()))
        return stop

    def _write_chain_row(self, dt: float):
        pc = self._post_cut_state()
        obs = self.pb.cut_observables()
        cl = pc["courant"]
        t = self._t()
        values = {"step": self.state.step, "time": t, **self._taus(t), "dt": dt, **pc["record"],
                  **{k: obs[k] for k in ("volume", "kinetic_energy", "momentum_z",
                                         "divergence_sq")},
                  "h_courant": cl["h"], "U_courant": cl["U"], "courant_chain": cl["chain"],
                  "dt_courant": pc["dt_courant"], "dt_capillary": pc["dt_capillary"],
                  "remesh_count": self.pb.remesh_count}
        self._chains.write("\t".join(
            str(values[c]) if c in ("step", "courant_chain", "remesh_count")
            else f"{float(values.get(c, math.nan)):.12e}" for c in CHAIN_COLUMNS) + "\n")

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
        # Select the fold closest to its own stop limit, using each node's local spacing.
        fold = fold_gap(r, z, r_max=focus_r, ratio=s.fold_ratio,
                        stop=s.fold_stop, stop_elements=s.fold_stop_elements, search_factor=FOLD_SEARCH)
        if fold.crossed:
            raise InterfaceSelfIntersection(
                f"the free surface crosses itself within r <= {focus_r:.6g} at t = {self._t():.15g}; "
                "the measured state is invalid and the last row before it is the last valid one")
        fold_limit = (max(s.fold_stop, s.fold_stop_elements * fold.h_elem)
                      if math.isfinite(fold.gap) else math.nan)
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
                "angles": angles, "focus_radius": focus_r, "profile": (r, z, ur, uz),
                "fold_gap": fold.gap, "fold_r": fold.r, "fold_z": fold.z, "fold_arc": fold.arc,
                "fold_h_elem": fold.h_elem, "fold_limit": fold_limit}

    # --- bookkeeping --------------------------------------------------------------------------

    def _event(self, kind: str, m: dict, **extra):
        rec = {"event": kind, "time": m["time"], "step": self.state.step,
               "r_j": m["r_j"], "z_axis": m["z_axis"], "ell": m["ell"],
               "axis_curvature": m["axis_curvature"],
               **(self._taus(m["time"]) if self._cut_record() is not None else {}), **extra}
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
               obs["dissipation"], self.pb.remesh_count, wall, *est,
               m.get("fold_gap", math.inf), m.get("fold_r", math.nan),
               m.get("fold_z", math.nan)] + list(m["angles"])
        if self.reconnect is not None:
            row += [self._taus(m["time"])[k] for k in POST_CUT_COLUMNS]
        self._table.write("\t".join(f"{v:.12e}" if isinstance(v, float) else str(v)
                                    for v in row) + "\n")
        if self._cut_record() is not None:
            self._write_chain_row(dt)

    def _snapshot(self, m: dict, reason: str):
        st = self.state
        r, z, ur, uz = m["profile"]
        header = {"time": m["time"], "step": st.step, "reason": reason, "r_j": m["r_j"],
                  "z_b": m["z_b"], "z_axis": m["z_axis"], "ell": m["ell"],
                  "remesh_count": self.pb.remesh_count}
        path = self.outdir / "profiles" / f"profile_{st.snapshots:06d}.txt"
        bubble = None
        if self._cut_record() is not None:  # the open chain here, the bubble beside it
            header.update(self._taus(m["time"]), chain="open")
            bubble = next(c for c in self._post_cut_state()["segs"] if not c["open"])
        np.savetxt(path, np.column_stack([r, z, ur, uz]), header=json.dumps(header) +
                   "\nr z u_r u_z", comments="# ")
        if bubble is not None:
            np.savetxt(path.with_name(path.stem + "_bubble.txt"),
                       np.column_stack([bubble[k] for k in ("r", "z", "u_r", "u_z")]),
                       header=json.dumps(dict(header, chain="bubble")) + "\nr z u_r u_z",
                       comments="# ")
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
                        "refine_blocked_until", "coarsen_held_until", "born_step")

    def _write_state(self, path: Path, *, kept: bool = False, **extra):
        """The pyoomph state at ``path`` and the driver's bookkeeping in the JSON beside it.

        The keys added for reconnections appear only where they apply, so that the dumps of other
        runs keep their keys: ``landed`` in runs with ``landing``; ``neck_samples`` in kept dumps
        and in runs that keep landing states; ``resume_at`` in dumps kept within a step; ``cut``
        after a cut."""
        st = self.state
        path.parent.mkdir(exist_ok=True)
        self.pb.save_state(str(path), quiet=True)
        focus = self.pb.active_mesh_template().focus
        meta = {k: getattr(st, k) for k in self._STATE_FIELDS + self._OPTIONAL_FIELDS}
        meta.update(time=self._t(), remesh_count=self.pb.remesh_count,
                    focus=None if focus is None else asdict(focus), dt_prev=st.dt_prev,
                    episodes=self.episodes.to_dict(), max_curvature_kind=MAX_CURVATURE_KIND)
        if self.landing:
            meta["landed"] = list(st.landed)
        if kept or self.landing or self.keep_stop_state:
            meta["neck_samples"] = [list(x) for x in st.neck_samples]
        if self._checks_pending:
            meta["resume_at"] = RESUME_REMESH_CHECK
        cut = getattr(self.pb, "cut", None)
        if cut is not None:
            meta["cut"] = cut
            if self.reconnect is not None:
                meta.update(self._taus(meta["time"]))
        meta.update(extra)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")

    def keep_state(self, path: Path, **extra) -> Path:
        """A dump outside the rotation of ``state_keep`` (``path`` under ``states/``)."""
        self._write_state(Path(path), kept=True, **extra)
        return Path(path)

    def save_state(self):
        """Dump the pyoomph state and the driver state; keep the last ``state_keep`` dumps."""
        st = self.state
        path = self.outdir / "states" / f"state_{st.step:08d}.dump"
        self._write_state(path)
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
        if st.born and "born_step" not in meta:  # older dump: wait post_birth_steps from here
            st.born_step = st.step
        reset = meta.get("max_curvature_kind") != MAX_CURVATURE_KIND
        if reset:  # possibly |z''(0)| of a bump, or of an earlier trough episode
            st.max_curvature, st.max_curvature_time = 0.0, math.nan
        if "refine_blocked_until" not in meta and st.last_remesh_step > st.step:
            # an older driver backed off by moving the last rebuild into the future, which also
            # blocked coarsening: keep only the wait of the refinement
            st.refine_blocked_until = st.last_remesh_step + self.s.min_steps_between_remesh
            st.last_remesh_step -= self.s.remesh_backoff
        st.dt_prev = meta.get("dt_prev")
        st.landed = [float(x) for x in meta.get("landed", [])]
        st.neck_samples = [[float(t), float(r)] for t, r in meta.get("neck_samples", [])]
        # a dump kept within its step: the run that wrote it went on to the snapshot and remesh
        # checks of that step, and so does the restart before its first step (run())
        self._checks_pending = meta.get("resume_at") == RESUME_REMESH_CHECK
        st.last_state_t = meta["time"]
        self.pb.remesh_count = meta["remesh_count"]
        if meta.get("focus"):
            self.pb.active_mesh_template().focus = Focus(**meta["focus"])
        if "episodes" in meta:
            self.episodes = EpisodeTracker.from_dict(meta["episodes"])
            self.episodes.hys = self.s.eta_hys
        else:  # the episode in progress began at a time this dump does not record
            self.episodes = EpisodeTracker(hys=self.s.eta_hys, history_known=False)
        if meta.get("cut") is not None:
            if self.reconnect is None:
                raise RuntimeError("a state written after a cut is continued only by a driver made "
                                   "with the cut's reconnect settings (post-cut stops on both chains)")
            if json.loads(json.dumps(self.reconnect.to_dict())) != meta["cut"].get("settings"):
                raise ValueError("the reconnect settings differ from those of the cut that wrote "
                                 "the state")
            from .surgery import restore_cut

            restore_cut(self.pb, meta["cut"])
        ep = self.episodes
        resume = {"resume_at": RESUME_REMESH_CHECK} if self._checks_pending else {}
        self._event("restart", self.measure(), from_time=meta["time"],
                    episode_history="episodes" in meta, episode=ep.index, episode_kind=ep.kind,
                    episode_start=ep.start, episode_partial=ep.partial, h_crest=ep.h_crest,
                    t_crest=ep.t_crest, crest_jet=ep.crest_jet, max_curvature_reset=reset,
                    coarsen_held_until=st.coarsen_held_until, **resume)

    def reset_after_cut(self, record: dict):
        """Restart the bookkeeping for the surface after a cut (``surgery.cut``).

        The open chain now ends on the axis in a fresh cap: a cavity bottom of a new cycle. The
        limiters of the local scale and the step, the singular-time series, the largest trough
        curvature, the jet and the neck are reset, so that the cap's curvature is not read as a
        collapse and the cycle is not counted twice; the next step is the cut's ``dt0``."""
        if self.reconnect is None:
            raise RuntimeError("a driver that continues after a cut needs the reconnect settings")
        st = self.state
        dt0 = float(record["settings"]["dt0"])
        st.ell_prev = st.t_prev = math.nan
        st.dt_prev = dt0 / min(self.s.growth, self.reconnect.dt_growth)  # the first step is dt0
        self._reset_singular()
        st.last_singular_t = -math.inf
        st.max_curvature, st.max_curvature_time = 0.0, math.nan
        st.born, st.born_step = False, -1
        st.neck_seen = False
        st.neck_samples = []
        st.cavity_seen = True
        st.cycle += 1
        st.coarsen_held_until = -math.inf
        st.refine_blocked_until = -10**9
        self._checks_pending = False  # the cut rebuilt the mesh; the next step follows at once
        m = self.measure()
        geometry = record.get("geometry", {})
        self._event("cut", m, cycle=st.cycle, delta=record["delta"], rmin=record.get("rmin"),
                    r_neck=record["r_neck"], z_neck=record["z_neck"],
                    t_pinch=record.get("pinch", {}).get("t_p"),
                    bridge=geometry.get("bridge"), removed_gas=geometry.get("removed_gas"),
                    in_cut_radii=record.get("in_cut_radii"),
                    delta_KE=record.get("projection", {}).get("delta_KE"),
                    transfer=record.get("transfer"), focus=record.get("focus"))
        self._snapshot(m, "cut")
        return m

    def _neck_sample(self, m: dict):
        """Offer the measured neck to the series from which :meth:`pinch_time` fits; no neck
        clears it, and a restart does not offer the dumped state again."""
        samples, r_n, t = self.state.neck_samples, m["r_neck"], m["time"]
        if not (math.isfinite(r_n) and r_n > 0.0):
            samples.clear()
        elif not samples or t > samples[-1][0]:
            samples.append([t, r_n])
            del samples[:-NECK_SAMPLES]

    def pinch_time(self) -> dict:
        """The pinch time ``t_p`` of the neck, from a power law ``r_n = A (t_p - t)^alpha`` fitted
        (:func:`singular.singular_time`, with the run's singular-time settings) to the latest
        neck samples over which the neck closes, back to ``PINCH_FIT_SPAN`` times the latest radius.

        The record gives the fit window (times, radii, number of samples): a fit over another
        window of the same run gives another ``t_p``, since the exponent of a real neck drifts as it
        closes, so a study in the cut radius compares its cuts at one common ``t_p``. NaN (``None``
        in JSON) when the series is too short or the fit is ill-conditioned."""
        s, samples = self.s, self.state.neck_samples
        out = {"t_p": math.nan, "alpha": math.nan, "samples": 0, "t_window": None,
               "r_window": None, "span": PINCH_FIT_SPAN, "source": "neck_samples"}
        if not samples:
            return out
        i, r_last = len(samples) - 1, samples[-1][1]
        while i > 0 and samples[i][1] < samples[i - 1][1] <= PINCH_FIT_SPAN * r_last:
            i -= 1
        ts, xs = np.asarray(samples[i:], dtype=float).T
        t0, alpha = singular_time(ts, xs, min_samples=s.singular_samples,
                                  alpha_range=s.singular_alpha_range,
                                  max_residual=s.singular_residual,
                                  max_extrapolation=s.singular_extrapolation)
        out.update(t_p=t0, alpha=alpha, samples=len(ts), t_window=[float(ts[0]), float(ts[-1])],
                   r_window=[float(xs[0]), float(xs[-1])])
        return out

    def _keep_landings(self, m: dict):
        """Keep the landing state of each cut radius of ``landing`` that the neck has just passed:
        the first measured state with ``delta/r_n >= LANDING_RATIO``, unless the neck is already
        narrower than ``delta/MAX_LANDING_RATIO`` (a cut there is refused: ``landing-missed``,
        reason ``neck-passed``)."""
        r_n = m["r_neck"]
        if not self.landing or not (math.isfinite(r_n) and r_n > 0.0):
            return
        st = self.state
        for delta in self.landing:
            if delta in st.landed or delta < LANDING_RATIO * r_n:
                continue
            st.landed.append(delta)  # before the dump, so that a restart from it skips this radius
            ratio = delta / r_n
            info = {"delta": delta, "r_neck": r_n, "z_neck": m["z_neck"], "delta_over_r_neck": ratio}
            if ratio > MAX_LANDING_RATIO:
                self._event("landing-missed", m, **info, reason="neck-passed",
                            max_landing_ratio=MAX_LANDING_RATIO)
                continue
            pinch = self.pinch_time()
            path = self.keep_state(self.outdir / "states" / f"landing_{delta:.6g}.dump",
                                   landing=info, pinch=pinch)
            self._event("landing", m, **info, path=str(path), t_p=pinch["t_p"],
                        alpha_p=pinch["alpha"])

    def _landings_left(self, m: dict, stop: dict):
        """At the stop: ``landing-missed`` (reason ``stopped-first``) for each cut radius the run
        ended before. They stay out of ``landed``, so that a continuation can still land them."""
        st, r_n = self.state, m["r_neck"]
        for delta in self.landing:
            if delta not in st.landed:
                self._event("landing-missed", m, delta=delta, r_neck=r_n, z_neck=m["z_neck"],
                            delta_over_r_neck=delta / r_n if r_n > 0.0 else math.nan,
                            reason="stopped-first", stop_reason=stop.get("reason"),
                            landing_ratio=LANDING_RATIO)

    def _landing_summary(self) -> dict:
        """The cut radii this run kept and those it missed, with the reason."""
        events = self.state.events
        return {"radii": list(self.landing),
                "kept": [e["delta"] for e in events if e["event"] == "landing"],
                "missed": [{"delta": e["delta"], "reason": e["reason"]} for e in events
                           if e["event"] == "landing-missed"]}

    # --- main loop ----------------------------------------------------------------------------

    def _geometry_stop(self, m):
        neck_limit = self.s.neck_limit
        if math.isfinite(m["r_neck"]) and m["r_neck"] < neck_limit:
            return {"reason": "pinch-off-approach", "trigger": "neck-radius",
                    "r_neck": m["r_neck"], "z_neck": m["z_neck"], "neck_limit": neck_limit}
        if m.get("fold_gap", math.inf) < m.get("fold_limit", math.nan):
            return {"reason": "self-contact-approach", "trigger": "fold-gap",
                    "fold_gap": m["fold_gap"], "fold_r": m["fold_r"], "fold_z": m["fold_z"],
                    "fold_arc": m["fold_arc"], "fold_h_elem": m["fold_h_elem"],
                    "fold_limit": m["fold_limit"]}
        return None

    def _stop(self, m):
        """The stop the measured state calls for, if any: a geometry stop (after a cut, those of
        both chains too) before the jet stop."""
        return self._geometry_stop(m) or self._post_cut_stop() or self._jet_stop(m)

    def _jet_stop(self, m):
        """Stop once the born jet is unambiguous: at least ``post_birth_steps`` accepted steps after
        its birth, it stands more than ``post_birth_aspect r_j`` above its base."""
        st, s = self.state, self.s
        if not st.born or st.step - st.born_step < s.post_birth_steps:
            return None
        rise = m["z_axis"] - m["z_b"]
        if m["r_j"] > 0.0 and rise > s.post_birth_aspect * m["r_j"]:
            return {"reason": "jet-confirmed", "trigger": "jet-aspect", "rise": rise,
                    "r_j": m["r_j"], "aspect": rise / m["r_j"], "born_step": st.born_step,
                    "post_birth_aspect": s.post_birth_aspect}
        return None

    def _remesh_checks(self, m: dict) -> tuple[dict, dict | None]:
        """The remesh check of a measured state that has passed its stop checks; after a rebuild,
        the measurement of the new mesh, offered for landing and checked for the stops again.
        Returns the state the next step starts from and the stop it calls for, if any."""
        self._checks_pending = False
        if self._maybe_remesh(m):
            m = self.measure()
            self._keep_landings(m)
            return m, self._stop(m)
        return m, None

    def run(self, newton_tolerance: float) -> dict:
        s, st, pb = self.s, self.state, self.pb
        r = self.reconnect
        if r is not None and self._cut_record() is None:
            raise RuntimeError("a driver made with reconnect settings runs after a cut: make the "
                               "cut (surgery.cut with this driver) or restore a post-cut state")
        t_cut = self._cut_times()[0]
        m = self.measure()
        self._track(m)
        self._write_row(m, 0.0, 0.0, self._singular(m))
        self._neck_sample(m)
        self._snapshot(m, "start")  # also stands for the snapshot check of a dump kept in its step
        self._keep_landings(m)
        stop = {"reason": "end-time"}
        if self._checks_pending and self._stop(m) is None:
            # restarted from a dump kept within its step (a landing or a stop state): make the
            # remesh check its run made after keeping it, so that the restart steps from the mesh
            # that run stepped from; the loop checks the stops of the new mesh
            m, _ = self._remesh_checks(m)
        while True:
            found = self._stop(m)
            if found is not None:
                stop = found
                break
            t = self._t()
            if s.t_end - t <= max(1e-14, 8 * math.ulp(s.t_end)):
                break
            if r is not None and t - t_cut >= r.tau_end - max(1e-14, 8 * math.ulp(r.tau_end)):
                stop = {"reason": "tau-end", "tau_end": r.tau_end, **self._taus(t)}
                break
            if st.step >= s.max_steps:
                stop = {"reason": "max-steps"}
                break
            dt = min(s.dt_max, s.courant * m["h_focus"] / m["U"],
                     s.capillary * math.sqrt(m["ell"] ** 3), s.t_end - t)
            if r is not None:  # both chains: every node's Courant limit and the tips' capillary time
                pc = self._post_cut_state()
                dt = min(dt, pc["dt_courant"], pc["dt_capillary"], t_cut + r.tau_end - t)
                if st.dt_prev is not None:
                    dt = min(dt, r.dt_growth * st.dt_prev)
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
                # after a cut no Newton failure is an expected end
                if r is None and type(exc) is RuntimeError and exc.args == ("OomphException",):
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
            self._checks_pending = True  # until the remesh check of this step
            self._track(m)
            self._write_row(m, dt, time.time() - wall0, self._singular(m))
            self._neck_sample(m)

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
                st.born_step = st.step
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
            # with this step's bookkeeping, as every dump; kept before the snapshot and remesh
            # checks of the step, which a restart from it makes first (resume_at)
            self._keep_landings(m)
            found = self._stop(m)
            if found is not None:
                stop = found
                break
            self._maybe_snapshot(m)
            m, found = self._remesh_checks(m)
            if found is not None:
                stop = found
                break
            if self._t() - st.last_state_t >= s.state_dt:
                self.save_state()
        if self.keep_stop_state and stop.get("trigger") == "neck-radius":
            # the measured state of an accepted step: a landing state for a reconnection
            stop["kept_state"] = str(self.keep_state(
                self.outdir / "states" / f"stop_{st.step:08d}.dump", stop=stop,
                pinch=self.pinch_time()))
        self._landings_left(m, stop)
        self._event("stop", m, **stop)
        self._snapshot(m, "stop")
        out = {"stop": stop, "steps": st.step, "remesh_count": pb.remesh_count,
               "jet_born": st.born, "max_axis_curvature_before_birth": st.max_curvature,
               "max_axis_curvature_time": st.max_curvature_time, "snapshots": st.snapshots}
        if self.landing:
            out["landing"] = self._landing_summary()
        if r is not None:
            t_cut, t_p = self._cut_times()
            out["cut"] = {"t_cut": t_cut, "t_p": t_p, "tau_end": r.tau_end, **self._taus(self._t())}
        return out
