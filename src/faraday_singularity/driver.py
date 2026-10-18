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
reaches ``t + dt`` in several accepted sub-steps, and only its end is measured.

Element distortion (pyoomph's ``RemeshWhen``: an element shrunk below 0.6 or grown beyond 1.75
times its size at the build, or of a fifth of its quality then) also asks for a rebuild. Before the
driver has set a focus, pyoomph makes that rebuild itself, inside the solve of the sub-step that
asked, at the initial sizing, as before. Once the driver has set a focus (at its first refinement,
by a cut, or restored from a dump), pyoomph only flags the distortion
(``do_call_remeshing_when_necessary`` off) and the driver rebuilds after the step, before it
measures it (:meth:`CollapseDriver._distortion_remesh`): it measures the step, refreshes the focus
to ``min(focus size, h_near)``, which does not coarsen the mesh near the collapse, within the
current focus radius (after a cut: every feature's region at its current place, at the size it
has, once the stops have been checked), rebuilds and measures again (event ``remesh``, ``reason``
``distortion``, with the trigger and ``h_near`` before and after). So the rebuild no longer reuses
a focus set at an earlier, smaller scale, and every rebuild in the refined phase is the driver's
and is logged. A
driver step makes at most one such rebuild, however many of its sub-steps were flagged. Neither
kind of distortion rebuild starts or ends a hold, nor counts for ``min_steps_between_remesh``; the
stop checks and the rule above follow on the rebuilt mesh, as after pyoomph's own. The run result
counts both (``distortion_remeshes``, ``pyoomph_remeshes``).

The run stops right before a pinch-off, when a neck becomes narrower than
``max(neck_stop, neck_stop_elements h_floor)``, at the
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
neck stop and the step rule above apply to the open chain as to any surface; the fold stop is the
post-cut one, on both chains (below). In addition:

* the jets of the bubble's two tips are measured as the open chain's (``reconnect.bubble_end`` and
  ``reconnect.tip_jet``; the bubble's top is mirrored, so that the jet the bridge drives down into
  the bubble is measured as one rising from a cavity, its base the highest point of the bubble's
  top near the axis), with the flux through the disc at the base (:meth:`_complete_tip_jet`):
  ``chains.txt`` holds ``r_j``, ``z_b``, ``Q_j``, ``V_j``, ``We_j`` and ``jet_born`` of each of
  the three tips (the cavity's as in ``collapse.txt``; ``Q_j`` and ``V_j`` positive along each jet,
  downwards from the bubble's top), and the events ``tip-jet-born`` and ``tip-jet-gone`` record the
  bubble's;
* the local scale is the smallest of the open chain's scales, the bubble's (``ell_bubble`` in
  ``chains.txt``: the jet-base radius of a resolved bump at either tip, the bubble's narrowest neck
  and its length; not the radii of curvature of its tips, which depend on the mesh at a chain's end
  and bound the step instead) and the radius of curvature at the base of each jet, the apex of the
  gas groove around it (``R_apex``: :meth:`CollapseDriver._apexes`, ``reconnect.jet_base_apex``),
  so that a jet on either chain sets the step; the open chain's speed ``U`` (which then includes
  the bubble's nodes and jets) is taken over the region of the open chain's own scale, which a
  smaller scale of the bubble does not narrow;
* the focus size of the rule is floored at ``h_floor`` (:meth:`CollapseDriver._focus_floor`), and
  the mesh follows each feature of both chains in its own region, sized by its own scale
  (``reconnect.feature_regions``, ``mesh.FocusRegion``): the tips the cut made (the cavity's and
  the bubble's top) at ``clip(ell_tip/elements_per_scale, h_floor, h_cut)``, ``ell_tip`` the
  smaller of the tip's radius of curvature and the radius of the jet on it, ``h_cut`` the cut's
  focus size (``focus_size`` cut radii), the coarsest size they may have, within ``focus_extent
  ell_tip`` of the tip, the rest of the cut's region (``focus_radius`` cut radii and the jet on the
  tip) kept at ``h_cut`` (``<tip>_cut``); the base of each jet (``jet_base_<tip>``) at
  ``clip(R_pred/elements_per_scale, h_floor, h_cut)`` within ``max(apex_extent R_apex, 4 target)``,
  ``R_pred`` the apex radius predicted ``apex_lookahead`` steps ahead from its rate of change
  (:meth:`CollapseDriver._apex_prediction`, never below half the radius read); the bubble's
  bottom, which the landing state held as the cavity's bottom, at the size its jet asks for but
  never coarser than at the landing (the cut record's ``focus.kept``); each neck at its own scale;
  and, where the guard of the interface (next item) asks for it, a ``kink_<chain>`` region at a
  quarter of the bent element's chord (``reconnect.kink_regions``), released once the chain has read
  no such element for ``coarsen_hold`` local times and coarsened back by the bounded factor. The
  rule (:meth:`CollapseDriver._maybe_remesh_features`) rebuilds when some region's longest element
  exceeds ``refine_ratio`` times its target, or, once the hold has passed, when some region's
  shortest element (outside the finer regions within it) lies below ``coarsen_ratio`` times it,
  refining or coarsening each region by at most the bounded factors; a rebuild that changed none of
  the regions that asked waits ``remesh_backoff`` steps (event ``remesh-ineffective``,
  ``direction`` ``coarsen`` for a coarsening). ``chains.txt`` gives, at each tip, the longest
  element of its region and the size the current mesh has there (``h_<tip>``, ``size_<tip>``), the
  apex radius, the gap of the groove's closest fold pair and its opening angle at the base of its
  jet (``R_apex_<tip>``, ``groove_gap_<tip>``, ``groove_angle_<tip>``), and, per chain, the largest
  turn of an element and offset of a midside node (``edge_turn_<chain>``,
  ``edge_midside_<chain>``); the ``remesh`` events every region;
* the interface is read on its quadratic elements after every accepted step and on the input of
  every rebuild (``reconnect.interface_guard``): an element whose tangent turns by more than
  ``edge_turn_refine`` degrees across it, or whose midside node lies more than ``midside_refine``
  chords off mid-chord, asks for a refinement (above); the run stops (``interface-unresolved``,
  trigger ``edge-turn``, ``edge-midside`` or ``feature-below-floor``) beyond
  ``edge_turn_stop_fresh`` on a freshly built mesh, ``edge_turn_stop`` on any, ``midside_stop``
  chords, or where the element
  that asks is already at ``h_floor``. A rebuilt mesh is accepted only if the Bezier lower bound of
  the scaled Jacobian of every curved element is at least ``jacobian_min``
  (``FaradayProblem.curved_jacobian_bound``; trigger ``curved-jacobian``), gmsh's optimiser of
  curved elements is asked for ``high_order_threshold``, an element whose quality falls below
  ``quality_floor`` asks for a rebuild (trigger ``absolute-quality``), and pyoomph's net for an
  element turned inside out within a step is armed (``inverted_element_remesh``; its rebuilds are
  counted, event ``remesh`` with ``reason`` ``inverted-element``);
* the stops before a second change of topology, on both chains (``reconnect.post_cut_stops``: the
  bridge, the chains, the bubble's tips, a new neck, a closing fold, which across the groove at the
  base of a jet is the ``annular-contact-approach``, a spline through the nodes of either chain
  that contacts itself, the bubble draining, then the guard of the interface), and ``tau-end`` once
  ``tau = t - t_cut`` reaches the cut's ``tau_end``. They are checked before every rebuild, the
  rebuilds for distortion included, since a state that calls for one is not rebuilt
  (``distortion_rebuild_skipped`` in its stop record): the spline test is the rebuild's own, so no
  rebuild meets a self-intersecting spline; a contact is reported as a ``fold`` of the resolved
  surface or a ``kink`` at the scale of the mesh (``contact``);
* the step is also at most ``courant h/|u|`` over every node of both chains
  (``reconnect.courant_limit``: the tips and jets of either chain wherever they go), at most
  ``capillary R^(3/2)`` with ``R`` the smallest radius of curvature of the three tips on the axis
  and of the bases of their jets, and grows by at most the cut's ``dt_growth``;
* no Newton failure ends the run cleanly: after a cut it is a failed run, whatever the state, and
  the profiles of the state the failing step started from are written first
  (``profile_<step>_lastgood.txt`` and its ``_bubble`` beside it); while an element of either
  chain turns by more than ``edge_turn_refine`` degrees a profile is written at every step;
* ``collapse.txt`` gets the columns ``tau`` and ``tau_p = t - t_p`` (``t_p`` the pinch time the cut
  recorded, NaN without one), ``chains.txt`` a row per step with both chains
  (``reconnect.post_cut_record``: the three tips, bridge, bubble, necks, budgets) and the step
  limits of the state (``dt_courant``, ``dt_capillary``), the profiles a ``_bubble`` file beside
  each one of the open chain, and every event, profile header and dump after the cut ``tau`` and
  ``tau_p``;
* ``h_near`` is read where the focus sets it, on both chains, the coarser counting: every mesh
  after the cut keeps the cut's sizing at the tips on the axis and along the bridge, so only edges
  where that sizing asks for at least ``CUT_SIZING_MARGIN`` focus sizes count
  (:meth:`CollapseDriver._h_near_after_cut`; the rule above reads its regions instead);
* the rotating dumps follow at least each time the time after the cut doubles, besides every
  ``state_dt`` (:meth:`CollapseDriver._state_interval`), and the state the run ends on is dumped
  as well (``end_state`` in the stop record), whatever the stop: the latest dump, which the rotation
  keeps (``state_keep`` is at least one); the dumps also keep which of the bubble's tips carry a
  jet (``tip_born``), the apex radii of the last step (``apex_history``) and the kink regions
  (``kinks``);
* landing states and the stop state are not kept (one cut per run).

With the events block (``events.EventSettings``) the stops after a cut are those of the events
branch (``events.post_cut_events``), and every change of topology is an event the stops request.
At each check of the stops (the loop's top, after a step, after a rebuild) the run meets the
request (:meth:`CollapseDriver._meet_event`): the cut of a landed liquid neck it makes
(``surgery.liquid_cut``), and goes on from the fresh mesh (:meth:`CollapseDriver.
reset_after_event`), whose stops it checks again, as those of any mesh freshly built; on every
other event, the event after ``max_events``, one the cascade test of its tip judges to shrink
(``events.CascadeMonitor``), a cut refused and one whose hard budgets do not close it stops
resumably, with the dump of the state before the event (``precut_e<k>``). ``events.txt`` holds a
row per event made.

Dumps written after a cut carry its record, which :meth:`CollapseDriver.restore` restores
(``surgery.restore_cut``), only into a driver whose reconnect settings are those of the cut but for
``tau_end`` (``reconnect.CONTINUATION_KEYS``), which a continuation may raise; the run then
continues in the post-cut mode.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .collapse import (Profile, axis_curvature, cone_angles, cone_fit, gauss_disc_points,
                       jet_base, necks)
from .episodes import ETA_HYS, EpisodeTracker
from .events import (EVENT_CASCADE, EVENT_CLASSES, EVENT_LIMIT, EVENT_REQUEST, EVENT_STOPS,
                     EVENT_UNCLOSED, EVENT_UNHANDLED, FAR_TIPS, IMPLEMENTED_HANDLERS,
                     LIQUID_CUT_REFUSED, PRECUT_EVENT, SIZING_KEYS, CascadeMonitor, EventSettings,
                     event_neck, event_tip, events_done, fold_stop_after_cut, ghost_place,
                     ghost_recontacts, minimum_dt_after_cut)
from .mesh import Focus, FocusRegion
from .reconnect import (AXIS, BUBBLE_TIP_SIGNS, CONTINUATION_KEYS, CUT_SIZING_MARGIN, JET_TIPS,
                        LANDING_RATIO, MAX_LANDING_RATIO, Chain, CutGeometry, ReconnectSettings,
                        bubble_end, courant_limit, cut_size_field, edge_reading, feature_regions,
                        groove_gap, interface_guard, jet_base_apex, kink_regions, post_cut_record,
                        post_cut_stops, region_spacing, tau_end_reached, tip_jet)
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
    state_keep: int = 4  # at least one: the latest dump, the end state after a cut, is kept
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
        keep = self.state_keep
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            # the rotation keeps the latest state_keep dumps: with none, a dump would be deleted as
            # soon as it is written, the dump of the state a run after a cut ends on included
            raise ValueError("state_keep must be an integer of at least one (the latest dump, "
                             "after a cut the end state a continuation starts from, is kept)")

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
                 "remesh_count", "ell_bubble") + tuple(
                     f"{c}_{tip}" for tip in JET_TIPS
                     for c in ("r_j", "z_b", "Q_j", "V_j", "We_j", "jet_born")) + tuple(
                     # the mesh at each tip: the longest element within its focus region and the
                     # size the region asks for (CollapseDriver._feature_regions)
                     f"{c}_{tip}" for tip in JET_TIPS for c in ("h", "size")) + tuple(
                     # the base of each tip's jet, the apex of the groove around it: its radius of
                     # curvature, the gap of the groove's closest fold pair and its opening angle
                     # (reconnect.jet_base_apex, reconnect.groove_gap)
                     f"{c}_{tip}" for tip in JET_TIPS
                     for c in ("R_apex", "groove_gap", "groove_angle")) + (
                     # the worst quadratic element of each chain (reconnect.edge_reading): the turn
                     # of its tangent (degrees) and the offset of a midside node from mid-chord
                     "edge_turn_open", "edge_turn_bubble", "edge_midside_open",
                     "edge_midside_bubble")
# chains.txt columns written as integers (NaN where the state has no bubble)
INT_CHAIN_COLUMNS = ("step", "courant_chain", "remesh_count") + tuple(
    f"jet_born_{tip}" for tip in JET_TIPS)
# after a cut with the events block: events.txt, a row per event the run makes (its record is
# events/event_<k>.json): the closures F1-F6 relative to the removed head's own, the jet's base
# before the event and after it, the interval since the event before at its tip and the verdict of
# the cascade, the totals of the removals since the gas cut
EVENT_COLUMNS = ("k", "step", "time", "tau", "tau_p", "tau_over_delta2", "kind", "class", "chain",
                 "tip", "generation", "r_n", "s", "r_n_over_s", "z_n", "z_tip_old", "z_tip_new",
                 "eta", "V_rem", "Pz_rem", "KE_rem", "U_ghost", "du_stump", "r_j", "z_b",
                 "r_j_over_r_n", "neck_to_base_over_r_j", "r_j_after", "Re_n", "We_n", "F1", "F2",
                 "F3", "F4", "F5", "F6", "phi_relative_error", "predicted_unlocated",
                 "reported_unlocated", "dt_history", "interval_over_delta2", "events_per_T",
                 "cascade", "V_removed_total", "Pz_removed_total", "KE_removed_total", "wall")
# drops.txt, a row per change of the ghost of a removed head: made (status free), and each of its
# recontacts (status rear or front, events.ghost_recontacts), with its ends on the axis then, the
# tip it met and its height
DROP_COLUMNS = ("k", "status", "step", "time", "tau", "tau_p", "tip", "far_tip", "z_front",
                "z_rear", "z_tip", "U", "v_ret", "V")


def _table_cell(v) -> str:
    """A cell of the tables of the events: numbers as collapse.txt writes them, text as it is."""
    if v is None:
        return "nan"
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.12e}"
    return str(v)


def _finite(x):
    """A number of a record, None where it is not finite (as the cut's records have it)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@dataclass
class _State:
    step: int = 0
    dt_prev: float | None = None
    last_remesh_step: int = -10**9
    refine_blocked_until: int = -10**9  # first step at which a refining rebuild may follow
    coarsen_held_until: float = -math.inf  # time before which the focus is not coarsened
    # after a cut: first step at which a coarsening rebuild may follow one that did not coarsen
    coarsen_blocked_until: int = -10**9
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
    # after a cut: whether a jet stands on each of the bubble's tips (reconnect.BUBBLE_TIP_SIGNS)
    tip_born: dict = field(default_factory=dict)
    # after a cut: the radius of the base of each tip's jet at the end of the last accepted step,
    # [t, R_apex] by tip, from which its rate of change is read, and the kink regions by chain
    # (reconnect.kink_regions)
    apex_history: dict = field(default_factory=dict)
    kinks: dict = field(default_factory=dict)
    # after a cut with the events block: the legs of the run, a restart or the cut each starting one
    legs: list = field(default_factory=list)


def _in_ulps(dt: float, t: float) -> float | None:
    """``dt`` in units of the spacing of doubles at ``t`` (``math.ulp``); None at ``t = 0``, where
    that spacing is the subnormals'."""
    return None if t == 0.0 else dt / math.ulp(t)


class CollapseDriver:
    def __init__(self, problem, settings: CollapseSettings, outdir: Path, *,
                 keep_stop_state: bool = False, landing=(),
                 reconnect: ReconnectSettings | None = None,
                 events: EventSettings | None = None, leg_info: dict | None = None):
        self.pb = problem
        self.s = settings
        self.keep_stop_state = keep_stop_state
        self.landing = tuple(sorted((float(x) for x in landing), reverse=True))
        check_landing(self.landing, settings)
        # the post-cut mode: the run continues after a cut made with these settings
        self.reconnect = reconnect
        if reconnect is not None and (self.landing or keep_stop_state):
            raise ValueError("a run after a cut keeps no landing or stop states (one cut per run)")
        # the events after the cut (events.EventSettings), and what each leg of the run records of
        # its source (the commit): only after a cut
        self.events = events
        self.leg_info = dict(leg_info or {})
        if events is not None and reconnect is None:
            raise ValueError("the events block applies after a cut: a driver with it needs the "
                             "reconnect settings")
        if events is not None:  # oomph-lib's smallest step after the cut, against the floor then
            minimum_dt_after_cut(events.post, settings.dt_min if events.post.dt_min is None
                                 else events.post.dt_min)
        self._post_cut_cache: tuple | None = None
        # the measured state has had its stop checks but not yet its snapshot and remesh checks
        self._checks_pending = False
        # rebuilds asked for by element distortion in this run: the driver's, once a focus is set,
        # and pyoomph's own inside the solve before that
        self._distortion_remeshes = 0
        self._pyoomph_remeshes = 0
        # after a cut: the sub-steps flagged for a rebuild that a stop of the step's state cancelled
        self._distortion_skipped = 0
        # after a cut: the rebuilds pyoomph made inside a step for an element turned inside out
        # (RemeshWhen's on_inverted_element, armed by run())
        self._inversion_remeshes = 0
        # the remesh count at the end of the last accepted step: a mesh built since is fresh (the
        # guard of the interface and the check of its curved elements, after a cut)
        self._stepped_rc: int | None = None
        self._nets_armed = False  # after a cut: pyoomph's inverted-element net (run())
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
        # with the events block: a row per event the run makes (events.txt), and what an event in
        # the making hands its record (_meet_event, _on_event_made)
        self._event_rows = self._drops = None
        self._event_before: tuple | None = None
        self._event_after: dict | None = None
        if events is not None:
            self._event_rows = open(self.outdir / "events.txt", "w", buffering=1)
            self._event_rows.write("# " + "\t".join(EVENT_COLUMNS) + "\n")
            self._drops = open(self.outdir / "drops.txt", "w", buffering=1)
            self._drops.write("# " + "\t".join(DROP_COLUMNS) + "\n")
        (self.outdir / "collapse_settings.json").write_text(
            json.dumps(asdict(settings), indent=2) + "\n")

    def close(self):
        self._table.close()
        self._events.close()
        if self._chains is not None:
            self._chains.close()
        for f in (self._event_rows, self._drops):
            if f is not None:
                f.close()
        if self._nets_armed:  # the detection is process-wide (pyoomph); later problems start clean
            disarm = getattr(self.pb, "disarm_inverted_element_remesh", None)
            if disarm is not None:
                disarm()
            self._nets_armed = False

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
        """Both chains of the current state, their record (``reconnect.post_cut_record``), the jets
        of the bubble's tips (``reconnect.tip_jet``, without the flux), the base of each tip's jet
        (``apexes``: :meth:`_apexes`), the quadratic elements of each chain (``edges``:
        ``reconnect.edge_reading``) and the step limits they set, computed once per state."""
        key = (self._t(), self.pb.remesh_count, self.state.step)
        if self._post_cut_cache is not None and self._post_cut_cache[0] == key:
            return self._post_cut_cache[1]
        s = self.s
        segs = self.pb.interface_segments()
        rec = post_cut_record(segs) if len(segs) == 2 else {}
        cl = courant_limit(segs)
        tips, apexes = {}, {}
        if len(segs) == 2:
            bubble = next(c for c in segs if not c["open"])
            tips = {name: tip_jet(bubble_end(bubble, name), r_search=s.r_search,
                                  rise_tol=self._rise_tol(), flat_aspect=s.flat_aspect,
                                  jet_aspect=s.jet_aspect, jet_max_radius=s.jet_max_radius)
                    for name in BUBBLE_TIP_SIGNS}
            apexes = self._apexes(next(c for c in segs if c["open"]), bubble, tips)
        edges = {("open" if c["open"] else "bubble"):
                 edge_reading(np.column_stack([c["r"], c["z"]])) for c in segs}
        R_apex = [a["R_apex"] for a in apexes.values()
                  if math.isfinite(a["R_apex"]) and a["R_apex"] > 0.0]
        R_apex_min = min(R_apex) if R_apex else math.inf
        # the capillary time of the sharpest tip on the axis and of the sharpest jet base
        R = min([x for x in (rec.get("R_tip_min", math.nan), R_apex_min) if math.isfinite(x)],
                default=math.nan)
        state = {"segs": segs, "record": rec, "courant": cl, "tips": tips, "apexes": apexes,
                 "edges": edges, "R_apex_min": R_apex_min,
                 "dt_courant": self.reconnect.courant * cl["h_over_u"],
                 "dt_capillary": self.s.capillary * R**1.5 if math.isfinite(R) else math.inf}
        self._post_cut_cache = (key, state)
        return state

    def _apexes(self, cavity: dict, bubble: dict, tips: dict) -> dict:
        """The base of the jet on each tip on the axis after a cut where a resolved bump stands (the
        rule by which its radius is a scale): the base (``r``, ``z``, the jet-base radius and
        height in the frame of the axis), its node along its chain (``chain``, ``index``), the
        vertex of the apex there (``apex_r``, ``apex_z``), its radius of curvature ``R_apex`` and
        the opening angle of the groove around the jet (``reconnect.jet_base_apex``), and the fold
        of that groove (``groove_gap``, ``groove_limit``, ``groove_arc``: ``reconnect.groove_gap``,
        with the driver's fold limits). By tip name."""
        s = self.s
        out = {}
        r, z = np.asarray(cavity["r"], dtype=float), np.asarray(cavity["z"], dtype=float)
        jb = jet_base(Profile(r, z), s.r_search, self._rise_tol())
        rise = float(z[0]) - jb.z_b
        if jb.jet_born and jb.r_j > 0.0 and not rise <= s.flat_aspect * jb.r_j:
            a = jet_base_apex(r, z, jb.index)
            if a["apex"] > 0:
                out["cavity_tip"] = {**a, "chain": "open", "index": a["apex"], "r": jb.r_j,
                                     "z": jb.z_b}
        n = len(bubble["r"])
        for name, tj in tips.items():
            if not (tj["bump"] and tj["r_j"] > 0.0 and not tj["flat_bump"]):
                continue
            end = bubble_end(bubble, name)
            a = jet_base_apex(end["r"], end["z"], tj["index"])
            if a["apex"] <= 0:
                continue
            a["apex_z"] *= end["sign"]  # in the frame of the axis
            out[name] = {**a, "chain": "bubble",
                         "index": a["apex"] if end["sign"] > 0 else n - 1 - a["apex"],
                         "r": tj["r_j"], "z": tj["z_b"]}
        for a in out.values():
            c = cavity if a["chain"] == "open" else bubble
            g = groove_gap(c["r"], c["z"], a["index"], ratio=s.fold_ratio, stop=self._fold_stop(),
                           stop_elements=s.fold_stop_elements)
            a.update(groove_gap=g["gap"], groove_limit=g["limit"], groove_arc=g["arc"])
        return out

    def _fresh_mesh(self) -> bool:
        """Whether the mesh was built since the last accepted step (the cut's, or a rebuild's)."""
        return self._stepped_rc is None or self.pb.remesh_count != self._stepped_rc

    def _post_cut_stop(self) -> dict | None:
        """The stops before a second change of topology, on both chains (post-cut mode), with the
        guard of the interface on its quadratic elements; on a freshly built mesh also the check of
        its curved elements (:meth:`_jacobian_stop`): a rebuild is accepted only if every element
        is valid."""
        cut = self._cut_record()
        if cut is None:
            return None
        pc = self._post_cut_state()
        segs = pc["segs"]
        # the chains as a rebuild takes them (mesh.FaradayCylinderMesh._current_chains: the ends on
        # the axis put on r = 0, the wall node already on the wall), so that the spline test of
        # the stops is the rebuild's own
        chains = []
        for c in segs:
            P = np.column_stack([c["r"], c["z"]]).astype(float)
            for k, end in ((0, 0), (-1, 1)):
                if c["end_types"][end] == AXIS:
                    P[k, 0] = 0.0
            chains.append(Chain(P, tuple(c["end_types"])))
        apexes: dict[str, list] = {}
        for tip, a in pc["apexes"].items():
            apexes.setdefault(a["chain"], []).append({"tip": tip, **a})
        s = self.s
        fresh = self._fresh_mesh()
        if self.events is not None:
            # the events branch: invalid states, the end of the window, then the events, each a
            # request (reason event-request) the run meets with its handler (_meet_event)
            stop = post_cut_stops(chains, CutGeometry(**cut["geometry"]), self.reconnect,
                                  cut["sizing"]["h_tip"], neck_elements=s.neck_stop_elements,
                                  fold_ratio=s.fold_ratio, fold_stop=s.fold_stop,
                                  fold_stop_elements=s.fold_stop_elements, fold_search=FOLD_SEARCH,
                                  apexes=apexes, guard={"fresh": fresh, "h_floor": self._h_floor(),
                                                        "refine_ratio": s.refine_ratio},
                                  events=self.events,
                                  invalid=self._jacobian_stop if fresh else None)
            if stop is not None:
                stop.update(self._taus(self._t()))
            return stop
        stop = post_cut_stops(chains, CutGeometry(**cut["geometry"]), self.reconnect,
                              cut["sizing"]["h_tip"], neck_elements=s.neck_stop_elements,
                              fold_ratio=s.fold_ratio, fold_stop=s.fold_stop,
                              fold_stop_elements=s.fold_stop_elements, fold_search=FOLD_SEARCH,
                              apexes=apexes, guard={"fresh": fresh, "h_floor": s.h_floor,
                                                    "refine_ratio": s.refine_ratio})
        if stop is None and fresh:
            stop = self._jacobian_stop()
        if stop is not None:
            stop.update(self._taus(self._t()))
        return stop

    def _event_stop(self, reason: str, request: dict, **extra) -> dict:
        """The stop of the run at an event (``events.post_cut_events``' request; ``reason`` one of
        ``events.EVENT_STOPS``), with the event's number ``k`` (one more than the events made) and
        what found it; at the end of the run the state is kept as the state before the event,
        ``precut_e<k>`` (``run``), unless the cut of the event kept it already, from which a run
        continues (with the event's handler, once one exists)."""
        out = {key: v for key, v in request.items() if key != "reason"}
        return {"reason": reason, "k": len(self._made_events()) + 1, **out, **extra}

    def _unhandled_event(self, request: dict) -> dict:
        """The stop of an event whose handler the driver does not make
        (``events.IMPLEMENTED_HANDLERS``): ``event-unhandled``, a resumable stop."""
        return self._event_stop(EVENT_UNHANDLED, request)

    def _keep_event_state(self, stop: dict) -> Path:
        """The dump of the state before an event the run stops on (``precut_e<k>``, outside the
        rotation, ``cut_stage`` ``precut-event``, with the event as the stop found it): a
        continuation from it meets the event again, with its handler once one exists, and the
        keys of every class of events apply anew from it (``events.EventSettings.
        refused_changes``)."""
        k = int(stop["k"])
        event = {key: v for key, v in stop.items() if key != "reason"}
        return self.keep_state(self.outdir / "states" / f"precut_e{k:03d}.dump",
                               cut_stage=PRECUT_EVENT, event=event)

    def _jacobian_stop(self) -> dict | None:
        """``interface-unresolved`` (trigger ``curved-jacobian``) for a mesh with an element whose
        scaled Jacobian may fall below the cut's ``jacobian_min`` (``reconnect.p2_jacobian_bounds``,
        through ``FaradayProblem.curved_jacobian_bound``); None from a problem without the bound."""
        bound = getattr(self.pb, "curved_jacobian_bound", None)
        if bound is None:
            return None
        j = bound(self.reconnect.jacobian_min)
        if j is None or not j["min"] < self.reconnect.jacobian_min:
            return None
        return {"reason": "interface-unresolved", "trigger": "curved-jacobian",
                "jacobian_bound": j["min"], "limit": self.reconnect.jacobian_min, "r": j["r"],
                "z": j["z"], "on_interface": j["on_interface"], "elements_below": j["below"],
                "fresh": True}

    def _h_near_after_cut(self, r, z, near_r: float, focus_r: float, h_near: float) -> float:
        """The spacing near the collapse after a cut, where the focus sets it, on both chains.

        Every mesh after the cut keeps the cut's sizing at the tips on the axis and along the bridge
        (``mesh.CutSizing``), so the tips are meshed at about ``h_tip`` whatever focus the driver
        sets. Read there, the spacing could never rise above about ``h_tip`` and the coarsening
        would rebuild the mesh at every step without effect, once the target focus exceeds
        ``h_tip/coarsen_ratio``. It is therefore read only on edges whose two nodes the cut's sizing
        leaves to the focus: where that sizing (``reconnect.cut_size_field``) asks for at least
        ``CUT_SIZING_MARGIN`` times the focus size of the current mesh; within ``near_r`` of the
        axis (the usual window), else within the focus region ``focus_r``. On the open chain
        (``r``, ``z``), without any such edge, as before the cut (``h_near``); a bubble without one
        does not count. The coarser of the two chains is returned: a chain whose spacing near the
        axis is coarser than the target asks for a refinement although the other is fine (a jet on
        the bubble while the open chain's cap flattens), and a coarsening needs both to be fine.
        Both chains' nodes within ``focus_r`` of the axis are attractors of the focus
        (``mesh.FaradayCylinderMesh``)."""
        sizing = self._cut_record().get("sizing") or {}
        if not all(k in sizing for k in ("h_tip", "h_bridge", "grading")):
            return h_near
        segs = self._post_cut_state()["segs"]
        tips = [float(c["z"][k]) for c in segs for k, end in ((0, 0), (-1, 1))
                if c["end_types"][end] == AXIS]
        focus = self.pb.active_mesh_template().focus
        size = focus.size if focus is not None else self.pb.mesh_settings.h_axis

        def spacing(r, z) -> float | None:
            r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
            ds = np.hypot(np.diff(r), np.diff(z))
            near, in_focus = r <= near_r, r <= focus_r
            free = cut_size_field(r, z, tips, sizing) >= CUT_SIZING_MARGIN * size
            free_edges = free[:-1] & free[1:]
            for edges in ((near[:-1] | near[1:]) & free_edges,
                          (in_focus[:-1] | in_focus[1:]) & free_edges):
                if edges.any():
                    return 2.0 * float(ds[edges].min())
            return None

        h = spacing(r, z)
        h = h_near if h is None else h
        bubble = next((c for c in segs if not c["open"]), None)
        if bubble is not None:
            h_bubble = spacing(bubble["r"], bubble["z"])
            if h_bubble is not None:
                h = max(h, h_bubble)
        return h

    def _focus_floor(self) -> float:
        """The smallest focus size of the driver's rule: ``h_floor``, after a cut too. A cut sizes
        its features by their own scales, between ``h_floor`` and the cut's focus size
        (``reconnect.feature_regions``): the cut's focus size is the coarsest size it allows near
        the cut, not a floor, which met a sharp tip or the apex of a groove at the base of a jet
        with one or two elements. With the events block ``post.h_floor`` after the cut."""
        return self._h_floor()

    # --- after a cut: the limits and floors of the events block's post ------------------------

    def _post(self):
        """The events block's ``post`` after a cut; None before it or without the block, where
        every limit is the collapse's own."""
        if self.events is None or self._cut_record() is None:
            return None
        return self.events.post

    def _h_floor(self) -> float:
        """The smallest element size: ``h_floor``, after a cut ``post.h_floor`` if set."""
        post = self._post()
        return self.s.h_floor if post is None or post.h_floor is None else post.h_floor

    def _rise_tol(self) -> float:
        """The rise of a bump that makes a jet: ``rise_tol``, after a cut ``post.rise_tol`` if
        set."""
        post = self._post()
        return self.s.rise_tol if post is None or post.rise_tol is None else post.rise_tol

    def _dt_min(self) -> float:
        """The step-size floor: ``dt_min``, after a cut ``post.dt_min`` if set."""
        post = self._post()
        return self.s.dt_min if post is None or post.dt_min is None else post.dt_min

    def _fold_stop(self) -> float:
        """The absolute part of the fold limit: ``fold_stop``, after a cut with the block
        ``gap.kappa_g delta`` (``events.fold_stop_after_cut``)."""
        if self._post() is None:
            return self.s.fold_stop
        return fold_stop_after_cut(self.events, self.reconnect.delta, self.s.fold_stop)

    def _minimum_dt(self) -> float:
        """oomph-lib's smallest step after the cut with the block: ``post.minimum_dt``, or 1e-3
        times the driver's step-size floor (:meth:`_dt_min`), never above oomph-lib's default
        and at most half that floor (``events.minimum_dt_after_cut``, checked at the driver's
        construction)."""
        return minimum_dt_after_cut(self.events.post, self._dt_min())

    def _lower_minimum_dt(self) -> float:
        """After a cut with the block, at the cut and at every restart from a state after it:
        oomph-lib's smallest step (``Problem::Minimum_dt``, pyoomph's ``DTSF_minimum_dt``, 1e-12 by
        default) set to :meth:`_minimum_dt`. oomph-lib gives up on a step below it, and also on an
        accepted step whose next candidate, four times the step, would fall below it, so that at
        the steps a small cut radius needs (about 1e-3 delta^2) the default would end the run on a
        converged step and lose the halving that rescues a failed one. Before the cut, and without
        the block, it stays at the default: the classification of a failure before a cut reads
        oomph-lib's message there (:func:`classify_failure`)."""
        value = self._minimum_dt()
        self.pb.DTSF_minimum_dt = value
        return value

    def _leg_limits(self) -> dict:
        """After a cut with the block: the limits and floors a leg runs with, each the block's
        ``post`` key or the collapse's own (the floor of the necks' regions the cut's focus size
        without ``post.neck_floor``), and ``in_delta``, the lengths in units of the cut radius and
        the times in units of its square: a study in ``delta`` then shows which of them it scaled
        with ``delta``, a key no case set staying at its absolute value."""
        delta = self.reconnect.delta
        cut = self._cut_record()
        post = self.events.post
        neck_floor = (self.reconnect.focus_size * float(cut.get("rmin") or cut["delta"])
                      if post.neck_floor is None else post.neck_floor)
        lengths = {"h_floor": self._h_floor(), "neck_floor": neck_floor,
                   "rise_tol": self._rise_tol(), "fold_stop": self._fold_stop(),
                   "neck_limit": self._neck_limit()}
        times = {"dt_min": self._dt_min(), "minimum_dt": self._minimum_dt()}
        return {"delta": delta, **lengths, **times,
                "in_delta": {**{k: v / delta for k, v in lengths.items()},
                             **{k: v / delta**2 for k, v in times.items()}}}

    def _neck_limit(self) -> float:
        """The open chain's neck stop, ``max(neck_stop, neck_stop_elements h_floor)``, with the
        floor of :meth:`_h_floor`."""
        if self._post() is None:
            return self.s.neck_limit
        return max(self.s.neck_stop, self.s.neck_stop_elements * self._h_floor())

    def _apex_prediction(self, apexes: dict) -> dict:
        """The base of each jet (``_post_cut_state``'s ``apexes``) with the radius predicted
        ``apex_lookahead`` steps of the last step ahead, ``R_pred = R_apex - lookahead dt
        |dR_apex/dt|``, its rate read against the end of the last accepted step (``apex_history``),
        but never below half the radius read, so that a jump of the reading between two meshes
        does not refine by more than that; ``R_apex`` without a history."""
        st, r = self.state, self.reconnect
        t = self._t()
        out = {}
        for tip, a in apexes.items():
            a = dict(a)
            R, prev = a["R_apex"], st.apex_history.get(tip)
            a["R_pred"] = R
            if prev is not None and math.isfinite(R) and st.dt_prev and t > prev[0] \
                    and math.isfinite(prev[1]):
                rate = abs(R - prev[1]) / (t - prev[0])
                a["R_pred"] = max(R - r.apex_lookahead * st.dt_prev * rate, 0.5 * R)
            out[tip] = a
        return out

    def _feature_regions(self, cavity: dict, tips: dict, U: float = 0.0) -> tuple[list[dict], dict]:
        """After a cut: the features of both chains (``reconnect.feature_regions``: the tips the cut
        made at their own scale, never coarser than the cut's focus size, the bases of the jets,
        the bubble's bottom never coarser than at the landing, the necks) and the ``kink_<chain>``
        regions where the guard of the interface asks for a refinement (``reconnect.kink_regions``,
        held for ``coarsen_hold`` local times of their chord at the speed ``U``), each with the size
        it has in the current mesh (``size``, NaN for a feature the current focus has no region
        for) and the longest and shortest element within its region (``h_max``, ``h_min``,
        ``reconnect.region_spacing``; the shortest outside the finer regions within it, with the
        reach of their grading); empty in a state whose topology has changed, which a stop records.
        Also returns the kink regions as entries, which the step commits at its end."""
        pc = self._post_cut_state()
        segs = pc["segs"]
        if len(segs) != 2 or not pc["record"]:
            return [], {}
        s, cut, st = self.s, self._cut_record(), self.state
        regions = feature_regions(segs, cavity, tips, pc["record"], cut, self.reconnect,
                                  h_floor=self._h_floor(), h_axis=self.pb.mesh_settings.h_axis,
                                  elements_per_scale=(s.elements_per_scale
                                                      if self.reconnect.elements_per_scale is None
                                                      else self.reconnect.elements_per_scale),
                                  focus_extent=s.focus_extent,
                                  apexes=self._apex_prediction(pc["apexes"]),
                                  **self._post_region_keys())
        h_floor = self._h_floor()
        refine = interface_guard(pc["edges"], self.reconnect, fresh=self._fresh_mesh(),
                                 h_floor=h_floor, refine_ratio=s.refine_ratio)["refine"]
        kinks = kink_regions(st.kinks, pc["edges"], refine, t=self._t(), h_floor=h_floor,
                             hold=lambda L: s.coarsen_hold * local_time(L, U))
        regions += [{"name": g["name"], "target": g["target"], "r": g["r"], "z": g["z"],
                     "radius": g["radius"], "ell": math.inf, "floor": h_floor,
                     "released": g["released"]} for g in kinks.values()]
        focus = self.pb.active_mesh_template().focus
        axis_tips = [float(c["z"][k]) for c in segs for k, end in ((0, 0), (-1, 1))
                     if c["end_types"][end] == AXIS]
        for g in regions:
            current = focus.region(g["name"]) if focus is not None else None
            g["size"] = current.size if current is not None else math.nan
        grading = s.focus_grading
        for g in regions:
            own = g["size"] if math.isfinite(g["size"]) else g["target"]
            finer = []
            for f in regions:
                other = f["size"] if math.isfinite(f["size"]) else f["target"]
                if f is not g and other < own:
                    finer.append((f["r"], f["z"], f["radius"] + (own - other) / grading))
            g["h_max"], g["h_min"] = region_spacing(segs, g, axis_tips, cut.get("sizing"), own,
                                                    finer)
        return regions, kinks

    def _post_region_keys(self) -> dict:
        """The keys of the events block's ``post`` that size the regions after the cut
        (``reconnect.feature_regions``); none without the block."""
        if self.events is None:
            return {}
        post = self.events.post
        return {"bubble_bottom_sizing": post.bubble_bottom_sizing,
                "bubble_bottom_size": post.bubble_bottom_size, "neck_floor": post.neck_floor,
                "neck_elements_per_scale": post.neck_elements_per_scale}

    def _commit_post_cut(self, m: dict):
        """After a cut, at the end of an accepted step (its last measurement ``m``): keep the radius
        of each jet's base, from which the next step reads its rate, and the kink regions, dropping
        each released one whose size in the current mesh is back at the chord it met."""
        if self._cut_record() is None:
            return
        st = self.state
        t = m["time"]
        st.apex_history = {tip: [t, float(a["R_apex"])]
                           for tip, a in self._post_cut_state()["apexes"].items()
                           if math.isfinite(a["R_apex"])}
        focus = self.pb.active_mesh_template().focus
        kinks = {}
        for chain, g in (m.get("kinks") or {}).items():
            current = focus.region(g["name"]) if focus is not None else None
            if g["released"] and (current is None or current.size >= g["h_before"] * (1 - 1e-9)):
                continue
            kinks[chain] = dict(g)
        st.kinks = kinks

    def _regions_focus(self, regions: list[dict], *, keep: bool) -> Focus:
        """The focus of a rebuild after a cut: one region per feature (``mesh.FocusRegion``) at its
        current centre and radius. A rebuild of the rule (:meth:`_maybe_remesh_features`) sizes
        each at its target, refining it by at most ``max_refine_factor`` on its longest element and
        coarsening it by at most ``max_coarsen_factor`` on its size in the current mesh (on its
        shortest element for a feature that has no region yet), but never coarser than a size it
        keeps from the landing (``kept``); a rebuild for distortion (``keep``) keeps the size of
        every region the current mesh has, so that it coarsens nothing. The tips the cut made have
        the cut's focus size as their target, so every rebuild meshes them at it."""
        s = self.s
        old = self.pb.active_mesh_template().focus
        out = []
        for g in regions:
            prev = old.region(g["name"]) if old is not None else None
            if keep and prev is not None:
                size = prev.size
            else:
                size = g["target"]
                if math.isfinite(g["h_max"]):
                    size = max(size, g["h_max"] / s.max_refine_factor)
                bound = prev.size if prev is not None else g["h_min"]
                if math.isfinite(bound):
                    size = min(size, s.max_coarsen_factor * bound)
                if g.get("kept") is not None:
                    size = min(size, g["kept"])
            out.append(FocusRegion(g["name"], size=size, r=g["r"], z=g["z"], radius=g["radius"]))
        return Focus(size=min(x.size for x in out), radius=max(x.radius for x in out),
                     grading=s.focus_grading, regions=tuple(out))

    @staticmethod
    def _region_record(regions: list[dict], focus: Focus | None = None) -> list[dict]:
        """The regions of a ``remesh`` event: the measured ones, with the size the rebuild set."""
        out = []
        for g in regions:
            new = focus.region(g["name"]) if focus is not None else None
            out.append({k: g[k] for k in ("name", "target", "size", "h_max", "h_min", "r", "z",
                                          "radius")}
                       | ({"new_size": new.size} if new is not None else {}))
        return out

    def _write_chain_row(self, m: dict, dt: float):
        pc = self._post_cut_state()
        obs = self.pb.cut_observables()
        cl = pc["courant"]
        t = self._t()
        values = {"step": self.state.step, "time": t, **self._taus(t), "dt": dt, **pc["record"],
                  **{k: obs[k] for k in ("volume", "kinetic_energy", "momentum_z",
                                         "divergence_sq")},
                  "h_courant": cl["h"], "U_courant": cl["U"], "courant_chain": cl["chain"],
                  "dt_courant": pc["dt_courant"], "dt_capillary": pc["dt_capillary"],
                  "remesh_count": self.pb.remesh_count,
                  "ell_bubble": m.get("ell_bubble", math.inf)}
        # the jets of the three tips: the open chain's as in collapse.txt, the bubble's beside it
        jets = {"cavity_tip": m, **m.get("tips", {})}
        for tip, jet in jets.items():
            for k in ("r_j", "z_b", "Q_j", "V_j", "We_j"):
                values[f"{k}_{tip}"] = jet.get(k, math.nan)
            values[f"jet_born_{tip}"] = int(bool(jet.get("jet_born", False)))
        for g in m.get("regions", []):
            if g["name"] in JET_TIPS:
                values[f"h_{g['name']}"], values[f"size_{g['name']}"] = g["h_max"], g["size"]
        for tip, a in pc["apexes"].items():  # the base of each jet: NaN without a jet
            for k in ("R_apex", "groove_gap", "groove_angle"):
                values[f"{k}_{tip}"] = a[k]
        for chain, rd in pc["edges"].items():
            values[f"edge_turn_{chain}"] = rd["turn"]
            values[f"edge_midside_{chain}"] = rd["midside"]

        def cell(c):
            v = values.get(c, math.nan)
            if c in INT_CHAIN_COLUMNS:
                return str(int(v)) if isinstance(v, (int, np.integer)) else "nan"
            return f"{float(v):.12e}"

        self._chains.write("\t".join(cell(c) for c in CHAIN_COLUMNS) + "\n")

    def _bubble_features(self) -> tuple[dict | None, dict, float]:
        """After a cut: the bubble chain, the jet bases of its two tips (``reconnect.tip_jet``, by
        name, without the flux) and the smallest of its scales, ``ell_bubble`` (``inf`` without a
        bubble): the jet-base radius of each tip where a resolved bump stands (the open chain's
        rule), its narrowest neck and its length, the gap between its tips.

        The radii of curvature of the bubble's tips are not among them. Read at the end of a chain,
        which every rebuild re-splines through the current nodes with no condition on the axis, the
        radius of a tip depends on the mesh there: a tip whose spline leaves the axis at a slope
        (on the synthetic cavity of the tests, from its coarse first mesh) is a cone, whose fitted
        radius shrinks with the element length, so each refinement reads it sharper and, as a
        scale, would refine it again (on that cavity the radius read at the bubble's bottom fell
        from 0.016 to 6.6e-4 in four rebuilds, the step with it). They bound the step instead
        (``dt_capillary``, from ``R_tip_min``), and the fresh top's radius sizes the mesh at that
        tip (``reconnect.feature_regions``), floored at ``h_floor``."""
        pc = self._post_cut_state()
        bubble = next((c for c in pc["segs"] if not c["open"]), None)
        if bubble is None or len(pc["segs"]) != 2:  # a changed topology, which a stop records
            return None, {}, math.inf
        tips = {name: dict(tj) for name, tj in pc["tips"].items()}  # completed by measure()
        rec = pc["record"]
        scales = [rec["r_neck_bubble"], rec["z_bubble_top"] - rec["z_bubble_bottom"]]
        scales += [tj["r_j"] for tj in tips.values()
                   if tj["bump"] and tj["r_j"] > 0.0 and not tj["flat_bump"]]
        finite = [float(x) for x in scales if math.isfinite(x) and x > 0.0]
        return bubble, tips, min(finite) if finite else math.inf

    def _complete_tip_jet(self, name: str, tj: dict, h_focus: float,
                          probe: tuple[list[float], bool] | None = None):
        """Complete the jet of one of the bubble's tips (:meth:`_bubble_features`) in place, as
        :meth:`measure` does the open chain's: whether its axis end is the lowest point of the end
        again (``axis_lowest``), whether a jet stands there (``jet_born``: one did at the last step
        and the end is not lowest again, or it passes the test of a born jet now), and, while one
        stands, the flux ``Q_j`` through the disc of radius ``r_j`` at the base, the mean speed
        ``V_j = Q_j/(pi r_j^2)`` and ``We_j = V_j^2 r_j``. ``Q_j`` and ``V_j`` are positive along
        the jet, into the bubble: downwards from the bubble's top. Without ``probe`` (the axial
        velocity at :meth:`_tip_probe_points`, which :meth:`measure` locates with the others) it
        reads the velocity itself."""
        s = self.s
        flat = tj["index"] > 0 and tj["rise"] <= s.flat_aspect * max(tj["r_j"], h_focus)
        tj["axis_lowest"] = bool(tj["index"] == 0 or flat)
        tj["jet_born"] = bool((self.state.tip_born.get(name, False) and not tj["axis_lowest"])
                              or tj["jet"])
        tj.update(Q_j=math.nan, V_j=math.nan, We_j=math.nan, flux_found=True)
        if tj["jet_born"] and tj["r_j"] > 0.0:
            _, ww = gauss_disc_points(tj["r_j"], s.probe_points)
            w, found = probe if probe is not None else self.pb.axial_velocity_at(
                self._tip_probe_points(tj))
            tj["flux_found"] = found
            if found:
                Q = BUBBLE_TIP_SIGNS[name] * float(np.dot(ww, w))
                V = Q / (math.pi * tj["r_j"] ** 2)
                tj.update(Q_j=Q, V_j=V, We_j=V**2 * tj["r_j"])

    def _probe(self, sets) -> list[tuple[list[float], bool]]:
        """The axial velocity at each set of points: one location of them all where the problem
        offers it (``FaradayProblem.axial_velocity_sets``), else one call per set."""
        many = getattr(self.pb, "axial_velocity_sets", None)
        if many is not None:
            return many(sets)
        return [self.pb.axial_velocity_at(pts) for pts in sets]

    def _tip_probe_points(self, tj: dict) -> list[tuple[float, float]]:
        """The points of the disc at the base of a tip's jet at which :meth:`_complete_tip_jet`
        reads the axial velocity."""
        rr, _ = gauss_disc_points(tj["r_j"], self.s.probe_points)
        return [(x, tj["z_b"]) for x in rr]

    def _tip_jet_probed(self, name: str, tj: dict, h_focus: float) -> bool:
        """Whether :meth:`_complete_tip_jet` will read a velocity for this tip (a jet stands)."""
        s = self.s
        flat = tj["index"] > 0 and tj["rise"] <= s.flat_aspect * max(tj["r_j"], h_focus)
        lowest = bool(tj["index"] == 0 or flat)
        born = bool((self.state.tip_born.get(name, False) and not lowest) or tj["jet"])
        return born and tj["r_j"] > 0.0

    def _track_tips(self, m: dict, snapshot: bool = True):
        """After a cut: record the jets of the bubble's tips that the measured state ``m`` finds
        born or gone (events ``tip-jet-born`` and ``tip-jet-gone``, with the tip's name), and keep
        them for the next measurement and the dumps."""
        st = self.state
        for name, tj in m.get("tips", {}).items():
            if tj["jet_born"] == bool(st.tip_born.get(name, False)):
                continue
            st.tip_born[name] = tj["jet_born"]
            kind = "tip-jet-born" if tj["jet_born"] else "tip-jet-gone"
            self._event(kind, m, tip=name, tip_r_j=tj["r_j"], tip_z_b=tj["z_b"],
                        tip_z=tj["z_tip"], tip_V_j=tj["V_j"])
            if snapshot and tj["jet_born"]:
                self._maybe_snapshot(m, kind)

    # --- measurement --------------------------------------------------------------------------

    def measure(self) -> dict:
        s, pb = self.s, self.pb
        r, z, ur, uz = pb.interface_arrays()
        prof = Profile(r, z)
        kappa = axis_curvature(prof)
        rise_tol = self._rise_tol()
        jb = jet_base(prof, s.r_search, rise_tol)
        rise = float(z[0]) - jb.z_b
        flat_bump = jb.index > 0 and rise <= s.flat_aspect * jb.r_j
        cavity = self.state.cavity_seen or jb.index == 0
        born = (self.state.born or (cavity and jb.jet_born and jb.r_j < s.jet_max_radius
                                    and rise > max(rise_tol, s.jet_aspect * jb.r_j)))
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
        ell_open = min(min(scales) if scales else ell_cap, ell_cap)
        # after a cut the bubble's features count too (inf before a cut or without a bubble)
        after_cut = self._cut_record() is not None
        bubble, tips, ell_bubble = self._bubble_features() if after_cut else (None, {}, math.inf)
        # and the radius of curvature at the base of each jet, the apex of the groove around it
        R_apex = self._post_cut_state()["R_apex_min"] if after_cut else math.inf
        ell_measured = min(ell_open, ell_bubble, R_apex)
        ell = limit_scale(ell_measured, self.state.ell_prev, s.scale_growth)
        h_focus = float(np.clip(ell / s.elements_per_scale, self._focus_floor(), h_axis))

        Q = V = We = math.nan
        found = True
        # the discs at the base of every jet that stands, located in one call
        sets = {}
        if born and r_j > 0.0:
            rr, ww = gauss_disc_points(r_j, s.probe_points)
            sets[None] = [(x, jb.z_b) for x in rr]
        for name, tj in tips.items():
            if self._tip_jet_probed(name, tj, h_focus):
                sets[name] = self._tip_probe_points(tj)
        probes = dict(zip(sets, self._probe(list(sets.values()))))
        if None in probes:
            w, found = probes[None]
            if found:
                Q = float(np.dot(ww, w))
                V = Q / (math.pi * r_j**2)
                We = V**2 * r_j
        for name, tj in tips.items():
            self._complete_tip_jet(name, tj, h_focus, probes.get(name))

        near_r = max(2.0 * ell, 4.0 * h_focus)
        near = r <= near_r
        ds = np.hypot(np.diff(r), np.diff(z))
        near_edges = near[:-1] | near[1:]
        h_near = 2.0 * float(ds[near_edges].min()) if near_edges.any() else math.inf

        focus_r = min(pb.nd.radius, s.focus_extent * ell)
        # the region of the open chain's own scale, for its fold stop and its speeds: after a cut a
        # feature of the bubble can narrow the focus, but not the open chain's fold search
        own_r = focus_r if not after_cut else max(focus_r,
                                                  min(pb.nd.radius, s.focus_extent * ell_open))
        in_own = r <= own_r
        if after_cut:
            h_near = self._h_near_after_cut(r, z, near_r, focus_r, h_near)
        # Select the fold closest to its own stop limit, using each node's local spacing.
        fold = fold_gap(r, z, r_max=own_r, ratio=s.fold_ratio,
                        stop=s.fold_stop, stop_elements=s.fold_stop_elements, search_factor=FOLD_SEARCH)
        if fold.crossed:
            raise InterfaceSelfIntersection(
                f"the free surface crosses itself within r <= {own_r:.6g} at t = {self._t():.15g}; "
                "the measured state is invalid and the last row before it is the last valid one")
        fold_limit = (max(s.fold_stop, s.fold_stop_elements * fold.h_elem)
                      if math.isfinite(fold.gap) else math.nan)
        speeds = [float(np.max(np.hypot(ur[in_own], uz[in_own]))) if in_own.any() else 0.0,
                  abs(V) if math.isfinite(V) else 0.0, 1e-3]
        if bubble is not None:  # the bubble's nodes in that region and the jets of its tips
            br = np.asarray(bubble["r"], dtype=float)
            sel = br <= own_r
            if sel.any():
                speeds.append(float(np.max(np.hypot(np.asarray(bubble["u_r"])[sel],
                                                    np.asarray(bubble["u_z"])[sel]))))
            speeds += [abs(tj["V_j"]) for tj in tips.values() if math.isfinite(tj["V_j"])]
        U = max(speeds)

        ref = max(r_j if born else 0.0,
                  1.0 / abs(kappa) if math.isfinite(kappa) and kappa != 0 else 0.0)
        lo, hi = s.cone_window[0] * ref, s.cone_window[1] * ref
        start = jb.index if cavity else 0
        beta, _, npts = cone_fit(prof, start, lo, hi)
        angles = cone_angles(prof, start, s.cone_radii)

        flat = jb.index > 0 and rise <= s.flat_aspect * max(jb.r_j, h_focus)
        # after a cut: the features of both chains, each meshed at its own size
        regions, kinks = self._feature_regions(
            {"z_tip": float(z[0]), "kappa": kappa, "r_j": jb.r_j, "rise": rise,
             "bump": jb.jet_born and cavity, "flat_bump": flat_bump}, tips,
            U) if after_cut else ([], {})
        return {"time": self._t(), "axis_lowest": jb.index == 0 or flat, "flat_bump": flat_bump,
                "z_axis": float(z[0]), "w_axis": float(uz[0]), "axis_curvature": kappa,
                "cavity": cavity, "jet_born": born, "r_j": r_j,
                "z_b": jb.z_b if cavity else math.nan, "Q_j": Q, "V_j": V,
                "We_j": We, "flux_found": found, "r_neck": r_neck, "z_neck": z_neck, "ell": ell,
                "ell_measured": ell_measured, "h_focus": h_focus, "h_near": h_near, "U": U, "beta_fit": beta,
                "beta_fit_points": npts, "beta_window_lo": lo, "beta_window_hi": hi,
                "angles": angles, "focus_radius": focus_r, "profile": (r, z, ur, uz),
                "fold_gap": fold.gap, "fold_r": fold.r, "fold_z": fold.z, "fold_arc": fold.arc,
                "fold_h_elem": fold.h_elem, "fold_limit": fold_limit,
                "tips": tips, "ell_bubble": ell_bubble, "R_apex": R_apex, "regions": regions,
                "kinks": kinks}

    # --- bookkeeping --------------------------------------------------------------------------

    def _event(self, kind: str, m: dict, /, **extra):
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
            self._write_chain_row(m, dt)

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
            # none only in a state that has changed topology, which a stop records
            bubble = next((c for c in self._post_cut_state()["segs"] if not c["open"]), None)
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
        after_cut = self._cut_record() is not None
        if after_cut and m.get("regions"):
            return self._maybe_remesh_features(m)
        target = m["h_focus"]
        refining = m["h_near"] > s.refine_ratio * target
        if refining:
            if st.step < st.refine_blocked_until:
                return False
            size = max(target, m["h_near"] / s.max_refine_factor)
        elif m["h_near"] < s.coarsen_ratio * target:
            if m["time"] < st.coarsen_held_until:  # held in physical time, not in steps
                return False
            if after_cut and st.step < st.coarsen_blocked_until:
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
                    h_near_after=after["h_near"], **hold, **self._axis_repair_note())
        if refining and after["h_near"] > 0.9 * m["h_near"]:
            # the rebuild did not refine: do not try again on every step; a growing scale may
            # coarsen the focus once the hold has passed
            st.refine_blocked_until = st.step + s.remesh_backoff + s.min_steps_between_remesh
            self._event("remesh-ineffective", after, focus_size=size)
        elif after_cut and not refining and after["h_near"] < m["h_near"] / 0.9:
            # after a cut the mesh also follows the cut's sizing and from curvature, which a focus
            # cannot coarsen: a rebuild that did not coarsen is not tried again on every step
            st.coarsen_blocked_until = st.step + s.remesh_backoff + s.min_steps_between_remesh
            self._event("remesh-ineffective", after, focus_size=size, direction="coarsen",
                        coarsen_blocked_until=st.coarsen_blocked_until)
        return True

    def _maybe_remesh_features(self, m: dict) -> bool:
        """The remesh rule after a cut, feature by feature (``m["regions"]``,
        :meth:`_feature_regions`): a rebuild refines when the longest element of some feature's
        region exceeds ``refine_ratio`` times its target, and coarsens, once the hold and any wait
        after a coarsening without effect have passed, when the shortest element of some feature's
        region lies below ``coarsen_ratio`` times its target; every region is then sized by
        :meth:`_regions_focus`. A refinement that refined none of the regions that asked for it
        waits ``remesh_backoff`` steps, and so does a coarsening that coarsened none (events
        ``remesh-ineffective``, with ``direction`` ``coarsen`` for the latter). The hold on
        coarsening after a refining rebuild is that of the rule before the cut."""
        s, st = self.s, self.state
        regions = m["regions"]
        refine = [g for g in regions
                  if math.isfinite(g["h_max"]) and g["h_max"] > s.refine_ratio * g["target"]]
        coarsen = [g for g in regions
                   if math.isfinite(g["h_min"]) and g["h_min"] < s.coarsen_ratio * g["target"]]
        if refine:
            if st.step < st.refine_blocked_until:
                return False
            refining, asked = True, refine
        elif coarsen:
            if m["time"] < st.coarsen_held_until or st.step < st.coarsen_blocked_until:
                return False
            refining, asked = False, coarsen
        else:
            return False
        template = self.pb.active_mesh_template()
        focus = self._regions_focus(regions, keep=False)
        template.focus = focus
        self.pb.force_remesh()
        st.last_remesh_step = st.step
        hold = {}
        if refining:
            held = m["time"] + s.coarsen_hold * local_time(m["ell"], m["U"])
            st.coarsen_held_until = max(st.coarsen_held_until, held)
            hold["coarsen_held_until"] = st.coarsen_held_until
        after = self.measure()
        self._event("remesh", m, focus_size=focus.size, focus_radius=focus.radius,
                    h_near_before=m["h_near"], h_near_after=after["h_near"],
                    asked_by=[g["name"] for g in asked],
                    regions=self._region_record(regions, focus), **hold,
                    **self._axis_repair_note())
        done = {g["name"]: g for g in after.get("regions", [])}
        if refining and not any(done.get(g["name"], {}).get("h_max", math.inf) <= 0.9 * g["h_max"]
                                for g in asked):
            st.refine_blocked_until = st.step + s.remesh_backoff + s.min_steps_between_remesh
            self._event("remesh-ineffective", after, focus_size=focus.size,
                        asked_by=[g["name"] for g in asked])
        elif not refining and not any(
                done.get(g["name"], {}).get("h_min", -math.inf) >= g["h_min"] / 0.9 for g in asked):
            st.coarsen_blocked_until = st.step + s.remesh_backoff + s.min_steps_between_remesh
            self._event("remesh-ineffective", after, focus_size=focus.size, direction="coarsen",
                        asked_by=[g["name"] for g in asked],
                        coarsen_blocked_until=st.coarsen_blocked_until)
        return True

    def distortion_counts(self) -> dict:
        """The rebuilds element distortion asked for in this run (``run_status.json``): the
        driver's, once a focus is set (events ``remesh`` with ``reason`` ``distortion``), and
        pyoomph's own inside the solve before that (no event; ``remesh_count`` counts them)."""
        return {"distortion_remeshes": self._distortion_remeshes,
                "pyoomph_remeshes": self._pyoomph_remeshes}

    def _defers_distortion(self) -> bool:
        """Whether the rebuild that element distortion asks for in the next step is the driver's:
        once a focus is set. pyoomph then only flags it (``do_call_remeshing_when_necessary``
        off); before, it rebuilds inside the solve, as it always did."""
        defer = self.pb.active_mesh_template().focus is not None
        self.pb.do_call_remeshing_when_necessary = not defer
        return defer

    def _distortion_flags(self) -> list[dict]:
        """What element distortion flagged in the step just taken, the request cleared
        (``FaradayProblem.take_distortion_flags``); nothing from a problem that cannot flag."""
        take = getattr(self.pb, "take_distortion_flags", None)
        return take() if take is not None else []

    def _distortion_remesh(self, flags: list[dict]) -> dict:
        """The rebuild that element distortion asked for in the step just taken, once a focus is
        set; returns the measurement of the rebuilt mesh, from which the step goes on.

        The step is measured on its own mesh first. The focus is then refreshed: its size to
        ``min(focus size, h_near)``, so that the rebuild does not coarsen what the mesh near the
        collapse resolves (it resets the edges compressed below the focus size to it otherwise),
        and its radius to that of the measured scale (``focus_radius``), not of the scale at
        which the focus was last set; its grading is kept. Event ``remesh`` with ``reason``
        ``distortion``: the first flagged sub-step's reading (``trigger``, ``ratio`` and the
        extremes, ``FaradayProblem.distortion_reading``), the number of flagged sub-steps, the
        focus before and after and ``h_near`` before and after. No hold starts or ends, and
        ``last_remesh_step`` is kept, so that the rule of :meth:`_maybe_remesh` follows on the
        rebuilt mesh within the same step, as after pyoomph's own rebuild."""
        m = self.measure()
        template = self.pb.active_mesh_template()
        old = template.focus
        extra = {}
        if self._cut_record() is not None:
            # after a cut the stops come first: a state that calls for one is not rebuilt, so that
            # no rebuild meets a self-intersecting spline (reconnect.post_cut_stops); the run then
            # stops on it, at the end of this step
            if (self._geometry_stop(m) or self._post_cut_stop()) is not None:
                self._distortion_skipped = len(flags)
                return m
            if m.get("regions"):
                new = self._regions_focus(m["regions"], keep=True)
                extra["regions"] = self._region_record(m["regions"], new)
            else:
                new = Focus(size=min(old.size, m["h_near"]), radius=m["focus_radius"],
                            grading=old.grading)
        else:
            new = Focus(size=min(old.size, m["h_near"]), radius=m["focus_radius"],
                        grading=old.grading)
        template.focus = new
        self.pb.force_remesh()
        self._distortion_remeshes += 1
        after = self.measure()
        self._event("remesh", m, reason="distortion", **flags[0], flagged_substeps=len(flags),
                    focus_size=new.size, focus_radius=new.radius, focus_size_before=old.size,
                    focus_radius_before=old.radius, h_near_before=m["h_near"],
                    h_near_after=after["h_near"], **extra, **self._axis_repair_note())
        return after

    def _axis_repair_note(self) -> dict:
        """For the ``remesh`` event of a rebuild after a cut that put misplaced midside nodes of the
        axis back on their edges (``FaradayProblem.repair_axis_midsides``): ``axis_repair``, with
        the nodes repaired and the largest offset found; nothing otherwise."""
        repair = getattr(self.pb, "axis_repair", None)
        return {"axis_repair": dict(repair)} if repair and repair.get("repaired") else {}

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
                    focus=None if focus is None else focus.to_dict(), dt_prev=st.dt_prev,
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
                meta.update(self._taus(meta["time"]),
                            coarsen_blocked_until=st.coarsen_blocked_until,
                            tip_born=dict(st.tip_born),
                            apex_history={k: list(v) for k, v in st.apex_history.items()},
                            kinks={k: dict(v) for k, v in st.kinks.items()})
            if self.events is not None:  # only with the block: other dumps keep their keys
                meta.update(events_settings=self.events.to_dict(),
                            events=[dict(x) for x in self._made_events()],
                            legs=[dict(x) for x in st.legs],
                            # whether a step has run on the mesh since it was built (restore)
                            mesh_fresh=self._fresh_mesh())
        meta.update(extra)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")

    def keep_state(self, path: Path, **extra) -> Path:
        """A dump outside the rotation of ``state_keep`` (``path`` under ``states/``)."""
        self._write_state(Path(path), kept=True, **extra)
        return Path(path)

    def _state_interval(self) -> float:
        """The time from the last rotating dump to the next: ``state_dt``, and after a cut at most
        the time after the cut of the last dump (none since the cut: zero), so that the dumps
        follow at least each time the time after the cut doubles. The ``state_dt`` of a landing
        run, sized for the collapse before it, can exceed the whole run after a cut."""
        if self._cut_record() is None:
            return self.s.state_dt
        return min(self.s.state_dt, max(0.0, self.state.last_state_t - self._cut_times()[0]))

    def _save_end_state(self) -> Path:
        """After a cut: the rotating dump of the state the run ends on (as any dump, within its step
        if it ends there), unless that step has just been dumped. A continuation starts from it. It
        is the latest dump, which the rotation keeps (``state_keep`` is at least one)."""
        st = self.state
        path = self.outdir / "states" / f"state_{st.step:08d}.dump"
        if not st.states or st.states[-1] != path:
            self.save_state()
        if not (path.is_file() and path.with_suffix(".json").is_file()):
            raise RuntimeError(f"the dump of the end state {path} was not kept")
        return path

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
        if meta.get("cut") is not None:
            self._check_events_meta(meta)
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
        self._stepped_rc = self.pb.remesh_count  # the mesh of the dump has been stepped on
        if self.events is not None and meta.get("cut") is not None and meta.get("mesh_fresh") \
                and not set(self.events.changed_keys(meta.get("events_settings"))) & set(
                    SIZING_KEYS):
            # with the block, a mesh no step has run on since its build is fresh again: the checks
            # of a fresh mesh (an unresolved liquid neck, the guard's limits there, the curved
            # elements) apply to it as they did in the run that dumped it, unless the restart
            # changes the sizes of the mesh, which its rule of rebuilds then meets
            self._stepped_rc = None
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
            if not self.reconnect.continues_cut(meta["cut"].get("settings")):
                raise ValueError("the reconnect settings differ from those of the cut that wrote "
                                 "the state (a continuation may change only "
                                 f"{', '.join(CONTINUATION_KEYS)})")
            from .surgery import restore_cut

            restore_cut(self.pb, meta["cut"])
            st.coarsen_blocked_until = int(meta.get("coarsen_blocked_until", -10**9))
            # dumps written before the jets of the bubble's tips were tracked: none stands yet
            st.tip_born = {str(k): bool(v) for k, v in (meta.get("tip_born") or {}).items()}
            st.apex_history = {str(k): [float(x) for x in v]
                               for k, v in (meta.get("apex_history") or {}).items()}
            st.kinks = {str(k): dict(v) for k, v in (meta.get("kinks") or {}).items()}
            if self.events is not None:
                self.pb.events = [dict(x) for x in meta.get("events") or []]
                st.legs = [dict(x) for x in meta.get("legs") or []]
        ep = self.episodes
        resume = {"resume_at": RESUME_REMESH_CHECK} if self._checks_pending else {}
        m = self.measure()
        self._event("restart", m, from_time=meta["time"],
                    episode_history="episodes" in meta, episode=ep.index, episode_kind=ep.kind,
                    episode_start=ep.start, episode_partial=ep.partial, h_crest=ep.h_crest,
                    t_crest=ep.t_crest, crest_jet=ep.crest_jet, max_curvature_reset=reset,
                    coarsen_held_until=st.coarsen_held_until, **resume)
        if self.events is not None and meta.get("cut") is not None:
            self._start_leg(m, "restart", meta)

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
        st.refine_blocked_until = st.coarsen_blocked_until = -10**9
        st.tip_born = {}
        st.apex_history, st.kinks = {}, {}
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
        self._track_tips(m, snapshot=False)  # a jet already inside the bubble, say
        if self.events is not None:
            self.pb.events, st.legs = [], []
            self._start_leg(m, "cut", None)
        self._snapshot(m, "cut")
        return m

    # --- after a cut: the events block ---------------------------------------------------------

    def _made_events(self) -> list:
        """The events made after the cut (their records, ``events.EVENT_CLASSES`` kinds)."""
        return list(getattr(self.pb, "events", None) or [])

    def _check_events_meta(self, meta: dict):
        """Refuse to continue a state written after a cut under an events block it cannot take
        (``run_case._check_events``, which checks it before any output): a state written with the
        block under a driver without it, or a change of the keys of a class of events of which the
        state has made one, but from the state before an event (``precut_e<k>``)."""
        recorded = meta.get("events_settings")
        if self.events is None:
            if recorded is not None:
                raise RuntimeError("a state written with the events block after its cut is "
                                   "continued only by a driver with that block")
            return
        recut_from = ((meta.get("event") or {}).get("k")
                      if meta.get("cut_stage") == PRECUT_EVENT else None)
        refused = self.events.refused_changes(recorded, events_done(meta.get("events")),
                                              recut_from)
        if refused:
            raise ValueError("the events block changes the keys of a class of events of which the "
                             "state has made one: " + ", ".join(refused))

    def _start_leg(self, m: dict, start: str, meta: dict | None):
        """After a cut with the events block: a new leg of the run, at the cut (``start`` ``cut``)
        or at a restart from ``meta``, recorded in every later dump (``legs``) and as event
        ``leg``: when and from which step it starts, the source it runs (``leg_info``: the commit),
        the hash of the block and the keys in which it differs from the block the state was written
        with (``changed``; for a state written without the block, ``added``, from the block's
        defaults), oomph-lib's smallest step it sets (``minimum_dt``, :meth:`_lower_minimum_dt`;
        ``minimum_dt_ulps``, that step over the spacing of doubles at the leg's time), the limits
        and floors it runs with (``limits``, :meth:`_leg_limits`), and, from the state before an
        event, that event (``from_event_state``) and, where keys of a class of events change, the
        event from which they apply anew (``event_keys_changed_at``)."""
        st = self.state
        minimum_dt = self._lower_minimum_dt()
        leg = {"leg": len(st.legs), "start": start, "step": st.step, "time": m["time"],
               **self._taus(m["time"]), **self.leg_info, "settings_sha256": self.events.sha256(),
               "changed": [], "added": False, "minimum_dt": minimum_dt,
               # oomph-lib's smallest step in units of the spacing of doubles at the leg's time:
               # a step of a few of them no longer advances the time by itself
               "minimum_dt_ulps": _in_ulps(minimum_dt, m["time"]), "limits": self._leg_limits()}
        if meta is not None:
            recorded = meta.get("events_settings")
            leg["added"] = recorded is None
            leg["changed"] = self.events.changed_keys(recorded)  # without a record: the defaults
            if meta.get("cut_stage") == PRECUT_EVENT:
                k = (meta.get("event") or {}).get("k")
                leg["from_event_state"] = k
                if any(c.split(".", 1)[0] in EventSettings.CLASSES for c in leg["changed"]):
                    leg["event_keys_changed_at"] = k
        st.legs.append(leg)
        self._event("leg", m, **{k: v for k, v in leg.items() if k not in ("time", "step")})

    # --- after a cut: the events the run makes ------------------------------------------------

    def _cascade(self) -> CascadeMonitor:
        """The monitor of the cascades of the events the run has made, at each tip on the axis
        (``events.CascadeMonitor``, with ``T = t_p - t_cut``), from their records."""
        t_cut, t_p = self._cut_times()
        return CascadeMonitor.for_run(self.events, self.reconnect.delta, T=t_p - t_cut,
                                      records=self._made_events())

    def _meet_event(self, m: dict, request: dict) -> tuple[dict, dict | None]:
        """Meet an event the stops of the measured state ``m`` request (``events.
        post_cut_events``, with the block): ``(m, stop)`` for an event the run stops on, ``(m_new,
        None)`` for one made, from whose fresh mesh the run goes on (``m_new``, its measurement,
        which the caller checks for every stop again, as on any mesh freshly built).

        1. An event whose handler the driver does not make (``events.IMPLEMENTED_HANDLERS``) is a
           resumable stop (``event-unhandled``), and so is the event after ``max_events`` made
           (``max-events``) and one that the cascade test of its tip judges to shrink
           (``event-cascade``, ``events.CascadeMonitor``, with the volume of the head the request
           names), each with the dump of the state before it (``run``).
        2. The liquid cut (``surgery.liquid_cut``), with the run's last step as the history's,
           the elements per scale after the cut and the floor ``post.h_floor``: a refusal, which
           leaves the problem unchanged, stops the run resumably (``liquid-cut-refused``), and so
           do hard budgets that do not close (``event-unclosed``), whose problem holds a cut that
           is not accepted and is dumped no more; any other failure of the cut ends the run as a
           failed one, raised.
        3. The driver's bookkeeping of the cut made (:meth:`_on_event_made`), its row of
           ``events.txt``, and the commit of the jets' bases and the kink regions on the fresh
           mesh (:meth:`_commit_post_cut`)."""
        if request.get("handler") not in IMPLEMENTED_HANDLERS:
            return m, self._unhandled_event(request)
        if len(self._made_events()) >= self.events.max_events:
            return m, self._event_stop(EVENT_LIMIT, request, max_events=self.events.max_events)
        neck = event_neck(request)
        tip = event_tip(neck)
        verdict = self._cascade().check(tip, m["time"], float(neck.get("head_volume", math.nan)))
        if verdict["verdict"] == "shrink":
            return m, self._event_stop(EVENT_CASCADE, request, cascade=verdict)
        from .surgery import BudgetUnclosed, CutRefused, liquid_cut

        st, r = self.state, self.reconnect
        k = len(self._made_events()) + 1
        eps = self.s.elements_per_scale if r.elements_per_scale is None else r.elements_per_scale
        dt_prev = st.dt_prev if st.dt_prev else float(self._cut_record()["settings"]["dt0"])
        self._event_before, self._event_after = (m, tip, verdict), None
        try:
            made = liquid_cut(self.pb, self.events, r, self.outdir, request, k=k,
                              dt_prev=dt_prev, elements_per_scale=eps, h_floor=self._h_floor(),
                              driver=self, on_made=self._on_event_made)
        except (CutRefused, BudgetUnclosed) as e:
            rec = getattr(e, "record", None) or {}
            reason = LIQUID_CUT_REFUSED if isinstance(e, CutRefused) else EVENT_UNCLOSED
            extra = {"hard": (rec.get("closures") or {}).get("hard")} if reason == \
                EVENT_UNCLOSED else {}
            return m, self._event_stop(reason, request, precut=e.precut,
                                       failed_at=rec.get("failed_at"),
                                       error=rec.get("error", f"{type(e).__name__}: {e}"),
                                       record=f"events/event_{k:03d}.json", **extra)
        finally:
            self._event_before = None
        m = self._event_after
        self._write_event_row(made)
        if isinstance(made.get("ghost"), dict):
            self._write_drop_row(made, "free")
        self._commit_post_cut(m)
        return m, None

    def _on_event_made(self, rec: dict):
        """The driver's bookkeeping of a liquid cut made (``surgery.liquid_cut``'s ``on_made``,
        before the state after it is kept and its record written), into its record: what the
        driver measured of the state before it, the base of the jet whose neck it cut
        (``trigger``: ``r_j``, ``z_b``, ``r_j_over_r_n`` and ``neck_to_base_over_r_j``, the
        neck's distance from the base in jet radii), the event's place in the run (``generation``
        at its tip, ``step``, ``leg``, ``tau_over_delta2``, ``t_prime_over_T`` with ``T = t_p -
        t_cut``, the ``cascade`` verdict, the source ``commit``, ``dt.before``), the start of its
        ghost (``step0``, ``tip``, ``far_tip``, ``recontacts``); then :meth:`reset_after_event`,
        and the jet of the cut tip on the fresh mesh (``after``)."""
        m, tip, verdict = self._event_before
        st, delta = self.state, self.reconnect.delta
        jet = m if tip == "cavity_tip" else (m.get("tips") or {}).get(tip, {})
        r_j, z_b = float(jet.get("r_j", math.nan)), float(jet.get("z_b", math.nan))
        trig = rec.setdefault("trigger", {})
        r_n = float(trig.get("r_n") or math.nan)
        z_n = float(trig.get("z_n") if trig.get("z_n") is not None else math.nan)
        trig.update(r_j=_finite(r_j), z_b=_finite(z_b),
                    r_j_over_r_n=_finite(r_j / r_n) if r_n > 0.0 else None,
                    neck_to_base_over_r_j=_finite(abs(z_n - z_b) / r_j) if r_j > 0.0 else None)
        t_cut, t_p = self._cut_times()
        T, tau_p = t_p - t_cut, rec.get("tau_p")
        rec.update(generation=verdict["generation"], step=st.step, leg=len(st.legs) - 1,
                   tau_over_delta2=_finite(float(rec["tau"]) / delta**2),
                   t_prime_over_T=(_finite(float(tau_p) / T) if tau_p is not None
                                   and math.isfinite(T) and T != 0.0 else None),
                   cascade={key: _finite(v) if isinstance(v, float) else v
                            for key, v in verdict.items()},
                   commit=self.leg_info.get("commit"))
        rec.setdefault("dt", {})["before"] = _finite(st.dt_prev)
        if isinstance(rec.get("ghost"), dict):
            rec["ghost"].update(step0=st.step, tip=tip, far_tip=FAR_TIPS.get(tip),
                                recontacts=[])
        m_new = self.reset_after_event(rec)
        after = m_new if tip == "cavity_tip" else (m_new.get("tips") or {}).get(tip, {})
        rec["after"] = {key: _finite(after.get(key, math.nan))
                        for key in ("r_j", "z_b", "Q_j", "V_j")}
        rec["after"]["jet_born"] = bool(after.get("jet_born", False))
        self._event_after = m_new

    def reset_after_event(self, record: dict) -> dict:
        """Restart the bookkeeping that the rebuild of an event invalidates (a liquid cut,
        ``record`` its record) and measure the fresh mesh. Unlike :meth:`reset_after_cut` it starts
        no new cycle: the cycle, the births of the jets (of the cavity's tip and the bubble's), the
        singular-time series, the neck samples, the last step (the history's step of the cut,
        from which the step limits follow) and the holds of the rebuilds are kept. The growth
        limit of the local scale and the rate limit of the step restart (``ell_prev``,
        ``t_prev``), and the radius of the cut tip's jet base (``apex_history``) and the kink
        region of its chain are dropped; the fresh mesh is measured, the event recorded (event
        ``liquid-cut``), the jets of the bubble's tips tracked and a profile written (reason
        ``liquid-cut``). The remesh check of the step is not made on the fresh mesh, from which
        the run steps as from the cut's."""
        st = self.state
        st.ell_prev = st.t_prev = math.nan
        st.apex_history.pop(record.get("tip"), None)
        st.kinks.pop(record.get("chain"), None)
        self._checks_pending = False
        m = self.measure()
        k, removed = record.get("k"), record.get("removed") or {}
        trig = record.get("trigger") or {}
        self._event("liquid-cut", m, k=k, kind=record.get("kind"),
                    **{"class": record.get("class")}, chain=record.get("chain"),
                    tip=record.get("tip"), generation=record.get("generation"),
                    r_n=trig.get("r_n"), s=trig.get("s"), r_j_before=trig.get("r_j"),
                    V_rem=removed.get("V"), Pz_rem=removed.get("Pz"), KE_rem=removed.get("KE"),
                    cascade=(record.get("cascade") or {}).get("verdict"),
                    record=f"events/event_{int(k):03d}.json")
        self._track_tips(m, snapshot=False)
        self._snapshot(m, "liquid-cut")
        return m

    def _write_event_row(self, rec: dict):
        """The row of ``events.txt`` of an event made (its record)."""
        trig, cap = rec.get("trigger") or {}, rec.get("cap") or {}
        removed, ghost = rec.get("removed") or {}, rec.get("ghost") or {}
        closures = rec.get("closures") or {}
        rel, phi = closures.get("relative") or {}, closures.get("F3_phi") or {}
        tr, casc, total = rec.get("transfer") or {}, rec.get("cascade") or {}, \
            rec.get("removed_total") or {}
        values = {"k": rec.get("k"), "step": rec.get("step"), "time": rec.get("t"),
                  "tau": rec.get("tau"), "tau_p": rec.get("tau_p"),
                  "tau_over_delta2": rec.get("tau_over_delta2"), "kind": rec.get("kind"),
                  "class": rec.get("class"), "chain": rec.get("chain"), "tip": rec.get("tip"),
                  "generation": rec.get("generation"),
                  **{key: trig.get(key) for key in ("r_n", "s", "r_n_over_s", "z_n", "r_j", "z_b",
                                                    "r_j_over_r_n", "neck_to_base_over_r_j",
                                                    "Re_n", "We_n")},
                  "z_tip_old": cap.get("z_tip_old"), "z_tip_new": cap.get("z_tip_new"),
                  "eta": cap.get("eta"), "V_rem": removed.get("V"), "Pz_rem": removed.get("Pz"),
                  "KE_rem": removed.get("KE"), "U_ghost": ghost.get("U"),
                  "du_stump": ghost.get("du_stump"),
                  "r_j_after": (rec.get("after") or {}).get("r_j"),
                  **{f: rel.get(f) for f in ("F1", "F2", "F3", "F4", "F5", "F6")},
                  "phi_relative_error": phi.get("phi_relative_error"),
                  "predicted_unlocated": tr.get("predicted_unlocated"),
                  "reported_unlocated": tr.get("reported_unlocated"),
                  "dt_history": (rec.get("dt") or {}).get("history"),
                  "interval_over_delta2": casc.get("interval_over_delta2"),
                  "events_per_T": casc.get("events_per_T"), "cascade": casc.get("verdict"),
                  "V_removed_total": total.get("V"), "Pz_removed_total": total.get("Pz"),
                  "KE_removed_total": total.get("KE"), "wall": rec.get("wall_seconds")}
        self._event_rows.write("\t".join(_table_cell(values.get(c)) for c in EVENT_COLUMNS)
                               + "\n")

    def _follow_ghosts(self, m: dict):
        """After an accepted step, with the events block: the ghost of each head the run removed
        (its record's ``ghost``, ``events.ghost_place``), against the tips on the axis of the state
        ``m``, whose recontacts (``events.ghost_recontacts``: the stump catching the ghost's rear,
        its front reaching the far tip of the bubble) are flags only: each is added to the ghost
        (``recontacts``, ``status`` ``recontact``) in the record the dumps keep and in the record's
        file, where this run wrote it, as an event ``ghost-recontact`` and a row of ``drops.txt``.
        Nothing stops: the bubble matters only for the entrainment, its collapse and the
        budgets."""
        if self.events is None or self._cut_record() is None:
            return
        rec = self._post_cut_state()["record"]
        if not rec:
            return
        tips = {"cavity_tip": rec["z_cavity_tip"], "bubble_bottom": rec["z_bubble_bottom"],
                "bubble_top": rec["z_bubble_top"]}
        for e in self._made_events():
            g = e.get("ghost")
            if not isinstance(g, dict) or "step0" not in g:
                continue
            hits = ghost_recontacts(g, m["time"], self.state.step, tips)
            for hit in hits:
                hit = {key: _finite(v) if isinstance(v, float) else v
                       for key, v in {**hit, **self._taus(hit["t"])}.items()}
                g.setdefault("recontacts", []).append(hit)
                g["status"] = "recontact"
                self._event("ghost-recontact", m, k=e.get("k"), recontact=hit["kind"],
                            tip=hit["tip"], z_ghost=hit["z_ghost"], z_tip=hit["z_tip"])
                self._write_drop_row(e, hit["kind"], hit)
            if hits:
                self._rewrite_event_file(e)

    def _write_drop_row(self, rec: dict, status: str, hit: dict | None = None):
        """The row of ``drops.txt`` of a change of an event's ghost: made (``free``), or a
        recontact ``hit``."""
        g = rec["ghost"]
        t = float(g["t0"]) if hit is None else float(hit["t"])
        front, rear = ghost_place(g, t)
        values = {"k": rec.get("k"), "status": status,
                  "step": rec.get("step") if hit is None else hit["step"], "time": t,
                  **self._taus(t), "tip": g.get("tip"), "far_tip": g.get("far_tip"),
                  "z_front": front, "z_rear": rear,
                  "z_tip": None if hit is None else hit["z_tip"], "U": g.get("U"),
                  "v_ret": g.get("v_ret"), "V": g.get("V")}
        self._drops.write("\t".join(_table_cell(values.get(c)) for c in DROP_COLUMNS) + "\n")

    def _rewrite_event_file(self, rec: dict):
        """Put an event's ghost, as the run follows it, into its record's file
        (``events/event_<k>.json``), where this run wrote it, the polygon the file alone holds
        kept."""
        path = self.outdir / "events" / f"event_{int(rec['k']):03d}.json"
        if not path.is_file():  # made in an earlier leg, in its own run directory
            return
        full = json.loads(path.read_text())
        full["ghost"] = {**rec["ghost"], **({"polygon": full["ghost"]["polygon"]}
                                            if "polygon" in (full.get("ghost") or {}) else {})}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(full, indent=1) + "\n")
        tmp.replace(path)

    def _events_summary(self, stop: dict) -> dict:
        """After a cut with the block: the events of the run's result (``run_status.json``): how
        many were made, by kind and by tip, the last, the totals of the removals since the gas
        cut, the cascade's verdict at each tip, and the event the run stopped on, if any."""
        made = self._made_events()

        def count(key):
            out: dict = {}
            for e in made:
                out[str(e.get(key))] = out.get(str(e.get(key)), 0) + 1
            return out

        liquid = [e for e in made if e.get("kind") in EVENT_CLASSES["liquid"]]
        last = made[-1] if made else None
        return {"made": len(made), "max_events": self.events.max_events,
                "by_kind": count("kind"), "by_tip": count("tip"),
                "last": None if last is None else {key: last.get(key) for key in (
                    "k", "kind", "class", "chain", "tip", "t", "tau", "tau_p")},
                "removed_total": {key: sum(float((e.get("removed") or {}).get(key) or 0.0)
                                           for e in liquid) for key in ("V", "Pz", "KE")},
                "cascade": {tip: v["verdict"] for tip, v in self._cascade().verdicts().items()},
                "ghost_recontacts": {kind: sum(
                    1 for e in made for x in (e.get("ghost") or {}).get("recontacts") or ()
                    if x.get("kind") == kind) for kind in ("rear", "front")},
                "stop": ({key: stop.get(key) for key in ("reason", "k", "kind", "class",
                                                         "handler")}
                         if stop.get("reason") in EVENT_STOPS else None)}

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
        neck_limit = self._neck_limit()
        post = self._post()
        # with the block, after the cut, the open chain's neck stop may be off
        # (post.open_neck_stop): its liquid neck is then an event of the stops after the cut
        if (post is None or post.open_neck_stop) and math.isfinite(m["r_neck"]) \
                and m["r_neck"] < neck_limit:
            return {"reason": "pinch-off-approach", "trigger": "neck-radius",
                    "r_neck": m["r_neck"], "z_neck": m["z_neck"], "neck_limit": neck_limit}
        # after a cut the fold stop is the post-cut one, on both chains, which names a fold across
        # the groove at the base of a jet (annular-contact-approach; reconnect.post_cut_stops)
        if self._cut_record() is None and m.get("fold_gap", math.inf) < m.get("fold_limit",
                                                                               math.nan):
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
        if r is not None:
            self._arm_post_cut_nets()
        while True:
            found = self._stop(m)
            if found is not None and found["reason"] == EVENT_REQUEST:  # the events block only
                m, found = self._meet_event(m, found)
                if found is None:  # made: every stop again on the fresh mesh
                    continue
            if found is not None:
                stop = found
                break
            t = self._t()
            if s.t_end - t <= max(1e-14, 8 * math.ulp(s.t_end)):
                break
            if r is not None and tau_end_reached(t - t_cut, r.tau_end):
                stop = {"reason": "tau-end", "tau_end": r.tau_end, **self._taus(t)}
                break
            if st.step >= s.max_steps:
                stop = {"reason": "max-steps"}
                break
            # after a cut the rule of the focus applies only with reconnect.focus_courant
            focus_courant = r is None or r.focus_courant
            dt = min(s.dt_max, s.courant * m["h_focus"] / m["U"] if focus_courant else math.inf,
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
            if dt < self._dt_min():
                stop = {"reason": "dt-floor", "dt": dt}
                break
            st.ell_prev, st.t_prev = m["ell"], t
            wall0 = time.time()
            # temporal_error engages pyoomph's adaptive stepping: without an error estimator every
            # converged step is accepted, while a failed Newton solve is rejected and halved
            tol = max(newton_tolerance, s.newton_floor / dt)
            defer = self._defers_distortion()
            rebuilds = pb.remesh_count
            try:
                with self._post_cut_solve():
                    pb.run(t + dt, timestep=dt, maxstep=dt, outstep=False, temporal_error=1.0,
                           newton_solver_tolerance=tol)
            except RuntimeError as exc:
                # the problem now holds a rejected iterate: stop on the last measured state ``m``,
                # but only if oomph-lib gave up below its smallest step
                stop = None
                if not defer:  # sub-steps accepted before the failure may have been rebuilt
                    self._pyoomph_remeshes += pb.remesh_count - rebuilds
                flagged = self._distortion_flags()
                # after a cut no Newton failure is an expected end
                if r is None and type(exc) is RuntimeError and exc.args == ("OomphException",):
                    error = oomph_error_message(pb.oomph_log_tail())
                    if MIN_DT_FAILURE in error:
                        stop = classify_failure(m, s, error)
                if stop is None:
                    if r is not None:  # the state the failing step started from, for forensics
                        self._write_lastgood(m, exc)
                    raise
                stop["dt_attempted"] = dt
                if defer and flagged:
                    # sub-steps accepted before the failure asked for a rebuild that the driver
                    # makes only after the step: the failure came on the mesh they flagged
                    stop["distortion_flagged_substeps"] = len(flagged)
                break
            st.step += 1
            dt = self._t() - t
            st.dt_prev = dt
            inside = pb.remesh_count - rebuilds  # rebuilds pyoomph made inside the step
            self._stepped_rc = pb.remesh_count
            flags = self._distortion_flags()
            if not defer:  # pyoomph rebuilt inside the solve wherever distortion asked
                self._pyoomph_remeshes += inside
            elif inside:  # after a cut: for an element turned inside out (_arm_post_cut_nets)
                self._inversion_remeshes += inside
            if r is not None:  # an element below the cut's absolute quality floor
                flags += self._quality_floor_flags()
            # with a focus, the driver's rebuild, before the step is measured, as pyoomph's was
            m = self._distortion_remesh(flags) if defer and flags else self.measure()
            if defer and inside and r is not None:
                self._event("remesh", m, reason="inverted-element", rebuilds=inside)
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
            self._track_tips(m)  # after a cut, the jets of the bubble's tips
            self._follow_ghosts(m)  # with the events block, the ghosts of the removed heads
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
            if found is not None and found["reason"] == EVENT_REQUEST:  # the events block only
                m, found = self._meet_event(m, found)
                if found is None:  # made: its rebuild stands for the one a stop cancelled
                    self._distortion_skipped = 0
                    continue
            if found is not None:
                if self._distortion_skipped:  # the rebuild of this step, cancelled by this stop
                    found["distortion_rebuild_skipped"] = self._distortion_skipped
                stop = found
                break
            if r is not None and self._edge_turn_max() > r.edge_turn_refine:
                self._snapshot(m, "edge-turn")  # every step while an element bends this much
            else:
                self._maybe_snapshot(m)
            m, found = self._remesh_checks(m)
            if found is not None and found["reason"] == EVENT_REQUEST:  # the events block only
                m, found = self._meet_event(m, found)
                if found is None:
                    continue
            if found is not None:
                stop = found
                break
            self._commit_post_cut(m)
            if self._t() - st.last_state_t >= self._state_interval():
                self.save_state()
        # with the events block, a cut whose hard budgets do not close leaves the problem holding
        # a cut that is not accepted: nothing more is read from it, and the run ends on the state
        # before the event, which the cut kept
        unclosed = stop.get("reason") == EVENT_UNCLOSED
        if r is not None:  # after a cut the state the run ends on is a dump too
            if not unclosed:
                self._commit_post_cut(m)
            if stop.get("reason") in EVENT_STOPS:  # with the events block only
                if not stop.get("precut"):  # unless the cut of the event kept it
                    stop["precut"] = str(self._keep_event_state(stop))
                self._event(stop["reason"], m, k=stop["k"], kind=stop.get("kind"),
                            **{"class": stop.get("class")}, handler=stop.get("handler"),
                            precut=stop["precut"])
            if stop.get("window_end"):  # with the events block only
                self._event("window-end", m, reason=stop["reason"])
            stop["end_state"] = stop["precut"] if unclosed else str(self._save_end_state())
        if self.keep_stop_state and stop.get("trigger") == "neck-radius":
            # the measured state of an accepted step: a landing state for a reconnection
            stop["kept_state"] = str(self.keep_state(
                self.outdir / "states" / f"stop_{st.step:08d}.dump", stop=stop,
                pinch=self.pinch_time()))
        self._landings_left(m, stop)
        self._event("stop", m, **stop)
        if not unclosed:
            self._snapshot(m, "stop")
        out = {"stop": stop, "steps": st.step, "remesh_count": pb.remesh_count,
               "jet_born": st.born, "max_axis_curvature_before_birth": st.max_curvature,
               "max_axis_curvature_time": st.max_curvature_time, "snapshots": st.snapshots,
               **self.distortion_counts()}
        if self.landing:
            out["landing"] = self._landing_summary()
        if r is not None:
            t_cut, t_p = self._cut_times()
            out["cut"] = {"t_cut": t_cut, "t_p": t_p, "tau_end": r.tau_end, **self._taus(self._t()),
                          "inverted_element_remesh": self._nets_armed,
                          "inverted_element_remeshes": self._inversion_remeshes,
                          "axis_midside_repairs": getattr(pb, "axis_repairs_total", 0)}
            if self.events is not None:  # only with the block: other results keep their keys
                out["events"] = self._events_summary(stop)
                out["window_end"] = ({"reason": stop["reason"], **self._taus(m["time"])}
                                     if stop.get("window_end") else None)
        return out

    @contextlib.contextmanager
    def _post_cut_solve(self):
        """Around each step after a cut: at least ``reconnect.newton_min_iterations`` Newton
        iterations per solve (``FaradayProblem.min_newton_iterations``), and the snapshot that
        pyoomph's net for an inverted element takes before the step (``_snapshot_state``, through
        ``save_state`` into memory) uncompressed: zlib took about 8% of a step, and the snapshot
        is lossless either way. The dumps the driver writes keep their compression."""
        pb, r = self.pb, self.reconnect
        if r is None or not hasattr(pb, "release_newton_tolerance"):  # a FaradayProblem only
            yield
            return
        level = pb.states_compression_level
        pb.min_newton_iterations = r.newton_min_iterations
        pb.states_compression_level = None
        try:
            yield
        finally:
            pb.min_newton_iterations = 1
            pb.release_newton_tolerance()
            pb.states_compression_level = level

    # --- after a cut: the nets and the forensics ----------------------------------------------

    def _arm_post_cut_nets(self):
        """After a cut, before the first step: arm pyoomph's net for an element that turns inside
        out within a step (``reconnect.inverted_element_remesh``;
        ``FaradayProblem.arm_inverted_element_remesh``), which restores the start of the step,
        rebuilds and retries it (counted as ``inverted_element_remeshes``, event ``remesh`` with
        ``reason`` ``inverted-element``). The rebuilds of the driver are checked as every rebuild
        after the cut is; this one is pyoomph's, a last resort."""
        if self._nets_armed or not self.reconnect.inverted_element_remesh:
            return
        arm = getattr(self.pb, "arm_inverted_element_remesh", None)
        self._nets_armed = bool(arm()) if arm is not None else False

    def _quality_floor_flags(self) -> list[dict]:
        """After a step after a cut: a distortion reading with trigger ``absolute-quality`` when an
        element's quality has fallen below the cut's ``quality_floor``
        (``FaradayProblem.quality_floor_reading``), so that the driver rebuilds; nothing from a
        problem that cannot read it."""
        read = getattr(self.pb, "quality_floor_reading", None)
        if read is None:
            return []
        reading = read(self.reconnect.quality_floor)
        return [] if reading is None else [reading]

    def _edge_turn_max(self) -> float:
        """The largest turn of the tangent across a quadratic element of either chain (degrees)."""
        turns = [rd["turn"] for rd in self._post_cut_state()["edges"].values()
                 if math.isfinite(rd["turn"])]
        return max(turns, default=0.0)

    def _write_lastgood(self, m: dict, exc: BaseException):
        """After a failed step after a cut: the profiles of the last state the driver measured (the
        start of the failing step), from what it kept of them, as ``profile_<step>_lastgood.txt``
        and ``profile_<step>_lastgood_bubble.txt``: the problem itself holds the rejected
        iterate."""
        st = self.state
        r, z, ur, uz = m["profile"]
        header = {"time": m["time"], "step": st.step, "reason": "lastgood",
                  "remesh_count": self.pb.remesh_count, "error": f"{type(exc).__name__}: {exc}",
                  **self._taus(m["time"]), "chain": "open"}
        path = self.outdir / "profiles" / f"profile_{st.step:06d}_lastgood.txt"
        np.savetxt(path, np.column_stack([r, z, ur, uz]), header=json.dumps(header) +
                   "\nr z u_r u_z", comments="# ")
        cached = self._post_cut_cache[1] if self._post_cut_cache is not None else None
        bubble = next((c for c in cached["segs"] if not c["open"]), None) if cached else None
        if bubble is not None:
            np.savetxt(path.with_name(path.stem + "_bubble.txt"),
                       np.column_stack([bubble[k] for k in ("r", "z", "u_r", "u_z")]),
                       header=json.dumps(dict(header, chain="bubble")) + "\nr z u_r u_z",
                       comments="# ")
