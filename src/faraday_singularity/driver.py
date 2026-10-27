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
round-off floor that grows like ``1/dt``, so a fixed tolerance cannot be met at the smallest steps;
after a cut, ``dt`` is that of each sub-step pyoomph's halving makes within the driver's step
(``FaradayProblem.newton_floor_rule``), whose floor a tolerance fixed at the driver's step stays below.
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
  (``profile_<step>_lastgood.txt`` and its ``_bubble`` beside it), unless pyoomph's net for an
  inverted element took its snapshot before the step: then the problem is rolled back to that
  state and the run ends on it, the resumable stop ``step-failed`` with its end state
  (:meth:`CollapseDriver._failed_step_stop`); while an element of either chain turns by more than
  ``edge_turn_refine`` degrees a profile is written at every step; with jetcore's recentred far
  kinks, a guard stop on the bubble before the piercing at the tail of an end folded along the
  axis is met by the axis repair (:meth:`CollapseDriver._axis_repair`, ``axisrepair``);
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
resumably, with the dump of the state before the event (``precut_e<k>``) and the record of the
event (``events/event_<k>.json``, :meth:`CollapseDriver._write_stop_record`), as every event made
has its own. ``events.txt`` holds a row per event made. The dumps of the events are kept as
``post.keep_event_dumps`` says, and with ``post.prune_gmsh`` the meshes no dump refers to are
deleted after every dump and every event (:meth:`CollapseDriver._prune_meshes`).

With the block's ``ring.enabled`` the piercing of the bubble by its downward jet (class ``W1``) is
met by the ring cut (``ringsurgery.ring_cut``, :meth:`CollapseDriver._meet_ring`) and the run goes
on in the ring stage (``FaradayProblem.ring``): the open chain read as before, the ring apart
(``ring.ring_stage_events``, :meth:`CollapseDriver._ring_stage_state`), its regions frozen at the
cut (:meth:`CollapseDriver._frozen_ring_regions`), ``ring.txt`` a row per step, the profiles the
ring's beside the open chain's, the totals of ``budget.txt`` net of the fills, until the ring's
removal below ``bubble_stop`` of the bubble's volume at the gas cut (class ``W2``,
``ringsurgery.ring_removal``), after which the window ends (``ring.after_removal`` ``stop``) or
the run goes on with the open chain alone; a liquid cut in the ring stage, and every other event of
the ring, is a resumable stop (:meth:`CollapseDriver.reset_after_ring` restarts the bookkeeping a
ring surgery invalidates).

Dumps written after a cut carry its record, which :meth:`CollapseDriver.restore` restores
(``surgery.restore_cut``), only into a driver whose reconnect settings are those of the cut but for
``tau_end`` (``reconnect.CONTINUATION_KEYS``), which a continuation may raise; the run then
continues in the post-cut mode.

With the block's ``jetcore.enabled`` (``events.JetCoreSettings``, ``jetcore.py``) the open chain's
nodes about the base of the upward jet are its core (``jetcore.core_mask``: within ``max(core_radius
L, apex_extent R_apex)`` of the base, ``L = max(r_j, delta)``, not above the jet's tracked neck, and
the walls beyond the base within ``wall_radius L``; the whole open chain without a base), every
other node of the free surface the far region, and the Courant limit of the step is split between
them (``jetcore.courant_split``: ``reconnect.courant`` on the core, ``jetcore.far_courant`` on the
far nodes, the same number by default, which leaves the step as it was); the far region's sizing
follows the part's keys (:meth:`CollapseDriver._feature_regions`: each region's class,
``jetcore.region_class``; the far targets floored at ``far_floor``; with ``far_kinks``
``recentre`` the guard's far kinks in regions of their own, ``jetcore.far_kink_regions``, each
yield below the floor an event ``far-yield``; the far refine ratio; the ring's kinks at the
column's size; the profiles of an element that bends from the core's elements only, with
``edge_turn_snapshots`` ``core``; with ``profile_dt`` a profile at least that often, reason
``window``); before the piercing, with ``prep_gap``, the bubble's bottom and
the downward jet's tip are refined (``jetcore.prep_regions``, ``pierce_bottom``, ``pierce_tip``)
so that the ring cut admits the splice radius ``prep_rho s_t``, their nodes stepped at the core's
Courant number; ``jetcore.txt`` holds a row per step (``jetcore.COLUMNS``: what
set the step, the split limit, the core, the far regions) and every leg the resolved keys
(``limits.jetcore``). With ``core_elements_per_scale`` above the run's elements per scale every
rule's size on the core's nodes is ``q`` times what it asks for (``jetcore.core_ratio``): each
region whose ball holds core nodes has its twin on them (``jetcore.core_twins``, ``core_<name>``,
its spacing and that of the region it splits read apart), and the guard reads the core's turns at
``q`` times its limit (``jetcore.core_turns``). In the ring stage with ``far_kinks`` ``recentre``,
the guard's stop on the ring (an element turning beyond ``edge_turn_stop_fresh`` on a freshly built
mesh or beyond ``edge_turn_stop`` on one a step has run on, up to ``RING_REFINE_TURN_MAX``, or a
midside node beyond ``midside_stop``, at an element above the guard's floor) is met by rebuilding at
once (:meth:`CollapseDriver._ring_refine`, at most ``RING_REFINE_TRIES`` times per state): every
region at its size and the element in a far kink of its own (``kink_ring_r<n>_far``, ``refined``;
the one whose reach holds it refined further) at the size its turn asks for (``min(1/2,
edge_turn_refine/turn)`` of its chord, at least ``h_floor``), which follows its feature keeping that
size until the feature asks no more, and while such a region remains every far kink of the ring
follows its own feature, a new one a region of its own (``kink_ring_m<n>_far``;
``jetcore.far_kink_regions``), a yield (``far_yields``); an event ``ring-refine`` each, with the
base of the upward jet before and after (``base_guard``: beyond ``HEAD_FOLD_BASE_TOL`` the run
stops, ``interface-unresolved``, trigger ``ring-refine-base``); an element of the ring at the
guard's floor that still asks is carried there, within the limits of the turn and the midside node
(``ring.ring_floored_ok``), not a stop. A guard stop on the ring that the refine does not meet, at
an element at the floor (an edge turn or a midside node beyond the limits, a feature below the
floor), where that element's apex is a wedge of gas read at the scale of the floor
(``ring.floor_wedge``), is met by the fill of the wedge (:meth:`CollapseDriver._floor_fill_request`,
``groovesurgery``'s fill at the floor: ``RF``, ``contact`` ``floor``, the floor's fold limits, the
fill's hard closures and the base of the upward jet, ``G_base``), an event ``floor-fill`` and
``floor_fills`` in the run's events; the tip of a finger of liquid stays the stop. A rebuilt mesh's
curved element whose Jacobian bound falls below ``jacobian_min`` at an element of the ring
(``curved-jacobian``) is read as the guard's reading of the ring's element there
(:meth:`CollapseDriver._ring_jacobian_reading`) and met as those stops are: by the refine above the
floor, by the fill of a wedge at it. A run whose ring never meets any of these is unchanged.

With the block's ``groove.enabled`` (``events.GrooveEventSettings``) a closing gas groove (class
``A-groove``, of the open chain or the bubble; ``RF``, a fold of the ring) is met by its fill
(``groovesurgery.groove_fill``, :meth:`CollapseDriver._meet_groove`): the gas beyond the groove's
mouth becomes liquid, the run goes on from the fresh mesh (:meth:`CollapseDriver.
reset_after_groove`), the fill a negative removal in ``events.txt`` and in the totals of
``budget.txt``; a fill refused (a pocket of liquid, a sheet, among the refusals; one on the base or
the core of the upward jet; one beyond ``groove.max_fill V_b``) is a resumable stop
(``groove-fill-refused``). A groove on the head of a tracked neck in the far region (the
downward jet's head, any head off the upward jet's core) is never filled: with ``pinch.far_head``
the far-head cut of that head comes first (decision 22), the fill only off the far heads. With
``groove.sheets`` ``log`` a fold whose
pocket is liquid, a thin sheet, is no event (``groove.side_folds``) but a row of ``sheets.txt``
at most once per rebuild (:meth:`CollapseDriver._log_sheets`), and its faces closing to the scale
of the mesh a resumable stop (``sheet-contact``, never the fill).

With the block's ``pinch.far_head`` an event without a handler (a gas neck, a closing groove)
whose site lies on the head of a tracked neck in the far region is met by the liquid cut of that
neck (``events.far_head_request``, kind ``far-head``, class ``F-head``, an event ``far-head``;
its budgets the cut's hard closures). A closing groove from the upward jet's head onto the core (the
cavity's wall or the stem beside the base; ``head_fold`` in the request) is the cut of that head
too, its nodes but the neck off the core, and the cut is checked against the jet's base: read
before and after it (:meth:`CollapseDriver._head_fold_check`, ``base_guard`` in the record), a base
whose ``r_j``, ``V_j`` or ``Q_j`` moves by more than ``HEAD_FOLD_BASE_TOL`` relative, or ``z_b`` by
more than that times ``r_j``, is the hard closure ``F_base`` (``event-unclosed``). A tracked
liquid neck that a freshly built mesh does not resolve (the stop ``liquid-neck-unresolved``) whose
head is a far head (the downward jet's, any head off the upward jet's core) is met by the same cut
of that head at this neck, closing or not, and checked against the jet's base in the same way
(``events.unresolved_neck_request``, :meth:`CollapseDriver._unresolved_neck_request`,
``unresolved`` in the request); on a head of the core, or one not single, the stop stays. With
``jetcore.box_gate`` every ordinary rebuild may change
the liquid volume and axial momentum in the box of radius ``liquid.box_radius delta`` about the
core's base by at most ``box_gate_sigmas`` times ``liquid.sigma_V`` and ``liquid.sigma_Pz``
(:meth:`CollapseDriver._gate_before`, :meth:`CollapseDriver._gate_after`, ``box_gate`` in the
``remesh`` event); otherwise the run stops (``box-gate``) on the state before the rebuild, dumped
before it as ``states/gate_<step>.dump``, its ``end_state``, from which a continuation resumes.

With the block's ``pinch.enabled`` a tracked neck with a head lands only when it closes
(``events.closing_neck``): the driver keeps the history of each tracked neck, matched across the
accepted steps by its chain and height (``neck_history``, in every dump; restored with it,
:meth:`CollapseDriver._commit_neck_history`), from which the stops read the trigger.

After a cut the stops of a state are evaluated once (:meth:`CollapseDriver._post_cut_stop`, keyed
by the state, :meth:`CollapseDriver._stop_key`): the check at the loop's top of the state the step
has just checked gives the same answer without evaluating it again. With ``step_timing`` (the
runner's ``--step-timing``, no key of a case) ``step_timing.txt`` holds the wall time of the parts
of each step (:data:`STEP_TIMING_COLUMNS`). Neither changes a value written.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .collapse import (Profile, axis_curvature, cone_angles, cone_fit, gauss_disc_points,
                       jet_base, necks)
from .episodes import ETA_HYS, EpisodeTracker
from .events import (EVENT_CASCADE, EVENT_CLASSES, EVENT_LIMIT, EVENT_REQUEST, EVENT_SCHEMA,
                     EVENT_STOPS, EVENT_UNCLOSED, EVENT_UNHANDLED, FAR_TIPS, GROOVE_CLASSES,
                     GROOVE_FILL_REFUSED, IMPLEMENTED_HANDLERS, SHEET_CONTACT,
                     SHEET_RUPTURE_REFUSED,
                     LIQUID_CUT_REFUSED, PRECUT_EVENT, RING_CUT_REFUSED, RING_HANDLERS,
                     RING_REMOVAL_REFUSED, SIZING_KEYS, STOP_OUTCOMES, CascadeMonitor,
                     EventSettings, event_dump_policy, event_dumps_kept, event_neck, event_tip,
                     events_done, fold_stop_after_cut, ghost_place, ghost_recontacts,
                     liquid_necks, minimum_dt_after_cut)
from .mesh import Focus, FocusRegion
from .reconnect import (AXIS, BUBBLE_TIP_SIGNS, CONTINUATION_KEYS, CUT_SIZING_MARGIN, JET_TIPS,
                        LANDING_RATIO, MAX_LANDING_RATIO, Chain, CutGeometry, ReconnectSettings,
                        bubble_end, courant_limit, cut_size_field, edge_reading, feature_regions,
                        groove_gap, interface_guard, jet_base_apex, kink_regions, post_cut_record,
                        post_cut_stops, region_spacing, tau_end_reached, tip_jet)
from .selfcontact import InterfaceSelfIntersection, fold_gap
from .ringcontact import RingStepContact, RingStepFold, input_key
from .singular import singular_time

FOLD_RETRY_FRACTION = 0.25  # the retry of a step folded at a closing corner (batch B)
FOLD_RETRY_REACH = 8  # steps after its retry within which a crossing records it
CORNER_LOCATE = 8.0  # node spacings of a corner within which one of the step's inverted elements lies


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
# the cut of the upward jet's head folding onto the core (events.far_head_request's head_fold) must
# leave the jet's base as it was: r_j, V_j and Q_j within this relative change, z_b within this
# times r_j (CollapseDriver._head_fold_check; r1's cut of its curled head changed them by 2-3e-4)
HEAD_FOLD_BASE_TOL = 1e-3
PINCH_FIT_SPAN = 1.5  # the pinch time is fitted to the latest samples with r_n <= this * the last
# in the ring stage with jetcore's recentred far kinks, a mesh whose ring turns beyond the guard's
# limit (fresh or stepped, up to RING_REFINE_TURN_MAX) or whose midside node lies beyond its limit
# is rebuilt at once with that element in a far kink of its own at the size its turn asks for
# (kink_ring_r<n>_far, refined; every other region as it was), at most this many times per state
# (CollapseDriver._ring_refine); the base of the upward jet must stay within HEAD_FOLD_BASE_TOL
# across each such rebuild (base_guard)
RING_REFINE_TRIES = 3
# the largest turn of the ring's element the refine meets, on a stepped or a fresh mesh (beyond it
# the element folds back on itself, and the run stops as the guard stops it)
RING_REFINE_TURN_MAX = 150.0
# the ring's thinning liquid lip (decision 22, a nuisance; CollapseDriver._sheet_context,
# _ring_sheet_reading, _meet_sheet_rupture): resolved by SHEET_ELEMENTS elements across its
# thickness down to the floor whenever fewer than SHEET_REFINE_ELEMENTS lie across it (r1 and r1h
# before their contacts at tau/delta^2 25.35: the lip ran with eight across at 6e-6 and met its
# contact at one, the faces closing at about 90 over 2e-5; four keep it at least two across within a
# halving of its thickness, the next refine due at two), and ruptured once it is thinning at
# h_s = jetcore.sheet_rupture_floors h_floor or below (SHEET_RUPTURE_FLOORS without the key)
SHEET_ELEMENTS = 4.0
SHEET_REFINE_ELEMENTS = 2.0
SHEET_RUPTURE_FLOORS = 2.0
LIP_JUMP = 20.0  # a lip's section beyond this many thicknesses of the last is another feature
# a fresh mesh's curved element on the bubble's chain before the piercing (CollapseDriver.
# _tip_straighten): the surface's edges within this many times the distance of the element to the
# surface, or the chain's spacing there, are read for their midside nodes
TIP_STRAIGHTEN_REACH = 3.0
BASE_KEYS = ("r_j", "z_b", "V_j", "Q_j")


def base_guard(before: dict, after: dict, tol: float | None = None) -> dict:
    """The base of the upward jet (``BASE_KEYS`` of two measurements, ``before`` and ``after`` a
    change of the mesh): ``change`` of ``r_j``, ``V_j`` and ``Q_j`` relative to their value
    before, of ``z_b`` relative to ``r_j``; ``ok`` where each lies within ``tol``, by default
    ``HEAD_FOLD_BASE_TOL`` as the module holds it at the call (decision 22: the base, its walls and
    the stem stay as resolved)."""
    tol = HEAD_FOLD_BASE_TOL if tol is None else tol
    b = {k: float(before.get(k, math.nan)) for k in BASE_KEYS}
    a = {k: float(after.get(k, math.nan)) for k in BASE_KEYS}
    r_j = b["r_j"]
    change = {k: (a[k] - b[k]) / abs(b[k]) if b[k] else math.nan for k in ("r_j", "V_j", "Q_j")}
    change["z_b"] = (a["z_b"] - b["z_b"]) / r_j if r_j > 0.0 else math.nan
    ok = all(math.isfinite(v) and abs(v) <= tol for v in change.values())
    return {"before": b, "after": a, "change": change, "tol": tol, "ok": bool(ok)}


def base_guard_checked(before: dict, after: dict) -> dict:
    """:func:`base_guard` of a change of the far region's mesh (the ring's refine, the fill of a
    wedge at the floor): ``checked`` false, and ``ok``, where no jet stood before it (a reading not
    finite), whose base there is none to keep."""
    guard = base_guard(before, after)
    guard["checked"] = all(math.isfinite(v) for v in guard["before"].values())
    if not guard["checked"]:  # no born jet before it: no base to keep
        guard["ok"] = True
    return guard


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
# budget.txt, a row per step after the cut: the liquid of the mesh (its volume, axial momentum and
# kinetic energy, as chains.txt has them), what the events have removed of it since the gas cut,
# each removed head counted with its own at its cut, and the sums, the liquid the run would hold
# without the removals
BUDGET_COLUMNS = ("step", "time", "tau", "tau_p", "events", "volume", "momentum_z",
                  "kinetic_energy", "V_removed", "Pz_removed", "KE_removed", "volume_total",
                  "momentum_z_total", "kinetic_energy_total")
# box_noise.txt, with the block's post.log_box_noise, a row per box and ordinary rebuild after the
# cut (reason rule or distortion): the box (about a tracked neck with a head, neck_<tip>, or a tip
# on the axis, tip_<tip>), its centre on the axis and radius, and its budgets before and after the
# rebuild as a liquid cut takes them (the liquid volume and the area of the interface from the
# polylines, the axial momentum and the kinetic energy on the grid, with the points found there)
BOX_NOISE_COLUMNS = ("step", "time", "tau", "tau_p", "reason", "remesh_count", "box", "chain", "z",
                     "R", "V_before", "V_after", "dV", "A_before", "A_after", "dA", "Pz_before",
                     "Pz_after", "dPz", "KE_before", "KE_after", "dKE", "points_before",
                     "points_after")
BOX_NOISE_KEYS = ("V", "A", "Pz", "KE")  # the budgets of box_noise.txt, as dV, dA, dPz, dKE
# ring.txt, in the ring stage only (after the cut of a pierced bubble, ringsurgery.ring_cut): a row
# per step, the ring as ring.loop_record reads it (its volume, also in units of the bubble's at
# the gas cut, its section's area and centroid, its extremes, nodes, spacing, speed and Courant
# time), its fold (ring.loop_fold_gap) against its limit, the limit of its approach to the axis
# and its distance to the open chain, its worst quadratic element, and the volume, axial momentum
# and kinetic energy the ring surgeries have filled since the gas cut
RING_COLUMNS = ("step", "time", "tau", "tau_p", "dt", "ring_volume", "ring_volume_over_V_b",
                "ring_area_section", "ring_centroid_r", "ring_centroid_z", "ring_r_min",
                "ring_z_r_min", "ring_r_max", "ring_z_min", "ring_z_max", "ring_points",
                "ring_h_min", "ring_h_max", "ring_u_max", "ring_courant", "ring_fold_gap",
                "ring_fold_limit", "ring_axis_limit", "ring_open_gap", "edge_turn_ring",
                "edge_midside_ring", "V_filled", "Pz_filled", "KE_filled")
# step_timing.txt, with the driver's step_timing (a flag of the runner, no key of a case): a row
# per accepted step, the wall time of the parts of its iteration of the loop (seconds of
# perf_counter): the stops at the loop's top and the choice of the step, the solve, the
# measurement after it (with a rebuild for distortion), the bookkeeping and rows, the stops after
# the step, the snapshot, the remesh check, the commit after the cut and the dump; the total
STEP_TIMING_COLUMNS = ("step", "time", "dt", "top", "solve", "measure", "rows", "stops", "snapshot",
                       "remesh", "commit", "dump", "total")
# sheets.txt, with the block's groove.sheets log only: a row per sheet (a fold of a chain or of the
# ring whose pocket is liquid, groove.side_folds) at most once per rebuild: its chain and pair, the
# fold's limit and the limit of its faces' contact, the element length of its finer side, and what
# groove.sheet_reading reads of its pocket
SHEET_COLUMNS = ("step", "time", "tau", "tau_p", "remesh_count", "chain", "a", "b", "r", "z",
                 "thickness", "fold_limit", "h_fine", "contact_limit", "arc", "length",
                 "tip_distance", "tip_r", "tip_z", "area", "mean_thickness", "volume")


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


def _plain(x):
    """A value of an event's record as its file holds it (as ``surgery._jsonable`` has the cut's):
    numbers that are not finite None, numpy's values Python's."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _finite(x)
    return x


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
    # with jetcore: the yields of far kinks below the far floor made so far (jetcore.far_kink_regions)
    far_yields: int = 0
    # with pinch.enabled: the history of each tracked neck, matched across steps by its chain and
    # height (events.match_neck): {"next": the next key's number, "necks": {key: {chain, z_n, r_n,
    # samples [[t, r_n], ...]}}}, from which events.closing_neck reads whether it closes
    neck_history: dict = field(default_factory=dict)
    # in the ring stage with jetcore's recentred far kinks: the ring's liquid lip, its thinnest
    # section's thickness over the last accepted states ([t, thickness], sheetsurgery.LIP_SAMPLES)
    lip_samples: list = field(default_factory=list)
    # in the ring stage: the last retry of a step folded at a closing corner (batch B; its input's
    # key, ``ringcontact.input_key``, consumes the one retry of that input), and the cap of the
    # steps from its time (``t``, ``dt``), kept apart: a fill leaves the time and the cap but
    # changes the input, whose own fold is retried again
    fold_retry: dict | None = None
    fold_cap: dict | None = None


def _in_ulps(dt: float, t: float) -> float | None:
    """``dt`` in units of the spacing of doubles at ``t`` (``math.ulp``); None at ``t = 0``, where
    that spacing is the subnormals'."""
    return None if t == 0.0 else dt / math.ulp(t)


class CollapseDriver:
    def __init__(self, problem, settings: CollapseSettings, outdir: Path, *,
                 keep_stop_state: bool = False, landing=(),
                 reconnect: ReconnectSettings | None = None,
                 events: EventSettings | None = None, leg_info: dict | None = None,
                 step_timing: bool = False):
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
            from .jetcore import core_ratio  # the core refined, never coarsened

            core_ratio(events.jetcore.core_elements_per_scale,
                       settings.elements_per_scale if reconnect.elements_per_scale is None
                       else reconnect.elements_per_scale)
        self._post_cut_cache: tuple | None = None
        # with jetcore.box_gate: the stop of a rebuild whose box changed beyond it (_gate_after)
        self._gate_stop: dict | None = None
        # after a cut: the stops of the last state checked (_post_cut_stop), by its key, so that the
        # check of the same state at the loop's top after the step's own is not made again
        self._stop_cache: tuple | None = None
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
        self._step_snapshot = None  # after a cut: the state the last step started from
        self._axis_repairs: list[dict] = []  # after a cut: the folds of a tip repaired (this leg)
        self._tip_edge_balances: list[dict] = []  # signed changes of the audited tip edges
        self._tip_straightens: list[dict] = []  # after a cut: curved edges of the bubble made straight
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
        self._event_rows = self._drops = self._budget = None
        self._ring_rows = None  # ring.txt, opened in the ring stage only (_open_ring_rows)
        # sheets.txt, with groove.sheets log only, opened at its first row (_log_sheets): the
        # rebuild of its last rows and the rows written
        self._sheet_rows = None
        self._sheets_rc = None
        self._sheets_logged = 0
        self._dumps_dropped = 0  # the dumps of events the policy deleted (post.keep_event_dumps)
        self._meshes_pruned = 0  # the files of meshes no dump refers to deleted (post.prune_gmsh)
        self._event_before: tuple | None = None
        self._event_after: dict | None = None
        # with the block's post.log_box_noise: what each ordinary rebuild after the cut changes in
        # the budgets of a liquid cut's box (box_noise.txt; _box_noise_before, _box_noise_after)
        self._noise = self._noise_stats = None
        if events is not None:
            self._event_rows = open(self.outdir / "events.txt", "w", buffering=1)
            self._event_rows.write("# " + "\t".join(EVENT_COLUMNS) + "\n")
            self._drops = open(self.outdir / "drops.txt", "w", buffering=1)
            self._drops.write("# " + "\t".join(DROP_COLUMNS) + "\n")
            self._budget = open(self.outdir / "budget.txt", "w", buffering=1)
            self._budget.write("# " + "\t".join(BUDGET_COLUMNS) + "\n")
            if events.post.log_box_noise:
                self._noise = open(self.outdir / "box_noise.txt", "w", buffering=1)
                self._noise.write("# " + "\t".join(BOX_NOISE_COLUMNS) + "\n")
                self._noise_stats = {"rebuilds": 0, "rows": 0,
                                     "max_abs": dict.fromkeys(BOX_NOISE_KEYS, 0.0),
                                     "sum_sq": dict.fromkeys(BOX_NOISE_KEYS, 0.0),
                                     "count": dict.fromkeys(BOX_NOISE_KEYS, 0)}
        # with the block's jetcore.enabled: jetcore.txt, a row per step after the cut (jetcore.py),
        # and what set the step the row records (_dt_limiter)
        self._jetcore_rows = None
        self._dt_limiter = "start"
        if events is not None and events.jetcore.enabled:
            from .jetcore import COLUMNS as JETCORE_COLUMNS

            self._jetcore_rows = open(self.outdir / "jetcore.txt", "w", buffering=1)
            self._jetcore_rows.write("# " + "\t".join(JETCORE_COLUMNS) + "\n")
        # with step_timing (a flag of the runner): step_timing.txt, the wall time of the parts of
        # each step (STEP_TIMING_COLUMNS); it reads clocks only and changes no value
        self._timing = None
        self._ticks: dict = {}
        if step_timing:
            self._timing = open(self.outdir / "step_timing.txt", "w", buffering=1)
            self._timing.write("# " + "\t".join(STEP_TIMING_COLUMNS) + "\n")
        (self.outdir / "collapse_settings.json").write_text(
            json.dumps(asdict(settings), indent=2) + "\n")

    def _tick(self, part: str | None = None):
        """With step timing: the wall time since the last tick added to ``part`` of the current
        step (none at the first tick of a step); nothing without it."""
        if self._timing is None:
            return
        now = time.perf_counter()
        last = self._ticks.pop("_last", None)
        if part is not None and last is not None:
            self._ticks[part] = self._ticks.get(part, 0.0) + now - last
        self._ticks["_last"] = now

    def _write_timing(self, dt: float):
        """With step timing: the row of ``step_timing.txt`` of the step just ended."""
        if self._timing is None:
            return
        self._tick("dump")
        parts = {k: v for k, v in self._ticks.items() if k != "_last"}
        row = {"step": self.state.step, "time": self._t(), "dt": dt, **parts,
               "total": sum(parts.values())}
        self._timing.write("\t".join(_table_cell(row.get(c, 0.0)) for c in STEP_TIMING_COLUMNS)
                           + "\n")
        self._ticks = {"_last": self._ticks["_last"]}

    def close(self):
        self._table.close()
        self._events.close()
        if self._chains is not None:
            self._chains.close()
        for f in (self._event_rows, self._drops, self._budget, self._noise, self._ring_rows,
                  self._timing, self._jetcore_rows, self._sheet_rows):
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
        if self._ring() is not None:
            state = self._ring_stage_state()
            self._post_cut_cache = (key, state)
            return state
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
        core = self._core_state(segs, apexes, [])
        state = {"segs": segs, "record": rec, "courant": cl, "tips": tips, "apexes": apexes,
                 "edges": edges, "R_apex_min": R_apex_min, "core": core,
                 "dt_courant": self._dt_courant(cl, core),
                 "dt_capillary": self.s.capillary * R**1.5 if math.isfinite(R) else math.inf}
        self._post_cut_cache = (key, state)
        return state

    def _jetcore(self):
        """The events block's ``jetcore`` part where it is enabled, after a cut; None otherwise
        (every rule of the run then as without the part)."""
        if self.events is None or not self.events.jetcore.enabled or self._cut_record() is None:
            return None
        return self.events.jetcore

    def _far_courant(self) -> float:
        """The Courant number of the far nodes: ``jetcore.far_courant``, or the core's
        ``reconnect.courant``."""
        jet = self._jetcore()
        if jet is None or jet.far_courant is None:
            return self.reconnect.courant
        return jet.far_courant

    def _core_ratio(self) -> float | None:
        """With ``jetcore.core_elements_per_scale`` above the run's elements per scale after the cut
        (``reconnect.elements_per_scale`` or the collapse's): the factor of every size on the core's
        nodes (``jetcore.core_ratio``); None otherwise, every size as without the key."""
        jet = self._jetcore()
        if jet is None or jet.core_elements_per_scale is None:
            return None
        from .jetcore import core_ratio

        eps = (self.s.elements_per_scale if self.reconnect.elements_per_scale is None
               else self.reconnect.elements_per_scale)
        return core_ratio(jet.core_elements_per_scale, eps)

    def _core_state(self, segs: list, apexes: dict, loops: list) -> dict | None:
        """With ``jetcore``: the core of the open chain (``jetcore.core_mask``, about the base of
        its jet, ``apexes``' ``cavity_tip``) and the Courant limit split between the core and the
        far nodes of the chains and the loops (``split``, ``jetcore.courant_split``); None without
        the part or without an open chain."""
        jet = self._jetcore()
        opens = [c for c in segs if c["open"]]
        if jet is None or not opens:
            return None
        from .jetcore import core_mask, courant_split, prep_mask, prep_regions
        from .ring import loop_segment

        o = opens[0]
        core = core_mask(np.column_stack([o["r"], o["z"]]), o["end_types"],
                         apexes.get("cavity_tip"), delta=self.reconnect.delta,
                         core_radius=jet.core_radius, wall_radius=jet.wall_radius,
                         apex_extent=self.reconnect.apex_extent,
                         track_depth=self.events.liquid.track_depth)
        bubble = next((c for c in segs if not c["open"]), None)
        core["prep"] = prep_regions(bubble if self._ring() is None else None,
                                    delta=self.reconnect.delta, prep_gap=jet.prep_gap,
                                    prep_rho=jet.prep_rho, h_floor=self._h_floor())
        chains = list(segs) + [loop_segment(x) for x in loops]
        masks = [core["mask"] if c is o else prep_mask(c, core["prep"]) if c is bubble else None
                 for c in chains]
        core["split"] = courant_split(chains, masks, core=self.reconnect.courant,
                                      far=self._far_courant())
        return core

    def _dt_courant(self, cl: dict, core: dict | None) -> float:
        """The Courant limit of the step after the cut: ``reconnect.courant`` times the smallest
        ``h/|u|`` of every node (``cl``, ``reconnect.courant_limit``), or, with a far Courant number
        of its own, the split limit of the core (``core``, :meth:`_core_state`)."""
        if core is None or core["split"]["far_courant"] == self.reconnect.courant:
            return self.reconnect.courant * cl["h_over_u"]
        return core["split"]["dt"]

    def _ring(self) -> dict | None:
        """The problem's record of the ring stage (``FaradayProblem.ring``), after the ring cut of
        a pierced bubble (``ringsurgery.ring_cut``); None before it, and on a problem without it."""
        return getattr(self.pb, "ring", None)

    def _ring_stage_state(self) -> dict:
        """:meth:`_post_cut_state` in the ring stage, with the same keys: ``segs`` the chains of
        the free surface (the open chain alone, ``FaradayProblem.interface_segments`` setting the
        ring aside), ``record`` with every key of ``reconnect.post_cut_record``
        (``ring.ring_stage_record``: the bubble's volume and widest point the ring's, its other keys
        NaN; empty where the chains are not the open chain alone, a changed topology a stop
        records), the Courant limit over the open chain and the ring (``chain`` 1 the ring), no
        tips of a bubble, the base of the open chain's jet (``apexes``), the quadratic elements of
        the open chain and the ring (``edges``, ``ring`` the ring's) and the step limits; besides,
        the ring as ``FaradayProblem.interface_loops`` reads it (``loops``) and its record
        (``ring_record``, ``ring.loop_record``, None once it is removed)."""
        from .ring import loop_edges, loop_record, loop_segment, ring_stage_record

        segs = self.pb.interface_segments()
        loops = self.pb.interface_loops()
        opens = [c for c in segs if c["open"]]
        rec = (ring_stage_record(opens[0], loops[0] if len(loops) == 1 else None)
               if len(segs) == 1 and opens and len(loops) <= 1 else {})
        cl = courant_limit(segs + [loop_segment(x) for x in loops])
        apexes = self._apexes(opens[0], None, {}) if opens else {}
        edges = {("open" if c["open"] else "bubble"):
                 edge_reading(np.column_stack([c["r"], c["z"]])) for c in segs}
        if len(loops) == 1:
            edges["ring"] = loop_edges(np.column_stack([loops[0]["r"], loops[0]["z"]]))
        R_apex = [a["R_apex"] for a in apexes.values()
                  if math.isfinite(a["R_apex"]) and a["R_apex"] > 0.0]
        R_apex_min = min(R_apex) if R_apex else math.inf
        R = min([x for x in (rec.get("R_tip_min", math.nan), R_apex_min) if math.isfinite(x)],
                default=math.nan)
        ring_record = (loop_record(np.column_stack([loops[0]["r"], loops[0]["z"]]),
                                   np.column_stack([loops[0]["u_r"], loops[0]["u_z"]]))
                       if len(loops) == 1 else None)
        core = self._core_state(segs, apexes, loops)
        return {"segs": segs, "record": rec, "courant": cl, "tips": {}, "apexes": apexes,
                "edges": edges, "R_apex_min": R_apex_min, "core": core,
                "dt_courant": self._dt_courant(cl, core),
                "dt_capillary": self.s.capillary * R**1.5 if math.isfinite(R) else math.inf,
                "loops": loops, "ring_record": ring_record}

    def _apexes(self, cavity: dict, bubble: dict | None, tips: dict) -> dict:
        """The base of the jet on each tip on the axis after a cut where a resolved bump stands (the
        rule by which its radius is a scale): the base (``r``, ``z``, the jet-base radius and
        height in the frame of the axis), its node along its chain (``chain``, ``index``), the
        vertex of the apex there (``apex_r``, ``apex_z``), its radius of curvature ``R_apex`` and
        the opening angle of the groove around the jet (``reconnect.jet_base_apex``), and the fold
        of that groove (``groove_gap``, ``groove_limit``, ``groove_arc``: ``reconnect.groove_gap``,
        with the driver's fold limits). By tip name. In the ring stage (``bubble`` None) the open
        chain's alone."""
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
        n = len(bubble["r"]) if bubble is not None else 0
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

    def _stop_key(self) -> tuple:
        """The key of a state for its stops after the cut (:meth:`_post_cut_stop`): its time, its
        rebuild and step, whether a step has run on its mesh, the events made and the stage of the
        ring. The stops are pure functions of the state so keyed: the check at the loop's top of
        the state the step's own check read gives its answer again."""
        ring = self._ring()
        return (self._t(), self.pb.remesh_count, self.state.step, self._stepped_rc,
                len(self._made_events()), ring is not None,
                ring is not None and ring.get("removed") is not None)

    def _post_cut_stop(self) -> dict | None:
        """The stops before a second change of topology, on both chains (post-cut mode), with the
        guard of the interface on its quadratic elements; on a freshly built mesh also the check of
        its curved elements (:meth:`_jacobian_stop`): a rebuild is accepted only if every element
        is valid. Evaluated once per state (:meth:`_stop_key`), each caller given its own copy,
        which it may change."""
        if self._cut_record() is None:
            return None
        key = self._stop_key()
        if self._stop_cache is not None and self._stop_cache[0] == key:
            return copy.deepcopy(self._stop_cache[1])
        stop = self._post_cut_stop_uncached()
        self._stop_cache = (key, copy.deepcopy(stop))
        return stop

    def _post_cut_stop_uncached(self) -> dict | None:
        """:meth:`_post_cut_stop`, evaluated."""
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
        ring = self._ring()
        if self.events is not None and ring is not None:
            # the ring stage (ring.ring_stage_events): the open chain and the ring, read apart
            from .ring import ring_stage_events

            loops = [np.column_stack([x["r"], x["z"]]) for x in pc["loops"]]
            stop = ring_stage_events(chains, loops, CutGeometry(**cut["geometry"]),
                                     self.reconnect, cut["sizing"]["h_tip"], self.events, ring,
                                     neck_elements=s.neck_stop_elements, fold_ratio=s.fold_ratio,
                                     fold_stop=s.fold_stop,
                                     fold_stop_elements=s.fold_stop_elements,
                                     fold_search=FOLD_SEARCH, apexes=apexes,
                                     guard={"fresh": fresh, "h_floor": self._h_floor(),
                                            "refine_ratio": s.refine_ratio},
                                     invalid=self._jacobian_stop if fresh else None,
                                     pinch=self._pinch_context(),
                                     sheet=self._sheet_context(loops))
            if stop is not None:
                stop.update(self._taus(self._t()))
            return stop
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
                                  invalid=self._jacobian_stop if fresh else None,
                                  pinch=self._pinch_context())
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

    def _pinch_context(self) -> dict | None:
        """With the block's ``pinch.enabled``: what the stops read the trigger of the tracked necks
        from (``events.closing_readings``): the state's time and the history of the necks; None
        otherwise."""
        if self.events is None or not self.events.pinch.enabled:
            return None
        return {"t": self._t(), "history": self.state.neck_history}

    def _commit_neck_history(self, t: float):
        """With ``pinch.enabled``, at the end of an accepted step: the state's tracked necks
        (``events.liquid_necks``) added to their histories (``neck_history``, each matched by its
        chain and height, ``events.match_neck``; a neck without one starts its own, a history whose
        neck is gone is dropped), the latest ``events.NECK_HISTORY_SAMPLES`` samples kept."""
        from .events import NECK_HISTORY_SAMPLES, match_neck

        st = self.state
        named = self._named_post_cut_chains()
        old = (st.neck_history or {}).get("necks") or {}
        nxt = int((st.neck_history or {}).get("next", 1))
        new, used = {}, set()
        for x in (liquid_necks(named, self.events.liquid) if named else []):
            key = match_neck(old, x["chain"], x["z_n"], x["r_n"], used)
            if key is None:
                key, nxt, samples = f"{x['chain']}-{nxt}", nxt + 1, []
            else:
                used.add(key)
                samples = [list(v) for v in old[key]["samples"]]
            if not samples or t > samples[-1][0]:
                samples.append([float(t), float(x["r_n"])])
            new[key] = {"chain": x["chain"], "z_n": float(x["z_n"]), "r_n": float(x["r_n"]),
                        "samples": samples[-NECK_HISTORY_SAMPLES:]}
        st.neck_history = {"next": nxt, "necks": new}

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

    def _write_stop_record(self, m: dict, stop: dict) -> Path | None:
        """The record of the event the run stops on (``events/event_<k>.json``, schema
        ``faraday-event/2``), so that every event a run meets has one. For an event no handler was
        made for (``events.STOP_OUTCOMES``: ``outcome`` ``unhandled``, ``cascade`` or
        ``max-events``, ``reason`` the stop's): its kind, class and handler, the request as the
        stops found it, its chain and, for a liquid neck, the neck as its ``trigger`` and its tip;
        its step, leg, time (``tau``, ``tau_p``, ``tau_over_delta2``, ``t_prime_over_T``), the
        cascade's verdict where it was judged, the dump of the state before it (``dumps.precut``),
        the block, its hash and the commit. A cut refused or unclosed has written its own record
        (``surgery.liquid_cut``), to which the dump is added where the cut kept none. Returns the
        file, None where there is none."""
        k, reason = int(stop["k"]), stop["reason"]
        path = self.outdir / "events" / f"event_{k:03d}.json"
        if reason not in STOP_OUTCOMES:
            if path.is_file():
                full = json.loads(path.read_text())
                if not (full.get("dumps") or {}).get("precut"):
                    full["dumps"] = {**(full.get("dumps") or {}), "precut": stop["precut"]}
                    tmp = path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(full, indent=1) + "\n")
                    tmp.replace(path)
                return path
            return None
        st, delta = self.state, self.reconnect.delta
        t = float(m["time"])
        t_cut, t_p = self._cut_times()
        T = t_p - t_cut
        taus = self._taus(t)
        request = {key: v for key, v in stop.items()
                   if key not in ("reason", "k", "precut", "end_state", "cascade", "max_events")}
        neck = event_neck(stop) if stop.get("kind") in EVENT_CLASSES["liquid"] else {}
        tip = event_tip(neck) if neck.get("side") is not None else None
        cascade = stop.get("cascade")
        rec = {"schema": EVENT_SCHEMA, "k": k, "kind": stop.get("kind"),
               "class": stop.get("class"), "handler": stop.get("handler"),
               "outcome": STOP_OUTCOMES[reason], "reason": reason,
               "chain": stop.get("chain", neck.get("chain")), "tip": tip,
               "generation": (cascade or {}).get("generation"), "step": st.step,
               "leg": len(st.legs) - 1, "t": t, **taus, "delta": delta,
               "tau_over_delta2": taus["tau"] / delta**2,
               "t_prime_over_T": taus["tau_p"] / T if math.isfinite(T) and T != 0.0 else None,
               "trigger": neck, "request": request, "cascade": cascade,
               "max_events": stop.get("max_events"), "dumps": {"precut": stop["precut"]},
               "settings": self.events.to_dict(), "settings_sha256": self.events.sha256(),
               "commit": self.leg_info.get("commit")}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_plain(rec), indent=1) + "\n")
        return path

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
        stop = {"reason": "interface-unresolved", "trigger": "curved-jacobian",
                "jacobian_bound": j["min"], "limit": self.reconnect.jacobian_min, "r": j["r"],
                "z": j["z"], "on_interface": j["on_interface"], "elements_below": j["below"],
                "fresh": True}
        return self._tip_straighten(stop)

    def _tip_straighten(self, stop: dict) -> dict | None:
        """A fresh mesh's ``curved-jacobian`` at an element on the bubble's chain before the
        piercing, with jetcore's recentred far kinks (decision 22: the bubble a nuisance): the free
        surface's quadratic edges about the element (within ``TIP_STRAIGHTEN_REACH`` times the
        distance of its centroid to the surface, at least that times the chain's spacing there)
        whose midside node lies more than ``AXIS_MIDSIDE_LIMIT`` chords off the middle are put
        straight (``FaradayProblem.straighten_interface_edges``), if the volume between their
        curves and chords is within ``axisrepair.AXIS_REPAIR_BOUND delta^3``, and the bound read
        again; at least ``jacobian_min`` there, the stop is met (event ``tip-straighten``, None
        returned). r3 (fix5 leg 1 at aa48432, step 7339): the first rebuild after the repair of its
        folded tip meshed the surface's edge beside the new end, where it bends by 2e-6 within a
        chord of 4.4e-6, with its midside node a quarter of the chord off, the element's bound
        0.054 against 0.2. Otherwise the stop is returned (with ``tip_straighten``, the reading,
        where it was tried). A radius reaching the open chain is refused; a straightening whose
        Jacobian bound still fails is rolled back at every history level."""
        from .axisrepair import AXIS_REPAIR_BOUND

        jet = self._jetcore()
        if not stop.get("on_interface") or jet is None or jet.far_kinks != "recentre" \
                or self._ring() is not None or not hasattr(self.pb, "straighten_interface_edges"):
            return stop
        x = np.array([float(stop["r"]), float(stop["z"])])
        near = None
        for c in self.pb.interface_segments():
            P = np.column_stack([c["r"], c["z"]])
            if len(P) < 3:
                continue
            d = np.hypot(P[:, 0] - x[0], P[:, 1] - x[1])
            k = int(np.argmin(d))
            if near is None or d[k] < near[0]:
                h = float(np.max(np.hypot(*(P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]).T)))
                near = (float(d[k]), bool(c["open"]), h)
        if near is None or near[1]:  # nearest the open chain: the upward jet's, not a nuisance
            return stop
        radius = TIP_STRAIGHTEN_REACH * max(near[0], near[2])
        for chain in self.pb.interface_segments():
            if not chain["open"]:
                continue
            P = np.column_stack([chain["r"], chain["z"]])
            centres = 0.5 * (P[:-2:2] + P[2::2])
            if (np.any(np.hypot(P[:, 0] - x[0], P[:, 1] - x[1]) <= radius)
                    or np.any(np.hypot(centres[:, 0] - x[0], centres[:, 1] - x[1]) <= radius)):
                return stop
        limit = AXIS_REPAIR_BOUND * self.reconnect.delta ** 3
        read = self.pb.straighten_interface_edges(x[0], x[1], radius, apply=False)
        reading = {"radius": radius, "edges": read["edges"], "dV": read["dV"],
                   "dV_abs": read["dV_abs"], "bound_dV": limit}
        if not read["edges"] or read["dV_abs"] > limit:
            return dict(stop, tip_straighten=reading)
        snapshot, was_fresh = self.pb._snapshot_state(), self._fresh_mesh()
        self.pb.straighten_interface_edges(x[0], x[1], radius)
        self._post_cut_cache = None
        j = self.pb.curved_jacobian_bound(self.reconnect.jacobian_min)
        reading.update(jacobian_bound_after=j["min"] if j is not None else None)
        if j is not None and j["min"] < self.reconnect.jacobian_min:
            self.pb._restore_state(snapshot)
            self._stepped_rc = None if was_fresh else self.pb.remesh_count
            self.pb.invalidate_cached_mesh_data()
            self._post_cut_cache = None
            reading["rolled_back"] = True
            return dict(stop, tip_straighten=reading)
        self._tip_straightens.append({"step": self.state.step, **reading})
        here = {"time": self._t(), "r_j": math.nan, "z_axis": math.nan, "ell": math.nan,
                "axis_curvature": math.nan}  # read inside the stops: no measurement of its own
        self._event("tip-straighten", here, jacobian_bound=stop["jacobian_bound"],
                    element_r=stop["r"], element_z=stop["z"], **reading)
        return None

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
        out = {"delta": delta, **lengths, **times,
               "in_delta": {**{k: v / delta for k, v in lengths.items()},
                            **{k: v / delta**2 for k, v in times.items()}}}
        jet = self._jetcore()
        if jet is not None:  # with jetcore only: other legs keep their keys
            out["jetcore"] = {"core_courant": self.reconnect.courant,
                              "far_courant": self._far_courant(),
                              "core_radius": jet.core_radius, "wall_radius": jet.wall_radius}
            if jet.core_elements_per_scale is not None:  # only where it is set: legs as before
                out["jetcore"].update(core_elements_per_scale=jet.core_elements_per_scale,
                                      core_ratio=self._core_ratio())
        return out

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
        Also returns the kink regions as entries, which the step commits at its end.

        In the ring stage the features are the open chain's alone, the kink regions those of the
        open chain and the ring (``kink_ring``), and the ring's own regions are frozen at the cut
        (the problem's record of the stage, ``sizing.frozen`` and ``sizing.ring_cut``; none once
        the ring is removed): each emitted at its size, as its target and the size it keeps, with
        no element read in it (``h_max``, ``h_min`` NaN), so that a rebuild sizes it exactly at its
        size and none asks for one (:meth:`_regions_focus`); the guard's kink regions still refine
        the ring where an element bends."""
        pc = self._post_cut_state()
        segs = pc["segs"]
        ring = self._ring()
        if ring is not None:
            if len(segs) != 1 or not pc["record"]:
                return [], {}
        elif len(segs) != 2 or not pc["record"]:
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
        from .ring import ring_floored_ok

        refine = interface_guard(pc["edges"], self.reconnect, fresh=self._fresh_mesh(),
                                 h_floor=h_floor, refine_ratio=s.refine_ratio,
                                 floored_ok=ring_floored_ok(self.events)
                                 if ring is not None else ())["refine"]
        focus = self.pb.active_mesh_template().focus
        jet, core = self._jetcore(), pc.get("core")
        q = self._core_ratio() if core is not None else None
        if q is not None and "open" in pc["edges"]:
            # the finer core's elements read at q times the guard's turn (jetcore.core_turns)
            from .jetcore import core_turns

            extra = core_turns(pc["edges"], core, turn=q * self.reconnect.edge_turn_refine,
                               min_length=s.refine_ratio * h_floor)
            if len(extra):
                refine["open"] = np.union1d(refine.get("open", np.zeros(0, dtype=int)),
                                            extra).astype(int)
        if jet is not None and core is not None and jet.far_kinks == "recentre":
            kinks = self._split_kinks(refine, pc, core, U, focus)
        else:
            kinks = kink_regions(st.kinks, pc["edges"], refine, t=self._t(), h_floor=h_floor,
                                 hold=lambda L: s.coarsen_hold * local_time(L, U))
        regions += [{"name": g["name"], "target": g["target"], "r": g["r"], "z": g["z"],
                     "radius": g["radius"], "ell": math.inf, "floor": h_floor,
                     "released": g["released"]} for g in kinks.values()]
        axis_tips = [float(c["z"][k]) for c in segs for k, end in ((0, 0), (-1, 1))
                     if c["end_types"][end] == AXIS]
        frozen = self._frozen_ring_regions()
        regions += frozen
        if jet is not None and core is not None:
            regions += [dict(g) for g in core.get("prep") or ()]  # before the piercing
            self._far_sizing(regions, segs, core)
            if q is not None:  # the core meshed finer: the twins on its nodes
                from .jetcore import core_spec, core_twins

                o = next(c for c in segs if c["open"])
                P = np.column_stack([o["r"], o["z"]])
                regions += core_twins(regions, P, core, core_spec(core, P, jet.wall_radius),
                                      ratio=q, h_floor=h_floor)
        for g in regions:
            current = focus.region(g["name"]) if focus is not None else None
            g["size"] = current.size if current is not None else math.nan
        grading = s.focus_grading
        spaced = segs
        if ring is not None:  # the ring's elements count in the regions' spacing
            from .ring import loop_segment

            spaced = segs + [loop_segment(x) for x in pc["loops"]]
        split = self._twin_selection(regions, core)
        for g in regions:
            if g.get("frozen"):
                g["h_max"] = g["h_min"] = math.nan
                continue
            own = g["size"] if math.isfinite(g["size"]) else g["target"]
            finer = []
            for f in regions:
                other = f["size"] if math.isfinite(f["size"]) else f["target"]
                if f is not g and other < own and (split is None
                                                   or ("twin_of" in f) == ("twin_of" in g)):
                    finer.append((f["r"], f["z"], f["radius"] + (own - other) / grading))
            g["h_max"], g["h_min"] = region_spacing(spaced, g, axis_tips, cut.get("sizing"), own,
                                                    finer, select=None if split is None
                                                    else split(g))
        return regions, kinks

    @staticmethod
    def _twin_selection(regions: list[dict], core: dict | None):
        """With the core's twins (``jetcore.core_twins``): the nodes each region's spacing reads
        (``reconnect.region_spacing``'s ``select``), a function of the region: a twin's the core's
        nodes of the open chain, a region with a twin its other nodes, any other region every node;
        a twin's finer regions are twins and a region's finer regions other regions, so that each
        rule reads its own nodes as without the twins. None without twins."""
        twinned = {g["twin_of"] for g in regions if "twin_of" in g}
        if not twinned:
            return None
        mask = np.asarray(core["mask"], dtype=bool)

        def select(g):
            if "twin_of" in g:
                return lambda c: (mask if c.get("open")
                                  else np.zeros(len(c["r"]), dtype=bool))
            if g["name"] in twinned:
                return lambda c: ~mask if c.get("open") else np.ones(len(c["r"]), dtype=bool)
            return None

        return select

    def _split_kinks(self, refine: dict, pc: dict, core: dict, U: float, focus) -> dict:
        """With ``jetcore.far_kinks`` ``recentre``: the guard's kink regions split between the core
        and the far region (``jetcore.split_refine``): the core's by the rule without the part
        (``reconnect.kink_regions``, keyed by chain), their radius at most ``R_K``; the far
        region's in a region of their own per chain (``jetcore.far_kink_regions``, keyed
        ``<chain>_far``), at the far floor (and, with ``ring_kink_floor``, the ring's no finer than
        the column), yielding below it only past ``far_yield_turn``."""
        from .jetcore import far_kink_regions, split_refine

        s, st, jet = self.s, self.state, self._jetcore()
        h_floor = self._h_floor()

        def hold(L):
            return s.coarsen_hold * local_time(L, U)

        near, far = split_refine(refine, pc["edges"], core)
        kinks = kink_regions({k: v for k, v in st.kinks.items() if not v.get("far")},
                             pc["edges"], near, t=self._t(), h_floor=h_floor, hold=hold)
        if not core["whole"] and math.isfinite(core["R_K"]):
            for g in kinks.values():
                g["radius"] = min(g["radius"], core["R_K"])
        sizes = {g.name: g.size for g in (focus.regions if focus is not None else ())}
        h_col = self._column_size()
        floors = {"ring": h_col} if jet.ring_kink_floor and math.isfinite(h_col) else {}
        previous = self._lip_hold({k: v for k, v in st.kinks.items() if v.get("far")})
        regions = far_kink_regions(previous, pc["edges"], far, t=self._t(), hold=hold,
                                   h_far=jet.far_floor, h_floor=h_floor,
                                   yield_turn=jet.far_yield_turn, sizes=sizes, floors=floors)
        kinks.update(self._lip_cover(previous, regions))
        return kinks

    def _lip_read(self):
        """The ring's lip as the measurement holds it (decision 22, :meth:`_sheet_context`): its last
        sample ``(r, z, thickness)`` where the lip is handled (:meth:`_sheet_handled`) and was read
        at the state this one goes on from (the sample at ``t_prev`` or later, the start of the
        step; at this state's time after an event, which sets ``t_prev`` NaN); None otherwise (a
        lip no longer read holds nothing)."""
        st = self.state
        samples = st.lip_samples
        if not samples or len(samples[-1]) < 4 or not self._sheet_handled():
            return None
        t = self._t()
        t_from = st.t_prev if math.isfinite(st.t_prev) else t
        if float(samples[-1][0]) < t_from:
            return None
        return float(samples[-1][2]), float(samples[-1][3]), float(samples[-1][1])

    @staticmethod
    def _holds_lip(g: dict, lip) -> bool:
        """Whether the far kink ``g`` is the refinement that holds the lip ``lip`` (``(r, z,
        thickness)``): a refined region of the ring, not released, whose disc holds the lip's
        section."""
        return (lip is not None and g.get("chain") == "ring" and bool(g.get("refined"))
                and not g.get("released")
                and math.hypot(lip[0] - float(g["r"]), lip[1] - float(g["z"])) <= float(g["radius"]))

    def _lip_hold(self, previous: dict) -> dict:
        """The far kinks of the last accepted step (``previous``) with the refinement that holds
        the ring's lip (:meth:`_lip_read`, :meth:`_holds_lip`) renewed, its ``t_set`` now, before
        ``jetcore.far_kink_regions`` reads which to release and every target from them, so that
        the measurement, the remesh checks and the commit (``_commit_post_cut``) all keep it; its
        disc is kept about the lip after the reading by :meth:`_lip_cover`. The entries renewed
        are copies; ``previous`` is returned as it is otherwise."""
        lip = self._lip_read()
        if lip is None:
            return previous
        t = self._t()
        return {key: (dict(g, t_set=max(float(g["t_set"]), t)) if self._holds_lip(g, lip) else g)
                for key, g in previous.items()}

    def _lip_cover(self, previous: dict, regions: dict) -> dict:
        """``jetcore.far_kink_regions``' regions (``regions``) with the disc of each refinement that
        held the ring's lip before the reading (``previous``, :meth:`_holds_lip`) kept about the
        lip: a region the reading left as it was (its entry unchanged: no element of the guard
        renewed it, even at its own centre) follows the lip, recentred on its section
        (its radius and target as they were), so that the lip, which moves out along the plug at
        the sheet's speed, does not leave its refinement for a rebuild to coarsen it (r1s4, fix15
        leg 1 at bf80e74: the lip, 4.3e-6 thick, 3.0 elements across at step 8201, left the region
        its seventh refine had set at step 8163 and met the rebuild of step 8208 with 8.8e-6
        elements, half an element across); a region the guard's elements recentred on themselves
        (four of their chords wide, which can leave the lip outside it) is widened about its new
        centre to hold the ball about the lip that the held disc held (its radius the held disc's
        clearance about the lip's section, at least the lip's thickness), its target kept; a
        region released, gone, or still holding that ball is as the reading made it. The entries
        moved or widened are copies."""
        lip = self._lip_read()
        if lip is None:
            return regions
        out = dict(regions)
        for key, g in previous.items():
            new = out.get(key)
            if new is None or new.get("released") or not self._holds_lip(g, lip):
                continue
            if new == g:  # the reading left it as it was (no element renewed it): it follows the lip
                out[key] = dict(new, r=lip[0], z=lip[1])
                continue
            clearance = max(float(g["radius"]) - math.hypot(lip[0] - float(g["r"]),
                                                            lip[1] - float(g["z"])), lip[2])
            need = math.hypot(lip[0] - float(new["r"]), lip[1] - float(new["z"])) + clearance
            if need > float(new["radius"]):
                out[key] = dict(new, radius=need)
        return out

    def _column_size(self) -> float:
        """The element size of the ring's column (the record of the ring stage's ``h_col``); NaN
        before the ring and once it is removed."""
        ring = self._ring()
        if ring is None or ring.get("removed") is not None:
            return math.nan
        return float((ring.get("sizing") or {}).get("h_col", math.nan))

    def _far_sizing(self, regions: list[dict], segs: list, core: dict):
        """With ``jetcore``: the class of each region (``jetcore.region_class``, ``class``) and the
        far region's targets: at least ``far_floor`` (``floored``), but for the far kinks, which
        floor and yield on their own (``jetcore.far_kink_regions``); with ``ring_kink_floor`` the
        ring's kinks at least the column's size."""
        from .jetcore import FAR, region_class

        jet = self._jetcore()
        opens = [c for c in segs if c["open"]]
        P = np.column_stack([opens[0]["r"], opens[0]["z"]]) if opens else None
        h_col = self._column_size()
        for g in regions:
            g["class"] = region_class(g, core, P)
            if g["class"] != FAR or g["name"].endswith("_far"):
                continue
            if jet.far_floor is not None and g["target"] < jet.far_floor:
                g["target"], g["floored"] = jet.far_floor, True
            if jet.ring_kink_floor and g["name"] == "kink_ring" and math.isfinite(h_col) \
                    and g["target"] < h_col:
                g["target"], g["floored"] = h_col, True

    def _frozen_ring_regions(self) -> list[dict]:
        """The ring's regions frozen at its cut (the problem's record of the stage, ``sizing``:
        the copies of the bubble's finer regions, ``ring_<name>``, and the column's, ``ring_cut``),
        each at its size as target and as the size it keeps (``frozen``); none before the ring and
        once it is removed."""
        ring = self._ring()
        if ring is None or ring.get("removed") is not None:
            return []
        sizing = ring.get("sizing") or {}
        out = []
        for g in list(sizing.get("frozen") or []) + ([sizing["ring_cut"]]
                                                     if sizing.get("ring_cut") else []):
            size = float(g["size"])
            out.append({"name": g["name"], "target": size, "r": float(g["r"]),
                        "z": float(g["z"]), "radius": float(g["radius"]), "ell": math.inf,
                        "floor": size, "kept": size, "frozen": True})
        return out

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
            new = int(g.get("yields", 0)) - int((st.kinks.get(chain) or {}).get("yields", 0))
            if g.get("far") and new > 0:  # with jetcore's far kinks: a yield below the floor
                st.far_yields += new
                self._event("far-yield", m, name=g["name"], chain=g["chain"], turn=g["turn"],
                            target=g["target"], yields=g["yields"], r=g["r"], z=g["z"])
        st.kinks = kinks
        if self.events is not None and self.events.pinch.enabled:
            self._commit_neck_history(t)

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
            out.append(FocusRegion(g["name"], size=size, r=g["r"], z=g["z"], radius=g["radius"],
                                   core=g.get("core")))
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
                       | ({"new_size": new.size} if new is not None else {})
                       | ({"class": g["class"]} if "class" in g else {})
                       | {k: g[k] for k in ("twin_of", "core_nodes", "core_floored") if k in g})
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
        if self._budget is not None:  # with the events block only
            self._write_budget_row(values)
        if self._ring_rows is not None:  # in the ring stage only
            self._write_ring_row(dt)
        if self._jetcore_rows is not None:  # with jetcore only
            self._write_jetcore_row(m, dt)
        if self.events is not None and self.events.groove.sheets == "log":
            self._log_sheets()

    def _log_sheets(self):
        """With the block's ``groove.sheets`` ``log``: the sheets of the state, at most once per
        rebuild (the first state of a rebuild that has one), each a row of ``sheets.txt``
        (:data:`SHEET_COLUMNS`, opened at its first row): on each chain and the ring the fold whose
        pocket is liquid (``groove.side_folds``) that closes below the fold's limit at the scale of
        the resolved surface (``contact`` ``fold``), which the stops meet as no event, and what
        ``groove.sheet_reading`` reads of its pocket (thickness at the pair, length beyond it, tip,
        area, mean thickness, volume)."""
        if self.pb.remesh_count == self._sheets_rc:
            return
        from .groove import sheet_reading, side_folds
        from .reconnect import fold_contact

        s, pc, t = self.s, self._post_cut_state(), self._t()
        kw = {"ratio": s.fold_ratio, "stop": self._fold_stop(),
              "stop_elements": s.fold_stop_elements, "search_factor": FOLD_SEARCH}
        items = []
        for c in pc["segs"]:
            P = np.column_stack([c["r"], c["z"]]).astype(float)
            for k, end in ((0, 0), (-1, 1)):
                if c["end_types"][end] == AXIS:
                    P[k, 0] = 0.0
            items.append(("open" if c["open"] else "bubble", P, False))
        items += [("ring", np.column_stack([x["r"], x["z"]]), True)
                  for x in pc.get("loops") or []]
        rows = []
        for name, P, loop in items:
            x = side_folds(P, loop=loop, **kw)["liquid"]
            f = x["fold"]
            limit = max(kw["stop"], s.fold_stop_elements * f.h_elem)
            if not (math.isfinite(f.gap) and f.gap < limit) or fold_contact(f.arc,
                                                                             f.h_elem) != "fold":
                continue
            rd = sheet_reading(P, f.pair, loop=loop)
            rows.append({"step": self.state.step, "time": t, **self._taus(t),
                         "remesh_count": self.pb.remesh_count, "chain": name,
                         "a": rd["pair"][0], "b": rd["pair"][1], "fold_limit": limit,
                         "h_fine": x["h_fine"],
                         "contact_limit": self.events.groove.sheet_contact * x["h_fine"],
                         **{k: rd[k] for k in ("r", "z", "thickness", "arc", "length",
                                               "tip_distance", "tip_r", "tip_z", "area",
                                               "mean_thickness", "volume")}})
        if not rows:
            return
        if self._sheet_rows is None:
            self._sheet_rows = open(self.outdir / "sheets.txt", "w", buffering=1)
            self._sheet_rows.write("# " + "\t".join(SHEET_COLUMNS) + "\n")
        for row in rows:
            self._sheet_rows.write("\t".join(_table_cell(row[c]) for c in SHEET_COLUMNS)
                                   + "\n")
        self._sheets_rc = self.pb.remesh_count
        self._sheets_logged += len(rows)

    def _write_jetcore_row(self, m: dict, dt: float):
        """The row of ``jetcore.txt`` of a step after the cut (``jetcore.COLUMNS``): the step just
        taken and what set it (:meth:`_dt_limiter_of`), and, as ``chains.txt`` has its limits, the
        split Courant limit and the core of the state it ended on."""
        from .jetcore import COLUMNS

        pc, t = self._post_cut_state(), self._t()
        core = pc.get("core") or {}
        split = core.get("split") or {}
        values = {"step": self.state.step, "time": t, **self._taus(t), "dt": dt,
                  "limiter": self._dt_limiter, "dt_courant": pc["dt_courant"],
                  "dt_core": split.get("dt_core"), "dt_far": split.get("dt_far"),
                  "binds": split.get("binds"), "far_courant": split.get("far_courant"),
                  **{k: core.get(k) for k in ("L", "R_K", "A_r", "A_z", "z_cap")},
                  "core_nodes": (split.get("core") or {}).get("nodes"),
                  "far_nodes": (split.get("far") or {}).get("nodes"),
                  "far_regions": sum(1 for g in m.get("regions") or () if g.get("class") == "far"),
                  "far_floored": sum(1 for g in m.get("regions") or () if g.get("floored")),
                  "far_yields": self.state.far_yields, "prep": len(core.get("prep") or ()),
                  "s_t": (core.get("prep") or [{}])[0].get("s_t")}
        for cls in ("core", "far"):
            b = split.get(cls) or {}
            values.update({f"h_{cls}": b.get("h"), f"U_{cls}": b.get("U"), f"r_{cls}": b.get("r"),
                           f"z_{cls}": b.get("z"), f"chain_{cls}": b.get("chain")})
        self._jetcore_rows.write("\t".join(_table_cell(values.get(c)) for c in COLUMNS) + "\n")

    def _dt_limiter_of(self, dt: float, m: dict, t: float, focus_courant: bool) -> str:
        """With ``jetcore``: the candidate of the step (``run``) equal to the step ``dt``, its
        first in the order the step is chosen: ``dt_max``, ``focus`` (the collapse's Courant rule),
        ``capillary``, ``t_end``, ``courant-core`` or ``courant-far`` (the split limit, by which
        class binds), ``capillary-tip``, ``tau_end``, ``dt_growth``, ``scale-rate``, ``growth``;
        ``none`` where none is (the step-size floor stops it). Read only, it sets nothing."""
        s, st, r = self.s, self.state, self.reconnect
        pc = self._post_cut_state()
        split = (pc.get("core") or {}).get("split") or {}
        t_cut = self._cut_times()[0]
        cands = [("dt_max", s.dt_max),
                 ("focus", s.courant * m["h_focus"] / m["U"] if focus_courant else math.inf),
                 ("capillary", s.capillary * math.sqrt(m["ell"] ** 3)), ("t_end", s.t_end - t),
                 (f"courant-{split.get('binds', 'core')}", pc["dt_courant"]),
                 ("capillary-tip", pc["dt_capillary"]), ("tau_end", t_cut + r.tau_end - t),
                 ("dt_growth", r.dt_growth * st.dt_prev if st.dt_prev is not None
                  else math.inf)]
        if math.isfinite(st.ell_prev) and st.t_prev < t:
            rate = (st.ell_prev - m["ell"]) / (t - st.t_prev)
            if rate > 0:
                cands.append(("scale-rate", s.scale_rate * m["ell"] / rate))
        if st.dt_prev is not None:
            cands.append(("growth", s.growth * st.dt_prev))
        return next((name for name, v in cands if v == dt), "none")

    def _filled(self) -> dict:
        """What the ring surgeries and the fills of grooves made have filled of the gas since the
        gas cut (the column of the ring cut, the ring at its removal, the gas beyond a groove's
        mouth): the volume, axial momentum and kinetic energy of the liquid each put there (its
        record's ``filled``), summed; zero without one."""
        kinds = EVENT_CLASSES["ring"] + EVENT_CLASSES["gap"]
        rings = [e for e in self._made_events() if e.get("kind") in kinds
                 and isinstance(e.get("filled"), dict)]
        return {"events": len(rings), **{key: sum((float(e["filled"].get(key) or 0.0)
                                                   for e in rings), 0.0)
                                         for key in ("V", "Pz", "KE")}}

    def _removed(self) -> dict:
        """What the events made have removed of the liquid since the gas cut: the volume, axial
        momentum and kinetic energy of each removed head at its cut (its record's ``removed``),
        summed, and the number of events; with ring surgeries or fills of grooves made, net of what
        they filled (:meth:`_filled`, a fill a negative removal, counted among the events), so that
        the fill's own amounts make no jump in the totals of ``budget.txt`` (the projection's change
        of the whole liquid's momentum and the noise of the rebuild still enter them). Signed
        tip-edge corrections are also subtracted, without adding to the topology-event count."""
        liquid = [e for e in self._made_events() if e.get("kind") in EVENT_CLASSES["liquid"]]
        out = {"events": len(liquid), **{key: sum((float((e.get("removed") or {}).get(key) or 0.0)
                                                   for e in liquid), 0.0)
                                         for key in ("V", "Pz", "KE")}}
        filled = self._filled()
        if filled["events"]:
            out = {"events": out["events"] + filled["events"],
                   **{key: out[key] - filled[key] for key in ("V", "Pz", "KE")}}
        if self._tip_edge_balances:
            out = {"events": out["events"],
                   **{key: out[key] - sum(x[key] for x in self._tip_edge_balances)
                      for key in ("V", "Pz", "KE")}}
        return out

    def _open_ring_rows(self):
        """``ring.txt`` of this leg's directory, opened at the ring cut or at a restart in the ring
        stage (a new file per leg, as every table of a leg)."""
        if self._ring_rows is None:
            self._ring_rows = open(self.outdir / "ring.txt", "w", buffering=1)
            self._ring_rows.write("# " + "\t".join(RING_COLUMNS) + "\n")

    def _write_ring_row(self, dt: float):
        """The row of ``ring.txt`` of a step of the ring stage (:data:`RING_COLUMNS`); NaN for
        the ring once it is removed."""
        from .ring import closed, loop_fold_gap, ring_axis_limits

        pc, ring, t = self._post_cut_state(), self._ring(), self._t()
        rr = pc.get("ring_record") or {}
        filled = self._filled()
        V_b = float((ring or {}).get("V_b", math.nan))
        values = {"step": self.state.step, "time": t, **self._taus(t), "dt": dt,
                  "V_filled": filled["V"], "Pz_filled": filled["Pz"], "KE_filled": filled["KE"]}
        if rr:
            from shapely.geometry import LineString

            L = np.column_stack([pc["loops"][0]["r"], pc["loops"][0]["z"]])
            fold = loop_fold_gap(L, ratio=self.s.fold_ratio, stop=self._fold_stop(),
                                 stop_elements=self.s.fold_stop_elements,
                                 search_factor=FOLD_SEARCH)
            opens = [c for c in pc["segs"] if c["open"]]
            values.update({f"ring_{k}": rr[k] for k in ("r_min", "z_r_min", "r_max", "z_min",
                                                         "z_max", "points", "h_min", "h_max",
                                                         "u_max", "courant")})
            values.update(ring_volume=rr["V"], ring_volume_over_V_b=rr["V"] / V_b,
                          ring_area_section=rr["area_section"], ring_centroid_r=rr["r_c"],
                          ring_centroid_z=rr["z_c"], ring_fold_gap=fold.gap,
                          ring_fold_limit=max(self._fold_stop(),
                                              self.s.fold_stop_elements * fold.h_elem)
                          if math.isfinite(fold.gap) else math.nan,
                          ring_axis_limit=max(ring_axis_limits(ring, self.events)),
                          ring_open_gap=float(LineString(closed(L)).distance(LineString(
                              np.column_stack([opens[0]["r"], opens[0]["z"]])))) if opens
                          else math.nan)
            rd = pc["edges"].get("ring") or {}
            values.update(edge_turn_ring=rd.get("turn", math.nan),
                          edge_midside_ring=rd.get("midside", math.nan))
        self._ring_rows.write("\t".join(_table_cell(values.get(c, math.nan))
                                         for c in RING_COLUMNS) + "\n")

    def _write_budget_row(self, values: dict):
        """The row of ``budget.txt`` of a step after the cut, from the values of its row of
        ``chains.txt``: the liquid of the mesh, the removals since the gas cut and their sums."""
        removed = self._removed()
        row = {**{key: values.get(key) for key in ("step", "time", "tau", "tau_p", "volume",
                                                   "momentum_z", "kinetic_energy")},
               "events": removed["events"], "V_removed": removed["V"],
               "Pz_removed": removed["Pz"], "KE_removed": removed["KE"]}
        for key, extra in (("volume", "V"), ("momentum_z", "Pz"), ("kinetic_energy", "KE")):
            row[f"{key}_total"] = float(values.get(key, math.nan)) + removed[extra]
        self._budget.write("\t".join(_table_cell(row.get(c)) for c in BUDGET_COLUMNS) + "\n")

    # --- after a cut with post.log_box_noise: the noise of a cut's box in an ordinary rebuild ---

    def _named_post_cut_chains(self) -> list:
        """The chains of the current state by name (``open``, ``bubble``), as the stops after a cut
        read them (the ends on the axis put on ``r = 0``); empty in a state whose topology has
        changed."""
        segs = self._post_cut_state()["segs"]
        if len(segs) != (1 if self._ring() is not None else 2):
            return []
        out = []
        for c in segs:
            P = np.column_stack([c["r"], c["z"]]).astype(float)
            for k, end in ((0, 0), (-1, 1)):
                if c["end_types"][end] == AXIS:
                    P[k, 0] = 0.0
            out.append(("open" if c["open"] else "bubble", Chain(P, tuple(c["end_types"]))))
        return out

    def _noise_boxes(self) -> list[dict]:
        """The boxes of the log of the noise of a rebuild (``post.log_box_noise``) about the current
        state: one about each tracked neck with a head (``events.liquid_necks``, status ``head``;
        ``neck_<tip>``) at the neck, of the radius its cut would take, ``max(liquid.box_radius
        delta, 2 |z_tip - z_n|)``, the reach of the head along the axis from the neck standing for
        the cut's ``L_rem``; and one about each tip on the axis (``tip_<tip>``) of radius
        ``liquid.box_radius delta``."""
        named = self._named_post_cut_chains()
        rec = self._post_cut_state()["record"]
        if not named or not rec:
            return []
        R0 = self.events.liquid.box_radius * self.reconnect.delta
        out, names = [], set()
        for nk in liquid_necks(named, self.events.liquid):
            if nk["status"] != "head":
                continue
            name = base = f"neck_{event_tip(nk)}"
            n = 2
            while name in names:  # two necks with heads at one tip
                name, n = f"{base}_{n}", n + 1
            names.add(name)
            z_n = float(nk["z_n"])
            out.append({"box": name, "chain": nk["chain"], "z": z_n,
                        "R": max(R0, 2.0 * abs(float(nk["z_tip"]) - z_n))})
        for tip in JET_TIPS:
            z = float(rec.get(f"z_{tip}", math.nan))
            if math.isfinite(z):
                out.append({"box": f"tip_{tip}", "chain": "open" if tip == "cavity_tip"
                            else "bubble", "z": z, "R": R0})
        return out

    def _box_budgets(self, boxes: list[dict]) -> list[dict]:
        """The budgets of each box of ``boxes`` in the current state, as a liquid cut takes them:
        the liquid volume and the area of the interface from the polylines clipped to it
        (``surgery.box_polyline_budgets``) and the axial momentum and the kinetic energy on its
        grid (``surgery.box_grid``, ``surgery.box_integrals``), with the grid's points found in the
        liquid."""
        from .surgery import box_grid, box_integrals, box_polyline_budgets

        chains = [c for _, c in self._named_post_cut_chains()]
        if self._ring() is not None:  # the ring, a hole of the liquid, bounds its gas too
            from .ring import closed

            chains += [Chain(closed(np.column_stack([x["r"], x["z"]])), ("loop", "loop"))
                       for x in self._post_cut_state()["loops"]]
        out = []
        for b in boxes:
            poly = box_polyline_budgets(chains, b["z"], b["R"], self.reconnect.reservoir_depth)
            grid = box_integrals(self.pb, box_grid(b["z"], b["R"]))
            out.append({"V": poly["V_liquid"], "A": poly["A"], "Pz": grid["Pz"],
                        "KE": grid["KE"], "points": grid["points"]})
        return out

    def _box_noise_before(self) -> dict | None:
        """Before an ordinary rebuild after the cut, with ``post.log_box_noise``: the boxes of the
        log about the state before it (:meth:`_noise_boxes`) and their budgets then; None
        otherwise, reading nothing."""
        if self._noise is None or self._post() is None:
            return None
        boxes = self._noise_boxes()
        return {"boxes": boxes, "before": self._box_budgets(boxes)}

    def _box_noise_after(self, noise: dict | None, reason: str):
        """After the rebuild before which ``noise`` was read (:meth:`_box_noise_before`; ``reason``
        ``rule`` or ``distortion``): the budgets of the same boxes on the new mesh, and a row of
        ``box_noise.txt`` per box with the change the rebuild made, the noise of the box from which
        ``liquid.sigma_V``, ``sigma_Pz`` and ``sigma_KE`` are set (the volume, the momentum and the
        energy; the area besides), kept for the run's result (:meth:`_box_noise_summary`)."""
        if noise is None:
            return
        after = self._box_budgets(noise["boxes"])
        stats, t = self._noise_stats, self._t()
        stats["rebuilds"] += 1
        for b, x, y in zip(noise["boxes"], noise["before"], after):
            row = {"step": self.state.step, "time": t, **self._taus(t), "reason": reason,
                   "remesh_count": self.pb.remesh_count, **b, "points_before": x["points"],
                   "points_after": y["points"]}
            for key in BOX_NOISE_KEYS:
                d = float(y[key]) - float(x[key])
                row.update({f"{key}_before": x[key], f"{key}_after": y[key], f"d{key}": d})
                if math.isfinite(d):
                    stats["max_abs"][key] = max(stats["max_abs"][key], abs(d))
                    stats["sum_sq"][key] += d * d
                    stats["count"][key] += 1
            stats["rows"] += 1
            self._noise.write("\t".join(_table_cell(row.get(c)) for c in BOX_NOISE_COLUMNS)
                              + "\n")

    def _gate_before(self) -> dict | None:
        """With ``jetcore.box_gate``, before an ordinary rebuild after the cut (the rule's or one
        for distortion): the box about the core's base (on the axis at the height of the jet's base,
        ``jetcore.core_mask``'s ``A_z``, or of the cavity's tip without one; radius
        ``liquid.box_radius delta``), its liquid volume and axial momentum as a liquid cut takes
        them (:meth:`_box_budgets`), and the state before the rebuild, dumped
        (``states/prebuild.dump``, outside the rotation); None otherwise."""
        jet = self._jetcore()
        if jet is None or not jet.box_gate:
            return None
        pc = self._post_cut_state()
        core, rec = pc.get("core") or {}, pc.get("record") or {}
        z = core.get("A_z", math.nan)
        if not math.isfinite(z):
            z = float(rec.get("z_cavity_tip", math.nan))
        if not math.isfinite(z):
            return None
        box = {"box": "core", "chain": "open", "z": float(z),
               "R": self.events.liquid.box_radius * self.reconnect.delta}
        path = self.outdir / "states" / "prebuild.dump"
        self._write_state(path, kept=True, box_gate={"box": box})
        return {"box": box, "before": self._box_budgets([box])[0], "dump": path}

    def _gate_after(self, gate: dict | None, reason: str) -> dict:
        """After the rebuild of :meth:`_gate_before`: the change of the box's volume and momentum
        against ``box_gate_sigmas`` times ``liquid.sigma_V`` and ``liquid.sigma_Pz`` (``box_gate``,
        for the ``remesh`` event; empty without the gate). Within both, the dump of the state before
        it is deleted; beyond either, it is kept as ``states/gate_<step>.dump`` and the run stops on
        it (``box-gate``, the stop of the next check, :meth:`_stop`), the state the rebuild left
        not accepted."""
        if gate is None:
            return {}
        after = self._box_budgets([gate["box"]])[0]
        jet, ls = self._jetcore(), self.events.liquid
        dV = float(after["V"]) - float(gate["before"]["V"])
        dPz = float(after["Pz"]) - float(gate["before"]["Pz"])
        lim_V, lim_Pz = jet.box_gate_sigmas * ls.sigma_V, jet.box_gate_sigmas * ls.sigma_Pz
        reading = {"reason": reason, **gate["box"], "dV": dV, "dPz": dPz, "limit_V": lim_V,
                   "limit_Pz": lim_Pz, "points_before": gate["before"]["points"],
                   "points_after": after["points"],
                   "passed": bool(abs(dV) <= lim_V and abs(dPz) <= lim_Pz)}
        dump = gate["dump"]
        if reading["passed"]:
            for f in (dump, dump.with_suffix(".json")):
                f.unlink(missing_ok=True)
            return {"box_gate": reading}
        kept = dump.with_name(f"gate_{self.state.step:08d}.dump")
        for f, g in ((dump, kept), (dump.with_suffix(".json"), kept.with_suffix(".json"))):
            f.replace(g)
        reading["dump"] = str(kept)
        self._gate_stop = {"reason": "box-gate", "trigger": reason,
                           **{k: v for k, v in reading.items() if k != "reason"},
                           "precut": str(kept), **self._taus(self._t())}
        return {"box_gate": reading}

    def _box_noise_summary(self) -> dict | None:
        """The log of the noise of the box (``post.log_box_noise``) in the run's result: the
        rebuilds and the rows logged, and the largest and the rms change of each budget over the
        rows (``max_abs``, ``rms``, in the solver's units); None without the log."""
        s = self._noise_stats
        if s is None:
            return None
        return {"rebuilds": s["rebuilds"], "rows": s["rows"], "max_abs": dict(s["max_abs"]),
                "rms": {k: math.sqrt(s["sum_sq"][k] / s["count"][k]) if s["count"][k] else None
                        for k in BOX_NOISE_KEYS}}

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
        self._write_ring_profile(path, header)
        st.snapshots += 1
        st.last_snapshot_t = m["time"]
        st.last_snapshot_ell = m["ell"]

    def _write_ring_profile(self, path: Path, header: dict, loops=None):
        """In the ring stage: beside the profile ``path`` of the open chain, the ring's
        (``<stem>_ring.txt``, header ``chain`` ``ring``), its closed polyline from its seam
        (``ring.closed``), instead of a bubble's; nothing before the ring or once it is removed."""
        if self._ring() is None:
            return
        if loops is None:
            loops = self._post_cut_state().get("loops") or []
        for x in loops[:1]:
            P = np.column_stack([x[k] for k in ("r", "z", "u_r", "u_z")])
            np.savetxt(path.with_name(path.stem + "_ring.txt"), np.vstack([P, P[:1]]),
                       header=json.dumps(dict(header, chain="ring")) + "\nr z u_r u_z",
                       comments="# ")

    def _maybe_snapshot(self, m: dict, forced: str | None = None):
        """A profile where one is due: ``forced`` (its reason), every ``snapshot_dt`` (``time``),
        when the local scale has changed by ``snapshot_decades`` (``scale``), and after the cut with
        ``jetcore.profile_dt`` at least every ``profile_dt`` since the last profile (``window``)."""
        st = self.state
        jet = self._jetcore()
        if forced is not None:
            self._snapshot(m, forced)
        elif self._t() - st.last_snapshot_t >= self.s.snapshot_dt:
            self._snapshot(m, "time")
        elif (math.isfinite(st.last_snapshot_ell) and m["ell"] > 0
              and abs(math.log10(m["ell"] / st.last_snapshot_ell)) >= self.s.snapshot_decades):
            self._snapshot(m, "scale")
        elif jet is not None and jet.profile_dt is not None \
                and self._t() - st.last_snapshot_t >= jet.profile_dt:
            self._snapshot(m, "window")

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
        jet = self._jetcore()
        far_ratio = (jet.far_refine_ratio if jet is not None and jet.far_refine_ratio is not None
                     else s.refine_ratio)
        refine = [g for g in regions if math.isfinite(g["h_max"]) and g["h_max"] > (
            far_ratio if g.get("class") == "far" else s.refine_ratio) * g["target"]]
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
        noise = self._box_noise_before()  # post.log_box_noise only
        gate = self._gate_before()  # jetcore.box_gate only
        self.pb.force_remesh()
        self._box_noise_after(noise, "rule")
        gated = self._gate_after(gate, "rule")
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
                    **self._axis_repair_note(), **gated)
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
        noise = self._box_noise_before()  # after a cut with post.log_box_noise only
        gate = self._gate_before()  # jetcore.box_gate only
        self.pb.force_remesh()
        self._box_noise_after(noise, "distortion")
        gated = self._gate_after(gate, "distortion")
        self._distortion_remeshes += 1
        after = self.measure()
        self._event("remesh", m, reason="distortion", **flags[0], flagged_substeps=len(flags),
                    focus_size=new.size, focus_radius=new.radius, focus_size_before=old.size,
                    focus_radius_before=old.radius, h_near_before=m["h_near"],
                    h_near_after=after["h_near"], **extra, **self._axis_repair_note(), **gated)
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
                if self._ring() is not None:  # the ring stage: its record, restored with it
                    meta["ring"] = self._ring()
                if self._jetcore() is not None:  # with jetcore only: other dumps keep their keys
                    meta["far_yields"] = st.far_yields
                if self.events.pinch.enabled:  # with pinch only
                    meta["neck_history"] = json.loads(json.dumps(st.neck_history))
                if st.lip_samples:  # the ring's lip read: other dumps keep their keys
                    meta["lip_samples"] = [[float(x) for x in v] for v in st.lip_samples]
                if st.fold_retry is not None:  # a fold retried (batch B): other dumps keep keys
                    meta["fold_retry"] = dict(st.fold_retry)
                if st.fold_cap is not None:
                    meta["fold_cap"] = dict(st.fold_cap)
        if self._tip_edge_balances:
            meta["tip_edge_balances"] = [dict(x) for x in self._tip_edge_balances]
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
        self._prune_meshes()  # with the events block only

    def _live_mesh_files(self) -> set:
        """The mesh files the problem's mesh templates were built from (pyoomph's ``_meshfile``,
        which the next dump refers to), resolved."""
        out = set()
        templates = list(getattr(self.pb, "_meshtemplate_list", None) or [])
        active = getattr(self.pb, "active_mesh_template", None)
        if active is not None:
            templates.append(active())
        for t in templates:
            get = getattr(t, "_get_template", None)
            for x in (t, get() if get is not None else None):
                f = getattr(x, "_meshfile", None)
                if f:
                    out.add(Path(f).resolve())
        return out

    def _prune_meshes(self) -> int:
        """After a cut with the events block and ``post.prune_gmsh``: delete the meshes gmsh wrote
        for the run (``_gmsh`` of the problem's output directory, each ``.msh`` with the files of
        its stem) that no dump refers to (a dump refers to the ``.msh`` its mesh was built from,
        pyoomph's ``get_state_file_mesh_files``: every dump in the output directory and in its
        subdirectories, and in the driver's ``states``) and that the problem does not hold
        (:meth:`_live_mesh_files`), so that a restart from any dump kept still finds its mesh.
        Nothing is deleted where the mesh the problem holds or any dump cannot be read. Returns the
        files deleted."""
        if self.events is None or not self.events.post.prune_gmsh or self._cut_record() is None:
            return 0
        where = getattr(self.pb, "get_output_directory", None)
        if where is None:  # no finite-element problem
            return 0
        root = Path(where())
        gmsh = root / "_gmsh"
        live = self._live_mesh_files()
        if not gmsh.is_dir() or not live:
            return 0
        from pyoomph.output.states import get_state_file_mesh_files

        kept = set(live)
        dumps = set(root.glob("*.dump")) | set(root.glob("*/*.dump")) | set(
            (self.outdir / "states").glob("*.dump"))
        for dump in dumps:
            try:
                kept.update(Path(f).resolve() for f in get_state_file_mesh_files(str(dump)))
            except (OSError, RuntimeError, ValueError):
                return 0
        removed = 0
        for msh in sorted(gmsh.glob("*.msh")):
            if msh.resolve() in kept:
                continue
            for f in sorted(gmsh.glob(glob.escape(msh.stem) + ".*")):
                if f != msh:
                    f.unlink(missing_ok=True)
                    removed += 1
            msh.unlink(missing_ok=True)  # last: a stem with its .msh is pruned again if need be
            removed += 1
        self._meshes_pruned += removed
        return removed

    def restore(self, meta: dict):
        """Resume the driver bookkeeping from a state dump's metadata."""
        st = self.state
        self._tip_edge_balances = [dict(x) for x in meta.get("tip_edge_balances", [])]
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
            st.far_yields = int(meta.get("far_yields", 0))
            st.neck_history = json.loads(json.dumps(meta.get("neck_history") or {}))
            st.lip_samples = [[float(x) for x in v] for v in meta.get("lip_samples") or []]
            st.fold_retry = dict(meta["fold_retry"]) if meta.get("fold_retry") else None
            st.fold_cap = dict(meta["fold_cap"]) if meta.get("fold_cap") else None
            if self.events is not None:
                self.pb.events = [dict(x) for x in meta.get("events") or []]
                st.legs = [dict(x) for x in meta.get("legs") or []]
                if meta.get("ring") is not None:
                    # the ring stage: its record and the ring's sizing on the template the loaded
                    # mesh is rebuilt from, before the state is measured
                    from .ringsurgery import restore_ring

                    self.pb.ring = json.loads(json.dumps(meta["ring"]))
                    restore_ring(self.pb, self.pb.ring)
                    self._post_cut_cache = self._stop_cache = None
                    self._open_ring_rows()
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
        self._stop_cache = None
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
        if meta.get("ring") is not None and not self.events.ring.enabled:
            # the class rule alone would admit it from a later precut_e<k> of the ring stage
            raise ValueError("a state of the ring stage (its dump holds ring) is continued only "
                             "under an events block with ring.enabled")
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
        and floors it runs with (``limits``, :meth:`_leg_limits`), the events made before it
        (``events_made``), and, from the state before an event, that event (``from_event_state``)
        and, where keys of a class of events change, the event from which they apply anew
        (``event_keys_changed_at``). A leg restarted from a dump writes the records of the events
        made before it into its own directory (:meth:`_copy_event_records`)."""
        st = self.state
        minimum_dt = self._lower_minimum_dt()
        leg = {"leg": len(st.legs), "start": start, "step": st.step, "time": m["time"],
               **self._taus(m["time"]), **self.leg_info, "settings_sha256": self.events.sha256(),
               "changed": [], "added": False, "minimum_dt": minimum_dt,
               # oomph-lib's smallest step in units of the spacing of doubles at the leg's time:
               # a step of a few of them no longer advances the time by itself
               "minimum_dt_ulps": _in_ulps(minimum_dt, m["time"]), "limits": self._leg_limits(),
               "events_made": len(self._made_events())}
        if meta is not None:
            recorded = meta.get("events_settings")
            leg["added"] = recorded is None
            leg["changed"] = self.events.changed_keys(recorded)  # without a record: the defaults
            if meta.get("cut_stage") == PRECUT_EVENT:
                k = (meta.get("event") or {}).get("k")
                leg["from_event_state"] = k
                if any(EventSettings.CLASS_OF_PART.get(c.split(".", 1)[0]) is not None
                       for c in leg["changed"]):
                    leg["event_keys_changed_at"] = k
        st.legs.append(leg)
        self._event("leg", m, **{k: v for k, v in leg.items() if k not in ("time", "step")})
        if meta is not None:
            self._copy_event_records()

    def _event_file(self, k) -> Path:
        """The file of the record of event ``k`` in this leg's directory."""
        return self.outdir / "events" / f"event_{int(k):03d}.json"

    def _write_event_copy(self, rec: dict, path: Path):
        """The record of an event made in an earlier leg, as this leg holds it, written into this
        leg's directory (``copy``: this leg and the leg that made it, ``of_leg``); the ghost's
        polygon is only in the file of the leg that made it."""
        full = dict(_plain(rec), copy={"leg": len(self.state.legs) - 1, "of_leg": rec.get("leg")})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(full, indent=1) + "\n")
        tmp.replace(path)

    def _copy_event_records(self):
        """At the start of a leg restarted from a dump: the record of each event made before it, as
        the dump holds it (the ghost's recontacts and the dumps deleted until then), into this
        leg's ``events/`` (:meth:`_write_event_copy`), so that the directory of every leg has the
        record of every event of its run up to its end; a file there already (a leg in the
        directory of the one before it) is kept."""
        for e in self._made_events():
            path = self._event_file(e["k"])
            if not path.is_file():
                self._write_event_copy(e, path)

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
        3. The driver's bookkeeping of the cut made (:meth:`_on_event_made`, which commits the
           jets' bases and the kink regions of the fresh mesh, :meth:`_commit_post_cut`, before the
           state after the event is kept) and its row of ``events.txt``.

        With the block's ``ring.enabled`` the ring's handlers (``events.RING_HANDLERS``) are made
        too (:meth:`_meet_ring`), and with ``groove.enabled`` the fill of a closing gas groove
        (``events.GROOVE_CLASSES``, :meth:`_meet_groove`), but not on a far head: with
        ``pinch.far_head`` an event without a handler the driver makes whose site lies on the head
        of a tracked neck in the far region (:meth:`_far_head`; a gas neck, a closing groove, a
        sheet's contact; the head may be curled, ``events.head_sides``) is met by the far-head cut
        of that head first, before any fill (decision 22: the heads are nuisances, the cut's
        volume and momentum hard checks), the fill made only off the far heads. In the ring stage
        the liquid cut of the open chain is made as before, through the ring stage's own rebuild
        and transfer by projection, which keep the ring (``surgery.liquid_cut``; its record
        ``stage`` ``ring``), not through pyoomph's parked plan, whose open chart cannot carry the
        ring's loop."""
        if request.get("handler") in RING_HANDLERS and self.events.ring.enabled:
            return self._meet_ring(m, request)
        if request.get("kind") == SHEET_CONTACT and request.get("chain") == "ring" \
                and (request.get("sheet_rupture") or {}).get("due") and self._sheet_handled():
            return self._meet_sheet_rupture(m, request)
        far = self._far_head(request) if request.get("handler") not in IMPLEMENTED_HANDLERS \
            else None
        if far is None and self.events.groove.enabled and \
                request.get("class") in GROOVE_CLASSES and request.get("kind") != SHEET_CONTACT:
            return self._meet_groove(m, request)
        if request.get("handler") not in IMPLEMENTED_HANDLERS:
            if far is None:
                return m, self._unhandled_event(request)
            self._event("far-head", m, far_kind=request.get("kind"),
                        far_class=request.get("class"), chain=far.get("chain"),
                        r_n=far.get("r_n"), s=far.get("s"), z_n=far.get("z_n"))
            request = far
        elif request.get("unresolved") is not None:  # an unresolved neck on a far head
            far_event = request.get("far_event") or {}
            self._event("far-head", m, far_kind=far_event.get("kind"), far_class=None,
                        chain=request.get("chain"), r_n=request.get("r_n"), s=request.get("s"),
                        z_n=request.get("z_n"))
        return self._liquid_cut_event(m, request)

    def _liquid_cut_event(self, m: dict, request: dict) -> tuple[dict, dict | None]:
        """Steps 2-3 of :meth:`_meet_event` for a request the liquid cut makes (a landed neck or a
        far head): the event after ``max_events`` made, the cascade, the cut and its
        bookkeeping."""
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
            check = {"base_check": self._head_fold_check(m)} \
                if request.get("head_fold") or request.get("unresolved") is not None else {}
            made = liquid_cut(self.pb, self.events, r, self.outdir, request, k=k,
                              dt_prev=dt_prev, elements_per_scale=eps, h_floor=self._h_floor(),
                              driver=self, on_made=self._on_event_made, **check)
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
        self._drop_event_dumps()
        self._prune_meshes()
        return m, None

    def _head_fold_check(self, m: dict):
        """The check of the upward jet's base for the cut of its head folding onto the core
        (``events.far_head_request``'s ``head_fold``, ``surgery.liquid_cut``'s ``base_check``):
        a callable that measures the fresh mesh and compares the base (``r_j``, ``z_b``, ``V_j``,
        ``Q_j`` of the open chain's jet) with its reading ``m`` before the cut, ``ok`` where
        ``r_j``, ``V_j`` and ``Q_j`` change by at most ``HEAD_FOLD_BASE_TOL`` relative and
        ``z_b`` by at most that times ``r_j`` (decision 22: the base, its walls and the stem stay
        as resolved)."""
        def check() -> dict:
            return base_guard(m, self.measure())

        return check

    def _far_head(self, request: dict) -> dict | None:
        """With ``pinch.far_head`` (and ``jetcore``): the request of the cut of the far head an
        event without a handler lies on (``events.far_head_request``, from the tracked necks of
        the state and the core of its open chain); None otherwise."""
        from .events import far_head_request

        if self.events is None or not self.events.pinch.far_head or self._jetcore() is None:
            return None
        named = self._named_post_cut_chains()
        if not named:
            return None
        core = self._post_cut_state().get("core")
        mask = None if core is None or core["whole"] else core["mask"]
        return far_head_request(request, liquid_necks(named, self.events.liquid), mask)

    def _unresolved_neck_request(self, stop: dict | None) -> dict | None:
        """With ``pinch.far_head`` (and ``jetcore``): for the stop ``liquid-neck-unresolved`` of a
        freshly built mesh whose neck has a far head, the request of the cut of that head at this
        neck (``events.unresolved_neck_request``, from the tracked necks of the state with their
        pinch readings and the core of its open chain), which :meth:`_meet_event` makes with the
        check of the upward jet's base; None otherwise, the stop staying (decision 22: the far
        heads are nuisances, the cut's volume and momentum hard checks)."""
        from .events import LIQUID_NECK_UNRESOLVED, closing_readings, unresolved_neck_request

        if stop is None or stop.get("reason") != LIQUID_NECK_UNRESOLVED:
            return None
        if self.events is None or not self.events.pinch.far_head or self._jetcore() is None:
            return None
        named = self._named_post_cut_chains()
        if not named:
            return None
        necks = liquid_necks(named, self.events.liquid)
        closing_readings(necks, self.events, self.reconnect.delta, self._pinch_context())
        core = self._post_cut_state().get("core")
        mask = None if core is None or core["whole"] else core["mask"]
        return unresolved_neck_request(stop, necks, mask)

    def _meet_ring(self, m: dict, request: dict) -> tuple[dict, dict | None]:
        """Meet a request of the ring's handlers (``ring``: the piercing, class ``W1``;
        ``ring-removal``: the ring emptied, class ``W2``), with the block's ``ring.enabled``: the
        event after ``max_events`` made is a resumable stop (``max-events``); the surgery
        (``ringsurgery.ring_cut``, ``ringsurgery.ring_removal``), with no test of a cascade (no
        tip of the axis makes them), the run's last step as the history's: a refusal, which leaves
        the problem unchanged, is a resumable stop (``ring-cut-refused``,
        ``ring-removal-refused``), and so are hard budgets that do not close (``event-unclosed``);
        any other failure ends the run as a failed one, raised. Made: its row of ``events.txt``,
        the dumps the policy drops, the meshes pruned; after the removal with
        ``ring.after_removal`` ``stop`` the window ends (``bubble-vanishing``, ``window_end``,
        ``ring_removed``), otherwise the run goes on from the fresh mesh."""
        if len(self._made_events()) >= self.events.max_events:
            return m, self._event_stop(EVENT_LIMIT, request, max_events=self.events.max_events)
        from .ringsurgery import ring_cut, ring_removal
        from .surgery import BudgetUnclosed, CutRefused

        st, r = self.state, self.reconnect
        k = len(self._made_events()) + 1
        eps = self.s.elements_per_scale if r.elements_per_scale is None else r.elements_per_scale
        dt_prev = st.dt_prev if st.dt_prev else float(self._cut_record()["settings"]["dt0"])
        removal = request.get("handler") == "ring-removal"
        surgery = ring_removal if removal else ring_cut
        self._event_after = None
        try:
            made = surgery(self.pb, self.events, r, self.outdir, request, k=k, dt_prev=dt_prev,
                           elements_per_scale=eps, h_floor=self._h_floor(), driver=self,
                           on_made=self._on_ring_made)
        except (CutRefused, BudgetUnclosed) as e:
            rec = getattr(e, "record", None) or {}
            reason = (EVENT_UNCLOSED if isinstance(e, BudgetUnclosed) else
                      RING_REMOVAL_REFUSED if removal else RING_CUT_REFUSED)
            extra = {"hard": (rec.get("closures") or {}).get("hard")} if reason == \
                EVENT_UNCLOSED else {}
            return m, self._event_stop(reason, request, precut=e.precut,
                                       failed_at=rec.get("failed_at"),
                                       error=rec.get("error", f"{type(e).__name__}: {e}"),
                                       record=f"events/event_{k:03d}.json", **extra)
        m = self._event_after
        self._write_ring_event_row(made)
        self._drop_event_dumps()
        self._prune_meshes()
        if removal and self.events.ring.after_removal == "stop":
            return m, {"reason": "bubble-vanishing", "window_end": True, "ring_removed": k,
                       "bubble_volume": (made.get("filled") or {}).get("V"),
                       **self._taus(m["time"])}
        return m, None

    def _meet_groove(self, m: dict, request: dict) -> tuple[dict, dict | None]:
        """Meet a closing gas groove (class ``A-groove`` or ``RF``) with the block's
        ``groove.enabled``: the event after ``max_events`` made is a resumable stop
        (``max-events``); the fill (``groovesurgery.groove_fill``), with no test of a cascade (no
        tip of the axis makes it), the run's last step as the history's, the stops' fold limits,
        and with ``jetcore`` the mask of the open chain's core, whose nodes no fill takes: a
        refusal, which leaves the problem unchanged, is a resumable stop (``groove-fill-refused``),
        and so are hard budgets that do not close (``event-unclosed``); any other failure ends the
        run as a failed one, raised. Made: its row of ``events.txt``, the dumps the policy drops,
        the meshes pruned, and the run goes on from the fresh mesh. A groove on a far head never
        comes here: :meth:`_meet_event` cuts that head first."""
        if len(self._made_events()) >= self.events.max_events:
            return m, self._event_stop(EVENT_LIMIT, request, max_events=self.events.max_events)
        from .groovesurgery import groove_fill
        from .surgery import BudgetUnclosed, CutRefused

        st, r, s = self.state, self.reconnect, self.s
        k = len(self._made_events()) + 1
        eps = s.elements_per_scale if r.elements_per_scale is None else r.elements_per_scale
        dt_prev = st.dt_prev if st.dt_prev else float(self._cut_record()["settings"]["dt0"])
        core = self._post_cut_state().get("core")
        fold = {"ratio": s.fold_ratio, "stop": self._fold_stop(),
                "stop_elements": s.fold_stop_elements}
        floor = request.get("floor")
        local_fill = floor or request.get("spline_fill") or request.get("step_fill")
        base_check = None
        if local_fill is not None:  # the local gas wedge's own fold limits
            fold = dict(local_fill["fold"])

            def base_check():
                return base_guard_checked(m, self.measure())
        self._event_after = None
        try:
            made = groove_fill(self.pb, self.events, r, self.outdir, request, k=k,
                               dt_prev=dt_prev, elements_per_scale=eps, h_floor=self._h_floor(),
                               fold=fold, core=None if core is None else core["mask"],
                               driver=self, on_made=self._on_groove_made,
                               **({"base_check": base_check} if base_check is not None else {}))
        except (CutRefused, BudgetUnclosed) as e:
            rec = getattr(e, "record", None) or {}
            reason = EVENT_UNCLOSED if isinstance(e, BudgetUnclosed) else GROOVE_FILL_REFUSED
            extra = {"hard": (rec.get("closures") or {}).get("hard")} if reason == \
                EVENT_UNCLOSED else {}
            return m, self._event_stop(reason, request, precut=e.precut,
                                       failed_at=rec.get("failed_at"),
                                       error=rec.get("error", f"{type(e).__name__}: {e}"),
                                       record=f"events/event_{k:03d}.json", **extra)
        m = self._event_after
        if floor is not None:  # countable: an event of its own besides the fill's
            self._event("floor-fill", m, k=k, guard=floor["guard"], wedge=floor["wedge"],
                        V_fill=(made.get("filled") or {}).get("V"),
                        base=made.get("base_guard"))
        elif local_fill is not None:
            kind = "step-contact-fill" if request.get("step_fill") is not None else "spline-fill"
            self._event(kind, m, k=k, guard=local_fill["guard"], wedge=local_fill["wedge"],
                        V_fill=(made.get("filled") or {}).get("V"), base=made.get("base_guard"))
        self._write_ring_event_row(made)
        self._drop_event_dumps()
        self._prune_meshes()
        return m, None

    def _meet_sheet_rupture(self, m: dict, request: dict) -> tuple[dict, dict | None]:
        """Meet the ring's lip thinning at ``h_s`` or below (decision 22; the request's
        ``sheet_rupture`` due): the event after ``max_events`` made is a resumable stop
        (``max-events``); the rupture (``sheetsurgery.sheet_rupture``) with the run's last step as
        the history's, the lip's reading and a check of the upward jet's base (the hard closure
        ``S_base``): a refusal, which leaves the problem unchanged, is a resumable stop
        (``sheet-rupture-refused``), and so are hard budgets that do not close
        (``event-unclosed``); any other failure ends the run as a failed one, raised. Made: the
        lip's history cleared before the state after it is kept (``on_made``; the history in the
        record's ``lip`` and in ``precut_e<k>``), so that a restart from ``postcut_e<k>`` goes on
        as the run does; an event ``sheet-rupture``, its row of ``events.txt``, the dumps the
        policy drops, the meshes pruned, and the run goes on from the fresh mesh."""
        if len(self._made_events()) >= self.events.max_events:
            return m, self._event_stop(EVENT_LIMIT, request, max_events=self.events.max_events)
        from .sheetsurgery import sheet_rupture
        from .surgery import BudgetUnclosed, CutRefused

        st, r = self.state, self.reconnect
        k = len(self._made_events()) + 1
        dt_prev = st.dt_prev if st.dt_prev else float(self._cut_record()["settings"]["dt0"])
        lip = {"samples": [list(x) for x in st.lip_samples], **(request.get("sheet_rupture") or {})}

        def base_check():
            return base_guard_checked(m, self.measure())

        def on_made(rec):  # before the state after the rupture is kept (postcut_e<k>)
            self._on_groove_made(rec)
            # the lip's history ends with it: the state after it starts with none, as the live
            # run goes on (the history kept in the record's lip and in precut_e<k>)
            st.lip_samples = []

        self._event_after = None
        try:
            made = sheet_rupture(self.pb, self.events, r, self.outdir, request, k=k,
                                 dt_prev=dt_prev, h_floor=self._h_floor(), lip=lip, driver=self,
                                 on_made=on_made, base_check=base_check)
        except (CutRefused, BudgetUnclosed) as e:
            rec = getattr(e, "record", None) or {}
            reason = EVENT_UNCLOSED if isinstance(e, BudgetUnclosed) else SHEET_RUPTURE_REFUSED
            extra = {"hard": (rec.get("closures") or {}).get("hard")} if reason == \
                EVENT_UNCLOSED else {}
            return m, self._event_stop(reason, request, precut=e.precut,
                                       failed_at=rec.get("failed_at"),
                                       error=rec.get("error", f"{type(e).__name__}: {e}"),
                                       record=f"events/event_{k:03d}.json", **extra)
        m = self._event_after
        removed = made.get("removed") or {}
        self._event("sheet-rupture", m, k=k, w=(made.get("geometry") or {}).get("w"),
                    V=removed.get("V"), Pz=removed.get("Pz"), KE=removed.get("KE"),
                    centroid=removed.get("centroid"), base=made.get("base_guard"))
        self._write_ring_event_row(made)
        self._drop_event_dumps()
        self._prune_meshes()
        return m, None

    def _on_groove_made(self, rec: dict):
        """The driver's bookkeeping of a fill made (``groovesurgery.groove_fill``'s ``on_made``,
        before the state after it is kept and its record written), into its record: the event's
        place in the run (``step``, ``leg``, ``tau_over_delta2``, ``t_prime_over_T``, the source
        ``commit``, ``dt.before``); then :meth:`reset_after_groove` and the commit of the jets'
        bases and the kink regions of the fresh mesh (:meth:`_commit_post_cut`), which the state
        after the event, kept next, holds as the run steps from it."""
        st, delta = self.state, self.reconnect.delta
        t_cut, t_p = self._cut_times()
        T, tau_p = t_p - t_cut, rec.get("tau_p")
        rec.update(step=st.step, leg=len(st.legs) - 1,
                   tau_over_delta2=_finite(float(rec["tau"]) / delta**2),
                   t_prime_over_T=(_finite(float(tau_p) / T) if tau_p is not None
                                   and math.isfinite(T) and T != 0.0 else None),
                   commit=self.leg_info.get("commit"))
        rec.setdefault("dt", {})["before"] = _finite(st.dt_prev)
        m_new = self.reset_after_groove(rec)
        self._commit_post_cut(m_new)
        self._event_after = m_new

    def reset_after_groove(self, record: dict) -> dict:
        """Restart the bookkeeping that the fill of a groove invalidates (``record`` its record)
        and measure the fresh mesh, as :meth:`reset_after_event` does for a liquid cut: no new
        cycle, the births, series, holds and last step kept; the growth limit of the local scale
        and the rate limit of the step restart, the kink regions of the chain filled are dropped
        and so is the apex history of the jet whose base the groove straddled, and the remesh
        check of the step is not made on the fresh mesh (``_checks_pending`` cleared, so that the
        state after the event is kept without ``resume_at``); the event ``groove-fill`` is
        recorded and a profile written (reason ``groove-fill``)."""
        st = self.state
        st.ell_prev = st.t_prev = math.nan
        chain = record.get("chain")
        st.kinks.pop(chain, None)
        st.kinks.pop(f"{chain}_far", None)  # jetcore's far kink of the chain
        jet = (record.get("request") or {}).get("jet")
        if jet is not None:
            st.apex_history.pop(jet, None)
        self._checks_pending = False
        self._post_cut_cache = self._stop_cache = None
        m = self.measure()
        k = record.get("k")
        filled = record.get("filled") or {}
        self._event("groove-fill", m, k=k, kind=record.get("kind"),
                    **{"class": record.get("class")}, chain=chain, V_fill=filled.get("V"),
                    Pz_fill=filled.get("Pz"), KE_fill=filled.get("KE"),
                    mouth=(record.get("geometry") or {}).get("mouth"),
                    record=f"events/event_{int(k):03d}.json" if k is not None else None)
        self._track_tips(m, snapshot=False)
        self._snapshot(m, "groove-fill")
        return m

    def _write_ring_event_row(self, rec: dict):
        """The row of ``events.txt`` of a ring surgery made, or of the fill of a groove: its chain
        (``ring``; the chain filled for a fill), its fill as a negative removal (``V_rem``,
        ``Pz_rem``, ``KE_rem``; a lip's rupture its removal), its closures R1-R6 (Q1-Q6 for the
        removal, G1-G6 for a fill, S1-S6 for a lip's rupture)
        relative to their scales in ``F1``-``F6``, the cutoff identity, the transfer's counts, the
        step of the history and its wall time; every other column empty."""
        filled = rec.get("filled") or {}
        closures = rec.get("closures") or {}
        rel = closures.get("relative") or {}
        groove = rec.get("kind") in EVENT_CLASSES["gap"]
        prefix = ("Q" if rec.get("kind") == "ring-collapse" else "G" if groove
                  else "S" if rec.get("kind") == "sheet-rupture" else "R")
        phi = closures.get(f"{prefix}3_phi") or {}
        tr = rec.get("transfer") or {}
        neg = {key: (-float(filled[x]) if filled.get(x) is not None else None)
               for key, x in (("V_rem", "V"), ("Pz_rem", "Pz"), ("KE_rem", "KE"))}
        values = {"k": rec.get("k"), "step": rec.get("step"), "time": rec.get("t"),
                  "tau": rec.get("tau"), "tau_p": rec.get("tau_p"),
                  "tau_over_delta2": rec.get("tau_over_delta2"), "kind": rec.get("kind"),
                  "class": rec.get("class"), "chain": rec.get("chain") if groove else "ring",
                  **neg,
                  **{f"F{i}": rel.get(f"{prefix}{i}") for i in range(1, 7)},
                  "phi_relative_error": phi.get("phi_relative_error"),
                  "predicted_unlocated": tr.get("predicted_unlocated"),
                  "reported_unlocated": tr.get("reported_unlocated"),
                  "dt_history": (rec.get("dt") or {}).get("history"),
                  "wall": rec.get("wall_seconds")}
        self._event_rows.write("\t".join(_table_cell(values.get(c)) for c in EVENT_COLUMNS)
                               + "\n")

    def _drop_event_dumps(self) -> list[str]:
        """Under ``post.keep_event_dumps`` ``first<N>-last<M>`` (``events.event_dumps_kept``,
        over the events of every leg): delete the dumps of the events made that the policy no
        longer keeps, of those this run wrote (``states/precut_e<k>``, ``postcut_e<k>``), noting
        them in the event's record (``dumps.dropped``), in the one the dumps keep and in its file.
        Returns the dumps deleted."""
        policy = event_dump_policy(self.events.post.keep_event_dumps)
        if policy is None:
            return []
        made = self._made_events()
        kept = event_dumps_kept([e["k"] for e in made], policy)
        out = []
        for e in made:
            k = int(e["k"])
            dropped = False
            for stage in ("precut", "postcut"):
                path = self.outdir / "states" / f"{stage}_e{k:03d}.dump"
                if stage in kept.get(k, ()) or not path.is_file():
                    continue
                for f in (path, path.with_suffix(".json")):
                    f.unlink(missing_ok=True)
                dumps = e.setdefault("dumps", {})
                dumps["dropped"] = sorted(set(dumps.get("dropped") or []) | {stage})
                out.append(str(path))
                dropped = True
            if dropped:
                self._rewrite_event_file(e, ("dumps",))
        self._dumps_dropped += len(out)
        return out

    def _on_event_made(self, rec: dict):
        """The driver's bookkeeping of a liquid cut made (``surgery.liquid_cut``'s ``on_made``,
        before the state after it is kept and its record written), into its record: what the
        driver measured of the state before it, the base of the jet whose neck it cut
        (``trigger``: ``r_j``, ``z_b``, ``r_j_over_r_n`` and ``neck_to_base_over_r_j``, the
        neck's distance from the base in jet radii), the event's place in the run (``generation``
        at its tip, ``step``, ``leg``, ``tau_over_delta2``, ``t_prime_over_T`` with ``T = t_p -
        t_cut``, the ``cascade`` verdict, the source ``commit``, ``dt.before``), the start of its
        ghost (``step0``, ``tip``, ``far_tip``, ``recontacts``); then :meth:`reset_after_event`,
        the commit of the jets' bases and the kink regions of the fresh mesh
        (:meth:`_commit_post_cut`), which the state after the event, kept next, holds as the run
        steps from it, and the jet of the cut tip on the fresh mesh (``after``)."""
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
        # the jets' bases and the kink regions of the fresh mesh, before the state after the event
        # is kept: a restart from that dump steps from the mesh as the run does
        self._commit_post_cut(m_new)
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
        st.kinks.pop(f"{record.get('chain')}_far", None)  # jetcore's far kink of the chain
        self._checks_pending = False
        self._stop_cache = None
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

    def _on_ring_made(self, rec: dict):
        """The driver's bookkeeping of a ring surgery made (``ringsurgery.ring_cut`` or
        ``ring_removal``'s ``on_made``, before the state after it is kept and its record written),
        into its record: the event's place in the run (``step``, ``leg``, ``tau_over_delta2``,
        ``t_prime_over_T``, the source ``commit``, ``dt.before``); then :meth:`reset_after_ring`
        and the commit of the jets' bases and the kink regions of the fresh mesh
        (:meth:`_commit_post_cut`), which the state after the event, kept next, holds as the run
        steps from it."""
        st, delta = self.state, self.reconnect.delta
        t_cut, t_p = self._cut_times()
        T, tau_p = t_p - t_cut, rec.get("tau_p")
        rec.update(step=st.step, leg=len(st.legs) - 1,
                   tau_over_delta2=_finite(float(rec["tau"]) / delta**2),
                   t_prime_over_T=(_finite(float(tau_p) / T) if tau_p is not None
                                   and math.isfinite(T) and T != 0.0 else None),
                   commit=self.leg_info.get("commit"))
        rec.setdefault("dt", {})["before"] = _finite(st.dt_prev)
        kind = "ring-removal" if rec.get("kind") == "ring-collapse" else "ring-cut"
        m_new = self.reset_after_ring(rec, kind)
        self._commit_post_cut(m_new)
        self._event_after = m_new

    def reset_after_ring(self, record: dict, kind: str) -> dict:
        """Restart the bookkeeping that a ring surgery invalidates (``record`` its record; ``kind``
        ``ring-cut`` or ``ring-removal``) and measure the fresh mesh, as :meth:`reset_after_event`
        does for a liquid cut: no new cycle, the births, series, holds and last step kept; the
        growth limit of the local scale and the rate limit of the step restart, the apex histories
        of the bubble's two tips and its kink region are dropped (at the cut the bubble's tips are
        gone; ``tip_born`` is kept for the record), every ghost whose far tip was one of the
        bubble's loses it (``far_tip`` None, ``status_note``), and the remesh check of the step is
        not made on the fresh mesh (``_checks_pending`` cleared, so that the state after the event
        is kept without ``resume_at``); the event ``kind`` is recorded, ``ring.txt`` opened and a
        profile written (reason ``kind``)."""
        st = self.state
        st.ell_prev = st.t_prev = math.nan
        for tip in BUBBLE_TIP_SIGNS:
            st.apex_history.pop(tip, None)
        st.kinks.pop("bubble", None)
        st.kinks.pop("bubble_far", None)  # jetcore's far kink of the bubble
        if kind == "ring-cut":
            for e in self._made_events():
                g = e.get("ghost")
                if isinstance(g, dict) and g.get("far_tip") in BUBBLE_TIP_SIGNS:
                    g.update(far_tip=None, status_note="far tip gone (ring)")
        self._checks_pending = False
        # the stage begins or ends at the same step and rebuild
        self._post_cut_cache = self._stop_cache = None
        self._open_ring_rows()
        m = self.measure()
        k = record.get("k")
        filled = record.get("filled") or {}
        self._event(kind, m, k=k, kind=record.get("kind"), **{"class": record.get("class")},
                    chain=record.get("chain"), V_fill=filled.get("V"), Pz_fill=filled.get("Pz"),
                    KE_fill=filled.get("KE"),
                    record=f"events/event_{int(k):03d}.json" if k is not None else None)
        self._track_tips(m, snapshot=False)
        self._snapshot(m, kind)
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
        file in this leg's directory (:meth:`_rewrite_event_file`, a copy for an event an earlier
        leg made), as an event ``ghost-recontact`` and a row of ``drops.txt``.
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

    def _rewrite_event_file(self, rec: dict, keys: tuple[str, ...] = ("ghost",)):
        """Put the ``keys`` of an event's record as the run keeps it (its ghost as the run follows
        it, the dumps the policy deleted) into the record's file in this leg's directory
        (``events/event_<k>.json``), the ghost's polygon, which the file of the leg that made the
        event alone holds, kept; for an event made in an earlier leg, whose directory this leg does
        not write, the record as this leg holds it, as a copy (:meth:`_write_event_copy`) where
        there is no file yet."""
        path = self._event_file(rec["k"])
        if not path.is_file():  # made in an earlier leg, in its own run directory
            self._write_event_copy(rec, path)
            return
        full = json.loads(path.read_text())
        for key in keys:
            if key == "ghost":
                full["ghost"] = {**rec["ghost"],
                                 **({"polygon": full["ghost"]["polygon"]}
                                    if "polygon" in (full.get("ghost") or {}) else {})}
            else:
                full[key] = rec.get(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(full, indent=1) + "\n")
        tmp.replace(path)

    def _events_summary(self, stop: dict) -> dict:
        """After a cut with the block: the events of the run's result (``run_status.json``): how
        many were made, by kind and by tip, the last, the totals of the removals since the gas
        cut, the cascade's verdict at each tip, the log of the noise of a cut's box in the
        ordinary rebuilds (``box_noise``, :meth:`_box_noise_summary`, None without
        ``post.log_box_noise``), and the event the run stopped on, if any, with its record
        (:meth:`_write_stop_record`)."""
        made = self._made_events()

        def count(key):
            out: dict = {}
            for e in made:
                out[str(e.get(key))] = out.get(str(e.get(key)), 0) + 1
            return out

        last = made[-1] if made else None
        floors = sum(1 for e in made if (e.get("request") or {}).get("floor") is not None)
        splines = sum(1 for e in made if (e.get("request") or {}).get("spline_fill") is not None)
        stepped = sum(1 for e in made if (e.get("request") or {}).get("step_fill") is not None)
        return {"made": len(made), "max_events": self.events.max_events,
                "by_kind": count("kind"), "by_tip": count("tip"),
                "last": None if last is None else {key: last.get(key) for key in (
                    "k", "kind", "class", "chain", "tip", "t", "tau", "tau_p")},
                "removed_total": {key: v for key, v in self._removed().items()
                                  if key != "events"},
                "cascade": {tip: v["verdict"] for tip, v in self._cascade().verdicts().items()},
                "ghost_recontacts": {kind: sum(
                    1 for e in made for x in (e.get("ghost") or {}).get("recontacts") or ()
                    if x.get("kind") == kind) for kind in ("rear", "front")},
                "dumps_dropped": self._dumps_dropped, "meshes_pruned": self._meshes_pruned,
                "box_noise": self._box_noise_summary(),
                **({"ring": self._ring_summary()} if self._ring() is not None else {}),
                **({"sheets": self._sheets_logged} if self._sheets_logged else {}),
                **({"floor_fills": floors} if floors else {}),
                **({"spline_fills": splines} if splines else {}),
                **({"step_contact_fills": stepped} if stepped else {}),
                **({"tip_edge_straightens": len(self._tip_edge_balances)}
                   if self._tip_edge_balances else {}),
                "stop": ({key: stop.get(key) for key in ("reason", "k", "kind", "class",
                                                         "handler", "record")}
                         if stop.get("reason") in EVENT_STOPS else None)}

    def _ring_summary(self) -> dict:
        """The ring of the run's result in the ring stage: the event of its cut, that of its
        removal (None while it lasts) and the time after the gas cut of the removal."""
        ring = self._ring() or {}
        removed = ring.get("removed") or {}
        return {"cut": ring.get("k_cut"), "removed": removed.get("k"),
                "tau_removed": removed.get("tau")}

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
        """The stop the measured state calls for, if any: a rebuild the box gate refused
        (``box-gate``, :meth:`_gate_after`), a geometry stop (after a cut, those of both chains
        too) before the jet stop."""
        if self._gate_stop is not None:
            return dict(self._gate_stop)
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

    def _tip_edge_straighten(self, m: dict, stop: dict | None, *,
                             failed_step: dict | None = None) -> tuple[dict, dict | None]:
        """Meet a fresh edge turn or a failed step's terminal dip by audited straightening.

        Only the final three quadratic elements of the downward tip qualify;
        the ring and upward chain are excluded. Signed sliver quadrature,
        local volume/momentum closures and the upward-base check are hard. The native
        affected triangle is integrated independently of the interface sliver: its actual
        volume change must match the signed curve/chord volume, and its actual momentum
        must match the planned nodal change, at the existing R1/R2 relative bounds. Book
        that triangle's full momentum/energy change, including the velocity reset. Refuse
        a recurring fold, a repeated tip operation within the axis-repair window, or a
        cumulative absolute volume above its bound. Records are events/tip_edge_<step>_<n>.json;
        event tip-edge-straighten and persisted tip_edge_balances account for a made operation.
        A hard refusal restores the native input and freshness (tip-edge-unclosed); later
        events and window ends are dispatched after acceptance. The tip_edge unit/FEM tests
        and 6ef3c4d identity pins cover this contract.

        A terminal dip is eligible only after an inversion failure was rolled back.
        Normal steps retain ordinary refinement. Refused admission leaves that failed
        step stopped; it never creates a new stop on an otherwise admissible state.
        """
        dimple = (stop is None and failed_step is not None
                  and failed_step.get("reason") == "step-failed"
                  and failed_step.get("error", "").startswith("InvertedElementRemeshRequest:"))
        if not dimple and (stop is None or stop.get("reason") != "interface-unresolved"
                or stop.get("trigger") != "edge-turn" or stop.get("chain") != "bubble"
                or not stop.get("fresh") or not self._fresh_mesh()):
            return m, stop
        if self._ring() is not None:
            return m, stop
        jet = self._jetcore()
        if jet is None or jet.far_kinks != "recentre" \
                or not hasattr(self.pb, "straighten_interface_edges"):
            return m, stop
        from .axisrepair import AXIS_REPAIR_BOUND, AXIS_REPAIR_RECUR_STEPS
        from .tipcontact import (closures, edge_patch, patch_budget, patch_jacobian_bound, sliver_momentum,
                                 swept_edge, terminal_dimple)
        from .surgery import box_integrals, site_grid

        sigma, pb = self.events.groove, self.pb
        if stop is not None and (stop.get("axis_fold_recurs") is not None
                                 or self._axis_fold_of(stop) is not None):
            return m, stop
        segments = pb.interface_segments()
        bubble = next((c for c in segments if not c["open"]), None)
        if bubble is None:
            return m, stop
        P = np.column_stack([bubble["r"], bubble["z"]])
        if dimple:
            from .problem import AXIS_MIDSIDE_LIMIT

            stop = terminal_dimple(P, self.reconnect.edge_turn_refine, AXIS_MIDSIDE_LIMIT)
            if stop is None:
                return m, None
            stop["fresh"] = self._fresh_mesh()
        i = int(stop.get("index", -1))
        if i < len(P) - 6 or i >= len(P) - 1 or i % 2 != 1:
            return m, stop
        radius = TIP_STRAIGHTEN_REACH * float(stop["L"])
        centre = np.array([stop["r"], stop["z"]])
        for c in segments:
            if c["open"]:
                Q = np.column_stack([c["r"], c["z"]])
                midpoints = 0.5 * (Q[:-2:2] + Q[2::2])
                if (
                    np.min(np.linalg.norm(Q - centre, axis=1)) <= radius
                    or np.min(np.linalg.norm(midpoints - centre, axis=1)) <= radius
                ):
                    return m, stop
        read = pb.straighten_interface_edges(*centre, radius, apply=False)
        if len(read["edges"]) != 1:
            return m, stop
        edge = read["edges"][0]
        if (edge["r"], edge["z"]) != tuple(P[i]):
            return m, stop
        if self._tip_edge_balances and self.state.step - self._tip_edge_balances[-1]["step"] \
                < AXIS_REPAIR_RECUR_STEPS:
            return m, dict(stop, tip_edge_recurs={"last_step": self._tip_edge_balances[-1]["step"],
                                                "window": AXIS_REPAIR_RECUR_STEPS})
        if not (sigma.sigma_V > 0 and sigma.sigma_Pz > 0):
            return m, dict(stop, tip_edge_unavailable="uncalibrated volume or momentum noise")
        sweep = swept_edge(P[i - 1 : i + 2])
        dV, bound = sweep["dV"], AXIS_REPAIR_BOUND * self.reconnect.delta**3
        cumulative = sum(abs(x["V"]) for x in self._tip_edge_balances) + abs(dV)
        if not math.isfinite(dV) or cumulative > bound or np.any(sweep["weights"] <= 0):
            return m, dict(
                stop, tip_edge_volume={"dV": dV, "cumulative_abs": cumulative, "bound": bound}
            )
        velocity = np.column_stack([bubble["u_r"], bubble["u_z"]])[i - 1 : i + 2]
        speed = float(np.max(np.linalg.norm(velocity, axis=1)))
        grid = site_grid(*centre, sigma.box_radius * self.reconnect.delta)
        b = box_integrals(pb, grid, cutoff=True)
        before = {"V": b["V_grid"], "Pz": b["Pz"], "KE": b["KE"]}
        quad = sliver_momentum(pb, sweep) if dV < 0 else None
        saved = pb._snapshot_state()
        was_fresh = self._fresh_mesh()
        n = len(self._tip_edge_balances) + 1
        path = self.outdir / "events" / f"tip_edge_{self.state.step:08d}_{n}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "guard": stop,
            "edge": edge,
            "radius": radius,
            "dV": dV,
            "step": self.state.step,
            "before": before,
            "cumulative_abs_V": cumulative,
            "cumulative_bound": bound,
            "recurrence_window": AXIS_REPAIR_RECUR_STEPS,
        }
        if dimple:
            rec["failed_step"] = failed_step
            rec["inversion_site"] = "unknown; candidate selected by terminal geometry"
        try:
            patch = edge_patch(pb, P[i], 0.5 * (P[i - 1] + P[i + 1]),
                               0.5 * (velocity[0] + velocity[2]))
            mesh_before = patch_budget(patch, initial=True)
            mesh_planned = patch_budget(patch, planned=True)
            jacobian_before = patch_jacobian_bound(patch, initial=True)
            applied = pb.straighten_interface_edges(*centre, radius)
            if applied["edges"] != read["edges"]:
                raise RuntimeError("the straightening selection changed after admission")
            self._post_cut_cache = self._stop_cache = None
            after_m = self.measure()
            a = box_integrals(pb, grid, cutoff=True)
            after = {"V": a["V_grid"], "Pz": a["Pz"], "KE": a["KE"]}
            if dV >= 0:
                quad = sliver_momentum(pb, sweep)
            mesh_after = patch_budget(patch)
            jacobian_after = patch_jacobian_bound(patch)
            dP = mesh_after["Pz"] - mesh_before["Pz"]
            mesh = {"before": mesh_before, "after": mesh_after, "planned": mesh_planned}
            checks = closures(before, after, dV, dP, sigma, bound, speed, quad, mesh=mesh)
            base = base_guard_checked(m, after_m)
            if not base["ok"]:
                checks["hard"].append("T_base")
            jacobian = {"before": jacobian_before, "after": jacobian_after,
                        "limit": self.reconnect.jacobian_min}
            if not (math.isfinite(jacobian_before) and math.isfinite(jacobian_after)) \
                    or jacobian_after < self.reconnect.jacobian_min:
                checks["hard"].append("T_guard")
            rec.update(after=after, quadrature=quad, mesh=mesh, closures=checks, base=base,
                       jacobian=jacobian)
            if checks["hard"]:
                raise RuntimeError("the tip edge's hard checks did not close")
            guard = self._tip_edge_guard()
            rec["guard_after"] = guard
            if guard is not None:
                checks["hard"].append("T_guard")
                raise RuntimeError("a geometric guard still fails after straightening")
        except Exception as exc:
            pb._restore_state(saved)
            self._stepped_rc = None if was_fresh else pb.remesh_count
            pb.invalidate_cached_mesh_data()
            pb._jacobian_cache = None
            self._post_cut_cache = self._stop_cache = None
            rec.update(outcome="rolled-back", error=f"{type(exc).__name__}: {exc}"[:400])
            path.write_text(json.dumps(_plain(rec), indent=1) + "\n")
            return self.measure(), {
                "reason": "interface-unresolved",
                "trigger": "tip-edge-unclosed",
                "chain": "bubble",
                "record": f"events/{path.name}",
                "hard": rec.get("closures", {}).get("hard", []),
                "error": rec["error"],
                "guard": {k: v for k, v in stop.items() if k != "reason"},
                **self._taus(self._t()),
            }
        rec["outcome"] = "made"
        path.write_text(json.dumps(_plain(rec), indent=1) + "\n")
        self._tip_edge_balances.append(
            {
                "V": dV,
                "Pz": dP,
                "KE": mesh_after["KE"] - mesh_before["KE"],
                "step": self.state.step,
                "record": f"events/{path.name}",
            }
        )
        self._event(
            "tip-edge-straighten",
            after_m,
            record=f"events/{path.name}",
            closures=checks,
            base=base,
            dV=dV,
            dPz=dP,
            **({"inversion_site": rec["inversion_site"]} if dimple else {}),
        )
        self._snapshot(after_m, "tip-edge-straighten")
        # Later events and legitimate window ends are dispatched after the
        # committed operation; the transactional audit never calls _stop.
        found = self._stop(after_m)
        return self.measure(), found

    def _tip_edge_guard(self) -> dict | None:
        """Read geometric guards without invoking another repair.

        The node polygons must be simple and mutually disjoint; their rebuild splines use
        the configured vertex stride. Then read interface_guard and, only on a fresh mesh,
        the raw whole-mesh Jacobian bound. This transactional read offers no further repair;
        a result or a topology read error becomes the transaction's T_guard;
        event requests and window ends belong to the caller's later dispatch.
        """
        from .reconnect import Chain, chains_disjoint, spline_contact_stop
        from .surgery import _named_chains
        from shapely.geometry import LineString

        named = _named_chains(self.pb)[0]
        try:
            chains_disjoint([c for _, c in named])
            if any(not LineString(c.points).is_simple for _, c in named):
                raise RuntimeError("a chain's node polygon crosses itself")
            stride = 2 if self.reconnect.rebuild_spline == "vertices" else 1
            sampled = [(name, Chain(c.points[::stride], c.end_types))
                       if stride > 1 and len(c.points) % 2 == 1 else (name, c)
                       for name, c in named]
            contact = spline_contact_stop(sampled)
            if contact is not None:
                return contact
        except RuntimeError as exc:
            return {"reason": "interface-unresolved", "trigger": "tip-topology",
                    "error": str(exc)[:400]}
        guard = interface_guard(
            {name: edge_reading(c.points) for name, c in named},
            self.reconnect,
            fresh=self._fresh_mesh(),
            h_floor=self._h_floor(),
            refine_ratio=self.s.refine_ratio,
        )["stop"]
        if guard is not None:
            return guard
        if self._fresh_mesh():
            bound = self.pb.curved_jacobian_bound(self.reconnect.jacobian_min)
            if bound is not None and bound["min"] < self.reconnect.jacobian_min:
                return {"reason": "interface-unresolved", "trigger": "curved-jacobian", **bound}
        return None

    def _axis_fold_of(self, stop: dict | None) -> tuple[str, dict] | None:
        """The fold along the axis that a guard stop sits on (``axisrepair.axis_fold``): the
        guard's ``interface-unresolved`` (an edge turn, a midside node, a feature below the floor)
        on the bubble before the piercing, with jetcore's recentred far kinks (the nuisance region,
        decision 22), at an element within the folded tail of one of the bubble's ends on the axis
        and the node kept beside it; the end and the fold, or None."""
        if stop is None or stop.get("reason") != "interface-unresolved" \
                or stop.get("chain") != "bubble" or stop.get("trigger") not in (
                    "edge-turn", "edge-midside", "feature-below-floor") \
                or stop.get("index") is None:
            return None
        jet = self._jetcore()
        if jet is None or jet.far_kinks != "recentre" or self._ring() is not None:
            return None
        from .axisrepair import axis_fold
        from .surgery import _named_chains

        named = dict(_named_chains(self.pb)[0])
        if "bubble" not in named:
            return None
        P = named["bubble"].points
        i = int(stop["index"])
        for end in ("first", "last"):
            fold = axis_fold(P, end)
            if fold is None:
                continue
            near = fold["removed"] + [fold["kept"], fold["kept"] + (-1 if end == "last" else 1)]
            if min(abs(i - j) for j in near) <= 1:
                return end, fold
        return None

    def _axis_repair_due(self, stop: dict | None) -> bool:
        """Whether a stop is one :meth:`_axis_repair` meets (:meth:`_axis_fold_of`)."""
        return self._axis_fold_of(stop) is not None

    def _axis_repair(self, m: dict, stop: dict) -> tuple[dict, dict | None]:
        """Repair a jet's tip folded along the axis (``axisrepair.axis_repair``; decision 22, the
        lead's option A): the folded nodes dropped and the axis node put where the surface meets
        the axis by a rebuild from the planned chains, the sliver of gas between them filled, its
        volume and momentum hard above ``axisrepair.AXIS_REPAIR_BOUND delta^3`` (and ``U`` times
        it), the transfer's closures A1-A6 as a fill's, the upward jet's base checked; an event
        ``axis-repair`` and its record ``events/axis_repair_<n>.json``. A repair whose hard
        closures miss, or whose rebuild fails, is rolled back to the state before it (the snapshot
        taken first) and is the resumable stop ``axis-repair-unclosed`` (or ``axis-repair-failed``)
        for the cap of option B; a fold at the same end within ``AXIS_REPAIR_RECUR_STEPS`` steps of
        the last repair there is not repaired again (the stop as it is, ``axis_fold_recurs``).
        Returns the state the run goes on from and its stop."""
        from .axisrepair import AXIS_REPAIR_BOUND, AXIS_REPAIR_RECUR_STEPS, axis_repair
        from .surgery import BudgetUnclosed, _named_chains

        end, fold = self._axis_fold_of(stop)
        st, pb = self.state, self.pb
        last = [x for x in self._axis_repairs if x["end"] == end]
        if last and st.step - last[-1]["step"] < AXIS_REPAIR_RECUR_STEPS:
            return m, dict(stop, axis_fold_recurs={"end": end, "last_step": last[-1]["step"],
                                                   "repairs": len(last)})
        n = len(self._axis_repairs) + 1
        segs = _named_chains(pb)[1]
        velocity = np.column_stack([segs["bubble"]["u_r"], segs["bubble"]["u_z"]])
        within_bound = fold["V_sliver"] <= AXIS_REPAIR_BOUND * self.reconnect.delta**3
        snapshot = pb._snapshot_state() if within_bound else None
        was_fresh = self._fresh_mesh()
        path = self.outdir / "events" / f"axis_repair_{n:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        def base_check():
            return base_guard_checked(m, self.measure())

        reason = None
        try:
            record = axis_repair(pb, self.events, self.reconnect, self.outdir, "bubble", fold,
                                 velocity=velocity, base_check=base_check)
        except BudgetUnclosed as e:
            record, reason = getattr(e, "record", {}), "axis-repair-unclosed"
        except Exception as e:  # noqa: BLE001  (a rebuild that fails: rolled back, a stop)
            record = {**(getattr(e, "record", None) or {}), "kind": "axis-repair", "outcome": "failed",
                      "error": f"{type(e).__name__}: {e}"[:400]}
            reason = "axis-repair-failed"
        record.update(n=n, step=st.step, guard=stop, **self._taus(self._t()))
        path.write_text(json.dumps(_plain(record), indent=1) + "\n")
        if reason is not None:
            if snapshot is not None:
                pb._restore_state(snapshot)
                self._stepped_rc = None if was_fresh else pb.remesh_count
                pb.invalidate_cached_mesh_data()
                self._post_cut_cache = None
                m = self.measure()
            return m, {"reason": reason, "chain": "bubble", "end": end,
                       "hard": (record.get("closures") or {}).get("hard"),
                       "error": record.get("error"), "record": f"events/{path.name}",
                       "guard": {k: v for k, v in stop.items() if k != "reason"},
                       **self._taus(self._t())}
        st.last_remesh_step = st.step
        after = self.measure()
        self._keep_landings(after)
        self._axis_repairs.append({"end": end, "step": st.step,
                                   "V_sliver": record["fold"]["V_sliver"]})
        self._event("axis-repair", after, n=n, end=end, chain="bubble",
                    removed=len(fold["removed"]), V_sliver=record["fold"]["V_sliver"],
                    V_sliver_over_delta3=record["fold"]["V_sliver_over_delta3"],
                    new_end=record["fold"]["new_end"], hard=record["closures"]["hard"],
                    flags=record["closures"]["flags"], base=record.get("base_guard"),
                    record=f"events/{path.name}")
        self._snapshot(after, "axis-repair")
        return after, self._stop(after)

    def _refined_chain(self) -> str | None:
        """The chain whose guard stops the ring refine meets: the ring, in the ring stage before its
        removal, with jetcore's recentred far kinks; None otherwise. (The bubble before the
        piercing is not met so: r3's stop at e923ead, step 7338, is the downward jet's tip folded
        along the axis, a node of it at r < 0, which a rebuild at the size its turn asks for exposes
        as an axis contact, not a feature it resolves.)"""
        jet = self._jetcore()
        if jet is None or jet.far_kinks != "recentre":
            return None
        ring = self._ring()
        return "ring" if ring is not None and ring.get("removed") is None else None

    def _ring_jacobian_reading(self, stop: dict | None) -> dict | None:
        """A rebuilt mesh's ``curved-jacobian`` stop at an element on the ring's loop read as the
        guard's reading of the ring's quadratic element there: in the ring stage before the ring's
        removal with jetcore's recentred far kinks, a stop ``on_interface`` whose element's
        centroid lies within a chord of the ring's nearest element (and no nearer to an element
        of the open chain) is returned with ``chain`` ``ring`` and that element's ``index``, ``L``
        (its chord), ``turn``, ``f`` and midpoint (``r``, ``z``; the bad element's centroid kept as
        ``element``), so that the ring refine and the fill at the floor meet it as they meet the
        guard's own stops on the ring (r1 at 63293b0 on 2 threads, step 7501: a corner of the ring
        turning by 89.8 degrees within one element of 3.3e-6, just below the fresh stop, rebuilt for
        distortion into an element whose Jacobian bound fell to 0.09 against 0.2). Any other stop
        is returned as it is."""
        if stop is None or stop.get("reason") != "interface-unresolved" \
                or stop.get("trigger") != "curved-jacobian" or not stop.get("on_interface") \
                or stop.get("chain") is not None:
            return stop
        chain = self._refined_chain()
        if chain is None:
            return stop
        edges = self._post_cut_state().get("edges") or {}
        if chain not in edges:
            return stop
        x = np.array([float(stop["r"]), float(stop["z"])])

        def nearest(name):
            e = (edges.get(name) or {}).get("edges")
            if e is None or not len(e["L"]):
                return None, math.inf
            d = np.hypot(e["mid"][:, 0] - x[0], e["mid"][:, 1] - x[1])
            k = int(np.argmin(d))
            return (e, k), float(d[k])

        (ring_e, k), d_ring = nearest(chain)
        _, d_open = nearest("open")
        if not (d_ring <= float(ring_e["L"][k]) and d_ring <= d_open):
            return stop
        return dict(stop, chain=chain, index=int(ring_e["index"][k]), L=float(ring_e["L"][k]),
                    turn=float(ring_e["turn"][k]), f=float(ring_e["f"][k]),
                    r=float(ring_e["mid"][k, 0]), z=float(ring_e["mid"][k, 1]),
                    element={"r": float(x[0]), "z": float(x[1])})

    def _ring_refine_due(self, stop: dict | None) -> bool:
        """Whether a stop is one :meth:`_ring_refine` meets: the guard's ``interface-unresolved``
        on the ring (``chain`` ``ring``), on a freshly built mesh (beyond ``edge_turn_stop_fresh``)
        or on one a step has run on (beyond ``edge_turn_stop``: the rim of a sheet born within a
        few steps of the ring cut, before any rebuild could refine it), trigger ``edge-turn`` up to
        ``RING_REFINE_TURN_MAX`` or ``edge-midside``, or a rebuilt mesh's ``curved-jacobian`` at an
        element on the ring (:meth:`_ring_jacobian_reading`), at an element longer than
        ``refine_ratio`` times the run's ``h_floor`` (the guard's own floor), in the ring stage
        before the ring's removal with jetcore's recentred far kinks (:meth:`_refined_chain`).
        A liquid kink of the ring also qualifies, through :meth:`_ring_liquid_reading`, with
        hard zero-removal balances around its rebuild (:meth:`_ring_liquid_refine`), and so does a
        closing fold of the ring's gas across a kink whose elements lie above the floor's
        (:meth:`_ring_kink_reading`, trigger ``fold-kink``; r2, fix16 leg 1 at bf80e74, step 12662:
        the pair 6093/6096 three nodes apart, 1.21e-6 per element against ``refine_ratio
        h_floor`` = 1.17e-6), refined to the floor with the same audit, so that the fill at the
        floor (:meth:`_floor_fill_request`) meets the kink there if it stays."""
        stop = self._ring_reading(stop)
        if stop is None or stop.get("reason") != "interface-unresolved":
            return False
        if stop.get("chain") != "ring":
            return False
        trigger = stop.get("trigger")
        if trigger not in ("edge-turn", "edge-midside", "curved-jacobian", "liquid-kink",
                           "sheet-contact", "fold-kink"):
            return False
        if stop.get("chain") != self._refined_chain():
            return False
        if trigger in ("edge-turn", "curved-jacobian") and not float(stop.get("turn", math.inf)) \
                <= RING_REFINE_TURN_MAX:
            return False
        L = float(stop.get("L", math.nan))
        return math.isfinite(L) and L > self.s.refine_ratio * self._h_floor()

    def _ring_reading(self, stop: dict | None) -> dict | None:
        """The stop as the ring refine reads it: a rebuilt mesh's curved element on the ring
        (:meth:`_ring_jacobian_reading`), a gas kink of the ring as the guard's reading at the
        element between its pair (:meth:`_ring_kink_reading`), a liquid kink
        (:meth:`_ring_liquid_reading`) and a lip to resolve (:meth:`_ring_sheet_reading`); any
        other stop as it is."""
        return self._ring_sheet_reading(self._ring_liquid_reading(
            self._ring_kink_reading(self._ring_jacobian_reading(stop))))

    def _ring_liquid_reading(self, stop: dict | None) -> dict | None:
        """A liquid kink of the ring asks for refinement, never for a fill or film rupture."""
        if stop is None or stop.get("reason") != "self-contact-approach" \
                or stop.get("chain") != "ring" or stop.get("side") != "liquid" \
                or stop.get("contact") != "kink" or stop.get("trigger") != "fold-gap":
            return stop
        from .ringcontact import liquid_reading

        noise = self.events.ring
        if not (noise.sigma_V > 0.0 and noise.sigma_Pz > 0.0):
            stop["liquid_refine_unavailable"] = "ring volume and momentum noise floors are not calibrated"
            return stop
        loops = self._post_cut_state().get("loops") or []
        if len(loops) != 1:
            return stop
        return liquid_reading(np.column_stack([loops[0]["r"], loops[0]["z"]]), stop)

    def _ring_spline_fill_request(self, stop: dict | None) -> dict | None:
        """Fill a certified local gas wedge whose rebuild spline already crosses itself.

        The floor's fold limits read the wedge, without pretending that a spline
        crossing is a kink or assigning a phase from an absent side field.
        The fill carries its usual hard closures and the upward-base check.
        """
        if stop is None or stop.get("trigger") != "spline-contact" or stop.get("chain") != "ring":
            return None
        ring, jet = self._ring(), self._jetcore()
        if ring is None or ring.get("removed") is not None or jet is None \
                or jet.far_kinks != "recentre" or not self.events.groove.enabled:
            return None
        from .ringcontact import spline_wedge

        loops = self._post_cut_state().get("loops") or []
        if len(loops) != 1:
            return None
        P = np.column_stack([loops[0]["r"], loops[0]["z"]])
        wedge = spline_wedge(P, stop, stop_elements=self.s.fold_stop_elements)
        if wedge is None:
            return None
        return self._local_ring_fill_request(P, wedge, stop, kind="spline")

    def _local_ring_fill_request(self, P, wedge: dict, guard: dict, *, kind: str) -> dict:
        """The common existing fill protocol for a certified local gas pocket."""
        from .events import GROOVE_HANDLER, _request
        from .ring import FLOOR_FOLD_RATIO

        a, b = wedge["pair"]
        mid = 0.5 * (P[a] + P[b])
        # the wedge's own fold ratio: FLOOR_CORNER_RATIO for a corner (step_wedge)
        fold = {"ratio": float(wedge.get("ratio", FLOOR_FOLD_RATIO)), "stop": 0.0,
                "stop_elements": self.s.fold_stop_elements}
        req = _request("ring-fold", "RF", chain="ring", contact=kind, side="gas",
                       fold_gap=wedge["gap"], fold_r=float(mid[0]), fold_z=float(mid[1]),
                       fold_arc=wedge["arc"], fold_h_elem=wedge["h_elem"],
                       fold_limit=wedge["limit"], fold_pair=[a, b], kappa_g=self.events.gap.kappa_g,
                       **{f"{kind}_fill": {"guard": dict(guard), "wedge": wedge, "fold": fold}})
        req["handler"] = GROOVE_HANDLER
        req.update(self._taus(self._t()))
        return req

    def _ring_step_input(self) -> dict | None:
        """The ring before a watched step; node identities are used only without a rebuild."""
        loops = self.pb.interface_loops(with_node_ids=True)
        if len(loops) != 1:
            return None
        return {"P": np.column_stack([loops[0]["r"], loops[0]["z"]]),
                "U": np.column_stack([loops[0]["u_r"], loops[0]["u_z"]]),
                "node_ids": loops[0]["node_ids"].copy(),
                "remesh_count": self.pb.remesh_count, "t": self._t(),
                "key": input_key(loops[0]["r"], loops[0]["z"]),
                "defer": self._defers_distortion()}

    def _fold_retry_due(self, before: dict) -> bool:
        """Whether a step from this input (its ``key``, :func:`ringcontact.input_key`) may still
        be retried at a quarter: once per input, which a rollback restores bitwise and a fill or
        a refine at the same time changes."""
        retry = self.state.fold_retry
        return retry is None or retry.get("input") != before.get("key")

    def _ring_input_key(self) -> str | None:
        """The key of the ring's present input (the state a rollback restored), or None."""
        loops = self.pb.interface_loops(with_node_ids=True)
        return input_key(loops[0]["r"], loops[0]["z"]) if len(loops) == 1 else None

    def _ring_step_check(self, before: dict):
        """Reject a candidate local crossing before step counters or gas work are committed.

        A candidate that instead folds (a new alternating run) or turns an element inside out
        (pyoomph's net rebuilt within the step, an inverted element it logged within reach of
        the corner) at a closing gas corner of its input is never accepted
        (:class:`RingStepFold`): rolled back and retried once at a quarter of its step, so that
        the smaller step meets the crossing on a readable input (the stepped contact's fill);
        with that input's retry spent (``consumed``), the corner is filled on the input or the
        run stops resumably there (batch B; tangential node redistribution is the principled
        cure)."""
        from .ringcontact import loop_node_map, step_contact, step_fold

        if self.pb.remesh_count != before["remesh_count"]:
            # no index mapping is inferred across a rebuild; with the driver's deferred
            # distortion a rebuild within the step is pyoomph's net for an inverted element
            at = self._step_inversions() if before.get("defer") else None
            corner = None if at is None else self._corner_at(before["P"], before["U"], at)
            if corner is not None:
                raise RingStepFold({"trigger": "step-inverted", "chain": "ring",
                                    "net_rebuilds": int(self.pb.remesh_count
                                                        - before["remesh_count"]),
                                    "corner": corner, "attempted_time": self._t(),
                                    "consumed": not self._fold_retry_due(before)})
            return
        try:
            loops = self.pb.interface_loops(validate=False, with_node_ids=True)
        except RuntimeError as exc:
            raise RingStepContact({"unreadable": str(exc)[:400],
                                   "attempted_time": self._t()}) from exc
        if len(loops) != 1:
            raise RingStepContact({"unreadable": "the number of loops changed without a rebuild",
                                   "attempted_time": self._t()})
        P = np.column_stack([loops[0]["r"], loops[0]["z"]])
        if len(P) != len(before["P"]):
            raise RingStepContact({"unreadable": "the loop node count changed without a rebuild",
                                   "attempted_time": self._t()})
        reading = step_contact(P)
        if reading is None:
            from .ring import loop_crossings
            from shapely.geometry import LinearRing

            if not LinearRing(P).is_simple:
                raise RingStepContact({"unreadable": "the crossing is outside the local fill rule",
                                       "crossing_pairs": loop_crossings(P),
                                       "attempted_time": self._t(),
                                       "attempted_step": self.state.step+1})
        if reading is None:  # every candidate: a folded one is never accepted
            mapping = loop_node_map(loops[0]["node_ids"], before["node_ids"])
            if mapping is not None:
                fold = step_fold(before["P"][mapping], before["U"][mapping], P)
                if fold is not None:
                    raise RingStepFold(dict(fold, attempted_time=self._t(),
                                            consumed=not self._fold_retry_due(before)))
        if reading is not None:
            mapping = loop_node_map(loops[0]["node_ids"], before["node_ids"])
            reading.update(attempted_time=self._t(), attempted_step=self.state.step+1,
                           before_time=self.state.t_prev)
            retry = getattr(self.state, "fold_retry", None)
            if retry is not None and retry["step"] >= self.state.step+1-FOLD_RETRY_REACH:
                reading["fold_retry"] = dict(retry)  # the retry that led to this crossing
            if mapping is None:
                reading["unreadable"] = "the loop node identities changed"
            else:
                indices = reading["local_indices"]
                reading.update(node_ids=before["node_ids"].tolist(),
                               local_node_ids=loops[0]["node_ids"][indices].tolist(),
                               before_local=before["P"][mapping[indices]].tolist())
            raise RingStepContact(reading)

    def _meet_ring_step_fold(self, m: dict, fold: RingStepFold, t: float, dt: float, *,
                             restored: bool = False) -> dict:
        """Roll a folded or inverted candidate back to its input and arm one retry of that step
        at ``FOLD_RETRY_FRACTION`` of it (event ``ring-step-fold-retry``); no global Courant
        number changes, and the step after the retry grows as every step does. ``restored``: a
        failed step that :meth:`_failed_step_stop` rolled back already."""
        discarded = self._distortion_flags()
        if not restored:
            self._restore_step_snapshot()
            self._stop_cache = None
            m = self.measure()
        st = self.state
        cap = FOLD_RETRY_FRACTION * dt
        st.fold_retry = {"t": t, "input": self._ring_input_key(), "dt": cap, "dt_attempted": dt,
                         "step": st.step+1, "trigger": fold.reading.get("trigger")}
        if st.fold_cap is None or st.fold_cap["t"] != t or cap < st.fold_cap["dt"]:
            st.fold_cap = {"t": t, "dt": cap}
        self._event("ring-step-fold-retry", m, reading=dict(fold.reading),
                    dt_attempted=dt, dt_retry=cap, input=st.fold_retry["input"],
                    discarded_distortion=discarded)
        return m

    def _step_inversions(self) -> np.ndarray | None:
        """The positions of the inverted elements pyoomph's log reports for the present step
        (from ``_log_mark``), or None without any or without the stepped contact's
        preconditions (the ring stage, recentred far kinks, groove fill enabled)."""
        events, jet = getattr(self, "events", None), self._jetcore()
        if events is None or not events.groove.enabled or jet is None \
                or jet.far_kinks != "recentre" or self._refined_chain() != "ring":
            return None
        mark = getattr(self, "_log_mark", None)
        size = getattr(self.pb, "oomph_log_size", lambda: None)()
        tail = (self.pb.oomph_log_tail(max(size - mark, 0)) if mark is not None and size is not None
                else getattr(self.pb, "oomph_log_tail", lambda: "")())
        found = re.findall(r"around the position \(([-+0-9.eE]+), ([-+0-9.eE]+)\)", tail)
        return np.array([[float(x) for x in pair] for pair in found]) if found else None

    @staticmethod
    def _corner_at(P, U, at: np.ndarray) -> dict | None:
        """The sharpest closing gas corner of the input ``P`` (velocity ``U``) within reach of
        a reported inverted element ``at``: its printed precision (six significant digits, half
        a unit in the last) plus ``CORNER_LOCATE`` node spacings of the corner; else None."""
        from .ring import loop_spacing
        from .ringcontact import closing_corners

        res = np.vectorize(lambda v: 0.5 * 10.0 ** (math.floor(math.log10(abs(v))) - 5)
                           if v else 0.0)(at)
        spacing = loop_spacing(P)
        best = None
        for c in closing_corners(P, U):
            reach = res + CORNER_LOCATE * spacing[c["index"]]
            near = np.all(np.abs(P[c["index"]] - at) <= reach, axis=1)
            if near.any() and (best is None or c["turn"] > best["turn"]):
                best = dict(c, inverted_at=at[near].tolist(), inverted_reported=len(at),
                            inverted_reach=float(CORNER_LOCATE * spacing[c["index"]]))
        return best

    def _inverted_corner(self, exc: BaseException) -> dict | None:
        """After a failed step rolled back: the closing gas corner of its input at which
        pyoomph's net for an inverted element gave up (``InvertedElementRemeshRequest``), located
        by an inverted element the log reports within that step (:meth:`_step_inversions`,
        :meth:`_corner_at`); else None."""
        if type(exc).__name__ != "InvertedElementRemeshRequest":
            return None
        at = self._step_inversions()
        if at is None:
            return None
        loops = self.pb.interface_loops(with_node_ids=True)
        if len(loops) != 1:
            return None
        P = np.column_stack([loops[0]["r"], loops[0]["z"]])
        U = np.column_stack([loops[0]["u_r"], loops[0]["u_z"]])
        best = self._corner_at(P, U, at)
        if best is not None:
            best.update(node_count=len(P), node_ids=loops[0]["node_ids"].tolist())
        return best

    def _input_corner(self, corner: dict) -> dict | None:
        """A corner read on a step's input (its ``r``, ``z``) on the restored input: its index
        there (the node at exactly that place) and the restored loop's identities; else None."""
        loops = self.pb.interface_loops(with_node_ids=True)
        if len(loops) != 1:
            return None
        P = np.column_stack([loops[0]["r"], loops[0]["z"]])
        at = np.flatnonzero((P[:, 0] == corner["r"]) & (P[:, 1] == corner["z"]))
        if len(at) != 1:
            return None
        return dict(corner, index=int(at[0]), node_count=len(P),
                    node_ids=loops[0]["node_ids"].tolist())

    def _meet_ring_fold_again(self, m: dict, fold: RingStepFold) -> tuple[dict, dict | None]:
        """A candidate that folds or inverts at a closing corner of an input whose one retry is
        spent: never accepted. Rolled back to the input (event ``ring-step-fold-again``), then
        the corner is filled there as an inverted retry's is (:meth:`_meet_ring_corner`), or the
        run stops resumably on that input (``step-failed``) with the record."""
        discarded = self._distortion_flags()
        self._restore_step_snapshot()
        self._stop_cache = None
        m = self.measure()
        self._event("ring-step-fold-again", m, reading=dict(fold.reading),
                    discarded_distortion=discarded)
        corner = self._input_corner(fold.reading["corner"])
        if corner is None:
            return m, {"reason": "step-failed", "trigger": "stepped-contact-unhandled",
                       "chain": "ring", "error": "a retried step folded again at a corner the "
                       "restored input does not hold", "contact": dict(fold.reading),
                       **self._taus(self._t())}
        return self._meet_ring_corner(m, corner, trigger=fold.reading.get("trigger"))

    def _meet_ring_corner(self, m: dict, corner: dict, *,
                          trigger: str = "step-failed-inverted") -> tuple[dict, dict | None]:
        """Fill the closing gas corner of a rolled-back input whose step, retried at a quarter,
        folded or turned an element inside out there again (``trigger``): the crossing the retry
        located, filled on that input by the stepped contact's fill (its gas pocket certified by
        :func:`step_wedge`, the hard closures and the upward base's check of the groove fill);
        without an admissible fill a resumable stop (``step-failed``) on that input."""
        from .ringcontact import step_wedge

        loops = self.pb.interface_loops(with_node_ids=True)
        P = np.column_stack([loops[0]["r"], loops[0]["z"]])
        n, c = len(P), int(corner["index"])
        reading = {"trigger": "stepped-inverted", "chain": "ring", "contact": "inverted",
                   "node_count": n, "local_indices": [(c + k) % n for k in range(-2, 3)],
                   "nodes": [(c - 2) % n, (c - 1) % n, (c + 1) % n, (c + 2) % n],
                   "corner": {k: v for k, v in corner.items() if k != "node_ids"},
                   "crossings": 0, "fold_retry": dict(self.state.fold_retry or {}),
                   "again": trigger}
        wedge, phase = (None, "unreadable")
        if loops[0]["node_ids"].tolist() == corner["node_ids"]:
            wedge, phase = step_wedge(P, reading, stop_elements=self.s.fold_stop_elements)
        reading["pre_step_phase"] = phase
        self._event("ring-step-corner", m, reading=reading)
        if wedge is None:
            return m, {"reason": "step-failed", "trigger": "stepped-contact-unhandled",
                       "chain": "ring", "error": "a retried step folded or inverted again at a "
                       "closing corner without an admissible fill", "contact": reading,
                       **self._taus(self._t())}
        return self._meet_event(m, self._local_ring_fill_request(P, wedge, reading, kind="step"))

    def _archive_corner_refusal(self, m: dict, found: dict) -> dict:
        """Retain a recoverable refusal outside the numbered surgery-event namespace."""
        folder = self.outdir / "events"
        folder.mkdir(parents=True, exist_ok=True)
        n = 1
        while True:
            stem = f"corner_refusal_{self.state.step:08d}_{n}"
            path = folder / f"{stem}.json"
            try:
                stream = path.open("x")
                break
            except FileExistsError:
                n += 1
        with stream:
            original = self.outdir / found["record"]
            record = json.loads(original.read_text()) if original.is_file() else {
                "outcome": "refused", "stop": dict(found)}
            kept = dict(found, original_record=found["record"],
                        record=f"events/{path.name}")
            if found.get("precut") is not None:
                source = Path(found["precut"])
                target = source.with_name(f"{stem}.dump")
                source.rename(target)
                if source.with_suffix(".json").exists():
                    source.with_suffix(".json").rename(target.with_suffix(".json"))
                kept.update(original_precut=str(source), precut=str(target))
                record["dumps"] = {**(record.get("dumps") or {}), "precut": str(target)}
            record["recovery"] = {"step": self.state.step, "refusal": kept}
            stream.write(json.dumps(_plain(record), indent=1) + "\n")
            stream.flush()
            original.unlink(missing_ok=True)
        self._event("ring-step-corner-refused", m, refusal=kept,
                    record=kept["record"], precut=kept.get("precut"))
        return kept

    def _meet_ring_step_contact(self, m: dict, contact: RingStepContact) -> tuple[dict, dict | None]:
        """Restore a simple input, then fill its certified gas wedge with hard budgets.

        A liquid or unreadable pocket remains a stop on the restored input. The
        crossed candidate is never handed to the surgery or kept as an accepted state.
        """
        from .ringcontact import loop_node_map, step_wedge

        discarded = self._distortion_flags()  # the rejected candidate cannot request a rebuild
        self._restore_step_snapshot()
        self._stop_cache = None
        m = self.measure()
        reading = dict(contact.reading)
        loops = self.pb.interface_loops(with_node_ids=True)
        wedge, phase = None, "unreadable"
        if len(loops) == 1 and not reading.get("unreadable"):
            P = np.column_stack([loops[0]["r"], loops[0]["z"]])
            mapping = loop_node_map(reading["node_ids"], loops[0]["node_ids"])
            if mapping is not None:
                by_id = {int(k): i for i, k in enumerate(loops[0]["node_ids"])}
                idx = [by_id[k] for k in reading["local_node_ids"]]
                if np.array_equal(P[idx], reading["before_local"]):
                    reading["local_indices"] = idx
                    wedge, phase = step_wedge(P, reading, stop_elements=self.s.fold_stop_elements)
        reading["pre_step_phase"] = phase
        self._event("ring-contact-step-rollback", m, reading=reading,
                    discarded_distortion=discarded,
                    rejected_diagnostics={"after": m["time"], "through": reading["attempted_time"]})
        if wedge is None:
            return m, {"reason": "self-contact-approach", "trigger": "stepped-contact-unhandled",
                       "chain": "ring", "contact": reading, **self._taus(self._t())}
        request = self._local_ring_fill_request(P, wedge, reading, kind="step")
        return self._meet_event(m, request)

    def _sheet_handled(self) -> bool:
        """Whether the ring's liquid lip is resolved and ruptured (decision 22): the ring stage
        before the ring's removal, jetcore's recentred far kinks, ``groove.enabled`` with
        ``groove.sheets`` ``log`` (a sheet read apart from the gas's folds), and the ring's
        calibrated noise floors, which the zero-removal budgets of its refine and the closures of
        its rupture are floored at."""
        ring, jet = self._ring(), self._jetcore()
        if ring is None or ring.get("removed") is not None or jet is None \
                or jet.far_kinks != "recentre":
            return False
        g, noise = self.events.groove, self.events.ring
        return bool(g.enabled and g.sheets == "log" and noise.sigma_V > 0.0
                    and noise.sigma_Pz > 0.0)

    def _sheet_rupture_thickness(self) -> float:
        """``h_s``: ``jetcore.sheet_rupture_floors`` (``SHEET_RUPTURE_FLOORS`` without it) times the
        run's floor."""
        floors = getattr(self._jetcore(), "sheet_rupture_floors", None)
        return float(floors if floors is not None else SHEET_RUPTURE_FLOORS) * self._h_floor()

    def _sheet_context(self, loops) -> object | None:
        """The ring stage's reading of the ring's lip for its stops (``ring.ring_stage_events``'
        ``sheet``), where it is handled (:meth:`_sheet_handled`; None otherwise): a callable of the
        ring's liquid fold ``f`` and the element length of its finer side ``h_fine`` that adds the
        state's sample of the lip's thickness (``[t, f.gap, f.r, f.z]``, once per time, the last
        ``sheetsurgery.LIP_SAMPLES``; a rebuild at the same time adds none; a section more than
        ``LIP_JUMP`` thicknesses from the last one's is another feature, and the history restarts
        from it; the refinement that holds the lip is kept alive from it, :meth:`_lip_hold`),
        and returns ``h_s`` (:meth:`_sheet_rupture_thickness`), the thinning reading
        (``sheetsurgery.lip_thinning``) and ``refine_limit``, ``SHEET_REFINE_ELEMENTS h_fine``
        while the lip's elements lie above the refine's floor (``refine_ratio h_floor``), zero
        otherwise."""
        if not self._sheet_handled() or len(loops) != 1:
            return None
        from .sheetsurgery import LIP_SAMPLES, lip_thinning

        st, t = self.state, self._t()

        def sheet(f, h_fine):
            samples = st.lip_samples
            if samples and len(samples[-1]) >= 4 and math.hypot(
                    f.r - samples[-1][2], f.z - samples[-1][3]) > LIP_JUMP * max(f.gap,
                                                                                samples[-1][1]):
                samples.clear()  # another feature: its own history
            if not samples or t > samples[-1][0]:
                samples.append([t, float(f.gap), float(f.r), float(f.z)])
                del samples[:-LIP_SAMPLES]
            floor = self.s.refine_ratio * self._h_floor()
            refine = SHEET_REFINE_ELEMENTS * h_fine if h_fine > floor else 0.0
            return {"h_s": self._sheet_rupture_thickness(), "refine_limit": refine,
                    "thinning": lip_thinning(samples)}

        sheet.h_s = self._sheet_rupture_thickness()  # the stage reads a lip up to h_s
        return sheet

    def _ring_sheet_reading(self, stop: dict | None) -> dict | None:
        """The ring's ``sheet-contact`` that is not due for its rupture read as a request to
        resolve the lip (:meth:`_sheet_handled`): the guard's reading at the pair's middle with
        ``L`` the element length of the lip's finer side and ``target`` ``max(h_floor,
        gap/SHEET_ELEMENTS)``, trigger ``sheet-contact``, the request kept as ``liquid_contact``
        (so that :meth:`_ring_refine` makes it as the liquid kink's refine, with its hard
        zero-removal budgets, :meth:`_ring_liquid_refine`) and as ``sheet``. Any other stop is
        returned as it is."""
        if stop is None or stop.get("reason") != EVENT_REQUEST or stop.get("kind") != SHEET_CONTACT \
                or stop.get("chain") != "ring" or stop.get("side") != "liquid" \
                or (stop.get("sheet_rupture") or {}).get("due") or not self._sheet_handled():
            return stop
        gap, h = float(stop.get("fold_gap", math.nan)), float(stop.get("h_fine", math.nan))
        if not (math.isfinite(gap) and math.isfinite(h)):
            return stop
        return {"reason": "interface-unresolved", "trigger": "sheet-contact", "chain": "ring",
                "L": h, "r": float(stop["fold_r"]), "z": float(stop["fold_z"]),
                "target": max(self._h_floor(), gap / SHEET_ELEMENTS), "turn": math.nan,
                "liquid_contact": dict(stop), "sheet": {"gap": gap, "h_fine": h}}

    def _ring_kink_reading(self, stop: dict | None) -> dict | None:
        """The ring's ``self-contact-approach`` (trigger ``fold-gap``) of contact ``kink`` on its gas
        side (or of no side read) read as the guard's reading at the element between the fold's
        pair (``index`` the node midway along the shorter way round, ``L`` the fold's arc per
        quadratic element it spans, two node steps each, so that ``L`` is the length of the
        elements at the kink whatever the pair's spacing, ``arc`` the arc itself, trigger
        ``fold-kink``, the stop kept as ``kink``), so that the fill at the floor meets the wedge of
        gas closing across a kink as it meets the guard's stops at the floor (r1h, h3 leg 1 at
        aa48432, step 7254: the wedge of gas at the rim of the radial sheet the downward head
        drives out along the plug, r 4.28e-4, its pair 2769/2771 two nodes apart, the gap 1.3e-7
        over an arc of 9.7e-7 between elements of 1.4e-6, which the fill of a groove, made for a
        fold across a resolved arc, does not take; r1f, f3 leg 1 at a100ccc, step 11580: the pair
        2613/2616 three nodes apart, its arc of 1.36e-6 one and a half elements of 9.1e-7 at the
        floor, which the arc itself, above ``refine_ratio h_floor`` = 1.17e-6, had kept from the
        fill). Any other stop is returned as it is."""
        if stop is None or stop.get("reason") != "self-contact-approach" \
                or stop.get("trigger") != "fold-gap" or stop.get("chain") != "ring" \
                or stop.get("contact") != "kink" or stop.get("side", "gas") != "gas" \
                or stop.get("fold_pair") is None:
            return stop
        loops = self._post_cut_state().get("loops") or []
        if len(loops) != 1:
            return stop
        n = len(loops[0]["r"])
        a, b = (int(x) % n for x in stop["fold_pair"])
        d = (b - a) % n
        apex = (a + d // 2) % n if d <= n - d else (b + (n - d) // 2) % n
        arc = float(stop.get("fold_arc", math.nan))
        steps = min(d, n - d)  # the node steps between the pair the shorter way round
        return {"reason": "interface-unresolved", "trigger": "fold-kink", "chain": "ring",
                "index": int(apex), "L": arc * 2.0 / max(2, steps), "arc": arc,
                "r": float(loops[0]["r"][apex]), "z": float(loops[0]["z"][apex]),
                "kink": {k: v for k, v in stop.items() if k != "reason"}}

    def _floor_fill_request(self, stop: dict | None) -> dict | None:
        """The fill at the floor of a wedge of gas of the ring (decision 22; ``groovesurgery``'s
        fill at the floor) for a stop the refine does not meet: the guard's ``interface-unresolved``
        on the ring (trigger ``edge-turn``, ``edge-midside`` or ``feature-below-floor``, or a
        rebuilt mesh's ``curved-jacobian`` read on the ring, :meth:`_ring_jacobian_reading`) at an
        element at the run's floor (at most ``refine_ratio h_floor`` long), in the ring stage before
        the ring's removal, with jetcore's recentred far kinks and ``groove.enabled``, where the
        element's apex is a wedge of gas read at the scale of the floor (``ring.floor_wedge``: the
        loop turning left across it, a pair across the wedge closer than ``fold_stop_elements``
        element lengths and ``ring.FLOOR_FOLD_RATIO`` times its gap below the arc between them), or,
        where no wedge is, a corner of gas open up to about a right angle (the same reading with
        ``ring.FLOOR_CORNER_RATIO``; ``corner`` in the wedge's record), as at the root of the radial
        sheet the downward jet's head drives out along the plug; a closing fold of the ring's gas
        across a kink at the scale of the floor is read as the guard's reading at the element
        between its pair (:meth:`_ring_kink_reading`, trigger ``fold-kink``).
        Returns the request (``ring-fold``, ``RF``, the fill as its handler, the pair, its gap and
        limit, ``contact`` ``floor``, and ``floor``: the guard's reading, the wedge and the fold
        limits the fill reads its trigger again with) for :meth:`_meet_event`; None otherwise (a
        right turn, the tip of a finger of liquid, stays the stop). A local crossing of the ring's
        rebuild spline is first offered to :meth:`_ring_spline_fill_request`, which verifies the
        gas wedge instead of assigning it the kink's phase or floor scale."""
        from .events import GROOVE_HANDLER, _request
        from .ring import FLOOR_CORNER_RATIO, FLOOR_FOLD_RATIO, floor_wedge

        spline = self._ring_spline_fill_request(stop)
        if spline is not None:
            return spline
        stop = self._ring_kink_reading(self._ring_jacobian_reading(stop))
        if stop is None or stop.get("reason") != "interface-unresolved":
            return None
        if stop.get("chain") != "ring" or stop.get("trigger") not in (
                "edge-turn", "edge-midside", "feature-below-floor", "curved-jacobian", "fold-kink"):
            return None
        ring, jet = self._ring(), self._jetcore()
        if ring is None or ring.get("removed") is not None or jet is None \
                or jet.far_kinks != "recentre" or not self.events.groove.enabled:
            return None
        L = float(stop.get("L", math.nan))
        if not (math.isfinite(L) and L <= self.s.refine_ratio * self._h_floor()):
            return None
        loops = self._post_cut_state().get("loops") or []
        if len(loops) != 1 or stop.get("index") is None:
            return None
        P = np.column_stack([loops[0]["r"], loops[0]["z"]])
        ratio = FLOOR_FOLD_RATIO
        wedge = floor_wedge(P, int(stop["index"]), stop_elements=self.s.fold_stop_elements)
        if wedge is None:  # a corner of gas at the floor, open up to about a right angle
            ratio = FLOOR_CORNER_RATIO
            wedge = floor_wedge(P, int(stop["index"]), stop_elements=self.s.fold_stop_elements,
                                ratio=ratio)
            if wedge is not None:
                wedge["corner"] = True
        if wedge is None:
            return None
        a, b = wedge["pair"]
        mid = 0.5 * (P[a] + P[b])
        fold = {"ratio": ratio, "stop": 0.0, "stop_elements": self.s.fold_stop_elements}
        guard = {k: v for k, v in stop.items() if k != "reason"}
        req = _request("ring-fold", "RF", chain="ring", contact="floor", fold_gap=wedge["gap"],
                       fold_r=float(mid[0]), fold_z=float(mid[1]), fold_arc=wedge["arc"],
                       fold_h_elem=wedge["h_elem"], fold_limit=wedge["limit"], fold_pair=[a, b],
                       kappa_g=self.events.gap.kappa_g, side="gas",
                       floor={"guard": guard, "wedge": wedge, "fold": fold})
        req["handler"] = GROOVE_HANDLER
        req.update(self._taus(self._t()))
        return req

    def _ring_refine(self, m: dict, stop: dict | None) -> tuple[dict, dict | None]:
        """Meet the guard's stop on the ring (:meth:`_ring_refine_due`):
        the ring is a nuisance (decision 22) whose features a rebuild may not represent at the size
        the far kink's rule gave them, as the rim of the thin sheet that the downward jet and the
        plug drive out of the column at the collision of their liquids, whose turn a rebuild at the
        yielded size concentrates into one element. The mesh is rebuilt at once, every region at
        the size the mesh has, the element in a far kink of its own (``kink_ring_r<n>_far``,
        marked ``refined``; the refined one whose reach, its radius and four chords, holds the
        element refined further, never coarser) about the element, its radius at least four of
        its chords and the radius of every far kink of the ring whose disc holds it (the region
        that yielded there, which stays as it is, as every other region does), at
        ``max(h_floor, f L)``, ``f = min(1/2, edge_turn_refine/turn)`` (one half for a midside
        node), the size at which an element of a circle would turn by ``edge_turn_refine``; it
        follows its feature keeping that size until the feature asks no more
        (``jetcore.far_kink_regions``), and counts as a yield (``far_yields``); while it remains,
        every far kink of the ring follows its own feature and a new feature has a region of its
        own (``kink_ring_m<n>_far``). A ring with several sharp features, a sheet's rim and the
        corners of a fill, so keeps each refined.
        At most ``RING_REFINE_TRIES`` rebuilds per state, each an event ``ring-refine`` with the
        guard's reading, the region and the base of the upward jet before and after it
        (``base_guard``; ``checked`` false where no jet stood before it, whose base there is none to
        keep); a base that moves beyond ``HEAD_FOLD_BASE_TOL`` stops the run
        (``interface-unresolved``, trigger ``ring-refine-base``). The stops of the new mesh are
        checked again; a stop not met is returned as it is. Returns the state the run goes on from
        and its stop."""
        tries = 0
        while tries < RING_REFINE_TRIES and self._ring_refine_due(stop):
            tries += 1
            reading = self._ring_reading(stop)
            if reading.get("liquid_contact") is not None or reading.get("trigger") == "fold-kink":
                m, stop = self._ring_liquid_refine(m, reading, tries)
            else:
                m, stop = self._ring_refine_once(m, reading, tries)
        if stop is not None and tries and stop.get("reason") == "interface-unresolved":
            stop["ring_refines"] = tries
        return m, stop

    def _ring_liquid_refine(self, m: dict, reading: dict, attempt: int) -> tuple[dict, dict | None]:
        """A topology-preserving ring refinement with hard zero-removal budgets.

        A failed balance or base check rolls back to the saved state. Reaching
        the floor remains a stop. The ring's lip is resolved through this route too
        (:meth:`_ring_sheet_reading`) and ruptured once it thins at ``h_s``
        (:meth:`_meet_sheet_rupture`).
        """
        from .ringcontact import closures, plain_sample, sample
        from .surgery import site_grid

        ring = self.events.ring
        grid = site_grid(reading["r"], reading["z"], ring.box_radius * self.reconnect.delta)
        before = sample(self.pb, grid, self.reconnect.reservoir_depth)
        saved = self.pb._snapshot_state()
        fresh = self._fresh_mesh()
        focus = self.pb.active_mesh_template().focus
        st = self.state  # run() holds this object too; output counters must never rewind
        kept = (copy.deepcopy(st.kinks), st.far_yields, st.last_remesh_step)
        old_gate = getattr(self, "_gate_stop", None)
        after, stop = self._ring_refine_once(m, reading, attempt)
        sampled = sample(self.pb, grid, self.reconnect.reservoir_depth)
        checks = closures(before, sampled, ring, grid)
        base = base_guard_checked(m, after)
        if not base["ok"]:
            checks["hard"].append("C_base")
        rec = {"reading": reading, "attempt": attempt, "before": plain_sample(before),
               "after": plain_sample(sampled),
               "closures": checks, "base": base, "step": st.step, **self._taus(self._t())}
        path = self.outdir / "events" / f"ring_contact_{st.step:08d}_{self.pb.remesh_count}_{attempt}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        relative = f"events/{path.name}"
        if checks["hard"]:
            self.pb._restore_state(saved)
            self.pb.active_mesh_template().focus = focus
            self.pb.invalidate_cached_mesh_data()
            st.kinks, st.far_yields, st.last_remesh_step = kept
            self._stepped_rc = None if fresh else self.pb.remesh_count
            gate = getattr(self, "_gate_stop", None)
            self._gate_stop = old_gate
            if gate is not None and gate is not old_gate:
                # The box gate kept the state BEFORE the rejected refinement.
                # Retain its bytes with a label that cannot masquerade as a live gate.
                pre = Path(gate["precut"])
                if pre.parent.resolve() == (self.outdir / "states").resolve() and pre.is_file():
                    target = pre.with_name(f"ring_contact_prestate_{st.step:08d}_{attempt}.dump")
                    for src, dst in ((pre, target), (pre.with_suffix(".json"), target.with_suffix(".json"))):
                        if src.exists():
                            if dst.exists():
                                raise RuntimeError(f"the rolled-back gate receipt already exists: {dst}")
                            src.rename(dst)
                    rec["rolled_back_gate"] = {"reading": gate, "prestate": f"states/{target.name}"}
            self._post_cut_cache = self._stop_cache = None
            after = self.measure()
            stop = {"reason": "interface-unresolved", "trigger": "ring-contact-unclosed",
                    "chain": "ring", "hard": checks["hard"], "record": relative,
                    **self._taus(self._t())}
        path.write_text(json.dumps(_plain(rec), indent=1) + "\n")
        self._event("ring-contact-refine", after, record=relative, closures=checks, base=base)
        return after, stop

    def _ring_refine_once(self, m: dict, stop: dict, attempt: int) -> tuple[dict, dict | None]:
        """One rebuild of :meth:`_ring_refine`, on the chain of the stop (its regions
        ``kink_<chain>_r<n>_far``)."""
        st, r = self.state, self.reconnect
        chain = stop.get("chain", "ring")
        L, rr, zz = float(stop["L"]), float(stop["r"]), float(stop["z"])
        if stop["trigger"] in ("edge-turn", "curved-jacobian"):  # a curved element: by its turn
            factor = min(0.5, r.edge_turn_refine / max(float(stop["turn"]), 1e-12))
        else:
            factor = 0.5
        target = max(self._h_floor(), factor * L)
        if stop.get("target") is not None:  # a sheet: its thickness over SHEET_ELEMENTS
            target = max(self._h_floor(), min(target, float(stop["target"])))
        # the ring's live far kinks whose disc holds the element (the region that yielded there):
        # their radius and the chord they met carried over; a refined one whose reach holds it is
        # refined further, else the refinement is a region of its own; no other region changes
        live = {k: g for k, g in st.kinks.items() if g.get("far") and g.get("chain") == chain
                and not g.get("released")}
        holding = [g for g in live.values()
                   if math.hypot(rr - float(g["r"]), zz - float(g["z"])) <= float(g["radius"])]
        key = next((k for k in sorted(live) if live[k].get("refined")
                    and math.hypot(rr - float(live[k]["r"]), zz - float(live[k]["z"]))
                    <= float(live[k]["radius"]) + 4.0 * L), None)
        if key is None:
            n = [int(x.group(1)) for x in (re.fullmatch(chain + r"_r(\d+)_far", k)
                                           for k in st.kinks) if x]
            key = f"{chain}_r{max(n, default=0) + 1}_far"
        name = f"kink_{key}"
        old = st.kinks.get(key)
        radius = max([4.0 * L] + [float(g["radius"]) for g in holding])
        t = self._t()
        st.kinks[key] = {"name": name, "chain": chain, "r": rr, "z": zz, "radius": radius,
                         "target": min(target, float(old["target"])) if old is not None
                         and not old.get("released") else target,
                         "h_before": max([L] + [float(g["h_before"]) for g in holding]),
                         "t_set": t, "released": False,
                         "turn": float(stop.get("turn", math.nan)),
                         "yields": (int(old.get("yields", 0)) if old is not None else 0) + 1,
                         "yield_t": t, "far": True, "refined": True}
        target = st.kinks[key]["target"]
        st.far_yields += 1
        template = self.pb.active_mesh_template()
        focus = template.focus
        region = FocusRegion(name, size=target, r=rr, z=zz, radius=radius)
        regions = [region if g.name == name else g for g in focus.regions]
        if focus.region(name) is None:
            regions.append(region)
        template.focus = Focus(size=min(g.size for g in regions),
                               radius=max(g.radius for g in regions), grading=focus.grading,
                               regions=tuple(regions))
        noise = self._box_noise_before()  # post.log_box_noise only
        gate = self._gate_before()  # jetcore.box_gate only
        self.pb.force_remesh()
        self._box_noise_after(noise, "ring-refine")
        gated = self._gate_after(gate, "ring-refine")
        st.last_remesh_step = st.step
        after = self.measure()
        self._keep_landings(after)
        base = base_guard_checked(m, after)
        reading = {k: stop[k] for k in ("trigger", "turn", "midside", "limit", "L", "r", "z",
                                        "index", "jacobian_bound") if k in stop}
        self._event("ring-refine", after, attempt=attempt, reading=reading, name=name,
                    target=target, radius=radius, yields=st.kinks[key]["yields"], base=base,
                    **self._axis_repair_note(), **gated)
        self._snapshot(after, "ring-refine")
        if not base["ok"]:
            return after, {"reason": "interface-unresolved", "trigger": "ring-refine-base",
                           "chain": chain, "base": base, "reading": reading,
                           "attempt": attempt, **self._taus(self._t())}
        return after, self._stop(after)

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
        self._tick()  # step timing only
        failed_step_context = None
        while True:
            found = self._stop(m)
            if self._axis_repair_due(found):  # a jet's tip folded along the axis (nuisance)
                m, found = self._axis_repair(m, found)
            m, found = self._tip_edge_straighten(m, found)
            if self._ring_refine_due(found):  # a fresh mesh of the ring short of its feature
                m, found = self._ring_refine(m, found)
            found = self._floor_fill_request(found) or found  # a wedge of gas at the floor
            found = self._unresolved_neck_request(found) or found  # on a far head
            if found is not None and found["reason"] == EVENT_REQUEST:  # the events block only
                m, found = self._meet_event(m, found)
                if found is None:  # made: every stop again on the fresh mesh
                    continue
            if found is not None:
                stop = (dict(found, failed_step=failed_step_context)
                        if failed_step_context is not None else found)
                break
            t = self._t()
            if s.t_end - t <= max(1e-14, 8 * math.ulp(s.t_end)):
                stop = {"reason": "end-time"}
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
            if st.fold_cap is not None and st.fold_cap["t"] == t:  # a folded step's retry
                dt = min(dt, st.fold_cap["dt"])
            if self._jetcore_rows is not None and r is not None:  # what set the step, read only
                self._dt_limiter = self._dt_limiter_of(dt, m, t, focus_courant)
            if dt < self._dt_min():
                stop = {"reason": "dt-floor", "dt": dt}
                break
            st.ell_prev, st.t_prev = m["ell"], t
            self._tick("top")
            wall0 = time.time()
            # temporal_error engages pyoomph's adaptive stepping: without an error estimator every
            # converged step is accepted, while a failed Newton solve is rejected and halved
            tol = max(newton_tolerance, s.newton_floor / dt)
            self._step_newton_tolerance = newton_tolerance  # each sub-step's: _post_cut_solve
            defer = self._defers_distortion()
            rebuilds = pb.remesh_count
            self._log_mark = getattr(pb, "oomph_log_size", lambda: None)()  # this step's log
            try:
                with self._post_cut_solve():
                    pb.run(t + dt, timestep=dt, maxstep=dt, outstep=False, temporal_error=1.0,
                           newton_solver_tolerance=tol)
            except RingStepFold as fold:
                if not fold.reading.get("consumed"):
                    m = self._meet_ring_step_fold(m, fold, t, dt)
                    continue
                m, found = self._meet_ring_fold_again(m, fold)  # never accepted
                if found is None:
                    continue
                stop = (dict(found, failed_step=failed_step_context)
                        if failed_step_context is not None else found)
                break
            except RingStepContact as contact:
                m, found = self._meet_ring_step_contact(m, contact)
                if found is None:
                    continue
                stop = (dict(found, failed_step=failed_step_context)
                        if failed_step_context is not None else found)
                break
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
                        failed = self._failed_step_stop(exc, dt, rebuilds)
                        if failed is not None:  # rolled back: a resumable stop on that state
                            m, stop = failed
                            failed_step_context = dict(stop)
                            corner = self._inverted_corner(exc)
                            if corner is not None and self._fold_retry_due(
                                    {"key": self._ring_input_key()}):
                                # pyoomph's net gave up at a closing gas corner: the one retry
                                fold = RingStepFold(dict(corner, trigger="step-failed-inverted"))
                                m = self._meet_ring_step_fold(m, fold, t, dt, restored=True)
                                continue
                            if corner is not None:  # the retry failed there too: fill the corner
                                m, found = self._meet_ring_corner(m, corner)
                                if found is None:
                                    continue
                                no_fill = (found.get("reason") == "step-failed"
                                           and found.get("trigger") == "stepped-contact-unhandled")
                                if not (no_fill or found.get("reason") == GROOVE_FILL_REFUSED):
                                    stop = dict(found, failed_step=failed_step_context)
                                    break
                                # An unfillable native inversion keeps the old ring-refine
                                # recovery. Folded candidates never reach this failure path.
                                if found.get("reason") == GROOVE_FILL_REFUSED:
                                    found = self._archive_corner_refusal(m, found)
                                stop["corner_refusal"] = failed_step_context["corner_refusal"] = dict(found)
                            retry_state = (copy.deepcopy(st.kinks), st.far_yields,
                                           st.last_remesh_step)
                            retry_tip = (len(self._tip_edge_balances), st.snapshots,
                                         st.last_snapshot_t, st.last_snapshot_ell)
                            retry_focus = pb.active_mesh_template().focus
                            try:
                                retried = self._failed_step_retry(m, stop)
                            except Exception as retry_exc:  # a failed rebuild keeps the rollback
                                st.kinks, st.far_yields, st.last_remesh_step = retry_state
                                discarded = self._tip_edge_balances[retry_tip[0]:]
                                del self._tip_edge_balances[retry_tip[0]:]
                                st.snapshots, st.last_snapshot_t, st.last_snapshot_ell = retry_tip[1:]
                                pb.active_mesh_template().focus = retry_focus
                                self._restore_step_snapshot()
                                stop["retry_error"] = (
                                    f"{type(retry_exc).__name__}: {retry_exc}")[:400]
                                for balance in discarded:
                                    path = self.outdir / balance["record"]
                                    rec = json.loads(path.read_text())
                                    rec.update(outcome="rolled-back", retry_error=stop["retry_error"])
                                    path.write_text(json.dumps(_plain(rec), indent=1) + "\n")
                                if discarded:
                                    self._event("tip-edge-rollback", m,
                                                records=[x["record"] for x in discarded],
                                                error=stop["retry_error"])
                                m = self.measure()
                                self._event("step-retry-failed", m, error=stop["retry_error"])
                                break
                            if retried is not None:
                                m, retry_stop = retried
                                if (retry_stop is not None
                                        and retry_stop.get("trigger") in (
                                            "ring-refine-base", "tip-edge-unclosed")):
                                    stop = dict(retry_stop, failed_step=failed_step_context)
                                    break
                                # Guard stops and event requests are re-derived and met at the top.
                                continue
                            break
                    raise
                stop["dt_attempted"] = dt
                if defer and flagged:
                    # sub-steps accepted before the failure asked for a rebuild that the driver
                    # makes only after the step: the failure came on the mesh they flagged
                    stop["distortion_flagged_substeps"] = len(flagged)
                break
            self._tick("solve")
            st.step += 1
            failed_step_context = None
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
            if self._gate_stop is not None:  # jetcore.box_gate refused the rebuild for distortion
                stop = dict(self._gate_stop)
                break
            if defer and inside and r is not None:
                self._event("remesh", m, reason="inverted-element", rebuilds=inside)
            self._tick("measure")
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
            self._tick("rows")
            found = self._stop(m)
            if self._axis_repair_due(found):
                m, found = self._axis_repair(m, found)
            m, found = self._tip_edge_straighten(m, found)
            if self._ring_refine_due(found):  # after a rebuild for distortion
                m, found = self._ring_refine(m, found)
            found = self._floor_fill_request(found) or found
            found = self._unresolved_neck_request(found) or found
            if found is not None and found["reason"] == EVENT_REQUEST:  # the events block only
                m, found = self._meet_event(m, found)
                if found is None:  # made: its rebuild stands for the one a stop cancelled
                    self._distortion_skipped = 0
                    self._tick("stops")
                    continue
            if found is not None:
                if self._distortion_skipped:  # the rebuild of this step, cancelled by this stop
                    found["distortion_rebuild_skipped"] = self._distortion_skipped
                stop = found
                break
            self._tick("stops")
            if r is not None and self._edge_turn_max() > r.edge_turn_refine:
                self._snapshot(m, "edge-turn")  # every step while an element bends this much
            else:
                self._maybe_snapshot(m)
            self._tick("snapshot")
            m, found = self._remesh_checks(m)
            if self._axis_repair_due(found):
                m, found = self._axis_repair(m, found)
            m, found = self._tip_edge_straighten(m, found)
            if self._ring_refine_due(found):  # after the rule's rebuild
                m, found = self._ring_refine(m, found)
            found = self._floor_fill_request(found) or found
            found = self._unresolved_neck_request(found) or found
            if found is not None and found["reason"] == EVENT_REQUEST:  # the events block only
                m, found = self._meet_event(m, found)
                if found is None:
                    continue
            if found is not None:
                stop = found
                break
            self._tick("remesh")
            self._commit_post_cut(m)
            self._tick("commit")
            if self._t() - st.last_state_t >= self._state_interval():
                self.save_state()
            self._write_timing(dt)
        if failed_step_context is not None:
            stop.setdefault("failed_step", failed_step_context)
        # with the events block, a cut whose hard budgets do not close leaves the problem holding
        # a cut that is not accepted: nothing more is read from it, and the run ends on the state
        # before the event, which the cut kept
        unclosed = stop.get("reason") in (EVENT_UNCLOSED, "box-gate")
        if r is not None:  # after a cut the state the run ends on is a dump too
            if not unclosed:
                self._commit_post_cut(m)
            if stop.get("reason") in EVENT_STOPS:  # with the events block only
                if not stop.get("precut"):  # unless the cut of the event kept it
                    stop["precut"] = str(self._keep_event_state(stop))
                record = self._write_stop_record(m, stop)
                if record is not None:
                    stop["record"] = f"events/{record.name}"
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
                          "axis_midside_repairs": getattr(pb, "axis_repairs_total", 0),
                          **({"axis_fold_repairs": len(self._axis_repairs)}
                             if self._axis_repairs else {}),
                          **({"tip_straightens": len(self._tip_straightens)}
                             if self._tip_straightens else {})}
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
        is lossless either way. The dumps the driver writes keep their compression. The first
        snapshot of the step is kept until the next (``_step_snapshot``): the state a failed step
        is rolled back to (:meth:`_failed_step_stop`)."""
        pb, r = self.pb, self.reconnect
        if r is None or not hasattr(pb, "release_newton_tolerance"):  # a FaradayProblem only
            yield
            return
        self._step_snapshot = None  # release the previous full snapshot before making this one's
        self._step_was_fresh = self._fresh_mesh()
        level = pb.states_compression_level
        pb.min_newton_iterations = r.newton_min_iterations
        pb.states_compression_level = None
        pb._step_snapshots = []  # the state the step starts from, as the net snapshots it
        watched = getattr(self, "events", None)
        before = (self._ring_step_input() if watched is not None and watched.groove.enabled
                  and self._refined_chain() == "ring" else None)
        tolerance = getattr(self, "_step_newton_tolerance", None)
        if tolerance is not None and hasattr(pb, "newton_floor_rule"):
            pb.newton_floor_rule = (tolerance, self.s.newton_floor)
        try:
            yield
            if before is not None and pb._step_snapshots:
                self._ring_step_check(before)
        finally:
            if hasattr(pb, "newton_floor_rule"):
                pb.newton_floor_rule = None
            pb.min_newton_iterations = 1
            pb.release_newton_tolerance()
            pb.states_compression_level = level
            self._step_snapshot = pb._step_snapshots[0] if pb._step_snapshots else None
            pb._step_snapshots = None

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
        """The largest turn of the tangent across a quadratic element of either chain (degrees);
        with ``jetcore.edge_turn_snapshots`` ``core``, of the core's elements only (those of the
        open chain whose midside node is a core node, ``jetcore.core_mask``)."""
        pc = self._post_cut_state()
        jet, core = self._jetcore(), pc.get("core")
        if jet is not None and core is not None and jet.edge_turn_snapshots == "core":
            e = (pc["edges"].get("open") or {}).get("edges")
            if e is None:
                return 0.0
            mask = np.asarray(core["mask"], dtype=bool)
            mid = np.clip(np.asarray(e["index"], dtype=int), 0, len(mask) - 1)
            turns = np.asarray(e["turn"], dtype=float)[mask[mid]]
            turns = turns[np.isfinite(turns)]
            return float(turns.max()) if len(turns) else 0.0
        turns = [rd["turn"] for rd in pc["edges"].values() if math.isfinite(rd["turn"])]
        return max(turns, default=0.0)

    def _failed_step_stop(self, exc: BaseException, dt: float,
                          rebuilds: int) -> tuple[dict, dict] | None:
        """After a cut, a step that failed (Newton at every step size oomph-lib tried, or
        pyoomph's net for an inverted element giving up after its rebuilds): the problem rolled back
        to the state the step started from (the snapshot the net took before it, kept by
        :meth:`_post_cut_solve`), measured again, and the stop ``step-failed`` (the error, the step
        tried, the rebuilds the net made), so that the run ends on that state with its end state
        dumped, a resumable stop, instead of failing without one (r1 at b46713a, step 7503: three
        rebuilds of the net and no end state). None without a snapshot (the net not armed): the
        failure is raised as before."""
        snap = getattr(self, "_step_snapshot", None)
        if snap is None:
            return None
        pb = self.pb
        inside = pb.remesh_count - rebuilds
        self._restore_step_snapshot()
        m = self.measure()
        stop = {"reason": "step-failed", "error": f"{type(exc).__name__}: {exc}"[:400],
                "dt_attempted": dt, "net_rebuilds": int(inside), **self._taus(self._t())}
        self._event("step-failed", m, error=stop["error"], dt_attempted=dt,
                    net_rebuilds=int(inside))
        return m, stop

    def _restore_step_snapshot(self):
        """Restore the failed step's input and its mesh age. Loading a dump increments pyoomph's
        remesh counter even when it restores a mesh that had already accepted a step."""
        self.pb._restore_state(self._step_snapshot)
        self._stepped_rc = None if self._step_was_fresh else self.pb.remesh_count
        self.pb.invalidate_cached_mesh_data()
        self._post_cut_cache = None

    def _failed_step_retry(self, m: dict, stop: dict) -> tuple[dict, dict | None] | None:
        """After a failed step rolled back (:meth:`_failed_step_stop`), one recovery per step.

        Before the ring, only a native inversion failure can admit the downward terminal
        dip to the existing signed tip audit. A refused admission keeps the original
        failed-step stop; a failed audit keeps its hard stop and restored input. Ordinary
        states never enter that route. In the ring stage with
        jetcore's recentred far kinks: one retry per step, the ring's sharpest element (turning by
        more than ``edge_turn_refine`` and above the guard's floor) rebuilt in a far kink of its own
        at the size its turn asks for, as the ring refine does (:meth:`_ring_refine_once`, event
        ``ring-refine`` with ``retry`` in its reading), and the step made again from the rebuilt
        mesh (r1h at b36c9c1, step 7251: the root of the radial sheet the downward head drives out
        along the plug, an element of 1.55e-5 turning by 85.6 degrees, below the guard's stop, and
        the step from it failed at every size). Returns the state and the stop of the rebuilt mesh,
        from which the loop's top goes on (its guard/event stops read again; a non-rederived
        ring-refine-base stop ends the run), or None where no retry is made (a
        second failure of the same step, no ring, no element to refine), the stop ``step-failed``
        then standing."""
        st, r = self.state, self.reconnect
        if getattr(self, "_retried_step", None) == st.step:
            return None
        if self._refined_chain() != "ring":
            if not stop.get("error", "").startswith("InvertedElementRemeshRequest:"):
                return None
            count = len(self._tip_edge_balances)
            self._retried_step = st.step
            after, found = self._tip_edge_straighten(m, None, failed_step=stop)
            if len(self._tip_edge_balances) > count or (found is not None
                    and found.get("trigger") == "tip-edge-unclosed"):
                return after, found
            if found is not None:
                stop["tip_edge_retry"] = found
            return None
        edges = (self._post_cut_state().get("edges") or {}).get("ring") or {}
        e = edges.get("edges")
        if e is None or not len(e["L"]):
            return None
        floor = self.s.refine_ratio * self._h_floor()
        ok = ((e["turn"] > r.edge_turn_refine) & (e["turn"] <= RING_REFINE_TURN_MAX)
              & (e["L"] > floor))
        if not ok.any():
            return None
        k = int(np.argmax(np.where(ok, e["turn"], -1.0)))
        reading = {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "ring",
                   "turn": float(e["turn"][k]), "limit": float(r.edge_turn_refine),
                   "L": float(e["L"][k]), "r": float(e["mid"][k, 0]), "z": float(e["mid"][k, 1]),
                   "index": int(e["index"][k]), "retry": True}
        self._retried_step = st.step
        self._event("step-retry", m, reading=reading, error=stop.get("error"))
        m, new_stop = self._ring_refine_once(m, reading, 1)
        return m, new_stop

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
        if cached and cached.get("loops"):
            self._write_ring_profile(path, header, cached["loops"])
