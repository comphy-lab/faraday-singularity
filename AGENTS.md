# faraday-singularity

Sharp-interface pyoomph FEM of the singular collapse of a Faraday-wave interface and the
Worthington jet it launches. The aim is to test the purely inertial self-similar theory of jet
birth, `r_j ∝ τ^α(β)`, against inertio-capillary scaling (exponent 2/3), and to resolve how
the jet speed diverges as the wave height approaches its singular value. The cone law holds only
where the velocity potential vanishes on the cone in `r_j ≪ r ≪ 1` at inception, and `We_j`
diverges in it only for `β` below Taylor's 49.29°; a truncated cone fed by a constant far-field
flux follows the line-sink law `r_j ∝ τ^(1/2)` instead. See `README.md` for the motivation and
references.

**Status:** axisymmetric solver at `Bo = 0` with linear verification cases (free decay and the
Faraday threshold of the `J0` mode), and a collapse driver (`src/faraday_singularity/driver.py`)
that writes the collapse and jet observables during the run (apex curvature, jet-base radius and
speed, cone angle, running estimates of the singular time from the apex and, from the jet-base
radius, of the origin of the jet's growth or of the time a closing base would reach the axis, crest
and trough episodes of the axis) and stops before any pinch-off. The production runner can
continue a run through a neck pinch-off by an explicit cut at the start of a restart
(`src/faraday_singularity/surgery.py`, the post-cut mode of the driver); it is tested on a synthetic
necked cavity only, not yet on a production collapse, and no study in the cut radius has been run. The repository
holds no collapse case files or results yet.

## Structure

```
faraday-singularity/
├── src/                    # reusable pyoomph code, as src/<package>/
├── simulationCases/        # entry points and pyoomph-case-v1 JSON cases
├── testCases/              # small software-contract tests
├── verificationCases/      # exact or independent solutions (e.g. linear Faraday theory)
├── validationCases/        # independent experimental or numerical data
├── modelComparisonCases/   # inertial-cone, line-sink and inertio-capillary similarity laws
├── postProcess/            # analysis and rendering
└── docs/                   # approved authored documents only
```

The canonical runner is `simulationCases/run_faraday.py`; `run_case.py` retains the compatibility entry point. Each evidence folder owns its own case files; do not
add a second top-level `cases/` tree.

## Starting point

- The solver adapts the surfactant-free path of the CoMPhy `faraday-surfactant-fem` code
  (container-frame forcing, free surface with kinematic and dynamic conditions, case-file runner)
  to an axisymmetric cylinder with an unstructured, hyperelastically smoothed mesh that is rebuilt
  from the live free surface (`src/faraday_singularity/`).
- Copy and adapt code deliberately, recording the source commit in the commit message. Do not
  import across repositories, and do not edit `faraday-surfactant-fem` from this repository.

## Environment

- pyoomph is pinned in `pyproject.toml` and `uv.lock` to the CoMPhy release `comphy-2026.09.21.1`
  (`f912ac10`), the same release as `faraday-surfactant-fem`, built from source with MPI. Build and
  verify it with the `pyoomph-env` workflow; never upgrade the pin implicitly.
- Open MPI 4 names its shared-memory transport `vader`, Open MPI 5 `sm`; `run_case.py` selects it.
- Local environment and build directories (`.venv*/`, `.pyoomph-builds/`) are ignored and
  never committed.

## Conventions

- Nondimensionalise in one parameters module (`parameters.py`: length `1/k`, capillary time,
  groups `Oh`, `Bo`, `Gamma`, `Omega`). Record the dimensionless groups with every run.
- The container radius is a zero of `J1` and the side wall is free slip with a sliding 90 degree
  contact line, so `J0(r)` is an exact linear mode; keep this when changing boundary conditions, or
  the verification comparator no longer applies.
- `testCases/` holds contract tests (`pytest testCases`); `verificationCases/README.md` states the
  verification acceptance criteria.
- For a mesh study, scale every mesh length together (surface and axis size, growth rate and the
  bulk cap); refining only the surface size leaves the bulk discretisation unchanged. Separate
  spatial from time-step error with `initial_value.response_bdf2`, since BDF2 dispersion
  otherwise dominates the linear comparison.
- The observable decides the solver. Before any production run, confirm that the solver writes
  the interface near the axis, the jet-base radius and position, the jet speed, the cone
  semi-angle β and the singular time, at the cadence needed for the intended exponent fit.
- A sharp-interface mesh cannot change topology by itself. Stop each simulation before any
  pinch-off or entrainment, and record why it stopped. The one exception is the reconnection
  through a neck pinch-off (next item). Before a cut, of failures of the solve, only oomph-lib giving up below
  its smallest step (the Newton solve failing at every step size) can end a run cleanly, and only
  when `classify_failure` attributes it to a closing neck or a vanishing local scale, as measured
  rather than growth-limited; any other failure, including a failed remesh, is a failed run. Runs also end cleanly at the neck-radius
  stop, the conservative fold-gap stop (`self-contact-approach`), the end time, the step-size
  floor and the step limit, and, only when a case sets a finite `post_birth_aspect`, once the born
  jet is unambiguous (`jet-confirmed`); such a run records jet birth but not jet formation. A fold-gap stop can also truncate a static sharp corner; it is not
  proof of entrapment or a singularity. `InterfaceSelfIntersection` is a failed run. Check stop
  criteria before advancing and after every remesh, before writing a state dump; after a cut also
  before every rebuild, the driver's rebuilds for distortion included, so that no rebuild meets a
  spline through the nodes that contacts itself (that is the clean `self-contact-approach` stop
  with trigger `spline-contact`; its `contact` says `fold`, a fold of the resolved surface, or
  `kink`, at the scale of the mesh).
- Reconnection through a neck pinch-off is made by restarts of the production runner
  (`simulationCases/run_faraday.py`; exact commands in `README.md`), never by a script of its own:
  (1) landing states, from a restart of an existing run under a derived case that adds
  `collapse.landing` (the first state with `delta/r_n >= 1.05` for each cut radius; a radius below
  1.05 times the neck-stop radius is refused before any output, but not in a cut or post-cut
  restart, where the inherited `landing` is inactive; `landing-missed` records one left
  without a state) or `keep_stop_state`, and a lower `neck_stop` if needed (`run_case.DERIVED_KEYS`:
  what the run keeps and where it stops, not the equations it solves, the dump cadence `state_dt`
  and `state_keep` (at least 1) and `collapse.reconnect.tau_end` included); (2) one cut run per
  `delta`, a restart from the landing state of that `delta` under the landing case plus a
  `collapse.reconnect` block and a `state_dt` for the run after the cut (`restart_mode` `cut`,
  `run_case.CUT_DERIVED_KEYS`); (3) continuations from any dump written after the cut (the `postcut`
  dump, the rotating dumps, the dump of the state the run ended on) under its case or one derived
  from it (`post-cut`). The `precut` dump of a cut run is a landing state: a restart from it under a
  case with a reconnect
  block cuts again. `run_case.restart_admission` decides the mode and checks it
  before output: the landing state's recorded `r_n` must put `delta/r_n` within the cut's
  `[min_landing_ratio, max_landing_ratio]` (a window that contains the landing window [1.05, 1.1]),
  and a post-cut continuation needs the reconnect settings the cut recorded but for `tau_end`
  (`reconnect.CONTINUATION_KEYS`), which it may raise, and a time after the cut before that
  `tau_end`. A reconnect block the
  parent did not have changes the trajectory at round-off even before a cut, so an ordinary restart
  may not add, drop or change it (`tau_end` aside). The cut (`surgery.cut`): pyoomph's planner gives the topology; its
  caps are always replaced by tangent arcs spliced at the first survivor beyond the planner's window
  (`cap_window_factor`) with a radius of at least `delta`, which sets the geometry (`in_cut_radii`;
  on slender threads at `delta/r_n = 1.05` a bridge of 13-14.5 `delta` and 55-58 `delta^3` of
  removed gas), so a `delta` study needs a sensitivity to that window; the tips, the bridge and the
  focus of the cut's remesh are sized in cut radii (`tip_size`, `bridge_size`, `focus_size`,
  `focus_radius`), not by the landing run's focus, except that the bubble's bottom, the landing's
  cavity bottom with its jet, keeps the landing focus where that is finer (`focus.kept`); gmsh's
  sizing from curvature is off after a cut by default (`curvature_sizing` 0): it reads the curvature
  of the spline through the nodes, which a jump in node spacing makes spurious, and refines there
  ungraded, which widens the jump rebuild after rebuild until the spline loops; the velocity is
  carried across and projected by a
  pressure impulse, refused unless the projection converges and the discrete energy identity holds
  (`strong_divergence_L2` is an order-`h` diagnostic, not a test); the bubble keeps the passive
  constant gas pressure (the zero-density limit). A refused or failed cut, whatever failed in it,
  ends the run as `failed` with its partial record and `failed_at` in `cut_diagnostics.json` and
  under `cut` in `run_status.json` (with the precut dump once written); never load the precut state
  back into that problem, and an interruption is never turned into an outcome of the cut. After
  the cut the collapse driver runs in its post-cut mode (`reconnect` settings): the open chain is
  measured as any surface and the jets of the bubble's two tips in the same way (the bubble's top
  mirrored; `r_j`, `z_b`, `Q_j`, `V_j`, `We_j` and `jet_born` of all three tips in `chains.txt`,
  `Q_j` and `V_j` positive along each jet), the local scale is the smallest over both chains (on
  the bubble its jets, neck and length, not the radii of curvature of its tips, which depend on the
  mesh at a chain's end) and the radius of curvature at the base of each jet, the apex of the gas
  groove around it (`R_apex`: circles fitted through vertices only, the smaller over 2 and 4 on
  either side); the focus is floored at `h_floor`, and the mesh follows each feature of both chains
  in its own region at its own scale (`reconnect.feature_regions`): the tips the cut made at
  `clip(ell_tip/elements_per_scale, h_floor, h_cut)` (`ell_tip` the smaller of the tip's radius of
  curvature and its jet's radius; `h_cut`, the cut's focus size, is the coarsest size allowed, not
  a fixed one) within `focus_extent ell_tip`, the rest of the cut's region (`focus_radius` cut radii
  and the jet) kept at `h_cut` (`<tip>_cut`); the base of each jet (`jet_base_<tip>`) at
  `clip(R_pred/elements_per_scale, h_floor, h_cut)` within `max(apex_extent R_apex, 4 target)`,
  `R_pred` the apex radius `apex_lookahead` steps ahead (at least half the reading; at the bubble's
  bottom capped at that bottom's own size, and none where the cap is reached); the bubble's bottom
  at its jet's size but never coarser than at the landing; each neck at its own scale; and a
  `kink_<chain>` region where the interface guard asks for one (`reconnect.kink_regions`). A rebuild
  refines or coarsens each region by the bounded factors when its longest element exceeds
  `refine_ratio`, or its shortest (outside the finer regions within it) lies below `coarsen_ratio`,
  times its target, a rebuild that changed none of the regions that asked waits `remesh_backoff`
  steps, and a rebuild for distortion keeps every region's size (`h_<tip>`, `size_<tip>` in
  `chains.txt`, the regions in the `remesh` events). The interface is read on its quadratic
  elements after every accepted step and on the input of every rebuild (`reconnect.p2_edges`,
  `interface_guard`): a tangent turning by more than `edge_turn_refine` (30) degrees across an
  element, or a midside node more than `midside_refine` (0.1) chords off mid-chord, asks for a
  refinement; beyond `edge_turn_stop_fresh` (90) on a freshly built mesh or `edge_turn_stop` (120) on
  any, beyond `midside_stop` (0.2), or where the element is at `h_floor` already, the run stops
  cleanly (`interface-unresolved`, trigger `edge-turn`, `edge-midside` or `feature-below-floor`). A
  rebuilt mesh is accepted only if the Bezier lower bound of the scaled Jacobian of every curved
  element reaches `jacobian_min` (0.2; trigger `curved-jacobian`, `reconnect.p2_jacobian_bounds`),
  gmsh's high-order optimiser is asked for `high_order_threshold` (0.3), an element below the
  absolute quality `quality_floor` (0.15) asks for a distortion rebuild, and pyoomph's net for an
  element turned inside out within a step is armed (`inverted_element_remesh`). Before that check,
  every rebuild after a cut puts back on its edge any midside node of the axis that the transfer of
  the solution (which interpolates node positions too) left more than 0.1 edge lengths off its
  middle, at the current and history levels (`FaradayProblem.repair_axis_midsides`; `axis_repair`
  in the `remesh` event, `axis_midside_repairs` in the run result). The stops before
  a second change of topology apply on both chains (`reconnect.post_cut_stops`: the fold stop on
  both chains, a fold pair that straddles the base of a jet being the `annular-contact-approach`,
  trigger `fold-gap`, with the jet, its apex and `R_apex`; a spline that contacts itself; the guard
  last) and at `tau_end`; the step also follows every node of both chains (`reconnect.courant_limit`)
  and the capillary time of the sharpest tip or jet base, and, with `focus_courant` (default),
  the collapse's rule `courant h_focus/U`; each step makes at least `newton_min_iterations` (2)
  Newton iterations whatever its tolerance (`FaradayProblem.min_newton_iterations`, set by the
  driver around each step: the tolerance is held at zero up to the test after the last required
  iteration); a new neck stops the run (`secondary-pinch-approach`) only `neck_depth` (0.25)
  below the lower of the widest points on either side of it along its chain, or below
  `neck_elements h_min` (`reconnect.chain_necks`; `neck_side` gas or liquid); with the events
  block a gas neck above `neck_elements h_min` must also be `neck_depth` below its adjacent
  shoulders, the largest `r` on each side before `r` falls below the neck's
  (`reconnect.adjacent_shoulders`), so a stationary pair on a nearly vertical wall is no
  `gas-neck`, while one below `neck_elements h_min` is a `gas-neck` whatever either depth, and the
  request records both depths; no Newton failure ends the run cleanly, and the profiles of the
  state the failing step started from are written first (`profile_<step>_lastgood{,_bubble}.txt`);
  with pyoomph's net for inverted elements armed, a failed step is rolled back to the state it
  started from (the net's snapshot before it, kept by `FaradayProblem._snapshot_state` within
  `CollapseDriver._post_cut_solve`) and ends the run as the resumable stop `step-failed` with that
  state dumped (`CollapseDriver._failed_step_stop`, event `step-failed`; `test_step_failure.py`,
  `test_step_failure_fem.py`, `test_step_failure_identity.py`), and a profile is written at every
  step while an element turns beyond `edge_turn_refine`; a loop of the ring whose spacing jumps
  beyond `ring.LOOP_STEP_RATIO` is in order when it is simple (`ring.validate_loop_order`); with
  jetcore's recentred far kinks, a guard stop on the bubble before the piercing at the tail of one
  of its ends folded along the axis (`axisrepair.axis_fold`: nodes beside the end within
  `AXIS_FOLD_TOL` of the spacing of the axis or across it) is met by the axis repair
  (`axisrepair.axis_repair`, `CollapseDriver._axis_repair`; decision 22, the downward jet's tip a
  nuisance): the folded nodes dropped and the axis node put where the surface meets the axis by a
  rebuild from the planned chains, the sliver filled, its volume and momentum hard above
  `AXIS_REPAIR_BOUND delta^3` for volume (`A_V`), and that bound times the folded nodes'
  largest speed for the sliver's momentum (`A_P`), integrated on its own triangle quadrature;
  every quadrature point located and its volume within the existing R1 volume tolerance, otherwise
  a hard `A_P_coverage` miss; the box-grid sample retained for A2; the transfer's closures A1-A6 as a fill's, the
  upward jet's base checked (`A_base`); the transfer checked as a fill's against the old chains,
  unlocated nodes only in the sliver and its band, chain names and ends kept, axis spans as planned;
  event `axis-repair`, record `events/axis_repair_<n>.json`,
  `axis_fold_repairs` in the run's result; a miss or a failed rebuild rolled back to the state
  before it, the resumable stop `axis-repair-unclosed` or `axis-repair-failed`, and a fold again
  at the same end within `AXIS_REPAIR_RECUR_STEPS` steps left the stop it is (`axis_fold_recurs`);
  `test_axis_repair.py`, `test_axis_repair_identity.py`; rotating dumps
  follow at least each time `tau` doubles and the end state is dumped; and every output carries `tau = t - t_cut` and `tau_p = t - t_p` (`collapse.txt`,
  `chains.txt` with `R_apex_<tip>`, `groove_gap_<tip>`, `groove_angle_<tip>`, `edge_turn_<chain>`
  and `edge_midside_<chain>`, profiles of both chains, events, dumps). `rebuild_spline` `vertices`
  (off by default) splines the rebuilds through the vertices only, for sensitivity tests; smoothing
  the default would hide the very feature under test. All these settings are cut keys: a
  continuation keeps them, and a state cut before they existed is cut again, not continued.
  Loading a state written after the cut restores the cut's mesh sizing, with gmsh's threshold for
  curved elements and the nodes of the rebuild's spline (`surgery.restore_cut`). One cut per run. A reconnected result
  counts only if the jets at a fixed time after the pinch time `t_p` do not depend on `delta`;
  compare the cuts of a study at one common `t_p` (each cut records its own fit and window,
  `pinch`), and report the result with that `delta` study. `simulationCases/reconnect_prototype.py`
  is a development wrapper of the same cut and driver (its own manifest schema; no figures, no
  continuation). The cut relies on private names of the pinned pyoomph release, pinned by
  `testCases/test_reconnect_fem.py`, and the cut of a liquid neck on more of them (the parking of
  its plan), pinned by `testCases/test_liquid_cut_fem.py`; check them when the pin moves. Planner
  tests on stored neck profiles run when `FARADAY_RECONNECT_PROFILES` names their directory; a
  synthetic slender thread covers the same planner classes by default.
- The events after the cut are set by `collapse.reconnect_events` (`events.EventSettings`: `liquid`,
  `gap`, `post`), a block beside the cut's own that needs a `reconnect` block; without it a run is
  the same to the byte as at 1b65916 (`testCases/test_events_fem.py`: a cut of the synthetic necked
  cavity, the run after it with rebuilds and dumps to its step limit, and a restart from a dump,
  which meet no neck, no jet at the bubble's bottom and no other stop, the meshes left in `_gmsh`
  too, by name and hash, and a run keeping fewer dumps than it writes, whose rotation runs the hook
  of the pruning;
  `testCases/test_events_legacy.py`: stand-in problems through those, the bubble's bottom with a jet
  and its kept focus, the necks of both chains, the open chain's neck stop, a groove at a jet's
  base, the dt floor and the stops after a cut; `testCases/test_events_run.py`: the runner of a
  post-cut continuation, the keywords it makes the driver with and its `run_status.json`, completed
  and failed, against the runner of 1b65916; a replay of a production run after its cut is the
  gate still to pass). It applies after a cut only: a cut restart may add, drop or change it
  (`run_case.CUT_EVENTS_DERIVED_KEYS`), a post-cut continuation may add or change it
  (`run_case.POST_CUT_DERIVED_KEYS`, `run_case._check_events`), and a run from the start or an
  ordinary restart with it is refused before any output; `run_case.CUT_DERIVED_KEYS` and the
  prototype, which refuses the block, are unchanged. Dumps written after the cut with it carry
  `events_settings`, `events` (the events made), `legs` (the cut and each restart: step, time,
  commit, hash of the block, changed keys, the limits and floors it runs with, resolved and in units
  of `delta` and `delta^2`, `limits`, the events made before it, `events_made`; events `leg`; a leg
  restarted from a dump writes the record of each event made before it, as the dump holds it, into
  its own `events/`, `copy` naming its leg and the leg that made it, the ghost's polygon only in the
  file of that leg, `CollapseDriver._copy_event_records`) and `mesh_fresh` (no step has run on the
  mesh since it was built: a restart checks it as a fresh mesh, as the run that dumped it did,
  unless it changes a key that sizes the mesh, `events.SIZING_KEYS`, whose new sizes the rule of
  rebuilds then meets), and only a case with the block continues them. The keys of `liquid` and
  `gap` are fixed
  once an event of their class was made, except in a restart from the state before an event
  (`precut_e<k>`, `cut_stage` `precut-event`), from which they apply anew (`event_keys_changed_at`
  in its leg); the keys of `post` change at any restart (a key a recorded block lacks counts at its
  default), `post.keep_event_dumps` (`all` or `first<N>-last<M>`, `events.event_dumps_kept`),
  `post.prune_gmsh` and `post.log_box_noise` among them, and so do the block's own
  (`events.EventSettings.KEYS`: `max_events`, the most events a run makes over its legs, and the
  cascade test
  `cascade_min_interval`, `cascade_ratio`, `cascade_count`). `post.bubble_bottom_sizing` `frozen`
  meshes the bubble's bottom never coarser than `post.bubble_bottom_size` instead of the landing
  focus (`reconnect.feature_regions`), except
  within a few of the cut's tip sizes of the bottom's tip, where the cut's tip attractor at every
  axis end (`tip_size delta`) still applies, and on the cut's own mesh, which has the size the cut
  keeps there (`focus.kept`) until the first rebuild after the cut. An unknown key of the `collapse`
  block is refused with a message. With the block, `reconnect.post_cut_stops` hands over to
  `events.post_cut_events` (without it the cut's stops are untouched): invalid states first (chains
  not two, crossed nodes or touching chains raise, spline contact, and on a fresh mesh a tracked
  liquid neck below `liquid.resolution h_loc`, `liquid-neck-unresolved`, the guard, the driver's
  curved-element check passed as `invalid`; on a mesh a step has run on the guard comes last, after
  the events, as in the cut's own stops), then the end of the window (`bubble-vanishing` with
  `window_end`), then the events as requests (`event-request`: `kind`, `class`, the `handler` of its
  class, `events.HANDLERS`, none for `L-unhandled`) in the order `piercing` (W1, the bubble's tips
  within `delta`, or `max(ring.gap delta, 3 h_min)` with `ring.enabled`), `liquid-neck` (L, landed
  at `r_n <= eps_d s`, nearest first; `L-unhandled` without a single head or with one beyond
  `max_head_volume s^3`, and with liquid events off at the
  cut's second-pinch test, its head never classified), a closing fold, below `max(kappa_g delta,
  fold_stop_elements h_elem)`, among the fold of each chain and that of each tracked head
  (`events.head_fold`, so that a closer fold elsewhere on its chain hides none): first one that
  reaches the head of a tracked neck with `r_n <= eps_max s` on either chain (`a-head`,
  `L-unhandled` for a head too large; the narrowest `r_n/s` first), then one at the scale of the
  mesh (`contact` `kink`), no event but the cut's own `self-contact-approach` or
  `annular-contact-approach`, then the groove closest to its limit (`groove`), then `gas-neck`
  (deep also against its adjacent shoulders unless below `neck_elements h_min`),
  `bridge-closing`, `chain-contact`. Heads and tracked necks: `events.head_sides`,
  `liquid_necks`, `head_of_fold`. The keys of `liquid` also set the cut of a neck that lands under
  them (`surgery.liquid_cut`, which refuses a request made under others), which the driver makes within
  the run (below) (`cap`, `cap_height`, `cap_fill`, `box_radius`, `projection`, `restart`, and the noise of the
  box in an ordinary rebuild that floors the limits of its budgets, `sigma_V`, `sigma_Pz`,
  `sigma_KE`, `surgery.NOISE_FLOORS`, none until calibrated; class keys like the rest of `liquid`;
  `post.log_box_noise` logs that noise: before and after every ordinary rebuild after the cut, the
  rule's and one for distortion, the budgets of a cut's box about each tracked neck with a head,
  `neck_<tip>` at the radius its cut would take, and each tip on the axis, `tip_<tip>` at
  `box_radius delta`, are read and their changes written to `box_noise.txt`, a row per box and
  rebuild, with the largest and rms change in the result's `events.box_noise`
  (`CollapseDriver._box_noise_before`, `_box_noise_after`), the reading changing no value,
  `testCases/test_events_fem.py`):
  the head removed and the stump capped at the neck (`events.truncate_chain`, refused unless its
  hard checks hold), the mesh rebuilt from a plan built in the project (`events.build_liquid_plan`,
  parked as pyoomph parks its own, `surgery.park_plan`) under the driver's focus with the cap's
  region, named after the cut tip, and the regions the cut tip and the cut neck had kept for that
  mesh under the event's names (`events.event_focus_regions`: `<tip>_e<k>`, `neck_<chain>_e<k>`;
  `mesh.renamed` in the record), so that a later rebuild, which sizes each feature from the region
  of its name, sizes the new tip from the cap, a rebuild for distortion included, and a new neck of
  the chain from its own spacing, not from the cut neck's size, the transfer checked against the
  geometry (`surgery.transfer_problems`:
  pyoomph's count of unlocated interface nodes within two of the count `events.predict_unlocated`
  predicts, every predicted one on the cap, none of the bulk or the axis, at most the stump's tip
  without a value), the cap's nodes given the old bulk velocity (refused unless found at
  `surgery.CAP_FILL_FOUND` of its samples along the cap), the projection, and the budgets within the
  box (`surgery.box_polyline_budgets`, `box_integrals`, the cutoff's identity in
  `project(..., box=)`), the volume and the momentum of the removal hard checks
  (`surgery.HARD_CLOSURES`; a miss is `surgery.BudgetUnclosed`, `outcome` `unclosed`, a stop to
  resume from `precut_e<k>` under calibrated floors), its kinetic energy (the transfer's against the
  head's, the projection's either way) and the area flags, the cutoff's identity recorded only; its
  record is `events/event_<k>.json`, its dumps `precut_e<k>` and `postcut_e<k>`
  (`testCases/test_liquid_cut_fem.py`, a synthetic state with a jet on each chain). The driver
  makes the liquid cut of an `L` or `A-head` event within the run (`CollapseDriver._meet_event`, at
  the stop checks of the loop's top, after a step and after a rebuild;
  `events.IMPLEMENTED_HANDLERS`) with the run's last step as the history's, and goes on from the
  fresh mesh, whose stops it checks again as those of a mesh freshly built (`reset_after_event`: no
  new cycle, the births, series, holds and last step kept; the scale's growth limit, the cut tip's
  apex history and its chain's kink region dropped; event `liquid-cut`, a `liquid-cut` profile; the
  jets' bases and the kink regions of the fresh mesh committed before `postcut_e<k>` is kept, so
  that a restart from it steps as the run does); it
  adds to the record the base of the cut jet before the cut (`trigger.r_j`, `z_b`, `r_j_over_r_n`,
  `neck_to_base_over_r_j`), the event's `generation` at its tip, `step`, `leg`, `tau_over_delta2`,
  `t_prime_over_T` (`T = t_p - t_cut`), the `cascade` verdict, the `commit`, the ghost's start and
  the jet after the cut (`after`), and writes a row of `events.txt` (`testCases/test_events_run.py`,
  stand-ins and a stand-in cut; `testCases/test_events_run_fem.py`, two events in one finite-element
  run, a refused cut, and a run resumed in a new process from the dump of an event a driver without
  the handler stopped on). The ghost of each removed head is followed after each step
  (`CollapseDriver._follow_ghosts`, `events.ghost_place`: rigidly at its mean axial speed, its rear
  retracting at `r_n^(-1/2)` until the ghost is as long as the sphere of its volume, `2 a`, so that
  its rear never passes its front): its recontacts with the stump and with the far tip of the
  bubble (`events.ghost_recontacts`, from `GHOST_SKIP_STEPS` after the cut) are flags only, in
  `ghost.recontacts` of the record and its file in the directory of the leg that flags them (the
  copy of a later leg, `CollapseDriver._rewrite_event_file`), events `ghost-recontact` and rows of
  `drops.txt` (with one per ghost made). Every removal enters the run's budget (`budget.txt`, a row
  per step after the cut: the mesh's volume, axial momentum and kinetic energy of `chains.txt`, the
  removals' since the gas cut, `CollapseDriver._removed`, and the sums; the volume and the momentum
  of each removal are hard checks of its cut, within its box). The dumps of events that
  `post.keep_event_dumps` no longer keeps are deleted as events come
  (`CollapseDriver._drop_event_dumps`, `dumps.dropped` in the record and its file), and with
  `post.prune_gmsh` the meshes of `_gmsh` that no dump of the run directory or of the driver's
  `states` refers to (pyoomph's `get_state_file_mesh_files`) and that the problem's templates do not hold, after every
  rotating dump and every event (`CollapseDriver._prune_meshes`, none where a dump cannot be read;
  `testCases/test_events_fem.py` restarts the dumps kept in new processes, and
  `testCases/test_events_run_fem.py` prunes the meshes of the event dumps a run deletes, against a
  twin that prunes none, and restarts the state it ended on). Every other event is a
  resumable stop with the dump of the state before it (`states/precut_e<k>.dump`, `cut_stage`
  `precut-event`, the event under `event`, also an event of its reason) besides the end state, and
  with the record of the event, so that every event a run meets has one (`events/event_<k>.json`,
  `CollapseDriver._write_stop_record`: `outcome` `unhandled`, `cascade` or `max-events`,
  `events.STOP_OUTCOMES`, with the request, the neck of a liquid event as its `trigger`, its tip,
  times and the dump; a cut refused or unclosed has its own, to which the dump is added; `record`
  in the stop):
  `event-unhandled` (no handler made: the ring of a piercing without `ring.enabled`, the groove
  fill without `groove.enabled`, the split of a gas neck, a liquid neck without a single head),
  `max-events` (the event
  after `max_events`),
  `event-cascade` (`events.CascadeMonitor` judges it to shrink: an interval at its tip below
  `cascade_min_interval delta^2`, or the last `cascade_count` pairs of ratios of intervals and of
  head volumes below `cascade_ratio`), `liquid-cut-refused` (the problem unchanged),
  `event-unclosed` (hard budgets that do not close: the problem holds a cut that is not accepted, so
  the run ends on the state before it, its `end_state`, and dumps nothing more); any other failure
  of a liquid cut fails the run. A continuation from such a dump is a post-cut restart from which
  the keys of every class apply anew (`events.EventSettings.refused_changes`), the events before it
  staying as made. The run's result with the block has `events` (made, by kind and tip, the last,
  the totals removed, the cascade's verdicts, the ghosts' recontacts, the dumps and meshes deleted,
  the log of the box's noise, `box_noise`, the event stopped on) and `window_end`
  (`bubble-vanishing`, also an event `window-end`), which
  `run_status.json` also has at its top level; a run failed in a liquid cut has `event` there
  (`run_case.event_failure_status`, the record's schema `events.EVENT_SCHEMA`). The tests of the
  branch on the stop states of runs after a cut run when `FARADAY_POSTCUT_RUNS` names their run
  directories (`os.pathsep`-separated); synthetic states scaled with `delta` by `2^-k` cover it by
  default. After the cut the driver reads its limits through accessors that return the collapse's
  own value before the cut and without the block (`_h_floor`, `_rise_tol`, `_dt_min`, `_fold_stop`,
  `_neck_limit`): `post.h_floor` (focus floor, feature regions, guard, kink regions, the open
  chain's neck limit), `post.rise_tol`, `post.dt_min`, `gap.kappa_g delta` (also the groove gaps of
  `chains.txt`), `post.neck_floor` and `post.neck_elements_per_scale` (the neck regions of
  `reconnect.feature_regions`; liquid events need `post.neck_floor`, without which the region of a
  neck stays at the cut's focus size and cannot follow it to its landing below the cut radius);
  `post.open_neck_stop` false skips the open chain's neck stop. At the cut and at each restart after
  it the driver with the block sets oomph-lib's smallest step (`pb.DTSF_minimum_dt`) to
  `post.minimum_dt` or 1e-3 times its step-size floor (`CollapseDriver._lower_minimum_dt`,
  `minimum_dt` in the leg, with `minimum_dt_ulps`, that step over the spacing of doubles at the
  leg's time), never above oomph-lib's default and at most half the step-size floor after the cut
  (`events.minimum_dt_after_cut`, refused before any output and where the driver is made: a step
  failing at the floor must still be halved), never before the cut, where `classify_failure` reads
  oomph-lib's message at 1e-12; at 1e-12 an accepted step below 2.5e-13 ends the solve and a failed
  one below 2e-12 is not halved (`testCases/test_events_fem.py`).
- The ring of a pierced bubble is the events block's `ring` part (`events.RingEventSettings`, off by
  default and written into the block and its hash only where it differs from its defaults, so that
  every recorded block, leg hash and dump is unchanged; without it, and without `ring.enabled`, runs
  are as before: `testCases/test_ring_identity.py` pins the stops, a stand-in piercing and the
  finite-element run after the synthetic cut with the block, and a restart from one of its dumps,
  against c4620f6, to the byte, besides the 1b65916, e330650 and 148340e pins). With `ring.enabled`
  the driver makes the handlers `ring` (W1, `ringsurgery.ring_cut`) and `ring-removal` (W2,
  `ringsurgery.ring_removal`) within the run (`CollapseDriver._meet_ring`, no cascade test): the gas
  cut on the axis between the plug and the downward jet's cap by the closure of
  `ring.ring_cut_geometry` (an arc, a vertical segment of radius `waist rho` and an arc, tangent to
  the cap at `rho = ring.rho s_t` and to the plug, refused where the old mesh does not hold `rho`,
  `RHO_MIN_NODES`), the bubble's rest kept as a ring, a closed loop of the free surface's own
  boundary `interface` (a boundary of its own would be refused by pyoomph at `load_state` of every
  earlier dump), at the passive constant pressure; the problem's chart
  `problem.RingTolerantArclength` (always present, the same chart without a loop) charts the loop
  beyond the open chain and turns the zeta transfer off with a loop on either mesh, the template
  meshes the ring as a hole of two splines (`mesh.FaradayCylinderMesh._define_ring_geometry`,
  `ring_stage`, `ring_sizing`, `pending_ring_geometry`), the rebuild is the project's own (no
  pyoomph plan is parked while a loop exists) and its transfer is checked against the nodes the
  geometry predicts unlocated (closure, column, axis gap); then the fill (`ring.fill`), the
  projection and the closures R1-R6 (`ringsurgery._ring_closures`: R1 the volume gained, R2 the
  momentum the transfer carried, R3 the projection's momentum balance in the box, the cutoff
  identity against the larger of its sides, and `R3_box`, the box's change of momentum by the
  projection against `P_ref`, the liquid cut's F3, all hard; R4-R6 flags; the transfer refused for a
  node without a value off the corner set or a closure without nodes), floored at
  `ring.sigma_*`, which every ring case must set from the box's noise; the ring's sizing frozen at
  the cut (`ring_<name>` copies of the bubble's finer regions, `ring_cut` the column's, emitted at
  their sizes by `CollapseDriver._frozen_ring_regions`). The ring stage (`FaradayProblem.ring`,
  dumped as `ring` with every state of it, restored by `ringsurgery.restore_ring` after
  `surgery.restore_cut`) reads the open chain as before and the ring apart
  (`FaradayProblem.interface_loops`; `interface_segments` sets loops aside); its stops are
  `ring.ring_stage_events` (the open chain's necks, folds, heads and gas neck as
  `events.post_cut_events` reads them; the ring's removal below `bubble_stop V_b`, W2; its approach
  to the axis, RA, its folds, RF (filled with `groove.enabled`, below), its approach to the
  open chain, X, each a resumable stop). A
  landed neck of the open chain in the ring stage is cut by `surgery.liquid_cut` as before the
  ring (record `stage` `ring`), its rebuild the ring stage's own (the template's
  `pending_ring_geometry`: the cut chain and the ring as it is, or no loop after the removal; no
  plan), the interface transferred by projection and the axis located in the old bulk
  (`FaradayProblem._ring_transfer`, reach `2 r_n`, which also turns the zeta transfer off with no
  loop), the cap clear of the ring (`ring.cap_ring_clearance`, check (v) of `truncate_chain`), the
  ring kept across the rebuild (`ring.loop_kept`: one loop as before, its nodes within
  `LOOP_NODE_SLACK`, its polyline's volume within `LOOP_NOISE_FLOORS ring.sigma_V` plus the drift
  `|S_before| + |S_after|` of `ring.loop_p2_excess`; `loop` in the record, `CutFailed`
  otherwise), the unlocated nodes predicted from the old chain and the ring, and the box's
  polylines holding the ring's gas; a liquid cut without the ring stage takes the unchanged path
  pinned by `test_liquid_cut_fem.py` and the pins. After the removal `ring.after_removal` `stop`
  ends the window (`bubble-vanishing`, `ring_removed`), `continue` goes on with the open chain
  alone; it is a key of the leg (`EventSettings.LEG_KEYS_OF_CLASSES`), every
  other ring key a class key, and a state of the ring stage is refused under a block without
  `ring.enabled` (`run_case._check_events`, `CollapseDriver._check_events_meta`). Files: `ring.txt`
  (`driver.RING_COLUMNS`), `_ring.txt` profiles, an `events.txt` row per surgery (its fill a
  negative removal), `budget.txt` totals net of the fills, `events.ring` in the run's result;
  `testCases/test_ring_settings.py`, `test_ring_geometry.py`, `test_ring_stops.py`,
  `test_ring_driver.py`, `test_ring_closures.py` (each hard closure beyond its floored limit, the
  transfer's no-value and closure checks) and `test_ring_fem.py` (a ring meshed, stepped, rebuilt,
  restarted in new processes and removed; the cut and the removal on a synthetic pierced state,
  `testCases/ring_fem_state.py`, and each hard closure of both violated in turn, a stop
  `event-unclosed` with `precut_e<k>`; F4, the liquid cut of the cavity's tip in the ring stage on
  that state with the tip necked, with the ring and after its removal) cover it.
- The events block's `groove` part (`events.GrooveEventSettings`, `groove.py`,
  `groovesurgery.py`; off by default, written into the block and its hash only where it differs
  from its defaults, its keys keys of the class `gap`, `EventSettings.CLASS_OF_PART`) makes the fill
  of a closing gas groove (`groovesurgery.groove_fill`, `CollapseDriver._meet_groove`) for
  `A-groove` (the open chain or the bubble, and the open chain in the ring stage) and `RF` (the
  ring's loop; its request names `groove-fill` only with the part), but never on a far head: with
  `pinch.far_head` a groove on the head of a tracked neck in the far region is met by the far-head
  cut of that head first (`CollapseDriver._meet_event`; decision 22), the fill only off the far
  heads: the pocket of the fold pair must be gas (`groove.side_of`; a liquid sheet is refused), the
  mouth where the walls lie `mouth` fold limits apart or have turned out of the groove
  (`groove.MOUTH_TURN`), a cubic closure C1 at the splices, the filled region at most `max_fill V_b`
  and never on the core or the base of the upward jet; the rebuild through the template's
  `pending_ring_geometry` (built by `_define_ring_geometry` also before the ring, the only path
  that sets it there) with the projection transfer; the bulk's unlocated nodes those inside the
  filled region within a twentieth of the old surface's shortest chord there; the projection's
  box a disc at the site (`surgery.site_grid`, `box_polyline_budgets(..., r_centre=)`,
  `project(box=<grid>)`), its integrals weighted by the cutoff `phi` of the identity G3
  (`box_integrals(..., cutoff=True)`, `project(..., cutoff=True)`: a cell at the box's edge whose
  sample a rebuild moves across a fast free surface weighs nothing; the unweighted sums kept as
  `unweighted`, the unweighted G2 reported as `G2_unweighted`, never a flag,
  `groovesurgery.unweighted_g2`; `test_fill_cutoff.py`, `test_fill_cutoff_identity.py`, runs
  without a fill b46713a's); G1-G3 and `G3_box` hard (`ringsurgery._ring_closures`, prefix `G`),
  but `G3_box` beyond its limit only a flag where the cutoff identity G3 holds and resolves it (its
  larger side at least `MERGE_RESOLVED` 0.5 of the unweighted disc momentum change,
  recorded as `G3_unweighted`, and at the momentum floor or above;
  `groovesurgery.merge_impulse`, `G3_merge` in the closures): the projection of a fill is the
  inelastic impulse of the merge it makes, which moves momentum out of any disc about the site
  physically, so the disc's change against `P_ref = V_fill U` tests no balance
  (`test_groove_closures.py`); `groove-fill-refused` a resumable stop, as is a fill whose hard
  closures miss (`event-unclosed`); the record kind `groove-fill` (`EVENT_CLASSES["gap"]`),
  a row of `events.txt`, `budget.txt` net of it (`CollapseDriver._filled`),
  `CollapseDriver.reset_after_groove`. With `groove.sheets` `log` (default `event`) the stops'
  chain and ring folds are selected per side (`groove.side_folds`, `side` in the request): a
  liquid pocket, a sheet, is no event but a row of `sheets.txt` (`driver.SHEET_COLUMNS`,
  `CollapseDriver._log_sheets`, `groove.sheet_reading`) at most once per rebuild, and its faces
  below `sheet_contact` element lengths of the finer side the kind `sheet-contact`, without a
  handler (`events.SHEET_CONTACT`, never routed to the fill); `test_groove_sheets.py`. Runs
  without the part, or with it and no groove met, are
  those of 89f7f49 (`testCases/test_groove_identity.py`), besides every earlier pin;
  `test_groove_settings.py`, `test_groove_geometry.py`, `test_groove_driver.py` (stand-ins, the
  far-head cut first among them), `test_groove_fem.py` (a finger of gas of a ring and a slot of the
  open chain filled, the refusals unchanged) and `test_far_head_first_fem.py` (a groove on the
  heads' state's downward head cut with the head, never filled) cover it, and
  `test_far_head_first_identity.py` pins runs that meet no groove or unhandled event on a far head
  and no curled head to cb12382. A wedge of gas of the ring whose apex the guard reads at an
  element at the run's floor (a stop the ring refine does not meet; `ring.floor_wedge`, the fold
  test read at the scale of the floor, `ring.FLOOR_FOLD_RATIO`) is met by the same fill
  (`CollapseDriver._floor_fill_request`: `RF`, `contact` `floor`, the request's `floor` with the
  guard's reading, the wedge and its fold limits; `groove_fill`'s `base_check`, the hard closure
  `G_base`; the transfer's former gas beyond `groovesurgery.LOCATOR_BAND` of the old elements'
  curve, `floor_bulk_inside`), event `floor-fill`, `floor_fills` in the run's events; the tip of
  a liquid finger stays the stop; where no wedge is, a corner of gas at the floor open up to about a
  right angle (`ring.FLOOR_CORNER_RATIO`, the root of the radial sheet the downward head drives
  out along the plug) is filled in the same way (`corner` in its wedge); a failed step in the ring
  stage is retried once after the ring's sharpest element above the floor is rebuilt as the ring
  refine rebuilds (`CollapseDriver._failed_step_retry`, event `step-retry`;
  `test_landing_identity.py` pins runs meeting neither against 6eff16c); a retry's hard
  `ring-refine-base` stop ends the run; other guard stops and event requests return to the
  handlers. Stops before another accepted step keep `failed_step` context. A failed retry
  restores its input mesh, focus and refinement bookkeeping (`step-retry-failed`, `retry_error`); an in-memory restore
  across a rebuild (`FaradayProblem._restore_state`: the rollback of a failed step or of an axis
  repair) puts the template's project attributes (`TEMPLATE_KEYS`: focus, cut and ring sizing)
  back on the sibling template pyoomph reloads the mesh from, which in a restarted run otherwise
  copies the original template's (`test_restore_template_identity.py`); after a cut each Newton
  solve's tolerance is `max(newton_tolerance, newton_floor/dt)` with dt the sub-step it makes
  (`FaradayProblem.newton_floor_rule`, set by `CollapseDriver._post_cut_solve`), not the driver's
  step alone (`test_newton_floor_identity.py`); a closing fold of the ring's gas across a kink at
  the floor's scale is read as the guard's reading at its apex and filled at the floor
  (`CollapseDriver._ring_kink_reading`, trigger `fold-kink`), and a fresh mesh's curved element
  beside the bubble's chain before the piercing has the surface's curved edges about it put
  straight within `AXIS_REPAIR_BOUND delta^3` (`CollapseDriver._tip_straighten`,
  `FaradayProblem.straighten_interface_edges`, event `tip-straighten`;
  `test_kink_tip_identity.py`); straightening that reaches the open chain is refused,
  and a Jacobian bound still below its limit restores the input; a rebuilt mesh's
  `curved-jacobian` at the ring is read as the
  ring element's reading there (`CollapseDriver._ring_jacobian_reading`) and met by the refine or
  the fill at the floor; `test_floor_fill.py`, `test_floor_fill_fem.py`,
  `test_floor_fill_identity.py` (runs never at the floor are 7ba32f9's).
- Local ring contacts (`ringcontact.py`, recentred far kinks and enabled groove fill): a
  single rebuild-spline crossing within two adjacent quadratic elements, on a simple node
  polygon, is filled only after its local pocket is independently read as gas (`spline-fill`,
  `spline_fills`). A candidate step's local polygon crossing is checked before acceptance;
  with the inverted-element net's step snapshot and an unchanged mesh, cyclic node
  identities map its pocket back to the simple input. Other non-simple candidates also
  roll back and keep `stepped-contact-unhandled`, without a fill.
  Without that snapshot this candidate-step route is off.
  The native input is restored before a gas fill (`ring-contact-step-rollback`, with discarded
  distortion flags and the rejected diagnostics interval; `step-contact-fill`,
  `step_contact_fills`). Unreadable or liquid pockets keep `stepped-contact-unhandled`.
  Both fills use the existing hard G closures, `G_base`, recorded filled volume and the
  curve/locator-band transfer prediction (`events/event_<k>.json`); refusals and failed
  closures keep the existing resumable stops. A liquid-side kink instead uses the ring
  refinement without rupture: positive calibrated ring volume and momentum noise floors
  are required (`liquid_refine_unavailable` otherwise). Its hard zero-removal checks are
  `C_V`, `C_P`, `C_loop` and `C_base`, with the existing base tolerance; an unclosed audit
  restores the input in place, retaining event and profile counters (`ring-contact-unclosed`).
  Records are `events/ring_contact_<step>_<remesh-count>_<attempt>.json`, event `ring-contact-refine`.
  Unit and stand-in routes, finite-element fills/refinement/rollback, and unchanged-run pins
  against 775df94 are in `test_ring_contact.py`, `test_ring_contact_fem.py` and
  `test_ring_contact_identity.py`. These routes preserve one simple ring; none ruptures a
  liquid film or changes the ring-removal threshold.
- The ring's thinning liquid lip (decision 22; `sheetsurgery.py`; recentred far kinks, the groove
  fill with `groove.sheets` `log`, the ring's calibrated noise floors, `CollapseDriver._sheet_handled`):
  the ring stage's stops read the ring's liquid fold through the driver's callback
  (`ring.ring_stage_events`' `sheet`, `CollapseDriver._sheet_context`), which keeps the lip's thickness
  history (`lip_samples` in the dumps; once per time, restarting at a jump of `driver.LIP_JUMP`
  thicknesses) and keeps the refined far kink that holds the lip alive while it lasts. A lip with
  fewer than `driver.SHEET_REFINE_ELEMENTS` elements across it, above the refine's floor, is a
  `sheet-contact` not due: it is refined to `max(h_floor, gap/driver.SHEET_ELEMENTS)` by the liquid
  kink's route with its hard zero-removal budgets (`CollapseDriver._ring_sheet_reading`, trigger
  `sheet-contact`, `_ring_liquid_refine`). A lip thinning (`sheetsurgery.lip_thinning`: a falling
  least-squares slope over at least `LIP_MIN_SAMPLES` of its last `LIP_SAMPLES`, the last below the
  first) at `h_s = jetcore.sheet_rupture_floors h_floor` (null: `driver.SHEET_RUPTURE_FLOORS`, 2; a
  key of the leg written into the block and its hash only where set, `EventSettings.LATE_KEYS`) or
  below is a `sheet-contact` due, met by the rupture (`CollapseDriver._meet_sheet_rupture`,
  `sheetsurgery.sheet_rupture`): the rim beyond the thinnest section removed and the remaining lip
  closed by a half circle of the section's diameter (`sheet_rupture_geometry`: the new loop simple
  and counter-clockwise, the removed region liquid, clear of the open chain by `OPEN_CLEARANCE`
  sections, at most `MAX_REMOVED_V_B V_b`), the ring stage's own rebuild with the projection
  transfer and the cap's region, the transfer checked (unlocated interface nodes only on the cap,
  none of the bulk or the axis), the projection, and the closures S1-S6 (`_ring_closures`, prefix
  `S`, the removal a negative fill: S1 the volume the disc loses against the rim's, S2 the momentum
  the transfer removes against the rim's, S3 and `S3_box` hard; S4-S6 flags) and `S_base`; event
  `sheet-rupture` (class `RS`, `EVENT_CLASSES["ring"]`), its record with the removed V, Pz, KE, the
  rim's centroid and the lip's history, a row of `events.txt` (`S` closures), `budget.txt` net of it
  (`filled` the negative removal); a refusal is the resumable stop `sheet-rupture-refused`, missed
  hard budgets `event-unclosed`. A lip at the floor thinner than one element and not thinning stays
  the `sheet-contact` stop. `test_sheet_rupture.py`; runs that meet no lip are dd01824's
  (`test_sheet_identity.py`).
- The events block's `jetcore` part (`events.JetCoreSettings`, `jetcore.py`; off by default, written
  into the block and its hash only where it differs from its defaults, every key a key of the leg,
  its sizing keys among `events.SIZING_KEYS`) resolves the upward jet as before and treats the rest
  of the free surface under simplifying assumptions, every feature kept in the domain: the core
  (`jetcore.core_mask`: the open chain's nodes within `max(core_radius L, apex_extent R_apex)` of
  the jet's base, `L = max(r_j, delta)`, not above the jet's tracked liquid neck, and the walls
  beyond the base within `wall_radius L`; the whole open chain without a base) is stepped at
  `reconnect.courant` and the far nodes at `far_courant` (`jetcore.courant_split`; by default the
  same number, the step unchanged to the bit); the far regions are floored at `far_floor`, the
  tracked necks' and the pre-pierce regions exempt (`jetcore.region_class`,
  `CollapseDriver._far_sizing`), the far kinks recentred with `far_kinks` `recentre`
  (`jetcore.far_kink_regions`: at the floor, below it only past `far_yield_turn`, halving per
  rebuild, event `far-yield`), refined at `far_refine_ratio`, the ring's kinks at the column's size
  with `ring_kink_floor` (in the ring stage the guard's stop on the ring, a turn beyond
  `edge_turn_stop_fresh` on a fresh mesh or `edge_turn_stop` on a stepped one, up to
  `driver.RING_REFINE_TURN_MAX`, or a midside node beyond `midside_stop`, is met by
  `CollapseDriver._ring_refine`: up to `driver.RING_REFINE_TRIES` rebuilds with the element in a far
  kink of its own, `kink_ring_r<n>_far` at `min(1/2, edge_turn_refine/turn)` of its chord, marked
  `refined` so that `jetcore.far_kink_regions` has it follow its feature at that size until the
  feature asks no more, every other region as it was, and while it remains every far kink of the
  ring follows its own feature, a new one `kink_ring_m<n>_far` (`jetcore.MULTI_FAR_KINKS`), event
  `ring-refine`, the upward jet's base held within `HEAD_FOLD_BASE_TOL` by `driver.base_guard` or
  the stop `ring-refine-base`; a ring element at the floor that asks is carried there,
  `ring.ring_floored_ok`, `reconnect.interface_guard`'s `floored_ok`; `test_ring_refine.py`,
  `test_ring_refine_fem.py`, and `test_ring_refine_identity.py`, runs that never meet it e923ead's),
  the edge-turn profiles from the core with `edge_turn_snapshots` `core`; `prep_gap` refines the
  bubble's bottom and the downward jet's tip before the piercing (`jetcore.prep_regions`) so that
  the ring cut admits `prep_rho s_t`; `profile_dt` adds profiles (reason `window`); `box_gate` stops
  the run (`box-gate`, on `states/gate_<step>.dump`, the state before the rebuild) where an ordinary
  rebuild changes the liquid volume or axial momentum of the box about the core's base by more than
  `box_gate_sigmas` times `liquid.sigma_V`, `sigma_Pz`;
  `core_elements_per_scale` (null: the run's elements per scale after the cut; never below it,
  `jetcore.core_ratio`, refused by the driver and the runner) meshes the core alone finer, every
  rule's size on its nodes `q = eps/core_elements_per_scale` times what it asks for: each region
  whose ball holds core nodes has a twin `core_<name>` of its ball at `q` times its target, floored
  at `h_floor` (`core_floored`), whose attractors are those core nodes (`jetcore.core_twins`,
  `core_spec`, `core_select`, `mesh.FocusRegion.core`, written into the focus records only where
  set), the twin's spacing and its region's read apart (`CollapseDriver._twin_selection`,
  `reconnect.region_spacing`'s `select`), the guard's turn on the core read at `q` times
  `edge_turn_refine` (`jetcore.core_turns`; its own kinks have no twin), a liquid cut renaming a
  twin with its region (`events.event_focus_regions`); the key is in the block and its hash only
  where it is set (`EventSettings.LATE_KEYS`), so every block recorded before it hashes as it did;
  `jetcore.txt` a row per step (`jetcore.COLUMNS`), `limits.jetcore` in each leg. The `pinch`
  part (`events.PinchSettings`, keys of the class `liquid`, `EventSettings.CLASS_OF_PART`) lands a
  tracked neck only when it closes (`events.closing_neck`, from the driver's per-neck history,
  `neck_history` in the dumps, `events.match_neck`), the request carrying the trigger and the liquid
  cut's admission reading it again, and with `far_head` meets an event without a handler the
  driver makes on a far head (a gas neck, a closing groove, before any fill, a sheet's contact) by
  the liquid cut of its neck (`events.far_head_request`, kind `far-head`, class `F-head`; neither
  the site nor any node of the head but its neck a core node of the open chain). A head whose rim
  curls back past its neck's height, every node beyond it outside the neck's radius and its
  half-section simple, is a head (`events.head_sides`, `curled` and `curl_depth` in its neck's
  record and request; `testCases/test_curled_head.py`), the head the chain from its tip to that
  root neck. A closing groove from the upward jet's head onto the core (one node of the pair on the
  head from the tip, the other a core node, the wall or the stem beside the base) is the cut of
  that head at its root neck too (`head_fold` in the request), the head's nodes but its neck off
  the core, and the cut is checked against the jet's base, read before and after on the fresh mesh
  (`surgery.liquid_cut`'s `base_check`, `CollapseDriver._head_fold_check`, `base_guard` in the
  record): `r_j`, `V_j`, `Q_j` within `driver.HEAD_FOLD_BASE_TOL` (1e-3) relative and `z_b` within
  that times `r_j`, or the hard closure `F_base` (`event-unclosed`); `test_head_fold.py`,
  `test_head_fold_fem.py`, and `test_head_fold_identity.py` (runs meeting none of it are
  3e00e92's). A tracked neck that a freshly built mesh does not resolve
  (`liquid-neck-unresolved`) whose head is a far head (no node of it but the neck a core node; on
  the bubble every node) is the same cut of that head at that neck, closing or not
  (`events.unresolved_neck_request`, `CollapseDriver._unresolved_neck_request` in the loop's
  dispatch beside the fill at the floor; `unresolved` and `far_event` in the request, an event
  `far-head` with `far_kind` `liquid-neck-unresolved`), with the same check of the base; a head of
  the core, or a neck without a single head, stays the stop; `test_unresolved_neck.py`,
  `test_unresolved_neck_fem.py`, and `test_unresolved_neck_identity.py` (runs meeting none of it
  are 7efe4df's).
  A key whose behaviour the driver does not make is refused (`events.PENDING_KEYS`).
  `testCases/test_jetcore.py` (stand-ins), `test_jetcore_fem.py` (the far floor and kinks on the
  synthetic headed state, the pre-pierce refinement on the pierced state, the far-head cut) and
  `test_jetcore_identity.py` (without the parts every run of 4a6c999 to the byte; with `jetcore`
  at its defaults the same solution, `jetcore.txt` added) cover it, and for
  `core_elements_per_scale` `test_jetcore_res.py` (stand-ins, the runner's admission),
  `test_jetcore_res_fem.py` (the core's elements halved on the heads' state and none off it; a
  restart under the key rebuilding at its sizes; named to run after `test_events_run_fem.py`,
  whose two runs that restart from dumps differ at round-off after a module building
  finite-element problems in the same process, as after `test_jetcore_fem.py`) and
  `test_jetcore_res_identity.py` (runs without
  the key, the production block and its hash, 3e00e92's to the byte). After a cut the stops of a
  state are evaluated once (`CollapseDriver._stop_key`), and the runner's `--step-timing` writes
  `step_timing.txt` (`driver.STEP_TIMING_COLUMNS`); neither changes a value.
- Element distortion (`RemeshWhen` at pyoomph's defaults: size ratio below 0.6 or above 1.75,
  quality ratio below 0.2, against the element at its build; never checked on the first solve on a
  new mesh) asks for a rebuild. Before the driver has set a focus, pyoomph makes it inside the
  solve, as it always did (counted as `pyoomph_remeshes`, no event). Once a focus is set (the
  first refinement, a cut, a restored dump) the driver turns `do_call_remeshing_when_necessary`
  off, so pyoomph only flags it (`FaradayProblem.take_distortion_flags`), and rebuilds after the
  step, before measuring it, at focus size `min(focus size, h_near)` (never coarser near the
  collapse; not floored at `h_floor`) within the current `focus_radius` (after a cut: each
  feature's region at its current place and the size it has, and only once the stops of the step's
  state have been checked), the grading kept: one
  rebuild per driver step however many sub-steps were flagged, a `remesh` event with `reason`
  `distortion` (trigger, ratio, extremes, focus and `h_near` before and after) and
  `distortion_remeshes` in `run_status.json`. These rebuilds neither start nor end a hold nor count
  for `min_steps_between_remesh`. This holds in every mode that steps through the driver (landing,
  cut and post-cut restarts, the prototype). A run without such a rebuild in the refined phase is
  the same to the byte as at 148340e (`testCases/test_distortion_remesh.py`); in one with it, the
  `diagnostics.txt` row of that step precedes the rebuild, as for the driver's other rebuilds.
- Each integral of `FaradayProblem.observables()` is evaluated once per state (`_integrals`: the
  time, the rebuild and a hash of the dofs and pinned values), the driver locates the probe discs
  of every jet in one call (`axial_velocity_sets`), and after a cut pyoomph's snapshot before each
  step is written uncompressed: none of these changes a value written.
- Runs of one case in fresh processes differ by about 1e-14 relative unless MKL runs on one thread
  (`MKL_NUM_THREADS=1`; `OMP_NUM_THREADS` alone does not do it); compare runs to round-off, or set
  it when a comparison must be exact.
- The wave height `h` of a collapse is `h_crest` from `postProcess/collapse_table.py
  --per-episode`: the largest elevation of the axis above the mean level over the crest episode
  before the trough that collapses. It is a wave height only in rows with `crest_jet` 0: when a
  jet stood on the axis during that crest, as after a jet launched from the trough before it,
  `h_crest` is the height of the jet tip.
- Quote an exponent only together with the resolution evidence that supports it.
- `postProcess/collapse_analysis.py` reports the evidence for the cone, line-sink and
  inertio-capillary laws and never picks one. Its primary test is time-free (`Q_j`, `We_j`, `q_j`
  against `r_j`); `β` comes from the profile at inception over the fixed window it records, not
  from the per-step `beta_fit`. `t0_fit` is the origin of a fitted law; keep "singular time" for
  the apex-based estimate.
- `postProcess/event_table.py` tabulates the events of runs after a cut with the block (a row per
  event, made or not, with its outcome and the handler of its class, in units of `delta` and
  `T = t_p - t_cut`, from `events/event_<k>.json`; a row per run and tip of `--tip`, the upward jet
  by default, with the stop and the event it stopped on, `budget.txt`'s drifts and the removals'
  shares of the whole liquid's axial momentum and kinetic energy (`*_share_liquid`, which the
  standing wave dominates), the removals at that tip, the momentum signed along its jet, against
  what the base carried into the jet (`Pz_jet`, `KE_jet`: the integrals of `Q_j V_j` and
  `Q_j V_j^2/2` over time, a plug flow; `*_removed_jet_share`) and, per event, the head's share of
  its cut's box (`Pz_rem_box_share`, `KE_rem_box_share`), the events made with a closure flagged,
  and the test of the base of that tip's jet widening after its own events, the slope of `ln r_j`
  against `ln t'` and of `Q_j` against `t'` over the rows of `chains.txt` after the step of the
  `--after-events`-th event made at that tip, `base_after_k`, not at any tip, and never the row of
  that step, the state before its cut; `stationary` where the slope and the relative change of
  `Q_j` over the window lie within their floors, `--stationary-alpha` and `--stationary-q`,
  recorded in the row, so that the round-off of a fit gives a base standing still no sign). A run
  is one directory or the directories of its legs in their order joined by `os.pathsep`, read as
  one continuous run: the records by `k`, a later leg's replacing an earlier's and an earlier leg's
  beyond the `events_made` a later leg starts with dropped; the rows of `chains.txt` and
  `budget.txt` of the legs before a leg beyond its first step dropped, and its own first row where
  they have one at that step. It reports and decides nothing.
- Run outputs, run manifests and environment receipts live outside this repository.
- Figures follow the `publication-plots` conventions.
- README, `docs/` and figure captions are public-candidate: no private paths, hosts or
  provisional findings.
