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
  `neck_elements h_min` (`reconnect.chain_necks`; `neck_side` gas or liquid); no Newton failure ends the run cleanly,
  and the profiles of the state the failing step started from are written first
  (`profile_<step>_lastgood{,_bubble}.txt`), as is a profile at every step while an element turns
  beyond `edge_turn_refine`; rotating dumps follow at least each time `tau` doubles and the end
  state is dumped; and every output carries `tau = t - t_cut` and `tau_p = t - t_p` (`collapse.txt`,
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
  which meet no neck, no jet at the bubble's bottom and no other stop;
  `testCases/test_events_legacy.py`: stand-in problems through those, the bubble's bottom with a jet
  and its kept focus, the necks of both chains, the open chain's neck stop, a groove at a jet's
  base, the dt floor and the stops after a cut; a replay of a production run after its cut is the
  gate still to pass). It applies after a cut only: a cut restart may add, drop or change it
  (`run_case.CUT_EVENTS_DERIVED_KEYS`), a post-cut continuation may add or change it
  (`run_case.POST_CUT_DERIVED_KEYS`, `run_case._check_events`), and a run from the start or an
  ordinary restart with it is refused before any output; `run_case.CUT_DERIVED_KEYS` and the
  prototype, which refuses the block, are unchanged. Dumps written after the cut with it carry
  `events_settings`, `events` (the events made), `legs` (the cut and each restart: step, time,
  commit, hash of the block, changed keys, the limits and floors it runs with, resolved and in units
  of `delta` and `delta^2`, `limits`; events `leg`) and `mesh_fresh` (no step has run on the mesh
  since it was built: a restart checks it as a fresh mesh, as the run that dumped it did, unless it
  changes a key that sizes the mesh, `events.SIZING_KEYS`, whose new sizes the rule of rebuilds then
  meets), and only a case with the block continues them. The keys of `liquid` and `gap` are fixed
  once an event of their class was made, except in a restart from the state before an event
  (`precut_e<k>`, `cut_stage` `precut-event`), from which they apply anew (`event_keys_changed_at`
  in its leg); the keys of `post` change at any restart (a key a recorded block lacks counts at its
  default), `post.keep_event_dumps` (`all` or `first<N>-last<M>`, `events.event_dumps_kept`) and
  `post.prune_gmsh` among them, and so do the block's own (`events.EventSettings.KEYS`:
  `max_events`, the most events a run makes over its legs, and the cascade test
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
  within `delta`), `liquid-neck` (L, landed at `r_n <= eps_d s`, nearest first; `L-unhandled`
  without a single head or with one beyond `max_head_volume s^3`, and with liquid events off at the
  cut's second-pinch test, its head never classified), a closing fold, below `max(kappa_g delta,
  fold_stop_elements h_elem)`, among the fold of each chain and that of each tracked head
  (`events.head_fold`, so that a closer fold elsewhere on its chain hides none): first one that
  reaches the head of a tracked neck with `r_n <= eps_max s` on either chain (`a-head`,
  `L-unhandled` for a head too large; the narrowest `r_n/s` first), then one at the scale of the
  mesh (`contact` `kink`), no event but the cut's own `self-contact-approach` or
  `annular-contact-approach`, then the groove closest to its limit (`groove`), then `gas-neck`,
  `bridge-closing`, `chain-contact`. Heads and tracked necks: `events.head_sides`, `liquid_necks`,
  `head_of_fold`. The keys of `liquid` also set the cut of a neck that lands under them
  (`surgery.liquid_cut`, which refuses a request made under others), which the driver makes within
  the run (below) (`cap`, `cap_height`, `cap_fill`, `box_radius`, `projection`, `restart`, and the noise of the
  box in an ordinary rebuild that floors the limits of its budgets, `sigma_V`, `sigma_Pz`,
  `sigma_KE`, `surgery.NOISE_FLOORS`, none until calibrated; class keys like the rest of `liquid`):
  the head removed and the stump capped at the neck (`events.truncate_chain`, refused unless its
  hard checks hold), the mesh rebuilt from a plan built in the project (`events.build_liquid_plan`,
  parked as pyoomph parks its own,
  `surgery.park_plan`), the transfer checked against the geometry (`surgery.transfer_problems`:
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
  apex history and its chain's kink region dropped; event `liquid-cut`, a `liquid-cut` profile); it
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
  `ghost.recontacts` of the record and its file, events `ghost-recontact` and rows of
  `drops.txt` (with one per ghost made). Every removal enters the run's budget (`budget.txt`, a row
  per step after the cut: the mesh's volume, axial momentum and kinetic energy of `chains.txt`, the
  removals' since the gas cut, `CollapseDriver._removed`, and the sums; the volume and the momentum
  of each removal are hard checks of its cut, within its box). The dumps of events that
  `post.keep_event_dumps` no longer keeps are deleted as events come
  (`CollapseDriver._drop_event_dumps`, `dumps.dropped` in the record and its file), and with
  `post.prune_gmsh` the meshes of `_gmsh` that no dump of the run directory or of the driver's
  `states` refers to (pyoomph's `get_state_file_mesh_files`) and that the problem's templates do not hold, after every
  rotating dump and every event (`CollapseDriver._prune_meshes`, none where a dump cannot be read;
  `testCases/test_events_fem.py` restarts the dumps kept in new processes). Every other event is a
  resumable stop with the dump of the state before it (`states/precut_e<k>.dump`, `cut_stage`
  `precut-event`, the event under `event`, also an event of its reason) besides the end state:
  `event-unhandled` (no handler made: the ring of a piercing, the groove fill, the split of a gas
  neck, a liquid neck without a single head), `max-events` (the event after `max_events`),
  `event-cascade` (`events.CascadeMonitor` judges it to shrink: an interval at its tip below
  `cascade_min_interval delta^2`, or the last `cascade_count` pairs of ratios of intervals and of
  head volumes below `cascade_ratio`), `liquid-cut-refused` (the problem unchanged),
  `event-unclosed` (hard budgets that do not close: the problem holds a cut that is not accepted, so
  the run ends on the state before it, its `end_state`, and dumps nothing more); any other failure
  of a liquid cut fails the run. A continuation from such a dump is a post-cut restart from which
  the keys of every class apply anew (`events.EventSettings.refused_changes`), the events before it
  staying as made. The run's result with the block has `events` (made, by kind and tip, the last,
  the totals removed, the cascade's verdicts, the ghosts' recontacts, the dumps and meshes deleted,
  the event stopped on) and `window_end` (`bubble-vanishing`, also an event `window-end`), which
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
  event, in units of `delta` and `T = t_p - t_cut`, from `events/event_<k>.json`; a row per run with
  `budget.txt`'s drifts and the test of the base of one jet's widening after the events, the slope
  of `ln r_j` against `ln t'` and of `Q_j` against `t'` over `chains.txt` from the
  `--after-events`-th event on). It reports and decides nothing.
- Run outputs, run manifests and environment receipts live outside this repository.
- Figures follow the `publication-plots` conventions.
- README, `docs/` and figure captions are public-candidate: no private paths, hosts or
  provisional findings.
