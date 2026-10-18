# faraday-singularity

Direct numerical simulation, with sharp-interface finite elements in
[pyoomph](https://github.com/pyoomph/pyoomph), of the singular collapse of a Faraday-wave
interface and the Worthington jet it launches.

**Status:** axisymmetric solver at zero Bond number with its linear verification cases, and a
collapse driver that records the collapse and jet observables during the run and stops before any
pinch-off. A reconnection continues a run through a neck pinch-off by an explicit cut (below); it is
tested on a synthetic necked cavity only, not yet on a production collapse. No collapse case files or results are in the repository yet.

## Motivation

A gas–liquid interface that collapses onto a cone launches a thin, fast Worthington jet. On a
vibrated free surface, Faraday waves drive such collapses and jet eruptions [1]. Cattaneo,
Presse and Supponen [2] showed that wall-attached microbubbles under gentle ultrasound jet
repeatedly through the Faraday instability of the bubble surface. A conical collapse sends a
jet towards the substrate; a parabolic collapse sends a pair of opposed jets. They measured the
jet speed against Faraday-wave height and bubble size, and rationalised the collapse with
scaling laws in which both inertia and capillarity enter.

Gordillo, Rodríguez-Rodríguez and Sanjay [3] argue that the birth of a Worthington jet from a
collapsing cone is purely inertial. Where the velocity potential vanishes on a cone of
semi-angle $\beta$ in an intermediate region $r_j \ll r \ll 1$ at jet inception, the local
collapse follows self-similar Euler solutions set by $\beta$. The jet-base radius then scales as
$r_j \propto \tau^{\alpha(\beta)}$, with $\alpha = 1/(2 - \nu(\beta))$ and $\tau$ the time
from inception, and the jet-base Weber number as $We_j = r_j v_j^2 \propto r_j^{2\nu(\beta) - 1}$.
$We_j$ therefore diverges as $r_j \to 0$ only for cones sharper than Taylor's,
$\beta < 49.3^\circ$, where $\nu < 1/2$ and $\alpha < 2/3$. For bursting bubbles that entrap a
tiny bubble, $\beta \simeq 38^\circ$ and $\alpha \simeq 0.63$, distinct from the
inertio-capillary exponent $2/3$. Jets from truncated cones, driven by a constant far-field
flux per unit length, follow instead $r_j \propto \tau^{1/2}$ with $We_j \propto r_j^{-1}$
[3, 5].

Which law applies is set by the flow far from the jet base, not by the cone angle alone [3].
Nor does [3] predict the jet speed as a function of the Faraday-wave height: relating $V_j$ to
$h - h_c$ needs a link between the wave height and the collapse that [3] does not provide,
and establishing that dependence is the second aim below. Near the singularity the
jet is thinnest and fastest, so the collapse geometry and the jet base are hardest to measure
exactly where the descriptions differ most. A clean simulation can resolve the geometry, the
jet base, the cone angle and the flow along the cavity wall together, with known parameters.

## Aim

Resolve the Faraday-wave-driven conical collapse and jet birth by direct numerical simulation,
and test:

1. whether the jet-base radius follows $r_j \propto \tau^{\alpha(\beta)}$, with $\beta$
   measured from the simulated collapse, rather than $\tau^{2/3}$;
2. how the jet speed $V_j$ diverges as the wave height $h$ approaches its singular value
   $h_c$, from both sides.

## Approach

- The solver adapts the surfactant-free path of the CoMPhy `faraday-surfactant-fem` code to an
  axisymmetric geometry: incompressible Navier–Stokes in the frame of a vertically vibrated
  cylindrical container, and a free surface with kinematic and dynamic conditions on an arbitrary
  Lagrangian–Eulerian mesh. The unstructured triangular mesh is smoothed hyperelastically and
  rebuilt from the current free surface when elements become too distorted.
- The container is a cylinder of radius $R_c = j_{1,1}/k$ with free-slip walls and a sliding 90°
  contact line, so the axisymmetric surface mode $J_0(kr)$ is exact in linear theory. Its trough
  lies on the axis.
- A sharp-interface mesh cannot change topology by itself. The simulations cover the collapse and
  the birth of the jet, and stop before any pinch-off or bubble entrainment, except through a
  reconnection: at a chosen neck radius $\delta$ the gas neck is cut with pyoomph's
  axisymmetric reconnection planner, the two new tips are rebuilt as circular arcs that meet the
  axis at right angles, the liquid velocity is carried onto the new mesh and projected onto a
  divergence-free field by a pressure impulse that vanishes on the free surfaces, and the run
  continues with the bubble at the passive constant gas pressure (the limit of zero gas density).
  A result of the cut is meaningful only if it does not depend on $\delta$.
- The step size and the mesh near the axis follow the local scale of the collapse: the apex radius
  of curvature, the jet-base radius or the narrowest neck, whichever is smallest. The scale may
  grow by at most a fixed factor per step, and each rebuild of the mesh coarsens it near the axis
  by at most a fixed factor.
- The mesh is also rebuilt when an element becomes too distorted (pyoomph's default criterion: an
  element shrunk below 0.6 or grown beyond 1.75 times its size when it was built, or with a fifth
  of its quality then). Before the mesh near the axis is first refined, pyoomph makes that rebuild
  within the solve. Afterwards the driver makes it, once the step has converged and before the
  step is measured: the element size near the collapse is set to the smaller of its current target
  and the measured spacing there, so that the rebuild never coarsens what the mesh resolves, over
  the region of the current local scale, and the rebuild is recorded.

## Model

Lengths are scaled with $1/k$, times with the capillary time $\sqrt{\rho/(\sigma k^3)}$. In the
container frame the body force is $-(Bo + \Gamma\cos\Omega t)\,\mathbf{e}_z$, with

- $Oh = \mu\sqrt{k/(\rho\sigma)}$, the Ohnesorge number;
- $Bo = \rho g/(\sigma k^2)$, the Bond number, zero in the present cases;
- $\Gamma = \rho A\Omega_d^2/(\sigma k^2)$, the container acceleration amplitude scaled by the
  capillary acceleration ($A$ the displacement amplitude, $\Omega_d$ the angular forcing
  frequency). With gravity, the usual reduced acceleration $A\Omega_d^2/g$ equals $\Gamma/Bo$;
- $\Omega$, the dimensionless forcing frequency, set to twice the inviscid frequency of the mode.

## Verification

`verificationCases/` compares the solver with the exact linear viscous theory of the $J_0$ mode:
free decay against the finite-depth dispersion relation and the linear initial-value response, and
forced growth or decay against the Floquet analysis of Kumar & Tuckerman [4]. The acceptance
criteria are stated in `verificationCases/README.md`.

## Layout

| Path | Contents |
|---|---|
| `src/` | reusable pyoomph code, as `src/<package>/` |
| `simulationCases/` | simulation entry points and `pyoomph-case-v1` case files |
| `testCases/` | small software-contract tests |
| `verificationCases/` | comparisons with exact or independently computed solutions |
| `validationCases/` | comparisons with independent experimental or numerical data |
| `modelComparisonCases/` | the inertial-cone, line-sink and inertio-capillary similarity laws |
| `postProcess/` | analysis and visualisation |
| `docs/` | approved authored documents |

Run outputs are not stored in this repository.

## Environment

pyoomph is built from source at the CoMPhy release `comphy-2026.09.21.1`
(`f912ac10cdbb170e58aca3a337f0aa4ed13cf1e0`), pinned in `pyproject.toml` and `uv.lock`, with MPI
enabled.

Runs of one case in fresh processes differ by about 1e-14 relative unless MKL, whose threaded
solve is the source, runs on one thread (`MKL_NUM_THREADS=1`; `OMP_NUM_THREADS` alone does not do
it); with it they have agreed to the byte, a restart after a cut included. Compare runs to
round-off, or set it when a comparison must be exact.

## Running

```bash
python run_case.py verificationCases/free-decay/free-decay-h0.025-n100.json --outdir <run-directory>
python verificationCases/compare_linear.py <run-directory> --out <analysis-directory>
pytest testCases
```

Each run directory receives `parameters.json` (inputs, dimensionless groups, numerical settings
and provenance), `run_status.json`, `diagnostics.txt` (axis height and velocity, mode amplitude,
volume, area, kinetic energy, dissipation and remesh count after every step) and free-surface
snapshots.

A case with a `collapse` block runs the collapse driver, which can resume from one of its state
dumps:

```bash
python run_case.py <collapse-case>.json --outdir <run-directory>
python run_case.py <collapse-case>.json --outdir <new-run-directory> \
    --restart <run-directory>/states/state_<step>.dump
python postProcess/collapse_table.py <run-directory> [...] --per-episode --out <table>.csv
python postProcess/collapse_analysis.py <run-directory> --out <analysis-directory> \
    [--rj-window <lo> <hi>]
```

Its run directory also receives:

- `collapse.txt`, one row per step: collapse cycle and crest/trough episode, apex height, velocity
  and curvature, jet-base radius `r_j` and height, flux, mean jet speed `V_j` and `We_j`, the
  narrowest neck, conservative fold-gap diagnostics (`fold_gap`, `fold_r`, `fold_z`), cone angles,
  resolution, global budgets, and a running estimate `t0_est`, with
  exponent `alpha_est`, of the origin of a power law in time fitted to the latest samples of a
  length (`t0_source`: 0 none; 1 the apex radius of curvature, an estimate of the singular time;
  2 the jet-base radius after birth: while the base closes in on the axis, the time at which it
  would reach it, after the samples; while the jet grows, the origin of that growth, a virtual one
  before birth when the jet is born at a finite radius);
- `events.jsonl`: cavities, jet birth, the first neck, crest and trough episodes (a trough carries
  the preceding crest height `h_crest` and `crest_jet`, whether a jet stood on the axis during that
  crest), remeshes, restarts and the stop. A remesh that element distortion asked for after the
  first refinement has `reason` `distortion`, with the criterion that fired (`trigger`, `ratio`),
  the extreme size and quality ratios, the element size near the collapse before and after
  (`focus_size_before`, `focus_size`) and the spacing there before and after (`h_near_before`,
  `h_near_after`); the earlier rebuilds, made by pyoomph within the solve, appear only in
  `remesh_count`. The `collapse` record of `run_status.json` counts both (`distortion_remeshes`,
  `pyoomph_remeshes`);
- `profiles/` (free-surface profiles with velocities) and `states/` (the latest state dumps).

A run ends at its end time, at the step-size floor or step limit, or right before a pinch-off or
the singular collapse. The `stop` record in `run_status.json` and `events.jsonl` names the reason:
`pinch-off-approach` when a neck becomes narrower than the stop radius (`trigger` `neck-radius`).
`self-contact-approach` (`trigger` `fold-gap`) is a conservative geometry cutoff in the focus
region: nonadjacent nodes are closer than `1/fold_ratio` of their intervening arc, and their gap
is below `max(fold_stop, fold_stop_elements * h_elem)`, using the local element length. The pair
closest to its own limit is recorded. This criterion can also stop at a static sharp corner;
it does not establish physical entrapment or a singularity. An already self-intersecting control
polygon, or a detected contact of its sampled spline, raises `InterfaceSelfIntersection` and
marks the run failed. The spline check is sampled and can miss loops below its sampling scale.
A run also stops when oomph-lib gives up on a step because the Newton solve fails at every step size
down to its smallest step (the message `less than the minimum dt` in the log): as a
`pinch-off-approach` if a neck is narrower than `neck_guard`, otherwise as a `singularity-approach`
if the measured local scale, before its growth per step is limited, is below `ell_guard` (both
with `trigger` `newton-failure` and the oomph-lib message as `error`, recorded from the last state
the driver measured, which can precede sub-steps that pyoomph accepted within the failing step
after halving it). Such runs are `completed`; any
other failure, including other oomph-lib errors and failed mesh generation or remeshing, marks the
run `failed` whatever the state. `collapse_table.py --per-episode` gives one row per trough episode,
and per collapse cycle within it, with the wave height `h_crest`, the crest height on the axis
above the mean level before that trough. Rows with `crest_jet` 1 had a jet standing on the axis
during that crest, so their `h_crest` is the height of the jet tip, not of a wave.

`collapse_analysis.py` sets out, for each collapse cycle that launches a jet, the evidence for
the three similarity laws of jet birth without choosing between them: the free slopes of
`Q_j`, `We_j` and `q_j = V_j r_j` against `r_j`, with the level of `We_j`; the cone angle and the
direction and decay of the surface velocity along the cavity wall at inception, over a fixed
intermediate window; and, as a secondary check, fits of `r_j(t)` with the origin fixed at the
apex-based singular time or fitted with the exponent. Every window and threshold is written to
its summary.

### Reconnection through a neck pinch-off

A sharp-interface mesh cannot pinch off, so a reconnected run is made in three restarts of the
production runner, each with the custody checks of any restart (the parent's normalised manifest,
a clean source commit, the case lineage) and the standard runner manifest
(`pyoomph-runner-manifest-v1`, which records `restart_mode`):

```bash
# 1. landing states: an existing run, restarted under a case derived from its own that adds
#    collapse.landing (the cut radii delta) and, if needed, a lower collapse.neck_stop
python simulationCases/run_faraday.py <landing-case>.json \
    --restart-from <run>/runtime/states/state_<step>.dump --out <landing-attempt>/runtime

# 2. one cut per delta: a case derived from the landing case that adds a collapse.reconnect block
#    with that delta, restarted from the landing state of that delta
python simulationCases/run_faraday.py <cut-case-delta>.json \
    --restart-from <landing-attempt>/runtime/states/landing_<delta>.dump --out <cut-attempt>/runtime

# 3. to go on after the cut run has ended (or was stopped): any dump it wrote after the cut, the
#    one it ended on included, under the same case or one derived from it in the keys a restart
#    may change (run_case.DERIVED_KEYS), a later collapse.reconnect.tau_end among them
python simulationCases/run_faraday.py <cut-case-delta>.json \
    --restart-from <cut-attempt>/runtime/states/state_<step>.dump --out <continuation>/runtime
```

`--check` validates any of these without writing output. The cut case is the landing case with,
in its `collapse` block, for example `"reconnect": {"delta": 0.001, "tau_end": 4e-5}` (the
settings of `src/faraday_singularity/reconnect.py`, `ReconnectSettings`) and a dump cadence for the
run after the cut, for example `"state_dt": 1e-5`, since the landing run's (0.1 by default) is
sized for the collapse before it; its other keys must be the landing run's
(`run_case.CUT_DERIVED_KEYS`), and its `landing` and `keep_stop_state`, inherited from the landing
case, are inactive after the cut (`inactive_after_cut` in `parameters.json`), so its cut radii are
not checked against its `neck_stop`, which a cut or continuation case may change. The run after a
cut also dumps its state at least each time the time after the cut doubles, and dumps the state it
ends on, whatever the stop. A run that ended at `tau_end` goes on from that dump under a case that
raises `tau_end` (or removes it): `tau_end` is the one reconnect setting a continuation may change,
and a continuation under the `tau_end` it ended at is refused before any output. The
`precut_<delta>` dump of a cut run is the landing state it cut: a restart from it under a case with
a reconnect block cuts again (`restart_mode` `cut`), never continues it before a cut.

**Landing states.** With `collapse.landing`, a list of cut radii `delta`, a run keeps for each the
first state whose neck radius `r_n` has fallen to `delta/1.05`, as `states/landing_<delta>.dump`, so
that one run supplies the landing states of a study in `delta`; with `collapse.keep_stop_state` it
keeps the state at its neck-radius stop as `states/stop_<step>.dump`. A cut radius that the run
cannot reach before its neck-radius stop (below 1.05 times `max(neck_stop, neck_stop_elements *
h_floor)`) is refused before any output, and a radius left without a landing state is recorded as
`landing-missed` with its reason. A landing state holds the latest samples of the closing neck and
the pinch time `t_p` fitted to them (a power law with a free exponent; `pinch`, with its window),
and a restart from one makes the remesh check of its step before stepping, as the run that wrote it
did. A restart may change only these keys, `collapse.neck_stop`, the cadence and number of
rotating dumps (`collapse.state_dt`, `collapse.state_keep`, at least 1, so that the latest dump
is kept; a dump writes the state and does not change it) and `collapse.reconnect.tau_end`
(`run_case.DERIVED_KEYS`); a `collapse.reconnect` block that differs from the parent's in any other
key makes the restart a cut, since the block compiles the reconnection terms into the problem and
changes its trajectory at round-off.

**The cut** (`src/faraday_singularity/surgery.py`) is made at the start of the cut run, from the
landing state, and refused unless `min_landing_ratio <= delta/r_n <= max_landing_ratio` (1.03 and
1.1 by default; the window must contain the landing window 1.05 to 1.1), since the cut geometry in
units of `delta` depends on `delta/r_n`. pyoomph's planner gives the topology; each of its caps is
replaced by a circular arc tangent to the old surface at the first surviving node beyond the
planner's exclusion window (`cap_window_factor`) whose radius is at least `delta`, and meeting the
axis at a right angle. On slender necks at `delta/r_n = 1.05` the bridge of liquid between the
bubble and the cavity is then 13-14.5 `delta` long and the cut removes 55-58 `delta^3` of gas; the
record gives the splice radii, the bridge and the removed gas in cut radii (`in_cut_radii`), and a
study in `delta` should vary the window too. The mesh of the cut is sized in cut radii, so that the
cuts of a study are meshed alike: the tips on the axis and the bridge (`tip_size`, `bridge_size`,
kept for every later remesh) and the focus of the cut's remesh (`focus_size` within
`focus_radius`), which replaces the focus the landing run left. Only the bubble's bottom, far from
the cut, keeps the landing focus where that is finer: it is the cavity's bottom of the landing
state, with its jet, and the cut makes no feature there (`focus.kept` in the record). gmsh's sizing
from curvature is off after a cut unless `curvature_sizing` asks for it: the caps do not need it
(an arc of radius at least `delta` meshed at `tip_size` of at most a quarter of `delta` has at
least 25 elements per turn), and gmsh reads the curvature of the spline through the nodes, which a
jump in node spacing makes spurious, so that it refines there without grading and widens the jump
at every rebuild, until the spline crosses itself. The liquid velocity is carried
across and projected onto a weakly divergence-free field by a pressure impulse that vanishes on
the free surfaces; the cut is refused unless the projection converges and the discrete energy
identity holds (the pointwise divergence of the velocity, `strong_divergence_L2`, is an order-`h`
discretisation diagnostic, not a test). The bubble keeps the passive constant gas pressure, the
limit of zero gas density. The run writes `cut_diagnostics.json` (the planner attempts, the replaced
caps, the volumes, the kinetic energy and axial momentum before and after the transfer and the
projection, also within 20 cut radii of the neck, the nodes the transfer could not locate in the
old mesh, the landing and cut foci, and the pinch time with its fit window) and keeps the
`precut_<delta>` and `postcut_<delta>` states. A refused or failed cut, whatever failed in it,
ends the run as `failed`, with its partial record and the stage it reached in
`cut_diagnostics.json` and under `cut` in `run_status.json`, which names the `precut` dump once it
was written. The liquid volume the cut adds is recorded both over the whole mesh, which
includes the drift of rebuilding the whole free surface, and within a box around the cut, which is
compared with the plan.

**After the cut** the collapse driver measures the open chain (the cavity, from its fresh tip to
the wall) as any free surface, jet included, and the jets of the bubble's two tips in the same way,
the bubble's top mirrored so that the jet the bridge drives down into the bubble is measured as one
rising from a cavity: for each of the three tips `chains.txt` holds the jet-base radius and height,
the flux through the base, the mean jet speed and `We_j` (`r_j_<tip>`, `z_b_<tip>`, `Q_j_<tip>`,
`V_j_<tip>`, `We_j_<tip>` and `jet_born_<tip>`, with `<tip>` one of `cavity_tip`, `bubble_bottom`
and `bubble_top`, and `Q_j` and `V_j` positive along each jet), so that the jets of a study in
`delta` can be compared. The local scale that sets the step is the smallest over both chains (on
the bubble its jets, its neck and its length, `ell_bubble`; the radii of curvature of its tips,
which depend on the mesh at the end of a chain, bound the step instead) and the radius of
curvature at the base of each jet, the apex of the gas groove around it (`R_apex`, from circles
fitted through the vertices of the elements only). The mesh follows each feature of both chains in
a region of its own, sized by the feature's own scale, so that no feature is met by only one or
two elements: the two tips the cut made, the cavity's and the bubble's top, at their own scale
(the smaller of the tip's radius of curvature and the radius of the jet on it) over
`elements_per_scale`, between the `h_floor` of the run and the cut's focus size, which is the
coarsest size allowed near the cut, not a fixed one, the rest of the cut's region staying at the
cut's focus size; the base of each jet at its apex radius over `elements_per_scale`, predicted
`apex_lookahead` steps ahead from its rate of change, within `apex_extent` apex radii of the base;
the bubble's bottom at the size its jet asks for, but never coarser than at the landing; each
neck at its own scale. A rebuild refines or coarsens a region by bounded factors when its longest
element exceeds `refine_ratio`, or its shortest (outside the finer regions within it) lies below
`coarsen_ratio`, times its target, and a rebuild that changed none of the regions that asked for
it waits before the next; a rebuild for element distortion keeps every region's size.
`chains.txt` gives at each tip the longest element of its region and the size the mesh has there
(`h_<tip>`, `size_<tip>`), at the base of each jet the apex radius, the gap of the closest fold
pair across the groove and the groove's opening angle (`R_apex_<tip>`, `groove_gap_<tip>`,
`groove_angle_<tip>`), and the `remesh` events every region.

The interface is also read on its quadratic elements, after every accepted step and on the input
of every rebuild: an element whose tangent turns by more than `edge_turn_refine` (30) degrees
across it, or whose midside node lies more than `midside_refine` (0.1) chords from mid-chord, is
refined in a region of its own (`kink_<chain>`, a quarter of its chord); beyond
`edge_turn_stop_fresh` (90) degrees on a freshly built mesh or `edge_turn_stop` (120) on any, beyond
`midside_stop` (0.2) chords, or where the element is already at `h_floor`, the run stops cleanly as
`interface-unresolved` (`trigger` `edge-turn`, `edge-midside` or `feature-below-floor`), since a
mesh that does not represent the interface admits no meaningful step. `chains.txt` records the
largest turn and midside offset of each chain (`edge_turn_<chain>`, `edge_midside_<chain>`). A
rebuilt mesh is accepted only if the lower bound of the scaled Jacobian of every curved element,
from the Bezier coefficients of its Jacobian determinant, is at least `jacobian_min` (0.2;
`trigger` `curved-jacobian`); gmsh's optimiser of curved elements is asked for
`high_order_threshold` (0.3); an element whose quality falls below `quality_floor` (0.15) asks for
a rebuild; and pyoomph's net for an element that turns inside out within a step is armed. The
transfer of the solution to a rebuilt mesh interpolates the node positions as well, and it has left
a midside node of the axis far off its edge, turning its element inside out; so every rebuild after
a cut first puts such a node (more than 0.1 edge lengths off the middle of its straight edge) back
at the mean of its edge's vertices, in position and velocity at the current and history levels,
and leaves every other node alone (`axis_repair` in the `remesh` event). The driver in addition
stops before a second change of topology: the bridge closing, the chains
approaching each other, the bubble's two tips meeting, a new neck narrower than `delta` on either
chain (only a neck at least `neck_depth`, 0.25, narrower than the lower of the widest points on
either side of it along the chain, or narrower than three of the smallest elements whatever its
depth: a ripple on a jet thinner than `delta` is no neck; the stop says whether the gas or the
liquid lies inside the neck), a closing fold on either chain, a spline through the nodes of either chain that contacts
itself (`self-contact-approach` with `trigger` `spline-contact`), the bubble draining away, or
`tau_end` after the cut. A closing fold whose pair of nodes straddles the base of a jet, the
annular groove around the jet closing, is the `annular-contact-approach` (`trigger` `fold-gap`,
with the jet, its apex and `R_apex`); any other is a `self-contact-approach`. These stops are
checked before every rebuild, those for element distortion included, with the rebuild's own test
of the spline, so that no rebuild meets a self-intersecting spline; a self-contact stop says whether
its two sides lie at least four elements apart along the surface, a fold of the resolved surface
(`contact` `fold`), or closer, a kink at the scale of the mesh (`kink`). The step also follows every
node of both chains (a Courant limit `courant h/|u|` over each node's element length), the
capillary time of the sharpest of the three tips on the axis and of the bases of their jets, and
the cut's `dt_growth`; with `focus_courant` (the default) also the collapse's rule, the finest focus
size over the largest surface speed anywhere. Each step after the cut makes at least
`newton_min_iterations` (2) Newton iterations whatever its tolerance: at the small steps after a
cut the tolerance `newton_floor/dt` passes the first iterate, whose systematic error grows into
noise at the scale of the elements. After a cut, no Newton failure ends a run cleanly; the profiles of the
state the failing step started from are written first (`profile_<step>_lastgood.txt` and its
`_bubble` beside it), and a profile is written at every step while an element of either chain
turns by more than `edge_turn_refine`. For sensitivity tests only, `rebuild_spline` `vertices`
splines each rebuild through the vertices of the elements rather than every node; it is off by
default, since smoothing would alter, and hide, the very feature under test. All these settings
belong to the cut: a continuation keeps them.
Every output after the cut carries the time after the cut `tau = t - t_cut` and after the pinch
`tau_p = t - t_p`: the columns `tau` and `tau_p` of `collapse.txt`, `chains.txt` (both chains per
step: the three tips with their speeds, radii of curvature and jets, the bridge, the bubble, the
necks, the budgets and the step limits), the profiles of the open chain with a `_bubble` profile
beside each, the events and the dumps. The dumps carry the cut's record, from which a restart
restores the cut and continues. The cuts of a study in `delta` are compared at fixed `t - t_p` with
one common `t_p`, since `t_cut` depends on `delta` and fits of `t_p` over different windows differ;
a reconnected result counts only if the jets do not depend on `delta`.

**Events after the cut.** A `collapse.reconnect_events` block beside `collapse.reconnect`
(`src/faraday_singularity/events.py`, `EventSettings`) governs how a run after the cut meets the
changes of topology that follow it. It has three parts: `liquid`, the necks on the liquid side at
the roots of the jets the cut launched; `gap`, the closing folds of either chain (a gas groove
narrowing); and `post`, the limits, floors and mesh sizes of the run after the cut, each by default
the collapse's own. A case without the block runs exactly as before the block existed: the tests
run a cut, the run after it with rebuilds and dumps, and a restart from one of those dumps, against
the source of commit `1b65916`, to the byte. The block applies after the cut only, and needs a
`reconnect` block: a cut restart may add, drop or change it together with the reconnect block
(`run_case.CUT_EVENTS_DERIVED_KEYS`), a continuation from a dump written after the cut may add or
change it (`run_case.POST_CUT_DERIVED_KEYS`), and a run from the start or an ordinary restart that
carries it is refused before any output. The dumps written after the cut with the block record it
(`events_settings`), the events made (`events`) and the legs of the run (`legs`, also as events
`leg`: the cut and each restart, with the step and time it starts at, the source commit, the hash
of the block and the keys in which it differs from the block the state was written with). Such a
dump continues only under a case with the block. The keys of `liquid` and of `gap` are the keys of
a class of events: once an event of the class has been made they may not change, except in a
restart from the state before an event (`precut_e<k>`), from which they apply anew; the keys of
`post` may change at any restart. With `post.bubble_bottom_sizing` `frozen` the bubble's bottom,
far from the cut, is meshed never coarser than `post.bubble_bottom_size` rather than never
coarser than at the landing (`kept`, the default), so that the bubbles of a study in `delta` are
meshed alike there.

With the block, the stops of a state after the cut come in three ranks
(`events.post_cut_events`). A state that is not valid stops the run: the chains not two, a spline
through the nodes that contacts itself, on a freshly built mesh a tracked liquid neck narrower than
`liquid.resolution` (4) element lengths at it (`liquid-neck-unresolved`), the guard of the interface
and the check of the curved elements; on a mesh a step has run on, the guard comes last, after the
events, as in the cut's own stops (an event's handler rebuilds the mesh). The bubble emptying (below `bubble_stop` of its volume at the
cut) ends the window (`bubble-vanishing`, with `window_end`). Every change of topology is then an
event, in this order: the bubble's two tips within `delta` of each other, the downward jet piercing
the bubble (`piercing`); a landed liquid neck (`liquid-neck`); a closing fold, one that lands the
neck of a head it reaches (`a-head`) on either chain before any closing groove (`groove`, the one
closest to its limit), each head's own fold searched apart from the fold of its chain so that a
closer fold elsewhere hides none; a gas neck (`gas-neck`); the bridge closing (`bridge-closing`)
and the chains approaching (`chain-contact`). A fold at the scale of the mesh (`contact` `kink`)
is no event: it stops the run as the cut's own stops do. A neck on the liquid side is followed from
`liquid.track_depth` (0.05) below its shoulder. Its head is the side of it whose far end lies on the
axis, that holds neither the wall end nor the bubble's widest point, lies on one side of the neck's
height and bounds a simple section of at most `liquid.max_head_volume` (20) times the cube of its
largest radius `s`; the neck lands once `r_n <= liquid.eps_d s` (0.2), the one nearest its landing
first. A fold of either chain closes at `max(gap.kappa_g delta, fold_stop_elements h_elem)`
(`kappa_g` 0.05; null for the collapse's absolute `fold_stop`), and one that reaches the head of a
tracked neck lands that neck at once if `r_n <= liquid.eps_max s` (0.5). With `liquid.enabled`
false a liquid neck is met at the cut's own test of a second pinch-off instead. Each event names
the handler of its class (`events.HANDLERS`); a landed neck without a single head or with one too
large, or met with liquid events off, whose head is then never classified, is of class
`L-unhandled`, which has none. No event has a handler yet: the run stops on the first
(`event-unhandled`, with the event's number `k`, its `kind`, `class` and the handler of its class)
and keeps the state before it as
`states/precut_e<k>.dump` (`cut_stage` `precut-event`), besides the state it ends on; a
continuation from it meets the event again, and its handler once one exists.

The limits and floors of the run after the cut are the block's `post` keys where set, the
collapse's own otherwise, so that a study in `delta` can scale them with `delta`: `h_floor` (the
smallest focus and feature size, the floor of the interface guard and its kink regions, and the open
chain's neck stop through `neck_stop_elements h_floor`), `rise_tol` (the rise that makes a jet),
`dt_min` (the step-size floor), and for the regions of the necks `neck_floor` (the cut's focus size
without it, so that a neck can be followed below the cut radius) and `neck_elements_per_scale`
(`reconnect.elements_per_scale` without it). The fold limit after the cut, also that of the groove
gaps `chains.txt` reports, is `gap.kappa_g delta`. With `post.open_neck_stop` false the open chain's
own neck stop (`collapse.neck_stop`) is off after the cut, as liquid events need: its liquid neck is
then followed as an event. At the cut and at every restart after it the block also sets oomph-lib's
smallest step (`DTSF_minimum_dt`, 1e-12 by default) to `post.minimum_dt`, or 1e-3 times the
step-size floor without it, never above the default, and records it in the leg, also in units of
the spacing of doubles at the leg's time (`minimum_dt_ulps`). A `minimum_dt` above the default, or
above half the step-size floor after the cut (a step failing at the floor would not be halved and
would end the run as a failed one), is refused before any output. oomph-lib gives up on a step below it, and
also after an accepted step whose next candidate, four times the step, would fall below it, so that
at the steps a small cut radius needs the default would end the run on a converged step. Before the
cut it keeps the default, on which the classification of a failure before a cut relies.

`simulationCases/reconnect_prototype.py` is a development tool beside this: it cuts a landing state
at `--delta` or `--delta-ratio` times its neck radius, with options that override the case's
reconnect block, and takes `--steps` steps under the same driver and post-cut mode, without the
events block, which it refuses. Its manifest has its own schema, so its outputs feed no figure, and
no runner continues its states.

## References

1. B. W. Zeff, B. Kleber, J. Fineberg and D. P. Lathrop, Singularity dynamics in curvature
   collapse and jet eruption on a fluid surface, *Nature* **403**, 401–404 (2000).
   [doi:10.1038/35000151](https://doi.org/10.1038/35000151)
2. M. Cattaneo, L. Presse and O. Supponen, Faraday wave singularities trigger microbubble
   jetting, *Phys. Rev. Lett.* **135**, 204002 (2025).
   [doi:10.1103/7c4p-y9jr](https://doi.org/10.1103/7c4p-y9jr)
3. J. M. Gordillo, J. Rodríguez-Rodríguez and V. Sanjay, Self-similar Worthington jets,
   arXiv:2607.08972 (2026). [arXiv:2607.08972](https://arxiv.org/abs/2607.08972)
4. K. Kumar and L. S. Tuckerman, Parametric instability of the interface between two fluids,
   *J. Fluid Mech.* **279**, 49–68 (1994).
   [doi:10.1017/S0022112094003812](https://doi.org/10.1017/S0022112094003812)
5. J. M. Gordillo and F. J. Blanco-Rodríguez, Theory of the jets ejected after the inertial
   collapse of cavities with applications to bubble bursting jets, *Phys. Rev. Fluids* **8**,
   073606 (2023).
   [doi:10.1103/PhysRevFluids.8.073606](https://doi.org/10.1103/PhysRevFluids.8.073606)
