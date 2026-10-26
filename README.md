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
python postProcess/event_table.py <run-directory>[:<next-leg-directory>...] [...] \
    --out <events>.csv --runs <runs>.csv [--tip cavity_tip bubble_top]
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
turns by more than `edge_turn_refine`. With pyoomph's net for inverted elements armed (the
default after a cut), a failed step is rolled back to the state it started from, the net's own
snapshot before the step, and the run ends on that state as the resumable stop `step-failed`
with it dumped, so that a continuation can resume from it; in the ring stage the step is first
made once more after the ring's sharpest element is rebuilt at the size its turn asks for.
In the region away from the upward jet,
the tip of a fast jet on the axis that folds along the axis within one element (its last element
running up the axis, a node of it across it) is repaired: the folded nodes are dropped and the
axis node put where the surface meets the axis, by a rebuild from the planned surface, and the
thin sliver of gas between them becomes liquid. Its geometric volume and independently sampled
axial momentum after transfer are hard checks above
`1e-4 delta^3` for volume and that bound times the folded nodes' largest speed for momentum
(a fold of a tip element is far smaller). Momentum is integrated on the sliver's own triangle
quadrature; every point must be located and its volume must reproduce the geometric volume
within the existing volume tolerance, or the momentum check is refused as unchecked. The budgets
of the rebuild and the base of the upward jet are checked as well. Its transfer must keep the
chain names and ends and the planned axis spans, with unlocated nodes only where the old chains and the sliver plus its band predict them;
a repair that misses is rolled back and the run stops resumably.
For sensitivity tests only, `rebuild_spline` `vertices`
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
the collapse's own. A case without the block runs exactly as before the block existed: the tests run
a cut, the run after it with rebuilds and dumps, and a restart from one of those dumps, against the
source of commit `1b65916`, to the byte, the meshes gmsh wrote included, also with fewer dumps kept
than written. That finite-element run meets no neck, no jet at the
bubble's bottom and no stop but its step limit; stand-in problems with prescribed chains cover those
against the same source (`testCases/test_events_legacy.py`), and the runner of a continuation after
the cut, with stand-ins of the problem and the driver, writes the `run_status.json` of that commit's
runner (`testCases/test_events_run.py`). A replay of a production run after its cut remains the
full check. The block applies after the cut only, and needs a `reconnect` block: a
cut restart may add, drop or change it together with the reconnect block
(`run_case.CUT_EVENTS_DERIVED_KEYS`), a continuation from a dump written after the cut may add or
change it (`run_case.POST_CUT_DERIVED_KEYS`), and a run from the start or an ordinary restart that
carries it is refused before any output. The dumps written after the cut with the block record it
(`events_settings`), the events made (`events`) and the legs of the run (`legs`, also as events
`leg`: the cut and each restart, with the step and time it starts at, the source commit, the hash of
the block, the keys in which it differs from the block the state was written with, and the limits
and floors it runs with, resolved, also in units of `delta` and `delta^2`, `limits`, so that a study
in `delta` shows which of them it scaled with `delta`, and the events made before it, `events_made`;
a leg restarted from a dump writes the record of each event made before it, as the dump holds it,
into its own `events/`, `copy` naming its leg and the leg that made it, so that the directory of
every leg holds the records of every event of its run up to its end) and whether a step has run on
its mesh since it was built (`mesh_fresh`): a restart from a mesh no step has run on checks it as a
fresh mesh, as the run that dumped it did, unless the restart changes a key that sizes the mesh.
Such a dump
continues only under a case with the block. The keys of `liquid` and of `gap` are the keys of a
class of events: once an event of the class has been made they may not change, except in a restart
from the state before an event (`precut_e<k>`), from which they apply anew; the keys of `post` may
change at any restart, and so may the block's own: `max_events` (1000), the most events a run makes
over all its legs, and the test of a cascade of events at one tip (`cascade_min_interval`, 0.25,
in `delta^2`; `cascade_ratio`, 0.7; `cascade_count`, 3). Of `post`, `keep_event_dumps` says which
dumps of the events a run keeps (`all`, or `first<N>-last<M>`: the state before each of the first
`N` events and both states of each of the last `M`), `prune_gmsh` (true) deletes the meshes
gmsh wrote for the run that no dump refers to and `log_box_noise` (false) logs the noise of a liquid
cut's box in the ordinary rebuilds (below). With `post.bubble_bottom_sizing` `frozen` the
bubble's bottom, far from the cut, is meshed never coarser than `post.bubble_bottom_size` rather
than never coarser than at the landing (`kept`, the default), so that the bubbles of a study in
`delta` are meshed alike there; within a few of the cut's tip sizes (`tip_size delta`) of the
bottom's tip the cut's sizing of every end on the axis still applies, and the cut's own mesh has the
size the cut keeps there from the landing until the first rebuild after the cut.

With the block, the stops of a state after the cut come in three ranks (`events.post_cut_events`). A
state that is not valid stops the run: the chains not two, a spline through the nodes that contacts
itself, on a freshly built mesh a tracked liquid neck narrower than `liquid.resolution` (4) element
lengths at it (`liquid-neck-unresolved`; with `pinch.far_head` a far head's is cut instead,
below), the guard of the interface and the check of the curved elements; on a mesh a step has run
on, the guard comes last, after the events, as in the cut's own stops (an event's handler rebuilds
the mesh). The bubble emptying (below `bubble_stop` of its volume
at the cut) ends the window (`bubble-vanishing`, with `window_end`). Every change of topology is
then an event, in this order: the bubble's two tips within `delta` of each other (within
`max(ring.gap delta, 3 h_min)` with `ring.enabled`, below), the downward jet piercing the bubble
(`piercing`); a landed liquid neck (`liquid-neck`); a closing fold, one that
lands the neck of a head it reaches (`a-head`) on either chain before any closing groove (`groove`,
the one closest to its limit), each head's own fold searched apart from the fold of its chain so
that a closer fold elsewhere hides none; a gas neck (`gas-neck`: narrower than `delta` and at
least `neck_depth` narrower both than the widest points on either side of it along its chain and
than its adjacent shoulders, the widest points before the surface narrows below it again, or
narrower than three of the smallest elements whatever its depth); the bridge closing
(`bridge-closing`) and the chains approaching (`chain-contact`). A fold at the scale of the mesh
(`contact` `kink`) is no event: it stops the run as the cut's own stops do. A neck on the liquid
side is followed from `liquid.track_depth` (0.05) below its shoulder. Its head is the side of it
whose far end lies on the axis, that holds neither the wall end nor the bubble's widest point, lies
on one side of the neck's height and bounds a simple section of at most `liquid.max_head_volume`
(20) times the cube of its largest radius `s`; the neck lands once `r_n <= liquid.eps_d s` (0.2),
the one nearest its landing first. A fold of either chain closes at `max(gap.kappa_g delta,
fold_stop_elements h_elem)` (`kappa_g` 0.05; null for the collapse's absolute `fold_stop`), and one
that reaches the head of a tracked neck lands that neck at once if `r_n <= liquid.eps_max s` (0.5).
With `liquid.enabled` false a liquid neck is met at the cut's own test of a second pinch-off
instead. Each event names the handler of its class (`events.HANDLERS`); a landed neck without a
single head or with one too large, or met with liquid events off, whose head is then never
classified, is of class `L-unhandled`, which has none. The run makes the cut of a landed liquid
neck, `L` or `A-head` (below), within itself and goes on; it stops on any other event, resumably
(`event-unhandled`, with the event's number `k`, its `kind`, `class` and the handler of its class:
the split of a gas neck is still to be built, the ring of a piercing is made only with
`ring.enabled` and the fill of a groove only with `groove.enabled`, below),
and keeps the state before it as `states/precut_e<k>.dump` (`cut_stage` `precut-event`), besides
the state it ends on, with the record of the event (`events/event_<k>.json`, `outcome`
`unhandled`); a continuation from it meets the event again, and its handler once one exists. The
bubble's tips meeting ends no window: without `ring.enabled` the piercing is such a stop. The
keys of `liquid` also set the cut of a landed neck: the cap that closes the stump (`liquid.cap`,
`hemispheroid`, of height `liquid.cap_height`, 1, in neck radii and at most one), the
velocity of the cap's nodes (`liquid.cap_fill`: `sample`, the old bulk velocity at their places, or
`transfer`, what pyoomph's transfer gives them, an arm only), the projection after the cut
(`liquid.projection`, true) and the first step after it (`liquid.restart`, null for the cut's
`restart`), and the radius of the budgets of the cut (`liquid.box_radius`, 4: within `max(box_radius
delta, 2 L_rem)` of the neck, `L_rem` the reach of the removed head along the axis) with the floors
of their limits (`liquid.sigma_V`, `sigma_Pz`, `sigma_KE`, 0: the noise of the box in an ordinary
rebuild, of its liquid volume, axial momentum and kinetic energy, in the solver's units, none until
calibrated). `post.log_box_noise` calibrates them: before and after every ordinary rebuild after the
cut (the rule's and one for distortion) it reads the budgets of such a box about each tracked neck
with a head, at the radius its cut would take, and about each tip on the axis, at `box_radius
delta`, and writes their changes to `box_noise.txt` (a row per box and rebuild), with the largest
and the rms change of each in `events.box_noise` of the run's result; the reading changes no value
of the run.

**The cut of a liquid neck** (`surgery.liquid_cut`) removes the head of a neck that lands under its
own `liquid` keys (a request made under others is refused) and closes the stump at the neck by a
half-spheroid of `liquid.cap_height` neck radii, tangent to the neck and inside the old liquid,
whose points start at the node spacing of the neck and grow to the cap's own
(`events.truncate_chain`); a cap that leaves the old liquid, folds back, makes a knife-edge angle at
the splice, has too few points for the neck's spacing or leaves chains that cannot be meshed is
refused, and tried once more at half a neck radius (`events.CAP_RETRY_HEIGHT`) unless that is its
height. pyoomph's planner sees only the necks of the gas, so the plan of the new chains is built in
the project (`events.build_liquid_plan`) and parked as pyoomph parks its own (`surgery.park_plan`):
the nodes kept keep their place on the old chart and the cap takes the chart just beyond the neck,
so that the remesh carries the fields across as through a pinch pyoomph planned. The mesh is built
under the driver's focus with a region at the cap (`r_n` over the elements per scale), named after
the cut tip, while the regions the cut tip and the cut neck had keep their sizes for this mesh
under the event's names (`<tip>_e<k>`, `neck_<chain>_e<k>`; `events.event_focus_regions`): a later
rebuild sizes each feature from the region of its name, so it sizes the new tip from the cap, a
rebuild for distortion included, and a new neck of the chain from its own spacing. The transfer is
checked against the geometry (`surgery.transfer_problems`): pyoomph reports counts, not nodes, so
its count of unlocated interface nodes must lie within two of the count of those farther than half
an element from every old element its locator tries (`events.predict_unlocated`), every predicted
one on the cap, which must have nodes, with no node of the bulk or the axis unlocated and at most
the stump's tip on the axis without a value. pyoomph gives the unlocated nodes the velocity of the
old surface near the neck, so the cap's nodes are given the old bulk velocity at their places,
sampled along the planned cap before the remesh (a cut is refused unless it is found at nine tenths
of the samples). The pressure-impulse projection follows, with the impulse zero on every free
surface, the cap's included, the step before the cut as the history's and the time unchanged. The
budgets are taken within the box of `liquid.box_radius`: the liquid volume and the area of the
interface from the polylines clipped to it, the axial momentum and the kinetic energy on a polar
grid, of the box and of the removed head, before the cut, after the transfer and after the
projection, and the identity of a smooth cutoff `phi` of the box in the projection's weak form, `int
(u+ - u~)_z phi dV = int Pi dphi/dz dV`. The volume and the axial momentum of the removed head must
close against the change of the box's, to a hundredth of the head's volume and a twentieth of its
momentum, and the projection may change the box's momentum by at most a twentieth of the head's,
each limit never below five times the noise of the box (`liquid.sigma_V`, `sigma_Pz`); the kinetic
energy the transfer takes against the head's (to a twentieth, or five times `liquid.sigma_KE`), the
energy the projection changes either way (a tenth of the head's, or three times `sigma_KE`) and the
area (a hundredth of the head's less the cap's) are flagged beyond their limits, and the identity is
recorded without a limit until the noise of the box is calibrated (`closures`). The record
(`events/event_<k>.json`, schema `faraday-event/2`) holds the trigger with the neck's Reynolds and
Weber numbers (and the base of the jet whose neck it cut, `r_j`, `z_b`, `r_j_over_r_n` and the
neck's distance from the base in jet radii, which the driver adds), the cap with its fill and the
elements per neck radius it was built with (`cap.fill`, `cap.elements_per_scale`), the removed head
(volume, axial momentum, kinetic energy, areas and reach), the head as the ghost a run is to follow
(its polygon, in the file only, its mean axial speed, equivalent radius, the retraction speed
`r_n^(-1/2)` of its rear and its two ends), the totals of the removals since the gas cut
(`removed_total`), the transfer, the projection and the budgets, and the states before and after the
cut are kept (`states/precut_e<k>.dump`, `states/postcut_e<k>.dump`). A refusal leaves the problem
unchanged; a failure after the remesh ends the run, and so do hard budgets that do not close, but as
a stop to resume (`outcome` `unclosed`, `surgery.BudgetUnclosed`): a restart from `precut_e<k>`,
under calibrated noise floors, cuts again. The tests make it on a synthetic state with a jet on each
chain (`testCases/test_liquid_cut_fem.py`).

**The events within the run.** At each check of its stops (before a step, after it and after a
rebuild) the run meets an event with a handler it makes, the cut of a landed liquid neck, by
making it: with its last step as the history's step of the cut, it removes the head, measures the
fresh mesh and checks its stops again as those of a mesh freshly built, then goes on from it. No new
cycle begins: the births of the jets, the series of the singular time, the holds of the rebuilds and
the last step are kept, while the growth limit of the local scale, the history of the cut tip's jet
base and the kink region of its chain restart; the bases of the jets and the kink regions of the
fresh mesh are committed before the state after the event is kept, so that a restart from it steps
as the run does. The driver adds to the event's record what it
measured of the state before it (the cut jet's base), the event's generation at its tip, its step
and leg, its times in units of `delta^2` and of `T = t_p - t_cut`, the verdict of the cascade test,
the source commit, the start of the ghost and the cut jet after the cut (`after`); `events.jsonl`
gets an event `liquid-cut`, `profiles/` a profile, and `events.txt` a row per event (the trigger,
the cap, the removed head and its ghost, the jet's base before and after, the closures relative to
the head's own, the transfer, the interval since the event before at its tip, the cascade's verdict,
the totals removed). The ghost of each removed head is followed after its cut, rigidly at its mean
axial speed with its rear retracting towards it at `r_n^(-1/2)` until the ghost is as long as the
sphere of its volume (twice its equivalent radius; a head shorter than that keeps its length), so
that its rear never passes its front: the stump's tip catching its rear and its front reaching the
far tip of the bubble (from two steps after the cut on) are flags only,
the bubble mattering only for the entrainment, its collapse and the budgets (`ghost.recontacts` in
the record and its file in the directory of the leg that flags them, an event `ghost-recontact`, a
row of `drops.txt`, which also has a row for each ghost made). Every removal enters the budget of
the run: `budget.txt` has a row per step
after the cut with the liquid of the mesh (its volume, axial momentum and kinetic energy, as in
`chains.txt`), the volume, momentum and kinetic energy the events have removed since the gas cut
(each head's at its cut, whose volume and momentum the cut checks against the change within its
box) and their sums. Under `post.keep_event_dumps` `first<N>-last<M>` the dumps of the events made
that the policy no longer keeps are deleted as the events come (`dumps.dropped` in their records
and the records' files). With `post.prune_gmsh` the run deletes, after every dump and every event,
the meshes gmsh wrote for it that no dump in its directory refers to and that the problem no longer
holds, so that every dump kept restarts with its mesh (nothing is deleted where a dump cannot be
read). The run stops resumably, with the dump of the state before the event, which is always kept,
and the record of the event (`outcome` `max-events` or `cascade`; a refused or unclosed cut's own
record, with that dump), at the event after
`max_events` (`max-events`), at one that the cascade test of its tip judges to shrink
(`event-cascade`: an interval since the last at the tip below `cascade_min_interval delta^2`, or
`cascade_count` successive ratios of the intervals and of the heads' volumes, each below
`cascade_ratio`; a cascade that grows, or stays stationary, is recorded), at a cut refused before it
changed anything (`liquid-cut-refused`) and at one whose hard budgets do not close
(`event-unclosed`), after which the problem holds a cut that is not accepted: the run then ends on
the state before the event (`end_state`) and dumps nothing more. Any other failure of a cut fails
the run. With the block `run_status.json` has, in its `collapse` record and at its top level,
`events` (the events made, by kind and by tip, the last, the totals removed, the cascade's
verdicts, the ghosts' recontacts, the dumps and meshes deleted, the log of the box's noise,
`box_noise`, and the event the run stopped on) and `window_end` (the end of the window the run
stopped at, the bubble emptying); a run that failed in a liquid cut records that event (`event`:
its number, kind, class, outcome, the stage it failed at, its record and the dump of the state
before it).

The block's `ring` part (off by default, and written into the block and its hash only where it
differs from its defaults, so that every block recorded before it is unchanged) makes the cut of a
bubble its downward jet pierces (`ringsurgery.ring_cut`, class `W1`) and keeps the bubble's rest as
a ring of gas, at the passive constant (zero) pressure of the whole free surface, until it
collapses. The ring is a closed loop of the free surface's own boundary: pyoomph refuses a problem
with equations on a boundary its mesh lacks, so a boundary of the ring's own could not load a state
written before the ring, while on the free surface's it needs nothing compiled; the problem's
arclength chart charts the loop beyond the open chain (`problem.RingTolerantArclength`, the same
chart as pyoomph's without a loop) and every rebuild with a loop transfers the interface by
projection onto the old one. With `ring.enabled` the piercing is met at `bubble_length <
max(ring.gap delta, 3 h_min)` (`gap`, 1). The cut (`ring.ring_cut_geometry`) splices the closure of
the liquid column it makes to the downward jet's cap at the radius `rho = ring.rho s_t` (`rho`, 0.5,
at most 0.8; `s_t` the radius of the jet's shoulder) and to the plug below: an arc, a vertical
segment of radius `ring.waist rho` (`waist`, 0.5) and an arc, tangent to both, its points at
`ring.cut_elements` per waist radius (null for the cut's elements per scale); it is refused before
anything changes where the old mesh does not hold `rho` (at least 1.5 element lengths at the
splices), where the closure leaves the old gas or makes a knife-edge angle, and where the ring would
be born below `bubble_stop` of the bubble's volume at the gas cut. The rebuild is the project's own,
the transfer checked against the nodes the geometry predicts it cannot locate (on the closure, in
the column, on the axis between the bubble's old tips), then the fill of the former gas
(`ring.fill`: `blend`, pyoomph's nearest-node blend, or `zero`) and the pressure-impulse projection
(`ring.projection`, `ring.restart`). Its budgets within `max(ring.box_radius delta, 2 L_fill)` of
the column (R1-R6): the liquid volume gained against the filled volume, the axial momentum the
transfer carried against the filled region's, and both momentum statements of the projection in the
box, the identity of the box's cutoff in the projection's weak form (R3, against the larger of its
two sides) and the box's own change of momentum (`R3_box`, against the filled volume times the
closing speed, as for a liquid cut), are hard checks, the kinetic energy and the area flags, each
limit floored at a multiple of the ring's own noise floors (`ring.sigma_V`, `sigma_Pz`, `sigma_KE`,
0 until set); a hard miss is a resumable stop (`event-unclosed`). The transfer is refused where a
node is left without a value off the corner set or the closure holds no node. The ring's mesh is
frozen at the cut: nowhere coarser than `max(h_cut, ring.size)` (`size`, null), the bubble's finer
regions kept as copies (`ring_<name>`) and the column's region (`ring_cut`) at their sizes. In the
ring stage the open chain is read as before (its necks, folds, heads and gas neck), and a landed
neck there is cut as before the ring (`surgery.liquid_cut`, the record's `stage` `ring`), but
through the ring stage's own rebuild and a transfer by projection onto the old interface, not
through pyoomph's plan, whose chart cannot carry the ring's loop: the cap must also clear the ring,
the ring must come through the rebuild kept (one loop, its nodes within a tenth of their count
before, the volume of its polyline within five times `ring.sigma_V` besides what resampling its
quadratic surface can change that volume by; `loop` in the record, a failure otherwise), and the
polylines of the cut's box hold the ring's gas where the box reaches it; after the ring's removal
the same rebuild has no loop. The ring's own events are resumable stops: its approach to the axis
(`ring-axis`, class `RA`, below `max(ring.axis_ratio r_w, 3 h_col)`), the closing of one of its
folds (`ring-fold`, `RF`; filled with `groove.enabled`, below) and its approach to the open chain
(`chain-contact`, `X`).
Once its volume falls below `bubble_stop` of the bubble's at the gas cut
(`ring-collapse`, class `W2`) it is removed (`ringsurgery.ring_removal`) by the same sequence and
closures (Q1-Q6), the removal logged; `ring.after_removal` `stop` (the default) then ends the window
(`bubble-vanishing`, with `window_end` and `ring_removed`), `continue` goes on with the open chain
alone. The ring's keys are those of a class of events, fixed once the ring is cut except from the
state before an event, but for `after_removal`, a key of the leg; a state of the ring stage is
continued only under a block with `ring.enabled`. In the ring stage `chains.txt` keeps its columns
(the bubble's volume and widest point the ring's, its other keys NaN), `ring.txt` has a row per step
(the ring's volume, also in units of the bubble's at the gas cut, its section, extremes, spacing,
speed and Courant time, its fold, the limit of its approach to the axis, its distance to the open
chain, and the volume, momentum and kinetic energy filled since the gas cut), the profiles have the
ring's beside the open chain's (`_ring.txt`), `events.txt` a row per ring surgery (its fill a
negative removal, its closures in `F1`-`F6`), the totals of `budget.txt` are net of the fills, so
that the fill's own amounts make no jump in them (the projection's change of the whole liquid's
momentum and the noise of the rebuild still enter them, as at a liquid cut), the dumps hold the
record of the stage (`ring`) and `run_status.json` the events of the ring (`events.ring`: the event
of the cut, that of the removal and its time).

The block's `groove` part (off by default, written into the block and its hash only where it differs
from its defaults; its keys are keys of the class `gap`, fixed once a fill is made except from the
state before an event) makes the fill of a closing gas groove, the handler of `A-groove` (a fold of
the open chain or the bubble) and of `RF` (a fold of the ring) (`groovesurgery.groove_fill`,
`groove.py`), but never on a far head: with `pinch.far_head` a groove on the head of a tracked neck
in the far region (the downward jet's head, any head off the upward jet's core) is met by the
far-head cut of that head first, the heads being nuisances whose cut has hard volume and momentum
checks, and only a groove off the far heads is filled. The pocket the fold pair closes off (on the
ring, the side of the smaller volume) must be gas: a pocket of liquid, a thin liquid sheet between
two faces of gas that the fold test meets alike, is refused. Every part of the groove narrower than
`groove.mouth` times the fold's limit (1.5) is filled at once: each wall is walked from the pair
away from the pocket until it lies that far from the other wall or has turned out of the groove (75
degrees from its tangent at the pair), and the two nodes reached are the splices. The closure is a
cubic curve C1 with both walls at the splices, its handles `groove.cap_height` times those of a
circular arc turning by the angle between the walls' tangents (1, tried once more at 0.5 if
refused), at `groove.cut_elements` points per mouth width (null for the run's elements per scale). A
fill is refused before anything changes where the pair lies at a chain's end, where no mouth is
found within the chain, where the closure leaves the old gas or makes a knife-edge angle, where the
filled region is not simple or not inside the old gas, where the new chains or the new loop are not
a mesh's, where it would take a node of the upward jet's core (`jetcore`) or straddle the base of
that jet, and where it would take more than `groove.max_fill` of the bubble's volume at the gas cut
(0.01, the share below which the ring is removed): a fill never splits the ring or a chain, and a
larger pocket (a lobe of the ring, a bubble pinching in two) waits for a handler of its own. The
rebuild is the project's own, before the ring as in its stage, the transfer by projection onto the
old interface checked against the nodes the geometry predicts it cannot locate (on the closure,
inside the filled region), then the velocity of the new liquid (`groove.fill`: `blend`, pyoomph's
nearest-node blend, or `zero`) and the pressure-impulse projection (`groove.projection`,
`groove.restart`). Its budgets within `max(groove.box_radius delta, 2 L_fill)` of the filled
region's centroid, a disc at the site (G1-G6): the liquid volume gained against the filled volume,
the axial momentum the transfer carried against the filled region's, the identity of the disc's
cutoff in the projection's weak form and the disc's change of momentum by the projection are hard
checks, the kinetic energies and the area flags, each limit floored at a multiple of the part's
noise floors (`groove.sigma_V`, `sigma_Pz`, `sigma_KE`, 0 until set). The momenta and kinetic
energies of the disc are weighted by its cutoff, the weight of the identity (one within half the
radius, zero at the edge), so that a cell at the disc's edge whose sample a rebuild moves across a
fast free surface weighs nothing; the momentum the transfer carried without that weight is reported
beside the closures (`G2_unweighted`), never a flag. The disc's change of momentum
beyond its limit (0.05 `V_fill U`) is only a flag where the identity holds and resolves it, its
larger side at least half the unweighted disc momentum change (`G3_unweighted`) and at the
momentum floor or above: a fill joins liquid that
moved at different velocities across the gas it removes, and the projection is the inelastic impulse
of that merge, which near free surfaces moves momentum out of any disc about the site, balanced at
the container's bottom, so that the identity, not the disc's change, is the momentum balance
(`G3_merge` in the closures). A hard miss is a resumable stop (`event-unclosed`), a refusal
`groove-fill-refused`. A fill made is an event of kind `groove-fill` (`events/event_<k>.json` with
the filled region, the closure and the pocket), a row of `events.txt` (its fill a negative removal,
its closures in `F1`-`F6`) and a fill in the totals of `budget.txt`, which stay net of it. The
triple of its length scale is `gap.kappa_g` (the trigger `kappa_g delta`), run as branches from the
state before the first fill. The fold test meets a thin liquid sheet between two faces of gas as it
meets a groove; with `groove.sheets` `log` (`event` by default) the stops select the fold of each
chain and of the ring on each side apart, so that a sheet hides no groove of gas: a fold whose
pocket is liquid is no event but a row of `sheets.txt` (its thickness at the pair, its length beyond
it, its tip, area, mean thickness and volume) at most once per rebuild, and only its faces closing
below `groove.sheet_contact` element lengths of the finer side (1) is one, `sheet-contact`, without
a handler (a resumable stop).

The block's `jetcore` part (off by default, and written into the block and its hash only where it
differs from its defaults; every key a key of the leg, its sizing keys among `events.SIZING_KEYS`)
sets a resolution policy for the run after the cut that resolves the upward jet as before and meshes
and steps the rest of the free surface under simplifying assumptions, every feature kept in the
domain (`jetcore.py`). The core is the part of the open chain about the base of the upward jet, the
apex of the groove around it: its nodes within `max(core_radius L, apex_extent R_apex)` of the base
(`L = max(r_j, delta)`; `core_radius`, 2), but not above the jet's tracked liquid neck, and the
walls beyond the base within `wall_radius L` (10); without a base of a jet the whole open chain.
Every other node of the free surface is far. The step's Courant limit is split, `reconnect.courant`
on the core and `far_courant` on the far nodes (by default the same number, which leaves the step
unchanged to the bit). The far regions' targets are floored at `far_floor` (absolute, null for none;
the regions of the tracked necks exempt, so that a neck is followed to its landing); with
`far_kinks` `recentre` the guard's far kinks have a region of their own per chain, recentred on the
worst element at each request (four chords wide, no growth), at the floor, and below it only while
the element turns by more than `far_yield_turn` degrees (75), halving at each rebuild down to the
run's `h_floor` (event `far-yield`); the far regions are refined beyond `far_refine_ratio` times
their targets (null for the collapse's `refine_ratio`); with `ring_kink_floor` the ring's kinks are
no finer than its column. In the ring stage a mesh of whose ring an element turns by more than the
guard allows (`reconnect.edge_turn_stop_fresh` on a fresh mesh, `edge_turn_stop` on one a step has
run on; up to 150 degrees), or whose midside node lies beyond `midside_stop`, is not a stop at
once: the mesh is rebuilt with that element in a far
kink of its own (`kink_ring_r<n>_far`) at the size its turn asks for (`min(1/2,
edge_turn_refine/turn)` of its chord, never below `h_floor`), every other region as the mesh has
it, at most three times per state (event `ring-refine`, a yield); that kink follows its feature
(the elements that ask within its reach), keeping its size, until the feature asks no more, and
while it remains every far kink of the ring follows its own feature, a new feature a region of its
own (`kink_ring_m<n>_far`, at most eight), so that several sharp features (a sheet's rim, the
corners of a fill) are all kept refined; the base of
the upward jet must stay within `driver.HEAD_FOLD_BASE_TOL` across each such rebuild, or the run
stops (`interface-unresolved`, trigger `ring-refine-base`). A thin sheet's rim, which a rebuild at
the yielded size cannot represent, is so followed instead of ending the run, and one thinner than
the run's `h_floor` is carried at the floor (no `feature-below-floor` stop on the ring) as long as
its elements keep within the guard's limits of the turn and the midside node. A stop of the guard
on the ring at an element at the floor that the refine does not meet, where the element's apex is
a wedge of gas read at the scale of the floor (the loop turning left across it, a pair across the
wedge closer than `fold_stop_elements` element lengths, with 1.5 times its gap below the arc
between them), is met by the fill of the wedge: the groove fill of the ring under the fold limits
of the floor, its hard closures, and the base of the upward jet besides (`G_base`); an event
`floor-fill` and `floor_fills` in the run's events count them; where no wedge is, a corner of gas at
the floor open up to about a right angle (the root of the radial sheet that the downward jet's head
drives out along the plug) is filled the same way. Its check of the transfer predicts
the former gas, a few elements wide, on the old elements' quadratic curve and beyond the band of
them that pyoomph's locator still takes (`groovesurgery.LOCATOR_BAND`, 0.07 of the element's
length; the transfer's `bulk_band`). A fresh curved edge beside the downward jet's tip may be put straight without a rebuild
when its signed volume and momentum changes, their local closures and the upward-base
check satisfy hard bounds. Repeated operations are limited by a recurrence window and a
cumulative volume bound. An independent integral of the affected finite element checks
its signed volume change and its momentum after the planned velocity reset; the full local
momentum and energy changes are accounted. The sliver is also integrated on its own positive quadrature; missing
points or a remaining guard stop restore the input. Its signed correction is retained in
checkpoints and budget totals. This operation applies only near the bubble's tip, before
piercing. Local contacts at the ring's rim can also be met under the same hard fill budgets and
upward-base check. A single local crossing of the rebuild spline on a simple node polygon
is filled only if its pocket reads as gas. With the inverted-element net's step snapshot, a crossing made by a candidate step is mapped by
node identity to its simple input on the unchanged mesh; the input is restored before any
fill, and distortion requests from the rejected step are discarded. Other non-simple candidates roll back without a fill; liquid or unreadable
pockets remain stopped. Without the net snapshot this route is off. A liquid-side kink on a simple ring can instead be refined without
rupture, with calibrated volume and momentum floors, a single-loop check and the upward-base
check; failure restores the input. Filled volume and closures are recorded for each fill;
the liquid refinement records zero added and removed volume. The ring remains a simple loop.
The tip of a finger of liquid at the floor stays a stop when these guards cannot be met. A
rebuilt mesh whose curved element at the ring falls below the Jacobian bound (`curved-jacobian`)
is read as the guard's reading of the ring's element there and met in the same way: by the ring
refine above the floor, by the fill of a wedge of gas at it. With
`edge_turn_snapshots` `core` only an element of the core that bends
beyond `reconnect.edge_turn_refine` writes a profile at every step. With `prep_gap` (null for none),
once the bubble is shorter than `prep_gap delta`, its bottom and the downward jet's tip are refined
(`pierce_bottom`, `pierce_tip`, of radius `s_t`, the jet's shoulder) at `prep_rho s_t / 2.25`, so
that the ring cut admits a splice radius of `prep_rho s_t` (0.125) with a margin of 1.5 on its
check; their nodes are stepped at the core's Courant number. `profile_dt` writes a profile at least
that often after the cut (reason `window`). With `box_gate` an ordinary rebuild (the rule's or one
for distortion) may change the liquid volume and axial momentum of the box of radius
`liquid.box_radius delta` about the core's base by at most `box_gate_sigmas` (5) times the
calibrated floors `liquid.sigma_V` and `liquid.sigma_Pz`; beyond either the run stops (`box-gate`)
on the state before the rebuild, dumped as `states/gate_<step>.dump`, from which a continuation
resumes. `jetcore.txt` records per step what set the step, the split Courant limit (with the node of
each class that sets its part) and the core, and each leg the resolved keys. The block's `pinch`
part (off by default, written only where it differs from its defaults; its keys are keys of the
class `liquid`) decides when a tracked neck with a head is cut: with `enabled` only once it closes,
`r_n <= r_c delta` (`r_c`, 0.2) with a quadratic fit of `r_n(t)` over the trailing `window
r_n^(3/2)` (0.05, at least `min_samples`, 8) closing at a constant or growing rate and a time to the
pinch at that rate, `(2/3) r_n / |dr_n/dt|`, at most `kappa r_n^(3/2)` (3), read from the history of
each tracked neck that the driver keeps across its steps and in its dumps (`neck_history`), neither
`liquid.eps_d` nor `liquid.eps_max` landing a neck; with `far_head` an event without a handler (a
gas neck, a closing groove, before any fill, a sheet's contact) whose site lies on the head of a
tracked neck in the far region is met by the liquid cut of that neck (kind `far-head`, class
`F-head`), with the cut's hard closures of the volume and the momentum removed; neither the site nor
any node of the head but its neck may lie in the upward jet's core. A head whose rim curls back past
the height of its neck (the root neck, the narrowest point of the stem below it), every node beyond
that height outside the neck's radius and its half-section a simple polygon, is a head too
(`curled`, with `curl_depth` in neck radii, in its neck's record): the head is the chain from its
tip to the root neck, the curled rim included. A closing groove from the upward jet's head onto its
core (the cavity's wall or the stem beside the base) is the cut of that head at its root neck too,
the head's nodes but its neck off the core, and the cut must leave the jet's base as it was: its
radius, speed and flux read before and after on the fresh mesh within a relative 1e-3 (its height
within that times its radius), or the cut is the hard closure `F_base` and the run stops resumably.
A tracked neck that a freshly built mesh does not resolve (`liquid-neck-unresolved`) whose head is a
far head (the downward jet's, any head off the upward jet's core) is cut at that neck in the same
way, closing or not, with the same check of the base; on a head of the core, or one not single, the
stop stays.
At their defaults neither part changes a value; whether a given set of keys leaves the upward jet
unchanged is a question for replays against runs without the parts.

`postProcess/event_table.py` tabulates the events of such runs, one row per event and one per run
and tip (`<run-directory> [...] --out <events>.csv --runs <runs>.csv`): each event, made or the one
a run stopped on, with its outcome and the handler of its class, in units of the cut radius and of
`T = t_p - t_cut` (the neck at its landing, the removed head, the interval since the event before
at its tip, the jet's base before the cut and its ratio to the neck, the neck's distance from the
base, the closures, the ghost's recontacts); each run with its stop and the event it stopped on,
its events per tip and the cascade's verdicts, the drift of its liquid with and without the
removals, the removals' shares of the whole liquid's axial momentum and kinetic energy (labelled
`_liquid`: the standing wave dominates them, and the heads of jets running in opposite directions
cancel there), the removals at the row's tip, the momentum signed along its jet, against what the
base of that jet carried into it (the integrals of `Q_j V_j` and `Q_j V_j^2/2` over time, a plug
flow through the base), each head's share of the momentum and the energy in the box of its cut, the
events made with a closure flagged, and a report on the base of the jet of each tip of `--tip`
(the upward jet into the cavity by default; `--tip cavity_tip bubble_top` adds the downward one)
after the `--after-events`-th event made at that tip (`base_after_k`; the events at the other tips
do not start it), over the rows of the steps after that event's, whose own row is the state before
its cut: the slopes of `ln r_j` against `ln t'` and of `Q_j` against `t'`, `t' = t - t_p`, with the
verdict `widens` when both are positive, `narrows` when the first is negative, `stationary` when
the first and the change of `Q_j` over the window relative to its median there both lie within
their floors (`--stationary-alpha`, `--stationary-q`, 1e-3 each, recorded in the row; within them
each counts as zero, so that the round-off of a fit gives a base standing still no sign), `mixed`
otherwise, and the base against the root neck of its chain. A run continued in several legs is
given as the directories of its legs in their order joined by the path separator
(`<leg-1>:<leg-2>`) and read as one continuous run: each event's record is the latest leg's, the
records of an earlier leg beyond the events a later leg starts with are dropped, and each step has
the row a continuous run writes. The report decides nothing about the run.

The limits and floors of the run after the cut are the block's `post` keys where set, the collapse's
own otherwise, so that a study in `delta` can scale them with `delta`: `h_floor` (the smallest focus
and feature size, the floor of the interface guard and its kink regions, and the open chain's neck
stop through `neck_stop_elements h_floor`), `rise_tol` (the rise that makes a jet), `dt_min` (the
step-size floor), and for the regions of the necks `neck_floor` (the cut's focus size without it;
liquid events need it set, so that a neck can be followed below the cut radius to its landing) and
`neck_elements_per_scale` (`reconnect.elements_per_scale` without it). The fold limit after the cut,
also that of the groove gaps `chains.txt` reports, is `gap.kappa_g delta`. With
`post.open_neck_stop` false the open chain's own neck stop (`collapse.neck_stop`) is off after the
cut, as liquid events need: its liquid neck is then followed as an event. At the cut and at every
restart after it the block also sets oomph-lib's smallest step (`DTSF_minimum_dt`, 1e-12 by default)
to `post.minimum_dt`, or 1e-3 times the step-size floor without it, never above the default, and
records it in the leg, also in units of the spacing of doubles at the leg's time
(`minimum_dt_ulps`). A `minimum_dt` above the default, or above half the step-size floor after the
cut (a step failing at the floor would not be halved and would end the run as a failed one), is
refused before any output. oomph-lib gives up on a step below it, and also after an accepted step
whose next candidate, four times the step, would fall below it, so that at the steps a small cut
radius needs the default would end the run on a converged step. Before the cut it keeps the default,
on which the classification of a failure before a cut relies.

After a cut the stops of a state are evaluated once per state, and the runner's `--step-timing`
flag (no key of the case) writes `step_timing.txt`, the wall time of the parts of each step; neither
changes a value written.

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
