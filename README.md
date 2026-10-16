# faraday-singularity

Direct numerical simulation, with sharp-interface finite elements in
[pyoomph](https://github.com/pyoomph/pyoomph), of the singular collapse of a Faraday-wave
interface and the Worthington jet it launches.

**Status:** axisymmetric solver at zero Bond number with its linear verification cases, and a
collapse driver that records the collapse and jet observables during the run and stops before any
pinch-off. A prototype reconnection continues a run through a neck pinch-off (below); it is not
yet verified for production. No collapse case files or results are in the repository yet.

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
  the birth of the jet, and stop before any pinch-off or bubble entrainment, except through the
  reconnection prototype: at a chosen neck radius $\delta$ the gas neck is cut with pyoomph's
  axisymmetric reconnection planner, the two new tips are rebuilt as circular arcs that meet the
  axis at right angles, the liquid velocity is carried onto the new mesh and projected onto a
  divergence-free field by a pressure impulse that vanishes on the free surfaces, and the run
  continues with the bubble at the passive constant gas pressure (the limit of zero gas density).
  A result of the cut is meaningful only if it does not depend on $\delta$.
- The step size and the mesh near the axis follow the local scale of the collapse: the apex radius
  of curvature, the jet-base radius or the narrowest neck, whichever is smallest. The scale may
  grow by at most a fixed factor per step, and each rebuild of the mesh coarsens it near the axis
  by at most a fixed factor.

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
  crest), remeshes, restarts and the stop;
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

Reconnection prototype. With `collapse.keep_stop_state` a run keeps the state at its neck-radius
stop as `states/stop_<step>.dump`; with `collapse.landing`, a list of cut radii `delta`, it keeps
for each the first state whose neck radius `r_n` has fallen to `delta/1.05`, as
`states/landing_<delta>.dump`, so that one run supplies the landing states of a study in `delta`.
A cut radius that the run cannot reach before its neck-radius stop (below 1.05 times
`max(neck_stop, neck_stop_elements * h_floor)`) is refused before any output, and a radius left
without a landing state is recorded as `landing-missed` with its reason. These kept states hold the
latest samples of the closing neck, from which the pinch time `t_p` is fitted, and a restart from
one makes the remesh check of its step before stepping, as the run that wrote it did. Either key,
and a different `collapse.neck_stop`, can be added to the case of an existing run and the run
restarted from one of its dumps under the derived case (with a new `case_id` if wanted); any other
change of the case is refused, a `collapse.reconnect` block included, since compiling the
reconnection terms into the problem changes its trajectory at round-off. The restart manifest
records the parent's case hash and the keys that differ.
`simulationCases/reconnect_prototype.py` loads a landing state, checks that the run which wrote it
has a normalised manifest with a clean source commit and the same case (or one derived from it in
those keys or in its `collapse.reconnect` block), cuts the neck at `--delta` (or `--delta-ratio`
times the neck radius; the settings are those of a `collapse.reconnect` block, see
`src/faraday_singularity/reconnect.py`), projects, and takes a few steps. A cut is refused unless
`min_landing_ratio <= delta/r_n <= max_landing_ratio` (1.03 and 1.1 by default; the window must
contain the landing window 1.05 to 1.1): the cut geometry in units of `delta` depends on
`delta/r_n`. It also depends on the exclusion window of pyoomph's planner (`cap_window_factor`),
which usually sets where the new caps are spliced; the record gives the splice radii, the bridge
and the removed gas in cut radii (`in_cut_radii`), and a study in `delta` should vary that window
too. The prototype writes `cut_diagnostics.json` (the planner attempts, the replaced caps, the
volumes, the kinetic energy and axial momentum before and after the transfer and the projection,
also within 20 cut radii of the neck, the nodes the transfer could not locate in the old mesh, and
the pinch time with its fit window), the kept `precut` and `postcut` states, `steps.txt` (budgets,
the three interface tips on the axis, the bridge of liquid between the bubble and the cavity, the
jet measurements on the open surface, the element size and speed of the Courant limit, and the
times after the cut `tau` and after the pinch `tau_p`) and the profiles of both interface chains.
A refused or failed cut writes its partial record to `cut_diagnostics.json` with the stage it
reached. The cuts of a study in `delta` are compared at fixed `t - t_p` with one common `t_p`, since
`t_cut` depends on `delta` and fits of `t_p` over different windows differ. The liquid volume the
cut adds is recorded both over the whole mesh, which includes the drift of rebuilding the whole
free surface, and within a box around the cut, which is compared with the plan. The continuation
stops before a second change of topology: the bridge closing, the chains approaching each other,
the bubble's two tips meeting, a new neck narrower than `delta` on either chain, a closing fold on
the bubble, or the bubble draining away; and at the collapse driver's neck and fold stops on the
open surface. Dumps written after the cut carry its record, from which loading such a state
restores the cut's mesh sizing; continuing a run from a post-cut state is not supported yet, and
both runners refuse it.

`collapse_analysis.py` sets out, for each collapse cycle that launches a jet, the evidence for
the three similarity laws of jet birth without choosing between them: the free slopes of
`Q_j`, `We_j` and `q_j = V_j r_j` against `r_j`, with the level of `We_j`; the cone angle and the
direction and decay of the surface velocity along the cavity wall at inception, over a fixed
intermediate window; and, as a secondary check, fits of `r_j(t)` with the origin fixed at the
apex-based singular time or fitted with the exponent. Every window and threshold is written to
its summary.

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
