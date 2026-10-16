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
  criteria before advancing and after every remesh, before writing a state dump.
- Reconnection through a neck pinch-off is made by restarts of the production runner
  (`simulationCases/run_faraday.py`; exact commands in `README.md`), never by a script of its own:
  (1) landing states, from a restart of an existing run under a derived case that adds
  `collapse.landing` (the first state with `delta/r_n >= 1.05` for each cut radius; a radius below
  1.05 times the neck-stop radius is refused before any output, `landing-missed` records one left
  without a state) or `keep_stop_state`, and a lower `neck_stop` if needed (`run_case.DERIVED_KEYS`:
  what the run keeps and where it stops, not the equations it solves); (2) one cut run per `delta`, a restart from the
  landing state of that `delta` under the landing case plus a `collapse.reconnect` block
  (`restart_mode` `cut`, `run_case.CUT_DERIVED_KEYS`); (3) continuations from any dump of the cut
  run under its case (`post-cut`). `run_case.restart_admission` decides the mode and checks it
  before output: the landing state's recorded `r_n` must put `delta/r_n` within the cut's
  `[min_landing_ratio, max_landing_ratio]` (a window that contains the landing window [1.05, 1.1]),
  and a post-cut continuation needs the reconnect settings the cut recorded. A reconnect block the
  parent did not have changes the trajectory at round-off even before a cut, so an ordinary restart
  may not add, drop or change it. The cut (`surgery.cut`): pyoomph's planner gives the topology; its
  caps are always replaced by tangent arcs spliced at the first survivor beyond the planner's window
  (`cap_window_factor`) with a radius of at least `delta`, which sets the geometry (`in_cut_radii`;
  on slender threads at `delta/r_n = 1.05` a bridge of 13-14.5 `delta` and 55-58 `delta^3` of
  removed gas), so a `delta` study needs a sensitivity to that window; the tips, the bridge and the
  focus of the cut's remesh are sized in cut radii (`tip_size`, `bridge_size`, `focus_size`,
  `focus_radius`), not by the landing run's focus; the velocity is carried across and projected by a
  pressure impulse, refused unless the projection converges and the discrete energy identity holds
  (`strong_divergence_L2` is an order-`h` diagnostic, not a test); the bubble keeps the passive
  constant gas pressure (the zero-density limit). A refused or failed cut ends the run as `failed`
  with its partial record and `failed_at` in `cut_diagnostics.json`; never load the precut state
  back into that problem, and an interruption is never turned into an outcome of the cut. After
  the cut the collapse driver runs in its post-cut mode (`reconnect` settings): the open chain is
  measured as any surface, the stops before a second change of topology apply on both chains
  (`reconnect.post_cut_stops`) and at `tau_end`, the step also follows every node of both chains
  (`reconnect.courant_limit`) and the capillary time of the sharpest tip, no Newton failure ends the
  run cleanly, and every output carries `tau = t - t_cut` and `tau_p = t - t_p` (`collapse.txt`,
  `chains.txt`, profiles of both chains, events, dumps). Loading a state written after the cut
  restores the cut's mesh sizing (`surgery.restore_cut`). One cut per run. A reconnected result
  counts only if the jets at a fixed time after the pinch time `t_p` do not depend on `delta`;
  compare the cuts of a study at one common `t_p` (each cut records its own fit and window,
  `pinch`), and report the result with that `delta` study. `simulationCases/reconnect_prototype.py`
  is a development wrapper of the same cut and driver (its own manifest schema; no figures, no
  continuation). The cut relies on private names of the pinned pyoomph release, pinned by
  `testCases/test_reconnect_fem.py`; check them when the pin moves. Planner tests on stored neck
  profiles run when `FARADAY_RECONNECT_PROFILES` names their directory; a synthetic slender thread
  covers the same planner classes by default.
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
- Run outputs, run manifests and environment receipts live outside this repository.
- Figures follow the `publication-plots` conventions.
- README, `docs/` and figure captions are public-candidate: no private paths, hosts or
  provisional findings.
