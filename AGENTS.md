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
and trough episodes of the axis) and stops before any pinch-off. A reconnection prototype
(`src/faraday_singularity/surgery.py`, `simulationCases/reconnect_prototype.py`) can continue a run
through a neck pinch-off by an explicit cut; it is not yet verified for production. The repository
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
  through a neck pinch-off: a run keeps its landing state at the neck-radius stop
  (`collapse.keep_stop_state`), and the cut is made from it at a chosen radius `delta`
  (`collapse.reconnect`, `reconnect.py`, `surgery.py`): pyoomph's planner gives the topology, its
  caps are always replaced by tangent arcs, the liquid velocity is carried across and projected
  onto a divergence-free field by a pressure impulse (refused unless the discrete energy identity
  holds), and the bubble keeps the passive constant gas pressure (the zero-density limit). A
  reconnected result counts only if the jets at a fixed time after the cut do not depend on
  `delta`; report it with that `delta` study. One cut per run; stop before any second change of
  topology (the stops in `reconnect.post_cut_stops`). The cut relies on private names of the
  pinned pyoomph release, pinned by `testCases/test_reconnect_fem.py`; check them when the pin
  moves. Planner tests on stored neck profiles run when `FARADAY_RECONNECT_PROFILES` names their
  directory. Of failures of the solve, only oomph-lib giving up below
  its smallest step (the Newton solve failing at every step size) can end a run cleanly, and only
  when `classify_failure` attributes it to a closing neck or a vanishing local scale, as measured
  rather than growth-limited; any other failure, including a failed remesh, is a failed run. Runs also end cleanly at the neck-radius
  stop, the conservative fold-gap stop (`self-contact-approach`), the end time, the step-size
  floor and the step limit, and, only when a case sets a finite `post_birth_aspect`, once the born
  jet is unambiguous (`jet-confirmed`); such a run records jet birth but not jet formation. A fold-gap stop can also truncate a static sharp corner; it is not
  proof of entrapment or a singularity. `InterfaceSelfIntersection` is a failed run. Check stop
  criteria before advancing and after every remesh, before writing a state dump.
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
