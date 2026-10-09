# faraday-singularity

Sharp-interface pyoomph FEM of the singular collapse of a Faraday-wave interface and the
Worthington jet it launches. The aim is to test the purely inertial self-similar theory of jet
birth, `r_j ∝ τ^α(β)`, against inertio-capillary scaling (exponent 2/3), and to resolve how
the jet speed diverges as the wave height approaches its singular value. See `README.md` for
the motivation and references.

**Status:** axisymmetric solver at `Bo = 0` with linear verification cases (free decay and the
Faraday threshold of the `J0` mode). Cavity-collapse production cases do not exist yet.

## Structure

```
faraday-singularity/
├── src/                    # reusable pyoomph code, as src/<package>/
├── simulationCases/        # entry points and pyoomph-case-v1 JSON cases
├── testCases/              # small software-contract tests
├── verificationCases/      # exact or independent solutions (e.g. linear Faraday theory)
├── validationCases/        # independent experimental or numerical data
├── modelComparisonCases/   # inertial vs inertio-capillary similarity models
├── postProcess/            # analysis and rendering
└── docs/                   # approved authored documents only
```

Runner scripts go at the repository root. Each evidence folder owns its own case files; do not
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
- The observable decides the solver. Before any production run, confirm that the solver writes
  the interface near the axis, the jet-base radius and position, the jet speed, the cone
  semi-angle β and the singular time, at the cadence needed for the intended exponent fit.
- A sharp-interface mesh cannot change topology. Stop each simulation before any pinch-off or
  entrainment, and record why it stopped.
- Quote an exponent only together with the resolution evidence that supports it.
- Run outputs, run manifests and environment receipts live outside this repository.
- Figures follow the `publication-plots` conventions.
- README, `docs/` and figure captions are public-candidate: no private paths, hosts or
  provisional findings.
