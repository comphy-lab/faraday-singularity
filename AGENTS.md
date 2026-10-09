# faraday-singularity

Sharp-interface pyoomph FEM of the singular collapse of a Faraday-wave interface and the
Worthington jet it launches. The aim is to test the purely inertial self-similar theory of jet
birth, `r_j ∝ τ^α(β)`, against inertio-capillary scaling (exponent 2/3), and to resolve how
the jet speed diverges as the wave height approaches its singular value. See `README.md` for
the motivation and references.

**Status:** scaffold only. No solver code, case files or environment pin exist yet.

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

- The CoMPhy `faraday-surfactant-fem` code supplies the sharp-interface ALE Faraday solver:
  modulated gravity, free surface with kinematic and dynamic conditions, Laplace-smoothed
  mesh, and a case-file runner. Reuse its surfactant-free path.
- Copy and adapt code deliberately, recording the source commit in the commit message. Do not
  import across repositories, and do not edit `faraday-surfactant-fem` from this repository.

## Environment

- Not yet pinned. Before the first solver code, pin pyoomph with the `pyoomph-env` workflow
  and commit `pyproject.toml` and `uv.lock`. Start from the release pinned by
  `faraday-surfactant-fem` unless a later fix is required; never upgrade the pin implicitly.
- Local environment and build directories (`.venv*/`, `.pyoomph-builds/`) are ignored and
  never committed.

## Conventions

- Nondimensionalise in one parameters module. Record the dimensionless groups (Ohnesorge,
  Bond, forcing amplitude and frequency) with every run.
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
