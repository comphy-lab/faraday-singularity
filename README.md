# faraday-singularity

Direct numerical simulation, with sharp-interface finite elements in
[pyoomph](https://github.com/pyoomph/pyoomph), of the singular collapse of a Faraday-wave
interface and the Worthington jet it launches.

**Status:** axisymmetric solver at zero Bond number and its linear verification cases. The
cavity-collapse cases and the jet observables are not implemented yet.

## Motivation

A gas–liquid interface that collapses onto a cone launches a thin, fast Worthington jet. On a
vibrated free surface, Faraday waves drive such collapses and jet eruptions [1]. Cattaneo,
Presse and Supponen [2] showed that wall-attached microbubbles under gentle ultrasound jet
repeatedly through the Faraday instability of the bubble surface. A conical collapse sends a
jet towards the substrate; a parabolic collapse sends a pair of opposed jets. They measured the
jet speed against Faraday-wave height and bubble size, and rationalised the collapse with
scaling laws in which both inertia and capillarity enter.

Gordillo, Rodríguez-Rodríguez and Sanjay [3] argue that the birth of a Worthington jet from a
collapsing cone is purely inertial. The local collapse follows self-similar Euler solutions
set by the cone semi-angle $\beta$. The jet-base radius therefore scales as
$r_j \propto \tau^{\alpha(\beta)}$, where $\tau$ is the time from the singular instant, and
the jet-base Weber number $We_j = r_j v_j^2$ diverges as $r_j \to 0$. For bursting bubbles,
$\alpha \simeq 0.63$, distinct from the inertio-capillary exponent $2/3$.

Because the exponents differ, the two descriptions predict different divergence of the jet
speed as the Faraday-wave height approaches its singular value. Near the singularity the jet
is thinnest and fastest, so the collapse geometry and the jet base are hardest to measure
exactly where the descriptions differ most. A clean simulation can resolve the geometry, the
jet base and the cone angle together, with known parameters.

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
- A sharp-interface mesh cannot change topology. The simulations cover the collapse and the
  birth of the jet, and stop before any pinch-off or bubble entrainment.

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
| `modelComparisonCases/` | comparisons with the inertial and inertio-capillary similarity models |
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
