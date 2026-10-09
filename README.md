# faraday-singularity

Direct numerical simulation, with sharp-interface finite elements in
[pyoomph](https://github.com/pyoomph/pyoomph), of the singular collapse of a Faraday-wave
interface and the Worthington jet it launches.

**Status:** repository scaffold. There is no solver code yet, and the model choices listed
under [Open choices](#open-choices) are unresolved.

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

- The solver will start from the surfactant-free path of the CoMPhy `faraday-surfactant-fem`
  code: incompressible Navier–Stokes in the container frame with modulated gravity, and a free
  surface with kinematic and dynamic conditions on an arbitrary Lagrangian–Eulerian mesh.
- A sharp-interface mesh cannot change topology. The simulations cover the collapse and the
  birth of the jet, and stop before any pinch-off or bubble entrainment.

## Open choices

1. Geometry: an axisymmetric standing Faraday wave on a flat vibrated layer, whose trough
   collapses at the axis, or a wall-attached bubble carrying the parametrically excited mode.
2. Forcing, and a definition of $h$ that is comparable with the experimental one.
3. The Ohnesorge and Bond numbers, which set the viscous cut-off of the singularity and the
   role of gravity.
4. The observables written during the run: the interface near the axis, the jet-base radius
   and position, the jet speed, $\beta$ and the singular time.
5. The verification case that gates production runs.

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

Not yet pinned. pyoomph will be built from source at a fixed CoMPhy release, recorded in
`pyproject.toml` and `uv.lock`.

## References

1. B. W. Zeff, B. Kleber, J. Fineberg and D. P. Lathrop, Singularity dynamics in curvature
   collapse and jet eruption on a fluid surface, *Nature* **403**, 401–404 (2000).
   [doi:10.1038/35000151](https://doi.org/10.1038/35000151)
2. M. Cattaneo, L. Presse and O. Supponen, Faraday wave singularities trigger microbubble
   jetting, *Phys. Rev. Lett.* **135**, 204002 (2025).
   [doi:10.1103/7c4p-y9jr](https://doi.org/10.1103/7c4p-y9jr)
3. J. M. Gordillo, J. Rodríguez-Rodríguez and V. Sanjay, Self-similar Worthington jets,
   arXiv:2607.08972 (2026). [arXiv:2607.08972](https://arxiv.org/abs/2607.08972)
