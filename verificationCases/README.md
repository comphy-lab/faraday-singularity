# Verification cases

These cases compare the finite-element solver with the exact linear theory of the equations it
implements: the axisymmetric `J0(r)` free-surface mode of a viscous layer in a cylinder of radius
`j_{1,1}` with free-slip walls, at `Bo = 0`. Lengths are in units of `1/k`, times in units of
`sqrt(rho/(sigma k^3))`, and `T_f = 2 pi/Omega` is the forcing period with `Omega = 2 omega0`.

## Comparators (`linear_theory/`)

- `faraday_linear.py`: the dispersion relation `D(s) + 1 + Bo = 0` of the finite-depth layer without a
  boundary-layer approximation, for a free-slip or no-slip bottom, and the Floquet (Hill) analysis
  of Kumar & Tuckerman (J. Fluid Mech. 279, 49, 1994) giving the neutral forcing `Gamma_c` and the
  Floquet exponent `mu(Gamma)`.
- `initial_value.py`: the linearised equations integrated in time from the state each simulation
  starts from, a surface `zeta = a0 J0(r)` with the liquid at rest. Its amplitude `a0 Z(t)` contains
  the viscous transient and the weakly excited bulk vortical modes that a pure normal mode omits.

The two routes are formulated differently (pressure-eliminated boundary-value problem against
primitive-variable collocation). They agree on normal modes and Floquet exponents to round-off.
`testCases/test_core.py` pins reference values and classical limits.

## Cases

All cases use `Oh = 0.04`, `Bo = 0`, depth `d = 2 pi` and the first mode (`R_c = j_{1,1}`).

| folder | cases | forcing | initial amplitude `a0` | length |
|---|---|---|---|---|
| `free-decay/` | `h` in {0.05, 0.025} x `T_f/dt` in {50, 100} | `Gamma = 0` | -0.01 | 24 `T_f` |
| `faraday-threshold/` | `Gamma` in {0.2, 0.3, 0.4} at (`h`, `T_f/dt`) = (0.05, 50) and (0.025, 100) | `Gamma cos(Omega t)` | -0.001 | 30 `T_f` |

`h` is the element size on the free surface and at the axis; elements grow away from them at rate
0.3 up to 0.4. The theory gives `Gamma_c = 0.2774` for these parameters, so the three amplitudes
give one decaying and two growing responses.

## Observables and acceptance criteria

The observable is the Hankel amplitude `a(t)` of the free surface on `J0`, written after every step.

- Free decay: the complex rate `s` from a damped-cosine fit over `t >= 4 T_f`, applied identically to
  the simulation and to `a0 Z(t)`. Accept if, on the finest case, the decay rate agrees within 1 %
  and the frequency within 0.1 %, and the differences decrease with refinement in `h` and `dt`.
- Faraday threshold: the Floquet exponent `mu` from `a(t + T_f) = -exp(mu T_f) a(t)` over
  `t >= 4 T_f`, applied identically to both histories. Accept if `|mu_sim - mu_lin| <= 2e-3` (3 % of
  the viscous decay rate `2 Oh`) for every `Gamma` on the finer setting, and if the threshold
  obtained by interpolating `mu_sim(Gamma)` to zero lies within 2 % of `Gamma_c`.
- Both: the largest pointwise difference `|a - a0 Z|/|a0|`, the volume drift and, for free decay,
  the energy-budget residual are reported alongside.

## Running

```bash
python run_case.py verificationCases/free-decay/free-decay-h0.025-n100.json --outdir <run-directory>
python verificationCases/compare_linear.py <run-directory> [...] --out <analysis-directory>
```
