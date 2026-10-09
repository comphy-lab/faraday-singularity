"""Dimensionless groups of the axisymmetric Faraday-collapse problem.

Scales
------
Length ``1/k``, where ``k`` is the wavenumber of the axisymmetric surface mode ``J0(k r)``; time
``t_c = sqrt(rho/(sigma k**3))``, the capillary time of that mode; velocity ``sqrt(sigma k/rho)``;
pressure ``sigma k``. In these units the mode wavenumber and the surface tension are both 1.

Dimensionless momentum balance in the frame of the vertically vibrated container::

    Du/Dt = -grad p + Oh lap u - (Bo + Gamma cos(Omega t)) e_z

with

* ``Oh = mu sqrt(k/(rho sigma))``, the Ohnesorge number built on ``1/k``;
* ``Bo = rho g/(sigma k**2)``, the gravitational Bond number (0 by default);
* ``Gamma = rho A Omega_d**2/(sigma k**2)``, the container acceleration amplitude ``A Omega_d**2``
  (``A`` the displacement amplitude, ``Omega_d`` the dimensional angular forcing frequency) scaled by
  the capillary acceleration ``sigma k**2/rho``: a Bond number built on the vibration instead of
  gravity. With gravity, the conventional ``A Omega_d**2/g`` equals ``Gamma/Bo``;
* ``Omega = Omega_d t_c``, the dimensionless forcing frequency.

Container: a cylinder of radius ``R_c = j_{1,n}`` (the n-th positive zero of ``J1``) and depth
``d = k H`` with a free-slip side wall and a 90 degree contact angle, so that ``J0(r)`` is an exact
mode of the linearised problem; the bottom is free slip. The inviscid linear frequency of the mode is
``omega0 = sqrt((1 + Bo) tanh d)`` and the forcing is applied at ``Omega = 2 omega0 * detuning``,
the centre of the first subharmonic tongue for ``detuning = 1``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from scipy.special import j0, jn_zeros


@dataclass(frozen=True)
class Fluid:
    Oh: float = 0.04
    Bo: float = 0.0


@dataclass(frozen=True)
class Container:
    mode: int = 1  # R_c = j_{1,mode}
    depth: float = 2 * math.pi  # d = k H


@dataclass(frozen=True)
class Forcing:
    Gamma: float = 0.0
    detuning: float = 1.0  # Omega / (2 omega0)


@dataclass(frozen=True)
class Initial:
    # Signed coefficient of J0(r) in the initial elevation; negative puts a trough on the axis.
    amplitude: float = -1.0e-3


@dataclass(frozen=True)
class CaseParameters:
    fluid: Fluid = field(default_factory=Fluid)
    container: Container = field(default_factory=Container)
    forcing: Forcing = field(default_factory=Forcing)
    initial: Initial = field(default_factory=Initial)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseParameters":
        return cls(
            fluid=Fluid(**data.get("fluid", {})),
            container=Container(**data.get("container", {})),
            forcing=Forcing(**data.get("forcing", {})),
            initial=Initial(**data.get("initial", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Dimensionless:
    """Derived dimensionless quantities of :class:`CaseParameters`."""

    Oh: float
    Bo: float
    Gamma: float
    radius: float  # R_c = j_{1,mode}
    depth: float  # d = k H
    omega0: float  # inviscid linear frequency of the J0 mode
    Omega: float  # forcing frequency
    a0: float  # signed J0 coefficient of the initial elevation
    J0_wall: float  # J0(R_c), the mode value at the wall

    @property
    def forcing_period(self) -> float:
        """Forcing period ``2 pi/Omega``; equals the mode period when forced at the tongue centre."""
        return 2 * math.pi / self.Omega

    @property
    def mode_period(self) -> float:
        return 2 * math.pi / self.omega0

    def reference_estimates(self) -> dict[str, float]:
        """Leading-order linear estimates, for orientation only (deep layer, small Oh).

        ``damping`` is the free-surface viscous decay rate ``2 Oh`` (Lamb); ``Gamma_c_estimate`` the
        damped-Mathieu threshold ``4 * damping * (1 + Bo)/omega0`` at the tongue centre. The exact
        linear values are computed in ``verificationCases``.
        """
        damping = 2.0 * self.Oh
        return {
            "damping": damping,
            "Gamma_c_estimate": 4.0 * damping * (1.0 + self.Bo) / self.omega0,
        }

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["forcing_period"] = self.forcing_period
        out["mode_period"] = self.mode_period
        out["reference_estimates"] = self.reference_estimates()
        return out


def nondimensionalise(p: CaseParameters) -> Dimensionless:
    fl, ct, fo = p.fluid, p.container, p.forcing
    if fl.Oh <= 0:
        raise ValueError("need Oh > 0")
    if fl.Bo < 0:
        raise ValueError("need Bo >= 0")
    if ct.mode < 1 or ct.depth <= 0:
        raise ValueError("need mode >= 1 and depth > 0")
    if fo.Gamma < 0 or fo.detuning <= 0:
        raise ValueError("need Gamma >= 0 and detuning > 0")

    radius = float(jn_zeros(1, ct.mode)[-1])
    omega0 = math.sqrt((1.0 + fl.Bo) * math.tanh(ct.depth))
    return Dimensionless(
        Oh=fl.Oh,
        Bo=fl.Bo,
        Gamma=fo.Gamma,
        radius=radius,
        depth=ct.depth,
        omega0=omega0,
        Omega=2.0 * omega0 * fo.detuning,
        a0=p.initial.amplitude,
        J0_wall=float(j0(radius)),
    )


def dimensional_reference(nd: Dimensionless, *, density: float, viscosity: float,
                          surface_tension: float) -> dict[str, float]:
    """SI equivalents of the dimensionless case for one liquid (orientation only).

    Fixing the liquid fixes ``k`` through ``Oh``: ``k = Oh**2 rho sigma/mu**2``.
    """
    k = nd.Oh**2 * density * surface_tension / viscosity**2
    t_c = math.sqrt(density / (surface_tension * k**3))
    accel = surface_tension * k**2 / density
    return {
        "k": k,
        "wavelength": 2 * math.pi / k,
        "depth": nd.depth / k,
        "container_radius": nd.radius / k,
        "time_scale": t_c,
        "velocity_scale": math.sqrt(surface_tension * k / density),
        "forcing_frequency_hz": nd.Omega / (2 * math.pi * t_c),
        "forcing_acceleration": nd.Gamma * accel,
        "forcing_displacement": nd.Gamma * accel / (nd.Omega / t_c) ** 2,
    }
