"""The gas in the collapsing neck, advanced over each step of the liquid (``collapse.gas``).

The liquid's free surface bounds a duct of gas (``geometry.py``). Three zones, as in Gekle &
Gordillo (2011, Sec. 2.3) with the pocket resolved down to its bottom:

* the atmosphere above ``z_hi``, held at ``p0`` through the characteristic top condition of
  ``q1d``;
* the Euler zone ``z_lo <= z <= z_hi``, the quasi-1D solver of ``q1d`` on a grid in ``z`` that is
  fine about the neck (``cells_per_neck`` cells per neck radius within ``half_width`` radii, growing
  by ``growth`` per cell outside, capped at ``dz_max_lo`` below and ``dz_max_hi`` above, and at most
  ``dz_cap``);
* the bubble zone between the cavity's bottom on the axis and ``z_lo``, a lumped compressible gas
  (``q1d``'s ``bz``). ``z_lo`` starts at the case's ``z_lo`` or ``h_bz`` above the bottom, and rises
  to ``h_bz`` above it whenever the bottom comes within ``h_bz/2`` of it, ``h_bz`` the larger of the
  key and ``h_bz_steps`` times the bottom's rise in the last step; the cells below the new ``z_lo``
  are merged into the bubble zone (``q1d.Q1D.regrid``).

The gas is dormant (no state, no traction) until the neck has closed to ``r_on``: the neck is the
narrowest interior minimum of the chain's radius within the gas zones (as the collapse driver
measures necks), and before one exists the smallest radius within ``neck_window``. The state at
which it first has is stored (``pending``); after the next step of the liquid the gas starts there,
quasi-steadily from the pair of chains of that step (the face velocity ``-(dVcum/dt)/S``, a
Bernoulli pressure recovered to the top, attached, or not recovered above the neck, separated, an
isentropic density, the bubble zone at the bottom cell's state; ``init`` ``rest``: at rest), and is
advanced over the step at once. Every later step of the liquid (``t_n -> t_n+1``):

1. :meth:`NeckGas.before_step`, just before the liquid's step: the chain the step starts from,
   after any rebuild of the mesh since the gas last saw it, is checked to be a duct and its cell
   volumes at the gas faces replace the stored ones by the exact adiabatic wall step (the jitter of
   a rebuild's spline, ``dV_remesh``); then the traction table (:meth:`NeckGas.traction`) of the
   applied variant for the step;
2. the liquid steps with that traction frozen (staggered coupling: the gas pressure lags one step);
3. :meth:`NeckGas.after_step`, on the stepped chain (before any rebuild): the gas is advanced from
   ``t_n`` to ``t_n+1``, the cross-section and the cumulative volume at the fixed faces linear in
   time between the two chains, sub-cycled at the gas's CFL number and at ``wall_frac`` of a cell's
   volume change; then the row of each variant (:data:`COLUMNS`), the budgets, the stops, and a
   regrid when the neck has fallen below ``regrid_ratio`` of the radius the grid was made for, when
   it has drifted by more than ``drift_regrid`` radii from the grid's centre, or when ``z_lo`` must
   rise.

The traction on the free surface is ``(p - p0)/(sigma k)`` (solver units: the pressures are in
``sigma k``): the cell pressures interpolated linearly in ``z`` between the cell centres, the bubble
zone's at ``z_lo`` and below, ``p0`` at ``z_hi`` and above (zero traction), continuous across both;
positive is a gas overpressure pushing into the liquid. ``coupling`` ``two-way`` applies it,
``one-way`` computes and records it but applies none, and then advances several ``variants`` on the
one liquid run (each its own gas, the first the reference of the stops).

Stops (:attr:`NeckGas.stop`, resumable stops of the driver): ``gas-geometry`` (the chain is no
duct), ``gas-budget`` (the gas mass of a variant off its budget by more than ``mass_tol`` relative,
a regrid changing it by more than ``remap_tol``, or the energy of the Euler cells off its budget by
more than ``energy_tol``), ``gas-stall`` (with ``s_passive`` given, the closing speed
``s = -dr_n/dt`` below ``stall_ratio s_passive(r_n)`` for ``stall_steps`` steps and at least
``stall_acoustic`` acoustic times ``r_n/c`` of the throat; or ``s < 0``, a reopening neck, for
``reopen_steps`` steps), ``gas-thread`` (the chain narrower than ``thread_radius r_n`` over more
than ``thread_length r_n`` about the neck), ``gas-continuum`` (the throat's Knudsen number
``lambda/(2 r_n)``, ``lambda = lambda0 rho0/rho``, above ``knudsen_max``).

The material values default to air and water at ``k = 1.8e3 m^-1`` (water at ``Oh = 0.005``):
``rho_g/rho_l`` 1.2/998, ``p0`` 1.013e5 Pa over ``sigma k`` (0.072 N/m x 1800 /m), ``gamma`` 1.4, the
gas viscosity 1.8e-5 Pa s in ``(rho_l sigma/k)^(1/2)`` and the mean free path 68 nm in ``1/k``
(standard values); a case of another liquid, gas or wavenumber sets its own.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from . import q1d
from .geometry import ChainGeom, GasGeometryError, chain_neck, single_valued

GAS_BLOCK = "gas"  # the key of the block in a case's collapse block

# air and water at k = 1.8e3 m^-1 (module docstring), in the solver's units
_RHO_L, _SIGMA, _K = 998.0, 0.072, 1.8e3
_RHO_G, _MU_G, _P0, _LAMBDA = 1.2, 1.8e-5, 1.013e5, 68e-9
DENSITY_RATIO = _RHO_G / _RHO_L
P0 = _P0 / (_SIGMA * _K)
MU = _MU_G * math.sqrt(_K / (_RHO_L * _SIGMA))
LAMBDA0 = _LAMBDA * _K
GAMMA = 1.4

COUPLINGS = ("two-way", "one-way")
INITS = ("qs", "rest")
STATUSES = ("dormant", "pending", "active", "retired")
STOPS = ("gas-geometry", "gas-budget", "gas-stall", "gas-thread", "gas-continuum")
XI_WINDOWS = (10.0, 20.0)  # the windows |z - z_n| <= xi r_n of the extremes of a row

# gas.txt: a row per variant per step of the liquid after the gas started (and one at its start)
COLUMNS = ("step", "time", "dt", "variant", "n_sub", "dt_gas_min", "N", "z_lo", "z_hi", "z_axis",
           "z_c", "r_n", "z_n", "neck_source", "s", "s_u", "We_n", "beta_lo", "beta_hi", "V_g",
           "V_bz", "m_bz", "p_bz", "m_below", "p_pk", "p_th", "rho_th", "T_th", "u_th", "M_th",
           "mdot_th", "mdot_ratio", "Q", "p_max10", "xi_max10", "p_min10", "M_max20",
           "p_th_over_s2", "p_max10_over_s2", "Kn_th", "m_total", "mass_error", "energy_error",
           "W_liquid", "W_gas", "dV_remesh", "omega_dt", "dp_th_over_s2", "regrid", "remeshed",
           "fallback", "retries", "fric_loss", "sep_loss", "F_z_liquid", "thread_over_r_n",
           "applied")


@dataclass(frozen=True)
class GasVariant:
    """One gas of the block: attached (``separated`` false) or separated beyond the throat (the
    Borda-Carnot loss), with laminar wall friction times ``fric`` (0: none), with Maxwell slip
    (``slip``) or without."""
    name: str
    separated: bool = False
    fric: float = 0.0
    slip: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not self.name.replace(
                "-", "").replace("_", "").isalnum():
            raise ValueError("collapse.gas.variants: a name of letters, digits, - and _")
        if not isinstance(self.separated, bool) or not isinstance(self.slip, bool):
            raise ValueError("collapse.gas.variants: separated and slip are true or false")
        if not _num(self.fric) or self.fric < 0:
            raise ValueError("collapse.gas.variants: fric is a non-negative number")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass(frozen=True)
class GasSettings:
    """The ``collapse.gas`` block (module docstring). ``z_hi`` and ``neck_window`` are heights of
    the case's cavity and have no default."""
    z_hi: float
    neck_window: tuple[float, float]
    coupling: str = "two-way"
    variants: tuple[GasVariant, ...] = (GasVariant("A"),)
    density_ratio: float = DENSITY_RATIO
    p0: float = P0
    gamma: float = GAMMA
    mu: float = MU
    lambda0: float = LAMBDA0
    z_lo: float | None = None
    h_bz: float = 2.5e-3
    h_bz_steps: float = 3.0
    r_on: float = 6e-2
    init: str = "qs"
    cells_per_neck: float = 8.0
    half_width: float = 40.0
    growth: float = 1.04
    dz_max_lo: float = 2e-3
    dz_max_hi: float = 4e-3
    dz_cap: float = 2e-3
    cfl: float = 0.4
    wall_frac: float = 0.02
    regrid_ratio: float = 0.5
    drift_regrid: float | None = 10.0
    omega_dt: float = 0.1
    s_passive: tuple[float, float] | None = None
    stall_ratio: float = 0.1
    stall_steps: int = 50
    stall_acoustic: float = 10.0
    reopen_steps: int = 5
    thread_radius: float = 2.0
    thread_length: float = 30.0
    knudsen_max: float = 0.1
    mass_tol: float = 1e-12
    remap_tol: float = 1e-14
    energy_tol: float = 1e-10
    angle_window: tuple[float, float] = (12.0, 24.0)

    def __post_init__(self):
        if self.coupling not in COUPLINGS:
            raise ValueError(f"collapse.gas.coupling must be one of {COUPLINGS}")
        if self.init not in INITS:
            raise ValueError(f"collapse.gas.init must be one of {INITS}")
        if not self.variants or not all(isinstance(v, GasVariant) for v in self.variants):
            raise ValueError("collapse.gas.variants: at least one variant")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("collapse.gas.variants: the names must differ")
        if self.coupling == "two-way" and len(self.variants) != 1:
            raise ValueError("collapse.gas: two-way coupling applies one variant")
        w = self.neck_window
        if not (isinstance(w, tuple) and len(w) == 2 and all(_num(x) for x in w) and w[0] < w[1]):
            raise ValueError("collapse.gas.neck_window must be two increasing heights")
        if not _num(self.z_hi) or not self.z_hi > w[0]:
            raise ValueError("collapse.gas.z_hi must be a height above the neck window's foot")
        if self.z_lo is not None and not (_num(self.z_lo) and self.z_lo < self.z_hi):
            raise ValueError("collapse.gas.z_lo must be null or a height below z_hi")
        positive = ("density_ratio", "p0", "h_bz", "r_on", "cells_per_neck", "half_width",
                    "dz_max_lo", "dz_max_hi", "dz_cap", "cfl", "wall_frac", "omega_dt",
                    "stall_ratio", "stall_acoustic", "thread_radius", "thread_length",
                    "knudsen_max", "mass_tol", "remap_tol", "energy_tol")
        for name in positive:
            if not _num(getattr(self, name)) or not getattr(self, name) > 0:
                raise ValueError(f"collapse.gas.{name} must be positive and finite")
        for name in ("mu", "lambda0", "h_bz_steps"):
            if not _num(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"collapse.gas.{name} must be non-negative and finite")
        if not _num(self.gamma) or not self.gamma > 1.0:
            raise ValueError("collapse.gas.gamma must exceed one")
        if not _num(self.growth) or not self.growth >= 1.0:
            raise ValueError("collapse.gas.growth must be at least one")
        if not _num(self.regrid_ratio) or not 0.0 < self.regrid_ratio < 1.0:
            raise ValueError("collapse.gas.regrid_ratio must lie in (0, 1)")
        if self.drift_regrid is not None and not (_num(self.drift_regrid)
                                                  and self.drift_regrid > 0):
            raise ValueError("collapse.gas.drift_regrid must be null or positive")
        if self.s_passive is not None and not (isinstance(self.s_passive, tuple)
                                               and len(self.s_passive) == 2
                                               and all(_num(x) for x in self.s_passive)
                                               and self.s_passive[0] > 0):
            raise ValueError("collapse.gas.s_passive must be null or [coefficient, exponent]")
        for name in ("stall_steps", "reopen_steps"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"collapse.gas.{name} must be a positive integer")
        a = self.angle_window
        if not (isinstance(a, tuple) and len(a) == 2 and all(_num(x) for x in a)
                and 0 < a[0] < a[1]):
            raise ValueError("collapse.gas.angle_window must be two increasing positive numbers")

    @classmethod
    def from_dict(cls, data) -> "GasSettings":
        """The block of a case, refusing keys it does not know."""
        if not isinstance(data, dict):
            raise ValueError(f"collapse.{GAS_BLOCK} must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"collapse.{GAS_BLOCK}: unknown key(s) " + ", ".join(unknown))
        for key in ("z_hi", "neck_window"):
            if key not in data:
                raise ValueError(f"collapse.{GAS_BLOCK}.{key} is required (a height of the case)")
        kw = dict(data)
        for key in ("neck_window", "s_passive", "angle_window"):
            if kw.get(key) is not None:
                if not isinstance(kw[key], (list, tuple)):
                    raise ValueError(f"collapse.{GAS_BLOCK}.{key} must be a list of two numbers")
                kw[key] = tuple(kw[key])
        if "variants" in kw:
            vs = kw["variants"]
            if not isinstance(vs, list) or not all(isinstance(v, dict) for v in vs):
                raise ValueError(f"collapse.{GAS_BLOCK}.variants must be a list of objects")
            vknown = {f.name for f in fields(GasVariant)}
            for v in vs:
                bad = sorted(set(v) - vknown)
                if bad:
                    raise ValueError(f"collapse.{GAS_BLOCK}.variants: unknown key(s) "
                                     + ", ".join(bad))
            kw["variants"] = tuple(GasVariant(**v) for v in vs)
        return cls(**kw)

    def to_dict(self) -> dict:
        """The block as its records hold it, every key resolved."""
        out = asdict(self)
        out["variants"] = [asdict(v) for v in self.variants]
        for key in ("neck_window", "s_passive", "angle_window"):
            if out[key] is not None:
                out[key] = list(out[key])
        return out

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def two_way(self) -> bool:
        return self.coupling == "two-way"


def split_gas_block(collapse: dict) -> tuple[dict, dict | None]:
    """Separate the ``gas`` block from a ``collapse`` block: the rest and the block (None
    without)."""
    rest = dict(collapse)
    block = rest.pop(GAS_BLOCK, None)
    if block is not None and not isinstance(block, dict):
        raise ValueError(f"collapse.{GAS_BLOCK} must be an object")
    return rest, block


# --- the gas start and the rows (as the one-way check computes them) -----------------------------

def init_state(sol: q1d.Q1D, S, Vc, dV, init: str, separated: bool, z_n: float):
    """The quasi-steady incompressible start of the one-way check on the faces of ``sol``: ``u``
    the mean of the face velocities ``-(dVcum/dt)/S``, a Bernoulli pressure recovered to the top
    (attached) or held above the neck (separated), an isentropic density, the bubble zone at the
    bottom cell's pressure; at rest with ``init`` ``rest``."""
    V = np.diff(Vc)
    Vbz = Vc[0]
    if init == "rest":
        sol.init_uniform(V, Vbz)
        return
    uf = -dV / S
    u = 0.5 * (uf[1:] + uf[:-1])
    rho0, p0 = sol.rho0, sol.p0
    zc = sol.zc
    if separated:
        i = int(np.clip(np.searchsorted(zc, z_n), 0, len(zc) - 1))
        p = np.where(zc < z_n, p0 + 0.5 * rho0 * (u[i] ** 2 - u ** 2), p0)
    else:
        p = p0 + 0.5 * rho0 * (u[-1] ** 2 - u ** 2)
    r = rho0 * (p / p0) ** (1 / sol.g)
    sol.init_uniform(V, Vbz, u=u, p=p, r=r)
    pb = float(p[0])
    sol.mb = rho0 * (pb / p0) ** (1 / sol.g) * Vbz


def gas_diagnostics(sol: q1d.Q1D, geom: ChainGeom, r_n: float, z_n: float) -> dict:
    """The throat, the extremes about the neck, the pocket and the gas below the neck, as the
    one-way check's diagnostics compute them on the chain ``geom`` of the row's time."""
    r, u, p = sol.prim(sol.U)
    c = np.sqrt(sol.g * p / r)
    M = u / c
    zc, zf = sol.zc, sol.zf
    i = int(np.clip(np.searchsorted(zf, z_n) - 1, 0, sol.N - 1))
    xi = (zc - z_n) / r_n
    out = dict(r_n=r_n, z_n=z_n, p_th=p[i], rho_th=r[i], u_th=u[i], M_th=M[i],
               Scell_th=sol.V[i] / sol.dz[i], mdot_th=r[i] * u[i] * sol.V[i] / sol.dz[i])
    for w in XI_WINDOWS:
        sel = np.abs(xi) <= w
        if sel.sum() == 0:
            sel = np.array([j == i for j in range(sol.N)])
        j = np.where(sel)[0]
        out[f"pmax{int(w)}"] = float(p[j].max())
        out[f"xi_pmax{int(w)}"] = float(xi[j[np.argmax(p[j])]])
        out[f"pmin{int(w)}"] = float(p[j].min())
        out[f"Mmax{int(w)}"] = float(np.abs(M[j]).max())
    Vc_n = float(geom.at(np.array([z_n]))[1][0])
    mb = sol.mb
    rb, pb = sol.bz_state()
    part = r[i] * (Vc_n - (float(np.sum(sol.V[:i])) + sol.Vbz))
    out["m_below"] = mb + float(np.sum(r[:i] * sol.V[:i])) + part
    out["V_below"] = Vc_n
    out["p_bz"] = pb
    out["m_bz"] = mb
    deep = zc < z_n - 20 * r_n
    Vd = float(np.sum(sol.V[deep])) + sol.Vbz
    out["p_pocket"] = (float(np.sum(p[deep] * sol.V[deep])) + pb * sol.Vbz) / Vd
    out["m_total"] = mb + float(np.sum(r * sol.V))
    out["cum_top"] = sol.cum["top_mass"]
    out["cum_bottom"] = sol.cum["bottom_mass"]
    out["N"] = sol.N
    out["i_th"] = i
    return out


def handover(sol: q1d.Q1D, geom: ChainGeom, z_n: float, kappas=(1.0, 1.4)) -> dict:
    """The gas below the neck at ``z_n``, as the entrained bubble would take it: the bubble zone,
    the Euler cells below the cell of ``z_n`` and that cell's share by volume. Its mass ``m_b``,
    total and kinetic energy ``E_b``, ``KE_b``, volume ``V_b = Vcum(z_n)``, the volume-mean
    pressure ``p_pk``, the pressure of the gas mixed at rest with its energy kept,
    ``p_b_mix = (gamma - 1)(E_b - KE_b)/V_b``, and ``C_kappa = p_b_mix V_b^kappa``; the rest of the
    gas (above ``z_n``) for the budget."""
    r, u, p = sol.prim(sol.U)
    E = sol.U[2]
    i = int(np.clip(np.searchsorted(sol.zf, z_n) - 1, 0, sol.N - 1))
    V_b = float(geom.at(np.array([z_n]))[1][0])
    share = (V_b - (float(np.sum(sol.V[:i])) + sol.Vbz)) / float(sol.V[i])
    share = min(max(share, 0.0), 1.0)
    w = np.zeros(sol.N)
    w[:i] = 1.0
    w[i] = share
    rb, pb = sol.bz_state()
    Eb_zone = sol.bz_energy()
    m_b = float(sol.mb + np.sum(w * r * sol.V))
    E_b = float(Eb_zone + np.sum(w * E * sol.V))
    KE_b = float(np.sum(w * 0.5 * r * u ** 2 * sol.V))
    Vgas = float(sol.Vbz + np.sum(w * sol.V))
    p_pk = float((pb * sol.Vbz + np.sum(w * p * sol.V)) / Vgas)
    p_mix = (sol.g - 1) * (E_b - KE_b) / Vgas
    out = {"z_n": float(z_n), "cell": i, "cell_share": float(share), "m_b": m_b, "E_b": E_b,
           "KE_b": KE_b, "V_b": V_b, "V_gas_below": Vgas, "p_pk": p_pk, "p_b_mix": float(p_mix),
           "Pz_b": float(np.sum(w * r * u * sol.V)), "p_bz": float(pb), "m_bz": float(sol.mb),
           "C_kappa": {f"{k:g}": float(p_mix * Vgas ** k) for k in kappas},
           "m_above": float(np.sum((1.0 - w) * r * sol.V)),
           "E_above": float(np.sum((1.0 - w) * E * sol.V)),
           "rho0": sol.rho0, "p0": sol.p0, "gamma": sol.g}
    return out


# --- the coupler --------------------------------------------------------------------------------

@dataclass
class _Variant:
    spec: GasVariant
    sol: q1d.Q1D | None = None
    m0: float = math.nan  # the gas mass (cells and zone) at the start, for the mass budget
    E0_cells: float = math.nan  # the energy of the Euler cells at the start
    V0: float = math.nan  # the gas volume (cells and zone) at the start, for the work of the gas
    W0: float = 0.0  # the wall work (cells and zone) at the start
    W_liquid: float = 0.0  # the work of the traction on the liquid's displacement, cumulative
    dV_remesh: float = 0.0  # cell volumes absorbed at syncs, cumulative (sum of |dV|)
    p_th: float = math.nan  # the throat pressure of the last row (for the step limit)
    last_row: dict = field(default_factory=dict)


class NeckGas:
    """The gas of a ``collapse.gas`` block over a run (module docstring)."""

    def __init__(self, settings: GasSettings):
        self.s = settings
        self.status = "dormant"
        self.variants = [_Variant(v) for v in settings.variants]
        self.zf: np.ndarray | None = None
        self.z_lo = settings.z_lo if settings.z_lo is not None else math.nan
        self.z_c = self.r_grid = math.nan
        self.t = math.nan  # time of the stored chain
        self.chain: tuple | None = None  # (r, z, u_r, u_z) the gas last saw
        self.neck: tuple | None = None  # (r_n, z_n, index, source) on that chain
        self.t_on = math.nan
        self.step_on = -1
        self.prev = None  # (t, r_n) of the last row, for s
        self.slow = {"steps": 0, "t0": math.nan}
        self.reopen = 0
        self.markers = {"r_peak": math.nan, "s_peak": -math.inf, "r_stall": math.nan,
                        "r_reopen": math.nan}
        self.table: tuple | None = None  # (z, values) of the traction the liquid's step uses
        self.stop: dict | None = None
        self.regrids: list = []
        self.n_sub = 0
        self.dt_gas_min = math.inf
        self.fem_dt = math.nan
        self._synced = False

    # --- what the driver reads --------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.status == "active"

    @property
    def applied(self) -> bool:
        """Whether the traction is applied to the liquid (two-way, active)."""
        return self.s.two_way and self.active

    def dt_limit(self) -> float:
        """The step limit of the liquid, ``omega_dt/omega`` with ``omega^2 = 2 gamma p_th/r_n^2``
        (the gas spring on the liquid wall), with the traction applied; inf otherwise."""
        if not self.applied:
            return math.inf
        v = self.variants[0]
        r_n = self.neck[0] if self.neck else math.nan
        if not (math.isfinite(v.p_th) and math.isfinite(r_n) and r_n > 0):
            return math.inf
        omega = math.sqrt(2.0 * self.s.gamma * v.p_th) / r_n
        return self.s.omega_dt / omega

    def traction(self, z) -> np.ndarray:
        """The traction ``(p - p0)/(sigma k)`` at the heights ``z`` for the liquid's next step: the
        table of :meth:`before_step` (zero before the gas is active and in one-way coupling)."""
        z = np.asarray(z, dtype=float)
        if not self.applied or self.table is None:
            return np.zeros_like(z)
        tz, tv = self.table
        return np.interp(z, tz, tv, left=tv[0], right=0.0)

    def would_be_traction(self, z) -> np.ndarray:
        """The traction of the first variant at ``z`` whether applied or not (zero without one)."""
        z = np.asarray(z, dtype=float)
        if self.table is None:
            return np.zeros_like(z)
        tz, tv = self.table
        return np.interp(z, tz, tv, left=tv[0], right=0.0)

    # --- the neck and the zones ---------------------------------------------------------------

    def _neck_of(self, chain, neck=None) -> tuple:
        r, z = chain[0], chain[1]
        if neck is not None:  # given (a replay of saved frames)
            r_n, z_n = float(neck[0]), float(neck[1])
            j = int(np.argmin(np.abs(np.asarray(z) - z_n) + np.abs(np.asarray(r) - r_n)))
            return r_n, z_n, j, "given"
        lo, hi = (self.z_lo, self.s.z_hi) if math.isfinite(self.z_lo) else self.s.neck_window
        return chain_neck(r, z, lo, hi, self.s.neck_window)

    def _h_bz(self, w_axis: float, dt: float) -> float:
        rise = abs(w_axis) * dt if (math.isfinite(w_axis) and math.isfinite(dt)) else 0.0
        return max(self.s.h_bz, self.s.h_bz_steps * rise)

    def _check(self, chain, z_lo):
        bad = single_valued(chain[0], chain[1], z_lo, self.s.z_hi)
        if bad is not None:
            raise GasGeometryError(f"the free surface is no duct of the gas: {bad['reason']}", bad)

    def _faces(self, r_n: float, z_n: float, z_lo: float) -> np.ndarray:
        s = self.s
        return q1d.make_grid(z_lo, s.z_hi, z_n, min(r_n / s.cells_per_neck, s.dz_cap),
                             s.half_width * r_n, ratio=s.growth, dz_max_lo=s.dz_max_lo,
                             dz_max_hi=s.dz_max_hi)

    # --- the step of the liquid ---------------------------------------------------------------

    def before_step(self, t: float, chain, *, step: int, dt_prev: float = math.nan,
                    neck=None) -> dict:
        """Before the liquid's step from ``t`` on ``chain`` (``r, z, u_r, u_z``, axis first): the
        activation test while dormant (the state stored, ``pending``), the sync and the traction
        table while active. Raises :class:`GasGeometryError` for a chain that is no duct."""
        chain = tuple(np.asarray(x, dtype=float).copy() for x in chain)
        out = {"status": self.status}
        if self.status in ("retired", "pending"):
            return out
        if self.status == "dormant":
            nk = self._neck_of(chain, neck)
            if not (math.isfinite(nk[0]) and nk[0] <= self.s.r_on):
                return out
            # the bottom's rise in a step: its velocity on the axis (u_z of the chain's first node)
            z_lo = self.s.z_lo if self.s.z_lo is not None else \
                float(chain[1][0]) + self._h_bz(float(chain[3][0]), dt_prev)
            self._check(chain, z_lo)
            self.z_lo = z_lo
            self.status = "pending"
            self.t, self.chain, self.neck = t, chain, self._neck_of(chain, neck)
            self.t_on, self.step_on = t, step
            return dict(out, status="pending", activated=True, r_n=nk[0], z_n=nk[1], z_lo=z_lo)
        # active: the chain the step starts from
        if t != self.t:
            raise RuntimeError(f"the gas is at t = {self.t!r}, the liquid's step starts at {t!r}")
        self._check(chain, self.z_lo)
        same = all(np.array_equal(a, b) for a, b in zip(chain, self.chain))
        dV = 0.0
        if not same:
            g = ChainGeom(*chain, self.z_lo, self.s.z_hi)
            Vc = g.at(self.zf)[1]
            Vnew = np.diff(Vc)
            for v in self.variants:
                d = float(np.sum(np.abs(Vnew - v.sol.V)) + abs(float(Vc[0]) - v.sol.Vbz))
                if d > 0.0:
                    v.sol.wall_step(Vnew, float(Vc[0]))
                    v.dV_remesh += d
                dV = max(dV, d)
            self.chain = chain
            self.neck = self._neck_of(chain, neck)
        self.table = self._table(self.variants[0].sol)
        self._synced = not same
        return dict(out, synced=not same, dV_remesh=dV)

    def _table(self, sol) -> tuple:
        r, u, p = sol.prim(sol.U)
        rb, pb = sol.bz_state()
        tz = np.r_[sol.zf[0], sol.zc, sol.zf[-1]]
        tv = np.r_[pb, p, sol.p0] - sol.p0
        return tz, tv

    def after_step(self, t1: float, chain1, *, step: int, neck=None) -> list[dict]:
        """After the liquid's step to ``t1`` on the stepped ``chain1`` (before any rebuild): start
        the gas if pending, advance it, and return its rows (one per variant, and at a start the
        rows of the start too). Sets :attr:`stop` when a stop applies."""
        if self.status not in ("pending", "active"):
            return []
        chain1 = tuple(np.asarray(x, dtype=float).copy() for x in chain1)
        rows = []
        t0, chain0 = self.t, self.chain
        try:
            self._check(chain1, self.z_lo)
            g0 = ChainGeom(*chain0, self.z_lo, self.s.z_hi)
            g1 = ChainGeom(*chain1, self.z_lo, self.s.z_hi)
        except GasGeometryError as exc:
            self.stop = {**exc.info, "trigger": exc.info.get("reason"), "reason": "gas-geometry",
                         "time": t1}
            return rows
        neck1 = self._neck_of(chain1, neck)
        if self.status == "pending":
            rows += self._start(t0, t1, g0, g1, step - 1)
        self.fem_dt = t1 - t0
        S0, V0, _, _ = g0.at(self.zf)
        S1, V1, R1, _ = g1.at(self.zf)
        rate = np.abs(np.diff(V1) - np.diff(V0)) / (t1 - t0)
        dt_wall = self.s.wall_frac * float(np.min(np.diff(V0) / np.maximum(rate, 1e-300)))

        def wall(t):
            if t == t1:
                return S1, V1, (V1 - V0) / (t1 - t0)
            x = (t - t0) / (t1 - t0)
            return S0 + x * (S1 - S0), V0 + x * (V1 - V0), (V1 - V0) / (t1 - t0)

        self.n_sub, self.dt_gas_min = 0, math.inf
        before = {}
        for v in self.variants:
            sol = v.sol
            before[v.spec.name] = (sol.V.copy(), sol.Vbz)
            t = t0
            while t < t1 - 1e-18:
                dt = min(sol.max_dt(), dt_wall, t1 - t)
                if t1 - t - dt < 1e-3 * dt:
                    dt = t1 - t
                try:
                    tn = sol.substep(t, dt, wall)
                except q1d.StepFailed as exc:
                    self.stop = {"reason": "gas-budget", "trigger": "gas-step-failed",
                                 "variant": v.spec.name, "error": str(exc), "time": t}
                    return rows
                self.dt_gas_min = min(self.dt_gas_min, tn - t)
                self.n_sub += 1
                t = tn
        rise = float(chain1[1][0] - chain0[1][0])  # the bottom's rise over the step
        self.t, self.chain, self.neck = t1, chain1, neck1
        rows += self._rows(t1, g1, neck1, step, before)
        self._synced = False
        self._stops(rows, g1, neck1)
        self._maybe_regrid(t1, g1, neck1, chain1, rise)
        return rows

    def _start(self, t0, t1, g0, g1, step) -> list[dict]:
        r_n, z_n = self.neck[0], self.neck[1]
        self.zf = self._faces(r_n, z_n, self.z_lo)
        S0, V0, _, _ = g0.at(self.zf)
        S1, V1, _, _ = g1.at(self.zf)
        x = (t0 - t0) / (t1 - t0)
        S, Vc, dV = S0 + x * (S1 - S0), V0 + x * (V1 - V0), (V1 - V0) / (t1 - t0)
        s = self.s
        for v in self.variants:
            sol = q1d.Q1D(self.zf, s.gamma, s.p0, s.density_ratio, mu=s.mu,
                          lam0=s.lambda0 if v.spec.slip else 0.0, fric=v.spec.fric,
                          separated=v.spec.separated, bc_bottom="bz", bc_top="p0", cfl=s.cfl)
            init_state(sol, S, Vc, dV, s.init, v.spec.separated, z_n)
            v.sol = sol
            tot = sol.totals()
            v.m0 = tot["mass_cells"] + tot["mass_bz"]
            v.E0_cells = tot["energy_cells"]
            v.V0 = tot["volume_cells"] + tot["volume_bz"]
        self.status = "active"
        self.r_grid, self.z_c = r_n, z_n
        self.fem_dt = math.nan
        return self._rows(t0, g0, self.neck, step, None)

    # --- rows, stops and regrids --------------------------------------------------------------

    def _rows(self, t, g, neck, step, before) -> list[dict]:
        s = self.s
        r_n, z_n, j, source = neck
        prev = self.prev
        # the closing speed between rows of the same neck: undefined where the neck the gas follows
        # changes kind (the window's smallest radius before a neck forms, then the neck itself,
        # which can be wider than the window's foot)
        same = prev is not None and (len(prev) < 3 or prev[2] == source)
        speed = -(r_n - prev[1]) / (t - prev[0]) if (same and t > prev[0]) else math.nan
        if before is not None or prev is None:
            self.prev = (t, r_n, source)
        z_ch, ur_ch = self.chain[1], self.chain[2]
        s_u = -float(ur_ch[j]) if 0 <= j < len(ur_ch) else math.nan
        lo, hi = s.angle_window
        rr = g.radius([z_n - hi * r_n, z_n - lo * r_n, z_n + lo * r_n, z_n + hi * r_n])
        beta_lo = math.degrees(math.atan((rr[0] - rr[1]) / ((hi - lo) * r_n)))
        beta_hi = math.degrees(math.atan((rr[3] - rr[2]) / ((hi - lo) * r_n)))
        Q = -float(g.at(np.array([z_n]))[2][0])
        thread = self._thread_length(r_n, j) / r_n
        rows = []
        for k, v in enumerate(self.variants):
            sol = v.sol
            d = gas_diagnostics(sol, g, r_n, z_n)
            tot = sol.totals()
            c = sol.cum
            m_total = tot["mass_cells"] + tot["mass_bz"]
            mass_error = (m_total + c["top_mass"] - v.m0) / v.m0
            dE = tot["energy_cells"] - v.E0_cells
            energy_error = (dE - (c["bottom_energy"] - c["top_energy"] + c["wall_work"]
                                  - c["merged_energy"])) / v.E0_cells
            W_gas = (c["wall_work"] + c["bz_wall_work"]) + s.p0 * (
                tot["volume_cells"] + tot["volume_bz"] - v.V0)
            if before is not None and self.table is not None:
                Vb, Vbz_b = before[v.spec.name]
                tz, tv = self.table
                T_c = np.interp(sol.zc, tz, tv, left=tv[0], right=0.0)
                v.W_liquid += -float(np.sum(T_c * (sol.V - Vb))) - float(tv[0]) * (sol.Vbz - Vbz_b)
            p_th = float(d["p_th"])
            dp = abs(p_th - v.p_th) if math.isfinite(v.p_th) else math.nan
            v.p_th = p_th
            c_th = math.sqrt(sol.g * p_th / d["rho_th"])
            omega = math.sqrt(2.0 * sol.g * p_th) / r_n
            s2 = speed ** 2 if math.isfinite(speed) else math.nan
            rho_th = float(d["rho_th"])
            mdot_ratio = float(d["mdot_th"]) / (sol.rho0 * Q) if Q != 0 else math.nan
            # the axial force of the gas's overpressure on the liquid: the bubble zone on the
            # pocket's bottom below z_lo, each cell on its wall (downwards where the duct widens)
            p_cells = sol.prim(sol.U)[2]
            Sf = g.at(sol.zf)[0]
            F_z = -float((d["p_bz"] - sol.p0) * Sf[0]) - float(np.sum((p_cells - sol.p0)
                                                                         * np.diff(Sf)))
            row = {"step": step, "time": t, "dt": self.fem_dt if before is not None else 0.0,
                   "variant": v.spec.name, "n_sub": self.n_sub if before is not None else 0,
                   "dt_gas_min": self.dt_gas_min if before is not None else math.nan,
                   "N": sol.N, "z_lo": float(sol.zf[0]), "z_hi": float(sol.zf[-1]),
                   "z_axis": float(z_ch[0]), "z_c": self.z_c, "r_n": r_n, "z_n": z_n,
                   "neck_source": source, "s": speed, "s_u": s_u,
                   "We_n": s2 * r_n, "beta_lo": beta_lo, "beta_hi": beta_hi,
                   "V_g": d["V_below"], "V_bz": float(sol.Vbz), "m_bz": d["m_bz"],
                   "p_bz": float(d["p_bz"]), "m_below": d["m_below"], "p_pk": d["p_pocket"],
                   "p_th": p_th, "rho_th": rho_th,
                   "T_th": (p_th / rho_th) / (sol.p0 / sol.rho0), "u_th": float(d["u_th"]),
                   "M_th": float(d["M_th"]), "mdot_th": float(d["mdot_th"]),
                   "mdot_ratio": mdot_ratio, "Q": Q, "p_max10": d["pmax10"],
                   "xi_max10": d["xi_pmax10"], "p_min10": d["pmin10"], "M_max20": d["Mmax20"],
                   "p_th_over_s2": (p_th - sol.p0) / s2, "p_max10_over_s2": (d["pmax10"]
                                                                               - sol.p0) / s2,
                   "Kn_th": s.lambda0 * sol.rho0 / rho_th / (2.0 * r_n),
                   "m_total": m_total, "mass_error": mass_error, "energy_error": energy_error,
                   "W_liquid": v.W_liquid, "W_gas": W_gas, "dV_remesh": v.dV_remesh,
                   "omega_dt": omega * self.fem_dt if before is not None else math.nan,
                   "dp_th_over_s2": dp / s2, "regrid": 0,
                   "remeshed": int(bool(getattr(self, "_synced", False)) and before is not None),
                   "fallback": int(c["fallback"]), "retries": int(c["retries"]),
                   "fric_loss": float(c["fric_loss"]), "sep_loss": float(c["sep_loss"]),
                   "F_z_liquid": F_z, "thread_over_r_n": thread,
                   "applied": int(self.s.two_way and k == 0),
                   "_c_th": c_th, "_diag": d}
            v.last_row = row
            rows.append(row)
        return rows

    def _stops(self, rows, g, neck):
        s = self.s
        r_n, z_n, j, _ = neck
        row = rows[-len(self.variants)]  # the first variant's row of this step
        speed = row["s"]
        for r in rows[-len(self.variants):]:
            if abs(r["mass_error"]) > s.mass_tol:
                self.stop = {"reason": "gas-budget", "trigger": "mass", "variant": r["variant"],
                             "mass_error": r["mass_error"], "mass_tol": s.mass_tol,
                             "time": r["time"]}
                return
            if abs(r["energy_error"]) > s.energy_tol:
                self.stop = {"reason": "gas-budget", "trigger": "energy",
                             "variant": r["variant"], "energy_error": r["energy_error"],
                             "energy_tol": s.energy_tol, "time": r["time"]}
                return
        if math.isfinite(speed):
            if speed > self.markers["s_peak"]:
                self.markers.update(s_peak=speed, r_peak=r_n)
            if speed < 0:
                self.reopen += 1
                if not math.isfinite(self.markers["r_reopen"]):
                    self.markers["r_reopen"] = r_n
            else:
                self.reopen = 0
            if s.s_passive is not None:
                passive = s.s_passive[0] * r_n ** s.s_passive[1]
                if speed <= s.stall_ratio * passive:
                    if self.slow["steps"] == 0:
                        self.slow["t0"] = row["time"]
                    self.slow["steps"] += 1
                    if not math.isfinite(self.markers["r_stall"]):
                        self.markers["r_stall"] = r_n
                else:
                    self.slow = {"steps": 0, "t0": math.nan}
        tau_ac = r_n / row["_c_th"]
        if self.slow["steps"] >= s.stall_steps and row["time"] - self.slow["t0"] >= \
                s.stall_acoustic * tau_ac:
            self.stop = {"reason": "gas-stall", "trigger": "slow", "r_n": r_n, "s": speed,
                         "steps": self.slow["steps"], "since": self.slow["t0"],
                         "tau_ac": tau_ac, **self.markers, "time": row["time"]}
            return
        if self.reopen >= s.reopen_steps:
            self.stop = {"reason": "gas-stall", "trigger": "reopen", "r_n": r_n, "s": speed,
                         "steps": self.reopen, **self.markers, "time": row["time"]}
            return
        length = row["thread_over_r_n"] * r_n
        if length > s.thread_length * r_n:
            self.stop = {"reason": "gas-thread", "r_n": r_n, "length": length,
                         "length_over_r_n": length / r_n, "time": row["time"]}
            return
        if row["Kn_th"] > s.knudsen_max:
            self.stop = {"reason": "gas-continuum", "Kn_th": row["Kn_th"], "r_n": r_n,
                         "knudsen_max": s.knudsen_max, "time": row["time"]}

    def _thread_length(self, r_n, j) -> float:
        """The height over which the chain about the neck node ``j`` stays narrower than
        ``thread_radius r_n``."""
        r, z = self.chain[0], self.chain[1]
        if not 0 <= j < len(r):
            return 0.0
        lim = self.s.thread_radius * r_n
        a = b = j
        while a > 0 and r[a - 1] < lim:
            a -= 1
        while b < len(r) - 1 and r[b + 1] < lim:
            b += 1
        return float(abs(z[b] - z[a]))

    def _maybe_regrid(self, t, g, neck, chain, rise):
        """A regrid when the neck has fallen below ``regrid_ratio`` of the grid's radius, has
        drifted ``drift_regrid`` radii from its centre, or when the bottom has come within
        ``h_bz/2`` of ``z_lo`` (``z_lo`` then rises to ``h_bz`` above it, ``h_bz`` at least
        ``h_bz_steps`` times the bottom's ``rise`` in the step); a regrid changing the gas mass of
        a variant by more than ``remap_tol`` is a ``gas-budget`` stop."""
        s = self.s
        r_n, z_n = neck[0], neck[1]
        reasons = []
        if r_n < s.regrid_ratio * self.r_grid:
            reasons.append("neck")
        if s.drift_regrid is not None and abs(z_n - self.z_c) > s.drift_regrid * r_n:
            reasons.append("drift")
        z_axis = float(chain[1][0])
        h_bz = max(s.h_bz, s.h_bz_steps * abs(rise))
        z_lo = self.z_lo
        if self.z_lo - z_axis < 0.5 * h_bz:
            z_lo = z_axis + h_bz
            reasons.append("z_lo")
        if not reasons:
            return
        zf_new = self._faces(r_n, z_n, z_lo)
        for v in self.variants:
            rec = v.sol.regrid(zf_new, lambda zq: g.at(np.asarray(zq))[1])
            b, a = rec["before"], rec["after"]
            m0 = b["mass_cells"] + b["mass_bz"]
            dm = (a["mass_cells"] + a["mass_bz"]) - m0
            E0 = b["energy_cells"] + b["energy_bz"]
            dE = (a["energy_cells"] + a["energy_bz"]) - E0
            if self.stop is None and abs(dm) > s.remap_tol * m0:
                self.stop = {"reason": "gas-budget", "trigger": "remap", "variant": v.spec.name,
                             "remap_mass_error": dm / m0, "remap_energy_error": dE / E0,
                             "remap_tol": s.remap_tol, "time": t}
            if v.last_row:
                v.last_row.update(regrid=1, remap_mass_error=dm / m0, remap_energy_error=dE / E0)
        self.regrids.append({"time": t, "reasons": reasons, "r_n": r_n, "z_n": z_n,
                             "z_lo": z_lo, "N": len(zf_new) - 1})
        self.zf, self.z_lo, self.r_grid, self.z_c = zf_new, z_lo, r_n, z_n

    # --- the cut, the profiles and the dumps ----------------------------------------------------

    def retire(self):
        """At a cut: the gas stops (no traction, no advance); its state is kept for the record."""
        self.status = "retired"
        self.table = None

    def handover(self, z_n: float | None = None) -> dict:
        """The partition of each variant's gas at the neck ``z_n`` (the last neck by default) on
        the chain the gas last saw (:func:`handover`), by variant; empty unless the gas is
        active."""
        if self.status != "active":
            return {}
        z_n = self.neck[1] if z_n is None else float(z_n)
        g = ChainGeom(*self.chain, self.z_lo, self.s.z_hi)
        return {v.spec.name: handover(v.sol, g, z_n) for v in self.variants}

    def profiles(self) -> dict:
        """The gas of each variant on its cells: ``zc``, ``S``, ``rho``, ``u``, ``p``."""
        out = {}
        for v in self.variants:
            if v.sol is None:
                continue
            r, u, p = v.sol.prim(v.sol.U)
            out[v.spec.name] = {"zc": v.sol.zc.copy(), "S": v.sol.V / v.sol.dz, "rho": r,
                                "u": u, "p": p}
        return out

    _SCALARS = ("status", "z_lo", "z_c", "r_grid", "t", "t_on", "step_on", "reopen", "n_sub",
                "dt_gas_min", "fem_dt", "_synced")
    _VSCALARS = ("m0", "E0_cells", "V0", "W0", "W_liquid", "dV_remesh", "p_th")

    def state_dict(self) -> tuple[dict, dict]:
        """``(arrays, meta)``: the whole state, the arrays for an ``.npz`` and the rest JSON, from
        which :meth:`load_state_dict` continues to the bit."""
        arrays, meta = {}, {k: getattr(self, k, None) for k in self._SCALARS}
        meta.update(neck=None if self.neck is None else list(self.neck),
                    prev=None if self.prev is None else list(self.prev), slow=dict(self.slow),
                    markers=dict(self.markers), stop=self.stop, regrids=list(self.regrids),
                    settings_sha256=self.s.sha256(), variants=[])
        if self.zf is not None:
            arrays["zf"] = self.zf
        if self.chain is not None:
            for k, x in zip(("r", "z", "ur", "uz"), self.chain):
                arrays[f"chain_{k}"] = x
        if self.table is not None:
            arrays["table_z"], arrays["table_v"] = self.table
        for i, v in enumerate(self.variants):
            vm = {k: getattr(v, k) for k in self._VSCALARS}
            vm["name"] = v.spec.name
            vm["last_row"] = {k: x for k, x in v.last_row.items() if not k.startswith("_")}
            if v.sol is not None:
                sd = v.sol.state_dict()
                for k in ("zf", "U", "V"):
                    arrays[f"v{i}_{k}"] = sd.pop(k)
                vm["sol"] = sd
            meta["variants"].append(vm)
        return {k: np.ascontiguousarray(x) for k, x in arrays.items()}, json.loads(
            json.dumps(meta, default=_plain_json))

    def load_state_dict(self, arrays, meta: dict):
        if meta.get("settings_sha256") != self.s.sha256():
            raise ValueError("the gas state was written under another collapse.gas block")
        for k in self._SCALARS:
            setattr(self, k, meta.get(k))
        self.neck = None if meta["neck"] is None else (float(meta["neck"][0]),
                                                       float(meta["neck"][1]),
                                                       int(meta["neck"][2]), str(meta["neck"][3]))
        self.prev = None if meta["prev"] is None else (float(meta["prev"][0]),
                                                       float(meta["prev"][1]),
                                                       *[str(x) for x in meta["prev"][2:3]])
        self.slow, self.markers = dict(meta["slow"]), dict(meta["markers"])
        self.stop, self.regrids = meta["stop"], list(meta["regrids"])
        self.zf = np.array(arrays["zf"], dtype=float) if "zf" in arrays else None
        self.chain = tuple(np.array(arrays[f"chain_{k}"], dtype=float)
                           for k in ("r", "z", "ur", "uz")) if "chain_r" in arrays else None
        self.table = (np.array(arrays["table_z"], dtype=float),
                      np.array(arrays["table_v"], dtype=float)) if "table_z" in arrays else None
        names = [vm["name"] for vm in meta["variants"]]
        if names != [v.spec.name for v in self.variants]:
            raise ValueError("the gas state holds other variants than the block")
        for i, (v, vm) in enumerate(zip(self.variants, meta["variants"])):
            for k in self._VSCALARS:
                setattr(v, k, vm[k])
            v.last_row = dict(vm.get("last_row") or {})
            if "sol" in vm:
                s = self.s
                sol = q1d.Q1D(np.array(arrays[f"v{i}_zf"], dtype=float), s.gamma, s.p0,
                              s.density_ratio)
                sd = dict(vm["sol"])
                sd.update(zf=arrays[f"v{i}_zf"], U=arrays[f"v{i}_U"], V=arrays[f"v{i}_V"])
                sol.load_state_dict(sd)
                v.sol = sol
            else:
                v.sol = None


GAS_FILE_SUFFIX = ".gas.npz"  # the gas's side file of a dump, beside it (<dump>.gas.npz)


def write_state_file(path, arrays: dict, meta: dict):
    """The gas's state at ``path``: an uncompressed ``.npz`` of ``arrays`` and the JSON ``meta``
    (``meta.json``), with fixed member dates, so that the same state gives the same bytes."""
    import zipfile

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.ascontiguousarray(arrays[name]),
                                          allow_pickle=False)
        info = zipfile.ZipInfo("meta.json", date_time=(1980, 1, 1, 0, 0, 0))
        zf.writestr(info, json.dumps(meta, sort_keys=True, default=_plain_json))


def read_state_file(path) -> tuple[dict, dict]:
    """``(arrays, meta)`` of a file of :func:`write_state_file`."""
    import zipfile

    arrays = {}
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        for name in zf.namelist():
            if name.endswith(".npy"):
                with zf.open(name) as fh:
                    arrays[name[:-4]] = np.lib.format.read_array(fh, allow_pickle=False)
    return arrays, meta


def chain_sha256(chain) -> str:
    """The hash of a chain's ``r, z, u_r, u_z`` (float64), to check that a restart restored the
    free surface a dump was written on."""
    h = hashlib.sha256()
    for x in chain:
        h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes())
    return h.hexdigest()


def _plain_json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON: {type(x).__name__}")
