"""Quasi-one-dimensional unsteady compressible gas flow in a duct with moving walls.

Equations (Gekle & Gordillo 2011, arXiv:1001.5402, Eqs. 13-15; dimensional form, rho_g, u, p):
    d(rho S)/dt + d(rho u S)/dz = 0
    d(rho u S)/dt + d((rho u^2 + p) S)/dz = p dS/dz  [+ friction] [+ separation loss]
    d(E S)/dt + d((E + p) u S)/dz = -p dS/dt|_z,        E = p/(gamma-1) + rho u^2/2
S(z, t) = pi r(z, t)^2 is prescribed (one-way coupling). Ideal gas, constant gamma.

Discretisation (written for this check, not taken from the paper; G&G used a Roe scheme):
  * fixed (between regrids) non-uniform finite-volume grid; cell volume V_i = int S dz from the
    exact piecewise-linear profile; face areas S_{i+1/2};
  * Strang splitting per step: wall half-step, flux step, wall half-step. The wall step is the
    exact solution of the local ODEs (rho S, rho u S const; p S^gamma const): it reproduces the
    adiabatic compression by the walls exactly for any rate;
  * flux step: HLLC Riemann solver, MUSCL reconstruction of (rho, u, p) with a minmod limiter on
    the non-uniform grid, SSP-RK2 in time, geometry frozen at the mid-step; the area source
    p_i (S_{i+1/2} - S_{i-1/2}) balances the pressure flux exactly for a gas at rest
    (well balanced);
  * optional friction (fully developed laminar duct flow with Maxwell slip,
    tau_w = 4 mu u/(r + 4 lambda), lambda = lambda0 rho0/rho), integrated exactly; the lost kinetic
    energy stays in the gas (stationary-wall Fanno flow);
  * optional separation limit: a distributed Borda-Carnot loss -rho u|u| |dS/dz| applied where
    the flow runs towards increasing area (u dS/dz > 0). For incompressible steady flow it makes
    dp/dz = 0 in the diverging part (no pressure recovery), i.e. a separated jet at the throat
    pressure discharging into dead gas; it does not touch continuity or the wall work, so the
    squeeze of the gas by the walls is the same in both limits.
Boundary conditions:
  * top ('p0'): Gekle & Gordillo Eqs. 19-22 (ghost cell at p0, isentropic, I+ invariant
    conserved); supersonic outflow extrapolates;
  * bottom ('bz'): lumped bubble zone below z_lo (uniform pressure, isentropic, mass tracked;
    volume V_bz(t) from the profiles): the ghost state is the bubble-zone gas (rho_b, p_b) moving
    with the displacement velocity u_e = -(dV_bz/dt)/S(z_lo); ('gg'): Gekle & Gordillo Eqs. 23-26
    (imposed u_e with the I- invariant; the bubble zone is incompressible at the pressure of the
    bottom cell); ('wall'): reflective.

Provenance and additions. The solver above is the one-way quasi-1D solver of the a-posteriori neck
gas check, copied verbatim (source sha256
643e827ff5b46b409af1b5358bdc45a82a3457e0c098f1b6f48f0bfb089e71c9); none of its arithmetic is changed,
so its verification values are reproduced to the bit. Added for the in-loop coupling
(``coupler.py``), none of which changes a value the solver computes:

  * budgets in ``cum``: the energy fluxes through the top and the bottom face (``top_energy``,
    ``bottom_energy``, positive upwards, as ``top_mass`` and ``bottom_mass``), the work of the walls
    on the Euler cells and on the bubble zone (``wall_work``, ``bz_wall_work``: the change of their
    total energy in the exact wall steps), the kinetic energy turned into heat by friction and by
    the separation loss (``fric_loss``, ``sep_loss``), the steps retried (``retries``) and what the
    bubble zone took in when ``z_lo`` rose (``merged_mass``, ``merged_energy``, and the axial
    momentum handed to the wall, ``merged_momentum``); :meth:`Q1D.totals` reads the inventories;
  * :meth:`Q1D.substep`, one Strang step on a wall motion given as a function of time, with the
    retry of the one-way driver: a failed second-order step is retaken at first order, then halved,
    up to six attempts;
  * :meth:`Q1D.state_dict` and :meth:`Q1D.load_state_dict`, the whole state as arrays (bitwise);
  * :meth:`Q1D.regrid`, the conservative remap of :meth:`Q1D.remap` onto new faces whose lowest face
    ``z_lo`` may have risen: the gas below the new ``z_lo`` is merged into the bubble zone (its mass
    and total energy added; its axial momentum, which a lumped zone at rest cannot hold, handed to
    the wall and counted).
"""
from __future__ import annotations

import copy
import math

import numpy as np


# ------------------------------------------------------------------ Riemann solver
def hllc(rL, uL, pL, rR, uR, pR, g):
    cL = np.sqrt(g * pL / rL)
    cR = np.sqrt(g * pR / rR)
    EL = pL / (g - 1) + 0.5 * rL * uL ** 2
    ER = pR / (g - 1) + 0.5 * rR * uR ** 2
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    den = rL * (SL - uL) - rR * (SR - uR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / den
    FL = np.array([rL * uL, rL * uL ** 2 + pL, uL * (EL + pL)])
    FR = np.array([rR * uR, rR * uR ** 2 + pR, uR * (ER + pR)])
    UL = np.array([rL, rL * uL, EL])
    UR = np.array([rR, rR * uR, ER])
    fL = rL * (SL - uL) / (SL - Ss)
    fR = rR * (SR - uR) / (SR - Ss)
    UsL = fL * np.array([np.ones_like(Ss), Ss, EL / rL + (Ss - uL) * (Ss + pL / (rL * (SL - uL)))])
    UsR = fR * np.array([np.ones_like(Ss), Ss, ER / rR + (Ss - uR) * (Ss + pR / (rR * (SR - uR)))])
    FsL = FL + SL * (UsL - UL)
    FsR = FR + SR * (UsR - UR)
    F = np.where(SL >= 0, FL, np.where(Ss >= 0, FsL, np.where(SR > 0, FsR, FR)))
    return F


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


# ------------------------------------------------------------------ grid
def make_grid(z_lo, z_hi, z_c, dz_min, half_width, ratio=1.04, dz_max_lo=2e-3, dz_max_hi=4e-3):
    """Faces of a grid that is uniform (dz_min) within z_c +- half_width and grows geometrically
    (ratio) outside, capped at dz_max_lo below and dz_max_hi above, ending exactly at z_lo, z_hi."""
    dz_max_lo = max(dz_max_lo, dz_min)
    dz_max_hi = max(dz_max_hi, dz_min)
    a = max(z_lo, z_c - half_width)
    b = min(z_hi, z_c + half_width)
    n = max(1, int(math.ceil((b - a) / dz_min)))
    core = list(np.linspace(a, b, n + 1))
    up, dz, z = [], dz_min, b
    while z < z_hi - 1e-14:
        dz = min(dz * ratio, dz_max_hi)
        z = z + dz
        up.append(min(z, z_hi))
    if len(up) >= 2 and up[-1] - up[-2] < 0.5 * (up[-2] - (up[-3] if len(up) > 2 else b)):
        up.pop(-2)
    dn, dz, z = [], dz_min, a
    while z > z_lo + 1e-14:
        dz = min(dz * ratio, dz_max_lo)
        z = z - dz
        dn.append(max(z, z_lo))
    if len(dn) >= 2 and dn[-2] - dn[-1] < 0.5 * ((dn[-3] if len(dn) > 2 else a) - dn[-2]):
        dn.pop(-2)
    zf = np.array(sorted(dn) + core + up)
    zf[0], zf[-1] = z_lo, z_hi
    return np.unique(zf)


# ------------------------------------------------------------------ solver
# the budgets added to ``Q1D.cum`` (module docstring), all zero at the start
BUDGET_KEYS = dict(top_energy=0.0, bottom_energy=0.0, wall_work=0.0, bz_wall_work=0.0, retries=0,
                   merged_mass=0.0, merged_energy=0.0, merged_momentum=0.0)
MAX_ATTEMPTS = 6  # attempts of a step in Q1D.substep: second order, first order, then halved


class StepFailed(RuntimeError):
    """A gas step that failed at every attempt of :meth:`Q1D.substep`."""


class Q1D:
    def __init__(self, zf, gamma, p0, rho0, mu=0.0, lam0=0.0, fric=0.0, separated=False,
                 bc_bottom="bz", bc_top="p0", cfl=0.4, order=2):
        self.g, self.p0, self.rho0 = gamma, p0, rho0
        self.mu, self.lam0, self.fric = mu, lam0, fric
        self.separated = separated
        self.bc_bottom, self.bc_top = bc_bottom, bc_top
        self.cfl, self.order = cfl, order
        self.set_grid(zf)
        self.mb = 0.0
        self.Kb = p0 / rho0 ** gamma
        self.cum = dict(top_mass=0.0, bottom_mass=0.0, sep_loss=0.0, fric_loss=0.0, fallback=0)
        self.cum.update(BUDGET_KEYS)
        self.pt, self.rt = p0, rho0  # stagnation state for the 'stag' test condition

    def set_grid(self, zf):
        self.zf = np.asarray(zf, float)
        self.dz = np.diff(self.zf)
        self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.N = len(self.dz)

    # --- state helpers
    def prim(self, U):
        r = U[0]
        u = U[1] / r
        p = (self.g - 1) * (U[2] - 0.5 * r * u ** 2)
        return r, u, p

    def cons(self, r, u, p):
        return np.array([r, r * u, p / (self.g - 1) + 0.5 * r * u ** 2])

    def init_uniform(self, V, Vbz, u=None, p=None, r=None):
        N = self.N
        r = np.full(N, self.rho0) if r is None else r
        u = np.zeros(N) if u is None else u
        p = np.full(N, self.p0) if p is None else p
        self.U = self.cons(r, u, p)
        self.V = V.copy()
        self.Vbz = Vbz
        self.mb = self.rho0 * Vbz
        self.Kb = self.p0 / self.rho0 ** self.g

    def bz_state(self, mb=None):
        mb = self.mb if mb is None else mb
        rb = mb / self.Vbz
        return rb, self.Kb * rb ** self.g

    def bz_energy(self) -> float:
        """The internal energy of the bubble zone, ``p_b V_bz/(gamma - 1)`` (zero without one)."""
        if self.bc_bottom != "bz" or not self.Vbz > 0.0:
            return 0.0
        return float(self.bz_state()[1] * self.Vbz / (self.g - 1))

    def totals(self) -> dict:
        """The inventories: the mass and total energy of the Euler cells and of the bubble zone,
        and their axial momentum (the zone's is zero)."""
        return {"mass_cells": float(np.sum(self.U[0] * self.V)),
                "mass_bz": float(self.mb) if self.bc_bottom == "bz" else 0.0,
                "energy_cells": float(np.sum(self.U[2] * self.V)),
                "energy_bz": self.bz_energy(),
                "momentum_cells": float(np.sum(self.U[1] * self.V)),
                "volume_cells": float(np.sum(self.V)), "volume_bz": float(self.Vbz)}

    # --- wall step: exact adiabatic compression at fixed z
    def wall_step(self, Vnew, Vbz_new):
        E0, Eb0 = float(np.sum(self.U[2] * self.V)), self.bz_energy()
        ratio = self.V / Vnew
        r, u, p = self.prim(self.U)
        r2 = r * ratio
        p2 = p * ratio ** self.g
        self.U = self.cons(r2, u, p2)
        self.V = Vnew.copy()
        self.Vbz = Vbz_new  # bubble zone: isentropic, mass fixed
        self.cum["wall_work"] += float(np.sum(self.U[2] * self.V)) - E0
        self.cum["bz_wall_work"] += self.bz_energy() - Eb0

    # --- ghost states
    def ghost_bottom(self, r, u, p, ue, mb):
        g = self.g
        if self.bc_bottom == "bz":
            rb, pb = self.bz_state(mb)
            return rb, ue, pb
        if self.bc_bottom == "gg":
            c1 = math.sqrt(g * p[0] / r[0])
            clb = c1 - (u[0] - ue) * (g - 1) / 2
            clb = max(clb, 1e-6 * c1)
            plb = ((g / clb ** 2) ** g * p[0] / r[0] ** g) ** (1.0 / (1.0 - g))
            rlb = g * plb / clb ** 2
            return rlb, ue, plb
        if self.bc_bottom == "wall":
            return r[0], -u[0], p[0]
        if self.bc_bottom == "stag":  # subsonic inflow from stagnation state (pt, rt) (tests)
            ct = math.sqrt(g * self.pt / self.rt)
            ug = u[0]
            cg2 = max(ct ** 2 - 0.5 * (g - 1) * ug ** 2, 1e-12 * ct ** 2)
            pg = self.pt * (cg2 / ct ** 2) ** (g / (g - 1))
            return g * pg / cg2, ug, pg
        if self.bc_bottom == "p0":  # open end at p0, mirror of the top condition (tests)
            c1 = math.sqrt(g * p[0] / r[0])
            if -u[0] >= c1:
                return r[0], u[0], p[0]
            rlb = r[0] * (self.p0 / p[0]) ** (1 / g)
            clb = math.sqrt(g * self.p0 / rlb)
            return rlb, u[0] - 2 / (g - 1) * (c1 - clb), self.p0
        raise ValueError(self.bc_bottom)

    def ghost_top(self, r, u, p):
        g = self.g
        if self.bc_top == "wall":
            return r[-1], -u[-1], p[-1]
        cN = math.sqrt(g * p[-1] / r[-1])
        if u[-1] >= cN:
            return r[-1], u[-1], p[-1]
        pub = self.p0
        rub = r[-1] * (pub / p[-1]) ** (1 / g)
        cub = math.sqrt(g * pub / rub)
        uub = 2 / (g - 1) * (cN - cub) + u[-1]
        return rub, uub, pub

    # --- flux operator
    def fluxes(self, U, Sf, ue, mb):
        g = self.g
        r, u, p = self.prim(U)
        gb = self.ghost_bottom(r, u, p, ue, mb)
        gt = self.ghost_top(r, u, p)
        rL = np.empty(self.N + 1); uL = np.empty(self.N + 1); pL = np.empty(self.N + 1)
        rR = np.empty(self.N + 1); uR = np.empty(self.N + 1); pR = np.empty(self.N + 1)
        if self.order == 2 and self.N > 2:
            W = np.array([r, u, p])
            h = np.diff(self.zc)
            d = np.diff(W, axis=1) / h  # slopes between centres, N-1
            sl = np.zeros_like(W)
            sl[:, 1:-1] = minmod(d[:, :-1], d[:, 1:])
            Wm = W - sl * (self.zc - self.zf[:-1])  # value at the lower face of each cell
            Wp = W + sl * (self.zf[1:] - self.zc)  # value at the upper face
            bad = (Wm[0] <= 0) | (Wm[2] <= 0) | (Wp[0] <= 0) | (Wp[2] <= 0)
            if bad.any():
                Wm[:, bad] = W[:, bad]
                Wp[:, bad] = W[:, bad]
                self.cum["fallback"] += int(bad.sum())
            rL[1:], uL[1:], pL[1:] = Wp
            rR[:-1], uR[:-1], pR[:-1] = Wm
        else:
            rL[1:], uL[1:], pL[1:] = r, u, p
            rR[:-1], uR[:-1], pR[:-1] = r, u, p
        rL[0], uL[0], pL[0] = gb
        rR[-1], uR[-1], pR[-1] = gt
        F = hllc(rL, uL, pL, rR, uR, pR, g)
        return F, p

    def rhs(self, U, Sf, ue, mb):
        F, p = self.fluxes(U, Sf, ue, mb)
        FS = F * Sf
        dS = np.diff(Sf)
        dU = -(FS[:, 1:] - FS[:, :-1])
        dU[1] += p * dS
        dU /= self.V
        return dU, F

    def flux_step(self, dt, Sf, ue):
        U0 = self.U
        k1, F1 = self.rhs(U0, Sf, ue, self.mb)
        U1 = U0 + dt * k1
        mb1 = self.mb - dt * F1[0, 0] * Sf[0]
        k2, F2 = self.rhs(U1, Sf, ue, mb1)
        Un = 0.5 * (U0 + U1 + dt * k2)
        Fb = 0.5 * (F1[:, 0] + F2[:, 0])
        Ft = 0.5 * (F1[:, -1] + F2[:, -1])
        r, u, p = self.prim(Un)
        if not (np.all(r > 0) and np.all(p > 0) and np.all(np.isfinite(Un))):
            return False
        self.U = Un
        # bubble zone update with the stage-averaged face flux (exact conservation)
        dm = dt * Fb[0] * Sf[0]
        if self.bc_bottom == "bz":
            rb, pb = self.bz_state()
            mnew = self.mb - dm
            if dm < 0:  # inflow into the bubble zone: mix the energy
                Eint = pb * self.Vbz / (self.g - 1) - dt * Fb[2] * Sf[0]
                pnew = (self.g - 1) * Eint / self.Vbz
                self.Kb = pnew / (mnew / self.Vbz) ** self.g
            self.mb = mnew
        self.cum["bottom_mass"] += dm
        self.cum["top_mass"] += dt * Ft[0] * Sf[-1]
        self.cum["bottom_energy"] += dt * Fb[2] * Sf[0]
        self.cum["top_energy"] += dt * Ft[2] * Sf[-1]
        return True

    def loss_step(self, dt, Sf):
        if self.fric <= 0 and not self.separated:
            return
        r, u, p = self.prim(self.U)
        ke0 = 0.5 * r * u ** 2
        u2 = u.copy()
        if self.fric > 0:
            rad = np.sqrt(self.V / self.dz / math.pi)
            lam = self.lam0 * self.rho0 / r
            kf = self.fric * 8 * self.mu / (r * rad * (rad + 4 * lam))
            u2 = u2 * np.exp(-kf * dt)
        kef = 0.5 * r * u2 ** 2  # after the friction, before the separation loss (budgets only)
        if self.separated:
            dS = np.diff(Sf)
            on = u2 * dS > 0
            u2 = np.where(on, u2 / (1 + dt * np.abs(u2) * np.abs(dS) / self.V), u2)
        ke1 = 0.5 * r * u2 ** 2
        p2 = p + (self.g - 1) * (ke0 - ke1)
        self.U = self.cons(r, u2, p2)
        self.cum["fric_loss"] += float(np.sum((ke0 - kef) * self.V))
        self.cum["sep_loss"] += float(np.sum((kef - ke1) * self.V))

    def max_dt(self):
        r, u, p = self.prim(self.U)
        c = np.sqrt(self.g * p / r)
        return self.cfl * float(np.min(self.dz / (np.abs(u) + c)))

    def remap(self, zf_new, vcum):
        """Conservative piecewise-constant remap of (rho, rho u, E) onto new faces; vcum(z) gives
        the gas volume below z at the current time."""
        zo = self.zf
        zu = np.unique(np.r_[zo, zf_new])
        Vu = vcum(zu)
        dV = np.diff(Vu)
        mid = 0.5 * (zu[1:] + zu[:-1])
        io = np.clip(np.searchsorted(zo, mid) - 1, 0, len(zo) - 2)
        inew = np.clip(np.searchsorted(zf_new, mid) - 1, 0, len(zf_new) - 2)
        tot = self.U[:, io] * dV
        Nn = len(zf_new) - 1
        Unew = np.zeros((3, Nn))
        for q in range(3):
            Unew[q] = np.bincount(inew, weights=tot[q], minlength=Nn)
        Vnew = np.diff(vcum(np.asarray(zf_new)))
        self.set_grid(zf_new)
        self.U = Unew / Vnew
        self.V = Vnew

    # --- additions for the in-loop coupling (module docstring) ----------------------------------

    def substep(self, t: float, dt: float, wall) -> float:
        """One Strang step from ``t``: wall half-step, flux step, loss step, wall half-step, with the
        wall motion ``wall(t) -> (S, Vcum, dVcum/dt)`` at the faces. A step that fails (a negative
        density or pressure) is restored and retaken at first order, then at first order with the
        step halved at each further attempt, :data:`MAX_ATTEMPTS` in all (the retry of the one-way
        driver); :class:`StepFailed` after the last. Returns the time reached, ``t`` plus the step
        actually taken."""
        snap = (self.U.copy(), self.V.copy(), self.Vbz, self.mb, self.Kb, copy.deepcopy(self.cum))
        for attempt in range(MAX_ATTEMPTS):
            th, tn = t + 0.5 * dt, t + dt
            Sh, Vch, dVh = wall(th)
            Sn, Vcn, _ = wall(tn)
            self.order = 2 if attempt == 0 else 1
            self.wall_step(np.diff(Vch), Vch[0])
            ue = -dVh[0] / Sh[0]
            ok = self.flux_step(dt, Sh, ue)
            if ok:
                self.loss_step(dt, Sh)
                self.wall_step(np.diff(Vcn), Vcn[0])
                break
            self.U, self.V, self.Vbz, self.mb, self.Kb = (snap[0].copy(), snap[1].copy(), snap[2],
                                                          snap[3], snap[4])
            self.cum = copy.deepcopy(snap[5])
            self.cum["retries"] += 1 + attempt
            if attempt >= 1:
                dt *= 0.5
        else:
            self.order = 2
            raise StepFailed(f"gas step failed at t = {t!r} after {MAX_ATTEMPTS} attempts")
        self.order = 2
        return tn

    SCALARS = ("g", "p0", "rho0", "mu", "lam0", "fric", "separated", "bc_bottom", "bc_top", "cfl",
               "order", "Vbz", "mb", "Kb", "pt", "rt")

    def state_dict(self) -> dict:
        """The whole state as arrays (``zf``, ``U``, ``V``), scalars and the budgets ``cum``:
        :meth:`load_state_dict` of it continues bitwise."""
        out = {"zf": self.zf.copy(), "U": self.U.copy(), "V": self.V.copy()}
        out.update({k: getattr(self, k) for k in self.SCALARS})
        out["cum"] = dict(self.cum)
        return out

    def load_state_dict(self, d: dict):
        for k in self.SCALARS:
            v = d[k]
            setattr(self, k, v.item() if hasattr(v, "item") else v)
        self.set_grid(np.asarray(d["zf"], dtype=float).copy())
        self.U = np.asarray(d["U"], dtype=float).copy()
        self.V = np.asarray(d["V"], dtype=float).copy()
        self.cum = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(d["cum"]).items()}
        for k, v in BUDGET_KEYS.items():
            self.cum.setdefault(k, v)

    def regrid(self, zf_new, vcum) -> dict:
        """Conservative remap onto the faces ``zf_new``, ``vcum(z)`` the gas volume below ``z`` on
        the current wall. With ``zf_new[0] == zf[0]`` it is :meth:`remap`. Where the lowest face
        rises, ``z_lo' = zf_new[0] > zf[0]``, the gas between the old and the new ``z_lo`` (whole
        cells and the share by volume of the cell the new face cuts) is merged into the bubble zone:
        its mass and total energy are added to the zone's, whose pressure follows from the energy
        at its new volume ``vcum(z_lo')``, and its axial momentum is handed to the wall
        (``cum["merged_momentum"]``). A lower ``z_lo`` is refused. Returns the inventories before
        and after and what was merged."""
        zf_new = np.asarray(zf_new, dtype=float)
        before = self.totals()
        z_lo, z_new = float(self.zf[0]), float(zf_new[0])
        merged = {"mass": 0.0, "energy": 0.0, "momentum": 0.0, "volume": 0.0}
        if z_new < z_lo:
            raise ValueError(f"regrid: the lowest face may only rise ({z_new!r} < {z_lo!r})")
        if z_new > z_lo:
            if self.bc_bottom != "bz":
                raise ValueError("regrid: a rising lowest face needs the bubble zone ('bz')")
            if not z_new < float(self.zf[-1]):
                raise ValueError("regrid: the new lowest face lies above the top")
            zo = self.zf
            k = int(np.searchsorted(zo, z_new, side="right"))  # faces zo[:k] lie at or below z_new
            cut = np.r_[zo[:k], z_new] if zo[k - 1] < z_new else zo[:k]
            dV = np.diff(vcum(cut))
            part = self.U[:, :len(dV)] * dV
            merged = {"mass": float(np.sum(part[0])), "momentum": float(np.sum(part[1])),
                      "energy": float(np.sum(part[2])), "volume": float(np.sum(dV))}
            Eb = self.bz_energy() + merged["energy"]
            mnew = self.mb + merged["mass"]
            # the old zone and the merged gas at the zone's new volume, mixed at rest
            self.Vbz = float(vcum(np.array([z_new]))[0])
            pnew = (self.g - 1) * Eb / self.Vbz
            self.mb = mnew
            self.Kb = pnew / (mnew / self.Vbz) ** self.g
            self.cum["merged_mass"] += merged["mass"]
            self.cum["merged_energy"] += merged["energy"]
            self.cum["merged_momentum"] += merged["momentum"]
            # the cells above: the part of the cut cell above z_new, then the rest, onto the new
            # faces (remap reads only the overlaps of the old cells with [z_new, z_hi])
            j = len(dV) - 1 if zo[k - 1] < z_new else len(dV)
            keep_U = self.U[:, j:]
            zo_kept = np.r_[z_new, zo[k:]] if zo[k - 1] < z_new else zo[k - 1:]
            self.set_grid(zo_kept)
            self.U = keep_U.copy()
            self.V = np.diff(vcum(zo_kept))
        self.remap(zf_new, vcum)
        after = self.totals()
        return {"before": before, "after": after, "merged": merged}
