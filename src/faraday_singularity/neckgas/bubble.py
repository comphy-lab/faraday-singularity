"""The entrained gas after the cut: a polytropic bubble (``reconnect.bubble_model``).

At the cut of the neck the gas below it (the coupled gas's partition at the cut's neck,
``coupler.handover``: its mass ``m_b``, its volume ``V_b`` and its pressure at rest with its
internal energy kept, ``p_b,mix``) becomes the gas of the bubble the cut makes, a closed volume of
uniform pressure on the polytrope

    p V^kappa = C,      C = p_b,mix V_b^kappa,

``kappa`` 1.4 (``polytropic``, adiabatic air) or 1 (``isothermal``). The bubble keeps its mass; its
volume ``V`` is that of the free surface that bounds it, read from the mesh before every step of the
liquid: the bubble's chain (from the axis to the axis) while it is one, the ring of a pierced bubble
(``ring.py``) after the ring's cut, which keeps the bubble's gas, all of it, on the same polytrope
(the column the ring cut closes holds no gas of its own: mass is conserved and the gas is compressed
or expanded isentropically into the ring), and none after the ring's removal (the gas leaves with
it, recorded). The cut itself changes the bubble's volume from ``V_b`` to the volume of the chain
it built (the bridge removed), on the same polytrope.

Before each step of the liquid the traction ``p(V) - p0`` (``sigma k`` units) is set at every node
of the gas's surface and zero elsewhere (the cavity above vents to the atmosphere), and held over
the step (staggered, as before the cut); after the step the volume of the stepped surface gives the
row of the step. Every change of volume between steps (a rebuild of the mesh, a surgery: a liquid
cut on the bubble, the fill of a groove of the bubble, the ring's cut) moves the gas along the same
polytrope and is recorded as a jump.

Recorded per step (``bubble_gas.txt``, :data:`COLUMNS`): the volume and pressure at the start (the
pressure the liquid sees) and at the end, the mass, the internal energy ``p V/(kappa - 1)``
(``kappa > 1``), the work of the gas on the liquid as the liquid sees it,
``W_traction = sum (p_start - p0)(V_end - V_start)``, and along the polytrope,
``W_poly = sum int (p - p0) dV`` over the steps, whose difference is the lag of the coupling, and
the jumps between steps. Units are the solver's (length ``1/k``, pressure ``sigma k``).
"""

from __future__ import annotations

import math

KAPPA = {"polytropic": 1.4, "isothermal": 1.0}
STAGES = ("bubble", "ring", "gone")

COLUMNS = ("step", "time", "dt", "stage", "V_start", "p_start", "T_start", "V_end", "p_end",
           "p_end_over_p0", "m", "rho_over_rho0", "E_end", "W_traction", "W_poly", "jump_V",
           "jump_W", "jumps", "nodes", "R_eq", "omega_dt")


def poly_work(C: float, kappa: float, V0: float, V1: float) -> float:
    """``int_V0^V1 p dV`` on ``p V^kappa = C``."""
    if kappa == 1.0:
        return C * math.log(V1 / V0)
    return C * (V1 ** (1.0 - kappa) - V0 ** (1.0 - kappa)) / (1.0 - kappa)


class BubbleGas:
    """The polytropic gas of the bubble after the cut (module docstring)."""

    def __init__(self, *, model: str, kappa: float, p0: float, rho0: float, m: float, C: float,
                 V_ref: float, p_ref: float, source: dict | None = None):
        if model not in KAPPA:
            raise ValueError(f"no bubble model {model!r}; the models are {tuple(KAPPA)}")
        if not (m > 0 and C > 0 and V_ref > 0 and p_ref > 0):
            raise ValueError("a polytropic bubble needs a positive mass, volume and pressure")
        self.model, self.kappa, self.p0, self.rho0 = model, float(kappa), float(p0), float(rho0)
        self.m, self.C, self.V_ref, self.p_ref = float(m), float(C), float(V_ref), float(p_ref)
        self.source = dict(source or {})
        self.stage = "bubble"
        self.V_start = self.V_end = math.nan  # the gas's volume at the start and end of a step
        self.p_start = math.nan
        self.t_start = math.nan
        self.W_traction = self.W_poly = 0.0
        self.jumps = 0
        self.jump_V_total = self.jump_W_total = 0.0
        self.transitions: list[dict] = []  # the cut, the ring's cut, the removal
        self._pending_jump = (0.0, 0.0)

    @classmethod
    def from_handover(cls, model: str, handover: dict, V_mesh: float, *, t: float,
                      step: int) -> "BubbleGas":
        """The bubble of the cut from the coupled gas's partition at the cut's neck
        (``coupler.handover`` of the variant the liquid felt): ``m_b``, the gas volume below the neck
        ``V_gas_below`` and ``p_b_mix``, at the volume ``V_mesh`` of the bubble's chain after the
        cut."""
        if model not in KAPPA:
            raise ValueError(f"no bubble model {model!r}; the models are {tuple(KAPPA)}")
        kappa = KAPPA[model]
        V, p = float(handover["V_gas_below"]), float(handover["p_b_mix"])
        gas = cls(model=model, kappa=kappa, p0=float(handover["p0"]),
                  rho0=float(handover["rho0"]), m=float(handover["m_b"]), C=p * V ** kappa,
                  V_ref=V, p_ref=p, source={k: handover[k] for k in (
                      "z_n", "m_b", "E_b", "KE_b", "V_b", "V_gas_below", "p_pk", "p_b_mix",
                      "p_bz", "m_above", "E_above") if k in handover})
        gas.V_end = V
        gas.transitions.append(gas._transition("cut", t, step, V, V_mesh, "bubble"))
        gas.V_end = V_mesh
        return gas

    # --- the state ----------------------------------------------------------------------------

    def pressure(self, V: float) -> float:
        return self.C / V ** self.kappa

    def energy(self, V: float) -> float:
        """The internal energy ``p V/(kappa - 1)`` (NaN for the isothermal gas)."""
        return self.pressure(V) * V / (self.kappa - 1.0) if self.kappa != 1.0 else math.nan

    def _transition(self, kind: str, t: float, step: int, V0: float, V1: float,
                    stage: str) -> dict:
        p0, p1 = self.pressure(V0), (self.pressure(V1) if V1 > 0 else math.nan)
        return {"kind": kind, "time": t, "step": step, "stage": stage, "V_before": V0,
                "V_after": V1, "p_before": p0, "p_after": p1, "m": self.m,
                "W_poly": poly_work(self.C, self.kappa, V0, V1) - self.p0 * (V1 - V0)
                if V1 > 0 else math.nan}

    # --- the step of the liquid ---------------------------------------------------------------

    def before_step(self, t: float, step: int, stage: str, V: float | None) -> float:
        """The traction ``p(V) - p0`` the step holds on the gas's surface, its volume ``V`` read on
        the mesh the step starts from (None without one: the gas is gone, the traction zero).
        A volume other than the last step's end is a jump along the polytrope (a rebuild or a
        surgery), a stage other than the last a transition (the ring's cut, the removal)."""
        if self.stage == "gone":
            return 0.0
        if V is None or not V > 0.0:
            self.transitions.append(self._transition("removed", t, step, self.V_end, 0.0,
                                                     "gone"))
            self.transitions[-1].update(E_removed=self.energy(self.V_end))
            self.stage = "gone"
            self.V_start = self.p_start = math.nan
            return 0.0
        if stage != self.stage:
            self.transitions.append(self._transition(f"{self.stage}-to-{stage}", t, step,
                                                     self.V_end, V, stage))
            self.stage = stage
        if math.isfinite(self.V_end) and V != self.V_end:
            dW = poly_work(self.C, self.kappa, self.V_end, V) - self.p0 * (V - self.V_end)
            self.jumps += 1
            self.jump_V_total += V - self.V_end
            self.jump_W_total += dW
            self._pending_jump = (V - self.V_end, dW)
        self.V_start, self.t_start = V, t
        self.p_start = self.pressure(V)
        return self.p_start - self.p0

    def after_step(self, t: float, step: int, V: float | None, *, nodes: int = 0) -> dict | None:
        """The row of the step just made, on the stepped surface's volume ``V``; None when the gas
        is gone or the volume cannot be read (a changed topology the stops record)."""
        if self.stage == "gone" or not math.isfinite(self.V_start) or V is None or not V > 0:
            return None
        dt = t - self.t_start
        self.W_traction += (self.p_start - self.p0) * (V - self.V_start)
        self.W_poly += poly_work(self.C, self.kappa, self.V_start, V) - self.p0 * (V - self.V_start)
        self.V_end = V
        p = self.pressure(V)
        R_eq = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)
        omega = math.sqrt(3.0 * self.kappa * self.p_start) / R_eq  # Minnaert, liquid density 1
        jV, jW = self._pending_jump
        self._pending_jump = (0.0, 0.0)
        return {"step": step, "time": t, "dt": dt, "stage": self.stage, "V_start": self.V_start,
                "p_start": self.p_start, "T_start": self.p_start - self.p0, "V_end": V, "p_end": p,
                "p_end_over_p0": p / self.p0, "m": self.m,
                "rho_over_rho0": self.m / V / self.rho0, "E_end": self.energy(V),
                "W_traction": self.W_traction, "W_poly": self.W_poly, "jump_V": jV,
                "jump_W": jW, "jumps": self.jumps, "nodes": nodes, "R_eq": R_eq,
                "omega_dt": omega * dt}

    # --- the dumps ----------------------------------------------------------------------------

    KEYS = ("model", "kappa", "p0", "rho0", "m", "C", "V_ref", "p_ref", "source", "stage",
            "V_start", "V_end", "p_start", "t_start", "W_traction", "W_poly", "jumps",
            "jump_V_total", "jump_W_total", "transitions")

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}

    @classmethod
    def from_dict(cls, d: dict) -> "BubbleGas":
        gas = cls(model=d["model"], kappa=d["kappa"], p0=d["p0"], rho0=d["rho0"], m=d["m"],
                  C=d["C"], V_ref=d["V_ref"], p_ref=d["p_ref"], source=d.get("source"))
        for k in cls.KEYS[9:]:
            setattr(gas, k, d[k])
        gas.transitions = [dict(x) for x in d["transitions"]]
        return gas

    def summary(self) -> dict:
        V = self.V_end
        return {"model": self.model, "kappa": self.kappa, "stage": self.stage, "m": self.m,
                "p_ref": self.p_ref, "V_ref": self.V_ref, "V": V,
                "p": self.pressure(V) if math.isfinite(V) and V > 0 else math.nan,
                "W_traction": self.W_traction, "W_poly": self.W_poly, "jumps": self.jumps,
                "jump_W_total": self.jump_W_total, "transitions": list(self.transitions)}
