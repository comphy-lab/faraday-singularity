"""Verification of the quasi-1D moving-wall gas solver (``faraday_singularity.neckgas.q1d``) on
exact and analytic limits.

T1 gas at rest in a static converging-diverging duct stays at rest (well-balanced area source);
T2 uniform compression of a closed duct: p S^gamma = const (exact wall step, any rate);
T3 steady choked nozzle with a normal shock: area-Mach relation, choked mass flux
   0.5787 rho_t c_t S*, throat p/p_t = 0.528 and the exact shock position (grid convergence);
T4 quasi-steady low-Mach flow driven by a shrinking lumped bubble zone: Bernoulli in the attached
   limit; constant pressure beyond the throat in the separated limit (the Borda-Carnot loss);
T5 incompressible squeeze of a uniformly shrinking open cylinder, S = S0 exp(-a t):
   exact p_c - p0 = rho a^2 (L/2)^2 / 2; then the compressible squeeze between that limit and
   adiabatic trapping (T2) as the squeeze Mach number a L/(2 c) grows;
T6 unsteady de Laval nozzle: throat pressure over bubble-zone pressure when the throat first
   becomes sonic, slow and fast closure.

The tests are those of the one-way solver this module was ported from, unchanged but for the
import and the output directory, so that the port is checked by reproducing their values
(``--compare``, to a relative ``--rtol``, 1e-12 by default) as well as by the exact limits each
test states. Results: ``verify_results.json`` and ``verify_arrays.npz`` in ``--out``.

Usage::

    python -I verify_q1d.py --out DIR [T1 T3 T4 T5 T6]
    python -I verify_q1d.py --out DIR --compare REFERENCE.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from faraday_singularity.neckgas import q1d  # noqa: E402

HERE = os.environ.get("NECKGAS_VERIFY_OUT", ".")

G = 1.4
GL = np.polynomial.legendre.leggauss(5)


def cell_volumes(Sfun, zf, t):
    x, w = GL
    a, b = zf[:-1], zf[1:]
    zq = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (b + a)[:, None]
    return np.sum(Sfun(zq, t) * w[None, :], axis=1) * 0.5 * (b - a)


def advance(sol, Sfun, t, dt, Vbz=lambda t: 1.0, dVbz=lambda t: 0.0):
    th, tn = t + 0.5 * dt, t + dt
    sol.wall_step(cell_volumes(Sfun, sol.zf, th), Vbz(th))
    Sh = Sfun(sol.zf, th)
    ue = -dVbz(th) / Sh[0]
    ok = sol.flux_step(dt, Sh, ue)
    if not ok:
        raise RuntimeError("step failed")
    sol.loss_step(dt, Sh)
    sol.wall_step(cell_volumes(Sfun, sol.zf, tn), Vbz(tn))
    return tn


def run(sol, Sfun, t_end, Vbz=lambda t: 1.0, dVbz=lambda t: 0.0, hook=None, dtmax=np.inf):
    t = 0.0
    while t < t_end - 1e-14:
        dt = min(sol.max_dt(), t_end - t, dtmax)
        t = advance(sol, Sfun, t, dt, Vbz, dVbz)
        if hook is not None and hook(sol, t):
            break
    return t


# ------------------------------------------------------------------ exact nozzle relations
def area_ratio(M, g=G):
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M ** 2)) ** ((g + 1) / (2 * (g - 1)))


def mach_from_area(A, supersonic, g=G):
    if abs(A - 1) < 1e-14:
        return 1.0
    f = lambda M: area_ratio(M, g) - A  # noqa: E731
    return brentq(f, 1.0 + 1e-12, 50.0) if supersonic else brentq(f, 1e-8, 1.0 - 1e-12)


def p_over_pt(M, g=G):
    return (1 + 0.5 * (g - 1) * M ** 2) ** (-g / (g - 1))


def shock_pt_ratio(M1, g=G):
    a = ((g + 1) * M1 ** 2 / 2) / (1 + 0.5 * (g - 1) * M1 ** 2)
    b = 2 * g * M1 ** 2 / (g + 1) - (g - 1) / (g + 1)
    return a ** (g / (g - 1)) * b ** (-1 / (g - 1))


def shock_area(Ae, pb_over_pt, g=G):
    def f(As):
        M1 = mach_from_area(As, True, g)
        r = shock_pt_ratio(M1, g)
        Me = mach_from_area(Ae * r, False, g)
        return r * p_over_pt(Me, g) - pb_over_pt
    return brentq(f, 1.0 + 1e-9, Ae - 1e-9)


def t1_rest():
    zf = np.linspace(0, 1, 201)
    S = lambda z, t: 1 + 8 * (z - 0.5) ** 2  # noqa: E731
    sol = q1d.Q1D(zf, G, 1.0, 1.0, bc_bottom="wall", bc_top="wall")
    sol.init_uniform(cell_volumes(S, zf, 0), 1.0)
    for _ in range(3000):
        advance(sol, S, 0.0, sol.max_dt())
    r, u, p = sol.prim(sol.U)
    return {"max_abs_u_over_c": float(np.max(np.abs(u)) / math.sqrt(G)),
            "max_rel_dp": float(np.max(np.abs(p - 1)))}


def t2_adiabatic():
    zf = np.linspace(0, 1, 101)
    f = lambda t: 1 / (1 + 3 * t)  # noqa: E731
    S = lambda z, t: (1 + 8 * (z - 0.5) ** 2) * f(t)  # noqa: E731
    sol = q1d.Q1D(zf, G, 1.0, 1.0, bc_bottom="wall", bc_top="wall")
    sol.init_uniform(cell_volumes(S, zf, 0), 1.0)
    out = []
    for T, dtm in ((1.0, np.inf), (1.0, 0.2)):  # CFL steps, and 5 huge steps
        sol.init_uniform(cell_volumes(S, zf, 0), 1.0)
        run(sol, S, T, dtmax=dtm)
        r, u, p = sol.prim(sol.U)
        exact = f(T) ** (-G)
        out.append({"dtmax": dtm, "p_over_exact_minus_1": float(np.max(np.abs(p / exact - 1))),
                    "max_abs_u": float(np.max(np.abs(u)))})
    return out


def t3_nozzle():
    pt, rt, pb = 1.0, 1.0, 0.6  # shock inside for 0.374 < pb/pt < 0.973 (A_e/A* = 3)
    ct = math.sqrt(G * pt / rt)
    S = lambda z, t: 1 + 8 * (z - 0.5) ** 2  # noqa: E731
    Ae = 3.0
    As = shock_area(Ae, pb / pt)
    zs_exact = 0.5 + math.sqrt((As - 1) / 8)
    mdot_star = (2 / (G + 1)) ** ((G + 1) / (2 * (G - 1))) * rt * ct * 1.0
    res = {"exact": {"A_shock": As, "z_shock": zs_exact, "mdot_star": mdot_star,
                     "p_throat_over_pt": p_over_pt(1.0)}, "runs": []}
    arrays = {}
    for N in (100, 200, 400, 800):
        zf = np.linspace(0, 1, N + 1)
        sol = q1d.Q1D(zf, G, pb, rt, bc_bottom="stag", bc_top="p0")
        sol.pt, sol.rt = pt, rt
        sol.init_uniform(cell_volumes(S, zf, 0), 1.0, r=np.full(N, rt), p=np.full(N, pt))
        run(sol, S, 40.0)
        r, u, p = sol.prim(sol.U)
        c = np.sqrt(G * p / r)
        M = u / c
        mdot = r * u * sol.V / sol.dz
        zc = sol.zc
        # exact Mach at cell centres (shock at zs_exact)
        Ae_z = 1 + 8 * (zc - 0.5) ** 2
        r_pt = shock_pt_ratio(mach_from_area(As, True))
        Mex = np.array([mach_from_area(a, False) if z < 0.5 else
                        (mach_from_area(a, True) if z < zs_exact else mach_from_area(a * r_pt, False))
                        for a, z in zip(Ae_z, zc)])
        away = np.abs(zc - zs_exact) > 0.05
        # computed shock: steepest pressure rise downstream of the throat
        j = int(np.argmax(np.where(zc > 0.5, np.r_[np.diff(p), 0], -np.inf)))
        zs = 0.5 * (zc[j] + zc[j + 1])
        i_th = int(np.argmin(np.abs(zc - 0.5)))
        res["runs"].append({"N": N, "mdot_over_star": float(np.median(mdot) / mdot_star),
                            "mdot_spread": float((mdot.max() - mdot.min()) / mdot_star),
                            "M_err_max_away": float(np.max(np.abs(M - Mex)[away])),
                            "M_err_L1_away": float(np.mean(np.abs(M - Mex)[away])),
                            "z_shock": float(zs), "z_shock_err": float(zs - zs_exact),
                            "p_throat_over_pt": float(np.interp(0.5, zc, p) / pt),
                            "M_throat": float(np.interp(0.5, zc, M))})
        arrays[f"t3_N{N}_zc"], arrays[f"t3_N{N}_M"], arrays[f"t3_N{N}_Mex"] = zc, M, Mex
    return res, arrays


def ramp(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp_int(t, Tr):
    """int_0^t ramp(t'/Tr) dt'."""
    if t >= Tr:
        return 0.5 * Tr + (t - Tr)
    x = t / Tr
    return Tr * (x ** 3 - 0.5 * x ** 4)


def t4_bernoulli(N=401, seps=(False, True)):
    p0, r0 = 1 / G, 1.0  # c0 = 1
    S = lambda z, t: 1 + 3 * (2 * z - 1) ** 2  # noqa: E731
    Q, V0, Tr, Tend = 0.06, 20.0, 20.0, 80.0
    Vbz = lambda t: V0 - Q * ramp_int(t, Tr)  # noqa: E731
    dVbz = lambda t: -Q * ramp(t / Tr)  # noqa: E731
    out, arrays = {}, {}
    for sep in seps:
        zf = np.linspace(0, 1, N)
        sol = q1d.Q1D(zf, G, p0, r0, separated=sep, bc_bottom="bz", bc_top="p0")
        sol.init_uniform(cell_volumes(S, zf, 0), Vbz(0))
        run(sol, S, Tend, Vbz, dVbz)
        r, u, p = sol.prim(sol.U)
        zc = sol.zc
        Sz = S(zc, 0)
        mdot_c = r * u * sol.V / sol.dz
        mdot = float(np.median(mdot_c))
        rb, pb = sol.bz_state()
        ith = int(np.argmin(np.abs(zc - 0.5)))
        uth = float(u[ith])
        q = 0.5 * r0 * uth ** 2
        ui = Q / Sz
        if not sep:
            p_inc = p[-1] + 0.5 * r0 * (ui[-1] ** 2 - ui ** 2)
        else:
            p_inc = np.where(zc <= 0.5, p[ith] + 0.5 * r0 * (ui[ith] ** 2 - ui ** 2), p[ith])
        o = {"M_throat": uth, "q_throat": q,
             "err_vs_incompressible_over_q": float(np.max(np.abs(p - p_inc)) / q),
             "throat_minus_top_over_q": float((p[ith] - p[-1]) / q),
             "bz_minus_throat_over_q": float((pb - p[ith]) / q),
             "mdot_over_rho0Q": float(mdot / (r0 * Q)),
             "mdot_cell_spread_over_mdot": float((mdot_c.max() - mdot_c.min()) / mdot)}
        if not sep:
            # exact steady isentropic compressible flow with the measured entropy, mass flux and top
            # pressure (subsonic branch)
            K = float(p[ith] / r[ith] ** G)
            h = lambda rho: G / (G - 1) * K * rho ** (G - 1)  # noqa: E731
            rt_ = (p[-1] / K) ** (1 / G)
            ht = h(rt_) + 0.5 * (mdot / (rt_ * Sz[-1])) ** 2

            def rho_at(Sv):
                f = lambda rho: h(rho) + 0.5 * (mdot / (rho * Sv)) ** 2 - ht  # noqa: E731
                return brentq(f, 0.9 * rt_, 1.1 * rt_)
            pc = np.array([K * rho_at(sv) ** G for sv in Sz])
            o["err_vs_compressible_over_q"] = float(np.max(np.abs(p - pc)) / q)
            o["compressible_minus_incompressible_over_q"] = float(np.max(np.abs(pc - p_inc)) / q)
            arrays["t4_A_pc"] = pc
        else:
            down = zc > 0.55
            o["downstream_p_spread_over_q"] = float((p[down].max() - p[down].min()) / q)
        out["separated" if sep else "attached"] = o
        k = "S" if sep else "A"
        arrays[f"t4_{k}_zc"], arrays[f"t4_{k}_p"], arrays[f"t4_{k}_pinc"] = zc, p, p_inc
    return out, arrays


def squeeze_F(g=G):
    xs = math.sqrt(2 / (g + 1))
    from scipy.integrate import quad
    return quad(lambda x: 1 - x * x / (1 - 0.5 * (g - 1) * x * x), 0, xs)[0]


def squeeze_exact(a, L, p0, K, g=G):
    """Steady squeeze of a uniformly shrinking open cylinder (S ~ exp(-a t)), length L, exits at
    p0, isentropic with entropy K: centre pressure and exit Mach number. u' = a/(1 - M^2),
    c^2 = c_c^2 - (g-1) u^2/2. Choked when the sonic point would lie inside the duct; then
    c_c = a L/(2 F)."""
    from scipy.integrate import quad
    F = squeeze_F(g)

    def z_of_u(u, cc):
        return quad(lambda w: 1 - w * w / (cc * cc - 0.5 * (g - 1) * w * w), 0, u)[0] / a

    def resid(pc):
        cc = math.sqrt(g * K ** (1 / g) * pc ** ((g - 1) / g))
        us = cc * math.sqrt(2 / (g + 1))
        if z_of_u(us, cc) < L / 2:  # would choke inside the duct: centre pressure too low
            return -p0, 1.0
        ue = brentq(lambda u: z_of_u(u, cc) - L / 2, 0.0, us)
        pe = pc * (1 - 0.5 * (g - 1) * ue ** 2 / cc ** 2) ** (g / (g - 1))
        return pe - p0, ue / math.sqrt(cc ** 2 - 0.5 * (g - 1) * ue ** 2)

    cc_choke = a * L / (2 * F)
    pc_choke = (cc_choke ** 2 / (g * K ** (1 / g))) ** (g / (g - 1))
    pe_choke = pc_choke * (2 / (g + 1)) ** (g / (g - 1))
    if pe_choke >= p0:
        return {"choked": True, "pc": pc_choke, "cc": cc_choke, "M_exit": 1.0}
    hi = 2 * p0 + 5 * (a * L / 2) ** 2 * (p0 / K) ** (1 / g)
    pc = brentq(lambda x: resid(x)[0], p0 * (1 + 1e-12), hi)
    return {"choked": False, "pc": pc, "cc": math.sqrt(g * K ** (1 / g) * pc ** ((g - 1) / g)),
            "M_exit": resid(pc)[1]}


def t5_squeeze():
    p0, r0 = 1 / G, 1.0
    L, Tr, Tend = 1.0, 10.0, 30.0
    out = {"F_gamma": squeeze_F(), "cases": []}
    arrays = {}
    zf = np.linspace(-L / 2, L / 2, 401)
    for edge in (0.1, 0.3, 0.6, 1.0, 2.0, 4.0):
        a = 2 * edge / L
        S = lambda z, t, a=a: np.exp(-a * ramp_int(t, Tr)) * np.ones_like(z)  # noqa: E731
        sol = q1d.Q1D(zf, G, p0, r0, bc_bottom="p0", bc_top="p0")
        sol.init_uniform(cell_volumes(S, zf, 0), 1.0)
        run(sol, S, Tend)
        r, u, p = sol.prim(sol.U)
        zc = sol.zc
        i = int(np.argmin(np.abs(zc)))
        pc_num = 0.5 * (p[i] + p[i - 1]) if abs(zc[i]) > 1e-12 else p[i]
        Kc = float(np.mean(p[i - 1:i + 1] / r[i - 1:i + 1] ** G))
        cc_num = float(np.sqrt(G * pc_num / np.mean(r[i - 1:i + 1])))
        ex = squeeze_exact(a, L, p0, Kc)
        c_edge = math.sqrt(G * p[-1] / r[-1])
        o = {"edge_speed_over_c0": edge, "a": a, "pc_over_p0": float(pc_num / p0),
             "exact_pc_over_p0": ex["pc"] / p0, "rel_err_pc_minus_p0": float((pc_num - ex["pc"]) / (ex["pc"] - p0)),
             "cc_over_c0": cc_num, "exact_cc_over_c0": ex["cc"], "choked": ex["choked"],
             "M_exit_num": float(u[-1] / c_edge), "M_exit_exact": ex["M_exit"],
             "incompressible_pc_over_p0": 1 + 0.5 * r0 * (a * L / 2) ** 2 / p0}
        if edge == 0.1:
            o["max_u_err_over_edge"] = float(np.max(np.abs(u - a * zc)) / edge)
        out["cases"].append(o)
        arrays[f"t5_e{edge}_zc"], arrays[f"t5_e{edge}_p"], arrays[f"t5_e{edge}_u"] = zc, p, u
    return out, arrays


def t6_delaval():
    """Throat closing in time while a lumped bubble zone pushes a fixed volume flux Q."""
    p0, r0 = 1 / G, 1.0
    Se, Q, V0, Tq = 4.0, 0.3, 100.0, 5.0  # V0 > Q T for the slowest closure
    out = []
    for T in (200.0, 40.0, 20.0, 12.0, 8.0):
        Sth = lambda t, T=T: 1.0 - 0.95 * min(t / T, 1.0)  # noqa: E731
        S = lambda z, t, Sth=Sth: Sth(t) + (Se - Sth(t)) * (2 * z - 1) ** 2  # noqa: E731
        zf = np.linspace(0, 1, 201)
        sol = q1d.Q1D(zf, G, p0, r0, bc_bottom="bz", bc_top="p0")
        Vbz = lambda t: V0 - Q * ramp_int(t, Tq)  # noqa: E731
        dVbz = lambda t: -Q * ramp(t / Tq)  # noqa: E731
        sol.init_uniform(cell_volumes(S, zf, 0), Vbz(0))
        rec = {"T_close_over_acoustic": T}
        hist = []

        def hook(sol, t):
            r, u, p = sol.prim(sol.U)
            i = int(np.argmin(sol.V / sol.dz))
            M = u[i] / math.sqrt(G * p[i] / r[i])
            rb, pb = sol.bz_state()
            hist.append((t, sol.V[i] / sol.dz[i], M, p[i] / pb, pb / p0, float(np.max(p) / p0)))
            if M >= 1.0 and "p_th_over_pbz" not in rec:
                rec.update(t=t, S_throat=float(sol.V[i] / sol.dz[i]), p_th_over_pbz=float(p[i] / pb),
                           p_bz_over_p0=float(pb / p0), p_th_over_p0=float(p[i] / p0))
            return False
        run(sol, S, 0.999 * T, Vbz, dVbz, hook=hook)
        H = np.array(hist)
        rec["M_throat_max"] = float(H[:, 2].max())
        rec["p_throat_over_pbz_final"] = float(H[-1, 3])
        rec["p_max_over_p0_final"] = float(H[-1, 5])
        rec["S_throat_final"] = float(H[-1, 1])
        out.append(rec)
    return out


def run_test(name):
    """Run one test and write verify_<name>.json and verify_<name>.npz."""
    arrays = {}
    if name == "T1":
        res = {"T1_rest": t1_rest(), "T2_adiabatic": t2_adiabatic()}
    elif name == "T3":
        r, arrays = t3_nozzle()
        res = {"T3_nozzle": r}
    elif name == "T4":
        r, arrays = t4_bernoulli()
        res = {"T4_bernoulli": r, "T4_attached_grid": {}}
        for N in (201, 801):
            o, _ = t4_bernoulli(N, (False,))
            res["T4_attached_grid"][str(N - 1)] = {k: o["attached"][k] for k in
                                                   ("err_vs_compressible_over_q", "throat_minus_top_over_q")}
    elif name == "T5":
        r, arrays = t5_squeeze()
        res = {"T5_squeeze": r}
    elif name == "T6":
        res = {"T6_delaval": t6_delaval()}
    else:
        raise ValueError(name)
    with open(os.path.join(HERE, f"verify_{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(HERE, f"verify_{name}.npz"), **arrays)
    return name


def merge():
    res, arrays = {}, {}
    for name in ("T1", "T3", "T4", "T5", "T6"):
        res.update(json.load(open(os.path.join(HERE, f"verify_{name}.json"))))
        a = np.load(os.path.join(HERE, f"verify_{name}.npz"))
        arrays.update({k: a[k] for k in a.files})
    with open(os.path.join(HERE, "verify_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(HERE, "verify_arrays.npz"), **arrays)
    return res


def differences(a, b, rtol: float, path: str = "") -> list[str]:
    """The places where two results differ beyond ``rtol`` (relative, absolute for a zero)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}: missing" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += differences(a[k], b[k], rtol, f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, rtol, f"{path}[{i}]")
        return out
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if math.isinf(a) or math.isinf(b) or math.isnan(a) or math.isnan(b):
        return [] if (a == b or (math.isnan(a) and math.isnan(b))) else [f"{path}: {a} != {b}"]
    scale = max(abs(a), abs(b))
    return [] if abs(a - b) <= rtol * scale else [f"{path}: {a!r} != {b!r}"]


def main():
    global HERE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", default=["T1", "T3", "T4", "T5", "T6"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="a verify_results.json to reproduce")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--processes", type=int, default=1)
    args = ap.parse_args()
    HERE = args.out
    os.environ["NECKGAS_VERIFY_OUT"] = HERE
    os.makedirs(HERE, exist_ok=True)
    names = args.tests
    if args.processes > 1:
        from multiprocessing import Pool
        with Pool(min(args.processes, len(names))) as pool:
            for n in pool.imap_unordered(run_test, names):
                print("done", n, flush=True)
    else:
        for n in names:
            run_test(n)
            print("done", n, flush=True)
    if set(names) >= {"T1", "T3", "T4", "T5", "T6"}:
        res = merge()
        if args.compare:
            ref = json.load(open(args.compare))
            bad = differences(res, ref, args.rtol)
            report = {"compare": args.compare, "rtol": args.rtol, "differences": bad,
                      "passed": not bad}
            with open(os.path.join(HERE, "compare.json"), "w") as f:
                json.dump(report, f, indent=1)
            print(json.dumps(report, indent=1))
            return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
