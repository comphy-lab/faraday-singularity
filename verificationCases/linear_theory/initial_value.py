"""Exact linear initial-value response of the J0 free-surface mode, released from rest.

A simulation starts from a deformed free surface ``zeta = a0 J0(r)`` with the liquid at rest. Its
linear response is not a pure normal mode: vorticity diffuses in from the surface, and with a
free-slip bottom slowly decaying bulk vortical modes are weakly excited. This module integrates the
full linearised problem in time from that state, so a simulated amplitude history ``a(t)`` can be
compared with ``a0 * Z(t)`` over the whole run, not only with a fitted exponent.

Model (planar notation; the J0/J1 axisymmetric case is identical because ``(1/r)(r J1)' = J0`` and
``(lap - 1/r^2) J1 = -J1``), with ``u = U(z,t) J1(r)``, ``w = W(z,t) J0(r)``, ``p' = P(z,t) J0(r)``,
``zeta = Z(t) J0(r)``::

    continuity        U + W_z = 0
    radial momentum   U_t =  P   + Oh (U_zz - U)
    axial momentum    W_t = -P_z + Oh (W_zz - W)
    z = 0:  U_z - W = 0,   P - 2 Oh W_z - (1 + Bo + Gamma cos(Omega t)) Z = 0,   Z_t = W
    z = -d: W = 0 and U = 0 (no-slip) or U_z = 0 (free-slip)

The container-frame base state is at rest with pressure ``-G(t) z``, ``G = Bo + Gamma cos(Omega t)``,
the same forcing convention as the simulations and ``faraday_linear``. The pressure is kept as an
unknown and every boundary condition is written from the stress tensor, independently of the
pressure-eliminated derivation in ``faraday_linear``. Chebyshev Gauss-Lobatto collocation in ``z``;
the velocity constraints are removed with a null-space basis and the pressure is solved for at each
instant, leaving ``y' = (A0 + Gamma cos(Omega t) A1) y`` with ``Z`` the last component.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    Dm = Dm - np.diag(Dm.sum(axis=1))
    return Dm, x


def descriptor(Oh, Bo, d, bottom, N):
    """``M x' = (L0 + g(t) L1) x`` for ``x = [U(0..N), W(0..N), P(0..N), Z]``, ``g = Gamma cos``.

    Node 0 is the free surface ``z = 0`` and node ``N`` the bottom ``z = -d``.
    """
    Dx, _ = cheb(N)
    D1 = (2.0 / d) * Dx
    D2 = D1 @ D1
    n = N + 1
    iU, iW, iP, iZ = 0, n, 2 * n, 3 * n
    m = 3 * n + 1
    eye = np.eye(n)
    M = np.zeros((m, m))
    L0 = np.zeros((m, m))
    L1 = np.zeros((m, m))
    kind = []
    r = 0
    for j in range(1, N):  # radial momentum at interior nodes
        M[r, iU + j] = 1.0
        L0[r, iP + j] = 1.0
        L0[r, iU:iU + n] += Oh * (D2[j] - eye[j])
        kind.append("mom")
        r += 1
    for j in range(1, N):  # axial momentum at interior nodes
        M[r, iW + j] = 1.0
        L0[r, iP:iP + n] = -D1[j]
        L0[r, iW:iW + n] += Oh * (D2[j] - eye[j])
        kind.append("mom")
        r += 1
    for j in range(n):  # continuity at every node
        L0[r, iU + j] = 1.0
        L0[r, iW:iW + n] = D1[j]
        kind.append("vcon")
        r += 1
    L0[r, iW + N] = 1.0  # W(-d) = 0
    kind.append("vcon")
    r += 1
    if bottom == "no-slip":
        L0[r, iU + N] = 1.0
    elif bottom == "free-slip":
        L0[r, iU:iU + n] = D1[N]
    else:
        raise ValueError(bottom)
    kind.append("vcon")
    r += 1
    L0[r, iU:iU + n] = D1[0]  # zero tangential stress at the surface
    L0[r, iW + 0] = -1.0
    kind.append("vcon")
    r += 1
    L0[r, iP + 0] = 1.0  # normal stress: P0 - 2 Oh W_z(0) - (1 + Bo + g) Z = 0
    L0[r, iW:iW + n] += -2.0 * Oh * D1[0]
    L0[r, iZ] = -(1.0 + Bo)
    L1[r, iZ] = -1.0
    kind.append("nstress")
    r += 1
    M[r, iZ] = 1.0  # kinematic: Z' = W(0)
    L0[r, iW + 0] = 1.0
    kind.append("kin")
    r += 1
    assert r == m
    return M, L0, L1, np.array(kind), (iU, iW, iP, iZ, n)


def reduced_ode(Oh, Bo, d, bottom, N):
    """Eliminate the velocity constraints and the pressure: ``y = [c; Z]``, ``y' = (A0 + g A1) y``."""
    M, L0, _, kind, (_, _, _, _, n) = descriptor(Oh, Bo, d, bottom, N)
    vcols = np.arange(0, 2 * n)
    pcols = np.arange(2 * n, 3 * n)
    Y = linalg.null_space(L0[kind == "vcon"][:, vcols])  # velocities v = Y c
    k = Y.shape[1]
    mom = np.where(kind == "mom")[0]
    Mv = M[np.ix_(mom, vcols)]
    Lvv = L0[np.ix_(mom, vcols)]
    Lvp = L0[np.ix_(mom, pcols)]
    ns = np.where(kind == "nstress")[0][0]
    aW = L0[ns, vcols]
    K = np.hstack([Mv @ Y, -Lvp[:, 1:]])
    R = np.linalg.inv(K)[:k]
    p0col = Lvp[:, 0]
    A0 = np.zeros((k + 1, k + 1))
    A1 = np.zeros((k + 1, k + 1))
    A0[:k, :k] = R @ (Lvv @ Y - np.outer(p0col, aW @ Y))
    A0[:k, k] = R @ p0col * (1.0 + Bo)
    A1[:k, k] = R @ p0col
    A0[k, :k] = Y[n + 0]
    return A0, A1


def normal_modes(Oh, Bo, d, bottom, N):
    """Eigenvalues of the Gamma = 0 reduced system (surface modes and bulk vortical modes)."""
    A0, _ = reduced_ode(Oh, Bo, d, bottom, N)
    return np.linalg.eigvals(A0)


def response(t, *, Oh, Bo, d, bottom="free-slip", Gamma=0.0, Omega=1.0, N=48,
             rtol=1e-10, atol=1e-13) -> np.ndarray:
    """Normalised elevation ``Z(t)`` for ``Z(0) = 1`` with the liquid initially at rest.

    ``t`` must be increasing and start at or after 0. ``N`` is the number of Chebyshev intervals in
    the layer; 48 resolves the boundary layers for ``Oh >= 0.01`` and ``d <= 2 pi``.
    """
    t = np.asarray(t, dtype=float)
    A0, A1 = reduced_ode(Oh, Bo, d, bottom, N)
    y0 = np.zeros(A0.shape[0])
    y0[-1] = 1.0

    def rhs(tt, y):
        return (A0 + Gamma * np.cos(Omega * tt) * A1) @ y

    def jac(tt, y):
        return A0 + Gamma * np.cos(Omega * tt) * A1

    sol = solve_ivp(rhs, (0.0, float(t[-1])), y0, method="Radau", jac=jac, t_eval=t,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"linear initial-value integration failed: {sol.message}")
    return sol.y[-1]


def response_bdf2(dt: float, steps: int, *, Oh, Bo, d, bottom="free-slip", Gamma=0.0,
                  Omega=1.0, N=48) -> tuple[np.ndarray, np.ndarray]:
    """``Z`` at ``t_n = n dt`` from the same time discretisation as the simulations.

    pyoomph starts from rest with one implicit Euler (BDF1) step and continues with constant-step
    BDF2, ``(3 y_{n+1} - 4 y_n + y_{n-1})/(2 dt) = A(t_{n+1}) y_{n+1}``. Applying the identical scheme
    to the linear system leaves only the spatial error in the difference between a simulation and
    this response, so mesh convergence can be measured separately from the time-step error.
    """
    A0, A1 = reduced_ode(Oh, Bo, d, bottom, N)
    m = A0.shape[0]
    eye = np.eye(m)
    y = np.zeros((steps + 1, m))
    y[0, -1] = 1.0
    for n in range(steps):
        A = A0 + Gamma * np.cos(Omega * (n + 1) * dt) * A1
        if n == 0:
            y[1] = np.linalg.solve(eye - dt * A, y[0])
        else:
            y[n + 1] = np.linalg.solve(3.0 * eye - 2.0 * dt * A, 4.0 * y[n] - y[n - 1])
    return dt * np.arange(steps + 1), y[:, -1]
