"""Exact linear viscous theory of the J0 free-surface mode: normal modes and Faraday Floquet analysis.

Dimensionless variables: length 1/k, time t_c = sqrt(rho/(sigma k^3)), velocity
sqrt(sigma k/rho), pressure sigma k.  Hence the lateral wavenumber is 1, the
surface tension is 1 and the kinematic viscosity is Oh.

Geometry and model
------------------
Liquid in -d < z < 0 at rest, free surface z = zeta(r, t), lateral structure
J0(r) (axisymmetric) or cos(x) (planar); both have horizontal Laplacian
eigenvalue -1, so the z-problem is identical.  Container-frame momentum

    du/dt = -grad p + Oh lap u - G(t) e_z,     G(t) = Bo + Gamma cos(Omega t).

Eliminating pressure, the vertical velocity w(z) e^{st} J0(r) satisfies

    (s - Oh (d_z^2 - 1)) (d_z^2 - 1) w = 0,
    =>  w in span{ e^{z}, e^{-z}, e^{qz}, e^{-qz} },  q^2 = 1 + s/Oh, Re q >= 0.

Free surface z = 0 (linearised):
    kinematic           w(0) = s zeta
    zero shear stress   w''(0) + w(0) = 0
    normal stress       (s + 3 Oh) w'(0) - Oh w'''(0) = -(1 + G(t)) zeta
Bottom z = -d:
    w = 0 and  free-slip: w'' = 0,   or   no-slip: w' = 0.

Define D(s) zeta := (s + 3 Oh) w'(0) - Oh w'''(0) where w solves the two bottom
conditions plus the kinematic and shear conditions for unit zeta.  Then
    normal modes (Gamma = 0):   D(s) + 1 + Bo = 0,
    Floquet/Hill system:        (D(s_n) + 1 + Bo) zeta_n
                                + (Gamma/2) (zeta_{n-1} + zeta_{n+1}) = 0,
with s_n = sigma + i n Omega (Floquet exponent sigma).

Closed form (free-slip):  D(s) = (s + 2Oh)^2 coth d - 4 Oh^2 q coth(q d).
Closed form (no-slip): see reference_mp.D_noslip_mp.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

BOTTOMS = ("free-slip", "no-slip")


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def omega0(Bo: float, d: float) -> float:
    """Inviscid angular frequency sqrt((1 + Bo) tanh d) of the k = 1 mode."""
    return float(np.sqrt((1.0 + Bo) * np.tanh(d)))


def _coth(x):
    """Numerically stable complex coth for arrays (no overflow for large |Re x|)."""
    x = np.asarray(x, dtype=complex)
    sgn = np.where(x.real < 0, -1.0, 1.0)
    xa = x * sgn
    e = np.exp(-2.0 * xa)
    return sgn * (1.0 + e) / (1.0 - e)


def D_free_slip_closed(s, Oh, d):
    """Closed-form D(s) for the free-slip bottom (module docstring)."""
    s = np.asarray(s, dtype=complex)
    q = np.sqrt(1.0 + s / Oh)
    cothd = 1.0 if np.isinf(d) else 1.0 / np.tanh(d)
    cothqd = np.ones_like(q) if np.isinf(d) else _coth(q * d)
    return (s + 2 * Oh) ** 2 * cothd - 4 * Oh**2 * q * cothqd


def D_fun(s, Oh, d, bottom="free-slip"):
    """D(s) from a 4x4 solve in the bounded exponential basis.

    Basis on -d <= z <= 0 (all functions bounded by 1 when Re q >= 0):
        phi1 = e^{z}, phi2 = e^{-(z+d)}, phi3 = e^{q z}, phi4 = e^{-q(z+d)}.
    Rows: w(-d) = 0; bottom condition (w''(-d) = 0 free-slip, w'(-d) = 0
    no-slip); w''(0) + w(0) = 0; w(0) = s (unit zeta).
    Returns D = (s + 3 Oh) w'(0) - Oh w'''(0).  Valid for both bottoms and
    all s except the degenerate point q = 1 (s = 0), where D(0) = 0 exactly
    (static deformation, w = 0) and is returned as such.
    """
    if bottom not in BOTTOMS:
        raise ValueError(f"bottom must be one of {BOTTOMS}")
    s_in = np.asarray(s, dtype=complex)
    s = np.atleast_1d(s_in).ravel()
    out = np.zeros(s.shape, dtype=complex)
    zero = np.abs(s) < 1e-300
    sz = s[~zero]
    if sz.size:
        q = np.sqrt(1.0 + sz / Oh)
        if np.isinf(d):
            Ed = 0.0
            Eq = np.zeros_like(q)
        else:
            Ed = np.exp(-d)
            Eq = np.exp(-q * d)
        n = sz.size
        M = np.zeros((n, 4, 4), dtype=complex)
        M[:, 0, :] = np.stack([np.full(n, Ed), np.ones(n), Eq, np.ones(n)], -1)
        if bottom == "no-slip":
            M[:, 1, :] = np.stack([np.full(n, Ed), -np.ones(n), q * Eq, -q], -1)
        else:
            M[:, 1, :] = np.stack([np.full(n, Ed), np.ones(n), q**2 * Eq, q**2], -1)
        M[:, 2, :] = np.stack(
            [np.full(n, 2.0), np.full(n, 2 * Ed), q**2 + 1, (q**2 + 1) * Eq], -1
        )
        M[:, 3, :] = np.stack([np.ones(n), np.full(n, Ed), np.ones(n), Eq], -1)
        rhs = np.zeros((n, 4, 1), dtype=complex)
        rhs[:, 3, 0] = sz
        c = np.linalg.solve(M, rhs)[..., 0]
        w1 = c[:, 0] - c[:, 1] * Ed + q * c[:, 2] - q * c[:, 3] * Eq
        w3 = c[:, 0] - c[:, 1] * Ed + q**3 * c[:, 2] - q**3 * c[:, 3] * Eq
        out[~zero] = (sz + 3 * Oh) * w1 - Oh * w3
    return out.reshape(s_in.shape) if s_in.ndim else out[0]


def dispersion(s, Oh, Bo, d, bottom="free-slip"):
    """F(s) = D(s) + 1 + Bo; normal modes are zeros of F."""
    return D_fun(s, Oh, d, bottom) + 1.0 + Bo


# ---------------------------------------------------------------------------
# Complex Newton with holomorphic finite-difference derivative
# ---------------------------------------------------------------------------
def _newton_complex(f, z0, tol=1e-15, maxit=100):
    z = complex(z0)
    for _ in range(maxit):
        fz = f(z)
        h = 1e-6 * max(1.0, abs(z))
        dfz = (f(z + h) - f(z - h)) / (2 * h)
        dz = fz / dfz
        z -= dz
        if abs(dz) <= tol * max(1.0, abs(z)):
            return z, True
    return z, False


# ---------------------------------------------------------------------------
# Deliverable 1: normal mode
# ---------------------------------------------------------------------------
def normal_mode(Oh, Bo=0.0, d=np.inf, bottom="free-slip", Oh_start=1e-9, steps_per_decade=40):
    """Complex growth rate s of the fundamental free-surface mode (Gamma = 0).

    Root of D(s) + 1 + Bo = 0 obtained by continuation in Oh from Oh_start
    (where s ~ i omega0) to the requested Oh, so the returned root is the one
    continuous with the inviscid s = i sqrt((1 + Bo) tanh d), Im s > 0.
    """
    w0 = omega0(Bo, d)
    if Oh <= Oh_start:
        path = [Oh]
    else:
        nstep = int(np.ceil(np.log10(Oh / Oh_start) * steps_per_decade))
        path = list(np.geomspace(Oh_start, Oh, nstep + 1))
    hist = []  # (sqrt(Oh), s)
    s = 1j * w0
    for o in path:
        if len(hist) >= 2:  # linear extrapolation in sqrt(Oh)
            (x1, s1), (x2, s2) = hist[-2], hist[-1]
            s = s2 + (s2 - s1) * (np.sqrt(o) - x2) / (x2 - x1)
        elif len(hist) == 1:
            s = hist[-1][1] - 2 * (o - hist[-1][0] ** 2)
        else:
            s = 1j * w0 - 2 * o
        s, ok = _newton_complex(lambda z: dispersion(z, o, Bo, d, bottom), s)
        if not ok:
            raise RuntimeError(f"normal_mode Newton failed at Oh={o}")
        hist.append((np.sqrt(o), s))
    if not (s.imag > 0):
        raise RuntimeError("continued root is not oscillatory (Im s <= 0)")
    return s


# ---------------------------------------------------------------------------
# Hill (Kumar & Tuckerman) matrices, real-mu formulation
# ---------------------------------------------------------------------------
def _realify_diag(a):
    """Real 2N x 2N matrix of multiplication by diag(a) acting on [Re; Im]."""
    P, R = np.diag(a.real), np.diag(a.imag)
    return np.block([[P, -R], [R, P]])


def hill_sub_operator(mu, Omega, N, Oh, Bo, d, bottom, Dfun=None):
    """Real matrix C with C v = (1/Gamma) v for the subharmonic (alpha = 1) family.

    Unknowns zeta_n, n = 0..N-1, with reality zeta_{-n-1} = conj(zeta_n) and
    truncation zeta_N = 0; v = [Re zeta_0..N-1, Im zeta_0..N-1].
    Equations: a_n zeta_n = -(Gamma/2)(zeta_{n-1} + zeta_{n+1}),
    a_n = D(mu + i(n + 1/2)Omega) + 1 + Bo.
    """
    Dfun = Dfun or (lambda s: D_fun(s, Oh, d, bottom))
    n = np.arange(N)
    s = mu + 1j * (n + 0.5) * Omega
    a = Dfun(s) + 1.0 + Bo
    Ainv = _realify_diag(1.0 / a)
    B = np.zeros((2 * N, 2 * N))
    for k in range(N):
        for j in (k - 1, k + 1):
            if 0 <= j < N:
                B[k, j] -= 0.5
                B[N + k, N + j] -= 0.5
    B[0, 0] -= 0.5  # zeta_{-1} = conj(zeta_0): Re part +x0
    B[N, N] += 0.5  # Im part -y0
    return Ainv @ B


def hill_harm_operator(mu, Omega, N, Oh, Bo, d, bottom, Dfun=None):
    """Real matrix C with C v = (1/Gamma) v for the harmonic (alpha = 0) family.

    Unknowns zeta_n, n = 0..N, zeta_{-n} = conj(zeta_n) (so zeta_0 real for
    real mu), truncation zeta_{N+1} = 0; v = [Re zeta_0..N, Im zeta_1..N].
    """
    Dfun = Dfun or (lambda s: D_fun(s, Oh, d, bottom))
    n = np.arange(N + 1)
    s = mu + 1j * n * Omega
    a = Dfun(s) + 1.0 + Bo
    a0 = a[0].real
    m = 2 * N + 1
    Ainv = np.zeros((m, m))
    Ainv[0, 0] = 1.0 / a0
    ai = 1.0 / a[1:]
    # x_n rows 1..N (indices 1..N), y_n rows (indices N+1..2N)
    for k in range(1, N + 1):
        ix, iy = k, N + k
        Ainv[ix, ix] = ai[k - 1].real
        Ainv[ix, iy] = -ai[k - 1].imag
        Ainv[iy, ix] = ai[k - 1].imag
        Ainv[iy, iy] = ai[k - 1].real
    B = np.zeros((m, m))
    B[0, 1] = -1.0  # a0 zeta_0 = -(Gamma/2)(zeta_{-1}+zeta_1) = -Gamma Re zeta_1
    for k in range(1, N + 1):
        for j in (k - 1, k + 1):
            if 0 <= j <= N:
                B[k, j] -= 0.5  # real parts
                if j >= 1:
                    B[N + k, N + j] -= 0.5  # imaginary parts (Im zeta_0 = 0)
    return Ainv @ B


def gamma_eigs(mu, Omega, N, Oh, Bo, d, bottom, alpha=1, Dfun=None):
    """Positive real Gamma eigenvalues (sorted) of the real-mu Hill problem."""
    op = hill_sub_operator if alpha == 1 else hill_harm_operator
    lam = np.linalg.eigvals(op(mu, Omega, N, Oh, Bo, d, bottom, Dfun))
    lam = lam[(lam.imag == 0.0) & (lam.real > 0)].real
    return np.sort(1.0 / lam)


def gamma_c(Oh, Bo, d, bottom, Omega, alpha=1, N=20, mu=0.0, Dfun=None):
    """Deliverable 2a: neutral (mu = 0) forcing amplitude of the first tongue."""
    g = gamma_eigs(mu, Omega, N, Oh, Bo, d, bottom, alpha, Dfun)
    return g[0] if g.size else np.inf


# ---------------------------------------------------------------------------
# Deliverable 2b: mu(Gamma) for the subharmonic family
# ---------------------------------------------------------------------------
def gamma_of_mu(mu, Omega, N, Oh, Bo, d, bottom, Dfun=None):
    return gamma_c(Oh, Bo, d, bottom, Omega, 1, N, mu, Dfun)


def fold_point(Omega, N, Oh, Bo, d, bottom, s_guess, Dfun=None):
    """Minimum of Gamma(mu) over real mu: the multiplier-collision point.

    Below Gamma_* the least-damped surface-mode Floquet exponents are a complex
    pair not locked to Omega/2; above it they are real-mu subharmonic.
    """
    def f(m):
        return gamma_of_mu(m, Omega, N, Oh, Bo, d, bottom, Dfun)

    span = max(abs(s_guess.real), 1e-3)
    res = optimize.minimize_scalar(
        f, bounds=(s_guess.real - 2 * span, s_guess.real + 2 * span),
        method="bounded", options={"xatol": 1e-13},
    )
    return res.x, res.fun


def mu_KT(Gamma, Omega, N, Oh, Bo, d, bottom, fold=None, s_guess=None, Dfun=None):
    """Largest real-mu subharmonic exponent by inverting Gamma(mu) (KT approach).

    Returns None when Gamma < Gamma_* (no real-mu subharmonic solution).
    """
    if fold is None:
        fold = fold_point(Omega, N, Oh, Bo, d, bottom, s_guess, Dfun)
    mu_s, G_s = fold
    if Gamma < G_s:
        return None
    def f(m):
        return gamma_of_mu(m, Omega, N, Oh, Bo, d, bottom, Dfun) - Gamma

    hi = mu_s + 0.05
    while f(hi) < 0:
        hi = mu_s + 2 * (hi - mu_s)
    if f(mu_s) >= 0:
        return mu_s
    return optimize.brentq(f, mu_s, hi, xtol=1e-15, rtol=1e-15, maxiter=500)


def hill_det(sigma, Gamma, Omega, N, Oh, Bo, d, bottom, Dfun=None):
    """Normalised Hill determinant for a general complex Floquet exponent sigma.

    n = -N..N, s_n = sigma + i n Omega.  Columns n not in {0, -1} are scaled
    by 1/a_n, so the result is analytic near sigma ~ i Omega/2 and of O(1).
    """
    Dfun = Dfun or (lambda s: D_fun(s, Oh, d, bottom))
    n = np.arange(-N, N + 1)
    a = Dfun(sigma + 1j * n * Omega) + 1.0 + Bo
    H = np.diag(a) + (Gamma / 2) * (np.eye(2 * N + 1, k=1) + np.eye(2 * N + 1, k=-1))
    sc = a.copy()
    sc[(n == 0) | (n == -1)] = 1.0
    return np.linalg.det(H / sc[None, :])


def floquet_track(Gammas, Omega, N, Oh, Bo, d, bottom, s0, dG=2e-3, Dfun=None):
    """Least-damped surface-mode Floquet exponent sigma(Gamma), complex.

    Continuation in Gamma from sigma = s0 (normal mode) at Gamma = 0.  At each
    step Newton is started from the extrapolated root and from its mirror
    conj(sigma) + i Omega; the root with the larger real part is kept.  Works
    on both sides of the collision point Gamma_*.
    """
    Gammas = np.asarray(Gammas, float)
    gmax = Gammas.max()
    grid = np.unique(np.concatenate([np.arange(0, gmax + dG, dG), Gammas]))
    grid = grid[grid <= gmax + 1e-15]
    def f(z, G):
        return hill_det(z, G, Omega, N, Oh, Bo, d, bottom, Dfun)

    out = {}
    hist = []
    sig = complex(s0)
    for G in grid:
        if len(hist) >= 2:
            (G1, z1), (G2, z2) = hist[-2], hist[-1]
            pred = z2 + (z2 - z1) * (G - G2) / (G2 - G1) if G2 != G1 else z2
        else:
            pred = sig
        cands = []
        for z0 in (pred, np.conj(pred) + 1j * Omega, pred + 1e-3, pred - 1e-3):
            z, ok = _newton_complex(lambda z: f(z, G), z0, tol=1e-13)
            if ok and abs(z.imag - Omega / 2) < Omega / 4:
                cands.append(z)
        if not cands:
            raise RuntimeError(f"floquet_track failed at Gamma={G}")
        sig = max(cands, key=lambda z: (z.real, -abs(z - pred)))
        hist.append((G, sig))
        out[G] = sig
    return np.array([out[g] for g in Gammas])
