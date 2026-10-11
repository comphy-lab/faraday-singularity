"""High-precision (mpmath) closed forms of D(s) for audit and root polishing (needs ``mpmath``).

Free-slip:
    D(s) = (s + 2 Oh)^2 coth d - 4 Oh^2 q coth(q d),   q^2 = 1 + s/Oh.

No-slip, with xi = z + d, S = sinh d, C = cosh d,
Sq = sinh(q d), Cq = cosh(q d):
    w = A f(xi) + B g(xi),  f = q sinh xi - sinh(q xi),  g = cosh xi - cosh(q xi)
    (these satisfy w(0) = w'(0) = 0 at the bottom identically).  At xi = d:
    f = q S - Sq,                  g = C - Cq,
    Tf = f'' + f = 2 q S - (q^2 + 1) Sq,     Tg = g'' + g = 2 C - (q^2 + 1) Cq,
    Nf = (s+3Oh) f' - Oh f''' = q C (s + 2 Oh) - 2 Oh q Cq,
    Ng = (s+3Oh) g' - Oh g''' = S (s + 2 Oh) - 2 Oh q Sq,
    kinematic A f + B g = s, shear A Tf + B Tg = 0  =>
    D(s) = s (Tg Nf - Tf Ng) / (f Tg - g Tf).
"""

import mpmath as mp


def D_freeslip_mp(s, Oh, d):
    s, Oh, d = mp.mpc(s), mp.mpf(Oh), mp.mpf(d)
    q = mp.sqrt(1 + s / Oh)
    return (s + 2 * Oh) ** 2 * mp.coth(d) - 4 * Oh**2 * q * mp.coth(q * d)


def D_noslip_raw_mp(s, Oh, d):
    """Direct (unsimplified) no-slip formula.  The e^{2qd} terms cancel in the
    denominator, so the working precision is raised with Re(q) d."""
    s0 = mp.mpc(s)
    qd_est = abs(mp.re(mp.sqrt(1 + s0 / mp.mpf(Oh)))) * mp.mpf(d)
    extra = int(2 * qd_est / mp.log(10)) + 30
    with mp.extradps(extra):
        s, Oh_, d_ = mp.mpc(s), mp.mpf(Oh), mp.mpf(d)
        q = mp.sqrt(1 + s / Oh_)
        S, C = mp.sinh(d_), mp.cosh(d_)
        Sq, Cq = mp.sinh(q * d_), mp.cosh(q * d_)
        f = q * S - Sq
        g = C - Cq
        Tf = 2 * q * S - (q**2 + 1) * Sq
        Tg = 2 * C - (q**2 + 1) * Cq
        Nf = q * C * (s + 2 * Oh_) - 2 * Oh_ * q * Cq
        Ng = S * (s + 2 * Oh_) - 2 * Oh_ * q * Sq
        val = s * (Tg * Nf - Tf * Ng) / (f * Tg - g * Tf)
    return +val


def D_noslip_mp(s, Oh, d):
    """Simplified no-slip closed form, using
    f Tg - g Tf = (q^2 - 1)(C Sq - q S Cq) and Oh (q^2 + 1) = s + 2 Oh:
        D = [4 Oh q P/(C Cq) + 4 Oh^2 q (q T Tq - 1) + P^2 (T Tq - q)] / (Tq - q T),
    P = s + 2 Oh, T = tanh d, Tq = tanh(q d)."""
    s, Oh, d = mp.mpc(s), mp.mpf(Oh), mp.mpf(d)
    q = mp.sqrt(1 + s / Oh)
    P = s + 2 * Oh
    T, Tq = mp.tanh(d), mp.tanh(q * d)
    inv = 1 / (mp.cosh(d) * mp.cosh(q * d))
    num = 4 * Oh * q * P * inv + 4 * Oh**2 * q * (q * T * Tq - 1) + P**2 * (T * Tq - q)
    return num / (Tq - q * T)


def D_mp(s, Oh, d, bottom):
    return D_freeslip_mp(s, Oh, d) if bottom == "free-slip" else D_noslip_mp(s, Oh, d)


def normal_mode_mp(Oh, Bo, d, bottom, s_guess, dps=40):
    """Polish a normal-mode root of D(s) + 1 + Bo = 0 at dps decimal digits."""
    with mp.workdps(dps):
        def f(z):
            return D_mp(z, Oh, d, bottom) + 1 + mp.mpf(Bo)

        return mp.findroot(f, mp.mpc(s_guess), tol=mp.mpf(10) ** (-dps + 5))
