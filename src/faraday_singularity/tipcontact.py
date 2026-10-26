"""Budgets of straightening a curved edge at a nuisance tip, without a rebuild."""

from __future__ import annotations

import math

import numpy as np


def terminal_dimple(points, turn_limit: float, offset_limit: float) -> dict | None:
    """Read an under-resolved annular dip on the downward tip's final P2 edge.

    The chain ends on the axis, walking towards the downward tip. Opposite
    axial tangent signs prove an interior minimum of the quadratic edge.
    Require the existing refinement turn and midside-offset thresholds too.
    """
    from .reconnect import p2_edges
    from .axisrepair import axis_fold

    P = np.asarray(points, dtype=float)
    if len(P) < 3 or len(P) % 2 != 1 or P[-1, 0] != 0:
        return None
    if axis_fold(P, "last") is not None:
        return None
    Q = P[-3:]
    if not np.isfinite(Q).all() or np.any(Q[:-1, 0] <= 0):
        return None
    tangent_z = 0.5 * (Q[0, 1] - Q[2, 1]) - 2 * (Q[1, 1] - Q[2, 1])
    initial_tangent_z = 2 * (Q[1, 1] - Q[0, 1]) - 0.5 * (Q[2, 1] - Q[0, 1])
    e = p2_edges(Q)
    L, turn = float(e["L"][0]), float(e["turn"][0])
    offset = float(np.linalg.norm(Q[1] - 0.5 * (Q[0] + Q[2]))) / L if L > 0 else 0
    if not (initial_tangent_z < 0 < tangent_z and turn > turn_limit and offset > offset_limit):
        return None
    return {"reason": "interface-unresolved", "trigger": "tip-dimple", "chain": "bubble",
            "index": len(P) - 2, "r": float(Q[1, 0]), "z": float(Q[1, 1]),
            "L": L, "turn": turn, "limit": turn_limit, "tangent_z": float(tangent_z),
            "initial_tangent_z": float(initial_tangent_z),
            "offset": offset, "fresh": False}


def swept_edge(points) -> dict:
    """The signed liquid change and positive quadrature of a quadratic edge's sliver.

    The chain bounds the gas counter-clockwise. Its replacement is the chord;
    minus the change in pi integral r^2 dz is the liquid gained. The positive
    tensor Gauss rule follows the strip from curve to chord, with 2 pi r.
    """
    P = np.asarray(points, dtype=float)
    x, w = np.polynomial.legendre.leggauss(5)
    s, ws = np.polynomial.legendre.leggauss(3)
    s, ws = (s + 1) / 2, ws / 2
    N = np.column_stack([x * (x - 1) / 2, 1 - x * x, x * (x + 1) / 2])
    dN = np.column_stack([x - 0.5, -2 * x, x + 0.5])
    chord = P.copy()
    chord[1] = 0.5 * (P[0] + P[2])
    # Translate z before the exact boundary integral to avoid cancellation.
    local = P - P[0] * [0, 1]
    local_chord = chord - P[0] * [0, 1]
    old, new = N @ local, N @ local_chord
    tangent, new_tangent = dN @ local, dN @ local_chord
    dV = -math.pi * math.fsum(
        w * (new[:, 0] ** 2 * new_tangent[:, 1] - old[:, 0] ** 2 * tangent[:, 1])
    )
    delta = new - old
    X = old[:, None, :] + s[None, :, None] * delta[:, None, :]
    J = np.abs(tangent[:, 0] * delta[:, 1] - tangent[:, 1] * delta[:, 0])
    weights = (2 * math.pi * X[:, :, 0] * J[:, None] * w[:, None] * ws[None, :]).ravel()
    X[:, :, 1] += P[0, 1]
    return {"dV": dV, "points": X.reshape(-1, 2), "weights": weights, "V_quad": math.fsum(weights)}


def sliver_momentum(problem, sweep) -> dict:
    """Read the liquid on the sliver itself; missing points are a hard refusal."""
    from .surgery import sample_fields

    X, weights = sweep["points"], sweep["weights"]
    values, found = sample_fields(problem, X[:, 0], X[:, 1], ("velocity_x", "velocity_y"))
    return {
        "Pz": math.fsum(weights * values["velocity_y"]) if found.all() else math.nan,
        "KE": math.fsum(0.5 * weights * (values["velocity_x"] ** 2 + values["velocity_y"] ** 2))
        if found.all()
        else math.nan,
        "found": int(found.sum()),
        "points": len(found),
        "V_quad": sweep["V_quad"],
    }


def closures(before, after, dV, dP, sigma, bound, speed, quadrature, *, mesh=None) -> dict:
    """Hard signed liquid closures, independent sliver bounds and coverage.

    Existing five-noise floors of the ring/axis repairs bound the local box
    changes. The sliver's volume preflight is redundant with admission; its
    momentum and the full affected-element momentum change are bounded. The
    native element's volume and planned momentum are independent hard checks
    at the existing R1/R2 relative limits, without a rebuild-noise floor.
    """
    from .ringsurgery import RING_BUDGET_LIMITS, RING_NOISE_FLOORS

    values = {
        "T_V": abs(after["V"] - before["V"] - dV),
        "T_P": abs(after["Pz"] - before["Pz"] - dP),
        "T_sliver_V": abs(dV),
        "T_sliver_P": abs(quadrature["Pz"]),
        "T_patch_P": abs(dP),
        "T_quad": abs(quadrature["V_quad"] - abs(dV)),
    }
    limits = {
        "T_V": RING_NOISE_FLOORS["R1"][1] * sigma.sigma_V,
        "T_P": RING_NOISE_FLOORS["R2"][1] * sigma.sigma_Pz,
        "T_sliver_V": bound,
        "T_sliver_P": bound * speed,
        "T_patch_P": bound * speed,
        "T_quad": RING_BUDGET_LIMITS["R1"] * abs(dV),
    }
    if mesh is not None:
        b, a, planned = (mesh[k] for k in ("before", "after", "planned"))
        values.update(
            T_V_mesh=abs(a["V"] - b["V"] - dV),
            T_V_plan=abs(planned["V"] - b["V"] - dV),
            T_P_mesh=abs(a["Pz"] - planned["Pz"]),
        )
        limits.update(
            T_V_mesh=RING_BUDGET_LIMITS["R1"] * abs(dV),
            T_V_plan=RING_BUDGET_LIMITS["R1"] * abs(dV),
            T_P_mesh=RING_BUDGET_LIMITS["R2"] * abs(planned["Pz"] - b["Pz"]),
        )
    hard = [k for k, v in values.items() if not math.isfinite(v) or v > limits[k]]
    if quadrature["found"] != quadrature["points"]:
        hard.append("T_coverage")
    return {
        "values": values,
        "limits": limits,
        "hard": hard,
        "V_added": max(dV, 0.0),
        "V_removed": max(-dV, 0.0),
        "Pz_liquid_change": dP,
    }


def edge_patch(problem, point, chord, velocity):
    """The native P2 triangle sharing the changed boundary midside node.

    This mesh uses conforming six-node triangles; a boundary midside node is
    shared only with its one bulk triangle. Unsupported elements are refused.
    Local node coordinates come from the pinned native element, not an assumed
    node order. No other node or element is modified by the straightening.
    """
    surface = []
    for face in problem.get_mesh("liquid/interface").elements():
        if face.nnode() == 3 and (face.node_pt(1).x(0), face.node_pt(1).x(1)) == tuple(point):
            surface.append(face)
    if len(surface) != 1:
        raise RuntimeError("the tip edge has no unique native boundary element")
    element = surface[0].get_bulk_element()
    if element.nnode() != 6 or element.dim() != 2:
        raise RuntimeError("the tip audit requires a conforming P2 triangle")
    nodes = [element.node_pt(k) for k in range(6)]
    selected = [k for k, n in enumerate(nodes) if (n.x(0), n.x(1)) == tuple(point)]
    if len(selected) != 1:
        raise RuntimeError("the changed midside node is not unique in its bulk element")
    fields = problem.get_mesh("liquid").get_nodal_field_indices()
    indices = [fields[k] for k in ("velocity_x", "velocity_y")]
    positions = np.array([[n.x(0), n.x(1)] for n in nodes])
    values = np.array([[n.value(k) for k in indices] for n in nodes])
    local = np.array([element.local_coordinate_of_node(k) for k in range(6)])

    def basis(X):
        return np.column_stack(
            [np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2]
        )

    inverse = np.linalg.inv(basis(local))
    planned_positions, planned_values = positions.copy(), values.copy()
    planned_positions[selected[0]] = chord
    planned_values[selected[0]] = velocity
    return {
        "nodes": nodes,
        "fields": indices,
        "inverse": inverse,
        "positions": positions,
        "values": values,
        "planned_positions": planned_positions,
        "planned_values": planned_values,
    }


def triangle_budget(positions, velocity, inverse, order=6):
    """Positive Duffy/Gauss integration of the P2 element's V, Pz and KE.

    Six points per direction integrate the degree-eight axisymmetric kinetic
    energy density exactly. Coordinates are translated before differentiation
    so the tip's tiny element is not subtracted from the domain height.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x + 1) / 2, w / 2
    S, T = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w) * (1 - S)
    S, T, W = S.ravel(), ((1 - S) * T).ravel(), W.ravel()
    basis = np.column_stack([np.ones(len(S)), S, T, S * S, S * T, T * T])
    ds = np.column_stack(
        [np.zeros(len(S)), np.ones(len(S)), np.zeros(len(S)), 2 * S, T, np.zeros(len(S))]
    )
    dt = np.column_stack(
        [np.zeros(len(S)), np.zeros(len(S)), np.ones(len(S)), np.zeros(len(S)), S, 2 * T]
    )
    origin = positions[0]
    coefficients = inverse @ (positions - origin)
    X = basis @ coefficients + origin
    A, B = ds @ coefficients, dt @ coefficients
    det = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
    if not (np.all(det > 0) or np.all(det < 0)) or np.any(X[:, 0] <= 0):
        raise RuntimeError("the tip audit found an inverted triangle or a negative radius")
    weights = 2 * math.pi * X[:, 0] * np.abs(det) * W
    U = basis @ (inverse @ velocity)
    return {
        "V": math.fsum(weights),
        "Pz": math.fsum(weights * U[:, 1]),
        "KE": math.fsum(0.5 * weights * np.sum(U * U, axis=1)),
    }


def patch_budget(patch, *, planned=False, initial=False):
    """Integrate initial, planned or actual native coordinates and velocities."""
    if planned or initial:
        prefix = "planned_" if planned else ""
        P, U = patch[prefix + "positions"], patch[prefix + "values"]
    else:
        P = np.array([[n.x(0), n.x(1)] for n in patch["nodes"]])
        U = np.array([[n.value(k) for k in patch["fields"]] for n in patch["nodes"]])
    return triangle_budget(P, U, patch["inverse"])
