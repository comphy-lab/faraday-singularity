"""Quadratic tip sliver geometry and independent hard balances."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faraday_singularity import tipcontact as t


def test_terminal_dimple_from_a_saved_p2_cap():
    # Three terminal nodes, with the common axial origin subtracted.
    P = np.array([[5.857513118657744e-6, 1.1011388534321047e-6],
                  [2.850491864724829e-6, -8.166955556276889e-8], [0.0, 0.0]])
    reading = t.terminal_dimple(P, 30.0, 0.1)
    assert reading is not None and reading["index"] == 1
    assert reading["initial_tangent_z"] < 0 < reading["tangent_z"]
    assert reading["trigger"] == "tip-dimple" and not reading["fresh"]
    assert reading["turn"] == pytest.approx(44.909064764353296, abs=1e-7)
    assert t.swept_edge(P)["dV"] == pytest.approx(-4.3900889792508587e-17, rel=1e-8, abs=0)
    # Translation preserves the reading; a straight or off-axis cap does not qualify.
    assert t.terminal_dimple(P + [0, 4.67311000646152], 30.0, 0.1) is not None
    assert t.terminal_dimple(P, 50.0, 0.1) is None
    assert t.terminal_dimple(P, 30.0, 0.2) is None
    Q = P.copy()
    Q[1] = 0.5 * (Q[0] + Q[2])
    assert t.terminal_dimple(Q, 30.0, 0.1) is None
    assert t.terminal_dimple(P + [1e-6, 0], 30.0, 0.1) is None
    assert t.terminal_dimple(P[::-1], 30.0, 0.1) is None
    Q = P.copy()
    Q[:, 1] *= -1
    assert t.terminal_dimple(Q, 30.0, 0.1) is None


@pytest.mark.parametrize("angle", [50, 60])
def test_a_coarse_convex_circular_cap_is_not_an_annular_dip(angle):
    from faraday_singularity.reconnect import p2_edges

    theta = np.radians([angle, angle / 2, 0])
    P = np.column_stack([np.sin(theta), np.cos(theta)])
    edge = p2_edges(P)
    offset = np.linalg.norm(P[1] - 0.5 * (P[0] + P[2])) / edge["L"][0]
    # This convex cap met all three old predicates, despite rising to its axis end.
    end_tangent = 0.5 * (P[0, 1] - P[2, 1]) - 2 * (P[1, 1] - P[2, 1])
    assert end_tangent > 0 and edge["turn"][0] > 30 and offset > 0.1
    assert t.terminal_dimple(P, 30.0, 0.1) is None


def test_only_a_native_inversion_failure_can_admit_a_dimple():
    from faraday_singularity.driver import CollapseDriver

    m = {"input": True}
    fake = SimpleNamespace()
    other = {"reason": "step-failed", "error": "RuntimeError: OomphException"}
    assert CollapseDriver._tip_edge_straighten(fake, m, None, failed_step=other) == (m, None)
    inverted = dict(other, error="InvertedElementRemeshRequest: folded")
    stop = {"reason": "other"}
    assert CollapseDriver._tip_edge_straighten(fake, m, stop, failed_step=inverted) == (m, stop)
    fresh_calls = []
    ordinary = SimpleNamespace(
        _fresh_mesh=lambda: fresh_calls.append(True) or True,
        _ring=lambda: None, _jetcore=lambda: None,
    )
    stop = {"reason": "interface-unresolved", "trigger": "edge-turn",
            "chain": "bubble", "fresh": True}
    assert CollapseDriver._tip_edge_straighten(ordinary, m, stop, failed_step=inverted) == (m, stop)
    assert fresh_calls == [True]  # the ordinary fresh-edge eligibility path was used


def test_a_swept_quadratic_edge_has_the_exact_revolved_sliver_volume():
    # r=R+x, z=h(1-x^2), chord z=0: 2 pi integral r z dr = 8 pi R h/3.
    R, h = 2.0, 0.2
    q = t.swept_edge([[R - 1, 0], [R, h], [R + 1, 0]])
    expected = 8 * math.pi * R * h / 3
    assert q["dV"] == pytest.approx(-expected)
    assert q["V_quad"] == pytest.approx(expected) and np.all(q["weights"] > 0)
    reverse = t.swept_edge([[R + 1, 0], [R, h], [R - 1, 0]])
    assert reverse["dV"] == pytest.approx(expected)
    assert reverse["V_quad"] == pytest.approx(expected)


@pytest.mark.parametrize("miss", ["volume", "momentum", "sliver", "coverage", None])
def test_signed_tip_balances_have_independent_hard_failures(miss):
    sigma = SimpleNamespace(sigma_V=1e-5, sigma_Pz=1e-5)
    before = {"V": 2.0, "Pz": 3.0}
    after = {"V": 2.001, "Pz": 3.002}
    quad = {"V_quad": 0.001, "points": 15, "found": 15, "Pz": 0.0}
    if miss == "volume":
        after["V"] += 0.1
    if miss == "momentum":
        after["Pz"] += 0.1
    if miss == "sliver":
        quad["V_quad"] = 0.002
    if miss == "coverage":
        quad["found"] = 14
    c = t.closures(before, after, 0.001, 0.002, sigma, 0.01, 1.0, quad)
    key = {"volume": "T_V", "momentum": "T_P", "sliver": "T_quad", "coverage": "T_coverage"}
    if miss in key:
        assert key[miss] in c["hard"]
    else:
        assert not c["hard"]
    assert c["V_added"] == 0.001 and c["V_removed"] == 0
    c = t.closures(
        after,
        before,
        -0.001,
        -0.002,
        sigma,
        0.01,
        1.0,
        {"V_quad": 0.001, "points": 15, "found": 15, "Pz": 0.0},
    )
    assert c["V_removed"] == 0.001 and c["V_added"] == 0


def test_other_stops_never_enter_the_tip_operation():
    from faraday_singularity.driver import CollapseDriver

    for stop in (
        None,
        {"reason": "other"},
        {"reason": "interface-unresolved", "trigger": "edge-turn", "chain": "open", "fresh": True},
        {
            "reason": "interface-unresolved",
            "trigger": "edge-turn",
            "chain": "bubble",
            "fresh": False,
        },
    ):
        fake = SimpleNamespace(_fresh_mesh=lambda: True)
        m = {"input": True}
        assert CollapseDriver._tip_edge_straighten(fake, m, stop) == (m, stop)


@pytest.mark.parametrize("value,key", [(0.02, "T_sliver_V"), (0.002, "T_sliver_P")])
def test_sliver_volume_and_momentum_bounds_are_independent(value, key):
    sigma = SimpleNamespace(sigma_V=1.0, sigma_Pz=1.0)
    dV = value if key == "T_sliver_V" else 0.001
    dP = 0.0
    c = t.closures(
        {"V": 0.0, "Pz": 0.0},
        {"V": dV, "Pz": dP},
        dV,
        dP,
        sigma,
        0.01,
        0.1,
        {"V_quad": dV, "found": 15, "points": 15, "Pz": value if key == "T_sliver_P" else 0.0},
    )
    assert c["hard"] == [key]


def test_sliver_momentum_and_energy_match_a_constant_velocity(monkeypatch):
    from faraday_singularity import surgery

    sweep = t.swept_edge([[1.0, 0.0], [2.0, 0.2], [3.0, 0.0]])

    def constant(problem, r, z, fields):
        return {"velocity_x": np.full(len(r), 3.0), "velocity_y": np.full(len(r), 4.0)}, np.ones(
            len(r), dtype=bool
        )

    monkeypatch.setattr(surgery, "sample_fields", constant)
    q = t.sliver_momentum(None, sweep)
    assert q["Pz"] == pytest.approx(4 * sweep["V_quad"])
    assert q["KE"] == pytest.approx(12.5 * sweep["V_quad"])
    assert q["found"] == q["points"] == 15


@pytest.mark.parametrize(
    "refusal", ["index", "open", "two-edges", "identity", "noise", "recurrence-unqualified"]
)
def test_geometric_refusals_leave_the_state_and_outputs_untouched(tmp_path, refusal):
    from faraday_singularity.driver import CollapseDriver

    P = np.column_stack([1 + np.arange(9) * 0.01, np.arange(9) * 0.01])
    i = 5
    edge = {"r": P[i, 0], "z": P[i, 1]}
    loops = [
        {"open": False, "r": P[:, 0], "z": P[:, 1], "u_r": np.zeros(9), "u_z": np.zeros(9)},
        {"open": True, "r": np.array([10.0, 11.0, 12.0]), "z": np.array([1.0, 1.0, 1.0])},
    ]
    read = {"edges": [edge]}
    if refusal == "index":
        i = 1
    if refusal == "open":
        loops[1]["r"][1], loops[1]["z"][1] = P[i]
    if refusal == "two-edges":
        read["edges"] = [edge, edge]
    if refusal in ("identity", "recurrence-unqualified"):
        read["edges"] = [dict(r=0.0, z=0.0)]
    noise = 0.0 if refusal == "noise" else 1.0
    balances = [dict(V=0.0, Pz=0.0, KE=0.0, step=9)] if refusal == "recurrence-unqualified" else []
    fake = SimpleNamespace(
        _fresh_mesh=lambda: True,
        _ring=lambda: None,
        _jetcore=lambda: SimpleNamespace(far_kinks="recentre"),
        _axis_fold_of=lambda stop: None,
        _tip_edge_balances=balances,
        state=SimpleNamespace(step=10),
        pb=SimpleNamespace(
            interface_segments=lambda: loops, straighten_interface_edges=lambda *a, **kw: read
        ),
        events=SimpleNamespace(groove=SimpleNamespace(sigma_V=noise, sigma_Pz=noise)),
        outdir=tmp_path,
    )
    stop = dict(
        reason="interface-unresolved",
        trigger="edge-turn",
        chain="bubble",
        fresh=True,
        index=i,
        L=0.001,
        r=P[i, 0],
        z=P[i, 1],
    )
    before = P.copy()
    _, got = CollapseDriver._tip_edge_straighten(fake, {}, stop)
    assert got == stop if refusal != "noise" else "tip_edge_unavailable" in got
    np.testing.assert_array_equal(P, before)
    assert not list(tmp_path.iterdir())


def test_mesh_closure_refuses_a_no_op_and_a_wrong_signed_volume():
    sigma = SimpleNamespace(sigma_V=1.0, sigma_Pz=1.0)
    for actual in (0.0, -0.001):
        mesh = {
            "before": {"V": 1.0, "Pz": 1.0},
            "planned": {"V": 1.001, "Pz": 1.0},
            "after": {"V": 1.0 + actual, "Pz": 1.0},
        }
        q = {"V_quad": 0.001, "points": 15, "found": 15, "Pz": 0.0}
        c = t.closures(
            {"V": 1.0, "Pz": 1.0},
            {"V": 1.0 + actual, "Pz": 1.0},
            0.001,
            0.0,
            sigma,
            0.01,
            1.0,
            q,
            mesh=mesh,
        )
        assert "T_V_mesh" in c["hard"] and "T_V" not in c["hard"]


def triangle_fixture():
    local = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    s, z = local.T
    basis = np.column_stack([np.ones(6), s, z, s * s, s * z, z * z])
    positions = np.column_stack([2 + s + 0.25 * s * s, z + 0.25 * z * z])
    velocity = np.column_stack([s * s, z * z])
    return positions, velocity, np.linalg.inv(basis)


def test_curved_p2_triangle_integrals_match_exact_polynomial_moments():
    from fractions import Fraction as F
    from itertools import product

    def multiply(*terms):
        result = {(0, 0): F(1)}
        for term in terms:
            new = {}
            for (a, b), (c, d) in product(result, term):
                power = (a + c, b + d)
                new[power] = new.get(power, F(0)) + result[a, b] * term[c, d]
            result = new
        return result

    def integral(poly):
        return sum(
            value * F(math.factorial(a) * math.factorial(b), math.factorial(a + b + 2))
            for (a, b), value in poly.items()
        )

    r = {(0, 0): F(2), (1, 0): F(1), (2, 0): F(1, 4)}
    jac = {(0, 0): F(1), (1, 0): F(1, 2), (0, 1): F(1, 2), (1, 1): F(1, 4)}
    expected = {
        "V": 2 * math.pi * float(integral(multiply(r, jac))),
        "Pz": 2 * math.pi * float(integral(multiply(r, jac, {(0, 2): F(1)}))),
        "KE": math.pi * float(integral(multiply(r, jac, {(4, 0): F(1), (0, 4): F(1)}))),
    }
    got = t.triangle_budget(*triangle_fixture())
    for key, value in expected.items():
        assert abs(got[key] - value) <= 6**2 * math.ulp(value)


def test_patch_jacobian_maps_native_node_order_for_initial_and_planned_geometry():
    from faraday_singularity.reconnect import p2_jacobian_bounds

    P, _, inverse = triangle_fixture()
    planned = P.copy()
    planned[3, 1] += 0.2
    permutation = np.array([4, 2, 0, 5, 3, 1])
    patch = {"positions": P[permutation], "planned_positions": planned[permutation],
             "inverse": np.linalg.inv(np.linalg.inv(inverse)[permutation])}
    for positions, flags in ((P, {"initial": True}), (planned, {"planned": True})):
        expected = p2_jacobian_bounds(positions, np.arange(6)[None, :])[0]
        assert t.patch_jacobian_bound(patch, **flags) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("bad", ["negative-radius", "inverted", "zero-area"])
def test_triangle_budget_refuses_invalid_geometry(bad):
    P, U, inverse = triangle_fixture()
    if bad == "negative-radius":
        P[:, 0] *= -1
    elif bad == "zero-area":
        P[:, 1] = 0
    else:
        P[3, 0] = 100
    with pytest.raises(RuntimeError, match="inverted triangle or a negative radius"):
        t.triangle_budget(P, U, inverse)


@pytest.mark.parametrize("bad", ["no-face", "two-faces", "not-p2", "missing-node"])
def test_edge_patch_refuses_unsupported_native_readings(bad):
    point = (1.0, 0.5)
    node = SimpleNamespace(x=lambda k: point[k])
    bulk = SimpleNamespace(
        nnode=lambda: 3 if bad == "not-p2" else 6,
        dim=lambda: 2,
        node_pt=lambda k: SimpleNamespace(x=lambda j: 0.0),
    )
    face = SimpleNamespace(nnode=lambda: 3, node_pt=lambda k: node, get_bulk_element=lambda: bulk)
    faces = [] if bad == "no-face" else [face, face] if bad == "two-faces" else [face]
    pb = SimpleNamespace(get_mesh=lambda name: SimpleNamespace(elements=lambda: faces))
    with pytest.raises(RuntimeError):
        t.edge_patch(pb, point, point, point)


@pytest.mark.parametrize("bad", ["touching", "polygon", "spline", "jacobian"])
@pytest.mark.parametrize("fresh", [False, True])
def test_the_pure_guard_reads_topology_and_jacobian_failures(monkeypatch, bad, fresh):
    from faraday_singularity import reconnect, surgery, driver
    from faraday_singularity.driver import CollapseDriver

    A = reconnect.Chain(np.array([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0]]), ("axis", "fixed"))
    B = reconnect.Chain(np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]]), ("axis", "axis"))
    if bad == "touching":
        B = reconnect.Chain(A.points.copy(), ("axis", "axis"))
    elif bad == "polygon":
        B = reconnect.Chain(
            np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), ("axis", "axis")
        )
    monkeypatch.setattr(surgery, "_named_chains", lambda pb: ([("open", A), ("bubble", B)], {}))
    monkeypatch.setattr(
        reconnect,
        "spline_contact_stop",
        lambda named: (
            {"reason": "self-contact-approach", "trigger": "spline-contact"}
            if bad == "spline"
            else None
        ),
    )
    monkeypatch.setattr(driver, "interface_guard", lambda *a, **kw: {"stop": None})
    fake = SimpleNamespace(
        pb=SimpleNamespace(
            curved_jacobian_bound=lambda limit: {"min": 0.1 if bad == "jacobian" else 1.0}
        ),
        reconnect=SimpleNamespace(rebuild_spline="nodes", jacobian_min=0.2),
        _fresh_mesh=lambda: fresh,
        _h_floor=lambda: 0.01,
        s=SimpleNamespace(refine_ratio=1.5),
    )
    got = CollapseDriver._tip_edge_guard(fake)
    if not fresh and bad == "jacobian":
        assert got is None  # the whole-mesh bound belongs to fresh-mesh admission
        return
    assert got is not None
    assert got["trigger"] == (
        "spline-contact"
        if bad == "spline"
        else "curved-jacobian"
        if bad == "jacobian"
        else "tip-topology"
    )


def test_guard_uses_snapped_named_chains_instead_of_raw_edge_readings(monkeypatch):
    from faraday_singularity import reconnect, surgery, driver
    from faraday_singularity.driver import CollapseDriver

    A = reconnect.Chain(np.array([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0]]), ("axis", "fixed"))
    B = reconnect.Chain(np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]]), ("axis", "axis"))
    named = [("open", A), ("bubble", B)]
    monkeypatch.setattr(surgery, "_named_chains", lambda pb: (named, {}))
    monkeypatch.setattr(reconnect, "spline_contact_stop", lambda named: None)
    seen = []

    def guard(readings, *a, **kw):
        seen.append(readings)
        return {"stop": None}

    monkeypatch.setattr(driver, "interface_guard", guard)
    fake = SimpleNamespace(
        pb=SimpleNamespace(curved_jacobian_bound=lambda limit: {"min": 1.0}),
        reconnect=SimpleNamespace(rebuild_spline="nodes", jacobian_min=0.2),
        _fresh_mesh=lambda: True,
        _h_floor=lambda: 0.01,
        s=SimpleNamespace(refine_ratio=1.5),
        _post_cut_state=lambda: pytest.fail("raw edge readings must not be used"),
    )
    assert CollapseDriver._tip_edge_guard(fake) is None
    for name, chain in named:
        expected = reconnect.edge_reading(chain.points)
        for key in ("turn", "turn_r", "turn_z", "turn_L"):
            assert seen[0][name][key] == expected[key]
