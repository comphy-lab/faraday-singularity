"""The ``ring`` part of the events block (``events.RingEventSettings``) and the trigger of the
piercing request it sets, without a finite-element problem.

* the part's validation at each bound and each enumerated value; the block without it, at its
  defaults, the same to the byte and the hash as at c4620f6 (and so as every recorded block and
  dump), and with it written into the block, round-tripped, its unknown keys refused;
* the keys it changes against a record without it, the refusals of a restart (none from the state
  before an event, ``ring.rho`` after a ring event from a rotating dump, ``ring.after_removal``
  never, a key of the leg), the events of its class, the keys that size the mesh;
* the piercing request: without the part at ``bubble_length < delta`` as before, its fields those
  of c4620f6; with it at ``max(ring.gap delta, 3 h_min)``, with the gap in the request.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import events as ev  # noqa: E402
from faraday_singularity import reconnect as rc  # noqa: E402


def test_the_ring_part_is_validated_at_each_bound_and_value():
    ok = ev.RingEventSettings
    assert ok() == ok(enabled=False, gap=1.0, rho=0.5, waist=0.5, cut_elements=None,
                      fill="blend", projection=True, restart=None, box_radius=4.0, sigma_V=0.0,
                      sigma_Pz=0.0, sigma_KE=0.0, size=None, axis_ratio=0.5,
                      after_removal="stop")
    for good in ({"gap": 1.0}, {"gap": 1e-3}, {"rho": ev.RING_RHO_MAX}, {"rho": 1e-3},
                 {"waist": 0.99}, {"waist": 0.01}, {"axis_ratio": 0.99}, {"cut_elements": 16.0},
                 {"size": 1e-5}, {"fill": "zero"}, {"restart": "impulsive"},
                 {"after_removal": "continue"}, {"sigma_V": 1e-13}, {"enabled": True}):
        ok(**good)
    for bad in ({"gap": 0.0}, {"gap": 1.0001}, {"gap": math.nan}, {"rho": 0.0},
                {"rho": ev.RING_RHO_MAX * 1.0001}, {"waist": 0.0}, {"waist": 1.0},
                {"axis_ratio": 0.0}, {"axis_ratio": 1.0}, {"cut_elements": 0.0},
                {"cut_elements": -1.0}, {"size": 0.0}, {"box_radius": 0.0}, {"sigma_V": -1e-20},
                {"sigma_KE": math.inf}, {"fill": "sample"}, {"restart": "bdf2"},
                {"after_removal": "go"}, {"enabled": 1}, {"projection": "yes"},
                {"gap": True}):
        with pytest.raises(ValueError, match="ring"):
            ok(**bad)


def test_the_block_without_the_ring_is_unchanged_and_with_it_holds_it():
    """``to_dict`` leaves the part out at its defaults: the block, its hash and every recorded block
    are what they were before the part existed. Set, the part is written, and the block round-trips;
    an unknown key of the part is refused."""
    plain = ev.EventSettings()
    assert "ring" not in plain.to_dict()
    assert list(plain.to_dict()) == ["liquid", "gap", "post", "max_events",
                                     "cascade_min_interval", "cascade_ratio", "cascade_count"]
    with_ring = ev.EventSettings.from_dict({"ring": {"enabled": True, "gap": 0.5}})
    d = with_ring.to_dict()
    assert d["ring"]["enabled"] is True and d["ring"]["gap"] == 0.5
    assert ev.EventSettings.from_dict(json.loads(json.dumps(d))) == with_ring
    assert with_ring.sha256() != plain.sha256()
    # the part at its defaults, given explicitly, is the block without it
    assert ev.EventSettings.from_dict({"ring": {}}).to_dict() == plain.to_dict()
    assert ev.EventSettings.from_dict({"ring": {"enabled": False}}).sha256() == plain.sha256()
    with pytest.raises(ValueError, match="ring: unknown key"):
        ev.EventSettings.from_dict({"ring": {"radius": 1.0}})
    with pytest.raises(ValueError, match="ring must be an object"):
        ev.EventSettings.from_dict({"ring": True})


def test_the_ring_keys_changed_and_refused_at_a_restart():
    """Against a record without the part (as P1a's precut_e004 holds it) the part enabled changes
    exactly ``ring.enabled``; from the state before an event nothing is refused; from a rotating
    dump after a ring event a key of the class is refused, ``ring.after_removal`` never."""
    recorded = ev.EventSettings.from_dict({"liquid": {"enabled": True, "eps_d": 0.4},
                                           "post": {"open_neck_stop": False,
                                                    "neck_floor": 1e-6}}).to_dict()
    assert "ring" not in recorded
    block = ev.EventSettings.from_dict({**recorded, "ring": {"enabled": True}})
    assert block.changed_keys(recorded) == ["ring.enabled"]
    assert ev.EventSettings.from_dict(recorded).changed_keys(recorded) == []
    tuned = ev.EventSettings.from_dict({**recorded, "ring": {"enabled": True, "rho": 0.4,
                                                             "sigma_V": 1e-13}})
    assert tuned.changed_keys(recorded) == ["ring.enabled", "ring.rho", "ring.sigma_V"]
    liquid_made = {"liquid": 3, "gap": 0, "ring": 0}
    ring_made = {"liquid": 3, "gap": 0, "ring": 1}
    # from precut_e004: a restart may add the part, whatever the events made
    assert block.refused_changes(recorded, ring_made, recut_from=4) == []
    # from a dump written after liquid events only, the part may be added (no ring event yet)
    assert block.refused_changes(recorded, liquid_made, recut_from=None) == []
    # after the ring cut its keys are fixed, but for after_removal
    ringed = block.to_dict()
    rho = ev.EventSettings.from_dict({**ringed, "ring": {**ringed["ring"], "rho": 0.4}})
    assert rho.refused_changes(ringed, ring_made, recut_from=None) == ["ring.rho"]
    assert rho.refused_changes(ringed, ring_made, recut_from=5) == []
    go_on = ev.EventSettings.from_dict({**ringed, "ring": {**ringed["ring"],
                                                           "after_removal": "continue"}})
    assert go_on.changed_keys(ringed) == ["ring.after_removal"]
    assert go_on.refused_changes(ringed, ring_made, recut_from=None) == []
    assert ev.EventSettings.LEG_KEYS_OF_CLASSES == ("ring.after_removal",)
    assert "ring" in ev.EventSettings.CLASSES
    assert ev.EventSettings().class_keys("ring") == asdict(ev.RingEventSettings())


def test_the_events_of_the_ring_count_in_its_class_and_its_sizes_size_the_mesh():
    records = [{"kind": "liquid-neck"}, {"kind": "piercing"}, {"kind": "ring-collapse"},
               {"kind": "a-head"}]
    assert ev.events_done(records) == {"liquid": 2, "gap": 0, "ring": 2}
    assert ev.events_done(None) == {"liquid": 0, "gap": 0, "ring": 0}
    assert "ring.size" in ev.SIZING_KEYS and "ring.cut_elements" in ev.SIZING_KEYS
    assert ev.HANDLERS["W1"] == "ring" and ev.HANDLERS["W2"] == "ring-removal"
    assert ev.HANDLERS["RA"] is None and ev.HANDLERS["RF"] is None
    assert ev.RING_HANDLERS == ("ring", "ring-removal")
    assert ev.IMPLEMENTED_HANDLERS == ("liquid-cut",)
    assert {ev.RING_CUT_REFUSED, ev.RING_REMOVAL_REFUSED} <= set(ev.EVENT_STOPS)
    assert not {ev.RING_CUT_REFUSED, ev.RING_REMOVAL_REFUSED} & set(ev.STOP_OUTCOMES)


# --- the trigger of the piercing ------------------------------------------------------------------


def _small(length, z0=4.6):
    """A bubble of the given length on the axis, a triangle 1e-3 wide."""
    return rc.Chain(np.array([[0.0, z0], [1e-3, z0 + 0.5 * length], [0.0, z0 + length]]),
                    ("axis", "axis"))


def _request(length, events, h_min=1e-5):
    from test_reconnect_geometry import _cavity

    cav = _cavity()
    small = _small(length)
    s = rc.ReconnectSettings(delta=1e-3)
    return rc.post_cut_stops([small, cav], rc.cut_geometry([small, cav], 3.0, gas_before=1.0), s,
                             h_min, events=events)


def test_the_piercing_fires_at_delta_without_the_ring_and_at_its_gap_with_it():
    """Without the part the request comes at ``bubble_length < delta`` with the fields it had at
    c4620f6, in their order; with ``ring.enabled`` and ``gap`` 0.5 it does not come at 0.8 delta
    and comes at 0.4 delta, with its gap and the limit ``max(gap delta, 3 h_min)``; the limit never
    falls below ``3 h_min``."""
    off = ev.EventSettings()
    req = _request(8e-4, off)
    assert list(req) == ["reason", "kind", "class", "handler", "bubble_length", "limit"]
    assert (req["kind"], req["class"], req["handler"]) == ("piercing", "W1", "ring")
    assert req["limit"] == 1e-3 and req["bubble_length"] == pytest.approx(8e-4)
    assert _request(1.2e-3, off) is None or _request(1.2e-3, off)["kind"] != "piercing"
    on = ev.EventSettings.from_dict({"ring": {"enabled": True, "gap": 0.5}})
    late = _request(8e-4, on)
    assert late is None or late.get("kind") != "piercing"
    req = _request(4e-4, on)
    assert (req["kind"], req["class"]) == ("piercing", "W1")
    assert req["limit"] == 5e-4 and req["gap"] == 0.5
    assert list(req) == ["reason", "kind", "class", "handler", "bubble_length", "limit", "gap"]
    # the limit is floored at three tip sizes
    req = _request(4e-4, on, h_min=2e-4)
    assert req["limit"] == pytest.approx(6e-4)
    # enabled at the inherited gap: the same limit as without the part, to the bit
    same = ev.EventSettings.from_dict({"ring": {"enabled": True}})
    assert _request(8e-4, same)["limit"] == _request(8e-4, off)["limit"]
