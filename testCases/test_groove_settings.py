"""The ``groove`` part of the events block (``events.GrooveEventSettings``), without a
finite-element problem.

* the part's validation at each bound and each enumerated value;
* the block without it, at its defaults, the same to the byte and the hash as at 89f7f49 (and so
  as every recorded block and dump); with it written into the block and round-tripped, its unknown
  keys refused;
* its keys are keys of the class ``gap``: against a record without the part it changes exactly the
  keys set; from a rotating dump after a fill a key of the part, or ``gap.kappa_g``, is refused, and
  from the state before an event nothing is; a fill made counts in the class ``gap``;
* the RF request of the ring stage names the fill as its handler only with ``groove.enabled``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testCases"))

from faraday_singularity import events as ev  # noqa: E402

BASE = "89f7f49"  # the commit before the groove fill


def test_the_groove_part_is_validated_at_each_bound_and_value():
    ok = ev.GrooveEventSettings
    assert ok() == ok(enabled=False, mouth=1.5, cap_height=1.0, cut_elements=None, fill="blend",
                      projection=True, restart=None, box_radius=4.0, sigma_V=0.0, sigma_Pz=0.0,
                      sigma_KE=0.0, max_fill=0.01)
    for good in ({"enabled": True}, {"mouth": 1.0}, {"mouth": 4.0}, {"cap_height": 1.0},
                 {"cap_height": 0.01}, {"cut_elements": 4.0}, {"fill": "zero"},
                 {"projection": False}, {"restart": "impulsive"}, {"box_radius": 1.0},
                 {"sigma_V": 1e-16}, {"sigma_Pz": 1e-12}, {"sigma_KE": 1e-9}, {"max_fill": 1.0},
                 {"max_fill": 1e-6}):
        ok(**good)
    for bad in ({"enabled": 1}, {"mouth": 0.99}, {"mouth": math.nan}, {"cap_height": 0.0},
                {"cap_height": 1.01}, {"cut_elements": 0.0}, {"fill": "sample"},
                {"projection": "yes"}, {"restart": "bdf2"}, {"box_radius": 0.0},
                {"sigma_V": -1e-20}, {"sigma_KE": math.inf}, {"max_fill": 0.0},
                {"max_fill": 1.5}, {"mouth": True}):
        with pytest.raises(ValueError, match="groove"):
            ok(**bad)


def test_the_block_without_the_groove_part_is_89f7f49s_and_with_it_holds_it(tmp_path):
    """``to_dict`` leaves the part out at its defaults: the block and its hash are those of
    89f7f49 (the source of that commit, in a new process), also beside the other optional parts.
    Set, the part is written and the block round-trips; an unknown key is refused."""
    from test_events_fem import _reference_src

    blocks = [{}, {"liquid": {"enabled": True, "eps_d": 0.4},
                   "post": {"open_neck_stop": False, "neck_floor": 1e-6}},
              {"ring": {"enabled": True, "sigma_V": 1e-13}, "gap": {"kappa_g": 0.025}},
              {"jetcore": {"enabled": True, "far_floor": 1.5625e-5}}]
    script = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
              "from faraday_singularity import events as ev; "
              "print(json.dumps([[ev.EventSettings.from_dict(b).to_dict(), "
              "ev.EventSettings.from_dict(b).sha256()] for b in json.loads(sys.argv[2])]))")
    ref = _reference_src(tmp_path, BASE)
    got = {}
    for name, src in (("ref", ref), ("head", ROOT / "src")):
        out = subprocess.run([sys.executable, "-c", script, str(src), json.dumps(blocks)],
                             capture_output=True, text=True, check=True).stdout
        got[name] = json.loads(out)
    assert got["ref"] == got["head"]
    plain = ev.EventSettings()
    assert "groove" not in plain.to_dict()
    with_groove = ev.EventSettings.from_dict({"groove": {"enabled": True, "mouth": 2.0}})
    d = with_groove.to_dict()
    assert d["groove"]["enabled"] is True and d["groove"]["mouth"] == 2.0
    assert ev.EventSettings.from_dict(json.loads(json.dumps(d))) == with_groove
    assert with_groove.sha256() != plain.sha256()
    assert ev.EventSettings.from_dict({"groove": {}}).sha256() == plain.sha256()
    with pytest.raises(ValueError, match="groove: unknown key"):
        ev.EventSettings.from_dict({"groove": {"width": 1.0}})


def test_the_groove_keys_are_keys_of_the_class_gap():
    """Against a record without the part, the part enabled changes the keys set; after a fill
    (an event of the class ``gap``) a key of the part and ``gap.kappa_g`` are refused from a
    rotating dump, nothing from the state before an event; a fill made counts in ``gap``."""
    recorded = ev.EventSettings.from_dict({"ring": {"enabled": True}}).to_dict()
    block = ev.EventSettings.from_dict({"ring": {"enabled": True},
                                        "groove": {"enabled": True, "max_fill": 0.005}})
    assert block.changed_keys(recorded) == ["groove.enabled", "groove.max_fill"]
    assert ev.EventSettings.class_of("groove.mouth") == "gap"
    assert ev.EventSettings.class_of("gap.kappa_g") == "gap"
    assert block.refused_changes(recorded, {"gap": 0}, None) == []
    assert block.refused_changes(recorded, {"gap": 1}, None) == ["groove.enabled",
                                                                 "groove.max_fill"]
    assert block.refused_changes(recorded, {"gap": 1}, 7) == []
    other = ev.EventSettings.from_dict({"ring": {"enabled": True}, "gap": {"kappa_g": 0.025},
                                        "groove": {"enabled": True, "max_fill": 0.005}})
    assert other.refused_changes(block.to_dict(), {"gap": 1}, None) == ["gap.kappa_g"]
    done = ev.events_done([{"kind": "groove-fill"}, {"kind": "piercing"},
                           {"kind": "liquid-neck"}])
    assert done == {"liquid": 1, "gap": 1, "ring": 1}
    assert ev.GROOVE_FILL_REFUSED in ev.EVENT_STOPS
    assert ev.HANDLERS["A-groove"] == ev.GROOVE_HANDLER and ev.HANDLERS["RF"] is None
    assert ev.GROOVE_CLASSES == ("A-groove", "RF")


def test_the_ring_fold_names_the_fill_as_its_handler_only_with_the_part():
    """The ring's waist closing (``test_ring_stops``' peanut): ``RF`` without a handler, as at
    89f7f49, without the part, and with ``groove.enabled`` the same request naming the fill."""
    import test_ring_stops as rs

    t = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    rho = 0.02 + np.cos(t) ** 2
    peanut = np.column_stack([0.01 + 1e-3 * rho * np.cos(t), 4.68 + 1e-3 * rho * np.sin(t)])
    off = rs.stops([rs.flat_open()], [peanut])
    on = rs.stops([rs.flat_open()], [peanut], events=ev.EventSettings.from_dict(
        {"ring": {"enabled": True}, "groove": {"enabled": True}}))
    assert (off["class"], off["handler"]) == ("RF", None)
    assert (on["class"], on["handler"]) == ("RF", ev.GROOVE_HANDLER)
    assert {k: v for k, v in on.items() if k != "handler"} == {
        k: v for k, v in off.items() if k != "handler"}
