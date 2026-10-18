"""Contract tests of the ``collapse.reconnect_events`` block (``events.EventSettings``): parsing,
validation, and which of its keys a restart may change. No finite-element problem is built."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faraday_singularity import events as ev  # noqa: E402

LIQUID = {"liquid": {"enabled": True}, "post": {"open_neck_stop": False, "neck_floor": 2e-6}}


def test_the_block_parses_with_its_defaults_and_round_trips():
    s = ev.EventSettings.from_dict({})
    assert s == ev.EventSettings()
    assert not s.liquid.enabled and s.liquid.eps_d == 0.2 and s.liquid.eps_max == 0.5
    assert s.gap.kappa_g == 0.05 and s.post.open_neck_stop
    assert s.post.bubble_bottom_sizing == "kept" and s.post.h_floor is None
    # the cut of a liquid neck: a half sphere, the cap's nodes from the old bulk, the projection,
    # the restart of the cut, the budgets within 4 delta
    assert (s.liquid.cap, s.liquid.cap_height, s.liquid.cap_fill) == ("hemispheroid", 1.0, "sample")
    assert s.liquid.projection and s.liquid.restart is None and s.liquid.box_radius == 4.0
    # the floors of the limits of its budgets, the noise of the box: none until calibrated
    assert (s.liquid.sigma_V, s.liquid.sigma_Pz, s.liquid.sigma_KE) == (0.0, 0.0, 0.0)
    d = json.loads(json.dumps(s.to_dict()))
    assert ev.EventSettings.from_dict(d) == s
    full = ev.EventSettings.from_dict({
        "liquid": {"enabled": True, "eps_d": 0.1, "eps_max": 0.4, "track_depth": 0.1,
                   "max_head_volume": 10.0, "resolution": 6.0, "cap": "hemispheroid",
                   "cap_height": 0.5, "cap_fill": "transfer", "box_radius": 8.0,
                   "projection": False, "restart": "impulsive", "sigma_V": 4e-13,
                   "sigma_Pz": 1e-12, "sigma_KE": 2e-11},
        "gap": {"kappa_g": None},
        "post": {"open_neck_stop": False, "h_floor": 1e-7, "neck_floor": 2e-7,
                 "neck_elements_per_scale": 16, "rise_tol": 1e-7, "dt_min": 1e-14,
                 "minimum_dt": 1e-17, "bubble_bottom_sizing": "frozen",
                 "bubble_bottom_size": 5.53e-5}})
    assert ev.EventSettings.from_dict(json.loads(json.dumps(full.to_dict()))) == full
    assert full.gap.kappa_g is None and full.post.neck_elements_per_scale == 16
    assert full.liquid.restart == "impulsive" and not full.liquid.projection
    assert (full.liquid.sigma_V, full.liquid.sigma_Pz, full.liquid.sigma_KE) == (4e-13, 1e-12,
                                                                                  2e-11)
    assert len(full.sha256()) == 64 and full.sha256() != s.sha256()
    assert ev.EventSettings.from_dict(full.to_dict()).sha256() == full.sha256()


def test_the_keys_of_the_runs_events_parse_with_their_defaults_and_round_trip():
    """The block's own keys (the most events a run makes, the test of a cascade) and those of
    post that keep the dumps of the events and prune the meshes: their defaults, and a block that
    sets them all comes back from its JSON."""
    s = ev.EventSettings()
    assert (s.max_events, s.cascade_min_interval, s.cascade_ratio, s.cascade_count) == (
        1000, 0.25, 0.7, 3)
    assert s.post.keep_event_dumps == "all" and s.post.prune_gmsh is True
    d = s.to_dict()
    assert {k: d[k] for k in ev.EventSettings.KEYS} == {
        "max_events": 1000, "cascade_min_interval": 0.25, "cascade_ratio": 0.7,
        "cascade_count": 3}
    full = ev.EventSettings.from_dict({"max_events": 40, "cascade_min_interval": 0.0,
                                       "cascade_ratio": 0.5, "cascade_count": 2,
                                       "post": {"keep_event_dumps": "first20-last8",
                                                "prune_gmsh": False}})
    assert ev.EventSettings.from_dict(json.loads(json.dumps(full.to_dict()))) == full
    assert full.max_events == 40 and full.post.keep_event_dumps == "first20-last8"
    assert not full.post.prune_gmsh and full.sha256() != s.sha256()
    # they are keys of the leg, which any restart may change, also after events of every class
    assert full.changed_keys(s.to_dict()) == ["cascade_count", "cascade_min_interval",
                                              "cascade_ratio", "max_events",
                                              "post.keep_event_dumps", "post.prune_gmsh"]
    assert full.refused_changes(s.to_dict(), {"liquid": 3, "gap": 1}, None) == []
    # a block recorded before they existed ran with their defaults
    old = {k: v for k, v in s.to_dict().items() if k not in ev.EventSettings.KEYS}
    del old["post"]["keep_event_dumps"], old["post"]["prune_gmsh"]
    assert s.changed_keys(old) == [] and full.changed_keys(old) == full.changed_keys(s.to_dict())


def test_the_policy_of_the_dumps_of_events_keeps_the_first_before_and_the_last_both():
    """``post.keep_event_dumps``: ``all`` keeps both dumps of every event; ``first<N>-last<M>``
    the state before each of the first N events and both states of each of the last M."""
    assert ev.event_dump_policy("all") is None
    assert ev.event_dump_policy("first20-last8") == (20, 8)
    assert ev.event_dump_policy("first0-last0") == (0, 0)
    both = ("precut", "postcut")
    assert ev.event_dumps_kept([1, 2, 3], None) == {1: both, 2: both, 3: both}
    kept = ev.event_dumps_kept(range(1, 31), (20, 8))
    assert [k for k, v in kept.items() if v == ("precut",)] == list(range(1, 21))
    assert [k for k, v in kept.items() if v == both] == list(range(23, 31))
    assert [k for k, v in kept.items() if v == ()] == [21, 22]
    # the last ones keep both, also among the first
    kept = ev.event_dumps_kept([1, 2, 3], (2, 2))
    assert kept == {1: ("precut",), 2: both, 3: both}
    assert ev.event_dumps_kept([5, 6], (0, 0)) == {5: (), 6: ()}


@pytest.mark.parametrize("block, match", [
    ({"ghost_window_end": True}, "unknown key"),
    ({"liquid": {"eps": 0.2}}, r"liquid: unknown key\(s\) eps"),
    ({"post": {"courant_mode": "ale"}}, "post: unknown key"),
    ({"gap": {"mode": "fill"}}, "gap: unknown key"),
    ({"liquid": []}, "liquid must be an object"),
    ({"liquid": {"enabled": True}}, "open_neck_stop false"),
    ({"liquid": {"enabled": True}, "post": {"open_neck_stop": False}}, "post.neck_floor"),
    ({"liquid": {"enabled": 1}}, "true or false"),
    ({"liquid": {"eps_d": 0.6}}, "eps_d <= eps_max < 1"),
    ({"liquid": {"eps_max": 1.0}}, "eps_d <= eps_max < 1"),
    ({"liquid": {"eps_d": 0.0}}, "eps_d must be positive"),
    ({"liquid": {"track_depth": 1.0}}, r"track_depth must lie in \[0, 1\)"),
    ({"liquid": {"resolution": float("inf")}}, "resolution must be positive and finite"),
    ({"liquid": {"cap": "arc"}}, "cap must be one of"),
    ({"liquid": {"cap_height": 1.5}}, r"cap_height must lie in \(0, 1\]"),
    ({"liquid": {"cap_height": 0.0}}, "cap_height must be positive"),
    ({"liquid": {"cap_fill": "zero"}}, "cap_fill must be one of"),
    ({"liquid": {"box_radius": -4.0}}, "box_radius must be positive"),
    ({"liquid": {"projection": 1}}, "projection must be true or false"),
    ({"liquid": {"restart": "bdf2"}}, "restart must be one of"),
    ({"liquid": {"sigma_V": -1e-12}}, "sigma_V must be finite and not negative"),
    ({"liquid": {"sigma_Pz": float("inf")}}, "sigma_Pz must be finite and not negative"),
    ({"liquid": {"sigma_KE": True}}, "sigma_KE must be finite and not negative"),
    ({"gap": {"kappa_g": 0.0}}, "kappa_g must be positive and finite, or null"),
    ({"post": {"h_floor": -1e-7}}, "h_floor must be positive"),
    ({"post": {"dt_min": True}}, "dt_min must be positive"),
    ({"post": {"open_neck_stop": "no"}}, "true or false"),
    ({"post": {"bubble_bottom_sizing": "own"}}, "bubble_bottom_sizing must be one of"),
    ({"post": {"bubble_bottom_sizing": "frozen"}}, "bubble_bottom_size is given"),
    ({"post": {"bubble_bottom_size": 5e-5}}, "bubble_bottom_size is given"),
    ({"post": {"dt_min": 1e-14, "minimum_dt": 1e-14}}, "at most half the step-size floor"),
    ({"post": {"dt_min": 1e-12, "minimum_dt": 9e-13}}, "at most half the step-size floor"),
    ({"post": {"minimum_dt": 1e-11}}, "may not raise it"),
    ({"post": {"minimum_dt": 5e-12}}, "may not raise it"),
    ({"post": {"keep_event_dumps": "last8"}}, "keep_event_dumps must be 'all' or"),
    ({"post": {"keep_event_dumps": 20}}, "keep_event_dumps must be 'all' or"),
    ({"post": {"prune_gmsh": "yes"}}, "prune_gmsh must be true or false"),
    ({"max_events": 0}, "max_events must be a positive integer"),
    ({"max_events": 10.0}, "max_events must be a positive integer"),
    ({"cascade_count": True}, "cascade_count must be a positive integer"),
    ({"cascade_min_interval": -0.1}, "cascade_min_interval must be finite and not negative"),
    ({"cascade_ratio": 1.0}, r"cascade_ratio must lie in \(0, 1\)"),
    ({"cascade_ratio": 0.0}, r"cascade_ratio must lie in \(0, 1\)"),
])
def test_the_block_refuses_unknown_keys_and_invalid_values(block, match):
    with pytest.raises(ValueError, match=match):
        ev.EventSettings.from_dict(block)


def test_oomph_libs_smallest_step_after_the_cut_lowers_the_default_below_half_the_floor():
    """``events.minimum_dt_after_cut``: post.minimum_dt, or 1e-3 times the step-size floor after
    the cut but never above oomph-lib's default of 1e-12; it must lie at most half that floor, so
    that a step failing at the floor is still halved, which the collapse's own floor (a null
    post.dt_min) is checked against where the driver is made."""
    post = ev.PostCutSettings()
    assert ev.OOMPH_MINIMUM_DT == 1e-12
    assert ev.minimum_dt_after_cut(post, 1e-12) == 1e-3 * 1e-12
    assert ev.minimum_dt_after_cut(post, 1e-6) == 1e-12  # 1e-9 would raise oomph-lib's default
    assert ev.minimum_dt_after_cut(ev.PostCutSettings(minimum_dt=5e-13), 1e-12) == 5e-13
    with pytest.raises(ValueError, match="at most half the step-size floor"):
        ev.minimum_dt_after_cut(ev.PostCutSettings(minimum_dt=9e-13), 1e-12)
    with pytest.raises(ValueError, match="at most half the step-size floor"):
        ev.minimum_dt_after_cut(ev.PostCutSettings(minimum_dt=1e-12), 1e-12)
    # with post.dt_min the block checks itself; without it, minimum_dt waits for the floor
    assert ev.EventSettings.from_dict({"post": {"minimum_dt": 9e-13}}).post.minimum_dt == 9e-13
    assert ev.EventSettings.from_dict({"post": {"dt_min": 2e-12, "minimum_dt": 1e-12}})


def test_liquid_events_need_the_open_neck_stop_off():
    s = ev.EventSettings.from_dict(LIQUID)
    assert s.liquid.enabled and not s.post.open_neck_stop
    with pytest.raises(ValueError, match="must be an object"):
        ev.EventSettings.from_dict([])


def test_the_class_keys_are_those_of_liquid_and_gap():
    s = ev.EventSettings.from_dict(LIQUID)
    assert s.class_keys("liquid") == s.to_dict()["liquid"]
    assert s.class_keys("gap") == {"kappa_g": 0.05}
    with pytest.raises(ValueError, match="no class of events"):
        s.class_keys("post")


def test_changed_keys_count_a_key_the_record_lacks_at_its_default():
    s = ev.EventSettings.from_dict(LIQUID)
    recorded = s.to_dict()
    assert s.changed_keys(recorded) == []
    del recorded["liquid"]["resolution"]  # recorded before the key existed: its default
    assert s.changed_keys(recorded) == []
    other = ev.EventSettings.from_dict({**LIQUID, "liquid": {"enabled": True, "eps_d": 0.1},
                                        "post": {"open_neck_stop": False, "neck_floor": 2e-6,
                                                 "h_floor": 1e-7}})
    assert other.changed_keys(recorded) == ["liquid.eps_d", "post.h_floor"]
    assert ev.EventSettings().changed_keys(None) == []  # against the defaults
    assert other.changed_keys(None) == ["liquid.enabled", "liquid.eps_d", "post.h_floor",
                                        "post.neck_floor", "post.open_neck_stop"]


def test_a_class_is_fixed_after_its_first_event_but_from_the_state_before_one():
    """The keys of a class of events may not change once an event of it was made, but in a
    restart from the state before an event (precut_e<k>), from which they apply anew; the keys of
    the leg change at any restart; a state written before the block takes any block."""
    s = ev.EventSettings.from_dict(LIQUID)
    recorded = s.to_dict()
    other = ev.EventSettings.from_dict({"liquid": {"enabled": True, "eps_d": 0.1},
                                        "gap": {"kappa_g": 0.1},
                                        "post": {"open_neck_stop": False, "neck_floor": 2e-6,
                                                 "h_floor": 1e-7}})
    none = {"liquid": 0, "gap": 0}
    assert other.refused_changes(recorded, none, None) == []
    assert other.refused_changes(recorded, {"liquid": 2, "gap": 0}, None) == ["liquid.eps_d"]
    assert other.refused_changes(recorded, {"liquid": 2, "gap": 1}, None) == ["gap.kappa_g",
                                                                              "liquid.eps_d"]
    assert other.refused_changes(recorded, {"liquid": 2, "gap": 1}, 3) == []
    assert other.refused_changes(None, {"liquid": 2, "gap": 1}, None) == []
    legs = ev.EventSettings.from_dict({"liquid": {"enabled": True},
                                       "post": {"open_neck_stop": False, "neck_floor": 2e-6,
                                                "dt_min": 1e-14, "bubble_bottom_sizing": "frozen",
                                                "bubble_bottom_size": 5.53e-5}})
    assert legs.refused_changes(recorded, {"liquid": 5, "gap": 5}, None) == []


def test_events_done_count_the_made_events_by_class():
    records = [{"kind": "liquid-neck"}, {"kind": "a-head"}, {"kind": "groove-fill"},
               {"kind": "liquid-neck"}, {"kind": "piercing"}, "junk"]
    assert ev.events_done(records) == {"liquid": 3, "gap": 1}
    assert ev.events_done(None) == {"liquid": 0, "gap": 0}


def test_the_block_is_split_off_the_collapse_block():
    collapse = {"neck_stop": 1e-4, "reconnect": {"delta": 1e-3}, "reconnect_events": {}}
    rest, block = ev.split_events_block(collapse)
    assert rest == {"neck_stop": 1e-4, "reconnect": {"delta": 1e-3}} and block == {}
    assert "reconnect_events" in collapse  # the case is left alone
    assert ev.split_events_block({"neck_stop": 1e-4}) == ({"neck_stop": 1e-4}, None)
    with pytest.raises(ValueError, match="must be an object"):
        ev.split_events_block({"reconnect_events": [1]})
