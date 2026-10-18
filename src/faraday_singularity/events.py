"""Events after the gas cut: the ``collapse.reconnect_events`` block of a case.

After the cut of the gas neck (``surgery.cut``) the run meets further changes of topology: the neck
at the root of a jet the cut launched closes on the liquid side (a drop pinching off), a gas groove
around the base of a jet closes, the downward jet reaches the bubble's bottom, a second gas neck
forms. A sharp-interface mesh follows none of them by itself. The block gathers the settings of
how a run after the cut meets them, in a block of its own beside ``collapse.reconnect``
(:class:`~faraday_singularity.reconnect.ReconnectSettings`, the settings of the cut, which it does
not change): a case without it runs exactly as before the block existed.

The block has three parts and a few keys of its own:

* ``liquid`` (:class:`LiquidEventSettings`): a neck on the liquid side of either chain, at the root
  of a jet whose far end is on the axis (its head), followed from ``track_depth`` on and landed once
  its radius has closed to ``eps_d`` times the largest radius of its head;
* ``gap`` (:class:`GapSettings`): a closing fold of either chain, a gas groove narrowing, met at
  the gap ``kappa_g delta`` (``delta`` the cut radius of the gas cut) instead of the absolute
  ``fold_stop`` of the collapse;
* ``post`` (:class:`PostCutSettings`): the limits and floors of the run after the cut, each by
  default the collapse's own (``h_floor``, ``rise_tol``, ``dt_min``), the floor and resolution of
  the mesh at the necks, oomph-lib's smallest step after the cut, the open chain's neck stop, the
  sizing of the bubble's bottom, which dumps of the events the run keeps and the pruning of the
  meshes no dump refers to;
* of its own: the most events a run makes (``max_events``) and the test of a cascade of events at
  one tip (``cascade_min_interval``, ``cascade_ratio``, ``cascade_count``).

Which keys a restart may change: the keys of ``liquid`` and ``gap`` are the keys of their class of
events (:meth:`EventSettings.class_keys`), fixed once an event of the class has been made, except
in a restart from the state before an event (``precut_e<k>``), from which they apply anew; every
other key is a key of the leg, which any restart may change (:meth:`EventSettings.
refused_changes`). Lengths and times are in the solver's units (length ``1/k``, capillary time).

With the block the stops of a state after the cut are those of :func:`post_cut_events`, in three
ranks: a state that is not valid stops the run; the bubble emptying ends the window; every change of
topology is an event, returned as a request for its handler (``event-request``). No handler exists
yet: the driver stops on each event, resumably (``event-unhandled``, with the dump of the state
before it, ``precut_e<k>``), so that a run continues from it once its handler exists. A liquid neck
is followed from ``liquid.track_depth`` on (:func:`liquid_necks`, :func:`head_sides`) and lands at
``r_n <= liquid.eps_d s``, ``s`` the largest radius of its head; a fold of either chain closes at
``max(gap.kappa_g delta, fold_stop_elements h_elem)``, and one that reaches the head of a tracked
neck lands that neck once ``r_n <= liquid.eps_max s`` (:func:`head_of_fold`), before any groove,
the fold of each head searched apart from that of its chain (:func:`head_fold`).

The geometry of the cut of a liquid neck (:func:`truncate_chain`): the head is removed and the stump
closed at the neck by a half-spheroid ``r = r_n cos t``, ``z = z_n + direction eta r_n sin t``
(:func:`cap_at`), which leaves the neck tangentially and lies inside the old liquid. Its points
start at the node spacing of the neck and grow by at most ``CAP_GROWTH`` per point to its own
spacing (:func:`cap_spacing`), so that the spline a rebuild draws through them has no jump in
spacing at the splice. Hard checks refuse a cap that leaves the old liquid, folds back, makes a
knife-edge angle at the splice, removes no simple region, or leaves chains a mesh cannot be built on
(:class:`CapRefused`); the removed region ``Q`` (the head, the axis and the cap) gives the volume
and the areas the cut takes away.

Of the events a run makes, :class:`CascadeMonitor` follows the sequence at each tip on the axis
(:func:`event_tip`) and judges whether it shrinks (a cascade that stops the run), grows or stays
stationary; the removed head of each liquid cut is followed as a ghost that moves rigidly at its
mean axial speed while its rear retracts (:func:`ghost_place`), whose recontacts with the stump it
left and with the far tip of the bubble are flagged (:func:`ghost_recontacts`).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields

import numpy as np

EVENTS_BLOCK = "reconnect_events"  # the key of the block in a case's collapse block
OOMPH_MINIMUM_DT = 1e-12  # oomph-lib's smallest step by default (Problem::Minimum_dt)
BUBBLE_BOTTOM_SIZINGS = ("kept", "frozen")
CAP_SHAPES = ("hemispheroid",)  # the cap that closes the stump of a cut liquid neck
CAP_FILLS = ("sample", "transfer")  # the velocity of its nodes: the old bulk's, or the transfer's
# the kinds of the events a run makes, by class: the keys of a class are fixed after its first event
EVENT_CLASSES = {"liquid": ("liquid-neck", "a-head"), "gap": ("groove-fill",)}
# a dump of the state before an event, from which a restart may change the keys of its class
PRECUT_EVENT = "precut-event"
# the keys of the block that size the mesh after the cut: a restart that changes one of them does
# not take a dumped mesh no step had run on as fresh (the checks of a fresh mesh would meet the
# mesh of the old sizes again); the rule of its rebuilds remeshes it at the new sizes instead
SIZING_KEYS = ("post.h_floor", "post.neck_floor", "post.neck_elements_per_scale",
               "post.bubble_bottom_sizing", "post.bubble_bottom_size")


def _number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _positive(owner: str, name: str, value, *, optional: bool = False):
    if optional and value is None:
        return
    if not (_number(value) and value > 0):
        raise ValueError(f"collapse.{EVENTS_BLOCK}.{owner}.{name} must be positive and finite"
                         + (", or null" if optional else ""))


def _non_negative(owner: str, name: str, value):
    if not (_number(value) and value >= 0):
        raise ValueError(f"collapse.{EVENTS_BLOCK}.{owner}.{name} must be finite and not negative")


def _flag(owner: str, name: str, value):
    if not isinstance(value, bool):
        raise ValueError(f"collapse.{EVENTS_BLOCK}.{owner}.{name} must be true or false")


@dataclass(frozen=True)
class LiquidEventSettings:
    """``liquid``: the necks on the liquid side of either chain after the cut.

    A neck of a chain (``reconnect.chain_necks``, side ``liquid``) is tracked once its depth below
    the lower of the widest points on either side of it reaches ``track_depth``. Its head is the
    side of it whose far end lies on the axis, holds neither the wall end nor the bubble's widest
    point, lies on one side of the neck's height and is at most ``max_head_volume`` times the cube
    of its largest radius ``s``. The neck lands at the first checked state with ``r_n <= eps_d s``.
    A tracked neck narrower than ``resolution`` times the element length at it on a freshly built
    mesh stops the run (``liquid-neck-unresolved``). A fold one of whose sides lies on the head of a
    tracked neck (the head's bulb coming up to the groove wall around it) lands that neck at once if
    ``r_n <= eps_max s``. With ``enabled`` false no neck is tracked: a liquid neck is met where the
    cut's own test of a second pinch-off finds it (below ``delta``), as an event without a handler
    (class ``L-unhandled``, ``detected_by`` ``secondary-pinch``), its head never classified.

    The cut of a neck (``surgery.liquid_cut``) removes the head and closes the stump by a
    half-spheroid (``cap``, ``hemispheroid``) of height ``cap_height`` neck radii (one, a half
    sphere; a refused cap is tried once more at ``CAP_RETRY_HEIGHT`` neck radii, unless that is its
    height); the velocity of the cap's nodes is the old bulk's at their places (``cap_fill``
    ``sample``), or what pyoomph's transfer gives them (``transfer``, from the old surface near the
    neck: an arm only); with ``projection`` the pressure-impulse projection of the cut follows, and
    the first step after it is ``restart`` (``reconnect.restart`` when null). The budgets of the cut
    are taken within ``max(box_radius delta, 2 L_rem)`` of the neck, ``L_rem`` the reach of the
    removed head along the axis, and the limit of each is a multiple of the head's own but never
    below a multiple of the noise of the box in an ordinary rebuild (``surgery.NOISE_FLOORS``):
    ``sigma_V`` of its liquid volume, ``sigma_Pz`` of its axial momentum, ``sigma_KE`` of its
    kinetic energy, in the solver's units, none (zero) until calibrated."""
    enabled: bool = False
    eps_d: float = 0.2
    eps_max: float = 0.5
    track_depth: float = 0.05
    max_head_volume: float = 20.0
    resolution: float = 4.0
    cap: str = "hemispheroid"
    cap_height: float = 1.0
    cap_fill: str = "sample"
    box_radius: float = 4.0
    projection: bool = True
    restart: str | None = None
    sigma_V: float = 0.0
    sigma_Pz: float = 0.0
    sigma_KE: float = 0.0

    def __post_init__(self):
        from .reconnect import RESTARTS

        _flag("liquid", "enabled", self.enabled)
        for name in ("eps_d", "eps_max", "max_head_volume", "resolution", "cap_height",
                     "box_radius"):
            _positive("liquid", name, getattr(self, name))
        if not self.eps_d <= self.eps_max < 1.0:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid: eps_d <= eps_max < 1")
        if not (_number(self.track_depth) and 0.0 <= self.track_depth < 1.0):
            raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid.track_depth must lie in [0, 1)")
        if not self.cap_height <= 1.0:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid.cap_height must lie in (0, 1] (neck "
                             "radii): a taller cap reaches beyond the half sphere")
        for name, allowed in (("cap", CAP_SHAPES), ("cap_fill", CAP_FILLS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid.{name} must be one of {allowed}")
        _flag("liquid", "projection", self.projection)
        for name in ("sigma_V", "sigma_Pz", "sigma_KE"):
            _non_negative("liquid", name, getattr(self, name))
        if self.restart is not None and self.restart not in RESTARTS:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid.restart must be one of {RESTARTS}, "
                             "or null")


@dataclass(frozen=True)
class GapSettings:
    """``gap``: a closing fold of either chain after the cut (a gas groove narrowing) is met where
    its gap falls below ``max(kappa_g delta, fold_stop_elements h_elem)``, ``delta`` the cut radius
    and ``h_elem`` the element length there; ``kappa_g`` null keeps the collapse's absolute
    ``fold_stop`` in place of ``kappa_g delta``."""
    kappa_g: float | None = 0.05

    def __post_init__(self):
        _positive("gap", "kappa_g", self.kappa_g, optional=True)


@dataclass(frozen=True)
class PostCutSettings:
    """``post``: the limits and floors of the run after the cut, each null for the collapse's own.

    ``h_floor`` replaces the collapse's ``h_floor`` after the cut (the smallest focus and feature
    size, the floor of the interface guard and of its kink regions, and the open chain's neck stop
    through ``neck_stop_elements h_floor``); ``neck_floor`` is the smallest size of the region of a
    neck (the cut's focus size without it; liquid events need it set, so that the region follows
    a neck below the cut radius to its landing) and ``neck_elements_per_scale`` its elements per
    neck radius (``reconnect.elements_per_scale`` without it); ``rise_tol`` replaces the
    collapse's ``rise_tol`` (the rise of a bump above its base that makes a jet); ``dt_min`` the
    step-size floor of the driver; ``minimum_dt`` is oomph-lib's smallest step after the cut
    (``Problem::Minimum_dt``, :func:`minimum_dt_after_cut`: 1e-3 times the driver's floor without
    it, never above oomph-lib's default), below which it gives up on a step, and below which it also
    refuses an accepted step whose next candidate, four times the step, would fall; it lowers
    oomph-lib's default (1e-12), never raises it, and lies at most half the driver's floor, so
    that a step failing at that floor is still halved once; before the cut it stays at the
    default. ``open_neck_stop`` false drops the open chain's neck stop after the cut
    (``collapse.neck_stop``), which liquid events need.
    ``bubble_bottom_sizing`` ``frozen`` meshes the bubble's bottom (far from the cut) never coarser
    than ``bubble_bottom_size`` instead of never coarser than the landing state's focus there
    (``kept``), so that the bubbles of a study in ``delta`` are meshed alike, but within a few of
    the cut's tip sizes (``tip_size delta``) of the bottom's tip on the axis, where the cut's tip
    attractor still applies, and on the mesh of the cut itself, which has the size the cut keeps
    there (``focus.kept``) until the first rebuild after it (``reconnect.feature_regions``).
    ``keep_event_dumps`` says which dumps of the events made the run keeps: ``all``, or
    ``first<N>-last<M>``, the state before each of the first ``N`` events and both states of
    each of the last ``M`` (:func:`event_dumps_kept`; the state before an event the run stops on
    is always kept). With ``prune_gmsh`` the meshes gmsh wrote for the run that no dump it keeps
    refers to, and that the problem no longer holds, are deleted after every dump."""
    open_neck_stop: bool = True
    h_floor: float | None = None
    neck_floor: float | None = None
    neck_elements_per_scale: float | None = None
    rise_tol: float | None = None
    dt_min: float | None = None
    minimum_dt: float | None = None
    bubble_bottom_sizing: str = "kept"
    bubble_bottom_size: float | None = None
    keep_event_dumps: str = "all"
    prune_gmsh: bool = True

    def __post_init__(self):
        _flag("post", "open_neck_stop", self.open_neck_stop)
        for name in ("h_floor", "neck_floor", "neck_elements_per_scale", "rise_tol", "dt_min",
                     "minimum_dt", "bubble_bottom_size"):
            _positive("post", name, getattr(self, name), optional=True)
        if self.bubble_bottom_sizing not in BUBBLE_BOTTOM_SIZINGS:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.post.bubble_bottom_sizing must be one of "
                             f"{BUBBLE_BOTTOM_SIZINGS}")
        if (self.bubble_bottom_sizing == "frozen") != (self.bubble_bottom_size is not None):
            raise ValueError(f"collapse.{EVENTS_BLOCK}.post.bubble_bottom_size is given with "
                             "bubble_bottom_sizing frozen, and only then")
        if self.minimum_dt is not None and self.minimum_dt > OOMPH_MINIMUM_DT:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.post.minimum_dt lowers oomph-lib's smallest "
                             f"step, {OOMPH_MINIMUM_DT:g} by default; it may not raise it")
        if self.dt_min is not None:
            minimum_dt_after_cut(self, self.dt_min)
        event_dump_policy(self.keep_event_dumps)
        _flag("post", "prune_gmsh", self.prune_gmsh)


def event_dump_policy(value) -> tuple[int, int] | None:
    """``post.keep_event_dumps`` as ``(N, M)`` for ``first<N>-last<M>``, None for ``all``. Raises
    ``ValueError`` for any other value."""
    if value == "all":
        return None
    match = re.fullmatch(r"first(\d+)-last(\d+)", value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"collapse.{EVENTS_BLOCK}.post.keep_event_dumps must be 'all' or "
                         "'first<N>-last<M>' (e.g. 'first20-last8')")
    return int(match.group(1)), int(match.group(2))


def event_dumps_kept(made, policy) -> dict[int, tuple[str, ...]]:
    """Which dumps of the events ``made`` (their numbers ``k``) a run keeps under the policy
    ``(N, M)`` of :func:`event_dump_policy` (None: every one): by event, ``("precut",)`` for each
    of the first ``N`` and ``("precut", "postcut")`` for each of the last ``M``, nothing for the
    others; ``("precut", "postcut")`` for every one without a policy."""
    ks = sorted(int(k) for k in made)
    if policy is None:
        return {k: ("precut", "postcut") for k in ks}
    first, last = policy
    late = set(ks[-last:]) if last > 0 else set()
    return {k: ("precut", "postcut") if k in late else ("precut",) if i < first else ()
            for i, k in enumerate(ks)}


def minimum_dt_after_cut(post: PostCutSettings, dt_min: float) -> float:
    """oomph-lib's smallest step after the cut with the block, for the driver's step-size floor
    after it, ``dt_min`` (``post.dt_min``, or the collapse's own without it): ``post.minimum_dt``,
    or ``1e-3 dt_min`` but at most oomph-lib's default (:data:`OOMPH_MINIMUM_DT`). Raises
    ``ValueError`` unless it lies at most half ``dt_min``: oomph-lib halves a failed step only
    above its smallest step, so that a step failing at the driver's floor would otherwise end the
    run as a failed one, which no failure after a cut ends cleanly, before the floor stops it."""
    value = post.minimum_dt if post.minimum_dt is not None else min(1e-3 * dt_min,
                                                                      OOMPH_MINIMUM_DT)
    if not 2.0 * value <= dt_min:
        raise ValueError(f"collapse.{EVENTS_BLOCK}.post.minimum_dt ({value:g}) must lie at most "
                         f"half the step-size floor after the cut ({dt_min:g}): oomph-lib would "
                         "give up on a step the driver still allows, without halving it")
    return value


@dataclass(frozen=True)
class EventSettings:
    """The ``collapse.reconnect_events`` block (module docstring), with the keys of its own, of the
    leg: the most events a run makes (``max_events``, over all its legs; the run stops resumably at
    the next, ``max-events``) and the test of a cascade of events at one tip
    (:class:`CascadeMonitor`: ``cascade_min_interval`` in ``delta^2``, ``cascade_ratio``,
    ``cascade_count``)."""
    liquid: LiquidEventSettings = field(default_factory=LiquidEventSettings)
    gap: GapSettings = field(default_factory=GapSettings)
    post: PostCutSettings = field(default_factory=PostCutSettings)
    max_events: int = 1000
    cascade_min_interval: float = 0.25
    cascade_ratio: float = 0.7
    cascade_count: int = 3

    PARTS = {"liquid": LiquidEventSettings, "gap": GapSettings, "post": PostCutSettings}
    CLASSES = ("liquid", "gap")  # the parts whose keys are the keys of a class of events
    KEYS = ("max_events", "cascade_min_interval", "cascade_ratio", "cascade_count")  # of the leg

    def __post_init__(self):
        for name, cls in self.PARTS.items():
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"collapse.{EVENTS_BLOCK}.{name} must be a {cls.__name__}")
        if self.liquid.enabled and self.post.open_neck_stop:
            raise ValueError(f"collapse.{EVENTS_BLOCK}: liquid events need post.open_neck_stop "
                             "false (the open chain's neck stop would end the run before its "
                             "liquid neck lands)")
        if self.liquid.enabled and self.post.neck_floor is None:
            raise ValueError(f"collapse.{EVENTS_BLOCK}: liquid events need post.neck_floor (the "
                             "region of a neck is otherwise floored at the cut's focus size, "
                             "which cannot follow a neck to its landing below the cut radius)")
        for name in ("max_events", "cascade_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"collapse.{EVENTS_BLOCK}.{name} must be a positive integer")
        if not (_number(self.cascade_min_interval) and self.cascade_min_interval >= 0.0):
            raise ValueError(f"collapse.{EVENTS_BLOCK}.cascade_min_interval must be finite and not "
                             "negative (in delta^2)")
        if not (_number(self.cascade_ratio) and 0.0 < self.cascade_ratio < 1.0):
            raise ValueError(f"collapse.{EVENTS_BLOCK}.cascade_ratio must lie in (0, 1)")

    @classmethod
    def from_dict(cls, data) -> "EventSettings":
        """The block of a case, refusing keys it does not know (at every level)."""
        if not isinstance(data, dict):
            raise ValueError(f"collapse.{EVENTS_BLOCK} must be an object")
        unknown = sorted(set(data) - set(cls.PARTS) - set(cls.KEYS))
        if unknown:
            raise ValueError(f"collapse.{EVENTS_BLOCK}: unknown key(s) " + ", ".join(unknown))
        kw = {name: data[name] for name in cls.KEYS if name in data}
        for name, part in cls.PARTS.items():
            sub = data.get(name, {})
            if not isinstance(sub, dict):
                raise ValueError(f"collapse.{EVENTS_BLOCK}.{name} must be an object")
            known = {f.name for f in fields(part)}
            unknown = sorted(set(sub) - known)
            if unknown:
                raise ValueError(f"collapse.{EVENTS_BLOCK}.{name}: unknown key(s) "
                                 + ", ".join(unknown))
            kw[name] = part(**sub)
        return cls(**kw)

    def to_dict(self) -> dict:
        return {**{name: asdict(getattr(self, name)) for name in self.PARTS},
                **{name: getattr(self, name) for name in self.KEYS}}

    def sha256(self) -> str:
        """The hash of the block's canonical JSON, recorded with every leg of a run."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def class_keys(self, kind: str) -> dict:
        """The keys of a class of events (``liquid`` or ``gap``)."""
        if kind not in self.CLASSES:
            raise ValueError(f"no class of events {kind!r}; the classes are {self.CLASSES}")
        return asdict(getattr(self, kind))

    @classmethod
    def _completed(cls, recorded: dict) -> dict:
        """A recorded block with the keys it lacks at their defaults (a block recorded before a
        key existed ran with the key's default)."""
        out = cls().to_dict()
        for name, sub in (recorded or {}).items():
            if name in cls.PARTS:
                out[name] = {**out.get(name, {}), **(sub if isinstance(sub, dict) else {})}
            else:
                out[name] = sub
        return out

    def changed_keys(self, recorded: dict | None) -> list[str]:
        """The keys in which the block differs from a recorded one, dotted for those of a part
        (``liquid.eps_d``); a key the record lacks counts at its default."""
        old = self._completed(recorded or {})
        new = json.loads(json.dumps(self.to_dict()))
        out = []
        for name in sorted(set(old) | set(new)):
            if name not in self.PARTS:
                if name not in old or name not in new or old[name] != new[name]:
                    out.append(name)
                continue
            a, b = old.get(name, {}), new.get(name, {})
            out += [f"{name}.{k}" for k in sorted(set(a) | set(b), key=str)
                    if k not in a or k not in b or a[k] != b[k]]
        return out

    def refused_changes(self, recorded: dict | None, done: dict[str, int],
                        recut_from: int | None) -> list[str]:
        """The changes from the ``recorded`` block (``None``: a state written before the block
        existed, which may take any block) that a restart may not make: those of the keys of a class
        of which ``done`` counts an event already made, unless the restart is from the state before
        an event, ``precut_e<recut_from>``, from which the class keys apply anew (events before it
        stay as made). The keys of the leg (``post`` and the block's own, ``KEYS``) may change at
        any restart."""
        if recorded is None or recut_from is not None:
            return []
        return [k for k in self.changed_keys(recorded)
                if k.split(".", 1)[0] in self.CLASSES and done.get(k.split(".", 1)[0], 0) > 0]


def events_done(records) -> dict[str, int]:
    """The events made, by class (``EVENT_CLASSES``), from the records a dump keeps."""
    out = {name: 0 for name in EVENT_CLASSES}
    for rec in records or ():
        kind = rec.get("kind") if isinstance(rec, dict) else None
        for name, kinds in EVENT_CLASSES.items():
            if kind in kinds:
                out[name] += 1
    return out


def split_events_block(collapse: dict) -> tuple[dict, dict | None]:
    """Separate the ``reconnect_events`` block from a ``collapse`` block: the rest, for
    ``reconnect.split_collapse_block``, and the block (``None`` without it)."""
    rest = dict(collapse)
    block = rest.pop(EVENTS_BLOCK, None)
    if block is not None and not isinstance(block, dict):
        raise ValueError(f"collapse.{EVENTS_BLOCK} must be an object")
    return rest, block


# --- the events after the cut, as the stops find them ---------------------------------------------

# what reconnect.post_cut_stops returns, with the block, for an event: the driver meets it with its
# handler, and stops on one that has none yet, resumably (EVENT_UNHANDLED)
EVENT_REQUEST = "event-request"
EVENT_UNHANDLED = "event-unhandled"
EVENT_SCHEMA = "faraday-event/2"  # the schema of an event's record (events/event_<k>.json)
# the handler of each class of event (None: none, an unhandled event or one to be designed when it
# occurs), by class and not by kind: a liquid neck without a single head, with one too large, or
# met with liquid events off (L-unhandled) has none, whatever found it
HANDLERS = {"W1": "ring", "L": "liquid-cut", "A-head": "liquid-cut", "L-unhandled": None,
            "A-groove": "groove-fill", "G": "n-chain-split", "B": None, "X": None}
# the handlers the driver makes within the run (surgery.liquid_cut); an event whose handler is
# another, or none, is a resumable stop (EVENT_UNHANDLED)
IMPLEMENTED_HANDLERS = ("liquid-cut",)
# the stops of a run at an event, each with the dump of the state before the event (precut_e<k>),
# from which a run resumes: an event without a handler; a cascade of events that shrink (the
# CascadeMonitor); the most events (max_events); a liquid cut refused before anything changed;
# one whose hard budgets do not close (surgery.BudgetUnclosed), which leaves the problem holding a
# cut that is not accepted, so that the state before it is the run's end state
EVENT_CASCADE = "event-cascade"
EVENT_LIMIT = "max-events"
LIQUID_CUT_REFUSED = "liquid-cut-refused"
EVENT_UNCLOSED = "event-unclosed"
EVENT_STOPS = (EVENT_UNHANDLED, EVENT_CASCADE, EVENT_LIMIT, LIQUID_CUT_REFUSED, EVENT_UNCLOSED)
HEAD_SIDE_TOL = 0.05  # a head lies on one side of its neck's height to within this * r_n


def fold_stop_after_cut(events: EventSettings | None, delta: float, fold_stop: float) -> float:
    """The absolute part of the fold limit after the cut: ``kappa_g delta`` with the block's
    ``gap.kappa_g``, the collapse's ``fold_stop`` without the block or with ``kappa_g`` null."""
    if events is None or events.gap.kappa_g is None:
        return fold_stop
    return events.gap.kappa_g * delta


def head_sides(points, end_types, index: int, *, is_bubble: bool,
               tol: float = HEAD_SIDE_TOL) -> list[dict]:
    """The two sides of the neck at node ``index`` of a chain, each with the tests of a head: the
    side's far end lies on the axis (``axis_end``); it does not hold the widest node of the bubble
    chain (``holds_widest``; on the open chain the wall end, which no axis end is, rules the other
    side out); its nodes lie on the side of the neck's height ``z_n`` that its tip lies on, to
    within ``tol r_n`` (``one_sided``, with ``direction`` the sign of ``z_tip - z_n``); and its
    half-section, the side from the neck to its tip closed along the axis and by the chord at
    ``z_n``, is a simple polygon (``simple``). ``qualifies`` when all hold. Each side also gives its
    node range (``lo``, ``hi``, the neck included), its largest radius ``s``, the volume of its
    half-section revolved (``volume``) and its tip's height (``z_tip``)."""
    from shapely.geometry import LinearRing

    from .reconnect import AXIS, revolved_volume

    P = np.asarray(points, dtype=float)
    n, i = len(P), int(index)
    r_n, z_n = float(P[i, 0]), float(P[i, 1])
    widest = int(np.argmax(P[:, 0]))
    out = []
    for name, lo, hi, end in (("before", 0, i, 0), ("after", i, n - 1, 1)):
        side = P[lo:hi + 1]
        z_tip = float(P[0, 1] if end == 0 else P[-1, 1])
        direction = float(np.sign(z_tip - z_n))
        rec = {"side": name, "lo": lo, "hi": hi, "axis_end": end_types[end] == AXIS,
               "holds_widest": bool(is_bubble and lo <= widest <= hi),
               "direction": direction, "z_tip": z_tip, "s": float(side[:, 0].max()),
               "one_sided": bool(direction != 0.0
                                 and np.all(direction * (side[:, 1] - z_n) >= -tol * r_n)),
               "simple": False, "volume": math.nan}
        if rec["axis_end"]:
            path = side[::-1] if end == 0 else side  # from the neck to the tip on the axis
            section = np.vstack([path, [[0.0, z_n]]])
            rec["simple"] = bool(LinearRing(section).is_simple)
            rec["volume"] = revolved_volume(section)
        rec["qualifies"] = bool(rec["axis_end"] and not rec["holds_widest"] and rec["one_sided"]
                                and rec["simple"])
        out.append(rec)
    return out


def liquid_necks(named, ls: LiquidEventSettings) -> list[dict]:
    """The tracked necks on the liquid side of the ``(name, chain)`` pairs: every neck of a chain
    (``reconnect.chain_necks``) with side ``liquid`` and a depth of at least ``ls.track_depth``.

    Each gives its chain and node (``chain``, ``index``), ``r_n``, ``z_n``, ``depth`` and
    ``shoulder`` (``chain_necks``), the element length there, twice the shorter of its two segments
    (``h_loc``), and its head (:func:`head_sides`): where exactly one side qualifies, ``head`` (its
    node range), ``s`` its largest radius, ``direction``, ``z_tip`` and ``head_volume``, with
    ``status`` ``head``, or ``head-too-large`` beyond ``ls.max_head_volume s^3``; where none or both
    do, ``status`` ``no-head`` or ``two-heads``, ``head`` None and ``s`` the shoulder. The neck has
    landed (``landed``) once ``r_n <= eps_d s`` (``ratio``, ``r_n/(eps_d s)``, at most one)."""
    from .reconnect import chain_necks

    out = []
    for name, c in named:
        P = np.asarray(c.points, dtype=float)
        seg = np.hypot(*np.diff(P, axis=0).T)
        for nk in chain_necks(P, c.end_types):
            if nk["side"] != "liquid" or nk["depth"] < ls.track_depth:
                continue
            i = int(nk["index"])
            rec = {"chain": name, "index": i, "r_n": float(nk["r"]), "z_n": float(nk["z"]),
                   "depth": float(nk["depth"]), "shoulder": float(nk["shoulder"]),
                   "h_loc": 2.0 * float(min(seg[i - 1], seg[i]))}
            heads = [h for h in head_sides(P, c.end_types, i, is_bubble=not c.is_open)
                     if h["qualifies"]]
            if len(heads) == 1:
                h = heads[0]
                s = h["s"]
                rec.update(head=[h["lo"], h["hi"]], side=h["side"], s=s, direction=h["direction"],
                           z_tip=h["z_tip"], head_volume=h["volume"],
                           head_volume_over_s3=h["volume"] / s**3,
                           status="head" if h["volume"] <= ls.max_head_volume * s**3
                           else "head-too-large")
            else:
                rec.update(head=None, s=rec["shoulder"],
                           status="no-head" if not heads else "two-heads")
            rec["ratio"] = rec["r_n"] / (ls.eps_d * rec["s"])
            rec["landed"] = bool(rec["r_n"] <= ls.eps_d * rec["s"])
            out.append(rec)
    return out


def head_of_fold(pair, necks: list[dict]) -> dict | None:
    """The tracked neck (:func:`liquid_necks`, on the fold's chain) on whose head a node of the
    fold pair lies, beyond the neck itself: the bulb of a head coming up to the wall of the groove
    around it (the narrowest relative to its head, ``r_n/s``, if several); None otherwise."""
    hits = [x for x in necks if x.get("head") is not None
            and any(x["head"][0] <= k <= x["head"][1] and k != x["index"] for k in pair)]
    return min(hits, key=lambda x: x["r_n"] / x["s"]) if hits else None


def head_fold(points, head, exclude: int, *, ratio: float, stop: float, stop_elements: float,
              search_factor: float):
    """The fold of a chain among the pairs with a node on a head (``head``, the node range
    ``[lo, hi]`` of :func:`liquid_necks`, but its neck ``exclude``), selected as
    :func:`~faraday_singularity.selfcontact.fold_gap` selects the fold of the whole chain: the
    pairs it examines (within the search radius ``search_factor max(stop, 2 stop_elements
    spacing)`` of the node of the larger spacing), a fold where ``ratio D`` is below the arclength
    between them, the smallest ``D / max(stop, stop_elements (spacing_i + spacing_j))``. A closer
    fold elsewhere on the chain, which ``fold_gap`` selects instead, hides none on the head. Returns
    a ``selfcontact.FoldGap`` (``NO_FOLD`` without one)."""
    from scipy.spatial import cKDTree

    from .selfcontact import NO_FOLD, FoldGap

    P = np.asarray(points, dtype=float)
    n = len(P)
    lo, hi = int(head[0]), int(head[1])
    on = np.zeros(n, dtype=bool)
    on[max(lo, 0):hi + 1] = True
    on[int(exclude)] = False
    nodes = np.flatnonzero(on)
    if n < 3 or not len(nodes):
        return NO_FOLD
    ds = np.hypot(*np.diff(P, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    spacing = np.empty(n)
    spacing[0], spacing[-1] = ds[0], ds[-1]
    spacing[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    radii = search_factor * np.maximum(stop, 2.0 * stop_elements * spacing)
    # the pairs within the radius of their head node, and within that of their other node
    near = cKDTree(P).query_ball_point(P[nodes], radii[nodes])
    i = np.repeat(nodes, [len(x) for x in near])
    j = np.concatenate(near).astype(int) if len(i) else np.zeros(0, dtype=int)
    reach = cKDTree(P[nodes]).query_ball_point(P, radii)
    j2 = np.repeat(np.arange(n), [len(x) for x in reach])
    i2 = nodes[np.concatenate(reach).astype(int)] if len(j2) else np.zeros(0, dtype=int)
    i, j = np.concatenate([i, i2]), np.concatenate([j, j2])
    D = np.hypot(*(P[i] - P[j]).T)
    keep = (i != j) & (D <= np.maximum(radii[i], radii[j]))
    fold = keep & (ratio * D < np.abs(arc[i] - arc[j]))
    if not fold.any():
        return NO_FOLD
    score = D / np.maximum(stop, stop_elements * (spacing[i] + spacing[j]))
    k = np.flatnonzero(fold)[np.argmin(score[fold])]
    a, b = sorted((int(i[k]), int(j[k])))
    m = 0.5 * (P[a] + P[b])
    return FoldGap(float(np.hypot(*(P[a] - P[b]))), float(m[0]), float(m[1]), (a, b),
                   float(arc[b] - arc[a]), float(spacing[a] + spacing[b]), False)


def _neck_fields(x: dict) -> dict:
    keys = ("chain", "index", "r_n", "z_n", "depth", "shoulder", "h_loc", "s", "ratio", "status",
            "head", "side", "direction", "z_tip", "head_volume", "head_volume_over_s3")
    return {k: x[k] for k in keys if k in x}


def _request(kind: str, cls: str, **fields) -> dict:
    return {"reason": EVENT_REQUEST, "kind": kind, "class": cls, "handler": HANDLERS[cls],
            **fields}


def post_cut_events(chains, cut, settings, h_min: float, events: EventSettings, *,
                    neck_elements: float, fold_ratio: float, fold_stop: float,
                    fold_stop_elements: float, fold_search: float, apexes: dict | None,
                    guard: dict | None, invalid=None) -> dict | None:
    """The stops and events of a state after the cut with the events block, in their order
    (``reconnect.post_cut_stops`` with ``events``; its arguments).

    1. A state that is not valid, or that no step may start from, stops the run: the chains not
       two (``topology-changed``), chains that touch or a chain whose nodes have crossed (raised),
       a spline through the nodes that contacts itself (``self-contact-approach``, trigger
       ``spline-contact``), and on a freshly built mesh (``guard["fresh"]``) a tracked liquid
       neck narrower than ``liquid.resolution`` element lengths at it
       (``liquid-neck-unresolved``), the guard of the interface (``interface-unresolved``) and
       ``invalid()``, the check of the curved elements of a fresh mesh the driver passes. On a
       mesh a step has run on the guard comes last, after the events, as in the cut's own stops:
       an event's handler rebuilds the mesh.
    2. The end of the window: the bubble emptied below ``bubble_stop`` of its volume at the cut
       (``bubble-vanishing``, ``window_end``).
    3. The events, each returned as a request (reason ``event-request``, its ``kind``, ``class`` and
       the ``handler`` its class is to have, :data:`HANDLERS`, None for ``L-unhandled``): the
       bubble's two tips within ``max(delta, 3 h_min)``, the downward jet piercing the bubble
       (``piercing``, class ``W1``); a landed liquid neck, the one nearest its landing first
       (``liquid-neck``: class ``L``, or ``L-unhandled`` where the neck has no single head or one
       too large; with ``liquid.enabled`` false, a liquid neck at the cut's secondary-pinch test
       instead, ``L-unhandled``, its head never classified); a closing fold, one whose gap lies
       below ``max(kappa_g delta, fold_stop_elements h_elem)`` (``gap``), among the fold of each
       chain and that of each tracked head (:func:`head_fold`): first one that reaches the head of
       a tracked neck with ``r_n <= eps_max s`` and lands that neck (``a-head``, class ``A-head``,
       or ``L-unhandled`` for a head too large; the narrowest ``r_n/s`` first), on either chain,
       then one at the scale of the mesh (``contact`` ``kink``), which is no event and stops the
       run as the cut's stops do (``self-contact-approach``, or ``annular-contact-approach``
       across the base of a jet, trigger ``fold-gap``), then the closing groove closest to its
       limit (``groove``, class ``A-groove``, with the head's neck as ``a_head`` where one is
       reached); each with the jet whose base its pair straddles; a gas neck at the cut's
       secondary-pinch test (``gas-neck``, class ``G``); the bridge closing (``bridge-closing``,
       ``B``) and the chains approaching (``chain-contact``, ``X``)."""
    from .reconnect import (Chain, chains_disjoint, chain_necks, edge_reading, fold_contact,
                            interface_guard, revolved_volume, spline_contact_stop)
    from .selfcontact import InterfaceSelfIntersection, fold_gap

    if len(chains) != 2:
        return {"reason": "topology-changed", "chains": len(chains)}
    bubble = next(c for c in chains if not c.is_open)
    cavity = next(c for c in chains if c.is_open)
    named = (("bubble", bubble), ("open", cavity))
    ls = events.liquid
    # 1. the state itself
    chain_gap = chains_disjoint(chains)  # raises where two chains touch or cross
    stop_abs = fold_stop_after_cut(events, settings.delta, fold_stop)
    folds = []
    for name, c in named:
        fold = fold_gap(c.points[:, 0], c.points[:, 1], ratio=fold_ratio, stop=stop_abs,
                        stop_elements=fold_stop_elements, search_factor=fold_search)
        if fold.crossed:
            what = "bubble" if name == "bubble" else "open chain"
            raise InterfaceSelfIntersection(
                f"the {what}'s free surface crosses itself after the cut")
        folds.append((name, fold))
    stride = 2 if settings.rebuild_spline == "vertices" else 1
    stop = spline_contact_stop([(name, Chain(c.points[::stride], c.end_types))
                                if stride > 1 and len(c.points) % 2 == 1 else (name, c)
                                for name, c in named])
    if stop is not None:
        return stop
    necks = liquid_necks(named, ls) if ls.enabled else []
    fresh = bool(guard["fresh"]) if guard is not None else False
    if fresh:
        unresolved = [x for x in necks if x["r_n"] < ls.resolution * x["h_loc"]]
        if unresolved:
            x = min(unresolved, key=lambda x: x["r_n"] / x["h_loc"])
            return {"reason": "liquid-neck-unresolved", **_neck_fields(x),
                    "limit": ls.resolution * x["h_loc"], "resolution": ls.resolution,
                    "fresh": True}

    def guarded():  # the guard of the interface on its quadratic elements
        return interface_guard({name: edge_reading(c.points) for name, c in named}, settings,
                               fresh=fresh, h_floor=float(guard["h_floor"]),
                               refine_ratio=float(guard["refine_ratio"]))["stop"]

    if guard is not None and fresh:  # a mesh no step has run on: a rebuild that is not valid
        stop = guarded()
        if stop is not None:
            return stop
    if invalid is not None:
        stop = invalid()
        if stop is not None:
            return stop
    # 2. the end of the window
    vb = revolved_volume(bubble.points)
    if vb < settings.bubble_stop * cut.bubble_volume:
        return {"reason": "bubble-vanishing", "bubble_volume": vb, "window_end": True}
    # 3. the events
    zb = sorted([float(bubble.points[0, 1]), float(bubble.points[-1, 1])])
    zc = cavity.axis_ends()[0][1]
    length = zb[1] - zb[0]
    limit = max(settings.delta, 3.0 * h_min)
    if length < limit:
        return _request("piercing", "W1", bubble_length=length, limit=limit)
    neck_limit = max(settings.delta, neck_elements * h_min)

    def secondary(side):  # the cut's test of a new neck (reconnect.post_cut_stops)
        for name, c in named:
            for nk in chain_necks(c.points, c.end_types):
                if nk["side"] == side and nk["r"] < neck_limit and (
                        nk["depth"] >= settings.neck_depth or nk["r"] < neck_elements * h_min):
                    return {"chain": name, "r_neck": nk["r"], "z_neck": nk["z"],
                            "index": int(nk["index"]), "neck_limit": neck_limit,
                            "neck_shoulder": nk["shoulder"], "neck_depth": nk["depth"],
                            "neck_side": side}
        return None

    if ls.enabled:
        landed = sorted((x for x in necks if x["landed"]), key=lambda x: x["ratio"])
        if landed:
            x = landed[0]
            return _request("liquid-neck", "L" if x["status"] == "head" else "L-unhandled",
                            eps_d=ls.eps_d, **_neck_fields(x))
    else:
        found = secondary("liquid")
        if found is not None:  # no head classified: none to cut
            return _request("liquid-neck", "L-unhandled", detected_by="secondary-pinch", **found)
    # the closing folds: the one of each chain (fold_gap's, closest to its limit) and, so that a
    # closer fold elsewhere on its chain hides none, the one of each tracked head
    closing = {}

    def offer(name, fold):
        fold_limit = max(stop_abs, fold_stop_elements * fold.h_elem)
        if math.isfinite(fold.gap) and fold.gap < fold_limit:
            closing.setdefault((name, tuple(fold.pair)), (fold.gap / fold_limit, name, fold,
                                                          fold_limit))

    for name, fold in folds:
        offer(name, fold)
    points = {name: c.points for name, c in named}
    for x in necks:
        if x.get("head") is not None:
            offer(x["chain"], head_fold(points[x["chain"]], x["head"], x["index"],
                                        ratio=fold_ratio, stop=stop_abs,
                                        stop_elements=fold_stop_elements,
                                        search_factor=fold_search))
    lands, kinks, grooves = [], [], []
    for score, name, fold, fold_limit in closing.values():
        rec = {"chain": name, "contact": fold_contact(fold.arc, fold.h_elem),
               "fold_gap": fold.gap, "fold_r": fold.r, "fold_z": fold.z, "fold_arc": fold.arc,
               "fold_h_elem": fold.h_elem, "fold_limit": fold_limit,
               "fold_pair": list(fold.pair), "kappa_g": events.gap.kappa_g}
        a, b = fold.pair
        jet = next((x for x in (apexes or {}).get(name, []) if a < int(x["index"]) < b), None)
        if jet is not None:
            rec.update(jet=jet["tip"], apex_index=int(jet["index"]),
                       apex_r=jet.get("r", math.nan), apex_z=jet.get("z", math.nan),
                       R_apex=jet.get("R_apex", math.nan),
                       groove_angle=jet.get("groove_angle", math.nan))
        if rec["contact"] == "kink":
            # a contact at the scale of the mesh, not a groove of the resolved surface: no event
            # (nothing to fill or cut), the stop of the cut's own stops
            reason = "self-contact-approach" if jet is None else "annular-contact-approach"
            kinks.append((score, {"reason": reason, "trigger": "fold-gap", **rec}))
            continue
        head = head_of_fold(fold.pair, [x for x in necks if x["chain"] == name])
        if head is not None:
            rec["a_head"] = {**_neck_fields(head), "r_n_over_s": head["r_n"] / head["s"],
                             "eps_max": ls.eps_max}
            if head["r_n"] <= ls.eps_max * head["s"]:  # the head's neck lands
                lands.append(((head["r_n"] / head["s"], score), head, rec))
                continue
        grooves.append((score, rec))
    if lands:  # a fold that lands a head's neck before any groove, the narrowest r_n/s first
        _, head, rec = min(lands, key=lambda x: x[0])
        return _request("a-head", "A-head" if head["status"] == "head" else "L-unhandled", **rec)
    if kinks:
        return min(kinks, key=lambda x: x[0])[1]
    if grooves:  # the groove closest to its limit, on either chain
        return _request("groove", "A-groove", **min(grooves, key=lambda x: x[0])[1])
    found = secondary("gas")
    if found is not None:
        return _request("gas-neck", "G", **found)
    bridge = zc - zb[1]
    limit = max(settings.bridge_stop * cut.bridge, 3.0 * h_min)
    if bridge < limit:
        return _request("bridge-closing", "B", bridge=bridge, limit=limit)
    if chain_gap < 3.0 * h_min:
        return _request("chain-contact", "X", chain_gap=chain_gap, limit=3.0 * h_min)
    if guard is not None and not fresh:
        # a mesh a step has run on: the guard last, as in the cut's own stops (an event's handler
        # rebuilds the mesh)
        return guarded()
    return None


# --- the cut of a liquid neck: the head removed, the stump closed by a cap ------------------------

CAP_MIN_POINTS = 16  # points of a cap, at least
# the cap points within this many of its intervals (r_n / elements_per_scale of arc) of the neck,
# which leave the neck tangentially, need not clear it; with the spacing of the neck itself, the
# three nearest the neck
CAP_SKIP = 3
CAP_CLEARANCE = 1e-3  # every other cap point lies at least this times r_n inside the old liquid
CAP_BACKTRACK = 1e-9  # a cap may come back towards its tip by at most this times r_n
CAP_ANGLE_NODES = 4  # interior angles are read over the splice and this many nodes on either side
CAP_RETRY_HEIGHT = 0.5  # the height of the cap of the one retry of a refused cap, in neck radii
# from the neck the spacing of the cap's points grows (or shrinks) by at most this factor per point
# from the neck's node spacing to its own: the uniform Catmull-Rom spline a mesh is built on loops
# where the spacing of its points jumps, and the nodes at a neck can lie many times closer than a
# cap at r_n / elements_per_scale
CAP_GROWTH = 1.25


class CapRefused(RuntimeError):
    """The cap of a liquid cut fails a hard check before anything has changed: ``checks`` holds what
    was read, with the checks that failed under ``failed``."""

    def __init__(self, message: str, checks: dict | None = None):
        super().__init__(message)
        self.checks = dict(checks or {})


def _quarter_ellipse(r_n: float, eta: float, m: int = 4097) -> tuple[np.ndarray, np.ndarray]:
    """The parameter ``t`` of ``r = r_n cos t``, ``z = eta r_n sin t`` on ``[0, pi/2]`` and the
    arclength from ``t = 0`` (the neck) at it."""
    t = np.linspace(0.0, 0.5 * math.pi, m)
    d = np.hypot(np.diff(r_n * np.cos(t)), np.diff(eta * r_n * np.sin(t)))
    return t, np.concatenate([[0.0], np.cumsum(d)])


def cap_arc_length(r_n: float, eta: float) -> float:
    """Arclength of the cap of :func:`cap_at`, a quarter of the ellipse ``r = r_n cos t``,
    ``z = eta r_n sin t``: ``pi r_n / 2`` for ``eta = 1``."""
    return float(_quarter_ellipse(r_n, eta)[1][-1])


def cap_point_count(r_n: float, eta: float, elements_per_scale: float) -> int:
    """The intervals of a cap of radius ``r_n`` at its own spacing: one per ``r_n /
    elements_per_scale`` of its arc, the size the mesh gives the tip it makes, and at least
    ``CAP_MIN_POINTS``."""
    return max(CAP_MIN_POINTS,
               int(math.ceil(cap_arc_length(r_n, eta) / (r_n / elements_per_scale))))


def cap_at(r_n: float, z_n: float, direction: float, eta: float, u) -> np.ndarray:
    """The points of the cap that closes the stump of a cut liquid neck at the arclengths ``u`` from
    the neck ``(r_n, z_n)`` along it: ``r = r_n cos t``, ``z = z_n + direction eta r_n sin t``, the
    point at the cap's whole arclength put on the axis exactly. With ``eta = 1`` a half sphere
    centred on the axis at the neck's height: it leaves the neck tangentially, where the surface of
    a neck, a minimum of ``r``, is parallel to the axis, and it lies within the neck's radius of its
    centre, inside the old liquid as long as the head reaches beyond it; ``eta`` below one flattens
    it. A cut spaces the points from the neck's spacing (:func:`cap_spacing`)."""
    t, s = _quarter_ellipse(r_n, eta)
    u = np.asarray(u, dtype=float)
    theta = np.interp(u, s, t)
    cap = np.column_stack([r_n * np.cos(theta), z_n + direction * eta * r_n * np.sin(theta)])
    cap[u >= s[-1]] = (0.0, z_n + direction * eta * r_n)
    return cap


def cap_spacing(length: float, h_start: float, h_cap: float,
                growth: float = CAP_GROWTH) -> np.ndarray:
    """The arclengths of a cap's points from the neck (excluded) to its tip at ``length``
    (included): the first ``h_start`` from the neck (its node spacing there), each interval
    ``growth`` times longer or shorter than the one before it until it is ``h_cap``, the cap's own
    spacing, then ``h_cap``; the last between half and one and a half times the one before it.
    Raises ``ValueError`` unless ``h_start`` and ``h_cap`` are positive."""
    if not (h_start > 0.0 and h_cap > 0.0 and length > 0.0):
        raise ValueError(f"a cap needs a positive length and spacings, not {length!r}, "
                         f"{h_start!r} and {h_cap!r}")
    u, h = [0.0], float(h_start)
    while length - u[-1] > 1.5 * h:
        u.append(u[-1] + h)
        h = min(h * growth, h_cap) if h < h_cap else max(h / growth, h_cap)
    u.append(float(length))
    return np.asarray(u[1:])


def revolved_area(points) -> float:
    """Area of the surface of revolution of a polyline about the axis, each segment a frustum."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(P) < 2:
        return 0.0
    d = np.diff(P, axis=0)
    return float(np.sum(math.pi * (P[:-1, 0] + P[1:, 0]) * np.hypot(d[:, 0], d[:, 1])))


def gas_sections(chains, reservoir_depth: float) -> list[np.ndarray]:
    """The half-sections of the gas the chains bound (``reconnect.closed_section``: a closed chain
    closed along the axis, the open chain through the gas above the free surface)."""
    from .reconnect import closed_section

    return [closed_section(c.points, c.end_types, reservoir_depth) for c in chains]


def liquid_clearance(points, sections) -> np.ndarray:
    """The signed distance of each point from the gas the ``sections`` bound (:func:`gas_sections`):
    positive in the liquid, outside every section, negative inside one."""
    import shapely
    from shapely.geometry import Polygon

    P = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = shapely.points(P)
    out = np.full(len(P), np.inf)
    for section in sections:
        poly = Polygon(section)
        d = np.asarray(shapely.distance(poly.boundary, pts), dtype=float)
        inside = np.asarray(shapely.contains_xy(poly, P[:, 0], P[:, 1]), dtype=bool)
        out = np.minimum(out, np.where(inside, -d, d))
    return out


TIP_OF_END = {("bubble", 0): "bubble_bottom", ("bubble", -1): "bubble_top",
              ("open", 0): "cavity_tip"}  # the tip on the axis a head ends on (reconnect.JET_TIPS)


@dataclass
class LiquidCut:
    """The chain of a liquid cut (:func:`truncate_chain`): the head beyond the neck removed and the
    stump closed by a cap. ``points`` is the new chain in the order of the old one, ``origin`` the
    old node each point is (``-1`` on the cap), ``cap`` the cap from next to the neck to its tip
    on the axis, ``removed`` the half-section of the liquid removed (``Q``: the head from the neck
    to its tip on the axis, the axis up to the cap's tip, the cap back to the neck), of volume
    ``V_rem``; ``A_rem`` the area of the head's surface, ``A_cap`` that of the cap, ``L_rem`` the
    reach of ``Q`` along the axis from the neck's height; ``checks`` the hard checks as read."""
    chain: str
    index: int
    end: int  # 0 or -1: the end of the chain where the head was, and the cap is
    tip: str  # the tip on the axis of that end (reconnect.JET_TIPS)
    direction: float  # sign(z_tip - z_n)
    r_n: float
    z_n: float
    s: float
    eta: float
    points: np.ndarray
    origin: np.ndarray
    cap: np.ndarray
    removed: np.ndarray
    V_rem: float
    A_rem: float
    A_cap: float
    L_rem: float
    z_tip_old: float
    z_tip_new: float
    checks: dict

    def summary(self) -> dict:
        """What a record keeps of the cut, without its point arrays: ``tip_shift`` is how far the
        chain's tip on the axis moved towards the neck, the piece of the axis the gas gains."""
        return {"chain": self.chain, "index": self.index, "end": self.end, "tip": self.tip,
                "direction": self.direction, "r_n": self.r_n, "z_n": self.z_n, "s": self.s,
                "eta": self.eta, "cap_points": len(self.cap), "V_rem": self.V_rem,
                "A_rem": self.A_rem, "A_cap": self.A_cap, "L_rem": self.L_rem,
                "z_tip_old": self.z_tip_old, "z_tip_new": self.z_tip_new,
                "tip_shift": abs(self.z_tip_old - self.z_tip_new), "checks": dict(self.checks)}


def truncate_chain(named, neck: dict, *, eta: float, elements_per_scale: float,
                   min_angle: float, reservoir_depth: float) -> LiquidCut:
    """The chain of a liquid cut at a tracked neck with a single head (:func:`liquid_necks`, status
    ``head``) of the ``(name, chain)`` pairs ``named``: the head side removed, the neck node kept as
    the splice, and the stump closed by the cap of :func:`cap_at` of height ``eta``, its own spacing
    one :func:`cap_point_count` interval (``elements_per_scale`` elements per neck radius) reached
    from the neck's node spacing by :func:`cap_spacing`. Raises :class:`CapRefused` unless the hard
    checks hold: the cap has at least ``CAP_MIN_POINTS`` points (not where the nodes of the neck lie
    so far apart that the cap's spacing cannot be reached from theirs within its arc) and ends short
    of the head's tip; (i) every cap point but those within ``CAP_SKIP`` of its own intervals of the
    neck lies in the old liquid at least ``CAP_CLEARANCE r_n`` from the gas
    (:func:`liquid_clearance`); (ii) the cap, from its tip through the neck and two nodes beyond,
    comes back towards its tip by at most ``CAP_BACKTRACK r_n`` (``reconnect.cap_backtrack``); (iii)
    the interior angles over the neck and ``CAP_ANGLE_NODES`` nodes on either side are at least
    ``min_angle``; (iv) the removed half-section is a simple polygon of positive volume; (v) the new
    chains pass ``reconnect.check_chains`` (simple, disjoint, one open, their splines too). The
    chart, the last check, is the plan's (:func:`build_liquid_plan`)."""
    from shapely.geometry import LinearRing

    from .reconnect import Chain, cap_backtrack, check_chains, interior_angles, revolved_volume

    chains = dict(named)
    name = neck.get("chain")
    if name not in chains:
        raise CapRefused(f"no chain {name!r} to cut")
    if neck.get("status") != "head" or neck.get("head") is None:
        raise CapRefused(f"the neck has no single head to remove (status {neck.get('status')!r})")
    c = chains[name]
    P = np.asarray(c.points, dtype=float)
    n, i = len(P), int(neck["index"])
    r_n, z_n, s = float(neck["r_n"]), float(neck["z_n"]), float(neck["s"])
    if not (0 < i < n - 1) or abs(P[i, 0] - r_n) > 1e-12 * max(abs(r_n), 1e-300) \
            or abs(P[i, 1] - z_n) > 1e-12 * max(abs(z_n), 1.0):
        raise CapRefused("the neck is not a node of the chain it names")
    lo, hi = (int(x) for x in neck["head"])
    if (lo, hi) == (i, n - 1) and c.end_types[1] == "axis":
        end = -1
    elif (lo, hi) == (0, i) and c.end_types[0] == "axis":
        end = 0
    else:
        raise CapRefused(f"the head [{lo}, {hi}] of the neck at node {i} does not end on the axis")
    direction = float(np.sign(P[end, 1] - z_n))
    if direction == 0.0 or direction != float(neck.get("direction", direction)):
        raise CapRefused("the head's direction from the neck is not the neck's")
    body = P[:i + 1][::-1] if end == -1 else P[i:]  # from the neck away from the head
    head_path = P[i:] if end == -1 else P[i::-1]  # from the neck to the head's tip on the axis
    length = cap_arc_length(r_n, eta)
    h_cap = length / cap_point_count(r_n, eta, elements_per_scale)
    u = cap_spacing(length, float(np.hypot(*(body[1] - body[0]))), h_cap)
    cap = cap_at(r_n, z_n, direction, eta, u)
    n_cap = len(cap)
    if end == -1:
        points = np.vstack([P[:i + 1], cap])
        origin = np.concatenate([np.arange(i + 1), np.full(n_cap, -1)])
    else:
        points = np.vstack([cap[::-1], P[i:]])
        origin = np.concatenate([np.full(n_cap, -1), np.arange(i, n)])
    removed = np.vstack([head_path, cap[::-1]])
    z_tip_old, z_tip_new = float(head_path[-1, 1]), float(cap[-1, 1])
    checks = {"eta": eta, "points": n_cap, "spacing": h_cap, "first_spacing": float(u[0]),
              "skip_arc": CAP_SKIP * h_cap, "failed": []}
    failed = checks["failed"]
    if n_cap < CAP_MIN_POINTS:
        failed.append("points")  # the neck's nodes are too far apart for a cap at r_n
    if not direction * (z_tip_old - z_tip_new) > 0.0:
        failed.append("tip")  # the head ends within the cap
    # (i) the cap inside the old liquid, but where it leaves the neck tangentially
    clearance = liquid_clearance(cap, gas_sections([ch for _, ch in named], reservoir_depth))
    near = u <= checks["skip_arc"] * (1.0 + 1e-12)
    checks.update(clearance=float(clearance[~near].min()) if (~near).any() else math.inf,
                  clearance_limit=CAP_CLEARANCE * r_n,
                  skipped=int(near.sum()),
                  clearance_skipped=float(clearance[near].min()) if near.any() else math.nan)
    if not checks["clearance"] >= checks["clearance_limit"]:
        failed.append("inside")
    # (ii) monotone from its tip, through the neck and two nodes beyond it
    checks["backtrack"] = cap_backtrack(np.vstack([cap[::-1], body[:3]]))
    checks["backtrack_limit"] = CAP_BACKTRACK * r_n
    if not checks["backtrack"] <= checks["backtrack_limit"]:
        failed.append("backtrack")
    # (iii) no knife-edge angle about the splice: the inner angles of the polyline from one node
    # beyond CAP_ANGLE_NODES before the splice to one beyond as many after it
    around = np.vstack([body[CAP_ANGLE_NODES + 1:0:-1], body[:1], cap[:CAP_ANGLE_NODES + 1]])
    angles = interior_angles(around)
    checks.update(min_angle=float(angles.min()) if len(angles) else math.nan,
                  min_angle_limit=min_angle)
    if not checks["min_angle"] >= min_angle:
        failed.append("angle")
    # (iv) the removed half-section
    V_rem = revolved_volume(removed)
    checks["removed_simple"] = bool(LinearRing(removed).is_simple)
    if not (checks["removed_simple"] and V_rem > 0.0):
        failed.append("removed")
    # (v) the new chains, as a mesh is built on them
    new = [(k, Chain(points, c.end_types) if k == name else ch) for k, ch in named]
    try:
        check_chains([ch for _, ch in new])
        checks["chains"] = "ok"
    except RuntimeError as e:  # InterfaceSelfIntersection is one
        checks["chains"] = f"{type(e).__name__}: {e}"
        failed.append("chains")
    if failed:
        raise CapRefused(f"the cap of the liquid cut fails its checks: {', '.join(failed)}",
                         checks)
    return LiquidCut(chain=name, index=i, end=end, tip=TIP_OF_END.get((name, end), name),
                     direction=direction, r_n=r_n, z_n=z_n, s=s, eta=eta, points=points,
                     origin=origin, cap=cap, removed=removed, V_rem=V_rem,
                     A_rem=revolved_area(head_path), A_cap=revolved_area(np.vstack([P[i:i + 1],
                                                                                   cap])),
                     L_rem=float(np.max(np.abs(removed[:, 1] - z_n))), z_tip_old=z_tip_old,
                     z_tip_new=z_tip_new, checks=checks)


# --- the plan of a liquid cut, for pyoomph's transfer ---------------------------------------------

CHART_ORDER = ("open", "bubble")  # the old chart, and its table, run over the open chain first
CHART_GAP = 1.0  # each chain's chart starts this far beyond the end of the one before it


def interface_chart(named) -> dict[str, np.ndarray]:
    """The chart of the plan of a liquid cut on the old chains (``(name, chain)`` pairs, ``open``
    and ``bubble``): the arclength along each chain from its far end, the open chain from its end on
    the wall and a chain from the axis to the axis from its lower end (as ``surgery.
    orient_interface_chart`` orders the problem's chart), each chain from ``CHART_GAP`` beyond the
    end of the one before it in ``CHART_ORDER``, so that the ranges of the chains are disjoint. By
    chain name, for the nodes in the order of the chain."""
    from .reconnect import FIXED

    chains = dict(named)
    out, start = {}, 0.0
    for name in CHART_ORDER:
        if name not in chains:
            continue
        c = chains[name]
        P = np.asarray(c.points, dtype=float)
        seg = np.hypot(*np.diff(P, axis=0).T)
        if c.end_types[0] == FIXED:
            last = False
        elif c.end_types[1] == FIXED:
            last = True  # the wall end is the last node
        else:  # a chain from the axis to the axis: from its lower end
            last = bool(P[-1, 1] < P[0, 1])
        if last:
            zeta = start + np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        else:
            zeta = start + np.concatenate([[0.0], np.cumsum(seg)])
        out[name] = zeta
        start = float(zeta.max()) + CHART_GAP
    if set(chains) - set(out):
        raise ValueError(f"a plan charts only the chains {CHART_ORDER}, not {sorted(chains)}")
    return out


@dataclass
class LiquidPlan:
    """The plan of a liquid cut (:func:`build_liquid_plan`): pyoomph's ``SurgeryPlan`` (``plan``),
    the old chart as a table ``(r, z, zeta)`` of every node of the old chains, open chain first
    (``table``), the new chains as the mesh is built on them (``chains``, by name, in the order of
    the old ones), and the range of the chart the cap takes (``cap_zeta``, the neck's and the node's
    two beyond it on the head's side)."""
    plan: object
    table: np.ndarray
    chains: dict
    cap_zeta: tuple


def build_liquid_plan(named, cut: LiquidCut, reservoir_depth: float) -> LiquidPlan:
    """The plan of the cut ``cut`` (:func:`truncate_chain`) of one of the old chains ``named``
    (``(name, chain)`` pairs; the chain positions those of the interface nodes, which the table
    must match), for pyoomph's remesh and transfer as its planner would park it: a ``NewChain`` per
    chain, open chain first, each ordered by its chart (increasing), the nodes it keeps with their
    old chart and the index of their row in the table (``origin``), the cap's points (``origin``
    -1) with a strictly increasing ramp of the chart from the neck's to that of the node two beyond
    it on the head's side, so that the interface's own fields of the cap come from the old surface
    just beyond the neck; the gas spans of the axis as ``surgery.cut`` sets them, the gas volumes of
    the chains before and after (the gas gains the removed volume, ``volume_lost_by_removal``) and
    one event, a pinch at the neck's height. Raises :class:`CapRefused` unless the chart of every
    new chain is strictly increasing and the ranges of the chains are disjoint."""
    from pyoomph.meshes.axisymm_topology import NewChain, ReconnectionEvent, SurgeryPlan

    from .reconnect import FIXED, Chain, chain_volumes

    chains = dict(named)
    chart = interface_chart(named)
    order = [name for name in CHART_ORDER if name in chains]
    offsets, k = {}, 0
    for name in order:
        offsets[name] = k
        k += len(chains[name].points)
    table = np.vstack([np.column_stack([np.asarray(chains[name].points, dtype=float),
                                        chart[name]]) for name in order])
    zc = chart[cut.chain]
    n_old = len(zc)
    beyond = cut.index + (2 if cut.end == -1 else -2)
    beyond = min(max(beyond, 0), n_old - 1)
    z0, z1 = float(zc[cut.index]), float(zc[beyond])
    new_chains, built, failed = [], {}, []
    for name in order:
        c = chains[name]
        if name == cut.chain:
            P = np.asarray(cut.points, dtype=float)
            origin = np.asarray(cut.origin, dtype=int)
            zeta = np.empty(len(P))
            kept = origin >= 0
            zeta[kept] = zc[origin[kept]]
            # the cap from the neck to its tip: the chart ramps by arclength from z0 to z1
            cap = np.vstack([[cut.r_n, cut.z_n], cut.cap])
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(cap, axis=0).T))])
            ramp = z0 + (z1 - z0) * arc[1:] / arc[-1]
            ramp[-1] = z1
            zeta[~kept] = ramp if cut.end == -1 else ramp[::-1]
            rows = np.where(kept, origin + offsets[name], -1)
        else:
            P = np.asarray(c.points, dtype=float)
            zeta = np.asarray(chart[name], dtype=float)
            rows = np.arange(len(P)) + offsets[name]
        seg = np.hypot(*np.diff(P, axis=0).T)
        sizes = 2.0 * np.r_[seg[0], 0.5 * (seg[:-1] + seg[1:]), seg[-1]]
        ends = tuple(c.end_types)
        reservoir = tuple(reservoir_depth if e == FIXED else None for e in ends)
        built[name] = Chain(P.copy(), ends)
        if zeta[-1] < zeta[0]:  # ordered by its chart
            P, zeta, rows, sizes = P[::-1], zeta[::-1], rows[::-1], sizes[::-1]
            ends, reservoir = ends[::-1], reservoir[::-1]
        if not np.all(np.diff(zeta) > 0.0):
            failed.append(name)
        new_chains.append(NewChain(points=P.copy(), sizes=sizes.copy(), zeta=zeta.copy(),
                                   origin=rows.copy(), end_types=ends, reservoir=reservoir))
    ranges = sorted((float(nc.zeta[0]), float(nc.zeta[-1])) for nc in new_chains)
    if failed or any(a[1] >= b[0] for a, b in zip(ranges, ranges[1:])):
        raise CapRefused("the chart of the plan of the liquid cut is not strictly increasing on "
                         f"every chain, with disjoint ranges ({failed or ranges})",
                         {"failed": ["chart"], "chart_ranges": ranges})
    olds = [chains[name] for name in order]
    news = [built[name] for name in order]
    spans = np.array([[c.points[0, 1], c.points[-1, 1] + (reservoir_depth if c.is_open else 0.0)]
                      for c in news])
    plan = SurgeryPlan(events=[ReconnectionEvent("pinch", float(cut.z_n))], new_chains=new_chains,
                       axis_spans_inside=spans, axis_spans_outside=np.zeros((0, 2)),
                       fragment_volumes_before=chain_volumes(olds, reservoir_depth),
                       fragment_volumes_after=chain_volumes(news, reservoir_depth),
                       volume_lost_by_removal=float(cut.V_rem))
    return LiquidPlan(plan=plan, table=table, chains=built, cap_zeta=(z0, z1))


# pyoomph's transfer places a new interface node on the old interface by projection onto the old
# element nearest it, accepted within this fraction of that element's size (the larger extent of
# its bounding box; pointlocator max_projection_offset_factor); farther from every old element it
# blends the two nearest old nodes instead
PROJECTION_OFFSET = 0.5
# the elements pyoomph's locator tries for a point: those of the old node nearest it and, where none
# takes it, those of the k old nodes nearest it, k doubling from 8 up to this many (its search
# passes); a point none of them takes is not located, however near a farther element lies
LOCATOR_NEIGHBOURS = 256
LOCATOR_FIRST = 8  # the old nodes whose elements the prediction tries first, for every new node


def predict_unlocated(nodes, old_chains, *, factor: float = PROJECTION_OFFSET,
                      neighbours: int = LOCATOR_NEIGHBOURS) -> np.ndarray:
    """Which of the new interface nodes ``nodes`` (``(M, 2)``) pyoomph's transfer will not place on
    the old interface (``old_chains``: the old chains' nodes, each a chain of quadratic elements,
    vertex, midside node, vertex): those farther than ``factor`` times its size from every old
    element it tries (the distance to its two chords; the size the larger extent of the bounding
    box of its three nodes), the elements of the ``neighbours`` old nodes nearest each node, as far
    as pyoomph's locator widens its search (:data:`LOCATOR_NEIGHBOURS`). The elements of the
    ``LOCATOR_FIRST`` nearest are tried first, the rest only for the nodes none of those takes."""
    from scipy.spatial import cKDTree

    X = np.asarray(nodes, dtype=float).reshape(-1, 2)
    elems, conn, points, base = [], [], [], 0
    for P in old_chains:
        P = np.asarray(P, dtype=float).reshape(-1, 2)
        n = len(P)
        if n < 2:
            continue
        idx = 2 * np.arange((n - 1) // 2)
        if len(idx):
            elems.append(np.stack([P[idx], P[idx + 1], P[idx + 2]], axis=1))
            conn.append(base + np.stack([idx, idx + 1, idx + 2], axis=1))
        if (n - 1) % 2:  # an even count: the last segment as an element of its own
            elems.append(np.stack([P[-2:-1], 0.5 * (P[-2:-1] + P[-1:]), P[-1:]], axis=1))
            conn.append(base + np.array([[n - 2, n - 1, n - 1]]))
        points.append(P)
        base += n
    if not elems or not len(X):
        return np.zeros(len(X), dtype=bool)
    E = np.concatenate(elems)  # (n_e, 3, 2)
    size = np.max(np.ptp(E, axis=1), axis=1)
    old = np.concatenate(points)  # every old node once
    # the elements of each old node (its star), -1 where it has fewer than the most
    pairs = np.unique(np.column_stack([np.concatenate(conn).ravel(),
                                       np.repeat(np.arange(len(E)), 3)]), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=len(old))
    star = np.full((len(old), int(counts.max())), -1)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    star[pairs[:, 0], np.arange(len(pairs)) - first[pairs[:, 0]]] = pairs[:, 1]
    tree = cKDTree(old)

    def taken(Y, k):  # whether an element of the k old nodes nearest each point takes it
        k = min(k, len(old))
        _, near = tree.query(Y, k=k)
        cand = star[np.asarray(near).reshape(len(Y), k)].reshape(len(Y), -1)
        out = np.zeros(len(Y), dtype=bool)
        for col in range(cand.shape[1]):
            e = np.maximum(cand[:, col], 0)
            best = np.full(len(Y), np.inf)
            for a, b in ((0, 1), (1, 2)):
                A, B = E[e, a], E[e, b]
                d = B - A
                L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
                t = np.clip(np.einsum("ij,ij->i", Y - A, d) / L2, 0.0, 1.0)
                best = np.minimum(best, np.hypot(*(Y - A - t[:, None] * d).T))
            out |= (cand[:, col] >= 0) & (best <= factor * size[e])
        return out

    located = taken(X, min(LOCATOR_FIRST, neighbours))
    rest = np.flatnonzero(~located)
    if len(rest) and neighbours > LOCATOR_FIRST:
        located[rest] = taken(X[rest], neighbours)
    return ~located


NECK_KEYS = ("chain", "index", "r_n", "z_n", "depth", "shoulder", "h_loc", "s", "ratio", "status",
             "head", "side", "direction", "z_tip", "head_volume", "head_volume_over_s3")


def event_neck(request: dict) -> dict:
    """The tracked neck an event request lands (:func:`post_cut_events`): the request's own fields
    for a landed liquid neck (``liquid-neck``), its ``a_head`` for a fold that reaches a head
    (``a-head``)."""
    if request.get("kind") == "a-head":
        return {k: v for k, v in (request.get("a_head") or {}).items() if k in NECK_KEYS}
    return {k: request[k] for k in NECK_KEYS if k in request}


# --- the events a run makes: their tips, the cascade of events at a tip, the ghosts --------------

# the tip on the axis whose head the ghost of a cut moves towards, through the gas of the bubble;
# the head of the cavity's tip rises into the open gas, which ends on no tip
FAR_TIPS = {"bubble_top": "bubble_bottom", "bubble_bottom": "bubble_top", "cavity_tip": None}
GHOST_SKIP_STEPS = 2  # a ghost's recontacts are read from this many steps after its cut on


def event_tip(neck: dict) -> str:
    """The tip on the axis a tracked neck's head ends on (:func:`liquid_necks`: its chain and
    side, ``before`` the neck the chain's first end, ``after`` its last): ``cavity_tip``,
    ``bubble_bottom`` or ``bubble_top``."""
    end = 0 if neck.get("side") == "before" else -1
    return TIP_OF_END.get((neck.get("chain"), end), str(neck.get("chain")))


class CascadeMonitor:
    """The events made at each tip on the axis, in their order (their generation at the tip), and
    the verdict on the next: a cascade of events that shrink stops the run, one that grows or
    stays stationary is recorded.

    Of the events at a tip, ``interval`` is the time since the one before (``delta^2`` and
    ``T = t_p - t_cut`` give it in units of the cut radius, ``interval_over_delta2``, and as
    events per ``T``, ``events_per_T``) and ``V`` the volume of its head. From the third event
    at a tip each has a pair of ratios to the one before, of their intervals and of their
    volumes. The verdict on an event (:meth:`check`, before it is made):

    * ``shrink``: its interval is below ``min_interval delta^2``, or the last ``count`` pairs of
      ratios (its own the last) all lie below ``ratio``, both of the interval and of the volume;
    * ``grow``: the last ``count`` pairs all lie above ``1/ratio``, both;
    * ``stationary``: neither, with ``count`` pairs at least;
    * ``undetermined`` with fewer, and ``first`` for the first event at its tip."""

    def __init__(self, *, delta: float, min_interval: float, ratio: float, count: int,
                 T: float = math.nan):
        self.delta, self.min_interval, self.ratio, self.count = delta, min_interval, ratio, count
        self.T = T
        self.history: dict[str, list[dict]] = {}

    @classmethod
    def for_run(cls, events: EventSettings, delta: float, T: float = math.nan,
                records=()) -> CascadeMonitor:
        """The monitor of a run with the block ``events`` at cut radius ``delta``, with the events
        the run has made (``records``: their tip, time and head volume) in it."""
        out = cls(delta=delta, min_interval=events.cascade_min_interval,
                  ratio=events.cascade_ratio, count=events.cascade_count, T=T)
        for rec in records or ():
            trigger = rec.get("trigger") or {}
            if rec.get("outcome", "made") == "made" and rec.get("tip") is not None:
                out.add(rec["tip"], int(rec.get("k", 0)), float(rec["t"]),
                        float(trigger.get("head_volume", math.nan)))
        return out

    def check(self, tip: str, t: float, V: float) -> dict:
        """The verdict on an event at ``tip`` at time ``t`` with a head of volume ``V``, before it
        is made, with its generation at the tip, its interval and the ratios of its pair."""
        h = self.history.get(tip, [])
        out = {"tip": tip, "generation": len(h) + 1, "interval": math.nan,
               "interval_over_delta2": math.nan, "events_per_T": math.nan,
               "interval_ratio": math.nan, "volume_ratio": math.nan, "shrinking": 0,
               "growing": 0, "verdict": "first"}
        if not h:
            return out
        times = [x["t"] for x in h] + [float(t)]
        vols = [x["V"] for x in h] + [float(V)]
        gaps = [b - a for a, b in itertools.pairwise(times)]
        dt = gaps[-1]
        out.update(interval=dt, interval_over_delta2=dt / self.delta**2,
                   events_per_T=abs(self.T) / dt if math.isfinite(self.T) and dt > 0.0
                   else math.nan)
        pairs = []  # (interval ratio, volume ratio) of each event from the third at the tip on
        for j in range(1, len(gaps)):
            ri = gaps[j] / gaps[j - 1] if gaps[j - 1] > 0.0 else math.inf
            rv = vols[j + 1] / vols[j] if vols[j] > 0.0 else math.nan
            pairs.append((ri, rv))
        if pairs:
            out.update(interval_ratio=pairs[-1][0], volume_ratio=pairs[-1][1])
        for key, test in (("shrinking", lambda p: p[0] < self.ratio and p[1] < self.ratio),
                          ("growing", lambda p: p[0] > 1.0 / self.ratio
                           and p[1] > 1.0 / self.ratio)):
            n = 0
            for p in reversed(pairs):
                if not test(p):
                    break
                n += 1
            out[key] = n
        if dt < self.min_interval * self.delta**2 or out["shrinking"] >= self.count:
            out["verdict"] = "shrink"
        elif out["growing"] >= self.count:
            out["verdict"] = "grow"
        else:
            out["verdict"] = "stationary" if len(pairs) >= self.count else "undetermined"
        return out

    def add(self, tip: str, k: int, t: float, V: float) -> dict:
        """Record the event ``k`` at ``tip``; returns the verdict :meth:`check` gave it."""
        verdict = self.check(tip, t, V)
        self.history.setdefault(tip, []).append({"k": int(k), "t": float(t), "V": float(V),
                                                 "verdict": verdict})
        return verdict

    def verdicts(self) -> dict:
        """The verdict on the last event at each tip, as :meth:`check` gave it."""
        return {tip: dict(h[-1]["verdict"]) for tip, h in self.history.items() if h}


def ghost_place(ghost: dict, t: float) -> tuple[float, float]:
    """The ends on the axis of the ghost of a removed head (a liquid cut's ``ghost``) at time
    ``t``: it moves rigidly at its mean axial speed ``U``, so that its front, the head's old tip,
    is at ``z_front0 + U (t - t0)``, while its rear, at the stump's new tip at the cut, retracts
    towards the head at ``v_ret`` besides, ``z_rear0 + (U + direction v_ret) (t - t0)``."""
    dt = float(t) - float(ghost["t0"])
    U, d = float(ghost["U"]), float(ghost["direction"])
    return (float(ghost["z_front0"]) + U * dt,
            float(ghost["z_rear0"]) + (U + d * float(ghost["v_ret"])) * dt)


def ghost_recontacts(ghost: dict, t: float, step: int, tips: dict) -> list[dict]:
    """The recontacts of a ghost (:func:`ghost_place`) at time ``t`` and step ``step`` with the
    tips on the axis (``tips``: the heights of ``cavity_tip``, ``bubble_bottom``,
    ``bubble_top``), each kind once and only from ``GHOST_SKIP_STEPS`` steps after the cut
    (``step0``) on: ``rear`` when the tip of the stump (``tip``) has passed the ghost's rear,
    ``direction (z_rear - z_stump) < 0``, the stump catching up with the head it lost; ``front``
    when the ghost's front has reached the tip ahead of it (``far_tip``, :data:`FAR_TIPS`: the
    other end of the bubble), ``direction (z_front - z_far) >= 0``. Flags only: nothing stops."""
    if int(step) < int(ghost.get("step0", 0)) + GHOST_SKIP_STEPS:
        return []
    done = {x.get("kind") for x in ghost.get("recontacts") or ()}
    front, rear = ghost_place(ghost, t)
    d = float(ghost["direction"])
    out = []
    z_stump = tips.get(ghost.get("tip"), math.nan)
    if "rear" not in done and math.isfinite(z_stump) and d * (rear - z_stump) < 0.0:
        out.append({"kind": "rear", "t": float(t), "step": int(step), "z_ghost": rear,
                    "tip": ghost.get("tip"), "z_tip": float(z_stump)})
    far = ghost.get("far_tip")
    z_far = tips.get(far, math.nan) if far is not None else math.nan
    if "front" not in done and math.isfinite(z_far) and d * (front - z_far) >= 0.0:
        out.append({"kind": "front", "t": float(t), "step": int(step), "z_ghost": front,
                    "tip": far, "z_tip": float(z_far)})
    return out
