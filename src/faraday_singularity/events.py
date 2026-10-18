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
  the mesh at the necks, oomph-lib's smallest step after the cut, the open chain's neck stop and the
  sizing of the bubble's bottom.

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
neck lands that neck once ``r_n <= liquid.eps_max s`` (:func:`head_of_fold`).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

EVENTS_BLOCK = "reconnect_events"  # the key of the block in a case's collapse block
OOMPH_MINIMUM_DT = 1e-12  # oomph-lib's smallest step by default (Problem::Minimum_dt)
BUBBLE_BOTTOM_SIZINGS = ("kept", "frozen")
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
    (class ``L-unhandled``, ``detected_by`` ``secondary-pinch``), its head never classified."""
    enabled: bool = False
    eps_d: float = 0.2
    eps_max: float = 0.5
    track_depth: float = 0.05
    max_head_volume: float = 20.0
    resolution: float = 4.0

    def __post_init__(self):
        _flag("liquid", "enabled", self.enabled)
        for name in ("eps_d", "eps_max", "max_head_volume", "resolution"):
            _positive("liquid", name, getattr(self, name))
        if not self.eps_d <= self.eps_max < 1.0:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid: eps_d <= eps_max < 1")
        if not (_number(self.track_depth) and 0.0 <= self.track_depth < 1.0):
            raise ValueError(f"collapse.{EVENTS_BLOCK}.liquid.track_depth must lie in [0, 1)")


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
    there (``focus.kept``) until the first rebuild after it (``reconnect.feature_regions``)."""
    open_neck_stop: bool = True
    h_floor: float | None = None
    neck_floor: float | None = None
    neck_elements_per_scale: float | None = None
    rise_tol: float | None = None
    dt_min: float | None = None
    minimum_dt: float | None = None
    bubble_bottom_sizing: str = "kept"
    bubble_bottom_size: float | None = None

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
    """The ``collapse.reconnect_events`` block (module docstring)."""
    liquid: LiquidEventSettings = field(default_factory=LiquidEventSettings)
    gap: GapSettings = field(default_factory=GapSettings)
    post: PostCutSettings = field(default_factory=PostCutSettings)

    PARTS = {"liquid": LiquidEventSettings, "gap": GapSettings, "post": PostCutSettings}
    CLASSES = ("liquid", "gap")  # the parts whose keys are the keys of a class of events

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

    @classmethod
    def from_dict(cls, data) -> "EventSettings":
        """The block of a case, refusing keys it does not know (at every level)."""
        if not isinstance(data, dict):
            raise ValueError(f"collapse.{EVENTS_BLOCK} must be an object")
        unknown = sorted(set(data) - set(cls.PARTS))
        if unknown:
            raise ValueError(f"collapse.{EVENTS_BLOCK}: unknown key(s) " + ", ".join(unknown))
        kw = {}
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
        return {name: asdict(getattr(self, name)) for name in self.PARTS}

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
            out[name] = {**out.get(name, {}), **(sub if isinstance(sub, dict) else {})}
        return out

    def changed_keys(self, recorded: dict | None) -> list[str]:
        """The dotted keys (``liquid.eps_d``) in which the block differs from a recorded one; a
        key the record lacks counts at its default."""
        old = self._completed(recorded or {})
        new = json.loads(json.dumps(self.to_dict()))
        out = []
        for name in sorted(set(old) | set(new)):
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
        stay as made). The keys of the leg (``post``) may change at any restart."""
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
# the handler of each class of event (None: none, an unhandled event or one to be designed when it
# occurs), by class and not by kind: a liquid neck without a single head, with one too large, or
# met with liquid events off (L-unhandled) has none, whatever found it. The driver has no handler
# yet, so every event is a resumable stop
HANDLERS = {"W1": "ring", "L": "liquid-cut", "A-head": "liquid-cut", "L-unhandled": None,
            "A-groove": "groove-fill", "G": "n-chain-split", "B": None, "X": None}
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
