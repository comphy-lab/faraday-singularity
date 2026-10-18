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
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields

EVENTS_BLOCK = "reconnect_events"  # the key of the block in a case's collapse block
BUBBLE_BOTTOM_SIZINGS = ("kept", "frozen")
# the kinds of the events a run makes, by class: the keys of a class are fixed after its first event
EVENT_CLASSES = {"liquid": ("liquid-neck", "a-head"), "gap": ("groove-fill",)}
# a dump of the state before an event, from which a restart may change the keys of its class
PRECUT_EVENT = "precut-event"


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
    ``r_n <= eps_max s``. With ``enabled`` false a liquid neck is met as before the block, at the
    stop of the cut's settings (``secondary-pinch-approach`` below ``delta``)."""
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
    neck (the cut's focus size without it) and ``neck_elements_per_scale`` its elements per neck
    radius (``reconnect.elements_per_scale`` without it); ``rise_tol`` replaces the collapse's
    ``rise_tol`` (the rise of a bump above its base that makes a jet); ``dt_min`` the step-size floor
    of the driver; ``minimum_dt`` is oomph-lib's smallest step after the cut (``Problem::
    Minimum_dt``, 1e-3 times the driver's floor without it), below which it gives up on a step,
    and below which it also refuses an accepted step whose next candidate, four times the step,
    would fall; before the cut it stays at oomph-lib's default. ``open_neck_stop`` false drops the
    open chain's neck stop after the cut (``collapse.neck_stop``), which liquid events need.
    ``bubble_bottom_sizing`` ``frozen`` meshes the bubble's bottom (far from the cut) never coarser
    than ``bubble_bottom_size`` instead of never coarser than the landing state's focus there
    (``kept``), so that the bubbles of a study in ``delta`` are meshed alike."""
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
        if self.minimum_dt is not None and self.dt_min is not None \
                and not self.minimum_dt < self.dt_min:
            raise ValueError(f"collapse.{EVENTS_BLOCK}.post.minimum_dt must lie below dt_min: "
                             "oomph-lib would give up on a step before the driver's floor stops")


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
