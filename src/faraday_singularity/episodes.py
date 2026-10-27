"""Crest and trough episodes of the free surface on the axis, and the wave height before a collapse.

The axis elevation ``eta = z_axis - H`` is measured from the mean level ``H``, which is the liquid
depth ``d``: the volume is conserved and the container is a cylinder. A crest episode starts when
``eta`` rises above ``+hys``; a trough episode starts when ``eta`` falls below ``-hys``. Between the
two thresholds the episode in progress continues, so a wobble of the axis within ``±hys`` neither
starts nor ends one, and a trough episode can only follow a crest episode (or the start of the
record). Episodes are numbered in the order in which they start, crests and troughs alike, from 0;
``-1`` means that none has started yet.

The wave height of a trough episode, ``h_crest``, is the largest ``eta`` over the crest episode
immediately before it, reached at ``t_crest``. It is NaN when no crest preceded the trough, and when
that crest began before the record did: a run restarted without the episode history cannot know the
earlier part of the crest in progress. The first episode of such a record is therefore ``partial``,
whether it starts on the first sample or, when the record starts inside the band, at the first
threshold crossing.

While a jet stands on the axis, ``eta`` is the elevation of its tip. A jet launched from a trough
rises into the crest episode after it, so that crest, and the ``h_crest`` of the next trough, can
be the jet's rather than the wave's. Each sample therefore says whether a jet stands
(``jet_born``), and a trough carries ``crest_jet``: whether a jet stood at any sample of its crest
episode. ``h_crest`` is a wave height only when ``crest_jet`` is false. ``crest_jet`` is None when
``h_crest`` is unknown, and when the crest was in progress at a restart from a tracker state that
predates the flag and no jet was seen after it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

ETA_HYS = 0.02  # default threshold, in units of 1/k


@dataclass
class EpisodeTracker:
    hys: float = ETA_HYS
    history_known: bool = True  # False when the record starts in the middle of a run
    index: int = -1
    kind: int = 0  # +1 crest, -1 trough, 0 before the first episode
    start: float = math.nan
    partial: bool = False  # the episode in progress may have begun before the record
    samples: int = 0
    crest_max: float = math.nan  # running maximum of eta over the crest in progress
    crest_time: float = math.nan
    crest_has_jet: bool | None = None  # a jet stood at a sample of the crest in progress
    h_crest: float = math.nan  # wave height of the trough in progress
    t_crest: float = math.nan
    crest_jet: bool | None = None  # a jet stood in the crest of h_crest: not a wave height

    def update(self, t: float, eta: float, jet: bool = False) -> dict | None:
        """Add one sample, ``jet`` if a jet stands on the axis; return ``{"event": "crest" |
        "trough", ...}`` when an episode starts."""
        self.samples += 1
        if not math.isfinite(eta):
            return None
        if eta > self.hys and self.kind != 1:
            self._begin(1, t)
            self.crest_max, self.crest_time = eta, t
            self.crest_has_jet = bool(jet)
            return {"event": "crest", "episode": self.index, "partial": self.partial}
        if eta < -self.hys and self.kind != -1:
            known = self.kind == 1 and not self.partial
            self.h_crest = self.crest_max if known else math.nan
            self.t_crest = self.crest_time if known else math.nan
            self.crest_jet = self.crest_has_jet if known else None
            self._begin(-1, t)
            return {"event": "trough", "episode": self.index, "h_crest": self.h_crest,
                    "t_crest": self.t_crest, "crest_jet": self.crest_jet,
                    "partial": self.partial}
        if self.kind == 1:
            if eta > self.crest_max:
                self.crest_max, self.crest_time = eta, t
            if jet:
                self.crest_has_jet = True
        return None

    def _begin(self, kind: int, t: float):
        self.index += 1
        self.kind = kind
        self.start = t
        # without the history, the first episode of the record may continue one already in
        # progress, also when the record starts inside the band and crosses a threshold later
        self.partial = self.index == 0 and not self.history_known

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EpisodeTracker:
        return cls(**data)


def segment(t, eta, hys: float = ETA_HYS, history_known: bool = True, jet=None):
    """Episode index and kind of every sample of ``eta(t)``, and the episode-start events.

    The same :class:`EpisodeTracker` as during a run, applied to a whole record, with ``jet`` the
    samples at which a jet stands (none if not given); each event also carries its start time
    ``time``.
    """
    tracker = EpisodeTracker(hys=hys, history_known=history_known)
    index = np.empty(len(t), dtype=int)
    kind = np.empty(len(t), dtype=int)
    jets = np.zeros(len(t), dtype=bool) if jet is None else np.asarray(jet, dtype=bool)
    events = []
    for i, (ti, ei, ji) in enumerate(zip(t, eta, jets, strict=True)):
        ev = tracker.update(float(ti), float(ei), bool(ji))
        if ev is not None:
            events.append({"time": float(ti), **ev})
        index[i], kind[i] = tracker.index, tracker.kind
    return index, kind, events
