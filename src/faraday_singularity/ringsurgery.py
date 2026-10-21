"""The ring of a pierced bubble on the finite-element problem: the ring stage of a run.

The ring cut of a bubble its downward jet pierces leaves a ring of gas, a closed loop of the free
surface's own boundary ``interface`` with the passive constant (zero) gas pressure of the whole free
surface (``ring.py``). The ring stage lives on the problem (``FaradayProblem.ring``, its record,
dumped with every state of it) and on the mesh template (``ring_stage``, ``ring_sizing``: every
rebuild meshes the ring as a hole of the liquid at the sizing frozen at the cut,
``mesh.FaradayCylinderMesh._define_ring_geometry``). pyoomph rebuilds a loaded mesh from a sibling
of the problem's template, which in a new process holds neither, so a restart in the ring stage
puts them back (:func:`restore_ring`, called by ``driver.CollapseDriver.restore`` after the cut's
own ``surgery.restore_cut``); the chart of the loaded interface needs nothing, the problem's
arclength chart charting a loop whatever the template (``problem.RingTolerantArclength``).
"""

from __future__ import annotations

from .mesh import RingSizing


def restore_ring(problem, record: dict) -> None:
    """After loading a state of the ring stage: the stage and the ring's frozen sizing
    (``record["sizing"]``: ``h_ring``, ``grading``) on the template the loaded mesh is rebuilt from
    (``FaradayProblem.active_mesh_template``), with no ring geometry pending; the record on the
    problem (``FaradayProblem.ring``)."""
    sizing = (record or {}).get("sizing") or {}
    if "h_ring" not in sizing or "grading" not in sizing:
        raise RuntimeError("the record of the ring stage has no ring sizing")
    template = problem.active_mesh_template()
    template.ring_stage = True
    template.ring_sizing = RingSizing(h_ring=float(sizing["h_ring"]),
                                      grading=float(sizing["grading"]))
    template.pending_ring_geometry = None
    problem.ring = record
