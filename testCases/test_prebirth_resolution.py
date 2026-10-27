"""The evolving base sets resolution before the jet-birth aspect threshold."""

import math

import numpy as np

from test_bookkeeping import SurfaceProblem, _settings
from faraday_singularity.driver import CollapseDriver


def test_emerging_cavity_base_sets_resolution_before_birth(tmp_path):
    r0, rise = 0.1, 0.004
    pb = SurfaceProblem(lambda t, r: 5.0 + rise * ((r / r0) ** 2 - 1) ** 2)
    settings = _settings(flat_aspect=0.01, jet_aspect=0.1)
    driver = CollapseDriver(pb, settings, tmp_path)
    driver.state.cavity_seen = True
    try:
        measured = driver.measure()
        assert not measured["jet_born"] and not measured["flat_bump"]
        assert 0.09 < measured["r_j"] < 0.11
        assert measured["ell_measured"] <= measured["r_j"]
        assert measured["h_focus"] < pb.mesh_settings.h_axis
    finally:
        driver.close()


def test_flat_remnant_minimum_does_not_force_resolution(tmp_path):
    r0, rise = 0.1, 1e-7
    pb = SurfaceProblem(lambda t, r: 5.0 + rise * ((r / r0) ** 2 - 1) ** 2)
    driver = CollapseDriver(pb, _settings(flat_aspect=0.01), tmp_path)
    driver.state.cavity_seen = True
    try:
        measured = driver.measure()
        assert measured["flat_bump"]
        assert math.isclose(measured["h_focus"], pb.mesh_settings.h_axis)
    finally:
        driver.close()
