"""A quasi-one-dimensional compressible gas in the collapsing neck of the cavity.

``q1d`` solves the unsteady quasi-1D Euler equations of the gas in a duct whose walls move
(Gekle & Gordillo 2011, Int. J. Numer. Meth. Fluids, doi:10.1002/fld.2426, Eqs. 13-15). The other
modules couple it to the liquid: ``geometry`` reads the duct from the free surface,
``coupler`` advances the gas over each step of the liquid and returns its pressure as a normal
traction on the free surface (the ``collapse.gas`` block of a case).
"""
