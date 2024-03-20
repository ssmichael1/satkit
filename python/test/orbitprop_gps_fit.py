"""
This function reads an SP3 file containing precise GPS locaions in the ITRF frame

It then minimizes RMS difference between the precise locations and locations predicted
by high-precision propagator by tuning the initial velocity and the 
radiation pressure parameter Cr A /M

This can then be used to check accuracy of the high-precision propagator in the 
python tests

Currently we can get it accurate to < 10 meters over full day of propagation

"""

import satkit as sk
import numpy as np
import math as m
import numpy.typing as npt
from scipy.optimize import minimize
from sp3file import read_sp3file
import os

# File contains test calculation vectors provided by NASA
basedir = os.getenv(
    "SATKIT_TESTVEC_ROOT", default="../.." + os.path.sep + "satkit-testvecs"
)
fname = (
    basedir
    + os.path.sep
    + "orbitprop"
    + os.path.sep
    + "ESA0OPSFIN_20233640000_01D_05M_ORB.SP3"
)

# Get positions and times from SP3 file
[pitrf, timearr] = read_sp3file(fname)

# Rotate positions to the GCRF frame
pgcrf = np.stack(
    np.fromiter(
        (q * p for q, p in zip(sk.frametransform.qitrf2gcrf(timearr), pitrf)), list
    ),
    axis=0,
)

# Crude estimation of initial velocity
vgcrf = (pgcrf[1, :] - pgcrf[0, :]) / (timearr[1] - timearr[0]).seconds()

# Initial state for non-linear least squares is initial velocity
# and susceptibility to radiation pressuer : Cr A / m
v0 = np.array([vgcrf[0], vgcrf[1], vgcrf[2], 1 * 10**2 / 5000])


def minfunc(v):
    tstart = timearr[0]
    tend = timearr[-1]
    settings = sk.satprop.propsettings()
    settings.use_jplephem = False
    satprops = sk.satprop.satproperties_static()
    satprops.craoverm = v[3]

    res = sk.satprop.propagate(
        pgcrf[0, :],
        v[0:3],
        tstart,
        stoptime=tend,
        dt=sk.duration.from_minutes(5),
        propsettings=settings,
        satproperties=satprops,
    )
    return np.sum(np.sum((res["pos"] - pgcrf) ** 2, axis=1), axis=0)


# Minimize difference between true satellite state and propagated state
# by tuning initial velocity and CrAoverM
r = minimize(minfunc, v0, method="Nelder-Mead")
print(r)
