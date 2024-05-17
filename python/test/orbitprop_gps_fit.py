# %%
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

sp3names = ['mid22630.sp3', 'mid22631.sp3', 'mid22632.sp3',
          'mid22633.sp3', 'mid22634.sp3', 'mid22635.sp3',
          'mid22636.sp3']

sp3names = ['mid22630.sp3', 'mid22631.sp3', 'mid22632.sp3']
pitrf = np.zeros((0,3), np.float64)
timearr = np.array([])

for sp3name in sp3names:
    fname = (
        basedir
        + os.path.sep
        + "orbitprop"
        + os.path.sep
        + sp3name
    )
    [pitrf1, timearray1] = read_sp3file(fname)
    pitrf = np.append(pitrf, pitrf1, axis=0)
    timearr = np.append(timearr, timearray1)

print(pitrf.shape)
print(timearr.shape)
print(timearr[0], timearr[-1])

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
v0 = np.array([vgcrf[0], vgcrf[1], vgcrf[2], 0.01])


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

# %%
import plotly.graph_objects as go

settings = sk.satprop.propsettings()
settings.use_jplephem = False
satprops = sk.satprop.satproperties_static()
satprops.craoverm = r.x[3]
res = sk.satprop.propagate(
    pgcrf[0,:], r.x[0:3], timearr[0],
    stoptime=timearr[-1], dt=sk.duration(minutes=5),
    propsettings=settings, satproperties=satprops
)
perr = res['pos'] - pgcrf
fig = go.Figure()
fig.add_trace(go.Scatter(x=[t.datetime() for t in timearr], y=perr[:,0], mode='lines', name='X'))
fig.add_trace(go.Scatter(x=[t.datetime() for t in timearr], y=perr[:,1], mode='lines', name='Y'))
fig.add_trace(go.Scatter(x=[t.datetime() for t in timearr], y=perr[:,2], mode='lines', name='Z'))

# %%
