"""TLE lines and comparison helpers shared by several test files
(``from shared import ...``, like ``sp3file``)."""

import numpy as np

import satkit as sk

# ISS (ZARYA), epoch 2024-01-01 12:00 UTC
ISS_2024 = [
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005",
    "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560000 00001",
]

# ISS (ZARYA), epoch 2021-10-02 14:11 UTC (day 275), and its name line
ISS_NAME = "0 ISS (ZARYA)"
ISS_2021 = [
    "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357",
]

# STARLINK-3118, epoch 2024-01-30
STARLINK_3118 = [
    "1 49140U 21082L   24030.39663557  .00000076  00000-0  14180-4 0  9995",
    "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
]


def qtuple(q):
    """A quaternion's (w, x, y, z)"""
    return (q.w, q.x, q.y, q.z)


def same(a, b):
    """Exactly equal: quaternions component-wise, anything else as arrays"""
    if isinstance(a, sk.quaternion):
        return qtuple(a) == qtuple(b)
    return np.array_equal(np.asarray(a), np.asarray(b))
