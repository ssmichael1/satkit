# SGP4, Two-Line Element Sets (TLEs), and Orbital Mean-Element Messages (OMMs)

## Satellite Catalog

The United States maintains a public catalog of Earth-orbiting objects, including active satellites, rocket bodies, and debris.

There are multiple places to access the catalog, including:
* <https://www.celestrak.org/>
* <https://www.space-track.org/>

## SGP4

**SGP4 (Simplified General Perturbations No. 4)** is an analytical orbital propagation model created in the **1960s–1970s by NORAD** to efficiently predict the motion of **Earth-orbiting satellites** from **Two-Line Element (TLE)** data.

It was developed to support U.S. space surveillance as a fast, closed-form alternative to numerical integration, modeling Earth’s oblateness (J2–J4), atmospheric drag via the TLE *B\** term, and key secular and periodic perturbations.

Today, SGP4 is the standard propagator for TLEs published by organizations like NORAD and CelesTrak, and is widely used for satellite tracking, visualization, conjunction screening, and mission planning—though its accuracy is fundamentally limited by TLE quality and simplifying assumptions.

## Ephemeris Representation

### TLE
A **Two-Line Element Set (TLE)** is a compact, legacy format (originally constrained by punch-card era line lengths) for describing satellite orbits. Despite its age, it remains widely used because it is easy to publish and often provides sufficient accuracy for many applications.

In addition to the familiar (mean) Keplerian elements, TLEs include parameters related to perturbations (e.g., atmospheric drag via $B^*$, and derivatives of mean motion).

TLEs are designed to be used with **SGP4**, an analytic model that produces orbital state vectors (position and velocity) from the augmented mean elements in the TLE. The perturbations modeled include Earth oblateness (which produces precession) and drag.

TLEs are often preceded by an additional “line 0” containing the satellite name. For an overview, see <https://en.wikipedia.org/wiki/Two-line_element_set>.

### Orbital Mean-Element Messages
**Orbital Mean-Element Messages (OMMs)** are a more modern way of representing mean-element ephemerides. They are described by a CCSDS standard (<https://ccsds.org/Pubs/502x0b3e1.pdf>), although real-world sources may not adhere to the standard perfectly.

OMMs are commonly published as:
* JSON
* XML
* KVN (key–value notation)

The satkit Python interface does not load OMM files directly. Instead, it expects you to provide the decoded OMM as a Python `dict` (for example, parsed from JSON or XML). The interface supports OMM dictionary layouts produced by CelesTrak and Space-Track.




## Example Usage

### SGP4 state computation from TLE

```python
import satkit as sk

# The two-line element set
# Let's pick a random Starlink satellite
# The lines below were downloaded from https://www.celestrak.org
tle_lines = [
    '0 STARLINK-30477',
    '1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997',
    '2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746'
]

# Create a TLE object
starlink30477 = sk.TLE.from_lines(tle_lines)

# We want the orbital state at April 9 2024, 12:00pm UTC
thetime = sk.time(2024, 4, 9, 12, 0, 0)

# The state is output in the "TEME" frame, which is an approximate inertial
# frame that does not include precession or nutation
# pTEME is geocentric position in meters
# vTEME is geocentric velocity in meters / second
# for now we will ignore the velocity
pTEME, _vTEME = sk.sgp4(starlink30477, thetime)

# Suppose we want currrent latitude, longitude, and altitude of satellite:
# we need to rotate into an Earth-fixed frame, the ITRF
# We use a "quaternion" to represent the rotation.  Quaternion rotations
# in the satkit toolbox can be represented as multiplications of a 3-vector
pITRF = sk.frametransform.qteme2itrf(thetime) * pTEME

# Now lets make a "ITRFCoord" object to extract geodetic coordinates
coord = sk.itrfcoord(pITRF)

# Get the latitude, longitude, and
# altitude (height above ellipsoid, or hae) of the satellite
print(coord)

# this should produce:
# ITRFCoord(lat:  29.3890 deg, lon: 170.8051 deg, hae: 560.11 km)

```

### SGP4 State Computation from OMM representation of International Space Station (ISS)

```python

import satkit as sk
import requests
import json

# Query the current ephemeris for the International Space Station (ISS)
# from celestrak.org
url = 'https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=json'
with requests.get(url) as response:
    omm = response.json()

# Get a representative time from the output
epoch = sk.time(omm[0]['EPOCH'])
# create a list of times .. once every 10 minutes
time_array = [epoch + sk.duration(minutes=i*10) for i in range(6)]

# TEME (inertial) output from SGP4
pTEME, _vTEME = sk.sgp4(omm[0], time_array)

# Rotate to Earth-fixed
pITRF = [sk.frametransform.qteme2itrf(t) * p for t, p in zip(time_array, pTEME)]

# Geodetic coordinates of space station at given times
coord = [sk.itrfcoord(x) for x in pITRF]

```
