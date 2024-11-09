
# Two-Line Element Sets (TLEs)

A **Two-Line Element Set**, or **TLE**, is an ancient method, originally designed for computer punchcards, to describe satellite orbits.  Despite it's age, this format continues to be commonly used, likely because it is a compact way of describing satellite orbits with sufficient accuracy for *most* applications.  TLEs contain the "classic" Keplerian elements, as well as elements that describe purturbations such as drag and derivatives of the mean motion.

TLEs are designed to be used with the **Simplified General Perturbation Model** version 4, or **SGP-4**, an analytic model that produces orbital state vectors -- position and velocity -- from the augmented Keplerian elements in the TLE.  The purturbations to the Keplerian elements include factoring in the Earth oblateness (which procudes procession) and drag.

The TLEs will often be prepended by an additional line that provides the name of the satellite.  For an overview, see: <https://en.wikipedia.org/wiki/Two-line_element_set>

## Satellite Catalog

The United States maintains a public catalog of all Earth-orbiting satellites, to include active satellites, rocket bodies, debris, etc... 

The publicly-available catalog describes satellites as two-line element sets.  There are multiple places to access the catalog, including:
* <https://www.celestrak.org/>
* <https://www.space-track.org/>

## Example Usage

```python
import satkit as sk

# The two-line element set
# Lets pick a random StarLink satellite
# The lines below were downloaded from https://www.celestrack.org
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