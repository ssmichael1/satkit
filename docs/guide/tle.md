# SGP4, Two-Line Element Sets (TLEs), and Orbital Mean-Element Messages (OMMs)

## Satellite Catalog

The United States maintains a public catalog of Earth-orbiting objects, including active satellites, rocket bodies, and debris.

There are multiple places to access the catalog, including:

- <https://www.celestrak.org/>
- <https://www.space-track.org/>

## SGP4

**SGP4 (Simplified General Perturbations No. 4)** is an analytical orbital propagation model created in the **1960s-1970s by NORAD** to efficiently predict the motion of **Earth-orbiting satellites** from **Two-Line Element (TLE)** data.

It was developed to support U.S. space surveillance as a fast, closed-form alternative to numerical integration, modeling Earth's oblateness (J2-J4), atmospheric drag via the TLE *B\** term, and key secular and periodic perturbations.

Today, SGP4 is the standard propagator for TLEs published by organizations like NORAD and CelesTrak, and is widely used for satellite tracking, visualization, conjunction screening, and mission planning — though its accuracy is fundamentally limited by TLE quality and simplifying assumptions.

## Ephemeris Representation

### TLE

A **Two-Line Element Set (TLE)** is a compact, legacy format (originally constrained by punch-card era line lengths) for describing satellite orbits. Despite its age, it remains widely used because it is easy to publish and often provides sufficient accuracy for many applications.

In addition to the familiar (mean) Keplerian elements, TLEs include parameters related to perturbations (e.g., atmospheric drag via $B^*$, and derivatives of mean motion).

TLEs are designed to be used with **SGP4**, an analytic model that produces orbital state vectors (position and velocity) from the augmented mean elements in the TLE. The perturbations modeled include Earth oblateness (which produces precession) and drag.

TLEs are often preceded by an additional "line 0" containing the satellite name. For an overview, see <https://en.wikipedia.org/wiki/Two-line_element_set>.

### Orbital Mean-Element Messages

**Orbital Mean-Element Messages (OMMs)** are a more modern way of representing mean-element ephemerides. They are described by a CCSDS standard (<https://ccsds.org/Pubs/502x0b3e1.pdf>), although real-world sources may not adhere to the standard perfectly.

OMMs are commonly published as:

- JSON
- XML
- KVN (key-value notation)

OMMs are commonly published as JSON or XML. The `satkit.omm_from_url()` function fetches OMMs from a URL and auto-detects the format, returning a list of Python dictionaries that can be passed directly to `satkit.sgp4()`.

You can also provide OMM dictionaries manually (e.g. parsed from a local JSON file).

## Loading from URLs

Both TLEs and OMMs can be loaded directly from a URL:

```python
import satkit as sk

# Load TLEs from a URL
tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")

# Load OMMs from a URL (auto-detects JSON vs XML)
omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")

# OMMs work directly with sgp4()
pos, vel = sk.sgp4(omms[0], sk.time(2024, 6, 1))
```

## Example Usage

### SGP4 from TLE lines

```python
import satkit as sk

# The two-line element set
# Let's pick a random Starlink satellite
tle_lines = [
    '0 STARLINK-30477',
    '1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997',
    '2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746'
]

# Create a TLE object
starlink30477 = sk.TLE.from_lines(tle_lines)

# The state is output in the "TEME" frame
pTEME, _vTEME = sk.sgp4(starlink30477, sk.time(2024, 4, 9, 12, 0, 0))

# Rotate to Earth-fixed (ITRF) and get geodetic coordinates
thetime = sk.time(2024, 4, 9, 12, 0, 0)
pITRF = sk.frametransform.rotation(sk.frame.TEME, sk.frame.ITRF, thetime) * pTEME
coord = sk.itrfcoord(pITRF)
print(coord)
# ITRFCoord(lat:  29.3890 deg, lon: 170.8051 deg, hae: 560.11 km)
```

### SGP4 from a URL (TLE)

```python
import satkit as sk

# Load all space station TLEs directly from CelesTrak
tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
iss = tles[0]  # ISS is first

pos, vel = sk.sgp4(iss, sk.time(2024, 6, 1))
```

### SGP4 from a URL (OMM)

```python
import satkit as sk

# Load ISS ephemeris as OMM from CelesTrak
omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=json")

epoch = sk.time(omms[0]['EPOCH'])
time_array = [epoch + sk.duration(minutes=i*10) for i in range(6)]

# SGP4 propagation
pTEME, _vTEME = sk.sgp4(omms[0], time_array)

# Rotate to Earth-fixed and get geodetic coordinates
pITRF = [
    sk.frametransform.rotation(sk.frame.TEME, sk.frame.ITRF, t) * p
    for t, p in zip(time_array, pTEME)
]
coord = [sk.itrfcoord(x) for x in pITRF]
```

XML format works the same way -- just change the URL:

```python
omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=xml")
```
