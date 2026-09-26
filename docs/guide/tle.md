# SGP4, Two-Line Element Sets (TLEs), and Orbital Mean-Element Messages (OMMs)

## Satellite Catalog

The United States maintains a public catalog of Earth-orbiting objects, including active satellites, rocket bodies, and debris.

There are multiple places to access the catalog, including:

- <https://www.celestrak.org/>
- <https://www.space-track.org/>

## SGP4

**SGP4 (Simplified General Perturbations No. 4)** is an analytical orbital propagation model created in the **1960s-1970s by NORAD** to efficiently predict the motion of **Earth-orbiting satellites** from **Two-Line Element (TLE)** data.

It was developed to support U.S. space surveillance as a fast, closed-form alternative to numerical integration, modeling Earth's oblateness (J2-J4), atmospheric drag via the TLE *B\** term, and key secular and periodic perturbations. The model is documented in Spacetrack Report No. 3 ([Hoots & Roehrich 1980](references.md#hoots1980)) and, in its modern reference form with test cases, in [Vallado et al. (2006)](references.md#vallado2006); satkit's implementation is a line-by-line Rust port of the C++ code accompanying the latter and is verified against its test vectors.

satkit implements classic SGP4 only. Element sets flagged as **SGP4-XP** (ephemeris type 4 in TLE line 1, column 63, or `EPHEMERIS_TYPE: 4` in an OMM) parse normally, but `sgp4()` raises an error rather than propagating them: SGP4-XP is a different theory whose line 1 stores agom and a B term where a classic TLE stores nddot and $B^*$, and its reference implementation is distributed by the U.S. Space Force as binaries only. Such sets are rare in public catalogs.

### Gravity model, ops mode and time

`satkit.sgp4()` and Rust `sgp4::sgp4()` use the WGS72 gravity model and the AFSPC ops mode by default, the model the catalog element sets are fitted with; pass `gravconst` / `opsmode` (Rust: `sgp4_full`) for the others. Before 0.24 the Rust `sgp4()` default was WGS84 with the IMPROVED ops mode (14 m apart at epoch and ~280 m after a week for the ISS). python-sgp4's `Satrec.twoline2rv` also uses WGS72, with the IMPROVED ops mode; the two ops modes differ only for deep-space orbits below about 11.5° inclination. `TLE.fit_from_states()` fits with the same defaults and takes the same keywords, so a fitted TLE should be propagated with the settings it was fitted with.

The SGP4 initialization is cached in the TLE (or Rust `OMM`) after the first propagation, together with the elements, gravity model and ops mode it was built from; editing an element or passing another `gravconst` / `opsmode` rebuilds it, and TLE equality ignores it.

The time since epoch is the physical (SI) time elapsed between the element-set epoch and the requested time. Across a leap second this is one second more than the difference of the UTC labels that Vallado's reference code and python-sgp4 use, so satkit differs from them by 1 s of along-track motion (~7.6 km at LEO) per leap second between epoch and time. This is deliberate: the satellite really flies 86,401 s over a day with a leap second, and SGP4's mean motion is per SI day.

A time at which propagation fails (e.g. the orbit has decayed) gives a NaN row, with its code in the error array when `errflag=True`. An element set that cannot be initialized at all raises `RuntimeError`, unless `errflag=True`: then its rows are NaN and its initialization error code is reported at every time, so one bad element set does not fail a list.

Today, SGP4 is the standard propagator for TLEs published by organizations like NORAD and CelesTrak, and is widely used for satellite tracking, visualization, conjunction screening, and mission planning — though its accuracy is fundamentally limited by TLE quality and simplifying assumptions.

## Ephemeris Representation

### TLE

A **Two-Line Element Set (TLE)** is a compact, legacy format (originally constrained by punch-card era line lengths) for describing satellite orbits. Despite its age, it remains widely used because it is easy to publish and often provides sufficient accuracy for many applications.

In addition to the familiar (mean) Keplerian elements, TLEs include parameters related to perturbations (e.g., atmospheric drag via $B^*$, and derivatives of mean motion).

TLEs are designed to be used with **SGP4**, an analytic model that produces orbital state vectors (position and velocity) from the augmented mean elements in the TLE. The perturbations modeled include Earth oblateness (which produces precession) and drag.

TLEs are often preceded by an additional "line 0" containing the satellite name. The field-by-field format is documented in [Vallado et al. (2006)](references.md#vallado2006), Appendix A, and in CelesTrak's [TLE format description](https://celestrak.org/NORAD/documentation/tle-fmt.php).

### Orbital Mean-Element Messages

**Orbital Mean-Element Messages (OMMs)** are a more modern way of representing mean-element ephemerides. They are described by the CCSDS *Orbit Data Messages* standard ([CCSDS 502.0-B-3](references.md#ccsds502)), although real-world sources may not adhere to the standard perfectly.

OMMs are commonly published as:

- JSON
- XML
- KVN (key-value notation)

satkit reads the JSON and XML forms (KVN is not supported). In Python an OMM is a plain dictionary keyed by the CCSDS field names: `satkit.omm_from_url()`, `satkit.omm_from_file()` and `satkit.omm_from_text()` return a list of them, with every field the source provided (Space-Track's catalog extras such as `OBJECT_TYPE` and `RCS_SIZE` included) and numbers converted from Space-Track's quoted strings. `satkit.sgp4()` accepts these dictionaries directly, as well as the raw output of `json.load` on a CelesTrak or Space-Track response. `satkit.TLE.from_omm()` and `satkit.TLE.to_omm()` convert between the two representations.

A message whose `MEAN_ELEMENT_THEORY` is not SGP4, whose `TIME_SYSTEM` is not UTC, whose `REF_FRAME` is not TEME, whose `CENTER_NAME` is not EARTH, or whose `EPHEMERIS_TYPE` is 4 (SGP4-XP, which Space-Track distributes alongside classic SGP4 sets) is rejected rather than propagated with the wrong theory. `EPOCH` may be an RFC 3339 date-time or the CCSDS day-of-year form `YYYY-DDDThh:mm:ss`.

In Rust the same message is the `satkit::omm::OMM` struct, which implements `SGP4Source`, serializes back to JSON with `serde`, and converts with `OMM::from_tle` / `OMM::to_tle`.

## Loading TLEs

`TLE.from_lines()`, `TLE.from_file()` and `TLE.from_url()` accept 2-line and 3-line (named) element sets, any number of them, and always return a `list[TLE]`, even for a single element set. Input with no element sets raises `ValueError`.

A record that fails to parse raises `RuntimeError` naming the input line the record starts on and its satellite. So does a line 1 whose line 2 does not follow it, and a line 1 and line 2 with different satellite numbers. A line that is almost a data line (68 characters, or a leading space) is read as a satellite name, and the error then says so; a UTF-8 byte-order mark at the start of the input is ignored. A data line longer than the standard 69 characters is accepted (the extra columns are ignored), but if a field of such a record then fails to parse, the message says so: a field written one column too wide shifts every later column of the line. Checksums (column 69) are not verified unless requested with `check_checksum=True` (`from_lines` and `from_file`).

```python
import satkit as sk

lines = [
    "0 STARLINK-3118",
    "1 49140U 21082L   24030.39663557  .00000076  00000-0  14180-4 0  9995",
    "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
]
tle = sk.TLE.from_lines(lines, check_checksum=True)[0]

# An 8-digit eccentricity pushes the mean anomaly one column right
bad = lines[:2] + [lines[2].replace(" 0002663 ", " 00026630 ")]
try:
    sk.TLE.from_lines(bad)
except RuntimeError as e:
    print(e)
# TLE record starting at line 1 (sat 49140 "STARLINK-3118"): Could not parse mean anomaly:
# invalid float literal (line 3 is 70 characters; a TLE line is 69, so its columns may be shifted)
```

In Rust, `TLE::from_lines` stops at the first bad record in the same way. To keep the good records of a large catalog file and skip the malformed ones, iterate `TLE::records` instead, optionally with `.check_checksums(true)`:

```rust
let tles: Vec<satkit::TLE> = satkit::TLE::records(&lines).filter_map(Result::ok).collect();
```

## Loading from URLs

Both TLEs and OMMs can be loaded directly from a URL:

<!-- skip-test: needs the network (CelesTrak) -->
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
starlink30477 = sk.TLE.from_lines(tle_lines)[0]

# The state is output in the "TEME" frame
pTEME, _vTEME = sk.sgp4(starlink30477, sk.time(2024, 4, 9, 12, 0, 0))

# Rotate to Earth-fixed (ITRF) and get geodetic coordinates. Keyword
# arguments make the direction explicit at first sight; the same call with
# positional args is `rotation(sk.frame.TEME, sk.frame.ITRF, thetime)`.
thetime = sk.time(2024, 4, 9, 12, 0, 0)
pITRF = sk.frametransform.rotation(
    from_frame=sk.frame.TEME, to_frame=sk.frame.ITRF, tm=thetime,
) * pTEME
coord = sk.itrfcoord(pITRF)
print(coord)
# ITRFCoord(lat:  29.3890 deg, lon: 170.8051 deg, hae: 560.11 km)
```

### SGP4 from a URL (TLE)

<!-- skip-test: needs the network (CelesTrak) -->
```python
import satkit as sk

# Load all space station TLEs directly from CelesTrak
tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
iss = tles[0]  # ISS is first

pos, vel = sk.sgp4(iss, sk.time(2024, 6, 1))
```

### SGP4 from a URL (OMM)

<!-- skip-test: needs the network (CelesTrak) -->
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

<!-- skip-test: needs the network (CelesTrak) -->
```python
omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=xml")
```

### OMM from a local file, and conversion to a TLE

<!-- test-setup
import json
import satkit as sk
iss = sk.TLE.from_lines([
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357",
])[0]
with open("gp.json", "w") as f:
    json.dump([iss.to_omm()], f)
-->
```python
import satkit as sk

# JSON or XML, detected from the content; one dict per message
omms = sk.omm_from_file("gp.json")

# Everything the source provided is in the dict
print(omms[0]["OBJECT_NAME"], omms[0].get("OBJECT_TYPE"), omms[0].get("RCS_SIZE"))

# The same element set as a TLE object, and back again
tle = sk.TLE.from_omm(omms[0])
print("\n".join(tle.to_2line()))
assert sk.TLE.from_omm(tle.to_omm()).to_2line() == tle.to_2line()
```
