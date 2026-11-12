# SGP4, Two-Line Element Sets (TLEs), and Orbital Mean-Element Messages (OMMs)

## SGP4

**SGP4 (Simplified General Perturbations No. 4)** is the analytical propagation model NORAD created in the 1960s–1970s to predict the motion of Earth-orbiting satellites from **Two-Line Element (TLE)** data: a fast, closed-form alternative to numerical integration that models Earth's oblateness (J2–J4), atmospheric drag through the TLE $B^*$ term, and the main secular and periodic perturbations. It remains the standard propagator for the public catalog of Earth-orbiting objects that the United States maintains, available from <https://www.space-track.org/> and <https://www.celestrak.org/>; its accuracy is limited by the quality of the element sets and by the model's simplifying assumptions.

The model is documented in Spacetrack Report No. 3 ([Hoots & Roehrich 1980](references.md#hoots1980)) and, in its modern reference form with test cases, in [Vallado et al. (2006)](references.md#vallado2006); satkit's implementation is a line-by-line Rust port of the C++ code accompanying the latter and is verified against its test vectors.

satkit implements classic SGP4 only. Element sets flagged as **SGP4-XP** (ephemeris type 4 in TLE line 1, column 63, or `EPHEMERIS_TYPE: 4` in an OMM) parse normally, but `sgp4()` raises an error rather than propagating them: SGP4-XP is a different theory whose line 1 stores agom and a B term where a classic TLE stores nddot and $B^*$, and its reference implementation is distributed by the U.S. Space Force as binaries only. Such sets are rare in public catalogs.

### Gravity model, ops mode and time

`satkit.sgp4()` and Rust `sgp4::sgp4()` use the WGS72 gravity model and the AFSPC ops mode by default, the model the catalog element sets are fitted with; pass `gravconst` / `opsmode` (Rust: `sgp4_full`) for the others. (The Rust default changed in 0.24: see [Migrating to 0.24](../migration.md#sgp4-and-tles).) python-sgp4's `Satrec.twoline2rv` also uses WGS72, with the IMPROVED ops mode; the two ops modes differ only for deep-space orbits below about 11.5° inclination. `TLE.fit_from_states()` fits with the same defaults and takes the same keywords, so a fitted TLE should be propagated with the settings it was fitted with.

The SGP4 initialization is cached in the TLE (or Rust `OMM`) after the first propagation, together with the elements, gravity model and ops mode it was built from; editing an element or passing another `gravconst` / `opsmode` rebuilds it, and TLE equality ignores it.

The time since epoch is the physical (SI) time elapsed between the element-set epoch and the requested time. Across a leap second this is one second more than the difference of the UTC labels that Vallado's reference code and python-sgp4 use, so satkit differs from them by 1 s of along-track motion (~7.6 km at LEO) per leap second between epoch and time. This is deliberate: the satellite really flies 86,401 s over a day with a leap second, and SGP4's mean motion is per SI day.

A time at which propagation fails (e.g. the orbit has decayed) gives a NaN row, with its code in the error array when `errflag=True`. An element set that cannot be initialized at all raises `RuntimeError`, unless `errflag=True`: then its rows are NaN and its initialization error code is reported at every time, so one bad element set does not fail a list.

## Ephemeris Representation

### TLE

A **Two-Line Element Set (TLE)** is a compact, legacy format (originally constrained by punch-card era line lengths) for describing satellite orbits. Despite its age, it remains widely used because it is easy to publish and often provides sufficient accuracy for many applications.

In addition to the (mean) Keplerian elements, a TLE carries the drag term $B^*$ and the derivatives of the mean motion. The elements are SGP4 *mean* elements: they give a position and velocity only through SGP4, not through the two-body formulas.

TLEs are often preceded by an additional "line 0" containing the satellite name. The field-by-field format is documented in [Vallado et al. (2006)](references.md#vallado2006), Appendix A, and in CelesTrak's [TLE format description](https://celestrak.org/NORAD/documentation/tle-fmt.php).

### Orbital Mean-Element Messages

**Orbital Mean-Element Messages (OMMs)** are a more modern way of representing mean-element ephemerides. They are described by the CCSDS *Orbit Data Messages* standard ([CCSDS 502.0-B-3](references.md#ccsds502)), although real-world sources may not adhere to the standard perfectly.

OMMs are commonly published as:

- JSON
- XML
- KVN (key-value notation)
- CSV: a header row of the OMM keywords and one row per object. CCSDS defines only KVN and XML, but CSV is the most compact form, and it is what a CelesTrak GP query returns when no `FORMAT` is given

satkit reads the JSON, XML and CSV forms (KVN is not supported). In CSV an empty cell is an absent field, columns that are not OMM fields (`RMS`, `DATA_SOURCE`, Space-Track's extras) are kept as strings, and a row that fails to parse raises `ValueError` naming its line. In Python an OMM is a plain dictionary keyed by the CCSDS field names: `satkit.omm_from_url()`, `satkit.omm_from_file()` and `satkit.omm_from_text()` return a list of them, with every field the source provided (Space-Track's catalog extras such as `OBJECT_TYPE` and `RCS_SIZE` included) and numbers converted from Space-Track's quoted strings. `satkit.sgp4()` accepts these dictionaries directly, as well as the raw output of `json.load` on a CelesTrak or Space-Track response. `satkit.TLE.from_omm()` and `satkit.TLE.to_omm()` convert between the two representations.

A message whose `MEAN_ELEMENT_THEORY` is not SGP4, whose `TIME_SYSTEM` is not UTC, whose `REF_FRAME` is not TEME, whose `CENTER_NAME` is not EARTH, or whose `EPHEMERIS_TYPE` is 4 (SGP4-XP, which Space-Track distributes alongside classic SGP4 sets) is rejected rather than propagated with the wrong theory. `EPOCH` may be an RFC 3339 date-time or the CCSDS day-of-year form `YYYY-DDDThh:mm:ss`.

In Rust the same message is the `satkit::omm::OMM` struct, which implements `SGP4Source`, serializes back to JSON with `serde`, and converts with `OMM::from_tle` / `OMM::to_tle`.

## Catalog numbers above 99,999

The satellite catalog passed 99,999 in July 2026, and every object catalogued since has a six-digit number. A TLE has five columns for it, so:

- **Alpha-5 TLEs** (served by Space-Track) write the first two digits as a letter: `A0000` is 100000 and `Z9999` is 339999, with `I` and `O` unused. satkit reads these, `TLE.satnum` is the plain integer, and `TLE.to_2line()` writes them. A number above 339999 cannot be written as a TLE, and `to_2line()` raises rather than emit a malformed line.
- **CelesTrak serves no TLE at all for these objects.** A TLE request for one answers HTTP 404 with the text `No GP data found`, so `TLE.from_url()` raises. Ask for an OMM instead (`FORMAT=json`, `xml` or `csv`) and load it with `satkit.omm_from_url()`.
- **OMMs carry the catalog number as an integer** of any size, in every format. For new code, and for anything launched since mid-2026, OMM is the format to use.

## Loading TLEs

`TLE.from_lines()`, `TLE.from_file()` and `TLE.from_url()` accept 2-line and 3-line (named) element sets, any number of them, and always return a `list[TLE]`, even for a single element set. Input with no element sets raises `ValueError`.

A record that fails to parse raises `ValueError` naming the input line the record starts on and its satellite. So does a line 1 whose line 2 does not follow it, and a line 1 and line 2 with different satellite numbers. A line that is almost a data line (68 characters, or a leading space) is read as a satellite name, and the error then says so; a UTF-8 byte-order mark at the start of the input is ignored. A data line longer than the standard 69 characters is accepted (the extra columns are ignored), but if a field of such a record then fails to parse, the message says so: a field written one column too wide shifts every later column of the line. Checksums (column 69) are not verified unless requested with `check_checksum=True` (`from_lines` and `from_file`).

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
except ValueError as e:
    print(e)
# TLE record starting at line 1 (sat 49140 "STARLINK-3118"): Could not parse mean anomaly:
# invalid float literal (line 3 is 70 characters; a TLE line is 69, so its columns may be shifted)
```

To keep the good records of a large catalog file and leave out the malformed ones, pass `on_error="warn"` or `on_error="skip"` to any of the three loaders. `"warn"` emits one `satkit.TLEWarning` per record left out, with the message `ValueError` would have carried; `"skip"` is silent. Only the default `on_error="raise"` raises for what the input contains: with the other two the result may be an empty list, when every record was bad or the input held no records at all (`"warn"` then emits one warning saying so).

```python
import warnings

catalog = lines + bad
with warnings.catch_warnings(record=True) as skipped:
    warnings.simplefilter("always", sk.TLEWarning)
    tles = sk.TLE.from_lines(catalog, on_error="warn")
print(len(tles), "loaded;", len(skipped), "skipped")
for w in skipped:
    print(w.message)
# 1 loaded; 1 skipped
# TLE record starting at line 4 (sat 49140 "STARLINK-3118"): Could not parse mean anomaly:
# invalid float literal (line 6 is 70 characters; a TLE line is 69, so its columns may be shifted)
```

In Rust, `TLE::from_lines` and `TLE::from_url` stop at the first bad record in the same way. To keep the good records, iterate `TLE::records` instead, optionally with `.check_checksums(true)`, or use `TLE::records_from_url`:

```rust
let tles: Vec<satkit::TLE> = satkit::TLE::records(&lines).filter_map(Result::ok).collect();
```

[SGP4, TLEs and OMMs](../rust/sgp4.md) in the Rust section has a complete program: parsing, propagation, ground track, passes and fitting.

## Building a TLE

An element set can also be built in code. `sk.TLE()` takes its properties as keyword arguments, each checked as setting the property is, and any left out keep the values of an empty `TLE()` (name `"none"`, number 0, epoch J2000, elements zero). `TLE.fit_from_states()` takes `satnum`, `name` and `intl_desig` keywords for the identity of the fitted element set.

```python
import satkit as sk

tle = sk.TLE(
    name="ISS (ZARYA)", satnum=25544, intl_desig="98067A",
    epoch=sk.time(2021, 9, 21, 12, 25, 40.104192),
    inclination=51.6442, raan=208.5856, eccen=0.0001458,
    arg_of_perigee=47.2277, mean_anomaly=50.1624,
    mean_motion=15.48919419, bstar=5.868e-5,
)
print(tle.to_2line()[1])
# 2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419    04
```

## Loading from URLs

Both TLEs and OMMs can be loaded directly from a URL:

<!-- skip-test: needs the network (CelesTrak) -->
```python
import satkit as sk

# Load TLEs from a URL
tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")

# Load OMMs from a URL (auto-detects JSON, XML or CSV)
omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")

# OMMs work directly with sgp4()
pos, vel = sk.sgp4(omms[0], sk.time(2024, 6, 1))
```

A CelesTrak GP query with no `FORMAT` answers in CSV, which `omm_from_url()` reads; `TLE.from_url()` needs `FORMAT=tle` (or `3le` / `2le`), and gets HTTP 404 for an object numbered above 99,999 (see [Catalog numbers above 99,999](#catalog-numbers-above-99999)). CelesTrak rate-limits repeated identical queries, so save the response and reload it from disk rather than fetching on every run.

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
thetime = sk.time(2024, 4, 9, 12, 0, 0)
pTEME, _vTEME = sk.sgp4(starlink30477, thetime)

# Rotate to Earth-fixed (ITRF) and get geodetic coordinates
pITRF = sk.frametransform.rotation(
    from_frame=sk.frame.TEME, to_frame=sk.frame.ITRF, time=thetime,
) * pTEME
coord = sk.itrfcoord(pITRF)
print(coord)
# ITRFCoord(lat:  29.3890 deg, lon: 170.8051 deg, hae: 560.11 km)
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

# JSON, XML or CSV, detected from the content; one dict per message
omms = sk.omm_from_file("gp.json")

# Everything the source provided is in the dict
print(omms[0]["OBJECT_NAME"], omms[0].get("OBJECT_TYPE"), omms[0].get("RCS_SIZE"))

# The same element set as a TLE object, and back again
tle = sk.TLE.from_omm(omms[0])
print("\n".join(tle.to_2line()))
assert sk.TLE.from_omm(tle.to_omm()).to_2line() == tle.to_2line()
```
