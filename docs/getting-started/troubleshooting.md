# Troubleshooting & FAQ

Symptom-first answers to the problems people most often hit with `satkit`.
Each entry gives the cause and the fix in a few lines and links to the page
that covers the subject in full.

When something about data files looks wrong, start by asking satkit what it
is using:

```python
import satkit as sk

print(sk.__version__)
print(sk.utils.datadir())               # where downloads are written
print(sk.utils.data_search_dirs())      # where files are looked up, in order
print(sk.utils.is_offline())            # downloads forbidden?
print(sk.frametransform.eop_source())   # "finals2000A", "celestrak" or None
print(sk.frametransform.eop_coverage()) # (first, last_observed, last) or None
print(sk.spaceweather.coverage())       # (first, last_observed, last_daily, last) or None
```

## Warnings about Earth orientation and space weather

### "Warning: EOP data ends at … held constant"

```
Warning: EOP data ends at 2027-10-02T00:00:00.000000Z (MJD 61680); the request for MJD UTC = 62502 and all later epochs use the last entry's values held constant. ...
```

**Cause.** An Earth-fixed frame transform (or a propagation, or `gmst`/`gast`)
was asked for an epoch past the last row of the Earth-orientation (EOP)
table. Either the epoch is genuinely in the future beyond the IERS
predictions (about a year ahead), or the file on disk is old: satkit fetches
the EOP file on first use but does **not** refresh it on its own after that.

**Impact.** Polar motion and $\Delta UT1$ are held at their last values. They
drift by ~0.1″ and ~10 ms over a few months, i.e. metres of position error at
LEO.

**Fix.** Refresh the file, which also reloads the table in the running process:

```python
sk.utils.update_datafiles()
```

Check an epoch before relying on it with `sk.frametransform.eop_status(t)`,
and pass `sk.propsettings(require_eop_coverage=True)` to make `propagate`
raise instead of extrapolating. See [EOP coverage](datacoverage.md#eop-coverage).

### "Warning: no Earth Orientation Parameters (EOP) table is loaded"

**Cause.** No `finals2000A.all` or `EOP-All.csv` is in any search directory
and the first-use download failed (no network, `SATKIT_OFFLINE=1`, a proxy,
a read-only data directory).

**Impact.** Polar motion, $\Delta UT1$ and the celestial-pole offsets are
treated as zero, and frame transforms are off by up to ~0.5″ (metres at LEO).
`propagate()` refuses to run rather than integrate with a mis-oriented
gravity field (`RuntimeError: no Earth Orientation Parameters (EOP) table is loaded ...`).

**Fix.** Run `sk.utils.update_datafiles()` on a machine with network access,
or point `SATKIT_DATA` at a directory that holds the file. If the download
itself fails, the entries under [Data files and downloads](#data-files-and-downloads)
below cover the usual reasons.

### "Warning: EOP data not available for MJD UTC = … (too early)"

**Cause.** The epoch is before the start of the loaded EOP table. The IERS
`finals2000A.all` file satkit downloads begins in **1973**; CelesTrak's
`EOP-All.csv` reaches back to 1962, but it is only downloaded when both IERS
mirrors are unreachable. Despite what the warning says, refreshing the data
does not move the start of the table.

**Fix.** For 1962–1972, place a copy of
[`EOP-All.csv`](https://celestrak.org/SpaceData/EOP-All.csv) in
`sk.utils.datadir()` (or any search directory) and restart the process. With
both files present the table starts in 1962: satkit uses the file whose
observed record runs later and, when that is the IERS file, puts the CSV's
pre-1973 rows in front of it. satkit has no EOP data before 1962; zeros are
used there.

### Space-weather warnings

| message begins | cause | fix |
|---|---|---|
| `Warning: no space-weather table is loaded` | no space-weather files on disk and the first-use download failed. NRLMSISE-00 runs on $F_{10.7} = F_{10.7A} = 150$, $A_p = 4$, which can be wrong by a factor of two in density | `sk.utils.update_datafiles()` with network access, or `SATKIT_DATA` pointing at a directory with the files |
| `Warning: the space-weather table ends at …` | the epoch is past the last row; that row's values are used unchanged | the default table runs decades ahead (NASA MSAFE monthly forecast), so this usually means an old file or a table loaded by hand — refresh with `sk.utils.update_datafiles()`. For epochs past the MSAFE forecast, refreshing cannot help |
| `Warning: the space-weather record for … is a monthly prediction` | the record carries $F_{10.7}$ but no $K_p$/$a_p$ (a CelesTrak `SW-All.csv` left in a data directory by satkit 0.22 or earlier, or loaded by hand), so NRLMSISE-00 uses a quiet-time $A_p = 4$ | `sk.utils.update_datafiles()` fetches the GFZ / SWPC / MSAFE files, whose monthly rows do carry $A_p$ |

`sk.spaceweather.get(t)` itself raises `RuntimeError: No space weather record found for date`
when no table is loaded. See [Space weather coverage](datacoverage.md#space-weather-coverage)
for `sk.spaceweather.status(t)` and what each block of the table can tell the density model.

### How do I silence these warnings?

They are printed directly to **stderr** by the Rust library, once per process
each, not through Python's `warnings` module — so `warnings.filterwarnings(...)`
and `python -W ignore` have no effect, and in Jupyter they appear as a stderr
block under the cell. Turn them off with:

```python
sk.frametransform.disable_eop_time_warning()
sk.spaceweather.disable_space_weather_time_warning()
```

The EOP warnings name only the Rust function
(`satkit::earth_orientation_params::disable_eop_time_warning()`); in Python it
is `sk.frametransform.disable_eop_time_warning()`. Silencing a warning does not
change the result: prefer refreshing the data, or checking
`eop_status(t)` / `spaceweather.status(t)` explicitly.

### The Earth-orientation or space-weather data is old, although I have network access

**Cause.** satkit downloads these files on first use only when no copy exists
in *any* search directory, and never refreshes them on its own. A copy that
is already there — from an earlier run, a legacy `~/.satkit-data`, a directory
copied from another machine, or an offline bundle — is used as it is, and it
ages from the day it was written.

**Fix.** Run `sk.utils.update_datafiles()` periodically (it is cheap: see
[below](#update_datafiles-says-no-request-made-how-do-i-force-a-refresh)).
The fresh files go to `sk.utils.datadir()`. For Earth orientation, when
several copies exist satkit uses whichever has the later observed record; for
space weather, the GFZ / SWPC / MSAFE files are preferred over an older
CelesTrak `SW-All.csv`. See [Data Directories](datadirs.md).

## Data files and downloads

### Where are my data files, and which copy is being used?

Downloads are written to exactly one directory, `sk.utils.datadir()`:
`SATKIT_DATA` if set, else the directory given to `sk.utils.set_datadir()`,
else the platform user-data directory (`~/Library/Application Support/satkit-data`
on macOS, `~/.local/share/satkit-data` on Linux, `%LOCALAPPDATA%\satkit-data`
on Windows). Files are looked up across `sk.utils.data_search_dirs()` and used
from the first directory that contains them. Nothing is ever written inside
`site-packages`. Full table: [Data Directories](datadirs.md#where-satkit-looks-for-data-and-where-it-writes).

What needs no data directory at all: the IERS nutation tables and the EGM96 /
EGM2008 / JGM-2 / JGM-3 gravity models are compiled in, so SGP4, time scales,
gravity, Keplerian propagation and Lambert targeting work straight after install.
See [Data Files](datafiles.md).

### The first `propagate()` or ephemeris query takes a long time

The first call that needs the JPL ephemeris downloads it: DE440, 102 MB,
verified against a SHA-256 compiled into satkit. That happens once; later
loads read the file from `datadir()`. To take the hit up front (a container
build, a CI cache), run `sk.utils.update_datafiles()`; to use the 14 MB DE421
(1900–2053) instead, set `SATKIT_JPLEPHEM_FILE=lnxp1900p2053.421`. See
[Selecting a JPL ephemeris file](datadirs.md#selecting-a-jpl-ephemeris-file).

### `update_datafiles()` says "no request made" — how do I force a refresh?

```
  finals2000A.all: current (2.3 h old); no request made
```

This is intended. The Earth-orientation and space-weather files are only
re-requested once their publication cadence has passed (3 h for the GFZ
record, 24 h for the SWPC forecast and the EOP file, a week for MSAFE), and
then with a conditional request that costs a `304` if nothing changed. Calling
`update_datafiles()` at the top of every script is therefore fine.

To force a transfer anyway:

- `sk.utils.update_datafiles(overwrite=True)` re-downloads **everything**,
  including the 102 MB ephemeris;
- deleting a file's `<name>.http-cache` sidecar in `datadir()` forces a full
  fetch of just that file on the next `update_datafiles()`.

The keyword is `overwrite`. Other keyword arguments (`force=True`, say) are
currently ignored without an error. See
[How often EOP and space weather are refreshed](datadownloads.md#how-often-eop-and-space-weather-are-refreshed).

### "No writeable data directory" or "Read-only file system" (containers, shared installs)

```
RuntimeError: No writeable data directory: /data/satkit could not be created or is not writable (...). Set SATKIT_DATA to a directory satkit may write to
```

satkit only writes to `datadir()`. In a container with a read-only root
filesystem, a service account with no home directory, or a shared install,
point `SATKIT_DATA` at a writable volume (or call `sk.utils.set_datadir(path)`
with an existing directory before the first data access; `SATKIT_DATA`
takes precedence over it). Read-only directories are fine as **search**
locations: provision the files once (below) and they are read from there.
With `update_datafiles(dir=...)` the directory is used as given, so a
read-only one fails with the operating system's error (e.g.
`Read-only file system (os error 30)`).

### Downloads fail with `invalid peer certificate: UnknownIssuer`

```
RuntimeError: could not fetch https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt: io: invalid peer certificate: UnknownIssuer
```

**Cause.** Something between you and the server re-signed the TLS connection,
typically a corporate TLS-inspecting proxy whose private CA is not in the trust
store satkit is using. satkit verifies against the **operating system's**
trust store (macOS keychain, Windows certificate store, `/etc/ssl` on Unix),
and ignores `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE`.

**Fix.** Install the proxy's CA system-wide, or set
`SATKIT_CA_BUNDLE=/path/bundle.pem` to a PEM file containing that CA
**together with** the public roots (the file replaces the trust store). In a
minimal container with no system trust store, `SATKIT_CA_BUNDLE=webpki` uses
the Mozilla roots compiled into satkit. There is no "skip verification"
switch. See [Downloads behind a TLS-inspecting proxy](datadownloads.md#downloads-behind-a-tls-inspecting-proxy).

### Downloads have to go through an HTTP proxy

The standard `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` / `NO_PROXY`
environment variables are honoured for every download. Where outbound access
is blocked entirely, `SATKIT_DATA_URL` can name an internal mirror for the
hash-pinned files (the JPL ephemeris, ITU_GRACE16): it is tried first, plain
`http://` is accepted, and the files are still verified. The mirror is not
consulted for the Earth-orientation and space-weather files, which always come
from their producers; on such a network, provision those by copying a data
directory ([below](#running-offline-or-air-gapped)). See
[Environment variables and API](datadirs.md#environment-variables-and-api).

### HTTP 503 (or 403) from CelesTrak

```
CelesTrak returned HTTP 503 for https://celestrak.org/NORAD/elements/gp.php?... CelesTrak throttles repeated identical GP queries ...
```

**Cause.** CelesTrak rate-limits clients that fetch the same element sets
repeatedly — typically a script or notebook that calls `sk.TLE.from_url(...)`
or `sk.omm_from_url(...)` on every run. It asks for at most one request per
object every ~2 hours. A 403 can also come from a filtering proxy.

**Fix.** Do not retry in a loop. Save the response text once and parse the
saved copy with `sk.TLE.from_file(...)` / `sk.TLE.from_lines(...)` or
`sk.omm_from_file(...)` / `sk.omm_from_text(...)`, re-fetching only when you
need newer elements. satkit's own data refreshes do not contribute to this:
they go to the IERS, GFZ, NOAA and NASA servers (CelesTrak only as the
Earth-orientation fallback) and are rate-limited to each file's publication
cadence.

### Running offline or air-gapped

Everything compiled in works with no network. What needs files is the JPL
ephemeris (the `jplephem` module and the numerical propagator; the
low-precision analytic `sun`, `moon` and `planets` modules do not use it), the EOP table
(exact Earth-fixed frames; required by `propagate`), the space-weather table
(drag), and the ITU_GRACE16 gravity model if you select it. To provision a
machine that will be offline:

1. On a connected machine, fill a directory:
   `SATKIT_DATA=/path/to/satkit-data python -c "import satkit; satkit.utils.update_datafiles()"`.
2. Copy that directory to the target and set `SATKIT_DATA` to it there (or
   copy it to the platform user-data directory, or to `/usr/share/satkit-data`
   as a read-only system-wide location).
3. Optionally set `SATKIT_OFFLINE=1` there (or call `sk.utils.set_offline(True)`)
   so that a missing file raises immediately instead of attempting a connection.

The Earth-orientation and space-weather files still age on an offline
machine: repeat the copy periodically, and check `eop_coverage()` against the
epochs you work with. `pip install satkit[data]` is an alternative for the
ephemeris. See [Offline and air-gapped use](installation.md#offline-and-air-gapped-use)
and [Provisioning up front](datadirs.md#provisioning-up-front).

### "… is not present and cannot be downloaded (SATKIT_OFFLINE is set)"

```
RuntimeError: ... linux_p1550p2650.440 is not present and cannot be downloaded (SATKIT_OFFLINE is set). Provide it in the data directory (SATKIT_DATA) or install the `satkit-data` bundle; sources: https://github.com/ssmichael1/satkit-data/releases/download/data-v1/linux_p1550p2650.440, ...
```

Downloads are forbidden (by `SATKIT_OFFLINE`, or by `sk.utils.set_offline(True)`,
which gives the same message) and the file is in none of the search
directories. The message lists the URLs the file can be fetched from by hand;
put it in `datadir()` or any search directory. Check `sk.utils.is_offline()`
if you did not expect offline mode: a leftover `SATKIT_OFFLINE` in a CI
environment is a common cause.

## Installation

### `pip install satkit` tries to compile, or fails asking for Rust

Pre-built wheels exist for CPython 3.10–3.14 on Linux x86_64 and aarch64
(glibc), macOS on Apple silicon (arm64), and Windows x86_64. Anywhere else pip
falls back to the source distribution, which needs a stable Rust toolchain
([rustup](https://rustup.rs)). In particular:

- **Intel Macs** have had no wheel since 0.22. Install from conda-forge, which
  builds `osx-64`, or from source with `pip install --no-binary satkit satkit`.
- **Alpine and other musl-based Linux images** have no wheel; build from source
  or use a glibc-based image.
- **Python 3.9 and older** are not supported (`requires-python >= 3.10`).

See [Installation](installation.md).

### Should I use pip or conda?

Either. The conda-forge package (`conda install -c conda-forge satkit`) is
built from the same source distribution and behaves identically: core data
compiled in, ephemeris downloaded on first use, the same environment
variables. Differences: conda-forge also builds for Intel Macs, and there is
no conda counterpart of the optional `satkit-data` bundle — offline conda
users provision a directory with `update_datafiles()` and set `SATKIT_DATA`
(above). See [Conda](installation.md#conda).

## API changes that look like errors

### `DeprecationWarning: time.as_mjd() is deprecated since 0.23 ...`

In 0.23 the Python `as_*` conversion methods were renamed `to_*`, to pair with
the `from_*` constructors. The old names still work but warn, and are
**removed in 0.25**:

| deprecated | use |
|---|---|
| `time.as_date()`, `as_gregorian()`, `as_datetime()`, `as_mjd()`, `as_jd()`, `as_unixtime()`, `as_iso8601()`, `as_rfc3339()` | `to_date()`, `to_gregorian()`, `to_datetime()`, `to_mjd()`, `to_jd()`, `to_unixtime()`, `to_iso8601()`, `to_rfc3339()` |
| `time.datetime()` | `time.to_datetime()` |
| `quaternion.as_rotation_matrix()`, `as_euler()` | `to_rotation_matrix()`, `to_euler()` |
| `kepler.w` (constructor keyword and property) | `kepler.argp` (`w` is kept, with a warning) |

Python hides `DeprecationWarning` outside `__main__` by default; run with
`python -W error::DeprecationWarning` to find every remaining call.

### `TypeError: 'float' object is not callable` (or `'int'`, or `'satkit.weekday'`)

A zero-argument member that describes the object is a **property** — call it
without parentheses. Since 0.22.1 this includes `quaternion.norm`,
`time.day_of_year`, `time.weekday` and `tlefitstatus.converged`, alongside
`quaternion.angle` and `propresult.can_interp`:

```python
q = sk.quaternion.rotz(0.1)
t = sk.time(2024, 1, 1)
q.norm, t.day_of_year, t.weekday   # not q.norm(), t.day_of_year(), t.weekday()
```

The reverse also changed: `quaternion.conj` and `quaternion.conjugate` are
**methods** (`q.conj()`), like `q.inverse()`. Without the parentheses you get
the bound method, and using it fails with
`TypeError: unsupported operand type(s) for *: 'builtin_function_or_method' and ...`.
The rule: a property tells you something about the object; a method gives you
something to use instead of it.

## Results that look wrong

### Times are off by 69.184 s (or 37 s, or 18 s)

`satkit.time` values are unambiguous instants, but constructors and
conversions interpret numbers as **UTC** unless told otherwise, and times
print in UTC. TT − UTC is 69.184 s, TAI − UTC 37 s, GPS − UTC 18 s (since 2017).
Pass the scale explicitly:

```python
t = sk.time(2024, 1, 1, 12, 0, 0, scale=sk.timescale.TT)
print(t)                           # 2024-01-01T11:58:50.816000Z  (printed in UTC)
t.to_mjd(sk.timescale.TT)          # to_mjd() alone means UTC
sk.time.from_mjd(60310.0, sk.timescale.TT)
```

See the [Time Systems tutorial](../tutorials/Time%20Systems.ipynb).

### Adding a number to a time jumps by days

`time + float` adds **days**: `t + 60` is two months later, not one minute.
Use a `duration` for anything else:

```python
t + sk.duration.from_seconds(60)
t + sk.duration.from_hours(2)
```

### `RuntimeError: Must pass in year, month, day or year, month, day, hour, min, sec`

The `time` constructor takes either the date alone or all six fields:
`sk.time(2024, 1, 1)` or `sk.time(2024, 1, 1, 12, 0, 0)`, not
`sk.time(2024, 1, 1, 12)`.

### A `datetime` without a time zone is read as local time

`sk.time.from_datetime(dt)` (and SGP4 given a list of `datetime`s) converts the
instant the `datetime` represents, and Python treats a naive `datetime` as
**local** time. On a machine in New York, `datetime(2024, 1, 1, 12)` becomes
17:00 UTC. This is intentional and matches Python's own convention
(`datetime.timestamp()`, `datetime.now()`), and `to_datetime(utc=False)`
returns a naive local-time `datetime` that round-trips. For UTC, attach a time
zone (`tzinfo=datetime.timezone.utc`) or construct `sk.time` directly; avoid
`datetime.utcnow()`, which returns a naive UTC value (and is deprecated since
Python 3.12).

### SGP4 positions are 1000 times larger than expected, or in the wrong frame

`sk.sgp4()` returns **metres** and **metres/second** — like every other
position in satkit, not the kilometres of most SGP4 libraries — in the
**TEME** frame. Rotate before comparing with anything in GCRF or ITRF, and
before using an SGP4 state as the initial state of `propagate` (which works
in GCRF):

```python
import numpy as np

tle = sk.TLE.from_lines([
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  30306-3 0  9993",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815367432047",
])
p_teme, v_teme = sk.sgp4(tle, tle.epoch)
q = sk.frametransform.qteme2gcrf(tle.epoch)
state_gcrf = np.concatenate([q * p_teme, q * v_teme])
res = sk.propagate(state_gcrf, tle.epoch, duration_days=1.0)
```

`sk.frametransform.rotation(from_frame=sk.frame.TEME, to_frame=sk.frame.ITRF, tm=t)`
gives the rotation to the Earth-fixed frame. Remember that a TLE holds SGP4
*mean* elements, which are not osculating Keplerian elements. See
[TLEs, SGP4 & OMMs](../guide/tle.md).

### `ValueError: invalid Keplerian element incl = 51.6: inclination must be in [0, π] radians`

`sk.kepler` takes angles in **radians** and distances in **metres**. TLE
fields are in **degrees** (`tle.inclination`, `tle.raan`, …), so copying them
across needs `math.radians()`; only the inclination is range-checked, so a
RAAN or anomaly in degrees is accepted silently. A semi-major axis in
kilometres is accepted too (it is positive), and describes an orbit inside the
Earth. `sk.itrfcoord` has separate `latitude_deg=` / `latitude_rad=` keywords
and takes altitude in metres.

### `sgp4()` returns `nan`

SGP4 failed for that element set and time, most often because the satellite
has decayed or the time is far from the TLE epoch. Pass `errflag=True` to get
the error code for each output (`sk.sgp4_error.orbit_decay`, etc.):

```python
p, v, err = sk.sgp4(tle, times, errflag=True)
```

A TLE with ephemeris type 4 raises
`RuntimeError: Ephemeris type 4 (SGP4-XP) is not supported ...` instead:
satkit implements classic SGP4 only.

### My propagation disagrees with GMAT, STK or Orekit

Check the force model before suspecting the integrator. The `propsettings`
defaults are a quick, not a high-fidelity, configuration:

- **Gravity** is 4×4 by default (`gravity_degree`, `gravity_order`; maximum
  70). For precision LEO work use tens of degrees. The default model is EGM2008 since
  0.23 (EGM96 before); `gravity_model=sk.gravmodel.egm96` reproduces older results.
- **Drag and solar radiation pressure** are only applied when you pass
  `satproperties=sk.satproperties(cdaoverm=..., craoverm=...)`; without it
  neither force is included. Drag is skipped above ~700 km altitude.
- **Space weather and EOP** follow the tables above: past their coverage the
  inputs are forecasts or held constant.

See the [Force Model](../guide/forces.md) guide and
[Validation: GMAT Comparison](../guide/gmat_validation.md) for measured agreement.
