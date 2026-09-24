# Data Coverage

Two of satkit's data files are time-bounded tables that are refreshed daily: the
Earth-orientation parameters and the space-weather indices. Both answer queries
for epochs they do not actually cover, so it is worth knowing where an epoch
falls before trusting the result — particularly for propagation into the future.

Both subsystems expose the same two-function vocabulary: a `*_coverage()` that
returns the table bounds, and a `*_status(t)` that classifies one epoch against
them.

## EOP coverage

Every Earth-fixed frame transform, every UT1-based quantity (`gmst`, `gast`, Earth rotation angle), and the high-precision propagator depend on the EOP table, so it matters where an epoch falls relative to it:

| `satkit.frametransform.eop_status(t)` | meaning | what satkit does |
|---|---|---|
| `"observed"` | on or before the last observed (`O`) row | interpolates measured values |
| `"predicted"` | after the last observed row, inside the table | interpolates IERS predictions (~6 months ahead) |
| `"extrapolated"` | after the last row | holds the last row constant and prints a **one-time warning**. Polar motion drifts ~0.1″ and $\Delta UT1$ ~10 ms over a few months — metres of position error at LEO |
| `"before_table"` | before 1962 | zeros, one-time warning |
| `"not_loaded"` | no table at all (first use offline, or the fetch failed) | zeros, one-time warning; **`propagate` refuses to run** (`RuntimeError`) |

`satkit.frametransform.eop_coverage()` returns `(first, last_observed, last)` as `satkit.time` values, or `None` if nothing is loaded; `satkit.frametransform.eop_source()` reports which file the table came from (`"finals2000A"` or `"celestrak"`). For precision work, propagate with `satkit.propsettings(require_eop_coverage=True)`: the propagator then raises instead of extrapolating past the table, and the fix is simply to refresh the file:

```python
import satkit as sk

first, last_observed, last = sk.frametransform.eop_coverage()
if sk.frametransform.eop_status(t_end) == "extrapolated":
    sk.utils.update_datafiles()   # re-downloads finals2000A.all (and the space-weather files)
```

The warnings can be silenced with `satkit.frametransform.disable_eop_time_warning()`.

## Space weather coverage

The space-weather table is assembled from three sources, and which one an
epoch falls in decides how much the density model actually knows:

| block | source | cadence | $K_p$ | $a_p$ / $A_p$ |
|---|---|---|---|---|
| observed | GFZ Potsdam | daily, 1932 → about yesterday | 3-hourly | 3-hourly + daily |
| daily predicted | NOAA/SWPC 45-day forecast | daily, ~45 days | — | daily $A_p$, held across the eight slots |
| monthly predicted | NASA MSAFE | monthly, decades | — | daily $A_p$ (13-month-smoothed climatology), held across the eight slots |

Through satkit 0.22 the table was CelesTrak's merged space-weather file, whose
monthly rows carry $F_{10.7}$ but **no** geomagnetic data, so past the 45-day
forecast NRLMSISE-00 silently fell back to a quiet-time $A_p = 4$. Geomagnetic activity
is not a small correction — taking the 2024-05-11 Gannon storm ($A_p = 271$)
against 2024-05-14 ($A_p = 6$), with $F_{10.7}$ almost matched so the
geomagnetic term is isolated:

| altitude | storm | quiet | ratio |
|---|---|---|---|
| 300 km | 7.53e-11 | 4.51e-11 | 1.67x |
| 400 km | 1.80e-11 | 9.68e-12 | 1.86x |
| 550 km | 2.88e-12 | 1.35e-12 | 2.14x |

MSAFE closes that gap: its monthly rows carry a climatological $A_p$, so a
long-horizon drag run past the daily data runs on the expected level of
activity for that point in the solar cycle rather than on the quiet floor.
What it cannot give is storm timing — that is unknowable months ahead — and
the model has no 3-hourly structure past the observed record, so
`predicted_daily` and `predicted_monthly` are honest labels for the answer's
quality, not just the row's origin.

| `satkit.spaceweather.status(t)` | meaning | what satkit does |
|---|---|---|
| `"observed"` | on or before the last measured row | returns that day's record, with the 3-hourly $a_p$ history NRLMSISE-00 prefers |
| `"predicted_daily"` | inside the SWPC 45-day forecast | returns the forecast row: daily $F_{10.7}$ and $A_p$ |
| `"predicted_monthly"` | past the daily rows | returns that month's MSAFE row: smoothed $F_{10.7}$ and $A_p$. **One-time warning** only if the row carries no $A_p$ (a table loaded from a file that lacks it) |
| `"extrapolated"` | after the last row of the table | returns that row unchanged. **One-time warning** |
| `"before_table"` | before 1932 | `RuntimeError` |
| `"not_loaded"` | no table at all | `RuntimeError`, **one-time warning** |

`satkit.spaceweather.coverage()` returns
`(first, last_observed, last_daily, last)` as `satkit.time` values, or `None`
if nothing is loaded:

```python
import satkit as sk

first, last_observed, last_daily, last = sk.spaceweather.coverage()
if sk.spaceweather.status(t_end) != "observed":
    sk.utils.update_datafiles()   # observed record to ~yesterday, forecasts refreshed
```

Refreshing moves `last_observed` to about yesterday and `last_daily` about 45
days past it; `last` is decades out and never the binding constraint.

Every record carries its provenance in `data_type` — `"OBS"` measured and
definitive, `"OBS-P"` measured but still preliminary (GFZ's nowcast, the last
few days), `"INT"` interpolated across a gap, `"PRD"` daily prediction,
`"PRM"` monthly prediction:

```python
>>> sk.spaceweather.get(sk.time(2023, 3, 1))["data_type"]
'OBS'
```

The 81-day averages that feed NRLMSISE-00's $F_{10.7A}$ are computed across
the assembled series — a centred window reaches 40 days into the forecast —
with the same convention as CelesTrak's published columns (mean over
$[t-40, t+40]$ and $[t-80, t]$), which they reproduce to the rounding digit.

`satkit.spaceweather.init_from_path()` loads a GFZ table directly, for a
comparison that must pin against one fixed input file. The MSAFE percentile bands are in the file but not yet exposed
through the API. The warnings can be silenced with
`satkit.spaceweather.disable_space_weather_time_warning()`.
