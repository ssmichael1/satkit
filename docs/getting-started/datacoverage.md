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
    sk.utils.update_datafiles()   # re-downloads finals2000A.all (and SW-All.csv)
```

The warnings can be silenced with `satkit.frametransform.disable_eop_time_warning()`.

## Space weather coverage

`SW-All.csv` is not one table but three, and only the first is measurement:

| block | cadence | Kp / ap | what it is |
|---|---|---|---|
| observed | daily | yes | measured values, from GFZ Potsdam (geomagnetic) and DRAO / Natural Resources Canada (F10.7) |
| daily predicted | daily | yes | the NOAA/SWPC 45-day forecast |
| monthly predicted | **monthly** | **no** | monthly F10.7 only, running years ahead |

The last block is the one to watch. Its rows carry F10.7 but every `kp` and `ap`
field is the `-1` sentinel, so NRLMSISE-00 falls back to a quiet-time
$A_p = 4$ with no storm information at all. The table runs well past the daily
data — years of it — so a query never fails and nothing looks wrong:

```python
>>> satkit.spaceweather.get(satkit.time(2028, 3, 15))
{'date': 2028-03-01T00:00:00.000000Z, 'ap_avg': -1, 'data_type': 'PRM', ...}
```

Geomagnetic activity is not a small correction. Taking the 2024-05-11 Gannon
storm ($A_p = 271$) against 2024-05-14 ($A_p = 6$), with F10.7 almost matched at
213.7 and 219.8 so the geomagnetic term is isolated:

| altitude | storm | quiet | ratio |
|---|---|---|---|
| 300 km | 7.53e-11 | 4.51e-11 | 1.67x |
| 400 km | 1.80e-11 | 9.68e-12 | 1.86x |
| 550 km | 2.88e-12 | 1.35e-12 | 2.14x |

So a long-horizon drag run past the daily data is pinned to the quiet floor.
That is a limitation of the input data, not a satkit defect — but it should be a
visible one.

| `satkit.spaceweather.status(t)` | meaning | what satkit does |
|---|---|---|
| `"observed"` | on or before the last measured row | returns that day's record |
| `"predicted_daily"` | inside the NOAA/SWPC 45-day forecast | returns the forecast row; F10.7 and ap are both present |
| `"predicted_monthly"` | past the daily rows | returns the most recent **monthly** row: F10.7 only, no geomagnetic data, NRLMSISE-00 runs on $A_p = 4$. **One-time warning** |
| `"extrapolated"` | after the last row of the table | returns that row unchanged. **One-time warning** |
| `"before_table"` | before 1957 | `RuntimeError` |
| `"not_loaded"` | no table at all | `RuntimeError`, **one-time warning** |

`satkit.spaceweather.coverage()` returns
`(first, last_observed, last_daily, last)` as `satkit.time` values, or `None`
if nothing is loaded. `last_daily` is the boundary that matters for drag:

```python
import satkit as sk

first, last_observed, last_daily, last = sk.spaceweather.coverage()
if sk.spaceweather.status(t_end) == "predicted_monthly":
    sk.utils.update_datafiles()   # may move last_daily forward by up to 45 days
```

Refreshing helps only so far: the daily block ends about 45 days out by
construction, so any propagation beyond that horizon is in the monthly regime no
matter how fresh the file is.

Every record also carries its own provenance in `data_type` — `"OBS"` measured,
`"INT"` interpolated across a gap in the measured record, `"PRD"` daily
prediction, `"PRM"` monthly prediction:

```python
>>> sk.spaceweather.get(sk.time(2023, 3, 1))["data_type"]
'OBS'
```

The warnings can be silenced with
`satkit.spaceweather.disable_space_weather_time_warning()`.
