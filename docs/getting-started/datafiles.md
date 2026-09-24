# Data Files

`satkit` needs three kinds of data, and handles them differently by size and by how often they change:

| tier | files | how it is provided |
|---|---|---|
| **Compiled in** | IERS Conventions (2010) Tables 5.2a/b/d (nutation and CIO series); EGM96, EGM2008, JGM-2 and JGM-3 gravity coefficients to degree 70 | gzip'd into the library (~300 KB) and inflated on first use. Frame transforms and gravity work with **no data directory and no network** |
| **Downloaded once, on first use** | JPL DE440 ephemeris `linux_p1550p2650.440` (102 MB), or DE421 `lnxp1900p2053.421` (14 MB); the ITU_GRACE16 gravity model `ITU_GRACE16.gfc` (1.8 MB) | fetched the first time a planet, Sun or Moon position (or `gravmodel.itugrace16`) is needed, SHA-256 verified against a manifest compiled into satkit, written to the [data directory](datadirs.md#where-satkit-looks-for-data-and-where-it-writes) |
| **Refreshed** | `finals2000A.all` (Earth orientation, IERS); `Kp_ap_Ap_SN_F107_since_1932.txt` (observed space weather, GFZ Potsdam), `45-day-forecast.txt` (NOAA/SWPC) and `msafe-f10-prd.txt` (NASA MSFC monthly forecast) | change daily to monthly; fetched on first use and refreshed by `satkit.utils.update_datafiles()`. CelesTrak's `EOP-All.csv` is the Earth-orientation fallback, and its `SW-All.csv` is still read when present |

Everything that does not need the ephemeris or Earth orientation — gravity accelerations, the precession-nutation part of the frame chain, SGP4, time scales, Keplerian propagation, Lambert targeting — therefore works immediately after `pip install satkit`, offline. The numerical propagator needs the ephemeris (Sun and Moon) and the Earth-fixed frame chain needs the EOP file.

Two caveats on "offline". Frame transforms need Earth-orientation parameters as well as the compiled-in nutation tables: with an Earth-orientation file (`finals2000A.all` or `EOP-All.csv`) present in a search directory they are exact; with none at all (a first run with no network) they fall back to zero polar motion and $\Delta UT1$, warn once, and are off by up to ~0.5″ (metres at LEO), while `propagate()` refuses to run (`EopUnavailable`) rather than integrate with a tilted gravity field. And the ephemeris is only "offline" once it has been downloaded (or provisioned by hand): `SATKIT_OFFLINE=1` turns a missing ephemeris into an error, not a degraded answer.

## The files

- **linux_p1550p2650.440** — File containing the precise ephemerides of the planets and 400 large asteroids between the years 1550 and 2650, as modelled by the Jet Propulsion Laboratory (JPL) — the DE440 ephemeris of [Park et al. (2021)](../guide/references.md#park2021). Large (~100 MB); downloaded on first use. The smaller `lnxp1900p2053.421` (DE421, [Folkner et al. 2009](../guide/references.md#folkner2009), ~14 MB, 1900–2053) is an alternative — see [Selecting a JPL ephemeris file](datadirs.md#selecting-a-jpl-ephemeris-file).

- **tab5.2a.txt**, **tab5.2b.txt**, **tab5.2d.txt** — Tables 5.2a, 5.2b and 5.2d of the IERS Conventions (2010), Technical Note 36 ([Petit & Luzum 2010](../guide/references.md#petit2010)): the CIP $X$, $Y$ and CIO-locator $s$ series used in the precise rotation between the inertial International Celestial Reference Frame and the Earth-fixed International Terrestrial Reference Frame. Compiled in.

- **EGM96.gfc**, **EGM2008.gfc**, **JGM2.gfc**, **JGM3.gfc** — Gravity coefficients for EGM96 ([Lemoine et al. 1998](../guide/references.md#lemoine1998)), EGM2008 ([Pavlis et al. 2012](../guide/references.md#pavlis2012)), JGM-2 ([Nerem et al. 1994](../guide/references.md#nerem1994)) and JGM-3 ([Tapley et al. 1996](../guide/references.md#tapley1996)), in the ICGEM `.gfc` format ([Ince et al. 2019](../guide/references.md#ince2019)). Compiled in, truncated to degree 70, the evaluator's cap, so results are identical to the full files. A full-degree copy placed in a data directory is used in preference.

- **ITU_GRACE16.gfc** — Gravity coefficients for ITU_GRACE16 ([Akyilmaz et al. 2016](../guide/references.md#akyilmaz2016)), a GRACE-only satellite solution to degree 180. Licensed CC BY 4.0, so it is not compiled in: downloaded (1.8 MB, verified) on first use of `gravmodel.itugrace16`; with `SATKIT_OFFLINE=1` and no copy on disk, selecting it is a `RuntimeError`. Results derived from it should cite the model.

- **Kp_ap_Ap_SN_F107_since_1932.txt** — Observed space weather: the 3-hourly $K_p$ and $a_p$ geomagnetic indices, daily $A_p$, and the 10.7 cm solar radio flux $F_{10.7}$ (observed and adjusted to 1 AU), daily since 1932, from [GFZ Potsdam](https://kp.gfz.de/) — the producer of the IAGA-endorsed $K_p$ series, redistributing the DRAO / Natural Resources Canada flux alongside it. Licensed CC BY 4.0 ([Matzka et al. 2021](../guide/references.md#matzka2021)); results derived from it should cite the index. The file's sunspot-number column is CC BY-NC 4.0 (SILSO) and is **not** ingested. $F_{10.7}$ is the primary driver of thermospheric density at low-Earth-orbit altitudes and $A_p$ the geomagnetic one; both feed NRLMSISE-00. Refreshed every 3 hours.

- **45-day-forecast.txt** — The [NOAA/SWPC 45-day $A_p$ and $F_{10.7}$ forecast](https://services.swpc.noaa.gov/text/45-day-forecast.txt), daily resolution from the day after the observed record. US Government work. Refreshed daily.

- **msafe-f10-prd.txt** — The NASA Marshall Space Flight Center [Solar Activity Future Estimation (MSAFE)](https://www.nasa.gov/solar-cycle-progression-and-forecast/): monthly 13-month-smoothed $F_{10.7}$ **and $A_p$** with 95 / 50 / 5 percentile bands, for the balance of the current solar cycle and a mean cycle beyond. US Government work. NASA publishes one file a month under a month-specific name, so satkit stores it under this stable one and walks back from the current month to find the newest issue. Refreshed weekly. This is the source that carries geomagnetic data past the daily record — see [Data coverage](datacoverage.md#space-weather-coverage).

- **predicted-solar-cycle.json** — [NOAA/SWPC solar cycle forecast](https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json), monthly predicted $F_{10.7}$ ~5 years ahead. Still fetched, but with MSAFE in the table it is no longer reached by the density model; kept for `satkit.spaceweather.predicted_f107()`.

- **finals2000A.all** — Earth orientation parameters. This includes $\Delta UT1$, the difference between $UT1$ and $UTC$, as well as $x_p$ and $y_p$, the polar "wander" of the Earth rotation axis, and the $dX$, $dY$ celestial-pole offsets. It is the [IERS Rapid Service / Prediction Centre](../guide/references.md#iers-finals2000a)'s Bulletin A combined file: observed values from 1973 and about a year of predictions, updated daily, fetched from the USNO mirror and then the IERS data centre. When both mirrors are unreachable satkit falls back to CelesTrak's **EOP-All.csv** ([CelesTrak Space Data](../guide/references.md#celestrak-spacedata)), a repackaging of the same IERS series that reaches back to 1962 and carries about six months of predictions; an `EOP-All.csv` already in a data directory (a hand-provisioned machine, the `satkit-data` bundle) is still read, and when both files are present the one whose observed record runs later is used, with the CSV's pre-1973 rows kept in front of the IERS table. `satkit.frametransform.eop_source()` says which one is loaded. For dates beyond the file, the last entry's values are used (constant extrapolation) — see [EOP coverage](datacoverage.md#eop-coverage) below.

- **Leap seconds** — not a file: the UTC↔TAI leap-second table is compiled into the library (current through the most recent leap second, 2017-01-01, when UTC began lagging TAI by 37 s), and a future leap second will require a new `satkit` release. The table is transcribed from [IERS Bulletin C](../guide/references.md#bulletinc); UTC and leap seconds are defined by [ITU-R TF.460-6](../guide/references.md#itu460). (Releases through 0.21.2 also downloaded a reference `leap-seconds.list`; nothing ever read it, and it is no longer fetched.)

## Where to next

- **[Data directories](datadirs.md)** — where satkit looks for files and where
  it writes them, the environment variables, provisioning a machine up front,
  the optional `satkit-data` bundle, and choosing a JPL ephemeris.
- **[Downloads and refresh](datadownloads.md)** — where each file comes from,
  how downloads are hash-verified, the refresh cadence for the two daily
  tables, TLS-inspecting proxies, and what happens when a fetch fails.
- **[Data coverage](datacoverage.md)** — whether the Earth-orientation and
  space-weather tables actually cover the epoch you are propagating over, and
  what satkit does when they do not.
