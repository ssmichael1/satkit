# Space Weather

Daily geomagnetic ($K_p$, $a_p$) and solar-flux ($F_{10.7}$) indices — the
inputs NRLMSISE-00 consumes when `use_spaceweather` is enabled. The table is
assembled from three primary sources: the GFZ Potsdam observed record (CC BY
4.0), the NOAA/SWPC 45-day forecast and the NASA MSFC MSAFE monthly forecast
(both public domain).

Only the first block is measurement. Use
[`coverage`][satkit.spaceweather.coverage] and
[`status`][satkit.spaceweather.status] to find out where an epoch falls before
relying on the result; see
[Data coverage](../getting-started/datacoverage.md#space-weather-coverage) for
what each regime does and does not know. [`init_from_path`][satkit.spaceweather.init_from_path]
loads a file of your own, including CelesTrak's `SW-All.csv`.

::: satkit.spaceweather
