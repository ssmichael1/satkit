# Space Weather

Daily geomagnetic ($K_p$, $a_p$) and solar-flux ($F_{10.7}$) indices — the
inputs NRLMSISE-00 consumes when `use_spaceweather` is enabled. The table is
CelesTrak's `SW-All.csv`, which merges GFZ Potsdam (geomagnetic), DRAO /
Natural Resources Canada ($F_{10.7}$) and NOAA/SWPC (forecasts).

Only part of that table is measurement. Past the NOAA/SWPC 45-day forecast the
rows go monthly and carry **no geomagnetic data at all**, so NRLMSISE-00 falls
back to a quiet-time $A_p = 4$ — during a storm that is wrong by up to a factor
of two in density. Use [`coverage`][satkit.spaceweather.coverage] and
[`status`][satkit.spaceweather.status] to find out where an epoch falls before
relying on the result, and see
[Data coverage](../getting-started/datacoverage.md#space-weather-coverage) for
the measured numbers.

::: satkit.spaceweather
