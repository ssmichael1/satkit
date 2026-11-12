# Earth Orientation Parameters

The Earth Orientation Parameters (EOP) — UT1 − UTC, polar motion, the excess
length of day and the celestial pole offsets — come from the IERS
`finals2000A.all` table (Bulletin A): observed values from 1973, then about a
year of predictions. They feed UT1 and every ITRF frame transform.

`satkit.eop` mirrors [`satkit.spaceweather`](spaceweather.md): use
[`coverage`][satkit.eop.coverage] and [`status`][satkit.eop.status] to find
out where an epoch falls before relying on the result (see
[Data coverage](../getting-started/datacoverage.md#eop-coverage)), and
[`init_from_path`][satkit.eop.init_from_path] to pin satkit to a
`finals2000A.all` file of your own.

```python
import satkit as sk

t = sk.time(2024, 1, 1)
print(sk.eop.status(t))          # "observed"
dut1, xp, yp, lod, dx, dy = sk.eop.get(t)
first, last_observed, last = sk.eop.coverage()
```

The older names in [`satkit.frametransform`](frametransform.md)
(`earth_orientation_params`, `eop_coverage`, `eop_status`) remain and return
the same values; `frametransform.disable_eop_time_warning` silences the
one-time EOP warnings.

::: satkit.eop
