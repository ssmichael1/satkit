# SGP4, TLEs and OMMs

The [TLEs, SGP4 & OMMs](../guide/tle.md) guide covers the model, the element-set formats and the gravity-model and ops-mode choices; this page shows the Rust API.

## Loading element sets

- `TLE::load_3line(name, line1, line2)` and `TLE::load_2line(line1, line2)` parse one element set.
- `TLE::records(lines)` iterates over a catalog (any iterator of `&str` or `String`, such as `text.lines()`), yielding a `Result<TLE, tle::Error>` per record, so a malformed record can be skipped or reported with its line number; `.check_checksums(true)` also verifies column 69. `TLE::from_lines(&[String])` collects it and stops at the first bad record.
- `TLE::from_url` fetches a catalog (`download` feature). CelesTrak rate-limits repeated requests; cache the text rather than fetching it on every run.
- `satkit::omm::OMM` is a CCSDS Orbit Mean-Elements Message: `OMM::from_json_string`, `from_xml_string` (`omm-xml` feature), `from_text` / `from_file` (format detected from the content) and `from_url`. `OMM::from_tle` and `OMM::to_tle` convert, and `OMM` serializes with `serde`.

## Propagating

`sgp4::sgp4(&mut source, &times)` runs SGP4 for a slice of times on anything implementing `SGP4Source` (`TLE` and `OMM`), with the WGS72 gravity model and AFSPC ops mode the published element sets are fitted with; `sgp4_full` takes the gravity model and ops mode. It returns an `SGP4State` holding TEME position (m) and velocity (m/s) as 3×N matrices, one column per time, and an `SGP4Error` code per time (e.g. a decayed orbit). The source is `&mut` because the SGP4 initialization is cached in it; editing an element re-initializes automatically.

SGP4 output is in the TEME frame: rotate it to ITRF with `frametransform::qteme2itrf` (e.g. for geodetic coordinates) or to GCRF with `frametransform::rotation(Frame::TEME, Frame::GCRF, &t)`.

## Fitting

`TLE::fit_from_states(&states, &times, epoch)` fits a TLE (Levenberg–Marquardt, SGP4 as the model) to GCRF position/velocity samples given as `[f64; 6]` arrays, typically from a precise propagation or GNSS, and returns the TLE with a `TleFitResult` (status, residual norms, iterations). The fitted TLE carries the orbital elements and B\*; copy the catalog fields (name, number, designator) over yourself. `fit_from_states_full` takes the gravity model and ops mode.

## Example

The catalog and the OMM are inline, so this runs without the network.

```rust
--8<-- "examples/sgp4_tle.rs"
```

```text
$ cargo run --example sgp4_tle
ISS (ZARYA)      #25544  epoch 2024-01-01T12:00:00.000000Z  incl 51.64 deg
SHINSEI (MS-F2)  #05485  epoch 2024-11-19T10:29:41.764416Z  incl 32.06 deg

time (UTC)                   lat (deg)  lon (deg)  alt (km)
2024-01-01T12:00:00.000000Z     50.680    176.922     421.4
2024-01-01T12:15:00.000000Z     16.846   -126.773     415.5
...

passes above 10 deg elevation over the next day:
  2024-01-02 00:09:30 to 00:15:30, max 77.5 deg at 00:12:30
  2024-01-02 01:47:00 to 01:52:00, max 23.5 deg at 01:49:30
...

OMM vs TLE at epoch: 0.000 m apart

fit: Converged (step size tolerance) after 10 iterations, residual norm 0.000 m
  mean motion 15.48915330 rev/day (was 15.48915330), bstar 1.0270e-4 (was 1.0270e-4)
  1 25544U 98067A   24001.50000000  .00000000  00000-0  10270-3 0    08
  2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330    02
```

## See also

- [TLEs, SGP4 & OMMs](../guide/tle.md), [Two-Line Element Set](../tutorials/Two-Line%20Element%20Set.ipynb) and [TLE Fitting](../tutorials/TLE%20Fitting.ipynb).
- [`TLE`](https://docs.rs/satkit/latest/satkit/tle/struct.TLE.html), [`sgp4`](https://docs.rs/satkit/latest/satkit/sgp4/) and [`OMM`](https://docs.rs/satkit/latest/satkit/omm/struct.OMM.html) on docs.rs.
