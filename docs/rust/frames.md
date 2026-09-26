# Coordinate Frames

Rotations between frames are unit quaternions (`satkit::Quaternion`); applying one to a `Vector3` with `*` rotates the vector into the target frame. The [Coordinate Frames](../tutorials/Coordinate%20Frames.ipynb) and [Geodetic Coordinates](../tutorials/Geodetic%20Coordinates.ipynb) tutorials explain the frames and the IERS 2010 reduction; this page shows the Rust API.

## Rotations: `frametransform`

The `Frame` enum names the frames: `ITRF`, `TIRS`, `CIRS`, `GCRF`, `EME2000`, `ICRF`, `TEME`, and the orbit-local `RTN`, `NTW` and `LVLH`. Three functions in [`frametransform`](https://docs.rs/satkit/latest/satkit/frametransform/) cover every pair:

| Function | Frames | Returns |
|---|---|---|
| `rotation(from, to, &t)` | any pair of Earth / inertial frames, full IERS 2010 reduction | `Result<Quaternion>` |
| `rotation_approx(from, to, &t)` | ITRF and the inertial frames, the cheaper ~1″ reduction | `Result<Quaternion>` |
| `rotation_with_state(from, to, &t, &pos, &vel)` | any pair, including RTN / NTW / LVLH, whose axes come from a GCRF position and velocity | `Result<Quaternion>` |
| `transform_state(from, to, &t, &pos, &vel)` | position and velocity, including the Earth-rotation term ω × r between rotating and inertial frames | `Result<(Vector3, Vector3)>` |

The fixed-pair functions (`qitrf2gcrf`, `qgcrf2itrf`, `qteme2itrf`, `qteme2gcrf`, ...) return the quaternion directly, with no `Result`; `qteme2itrf` is the one SGP4 output needs. `to_gcrf(frame, &pos, &vel)` / `from_gcrf` give the RTN / NTW / LVLH rotation as a 3×3 matrix, and `gcrf_to_rtn` and friends name each one.

The rotations to and from ITRF read the Earth orientation parameters (UT1 − UTC, polar motion, celestial-pole offsets) from `finals2000A.all`. [`earth_orientation_params::status(&t)`](https://docs.rs/satkit/latest/satkit/earth_orientation_params/fn.status.html) says whether an epoch is observed, predicted, extrapolated or not covered. Without the file the rotations warn once and use zeros, which is off by up to ~12″; run `satkit::utils::update_datafiles` to fetch or refresh it.

## Geodetic coordinates: `ITRFCoord`

`ITRFCoord` is an Earth-fixed position. Build it from geodetic latitude, longitude (degrees or radians) and height above the WGS84 ellipsoid, or from an ITRF `Vector3`; read back `latitude_deg()`, `longitude_deg()`, `hae()` or `to_geodetic()`. `to_enu(&origin)` / `to_ned(&origin)` give the vector from `origin` in its local East-North-Up / North-East-Down frame, and `geodesic_distance` the distance and headings along the ellipsoid (Vincenty).

## Example

```rust
--8<-- "examples/frames.rs"
```

```text
$ cargo run --example frames
EOP status: Observed
UT1-UTC = -0.0164 s, polar motion = (0.0545", 0.4695")
station: ITRFCoord(lat:  42.3601 deg, lon: -71.0589 deg, altitude:  0.02 km)
  ITRF = [1532143.8, -4464572.7, 4275257.9] m
  GCRF = [4611302.5, 1053168.8, 4264320.4] m
full vs approximate reduction: 0.453 arcsec
station inertial speed = 344.2 m/s at 6368.5 km
TEME x-axis point at 400 km: ITRFCoord(lat:  -0.0001 deg, lon: -84.2628 deg, altitude: 400.00 km)
to target: ENU = [-248.9, -178.6, -7.4] km, azimuth 234.34 deg
  geodesic distance 306.5 km, initial heading 234.34 deg
velocity in RTN = [0.0, 7071.1, -0.0] m/s
```

## See also

- [Coordinate Frames](../tutorials/Coordinate%20Frames.ipynb), [Quaternions](../tutorials/Quaternions.ipynb) and [Geodetic Coordinates](../tutorials/Geodetic%20Coordinates.ipynb).
- [Maneuver Coordinate Frames](../guide/maneuver_frames.md): the RTN, NTW and LVLH axis conventions.
- [Data Coverage](../getting-started/datacoverage.md#eop-coverage): what the EOP table covers.
