# Orbit Ephemeris Messages (OEM)

API reference for the reader of CCSDS Orbit Ephemeris Messages: tables of position and velocity at a sequence of times, as exchanged between operators and agencies ([CCSDS 502.0-B-3](../guide/references.md#ccsds502)).

An OEM differs from the [OMM](tle.md) of the same standard: an OMM is one set of mean elements that needs SGP4 to give a state, while an OEM is the trajectory itself and is only interpolated.

- `OEM.from_file` and `OEM.from_text` read the KVN (text) and XML encodings, message versions 1.0 to 3.0.
- Positions and velocities are converted from the file's kilometers to **meters and m/s**; covariances to m², m²/s and m²/s².
- States stay in the frame the file names. `OEMSegment.frame` gives the matching `satkit.frame`, where there is one, for use with [`frametransform`](frametransform.md).
- `interp` uses the method and degree the file recommends (Lagrange or Hermite), or the ones passed to it, and never interpolates across segments.

```python
import satkit as sk

# An OEM normally comes from a file: sk.OEM.from_file("ISS.OEM_J2K_EPH.txt")
lines = [
    "CCSDS_OEM_VERS = 2.0",
    "CREATION_DATE = 2024-01-01T00:00:00",
    "ORIGINATOR = EXAMPLE",
    "META_START",
    "OBJECT_NAME = DEMOSAT",
    "OBJECT_ID = 2024-001A",
    "CENTER_NAME = EARTH",
    "REF_FRAME = EME2000",
    "TIME_SYSTEM = UTC",
    "START_TIME = 2024-01-01T00:00:00",
    "STOP_TIME = 2024-01-01T00:03:00",
    "INTERPOLATION = HERMITE",
    "INTERPOLATION_DEGREE = 7",
    "META_STOP",
    "2024-01-01T00:00:00 7000.000    0.000 0.0  0.000000 7.546053 0.0",
    "2024-01-01T00:01:00 6985.363  452.448 0.0 -0.487742 7.530274 0.0",
    "2024-01-01T00:02:00 6941.512  903.003 0.0 -0.973444 7.483003 0.0",
    "2024-01-01T00:03:00 6868.631 1349.782 0.0 -1.455075 7.404436 0.0",
]
oem = sk.OEM.from_text("\n".join(lines))

for seg in oem.segments:
    print(seg.object_name, seg.ref_frame, seg.start_time, seg.stop_time, len(seg))

seg = oem.segments[0]
print(seg.pos.shape, seg.vel.shape)  # (4, 3) arrays, meters and m/s

# Interpolate, with the method and degree the file recommends
times = [seg.start_time + sk.duration(seconds=s) for s in range(0, 180, 30)]
states = oem.interp(times)  # (6, 6): x, y, z (m), vx, vy, vz (m/s)

# EME2000 states to Earth-fixed
if seg.frame is not None and seg.center_name == "EARTH":
    pos_itrf, vel_itrf = sk.frametransform.transform_state(
        seg.frame, sk.frame.ITRF, times[0], states[0, :3], states[0, 3:]
    )
```

In Rust the reader is the [`satkit::oem`](https://docs.rs/satkit/latest/satkit/oem/) module, behind the `oem` Cargo feature (on by default).

::: satkit.OEM

::: satkit.OEMSegment

::: satkit.OEMCovariance
