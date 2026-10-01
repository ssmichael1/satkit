# Rust Quick Start

satkit is a Rust library first; the Python package is a binding over it. This section covers using the crate directly. Everything in [Learn](../tutorials/index.md) applies to both languages, since the models and conventions are the same; the pages here show the Rust API for each topic, with complete programs you can run.

## Add the crate

```bash
cargo add satkit
```

or in `Cargo.toml`:

```toml
[dependencies]
satkit = "0.24"
```

satkit builds on stable Rust. Its linear-algebra types (`Vector3`, `Matrix3`, `Quaternion`, ...) come from the [numeris](https://crates.io/crates/numeris) crate and are re-exported in [`satkit::mathtypes`](https://docs.rs/satkit/latest/satkit/mathtypes/); build vectors with `Vector3::from_array([x, y, z])`, or add `numeris` yourself for its `vector!` macro.

## Cargo features

| Feature | Default | What it does |
|---|---|---|
| `download` | on | HTTP(S) downloads through `ureq`: fetching data files on first use, `utils::update_datafiles`, `TLE::from_url` and `OMM::from_url`. Without it, data files must already be on disk ([Provisioning up front](../getting-started/datadirs.md#provisioning-up-front)) and the URL loaders are not compiled |
| `omm-xml` | on | Parsing OMM XML (`OMM::from_xml_string`, `from_xml_file`, and XML in `OMM::from_text` / `from_file`) through `quick-xml`. OMM JSON needs no feature |
| `chrono` | off | Interoperability with [chrono](https://crates.io/crates/chrono): `From` conversions between `Instant` and `chrono::DateTime`, and `TimeLike` for `chrono::DateTime`, so chrono times can be passed to every satkit function that takes a time |
| `hifitime` | off | Interoperability with [hifitime](https://crates.io/crates/hifitime): `From` conversions between `Instant` and `hifitime::Epoch` (through TAI, so leap seconds are kept), and `TimeLike` for `hifitime::Epoch` |

To leave out the downloader, for a build without an HTTP client:

```toml
satkit = { version = "0.24", default-features = false, features = ["omm-xml"] }
```

## Data files

The IERS nutation tables and the gravity models are compiled in, so time scales, SGP4, Kepler and Lambert, gravity, and the precession–nutation part of the frame chain work with no files and no network. The JPL ephemeris (for the Sun and Moon in the numerical propagator, and planetary positions) is downloaded on first use; the Earth orientation and space-weather tables are downloaded on first use and refreshed by [`utils::update_datafiles`](https://docs.rs/satkit/latest/satkit/utils/update_data/fn.update_datafiles.html):

```rust
// Download anything missing and refresh the Earth orientation and
// space-weather tables (needs the `download` feature)
satkit::utils::update_datafiles(None, false)?;
```

Frame transforms to or from ITRF use Earth orientation parameters (EOP). Without the EOP file they warn once and fall back to zero polar motion and UT1 = UTC, which is off by up to about 12″ (hundreds of meters at LEO), and `propagate` returns `orbitprop::Error::EopUnavailable`. The data pages apply unchanged to Rust:

- [Data Files](../getting-started/datafiles.md): what is compiled in, downloaded and refreshed.
- [Data Directories](../getting-started/datadirs.md): where satkit looks and writes, `SATKIT_DATA`, `SATKIT_OFFLINE`, and provisioning a machine up front (`utils::set_datadir`, `utils::set_offline`).
- [Data Coverage](../getting-started/datacoverage.md): checking that the tables cover your epochs (`earth_orientation_params::status`, `spaceweather::status`).

## Warnings and logging

satkit reports stale or missing data, fallbacks and download notices through the [`log`](https://docs.rs/log) facade, with the emitting module as the target (e.g. `satkit::earth_orientation_params`). With no logger installed they are printed to stderr. Once you install a logger it decides what is shown. [`env_logger`](https://docs.rs/env_logger) shows only errors unless `RUST_LOG` says otherwise, which would hide satkit's warnings, so give it a default:

```rust
fn main() {
    // Show satkit's warnings unless RUST_LOG says otherwise
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("satkit=warn"))
        .init();
    // ...
}
```

and filter per module at run time, e.g. `RUST_LOG=satkit=error` or `RUST_LOG=satkit=warn,satkit::spaceweather=error`. The EOP and space-weather warnings can also be turned off at the source with `earth_orientation_params::disable_eop_time_warning()` and `spaceweather::disable_space_weather_time_warning()`. See [Warnings and logging](../getting-started/troubleshooting.md#warnings-and-logging-how-do-i-silence-or-redirect-these-warnings) for what each warning means.

## Errors

Fallible functions return a `Result` with the error type of their module, each an enum built with `thiserror` so it can be matched:

| Module | Error type | Typical causes |
|---|---|---|
| time | `InstantError` | out-of-range date fields, unparsable strings, invalid leap second |
| `frametransform` | `frametransform::Error` | frame pair not supported by the chosen function |
| `tle`, `omm` | `tle::Error`, `omm::Error` | malformed records (with line number and satellite), checksum mismatch, HTTP errors |
| `sgp4` | `sgp4::Error` | initialization failure (eccentricity, decayed orbit); per-time failures are in `SGP4State::errcode` |
| `orbitprop` | `orbitprop::Error` | no EOP data, invalid settings, a failed ephemeris lookup |
| `jplephem` | `jplephem::Error` | ephemeris file missing (offline) or invalid, time outside its span |
| `kepler`, `lambert`, `lpephem` | `kepler::Error`, `lambert::Error`, `lpephem::Error` | out-of-domain elements, no convergence, no sunrise |
| data | `earth_orientation_params::Error`, `spaceweather::Error`, `utils::download::Error` | unreadable or unwritable data files, failed downloads |

Some are `#[non_exhaustive]`, so a `match` on them needs a wildcard arm. All implement `std::error::Error + Send + Sync`, so in an application `Box<dyn std::error::Error>` (as in the examples) or `anyhow::Result` collects them with `?`. The crate-level `satkit::Error` is deprecated; use the module errors.

## A first program

```rust
--8<-- "examples/quickstart.rs"
```

```text
$ cargo run --example quickstart
ISS (ZARYA) at 2024-01-01T12:46:00.000000Z
  latitude -51.019 deg, longitude -17.397 deg, altitude 441.6 km
```

## Examples

Every program on these pages is a file in [`examples/`](https://github.com/ssmichael1/satkit/tree/main/examples) of the repository, compiled in CI, and runnable from a checkout with `cargo run --example <name>`:

| Page | Example | Needs |
|---|---|---|
| this page | `quickstart` | EOP for the TEME → ITRF rotation |
| [Time and Time Scales](time.md) | `time_scales`, `chrono_interop`, `hifitime_interop` | EOP for UT1 only; `chrono_interop` and `hifitime_interop` need `--features chrono` / `--features hifitime` |
| [Coordinate Frames](frames.md) | `frames` | EOP for exact ITRF rotations |
| [SGP4, TLEs and OMMs](sgp4.md) | `sgp4_tle` | EOP for the TEME → ITRF / GCRF rotations |
| [Numerical Propagation](propagation.md) | `propagate_leo` | JPL ephemeris, EOP, space weather |
| [Sun, Moon and Ephemerides](ephemerides.md) | `sun_moon`, `ephemeris` | nothing; JPL ephemeris for `ephemeris` |
| [Kepler and Lambert](kepler-lambert.md) | `kepler_lambert` | nothing |

"EOP" and "space weather" are fetched on first use when the `download` feature is on; the JPL ephemeris is a one-time 102 MB download. The [API reference](api.md) on docs.rs documents every type and function.
