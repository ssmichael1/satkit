# Installation

The `satkit` package is hosted at <https://github.com/ssmichael1/satkit/>. The package is written natively in Rust, with Python bindings provided by the [PyO3](https://pyo3.rs/) Rust package.

## PIP

Pre-built Python binary packages are provided by the [PyPI](https://pypi.org) package manager, and are the simplest to install. Binary packages are provided for 64-bit x86 platforms running Windows, Linux, and macOS, as well as macOS systems on the ARM platform. To install via PyPI:

```bash
python -m pip install satkit
```

That is the whole install (~10 MB). The core data — the IERS nutation tables and the gravity models to degree 70 — is compiled into the package, so frame transforms, gravity, SGP4, time scales, Keplerian propagation and Lambert targeting work immediately, with no data directory and no network.

Two things are fetched later, on demand:

- **The JPL ephemeris** (DE440, 102 MB) is downloaded the first time a planetary or lunar position is needed — the first `propagate()`, `jplephem` query or `sun`/`moon` call. The download is SHA-256 verified against the manifest compiled into satkit, and is written to the platform user-data directory (`satkit.utils.datadir()`), never inside `site-packages`. Set `SATKIT_JPLEPHEM_FILE=lnxp1900p2053.421` to use the 14 MB DE421 (1900–2053) instead.
- **Earth orientation and space weather** (`EOP-All.csv`, `SW-All.csv`) are fetched from CelesTrak on first use and refreshed by `satkit.utils.update_datafiles()`; they change daily, so re-run that periodically.

To provision everything up front (a Docker image, a CI job, a machine that will later be offline):

```python
import satkit as sk
sk.utils.update_datafiles()   # ephemeris (verified) + EOP/SW + solar-cycle forecast
```

### Offline and air-gapped use

- `SATKIT_OFFLINE=1` forbids all network access: anything that would need a download raises `RuntimeError` naming the missing file and its sources instead of connecting.
- `SATKIT_DATA_URL=https://mirror.example/satkit-data` makes satkit fetch from a mirror first (plain `http://` is accepted for an internal mirror; downloads are still verified).
- `pip install satkit[data]` installs the optional **`satkit-data`** bundle (the ephemeris and full-degree gravity files, ~110 MB) into `site-packages`; satkit finds it automatically as a read-only source. Use it where a first-use download is unwelcome.
- `SATKIT_DATA=/path` names a directory that is both searched first and written to.
- `SATKIT_CA_BUNDLE=/path/bundle.pem` verifies downloads against that PEM file instead of the system trust store — for a network whose TLS is inspected by a proxy whose CA is not installed system-wide, or a container with no trust store at all (`SATKIT_CA_BUNDLE=webpki` uses the roots compiled into satkit). See [Data Files](datafiles.md#downloads-behind-a-tls-inspecting-proxy).

See [Data Files](datafiles.md) for the full search order per platform.

## Build from Source

The package can be downloaded and installed from the Rust source directly. The "pybindings" feature must be enabled. The simplest way to do this is to manually build the wheel package:

```bash
git clone https://github.com/ssmichael1/satkit
cd satkit
mkdir wheel
cd wheel
python -m pip wheel ..
```
