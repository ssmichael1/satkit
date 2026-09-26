# Installation

The `satkit` package is hosted at <https://github.com/ssmichael1/satkit/>. The package is written natively in Rust, with Python bindings provided by the [PyO3](https://pyo3.rs/) Rust package.

## PIP

Pre-built Python binary packages are provided by the [PyPI](https://pypi.org) package manager, and are the simplest to install. Binary packages are provided for Linux (x86_64 and aarch64), Windows (x86_64), and macOS on Apple silicon (arm64). Intel-based Macs are not shipped a wheel since 0.22: install from source with `pip install --no-binary satkit satkit` (needs a stable Rust toolchain) or via conda-forge, which builds `osx-64`. To install via PyPI:

```bash
python -m pip install satkit
```

That is the whole install (~10 MB). The core data — the IERS nutation tables and the gravity models to degree 70 — is compiled into the package, so frame transforms, gravity, SGP4, time scales, Keplerian propagation and Lambert targeting work immediately, with no data directory and no network.

Two things are fetched later, on demand, into the platform user-data directory (`satkit.utils.datadir()`, never inside `site-packages`):

- **The JPL ephemeris** (DE440, 102 MB, SHA-256 verified), the first time it is needed — the first `propagate()` or `satkit.jplephem` query. The `sun`, `moon` and `planets` modules are analytic low-precision models and never load it.
- **Earth orientation and space weather**, fetched on first use and refreshed by `satkit.utils.update_datafiles()`; they change daily, so re-run that periodically.

See [Data Files](datafiles.md) for what each file is and [Downloads and Refresh](datadownloads.md) for where it comes from.

### Offline and air-gapped use

Run `satkit.utils.update_datafiles()` once to provision everything up front (a Docker image, a CI job, a machine that will later be offline); the steps for copying the result to an air-gapped machine are in [Provisioning up front](datadirs.md#provisioning-up-front). `pip install satkit[data]` installs the optional [`satkit-data` bundle](datadirs.md#the-optional-satkit-data-bundle) instead. The environment variables that control the data directory, mirrors, offline mode and TLS (`SATKIT_DATA`, `SATKIT_DATA_URL`, `SATKIT_OFFLINE`, `SATKIT_CA_BUNDLE`, …) are listed in [Data Directories](datadirs.md#environment-variables-and-api), with the full search order per platform.

## Conda

The same package is on [conda-forge](https://anaconda.org/conda-forge/satkit), built from the PyPI source distribution by [conda-forge/satkit-feedstock](https://github.com/conda-forge/satkit-feedstock):

```bash
conda install -c conda-forge satkit
```

It behaves exactly like the wheel: the core data is compiled in, the JPL ephemeris is downloaded on first use, and the same environment variables apply. Intel Macs get a native `osx-64` build here.

## Build from Source

The package can be downloaded and installed from the Rust source directly. The "pybindings" feature must be enabled. The simplest way to do this is to manually build the wheel package:

```bash
git clone https://github.com/ssmichael1/satkit
cd satkit
mkdir wheel
cd wheel
python -m pip wheel ..
```

## Something not working?

See [Troubleshooting & FAQ](troubleshooting.md) for pip falling back to a source build, data warnings, and download failures.
