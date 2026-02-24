# Installation

The `satkit` package is hosted at <https://github.com/ssmichael1/satkit/>. The package is written natively in Rust, with Python bindings provided by the [PyO3](https://pyo3.rs/) Rust package.

## PIP

Pre-built Python binary packages are provided by the [PyPI](https://pypi.org) package manager, and are the simplest to install. Binary packages are provided for 64-bit x86 platforms running Windows, Linux, and macOS, as well as macOS systems on the ARM platform. To install via PyPI:

```bash
python -m pip install satkit
```

## Build from Source

The package can be downloaded and installed from the Rust source directly. The "pybindings" feature must be enabled. The simplest way to do this is to manually build the wheel package:

```bash
git clone https://github.com/ssmichael1/satkit
cd satkit
mkdir wheel
cd wheel
python -m pip wheel ..
```
