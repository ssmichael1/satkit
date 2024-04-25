# Installation

The `satkit` package is hosted at https://github.com/ssmichael1/satkit/.  The package is written natively in rust, with python bindings provided by the [pyo3](https://pyo3.rs/) rust package.

## PIP

Pre-build python binary packages are provided by the [PyPi](https://pypi.org) package manager, and are the simplest to install.  Binary packages are provided for the 64-bit x86 platforms running Windows, Linux, and MacOS, as well as MacOS systems on the ARM platgform. To install via PyPi:

```python
python -m pip install satkit
```

## Build from Source

The package can be downloaded and installed from the rust source directly.  The "pybindings" feature must be enabled.  The simplest way to do this is to manually build the wheel package:

```bash
git clone https://github.com/ssmichael1/satkit
cd satkit
mkdir wheel
cd wheel
python -m pip wheel ..
```