# Changelog


## 0.10.4 - 2026-02-26

### Data directory fix
- Fix data directory resolution for `satkit_data` pip package: the `data/` subdirectory was not being found because the path was missing the `/data` suffix and used the wrong parent level relative to the dylib
- Python `__init__.py` now uses Python's import system to locate `satkit_data` package and set the data directory, rather than relying solely on Rust-side path heuristics
- Update documentation to reflect correct data directory search order


## 0.10.3

  - Clean up LICENSE file so GitHub correctly detects MIT license
  - Point homepage and documentation URLs to satkit.dev
  - Add CNAME for GitHub Pages custom domain (satkit.dev)

## 0.10.1

  - Fix PyO3 0.28 deprecation warnings: add `from_py_object` to all `#[pyclass]` types deriving `Clone`
  - Use SPDX `license = "MIT"` in Cargo.toml and pyproject.toml (fixes crates.io license detection)
  - Fix broken release badge in README (wheels.yml was renamed to release.yml)
  - Rewrite README for clarity and conciseness
  - Fix `release.toml` regex so `cargo release` correctly syncs all version files

## 0.10.0

Starting with this release, Rust and Python versions are tracked with identical version numbers.

  - Add 20 unit tests for core physics modules: frame transforms, point gravity, atmospheric drag, ITRFCoord, Kepler elements, and earth gravity
  - Test count increases from 79 to 99 library tests

## Python 0.9.4
  - ensure python packages are built in release mode (they have been, but adding additoinal flags)
  - Restructure into Cargo workspace with separate Python bindings crate
  - Fix and modernize Python `.pyi` stub files: explicit typed signatures for `time`, `duration`, `itrfcoord`, `quaternion`, `kepler`, `propsettings`, and `propagate` (replaces `*args`/`**kwargs`)
  - Migrate documentation from Sphinx/Read the Docs to MkDocs + Material theme + GitHub Pages
  - Replace `pytz` with stdlib `zoneinfo` in sunrise/sunset tutorial
  - Add `cargo-release` config to sync `pyproject.toml` version from `Cargo.toml`
  - Drop Python 3.8 and 3.9 support; minimum is now Python 3.10
  - Add explanatory markdown to all Jupyter notebook tutorials
  - Fix MathJax rendering in notebook tutorials
  - Remove ReadTheDocs workflow; docs now deployed via GitHub Pages
  - Streamline CI: merge `wheels.yml` and `cargo-publish.yml` into single `release.yml` on `v*` tags
  - Add Python bindings test job to `build.yml` (runs on every push)
  - Move cibuildwheel testing out of `pyproject.toml` into dedicated CI job
  - Update PyO3 to 0.28.2 and numpy to 0.28

## Rust 0.9.4
  - Add prelude to expose commonly-used structs and methods
  - OMM no-longer exposed at top crate level (but is expose via prelude)
  - Make `epoch_instant` function public in OMM
  - OMM supports import of xml files with `omm-xml` feature (enabled by default)

## Python 0.9.3
  - bugfix: Fixed transposed state transition matrix in python bindings for high-precision propagator

## Rust 0.9.3, Python 0.9.2
  - bugfix: Fix (and document in python) the `to_enu` and `to_ned` functions in ITRFCoord

## Rust 0.9.2, Python 0.9.1
  - bugfix: handle high-precision propagation without error when duration is zero
  - Fix Rust documentation for SGP4 to accurately represent sgp4 source
  - Add a "zero" static function for duration to represent zero duration

## Rust 0.9.1
  - Functions that accept time as input now accept structs that implement the new `TimeLike` trait.
  - Add a `chrono` feature that enables interoperability with the chrono crate.  implement `TimeLike` for `chrono::DateTime`

## Rust 0.9.0, Python 0.9.0
  - Support Orbital Mean-Element Messages (OMM) in JSON format
  - Add OMM documentation, tests, and python example
  - Structured output of SGP4 propagator in rust
  - add "as_datetime" function in python for satkit.time (for consistent nomenclature)
  - Rename time-interval boundary nomenclature from `start/stop` to `begin/end` across Rust + Python APIs (including propagation functions and related settings/results).

-----------------

## Rust 0.8.4, Python 0.8.5
  - Add "x", "y", "z", "w" property getters to quaternion in python
  - Add pickle serialize/deserialize tests in python testing
  - allow for duration division by float and by other duration in python

## Rust 0.8.3, Python 0.8.4
  - Fix small python typing bugs to allow setting of properties without errors
  - add additional option for RK integrator
  - Add functions for phase of moon & fraction of moon illuminated
  - typo corrections in code

## Python 0.8.3
  - Use pyo3 0.27.1 (transparent to user but annoying API updates)
  - add typing for property setters that were left out of python bindings (e.g., propsettings)
  - Allow user to set values for kepler object following object creation

## Rust 0.8.2
  - Cleanup of referencing of nalgebra types: commonly used aliases are now all referenced from mathtypes.rs
  - Improve documentation

## Rust 0.8.1
  - Remove un-used dependencies

## Python 0.8.2

### Allow TLE setting of parameters
  - Allow setting of TLE parameters in python bindings

### Support for Python 3.14
  - Build wheels for Python 3.14


## Python 0.8.1

### Fix _version.py
  - export "version" and "_version"

## Python 0.8.0.  Rust 0.8.0

### TLE export to lines
  - functions for generating 69-character TLE lines from a TLE object

### Python use of "anyhow::Result"
  - Use "anyhow::Result" where appropriate in python bindings, as it is used in main code branch and compatible with pyo3

### Code Cleanup
  - Remove "clippy" warnings, mainly in comment indentation and a few inefficient code idioms

### Error Checking on from_datetime
  - Bounds checking on month, day, hour, minute, second

### TLE Fitting
  - Add functionality to generate TLE from high-precision state vectors

### Day of Year
  - Add functionality to compute 1-based day of year in "Instant" structure

## Python 0.7.4 Rust 0.7.4

### Separate Rust and python versions
 - Rust and python versions will now evolve independently

### Fix EOP-induced crashes in frame transform
 - Earth orientation parameters query returns None for invalid dates, which caused frame transform functions that rely on it to panic.  Instead, print warning to stderr in query when out of bounds, and for frame transforms set all values to zero when out of bounds.  Also add function to disable the warning from being shown even once

### Add workflow_dispatch trigger to GitHub Actions
 - Suggestion & contribution by "DeflateAwning"

### Docs cleanup and fix path for download of python scripts
 - Contribution by "DeflateAwning"

### No panic if TLE lines are too short
 - Return error rather than panic if not enough characters in TLE
 - Contribution by "DeflateAwning"


## 0.7.3 - 2025-07-30

### Python data file warning to stderr
 - Print warning to stderr in python bindings if importing with missing datafiles

## 0.7.1 - 2025-07-22
## 0.7.2 - 2025-07-22

### Quaternion python bindings
 - Add ability to create quaternion from (w,x,y,z) values in python bindings
 - Did not merge correctly, so upped version twice.

## 0.7.0 - 2025-07-17

### satkit-data package for python bindings
- For python bindings, necessary data files are now included in a separate package, ``satkit-data``, which is a dependency of ``satkit``


## 0.6.2 - 2025-07-14

### Jupyter notebook example fixes
- Fix jupyter notebook examples to work with time and duration casting as properties


## 0.6.1 - 2025-07-12

### Python comparison operators
- Add complete set of comparison operators for python bindings of time and duration

## 0.5.9 - 2025-07-10

### Python typing fixes
- Python typing fixes, too many to enumerate

### Python duration casting
- Python binding duration casting (e.g., seconds, minutes, hours, days) are now properties, not function calls, e.g. d.seconds instead of d.seconds()


## 0.5.8 - 2025-07-07

### Python fixes
- Fix issue with quaternion rotation on multiple vectors failing if memory is non-contiguous
- Fix indexing error with SGP4 when propagating with multiple TLEs

### Clippy Warnings
- minor code changes to remove rust clippy warnings


## 0.5.7 - 2025-04-25

### Linux ARM
- Include Python binaries for Linux 64-bit ARM

## 0.5.6 - 2025-04-6

### Anyhow
- Use "anyhow" crate for error handling, both in core code and in python bindings (it is very nice!)

## 0.5.5 - 2025-01-27

### Low-Precision Ephemeris
 - Coefficients for low-precision planetary ephemerides did not match paper or JPL website referenced in documentation.  Not sure where original nu mbers came from.  Some plantes (e.g., Mercury) matched.  Others (e.g., Mars) did not, although numbers for Mars did match a google search. Very strange ... regardless, update so they match.  Coefficients for years 1800 to 2050 AD are correct and remain unchanged.
 - Add Pluto as a planet for low-precision ephemerides (it is included in reference paper, but not JPL website)

### Two-Line Element Sets
- Fix issue where two-line element sets confuse satellite name for a line number if satellite name starts with a "1" or a "2"
