# Changelog

## 0.5.5 - 2025-01-27

### Low-Precision Ephemeris
 - Coefficients for low-precision planetary ephemerides did not match paper or JPL website referenced in documentation.  Not sure where original numbers came from.  Some plantes (e.g., Mercury) matched.  Others (e.g., Mars) did not, although numbers for Mars did match a google search. Very strange ... regardless, update so they match.  Coefficients for years 1800 to 2050 AD are correct and remain unchanged.
 - Add Pluto as a planet for low-precision ephemerides (it is included in reference paper, but not JPL website)

### Two-Line Element Sets
- Fix issue where two-line element sets confuse satellite name for a line number if satellite name starts with a "1" or a "2"

## 0.5.6 - 2025-04-6

### Anyhow
- Use "anyhow" crate for error handling, both in core code and in python bindings (it is very nice!)

## 0.5.7 - 2025-04-25

### Linux ARM
- Include Python binaries for Linux 64-bit ARM

## 0.5.8 - 2025-07-07

### Python fixes
- Fix issue with quaternion rotation on multiple vectors failing if memory is non-contiguous
- Fix indexing error with SGP4 when propagating with multiple TLEs

### Clippy Warnings
- minor code changes to remove rust clippy warnings


## 0.5.9 - 2025-07-10

### Python typing fixes
- Python typing fixes, too many to enumerate

### Python duration casting
- Python binding duration casting (e.g., seconds, minutes, hours, days) are now properties, not function calls, e.g. d.seconds instead of d.seconds()


## 0.6.1 - 2025-07-12

### Python comparison operators
- Add complete set of comparison operators for python bindings of time and duration

## 0.6.2 - 2025-07-14

### Jupyter notebook example fixes
- Fix jupyter notbook examples to work with time and duration casting as properties


## 0.7.0 - 2025-07-17

### satkit-data package for python bindings
- For python bindings, necessary data files are now included in a separate package, ``satkit-data``, which is a dependency of ``satkit``


## 0.7.1 - 2025-07-22
## 0.7.2 - 2025-07-22

### Quaternion python bindings
 - Add ability to create quaternion from (w,x,y,z) values in python bindings
 - Did not merge correctly, so upped version twice.

## 0.7.3 - 2025-07-30

### Python data file warning to stderr
 - Print warning to stderr in python bindings if importing with missing datafiles

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

## Python 0.8.1

### Fix _version.py
  - export "version" and "_version"

## Python 0.8.2

### Allow TLE setting of parameters
  - Allow setting of TLE parameters in python bindings

### Support for Python 3.14
  - Build wheels for Python 3.14


## Rust 0.8.1
  - Remove un-used dependencies
  
## Rust 0.8.2
  - Cleanup of referencing of nalgebra types: commonly used aliases are now all referenced from mathtypes.rs
  - Improve documentation


## Python 0.8.3
  - Use pyo3 0.27.1 (transparent to user but annoying API updates)
  - add typing for property setters that were left out of python bindings (e.g., propsettings)
  - Allow user to set values for kepler object following object creation

## Rust 0.8.3, Python 0.8.4
  - Fix small python typing bugs to allow setting of properties without errors
  - add additional option for RK integrator
  - Add functions for phase of moon & fraction of moon illuminated
  - typo corrections in code
