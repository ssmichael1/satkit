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

## Python 0.7.5 Rust 0.7.5
  ### Export TLE object as lines
  - Generatetwo 69-character lines of element set from TLE object

  ### Comparison operators for Instant and Duration
  - Implement Ord (previously only PartialOrd was implemented)