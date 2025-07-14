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

## 0.6.2 - 2025-07014

### Jupyter notebook example fixes
- Fix jupyter notbook examples to work with time and duration casting as properties
