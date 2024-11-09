# Time

The **satkit** package includes a `time` module that is able to represent and transform between times with various time scale, or epochs.

The `time` module duplicates much of the functionality of the native `datetime` package provided by python.  This separate module is necessary in the native rust implementaiton many low-level calculations, as these often require knowlege of time with various epochs (UTC, UT1, TT, GPS, etc..).  The native `datetime` package provides only **UTC** time.  

```{tip}
Many functions within the python package (e.g., frame transforms) require time as an input variable.  For *all* functions of time in the **satkit** package, for convenience, inputs can either either `satkit.time` object or `datetime.datetime` object.  For `datetime.datetime` objects, inputs are taken to be UTC.
```

```{warning}
Time is stored natively as a 64-bit floating point number representing the TAI modified Julian day. This should give you some sense of the precision.  Typically, microseconds can be achieved but different techniques are required for higher precision (e.g., nanoseconds)
```

## Time Scales

The `satkit.time` object can represent times with the following time scales, or epochs:

* `UTC`: Universal Time Coordinate
* ` TT`: Terrestrial Time
* `UT1`: UT1
* `TAI`: International Atomic Time
* `GPS`: Global Positioning System (GPS) time
* `TDB`: Barycentric Dynamical Time    

A description of the above time systems is at:
<https://gssc.esa.int/navipedia/index.php/Transformations_between_Time_Systems>


```{note}
**UT1** is a "universal time" linked to the Earth's rotation, which at small scales is stochastic.  It will be different than **UTC** by less than a second.  The difference between the two times, often denoted $\Delta UT1$, is part of the "Earth Orientation Parameters" file that must be periodically downloaded to update with new values.  This can be done with the `satkit.utils.update_datafiles()` function call, and requires access to the internet (it is proxy aware)
```




## Examples

### Object creation

```python
import satkit as sk
import datetime

# Test time creation functions

# Create from a string, attempting multiple formats
# (format below is iso8601)
t = sk.time('2020-03-01T18:20:30Z')
print(t)
# prints 2020-03-01T18:20:30Z
# Same thing, trying a different format (satkit tries to guess)
t = sk.time('2024/3/1 18:20:30')
print(t)
# prints 2024-03-01T18:20:30Z

# Create from a datetime object
dt = datetime.datetime(2020, 3, 1, 18, 20, 30, tzinfo=datetime.timezone.utc)
t = sk.time.from_datetime(dt)
print(t)
# prints 2020-03-01T18:20:30Z

# Create from a rfc3339 string (ISO 8601)
t = sk.time.from_rfctime('2020-03-01T18:20:30Z')
print(t)
# prints 2020-03-01T18:20:30Z
```

### Epochs

```python
import satkit as sk

# Create a time corresponding to April 9, 2024 at 12:00:00 UTC
instant = sk.time(2024, 4, 9, 12, 0, 0, scale=sk.timescale.UTC)
# Create a 2nd time corresponding to April 9, 2024 at 12:00:00 TAI
# This is slightly different than the previous time
instant2 = sk.time(2024, 4, 9, 12, 0, 0, scale=sk.timescale.TAI)

# Print the UTC time
print(instant)
# Print the TAI time
print(instant2)

# Take the difference between the 2
# This should be 37 seconds since there have been 37 leap seconds since 1972
# and TAI is monotonically increasing
duration = instant - instant2
print(duration)

# This will print:
# 2024-04-09 12:00:00.000Z
# 2024-04-09 11:59:23.000Z
# Duration: 37.000 seconds
```

### Julian Dates (and modified)
```python
# Create an object representing an instant in time
instant = sk.time(2024, 4, 9, 12, 0, 0, scale=sk.timescale.UTC)

# Convert to Julian date (default epoch is UTC)
jd = instant.as_jd()
print(jd)

# Convert to a modified julian date (default epoch is UTC)
mjd = instant.as_mjd()
print(mjd)

# Convert to a modified Julian date with Terrestrial Time as the epoch
mjd_tt = instant.as_mjd(sk.timescale.TT)
print(mjd_tt)
# This will print:
# 2460410.0
# 60409.5
# 60409.50080074074
```

### Durations
```python
# Create an instant
instant = sk.time(2024, 4, 9, 12, 0, 0)

# Add an hour to the instant
# Valid keywords are "hours", "minutes", "seconds", "days"
instant2 = instant + sk.duration(hours=1)


print(instant)
print(instant2)

# Take the differences between the two instants
# This should be 1 hour
dur = instant2 - instant
print(dur)
# Some rounding errors may occur
print(dur.seconds())
print(dur.hours())

# This outputs:
# 2024-04-09 12:00:00.000Z
# 2024-04-09 13:00:00.000Z
# Duration: 1 hours, 0 minutes, 0.000 seconds
# 3599.9999997904524
# 0.9999999999417923

```
