use crate::pyinstant::{PyInstant, TimeArg};
use crate::pyitrfcoord::PyITRFCoord;
use crate::pyutils;
use pyo3::prelude::*;
use satkit::lpephem::moon;
use satkit::Instant;

/// `NotPrincipalPhase` is a bad argument; everything else (the JPL
/// ephemeris missing or out of range) is a runtime failure
fn moon_err(e: moon::Error) -> PyErr {
    match e {
        moon::Error::NotPrincipalPhase(_) => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    }
}

#[derive(PartialEq, Eq)]
/// Enum representing moon phases
///
/// Each value covers a range of the moon phase angle (see :func:`phase`),
/// given here in degrees.
#[pyclass(name = "moonphase", eq, eq_int)]
pub enum MoonPhase {
    NewMoon = moon::MoonPhase::NewMoon as isize,
    WaxingCrescent = moon::MoonPhase::WaxingCrescent as isize,
    FirstQuarter = moon::MoonPhase::FirstQuarter as isize,
    WaxingGibbous = moon::MoonPhase::WaxingGibbous as isize,
    FullMoon = moon::MoonPhase::FullMoon as isize,
    WaningGibbous = moon::MoonPhase::WaningGibbous as isize,
    LastQuarter = moon::MoonPhase::LastQuarter as isize,
    WaningCrescent = moon::MoonPhase::WaningCrescent as isize,
}

crate::enum_pickle!(MoonPhase, "moon.moonphase");

impl From<&MoonPhase> for moon::MoonPhase {
    fn from(p: &MoonPhase) -> Self {
        match p {
            MoonPhase::NewMoon => Self::NewMoon,
            MoonPhase::WaxingCrescent => Self::WaxingCrescent,
            MoonPhase::FirstQuarter => Self::FirstQuarter,
            MoonPhase::WaxingGibbous => Self::WaxingGibbous,
            MoonPhase::FullMoon => Self::FullMoon,
            MoonPhase::WaningGibbous => Self::WaningGibbous,
            MoonPhase::LastQuarter => Self::LastQuarter,
            MoonPhase::WaningCrescent => Self::WaningCrescent,
        }
    }
}

impl From<moon::MoonPhase> for MoonPhase {
    fn from(p: moon::MoonPhase) -> Self {
        match p {
            moon::MoonPhase::NewMoon => Self::NewMoon,
            moon::MoonPhase::WaxingCrescent => Self::WaxingCrescent,
            moon::MoonPhase::FirstQuarter => Self::FirstQuarter,
            moon::MoonPhase::WaxingGibbous => Self::WaxingGibbous,
            moon::MoonPhase::FullMoon => Self::FullMoon,
            moon::MoonPhase::WaningGibbous => Self::WaningGibbous,
            moon::MoonPhase::LastQuarter => Self::LastQuarter,
            moon::MoonPhase::WaningCrescent => Self::WaningCrescent,
        }
    }
}

/// Approximate Moon position in the GCRF Frame
///
/// Notes:
///   * Algorithm 31 from Vallado for the moon in Mean of Date (MOD), then rotated from MOD to GCRF via Equations 3-88 and 3-89 in Vallado
///   * Valid with accuracy of about 0.3 degree in ecliptic longitude (0.36 degree worst case against JPL DE440 over 1950-2100), 0.2 degree in ecliptic latitude, and 1275 km in range
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element numpy array or Nx3 numpy array representing moon position in GCRF frame at input time[s].  Units are meters
#[pyfunction]
pub fn pos_gcrf(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_result_arr(&|t| Ok(moon::pos_gcrf(t)), time)
}

/// Approximate Moon position in the Mean-of-Date (MOD) Frame
///
/// Notes:
///   * Algorithm 31 from Vallado for the moon in Mean of Date (MOD)
///   * `pos_gcrf(t) = frametransform.qmod2gcrf(t) * pos_mod(t)`
///   * Valid with accuracy of about 0.3 degree in ecliptic longitude (0.36 degree worst case against JPL DE440 over 1950-2100), 0.2 degree in ecliptic latitude, and 1275 km in range
///   * Useful when comparing with Vallado's worked examples, which are given in MOD coordinates
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute position
///
/// Returns:
///     numpy.ndarray: 3-element numpy array or Nx3 numpy array representing moon position in MOD frame at input time[s].  Units are meters
#[pyfunction]
pub fn pos_mod(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_vec3_of_time_result_arr(&|t| Ok(moon::pos_mod(t)), time)
}

/// Approximate Moon phase angle
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute phase
///
/// Returns:
///     float|numpy.ndarray: scalar or numpy array representing moon phase at input time[s].  Units are radians
#[pyfunction]
pub fn phase(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_func_of_time_arr(moon::phase, time)
}

/// Moon phase name
///
/// Args:
///     time (satkit.time|numpy.ndarray|list): time[s] at which to compute phase name
///
/// Returns:
///     str|list: phase name string or list of phase name strings (e.g., "New Moon", "Waxing Crescent", etc.)
#[pyfunction]
pub fn phase_name(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_func_of_time_arr(|t: &Instant| MoonPhase::from(moon::phase_name(t)), time)
}

/// Fraction of moon illuminated
///
/// Args:
///    time (satkit.time|numpy.ndarray|list): time[s] at which to
///
/// Returns:
///    float|numpy.ndarray: scalar or numpy array representing fraction of moon illuminated at input time[s].  Range is 0.0 to 1.0
#[pyfunction]
pub fn illumination(time: &Bound<'_, PyAny>) -> anyhow::Result<Py<PyAny>> {
    pyutils::py_func_of_time_arr(moon::illumination, time)
}

/// Moonrise and moonset times on a calendar date at the given location.
///
/// The input selects the UTC calendar date (its time of day is ignored);
/// the returned moonrise and moonset are those of that date at the
/// location's longitude, i.e. in the local mean day from
/// 0h UTC - longitude / 15 hours to 24 hours later, as UTC times.  At
/// far-west longitudes an event can fall on the next UTC date (and at
/// far-east longitudes on the previous one).  These are the same day
/// semantics as ``satkit.sun.rise_set``.
///
/// To get the events of a *local* date, pass a timezone-aware datetime at
/// local noon: for time zones UTC-11 to UTC+11 local noon falls on the same
/// UTC date, while local midnight falls on the previous UTC date east of
/// Greenwich.
///
/// The Moon rises about 50 minutes later each day, so about once a month
/// there is no moonrise on a date (and on another no moonset): that event
/// is None.  At high latitudes the Moon can stay up or down all day,
/// giving (None, None), and a day can hold two moonrises (or moonsets):
/// the first is returned.
///
/// Notes:
///     * Rise and set are when the Moon's upper limb touches a sea-level
///       horizon with 34 arcmin of refraction (the USNO / Astronomical
///       Almanac definition): the topocentric altitude of the Moon's centre
///       is -34' - asin(R_moon / d), with d the topocentric distance and
///       R_moon = 1737.4 km.  Parallax is exact; altitude is measured from
///       the plane normal to the geodetic vertical, with no dip correction
///       for an elevated site.
///     * The altitude is sampled every 10 minutes (and near-horizon extrema
///       checked for a grazing Moon), and events refined to 0.05 s.
///     * Accuracy against Skyfield (DE440s) over 2024, 35 S to 62 N: with
///       use_jpl, 0.1 s (1.6 s for a grazing Moon at 62 N); built-in,
///       52 s at the equator, 89 s at 35 deg, 128 s at 52 deg, a few
///       minutes at 62 deg.  Real refraction varies with the weather by a
///       minute or more of time.
///
/// Args:
///     time (satkit.time|datetime.datetime|numpy.datetime64): time whose UTC calendar date selects the day
///     coord (satkit.itrfcoord): location at which to compute moonrise and moonset
///     use_jpl (bool, optional): use the JPL ephemeris (apparent position) and the full IERS 2010 Earth orientation instead of the built-in analytic Moon.  Default False
///
/// Returns:
///     (satkit.time | None, satkit.time | None): moonrise and moonset, UTC, each None if it does not happen that day
///
/// Raises:
///     RuntimeError: with use_jpl, if the JPL ephemeris is unavailable or does not cover the date
///
/// Example:
///
/// ```python
/// from datetime import datetime, timedelta, timezone
///
/// # Moonrise and moonset on 2024-03-15 (local date) in New York
/// eastern = timezone(timedelta(hours=-4))
/// coord = satkit.itrfcoord(latitude_deg=40.71, longitude_deg=-74.01)
/// noon = datetime(2024, 3, 15, 12, tzinfo=eastern)
/// rise, set = satkit.moon.rise_set(noon, coord)
/// print(f"Moonrise: {rise.to_datetime().astimezone(eastern) if rise else None}")
/// print(f"Moonset:  {set.to_datetime().astimezone(eastern) if set else None}")
/// ```
#[pyfunction(signature=(time, coord, *, use_jpl=false))]
pub fn rise_set(
    time: TimeArg,
    coord: &PyITRFCoord,
    use_jpl: bool,
) -> PyResult<(Option<PyInstant>, Option<PyInstant>)> {
    let (rise, set) = moon::riseset(&time.0, &coord.0, use_jpl).map_err(moon_err)?;
    Ok((rise.map(PyInstant), set.map(PyInstant)))
}

/// Times of the principal Moon phases in a time interval
///
/// Returns every New Moon, First Quarter, Full Moon and Last Quarter in
/// [start, end), in time order.
///
/// Notes:
///     * The principal phases are when the Moon's geocentric apparent
///       ecliptic longitude exceeds the Sun's by 0, 90, 180 and 270 degrees
///       (the Astronomical Almanac / USNO definition).  Full Moon is
///       therefore not exactly the instant of greatest illuminated fraction.
///     * With use_jpl, apparent JPL positions (light time and aberration,
///       which move the Sun by 20 arcsec and the phase times by ~40 s) in
///       the mean ecliptic of date; otherwise the built-in analytic Sun and
///       Moon (see phase)
///     * Accuracy over 2024: with use_jpl, 0.1 s of Skyfield (DE440s) and
///       40 s of USNO's minute-rounded times; built-in, up to 22 minutes
///       (the analytic Moon's 0.36 deg worst-case longitude error allows 42)
///
/// Args:
///     start (satkit.time|datetime.datetime|numpy.datetime64): start of the interval (inclusive)
///     end (satkit.time|datetime.datetime|numpy.datetime64): end of the interval (exclusive)
///     use_jpl (bool, optional): use apparent JPL positions instead of the built-in analytic Sun and Moon.  Default False
///
/// Returns:
///     list[tuple[satkit.moon.moonphase, satkit.time]]: (phase, time) pairs in time order; empty if end <= start
///
/// Raises:
///     RuntimeError: with use_jpl, if the JPL ephemeris is unavailable or does not cover the interval
///
/// Example:
///
/// ```python
/// # The full moons of 2024
/// for phase, t in satkit.moon.phase_times(satkit.time(2024, 1, 1), satkit.time(2025, 1, 1)):
///     if phase == satkit.moon.moonphase.FullMoon:
///         print(t)
/// ```
#[pyfunction(signature=(start, end, *, use_jpl=false))]
pub fn phase_times(
    start: TimeArg,
    end: TimeArg,
    use_jpl: bool,
) -> PyResult<Vec<(MoonPhase, PyInstant)>> {
    Ok(moon::phase_times(&start.0, &end.0, use_jpl)
        .map_err(moon_err)?
        .into_iter()
        .map(|(p, t)| (MoonPhase::from(p), PyInstant(t)))
        .collect())
}

/// Time of the next occurrence of a principal Moon phase
///
/// Returns the first time at or after `time` when the Moon reaches `phase`.
/// See phase_times for the definition and accuracy.
///
/// Args:
///     time (satkit.time|datetime.datetime|numpy.datetime64): time from which to search
///     phase (satkit.moon.moonphase): NewMoon, FirstQuarter, FullMoon or LastQuarter
///     use_jpl (bool, optional): use apparent JPL positions instead of the built-in analytic Sun and Moon.  Default False
///
/// Returns:
///     satkit.time: time of the phase, UTC
///
/// Raises:
///     ValueError: if phase is not one of the four principal phases (e.g. WaxingCrescent, which spans a range of phases)
///     RuntimeError: with use_jpl, if the JPL ephemeris is unavailable or does not cover the search
///
/// Example:
///
/// ```python
/// full = satkit.moon.next_phase(satkit.time(2024, 1, 1), satkit.moon.moonphase.FullMoon)
/// print(f"Next full moon: {full}")
/// ```
#[pyfunction(signature=(time, phase, *, use_jpl=false))]
pub fn next_phase(time: TimeArg, phase: &MoonPhase, use_jpl: bool) -> PyResult<PyInstant> {
    moon::next_phase(&time.0, phase.into(), use_jpl)
        .map(PyInstant)
        .map_err(moon_err)
}
