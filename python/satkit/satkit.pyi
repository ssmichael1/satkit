"""
Toolkit containing functions and classes used in satellite dynamics
calculations.
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import datetime

from collections.abc import Callable
from typing import Any, Generic, TypeVar, Optional

R = TypeVar("R")

class static_property(Generic[R]):
    def __init__(self, getter: Callable[[Any], R]) -> None:
        self.__getter = getter

    def __get__(self, obj: object, objtype: type) -> R:
        return self.__getter(objtype)

    @staticmethod
    def __call__(getter_fn: Callable[[Any], R]) -> "static_property[R]":
        return static_property(getter_fn)

class TLE:
    """Two-Line Element Set (TLE) representing a satellite ephemeris

    Stucture representing a Two-Line Element Set (TLE), a satellite
    ephemeris format from the 1970s that is still somehow in use
    today and can be used to calculate satellite position and
    velcocity in the "TEME" frame (not-quite GCRF) using the
    "Simplified General Perturbations-4" (SGP-4) mathemematical
    model that is also included in this package.

    For details, see: https://en.wikipedia.org/wiki/Two-line_element_set

    The TLE format is still commonly used to represent satellite
    ephemerides, and satellite ephemerides catalogs in this format
    are publicly availalble at www.space-track.org (registration
    required)

    TLEs sometimes have a "line 0" that includes the name of the satellite
    """

    @staticmethod
    def from_file(filename: str) -> list[TLE] | TLE:
        """Load TLEs from input text file
        Return a list of TLES loaded from input text file.

        If the file contains lines only represent a single TLE, the TLE will
        be output, rather than a list with a single TLE element

        Args:
            filename (str): name of textfile lines for TLE(s) to load

        Returns:
            list[TLE] | TLE: a list of TLE objects or a single TLE of lines for
            only 1 are passed in
        """

    @staticmethod
    def from_lines(lines: list[str]) -> list[TLE] | TLE:
        """Return a list of TLES loaded from input list of lines

            If the file contains lines only represent a single TLE, the TLE will
            be output, rather than a list with a single TLE element

        Args:
            lines (list[str]): list of strings with lines for TLE(s) to load

        Returns:
            list[TLE] | TLE: a list of TLE objects or a single TLE of lines for
            only 1 are passed in
        """

    @property
    def satnum(self) -> int:
        """Satellite number, or equivalently the NORAD ID"""

    @property
    def eccen(self) -> float:
        """Satellite eccentricity, in range [0,1]"""

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly in degrees"""

    @property
    def mean_motion(self) -> float:
        """Mean motion in revs / day"""

    @property
    def inclination(self) -> float:
        """Inclination, in degrees"""

    @property
    def epoch(self) -> time:
        """TLE epoch"""

    @property
    def arg_of_perigee(self) -> time:
        """Argument of Perigee, in degrees"""

    @property
    def mean_motion_dot(self) -> float:
        """1/2 of first derivative of mean motion, in revs/day^2

        Note:
            the "1/2" is because that is how number is stored in the TLE
        """

    @property
    def mean_motion_dot_dot(self) -> float:
        """1/6 of 2nd derivative of mean motion, in revs/day^3

        Note:
            The "1/6" is because that is how number is stored in the TLE
        """

    @property
    def name(self) -> str:
        """The name of the satellite"""

    @property
    def bstar(self) -> str:
        """Drag of the satellite

        should be rho0 * Cd * A / 2 / m

        Units (which are strange) is multiples of
        1 / Earth radius
        """

def sgp4(
    tle: TLE | list[TLE],
    tm: time | list[time] | npt.ArrayLike,
    **kwargs,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """SGP-4 propagator for TLE

    Note:
        Run Simplified General Perturbations (SGP)-4 propagator on Two-Line Element Set to
        output satellite position and velocity at given time
        in the "TEME" coordinate system

        A detailed description is at:
        https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf

    Args:
        tle (TLE | list[TLE]): TLE (or list of TLES) on which to operate
        tm (time | list[time] | npt.ArrayLike[time]): time(s) at which to compute position and velocity

    Keyword Args:
        gravconst (satkit.sgp4_gravconst): gravity constant to use.  Default is gravconst.wgs72
        opsmode (satkit.sgp4_opsmode): opsmode.afspc (Air Force Space Command) or opsmode.improved.  Default is opsmode.afspc
        errflag (bool): whether or not to output error conditions for each TLE and time output.  Default is False
                        (this is likely rarely needed, but can be useful for debugging)
                        (this may also flag a typing error ... I can't figure out how to get rid of it)

    Returns:
        tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]: position and velocity
        in meters and meters/second, respectively,
        in the TEME frame at each of the "Ntime" input times and each of the "Ntle" tles

        Additional return value if errflag is True:
        list[sgp4_error]: list of errors for each TLE and time output, if errflag is True

    Example:
        >>> lines = [
        >>>        "0 INTELSAT 902",
        >>>     "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
        >>>     "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300."
        >>> ]
        >>>
        >>> tle = satkit.TLE.single_from_lines(lines)
        >>>
        >>> # Compute TEME position & velocity at epoch
        >>> pteme, vteme = satkit.sgp4(tle, tle.epoch)
        >>>
        >>> # Rotate to ITRF frame
        >>> q = satkit.frametransform.qteme2itrf(tm)
        >>> pitrf = q * pteme
        >>> vitrf = q * vteme - np.cross(np.array([0, 0, satkit.univ.omega_earth]), pitrf)
        >>>
        >>> # convert to ITRF coordinate object
        >>> coord = satkit.itrfcoord.from_vector(pitrf)
        >>>
        >>> # Print ITRF coordinate object location
        >>> print(coord)
        ITRFCoord(lat:  -0.0363 deg, lon:  -2.2438 deg, hae: 35799.51 km)
    """

class sgp4_gravconst:
    """Gravity constant to use for SGP4 propagation"""

    @static_property
    def wgs72(self) -> sgp4_gravconst:
        """WGS-72"""

    @static_property
    def wgs72old(self) -> sgp4_gravconst:
        """WGS-72 Old"""

    @static_property
    def wgs84(self) -> sgp4_gravconst:
        """WGS-84"""

class sgp4_opsmode:
    """Ops Mode for SGP4 Propagation"""

    @static_property
    def afspc(self) -> int:
        """afspc (Air Force Space Command), the default"""

    @property
    def improved(self) -> int:
        """Improved"""

class gravmodel:
    """
    Earth gravity models available for use

    For details, see: http://icgem.gfz-potsdam.de/
    """

    @static_property
    def jgm3(self) -> gravmodel:
        """
        The "JGM3" gravity model

        This model is used by default in the orbit propagators
        """

    @static_property
    def jgm2(self) -> gravmodel:
        """
        The "JGM2" gravity model
        """

    @static_property
    def egm96(self) -> gravmodel:
        """
        The "EGM96" gravity model
        """

    @static_property
    def itugrace16(self) -> gravmodel:
        """
        the ITU Grace 16 gravity model
        """

def gravity(
    pos: list[float] | itrfcoord | npt.ArrayLike, **kwargs
) -> npt.NDArray[np.float64]:
    """Return acceleration due to Earth gravity at the input position

    Args:
        pos (list[float] | satkit.itrfcoord | npt.ArrayLike[np.float]): Position as ITRF coordinate or numpy 3-vector representing ITRF position in meters

    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.jgm3
        order (int): The order of the gravity model to use.  Default is 6, maximum is 16

    Returns:
        npt.ArrayLike[np.float]: acceleration in m/s^2 in the International Terrestrial Reference Frame (ITRF)


    Notes:
        *  For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.

    """

def gravity_and_partials(
    pos: itrfcoord | npt.NDArray[np.float64], **kwargs
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gravity and partial derivatives of gravity with respect to Cartesian coordinates

    Args:
        pos (itrfcoord | npt.ArrayLike[np.float]): Position as ITRF coordinate or numpy 3-vector representing ITRF position in meters


    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.jgm3
        order (int): The order of the gravity model to use.  Default is 6, maximum is 16

    Returns:
        tuple[npt.ArrayLike[np.float], np.arrayLike[np.float]]: acceleration in m/s^2 and partial derivative of acceleration with respect to ITRF Cartesian coordinate in m/s^2 / m


    For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.

    """

class solarsystem:
    """Solar system bodies for which high-precision ephemeris can be computed"""

    @static_property
    def Mercury(self) -> solarsystem:
        """Mercury"""

    @static_property
    def Venus(self) -> solarsystem:
        """Venus"""

    @static_property
    def EMB(self) -> solarsystem:
        """Earth-Moon Barycenter"""

    @static_property
    def Mars(self) -> solarsystem:
        """Mars"""

    @static_property
    def Jupiter(self) -> solarsystem:
        """Jupiter"""

    @static_property
    def Saturn(self) -> solarsystem:
        """Saturn"""

    @static_property
    def Uranus(self) -> solarsystem:
        """Uranus"""

    @static_property
    def Neptune(self) -> solarsystem:
        """Neptune"""

    @static_property
    def Pluto(self) -> solarsystem:
        """Pluto"""

    @static_property
    def Moon(self) -> solarsystem:
        """Moon"""

    @static_property
    def Sun(self) -> solarsystem:
        """Sun"""

class sgp4_error:
    """Represent errors from SGP-4 propagation of two-line element sets (TLEs)"""

    @static_property
    def success(self) -> sgp4_error:
        """Success"""

    @static_property
    def eccen(self) -> sgp4_error:
        """Eccentricity < 0 or > 1"""

    @static_property
    def mean_motion(self) -> sgp4_error:
        """Mean motion (revs / day) < 0"""

    @static_property
    def perturb_eccen(self) -> sgp4_error:
        """Perturbed eccentricity < 0 or > 1"""

    @static_property
    def semi_latus_rectum(self) -> sgp4_error:
        """Semi-Latus Rectum < 0"""

    @static_property
    def unused(self) -> sgp4_error:
        """Unused, but in base code, so keeping for completeness"""

    @static_property
    def orbit_decay(self) -> sgp4_error:
        """Orbit decayed"""

class weekday:
    """

    Represent the day of the week

    Values:
    * `Sunday`
    * `Monday`
    * `Tuesday`
    * `Wednesday`
    * `Thursday`
    * `Friday`
    * `Saturday`
    """

    @static_property
    def Sunday(self) -> weekday:
        """Sunday"""

    @static_property
    def Monday(self) -> weekday:
        """Monday"""

    @static_property
    def Tuesday(self) -> weekday:
        """Tuesday"""

    @static_property
    def Wednesday(self) -> weekday:
        """Wednesday"""

    @static_property
    def Thursday(self) -> weekday:
        """Thursday"""

    @static_property
    def Friday(self) -> weekday:
        """Friday"""

    @static_property
    def Saturday(self) -> weekday:
        """Saturday"""

class timescale:
    """
    Specify time scale used to represent or convert between the "satkit.time"
    representation of time

    Most of the time, these are not needed directly, but various time scales
    are needed to compute precise rotations between various inertial and
    Earth-fixed coordinate frames

    For an excellent overview, see:
    https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/MPG%20Book/Release/Chapter2-TimeScales.pdf

    Values:

    * `Invalid`: Invalid time scale
    * `UTC`: Universal Time Coordinate
    * `TT`: Terrestrial Time
    * `UT1`: UT1
    * `TAI`: International Atomic Time
    * `GPS`: Global Positioning System (GPS) time
    * `TDB`: Barycentric Dynamical Time
    """

    @static_property
    def Invalid(self) -> timescale:
        """Invalid time scale"""

    @static_property
    def UTC(self) -> timescale:
        """Universal Time Coordinate"""

    @static_property
    def TT(self) -> timescale:
        """Terrestrial Time"""

    @static_property
    def UT1(self) -> timescale:
        """UT1"""

    @static_property
    def TAI(self) -> timescale:
        """International Atomic Time
        (nice because it is monotonically increasing)
        """

    @static_property
    def GPS(self) -> timescale:
        """Global Positioning System (GPS) time"""

    @static_property
    def TDB(self) -> timescale:
        """Barycentric Dynamical Time"""

class time:
    """Representation of an instant in time

    This has functionality similar to the "datetime" object, and in fact has
    the ability to convert to an from the "datetime" object.  However, a separate
    time representation is needed as the "datetime" object does not allow for
    conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)

    Notes:
        * If no arguments are passed in, the created object represents the current time
        * If year is passed in, month and day must also be passed in
        * If hour is passed in, minute and second must also be passed in

    Args:
        year (int, optional): Gregorian year (e.g., 2024)
        month (int, optional): Gregorian month (1 = January, 2 = February, ...)
        day (int, optional): Day of month, beginning with 1
        hour (int, optional): Hour of day, in range [0,23] (optional), default is 0
        min (int, optional): Minute of hour, in range [0,59], default is 0
        sec (float, optional): floating point second of minute, in range [0,60), default is 0
        scale (satkit.timescale, optional): Time scale , default is satkit.timescale.UTC
        str (str, optional): string representation of time, in format "YYYY-MM-DD HH:MM:SS.sssZ" or if other will try to guess
    Returns:
        satkit.time: Time object representing input date and time, or if no arguments, the current date and time

    Example:
        >>> print(satkit.time(2023, 3, 5, 11, 3, 45.453))
        2023-03-05 11:03:45.453Z

        >>> print(satkit.time(2023, 3, 5))
        2023-03-05 00:00:00.000Z

    """

    def __init__(self, *args):
        """Create a time object representing input date and time

        This has functionality similar to the "datetime" object, and in fact has
        the ability to convert to an from the "datetime" object.  However, a separate
        time representation is needed as the "datetime" object does not allow for
        conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)

        Notes:
            * If no arguments are passed in, the created object represents the current time

        Args:
            year (int, optional): Gregorian year (e.g., 2024)
            month (int, optional): Gregorian month (1 = January, 2 = February, ...)
            day (int, optional): Day of month, beginning with 1
            hour (int, optional): Hour of day, in range [0,23] (optional), default is 0
            min (int, optional): Minute of hour, in range [0,59], default is 0
            sec (float, optional): floating point second of minute, in range [0,60), default is 0
            scale (satkit.timescale, optional): Time scale , default is satkit.timescale.UTC
            str (str, optional): string representation of time, in format "YYYY-MM-DD HH:MM:SS.sssZ" or if other will try to guess

        Returns:
            satkit.time: Time object representing input date and time, or if no arguments, the current date and time

        Example:
            >>> print(satkit.time(2023, 3, 5, 11, 3,45.453))
            2023-03-05 11:03:45.453Z

            >>> print(satkit.time(2023, 3, 5))
            2023-03-05 00:00:00.000Z
        """

    @staticmethod
    def now() -> time:
        """Create a "time" object representing the instant of time at the
        calling of the function.

        Returns:
            satkit.time: Time object representing the current time
        """

    @staticmethod
    def from_string(str: str) -> time:
        """
        Create a "time" object from input string

        Args:
            str (str): string representation of time, in format "YYYY-MM-DD HH:MM:SS.sssZ" or if other will try
            to intelligently parse, but no guarantees

        Note:
            * This is probably not what you want.  Use with caution.

        Returns:
            satkit.time: Time object representing input string

        Example:
            >>> print(satkit.time.from_string("2023-03-05 11:03:45.453Z"))
            2023-03-05 11:03:45.453Z
        """

    @staticmethod
    def from_rfc3339(rfc: str) -> time:
        """Create a "time" object from input RFC 3339 string

        Args:
            rfc (str): RFC 3339 string representation of time

        Notes:
            * RFC 3339 is a subset of ISO 8601
            * Only allows a subset of the format: "YYYY-MM-DDTHH:MM:SS.sssZ" or "YYYY-MM-DDTHH:MM:SS.ssssssZ"

        Returns:
            satkit.time: Time object representing input RFC 3339 string

        Example:
            >>> print(satkit.time.from_rfctime("2023-03-05T11:03:45.453Z"))
            2023-03-05 11:03:45.453Z
        """

    @staticmethod
    def strptime(str: str, format: str) -> time:
        """
        Create a "time" object from input string with given formatting

        Args:
            str (str): string representation of time
            format (str): format of the string

        Notes:
        * The format string is a subset of the strptime format string in the Python "datetime" module

        Format Codes:
        * %Y - year
        * %m - month with leading zeros (01-12)
        * %d - day of month with leading zeros (01-31)
        * %H - hour with leading zeros (00-23)
        * %M - minute with leading zeros (00-59)
        * %S - second with leading zeros (00-59)
        * %f - microsecond, allowing for trailing zeros
        * %b - abbreviated month name (Jan, Feb, ...)
        * %B - full month name (January, February, ...)

        Returns:
            satkit.time: Time object representing input string

        Example:
            # Note the microsecond %f actually is represented as milliseconds in the input string
            >>> print(satkit.time.strptime("2023-03-05 11:03:45.453Z", "%Y-%m-%d %H:%M:%S.%fZ"))
            2023-03-05 11:03:45.453Z
        """

    @staticmethod
    def from_date(year: int, month: int, day: int) -> time:
        """Return a time object representing the start of the input day (midnight)

        Args:
            year (int): Gregorian year (e.g., 2024)
            month (int): Gregorian month (1 = January, 2 = February, ...)
            day (int): Day of month, beginning with 1

        Returns:
            satkit.time: Time object representing the start of the input day (midnight)
        """

    @staticmethod
    def from_jd(jd: float, scale: timescale = timescale.UTC) -> time:
        """Return a time object representing input Julian date and time scale

        Args:
            jd (float): Julian date
            scale (timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            satkit.time: Time object representing input Julian date and time scale
        """

    @staticmethod
    def from_unixtime(ut: float) -> time:
        """Return a time object representing input unixtime

        Args:
            ut (float): unixtime, UTC seconds since Jan 1, 1970 00:00:00
                        (leap seconds are not included)

        Returns:
            satkit.time: Time object representing input unixtime
        """

    @staticmethod
    def from_gps_week_and_second(week: int, sec: float) -> time:
        """Return a time object representing input GPS week and second

        Args:
            week (int): GPS week number
            sec (float): GPS seconds of week
            scale (timescale, optional): Time scale.  Default is satkit.timescale.GPS

        Returns:
            satkit.time: Time object representing input GPS week and second
        """

        def weekday(self) -> weekday:
            """
            Return the day of the week

            Returns:
                satkit.weekday: Day of the week
            """

    @staticmethod
    def from_mjd(mjd: float, scale: timescale = timescale.UTC) -> time:
        """Return a time object representing input modified Julian date and time scale

        Args:
            mjd (float): Modified Julian date
            scale (satkit.timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            satkit.time: Time object representing input modified Julian date and time scale
        """

    def as_date(self) -> tuple[int, int, int]:
        """Return tuple representing as UTC Gegorian date of the time object.

        Returns:
            tuple[int, int, int]: Tuple with 3 elements representing the Gregorian year, month, and day of the time object

        Fractional component of day are truncated
        Month is in range [1,12]
        Day is in range [1,31]
        """

    @typing.overload
    @staticmethod
    def from_datetime(
        year: int,
        month: int,
        day: int,
        hour: int,
        min: int,
        sec: float,
        scale: timescale = timescale.UTC,
    ) -> time:
        """Create time object from 6 input arguments representing UTC Gregorian time.

        Args:
            year (int): Gregorian year
            month (int): Gregorian month (1 = January, 2 = February, ...)
            day (int): Day of month, beginning with 1
            hour (int): Hour of day, in range [0,23]
            min (int): Minute of hour, in range [0,59]
            sec (float): floating point second of minute, in range [0,60)
            scale (timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            satkit.time: Time object representing input UTC Gregorian time

        Example:
            >>> print(satkit.time.from_datetime(2023, 3, 5, 11, 3,45.453))
            2023-03-05 11:03:45.453Z
        """

    def as_gregorian(
        self, scale=timescale.UTC
    ) -> tuple[int, int, int, int, int, float]:
        """Return tuple representing as UTC Gegorian date and time of the time object.

        Args:
            scale (timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            tuple[int, int, int, int, int, float]: Tuple with 6 elements representing the Gregorian year, month, day, hour, minute, and second of the time object

        Month is in range [1,12]
        Day is in range [1,31]
        """

    @staticmethod
    def from_gregorian(
        year: int,
        month: int,
        day: int,
        hour: int,
        min: int,
        sec: float,
    ) -> time:
        """Create time object from 6 input arguments representing UTC Gregorian time.

        Args:
            year (int): Gregorian year
            month (int): Gregorian month (1 = January, 2 = February, ...)
            day (int): Day of month, beginning with 1
            hour (int): Hour of day, in range [0,23]
            min (int): Minute of hour, in range [0,59]
            sec (float): floating point second of minute, in range [0,60)

        Returns:
            satkit.time: Time object representing input UTC Gregorian time

        Example:
            >>> print(satkit.time.from_gregorian(2023, 3, 5, 11, 3,45.453))
            2023-03-05 11:03:45.453Z
        """

    @typing.overload
    @staticmethod
    def from_datetime(dt: datetime.datetime) -> time:
        """Convert input "datetime.datetime" object to an "satkit.time" object representing the same instant in time

        Args:
            dt (datetime.datetime): "datetime.datetime" object to convert

        Returns:
            satkit.time: Time object representing the same instant in time as the input "datetime.datetime" object
        """

    def datetime(self, utc: bool = True) -> datetime.datetime:
        """Convert object to "datetime.datetime" object representing same instant in time.

        Args:
            utc (bool, optional): Whether to make the "datetime.datetime" object represent time in the local timezone or "UTC".  Default is True

        Returns:
            datetime.datetime: "datetime.datetime" object representing the same instant in time as the "satkit.time" object

        Example:
            >>> dt = satkit.time(2023, 6, 3, 6, 19, 34).datetime(True)
            >>> print(dt)
            2023-06-03 06:19:34+00:00
            >>>
            >>> dt = satkit.time(2023, 6, 3, 6, 19, 34).datetime(False)
            >>> print(dt)
            2023-06-03 02:19:34
        """

    def as_mjd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as a Modified Julian Date
        with the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC
        """

    def as_jd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as Julian Date with
        the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC
        """

    def as_unixtime(self) -> float:
        """
        Represent time as unixtime

        (seconds since Jan 1, 1970 UTC, excluding leap seconds)

        Includes fractional comopnent of seconds
        """

    def as_iso8601(self) -> str:
        """
        Represent time as ISO 8601 string

        Returns:
            str: ISO 8601 string representation of time: "YYYY-MM-DDTHH:MM:SS.sssZ"
        """

    def as_rfc3339(self) -> str:
        """
        Represent time as RFC 3339 string

        Returns:
            str: RFC 3339 string representation of time: "YYYY-MM-DDTHH:MM:SS.sssZ"
        """

    def strftime(self, format: str) -> str:
        """
        Represent time as string with given format

        Args:
            format (str): format of the string

        Format Codes:
        * %Y - year
        * %m - month with leading zeros (01-12)
        * %d - day of month with leading zeros (01-31)
        * %H - hour with leading zeros (00-23)
        * %M - minute with leading zeros (00-59)
        * %S - second with leading zeros (00-59)
        * %f - microsecond, allowing for trailing zeros
        * %b - abbreviated month name (Jan, Feb, ...)
        * %B - full month name (January, February, ...)
        * %A - full weekday name (Sunday, Monday, ...)
        * %w - weekday as a decimal number (0=Sunday, 1=Monday, ...)

        Returns:
            str: string representation of time

        Example:
            >>> print(satkit.time(2023, 6, 3, 6, 19, 34).strptime("%Y-%m-%d %H:%M:%S"))
            2023-06-03 06:19:34
        """

    @typing.overload
    def __add__(self, other: duration) -> time:
        """
        Return a time object representing the input duration added to the current time

        Args:
            other (duration): duration to add to the current time

        Returns:
            satkit.time: Time object representing the input duration added to the current time

        """

    @typing.overload
    def __add__(self, other: float) -> time:
        """
        Return a time object representing the input number of days added to the current time

        Args:
            other (float): number of days to add to the current time

        Returns:
            satkit.time: Time object representing the input number of days added to the current time

        """

    @typing.overload
    def __add__(self, other: list[duration]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise addition of days to the "self" time object

        Args:
            other (list[duration]): array-like structure containing days to add to the current time

        Returns:
            npt.ArrayLike[time]: Array of time objects representing the element-wise addition of days to the current time
        """

    def __le__(self, other: time) -> bool:
        """
        Compare two time objects for less than or equal to

        Args:
            other (time): time object to compare with

        Returns:
            bool: True if "self" time is less than or equal to "other" time, False otherwise
        """

    def __lt__(self, other: time) -> bool:
        """
        Compare two time objects for less than

        Args:
            other (time): time object to compare with

        Returns:
            bool: True if "self" time is less than "other" time, False otherwise
        """

    def __ge__(self, other: time) -> bool:
        """
        Compare two time objects for greater than or equal to

        Args:
            other (time): time object to compare with

        Returns:
            bool: True if "self" time is greater than or equal to "other" time, False otherwise
        """

    def __gt__(self, other: time) -> bool:
        """
        Compare two time objects for greater than

        Args:
            other (time): time object to compare with

        Returns:
            bool: True if "self" time is greater than "other" time, False otherwise
        """

    def __eq__(self, value: object) -> bool:
        """
        Compare two time objects for equality

        Args:
            value (object): object to compare with

        Returns:
            bool: True if "self" time is equal to "value", False otherwise
        """

    def __ne__(self, value: object) -> bool:
        """
        Compare two time objects for inequality

        Args:
            value (object): object to compare with

        Returns:
            bool: True if "self" time is not equal to "value", False otherwise
        """

    @typing.overload
    def __add__(self, other: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise addition of duration to the "self" time object

        Args:
            other (npt.ArrayLike[Any]): array-like structure containing durations to add to the current time

        Returns:
            npt.ArrayLike[time]: Array of time objects representing the element-wise addition of durations to the current time

        """

    @typing.overload
    def __sub__(self, other: duration) -> time:
        """
        Return a time object representing the input duration subtracted from the current time

        Args:
            other (duration): duration to subtract from the current time

        Returns:
            satkit.time: Time object representing the input duration subtracted from the current time

        """

    @typing.overload
    def __sub__(self, other: time) -> duration:
        """
        Return a duration object representing the difference between the two times

        Args:
            other (time): time to subtract from the current time

        Returns:
            satkit.duration: Duration object representing the difference between the two times

        """

    @typing.overload
    def __sub__(self, other: float) -> time:
        """
        Return a time object representing the input number of days subtracted from the current time

        Args:
            other (float): number of days to subtract from the current time

        Returns:
            satkit.time: Time object representing the input number of days subtracted from the current time

        """

    @typing.overload
    def __sub__(self, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of days from the "self" time object

        Args:
            other (npt.ArrayLike[float]): array-like structure containing days to subtract from the current time

        Returns:
            npt.ArrayLike[time]: Array of time objects representing the element-wise subtraction of days from the current time

        """

    @typing.overload
    def __sub__(self, other: list[duration]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of duration from the "self" time object

        Args:
            other (list[duration]): array-like structure containing durations to subtract from the current time

        Returns:
            npt.ArrayLike[time]: Array of time objects representing the element-wise subtraction of durations from the current time
        """

    @typing.overload
    def __sub__(self, other: list[time]) -> npt.NDArray[Any]:
        """
        Return a numpy array of duration objects, with each object representing an element-wise subtraction of time from the "self" time object

        Args:
            other (list[time]): array-like structure containing times to subtract from the current time

        Returns:
            npt.ArrayLike[duration]: Array of duration objects representing the element-wise subtraction of times from the current time
        """

class duration:
    """
    Representation of a duration, or interval of time
    """

    def __init__(self, **kwargs):
        """Create a duration object representing input time duration

        Args:
            days (float, optional): Number of days, default is 0
            hours (float, optional): Number of hours, default is 0
            minutes (float, optional): Number of minutes, default is 0
            seconds (float, optional): Number of seconds, default is 0.0
            microseconds(float, optional): Number of microseconds, default is 0.0

        Notes:
            * If no arguments are passed in, the created object represents a duration of 0 seconds

        Returns:
            satkit.duration: Duration object representing input time duration

        Example:
            >>> print(satkit.duration(days=1, hours=2, minutes=3, seconds=4.5))
            Duration: 1 days, 2 hours, 3 minutes, 4.500 seconds

        """

    @staticmethod
    def from_days(d: float) -> duration:
        """Create duration object given input number of days. Note: a day is defined as 86,400 seconds

        Args:
            d (float): Number of days

        Returns:
            satkit.duration: Duration object representing input number of days
        """

    @staticmethod
    def from_seconds(s: float) -> duration:
        """Create duration object representing input number of seconds

        Args:
            s (float): Number of seconds

        Returns:
            satkit.duration: Duration object representing input number of seconds
        """

    @staticmethod
    def from_minutes(m: float) -> duration:
        """Create duration object representing input number of minutes

        Args:
            m (float): Number of minutes

        Returns:
            satkit.duration: Duration object representing input number of minutes
        """

    @staticmethod
    def from_hours(h: float) -> duration:
        """Create duration object representing input number of hours

        Args:
            h (float): Number of hours

        Returns:
            satkit.duration: Duration object representing input number of hours
        """

    @typing.overload
    def __add__(self, other: duration) -> duration:
        """Add a duration to another duration

        Args:
            other (duration): duration to add to the current duration

        Returns:
            duration: Duration object representing the sum, or concatenation, of both durations

        Example:
            >>> print(duration.from_hours(1) + duration.from_minutes(1))
            Duration: 1 hours, 1 minutes, 0.000 seconds
        """

    @typing.overload
    def __add__(self, other: float) -> duration:
        """Add a number of days to the current duration

        Args:
            other (float): number of days to add to the current duration

        Returns:
            duration: Duration object representing the input number of days added to the current duration

        Example:
            >>> print(duration.from_days(1) + 2.5)
            Duration: 3 days, 0 hours, 0 minutes, 0.000 seconds
        """

    @typing.overload
    def __add__(self, other: time) -> time:
        """Add a duration to a time

        Args:
            other (time): time to add the current duration to

        Returns:
            time: Time object representing the input time plus the duration

        Example:
            >>> print(duration.from_hours(1) + satkit.time(2023, 6, 4, 11,30,0))
            2023-06-04 13:30:00.000Z
        """

    def __sub__(self, other: duration) -> duration:
        """Take the difference between two durations

        Args:
            other (duration): duration to subtract from the current duration

        Returns:
            duration: Duration object representing the difference between the two durations

        Example:
            >>> print(duration.from_hours(1) - duration.from_minutes(1))
            Duration: 59 minutes, 0.000 seconds
        """

    def __mul__(self, other: float) -> duration:
        """Multiply (or scale) duration by given value

        Args:
            other (float): value by which to multiply duration

        Returns:
            duration: Duration object representing the input duration scaled by the input value

        Example:
            >>> print(duration.from_days(1) * 2.5)
            Duration: 2 days, 12 hours, 0 minutes, 0.000 seconds
        """

    def __gt__(self, other: duration) -> bool:
        """Compare two durations for greater than

        Args:
            other (duration): duration to compare with
        Returns:
            bool: True if "self" duration is greater than "other" duration, False otherwise

        Example:
            >>> print(duration.from_hours(1) > duration.from_minutes(30))
            True
        """

    def __lt__(self, other: duration) -> bool:
        """Compare two durations for less than

        Args:
            other (duration): duration to compare with
        Returns:
            bool: True if "self" duration is less than "other" duration, False otherwise

        Example:
            >>> print(duration.from_hours(1) < duration.from_minutes(30))
            False
        """

    def __ge__(self, other: duration) -> bool:
        """Compare two durations for greater than or equal to

        Args:
            other (duration): duration to compare with
        Returns:
            bool: True if "self" duration is greater than or equal to "other" duration, False otherwise

        Example:
            >>> print(duration.from_hours(1) >= duration.from_minutes(30))
            True
        """

    def __le__(self, other: duration) -> bool:
        """Compare two durations for less than or equal to

        Args:
            other (duration): duration to compare with
        Returns:
            bool: True if "self" duration is less than or equal to "other" duration, False otherwise

        Example:
            >>> print(duration.from_hours(1) <= duration.from_minutes(30))
            False
        """

    def __eq__(self, other: duration) -> bool:
        """Compare two durations for equality

        Args:
            other (duration): duration to compare with
        Returns:
            bool: True if "self" duration is equal to "other" duration, False otherwise

        Example:
            >>> print(duration.from_hours(1) == duration.from_minutes(60))
            True
        """

    def __ne__(self, other: duration) -> bool:
        """Compare two durations for inequality

        Args:
            other (duration): duration to compare with
        Returns:
            bool: True if "self" duration is not equal to "other" duration, False otherwise

        Example:
            >>> print(duration.from_hours(1) != duration.from_minutes(30))
            True
        """

    @property
    def days(self) -> float:
        """Floating point number of days represented by duration

        Returns:
            float: Floating point number of days represented by duration

        A day is defined as 86,400 seconds
        """

    @property
    def hours(self) -> float:
        """Floating point number of hours represented by duration

        Returns:
            float: Floating point number of hours represented by duration
        """

    @property
    def minutes(self) -> float:
        """Floating point number of minutes represented by duration

        Returns:
            float: Floating point number of minutes represented by duration
        """

    @property
    def seconds(self) -> float:
        """Floating point number of seconds represented by duration

        Returns:
            float: Floating point number of seconds represented by duration
        """

class quaternion:
    """Quaternion representing rotation of 3D Cartesian axes

    Quaternions perform right-handed rotation of a vector, e.g. rotation of +xhat 90 degrees by +zhat give +yhat

    This is different than the convention used in Vallado, but it is the way it is commonly used in mathematics and it is the way it should be done.

    For the uninitiated: quaternions are a more-compact and
    computationally efficient way of representing 3D rotations.
    They can also be multipled together and easily renormalized to
    avoid problems with floating-point precision eventually causing
    changes in the rotated vecdtor norm.

    For details, see:

    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Notes:
        * Under the hood, this is using the "UnitQuaternion" object in the rust "nalgebra" crate.
    """

    def __init__(self):
        """Return unit quaternion (no rotation)

        Returns:
            satkit.quaternion: Quaternion representing no rotation
        """

    @staticmethod
    def from_axis_angle(axis: npt.NDArray[np.float64], angle: float) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "angle" degrees about the given axis

        Args:
            axis (npt.ArrayLike[np.float64]): 3-element array representing axis of rotation
            angle (float): angle of rotation in radians

        Returns:
            satkit.quaternion: Quaternion representing rotation by "angle" degrees about the given axis
        """

    @staticmethod
    def from_rotation_matrix(
        mat: npt.NDArray[np.float64],
    ) -> quaternion:
        """Return quaternion representing identical rotation to input 3x3 rotation matrix

        Args:
            mat (npt.ArrayLike[np.float64]): 3x3 rotation matrix

        Returns:
            satkit.quaternion: Quaternion representing identical rotation to input 3x3 rotation matrix
        """

    @staticmethod
    def rotx(theta) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the xhat unit vector

        Args:
            theta (float): angle of rotation in radians

        Returns:
            satkit.quaternion: Quaternion representing right-handed rotation of vector by "theta" radians about the xhat unit vector

        Notes:
            Equivalent rotation matrix:
            | 1             0            0|
            | 0    cos(theta)  -sin(theta)|
            | 0    sin(theta)   cos(theta)|
        """

    @staticmethod
    def roty(theta) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the yhat unit vector

        Args:
            theta (float): angle of rotation in radians

        Returns:
            satkit.quaternion: Quaternion representing right-handed rotation of vector by "theta" radians about the yhat unit vector


        Notes:
            Equivalent rotation matrix:
            |  cos(theta)     0    sin(theta)|
            |           0     1             0|
            | -sin(theta)     0    cos(theta)|
        """

    @staticmethod
    def rotz(theta) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the zhat unit vector

        Args:
            theta (float): angle of rotation in radians

        Returns:
            satkit.quaternion: Quaternion representing right-handed rotation of vector by "theta" radians about the zhat unit vector

        Notes:
            Equivalent rotation matrix:
            |  cos(theta)     -sin(theta)   0|
            |  sin(theta)      cos(theta)   0|
            |           0               0   1|
        """

    @staticmethod
    def rotation_between(
        v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]
    ) -> quaternion:
        """Quaternion represention rotation between two input vectors

        Args:
            v1 (npt.ArrayLike[np.float64]): vector rotating from
            v2 (npt.ArrayLike[np.float64]): vector rotating to

        Returns:
            satkit.quaternion: Quaternion that rotates from v1 to v2
        """

    def as_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Return 3x3 rotation matrix representing equivalent rotation

        Returns:
            npt.ArrayLike[np.float64]: 3x3 rotation matrix representing equivalent rotation
        """

    def as_euler(self) -> tuple[float, float, float]:
        """Return equivalent rotation angle represented as rotation angles: ("roll", "pitch", "yaw") in radians:

        Returns:
            tuple[float, float, float]: Tuple with 3 elements representing the rotation angles in radians

        """

    def angle(self) -> float:
        """Return the angle in radians of the rotation

        Returns:
            float: Angle in radians of the rotation
        """

    def axis(self) -> npt.NDArray[np.float64]:
        """Return the axis of rotation as a unit vector

        Returns:
            npt.ArrayLike[np.float64]: 3-element array representing the axis of rotation as a unit vector
        """

    @property
    def conj(self) -> quaternion:
        """Return conjugate or inverse of the rotation

        Returns:
            satkit.quaternion: Conjugate or inverse of the rotation
        """

    @property
    def conjugate(self) -> quaternion:
        """Return conjugate or inverse of the rotation

        Returns:
            satkit.quaternion: Conjugate or inverse of the rotation
        """

    @typing.overload
    def __mul__(self, other: quaternion) -> quaternion:
        """Multiply by another quaternion to concatenate rotations

        Notes:
            * Multiply represents concatenation of two rotations representing the quaternions.  The left value rotation is applied after the right value, per the normal convention

        Args:
            other (quaternion): quaternion to multiply by

        Returns:
            quaternion: Quaternion representing concatenation of the two rotations
        """

    @typing.overload
    def __mul__(self, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply by a vector to rotate the vector

        Args:
            other (npt.ArrayLike[np.float64]): 3-element array representing vector to rotate or Nx3 array of vectors to rotate

        Returns:
            npt.ArrayLike[np.float64]: 3-element array representing rotated vector or Nx3 array of rotated vectors

        Example:
            >>> xhat = np.array([1,0,0])
            >>> q = satkit.quaternion.rotz(np.pi/2)
            >>> print(q * xhat)
            [0, 1, 0]
        """

    def slerp(
        self, other: quaternion, frac: float, epsilon: float = 1.0e-6
    ) -> quaternion:
        """Spherical linear interpolation between self and other

        Args:
            other (quaternion): Quaternion to perform interpolation to
            frac (float): fractional amount of interpolation, in range [0,1]
            epsilon (float, optional): Value below which the sin of the angle separating both quaternions must be to return an error. Default is 1.0e-6

        Returns:
            quaternion: Quaternion representing interpolation between self and other
        """

class kepler:
    """Represent Keplerian element sets and convert between cartesian


    Notes:
    * This class is used to represent Keplerian elements and convert between Cartesian coordinates
    * The class uses the semi-major axis (a), not the semiparameter
    * All angle units are radians
    * All length units are meters
    * All velocity units are meters / second
    """

    def __init__(
        self,
        a: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        nu: float,
        **kwargs,
    ):
        """Create Keplerian element set object from input elements


        Args:
            a (float) : Semi-major axis, meters
            e (float) : Eccentricity, unitless
            i (float) : Inclination, radians
            raan (float) : Right ascension of ascending node, radians
            argp (float) : Argument of perigee, radians
            nu (float) : True anomaly, radians
            true_anomaly (float, optional keyword) : True anomaly, radians
            mean_anomaly (float, optional keyword) : Mean anomaly, radians
            eccentric_anomaly (float, optional keyword) : Eccentric anomaly, radians

        Notes:
        * If "nu" is provided (6th argument), it will be used as the true anomaly
        * Anomaly may also be set via keyword arguments; if so, the there should only be
            5 input arguments

        Returns:
            satkit.kepler: Keplerian element set object
        """

    def to_pv(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert Keplerian element set to position and velocity vectors

        Returns:
            tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]: Tuple with two elements representing the position and velocity vectors
        """

    def propagate(self, dt: duration | float) -> kepler:
        """Propagate Keplerian element set by input duration

        Args:
            dt (duration | float): Duration by which to propagate the Keplerian element set
                                   If float, value is seconds

        Returns:
            satkit.kepler: Keplerian element set object after propagation
        """

    @property
    def mean_motion(self) -> float:
        """Mean motion, radians / second"""

    @property
    def true_anomaly(self) -> float:
        """True anomaly, radians"""

    @property
    def eccentric_anomaly(self) -> float:
        """Eccentric anomaly, radians"""

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly, radians"""

    @property
    def period(self) -> float:
        """Orbital period, seconds"""

    @property
    def a(self) -> float:
        """Semi-major axis, meters"""

    @property
    def eccen(tricity) -> float:
        """Eccentricity, unitless"""

    @property
    def inclination(self) -> float:
        """Inclination, radians"""

    @property
    def raan(self) -> float:
        """Right ascension of ascending node, radians"""

    @property
    def nu(self) -> float:
        """True anomaly, radians"""

    @property
    def w(self) -> float:
        """Argument of perigee, radians"""

    @staticmethod
    def from_pv(pos: npt.NDArray[np.float64], vel: npt.NDArray[np.float64]) -> kepler:
        """Create Keplerian element set from input position and velocity vectors

        Args:
            pos (npt.ArrayLike[np.float64]): 3-element array representing position vector
            vel (npt.ArrayLike[np.float64]): 3-element array representing velocity vector

        Returns:
            satkit.kepler: Keplerian element set object
        """

class itrfcoord:
    """Representation of a coordinate in the International Terrestrial Reference Frame (ITRF)

    Notes:

    * This coordinate object can be created from and also output to Geodetic coordinates (latitude, longitude, height above ellipsoid).
    * Functions are also available to provide rotation quaternions to the East-North-Up frame and North-East-Down frame at this coordinate


    Args:

        vec (numpy.ndarray, list, or 3-element tuple of floats, optional): ITRF Cartesian location in meters
        latitude_deg (float, optional): Latitude in degrees
        longitude_deg (float, optional): Longitude in degrees
        latitude_rad (float, optional): Latitude in radians
        longitude_rad (float, optional): Longitude in radians
        altitude (float, optional): Height above ellipsoid, meters
        height (float, optional): Height above ellipsoid, meters

    Returns:
        itrfcoord: New ITRF coordinate

    Example:

        * Create ITRF coord from Cartesian
        >>> coord = itrfcoord([ 1523128.63570828 -4461395.28873207  4281865.94218203 ])
        >>> print(coord)
        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)

        * Create ITRF coord from Geodetic
        >>> coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
        >>> print(coord)
        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)

    """

    def __init__(self, *args, **kwargs):
        """Representation of a coordinate in the International Terrestrial Reference Frame (ITRF)

        Notes:

        * This coordinate object can be created from and also output to Geodetic coordinates (latitude, longitude, height above ellipsoid).
        * Functions are also available to provide rotation quaternions to the East-North-Up frame and North-East-Down frame at this coordinate

        Args:
            vec (numpy.ndarray|list[float]|tuple[float, float, float], optional): ITRF Cartesian location in meters
            latitude_deg (float, optional): Latitude in degrees
            longitude_deg (float, optional): Longitude in degrees
            latitude_rad (float, optional): Latitude in radians
            longitude_rad (float, optional): Longitude in radians
            altitude (float, optional): Height above ellipsoid, meters
            height (float, optional): Height above ellipsoid, meters


        Returns:
            itrfcoord: New ITRF coordinate

        Example:
            * Create ITRF coord from Cartesian
            >>> coord = itrfcoord([ 1523128.63570828 -4461395.28873207  4281865.94218203 ])
            >>> print(coord)
            ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)

            * Create ITRF coord from Geodetic
            >>> coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
            >>> print(coord)
            ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)


        """

    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees"""

    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees"""

    @property
    def latitude_rad(self) -> float:
        """Latitude in radians"""

    @property
    def longitude_rad(self) -> float:
        """Longitude in radians"""

    @property
    def altitude(self) -> float:
        """Altitude above ellipsoid, in meters"""

    @property
    def geodetic_rad(self) -> tuple[float, float, float]:
        """Geodetic position in radians

        Returns:
            tuple[float, float, float]: Tuple with 3 elements representing the geodetic position. First element is latitude in radians, second is longitude in radians, and third is altitude in meters
        """

    @property
    def geodetic_deg(self) -> tuple[float, float, float]:
        """Geodetic position in degrees

        Returns:
            tuple[float, float, float]: Tuple with 3 elements representing the geodetic position. First element is latitude in degrees, second is longitude in degrees, and third is altitude in meters
        """

    @property
    def vector(self) -> npt.NDArray[np.float64]:
        """Cartesian ITRF coord as numpy array

        Returns:
            npt.NDArray[np.float64]: 3-element numpy array representing the ITRF Cartesian coordinate in meters
        """

    @property
    def qned2itrf(self) -> quaternion:
        """Quaternion representing rotation from North-East-Down (NED) to ITRF at this location

        Returns:
            satkit.quaternion: Quaternion representiong rotation from North-East-Down (NED) to ITRF at this location
        """

    @property
    def qenu2itrf(self) -> quaternion:
        """Quaternion representiong rotation from East-North-Up (ENU) to ITRF at this location

        Returns:
            satkit.quaternion: Quaternion representiong rotation from East-North-Up (ENU) to ITRF at this location
        """

    def __sub__(self, other: itrfcoord) -> npt.NDArray[np.float64]:
        """Subtract another ITRF coordinate from this one

        Args:
            other (itrfcoord): Other ITRF coordinate to subtract

        Returns:
            npt.NDArray[np.float64]: 3-element numpy array representing the difference in meters between the two ITRF coordinates
        """

    def geodesic_distance(self, other: itrfcoord) -> tuple[float, float, float]:
        """Use Vincenty formula to compute geodesic distance:
        https://en.wikipedia.org/wiki/Vincenty%27s_formulae

        Returns:
            tuple[float, float, float]: (distance in meters, initial heading in radians, heading at destination in radians)

        """

    def move_with_heading(self, distance: float, heading_rad: float) -> itrfcoord:
        """Move a distance along the Earth surface with a given initial heading

        Args:
            distance (float): Distance to move in meters
            heading_rad (float): Initial heading in radians

        Notes:
            Altitude is assumed to be zero

            Use Vincenty formula to compute position:
            https://en.wikipedia.org/wiki/Vincenty%27s_formulae

        Returns:
            tuple[float, float, float]: (distance in meters, initial heading in radians, heading at destination in radians)

        """

class consts:
    """Some constants that are useful for saetllite dynamics"""

    @static_property
    def wgs84_a(self) -> float:
        """WGS-84 semiparameter, in meters"""

    @static_property
    def wgs84_f(self) -> float:
        """WGS-84 flattening in meters"""

    @static_property
    def earth_radius(self) -> float:
        """Earth radius along major axis, meters"""

    @static_property
    def mu_earth(self) -> float:
        """Gravitational parameter of Earth, m^3/s^2"""

    @static_property
    def mu_moon(self) -> float:
        """Gravitational parameter of Moon, m^3/s^2"""

    @static_property
    def mu_sun(self) -> float:
        """Gravitational parameter of sun, m^3/s^2"""

    @static_property
    def GM(self) -> float:
        """Gravitational parameter of Earth, m^3/s^2"""

    @static_property
    def omega_earth(self) -> float:
        """Scalar Earth rotation rate, rad/s"""

    @static_property
    def c(self) -> float:
        """Speed of light, m/s"""

    @static_property
    def au(self) -> float:
        """Astronomical Unit, mean Earth-Sun distance, meters"""

    @static_property
    def sun_radius(self) -> float:
        """Radius of sun, meters"""

    @static_property
    def moon_radius(self) -> float:
        """Radius of moon, meters"""

    @static_property
    def earth_moon_mass_ratio(self) -> float:
        """Earth mass over Moon mass, unitless"""

    @static_property
    def geo_r(self) -> float:
        """Distance to Geosynchronous orbit from Earth center, meters"""

    @static_property
    def jgm3_mu(self) -> float:
        """Earth gravitational parameter from JGM3 gravity model, m^3/s^2"""

    @static_property
    def jgm3_a(self) -> float:
        """Earth semiparameter from JGM3 gravity model, m"""

    @static_property
    def jgm3_j2(self) -> float:
        """ "J2" gravity due oblateness of Earth from JGM3 gravity model, unitless"""

class satstate:
    """
    A convenience class representing a satellite position and velocity, and
    optionally 6x6 position/velocity covariance at a particular instant in time

    This class can be used to propagate the position, velocity, and optional
    covariance to different points in time.
    """

    def __init__(
        self,
        time: time,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64] | None = None,
    ):
        """Create a new satellite state

        Args:
            time (satkit.time): Time instant of this state
            pos (npt.NDArray[np.float64]): Position in meters in GCRF frame
            vel (npt.NDArray[np.float64]): Velocity in meters / second in GCRF frame
            cov (npt.NDArray[np.float64]|None, optional): Covariance in GCRF frame. Defaults to None.  If input, should be a 6x6 numpy array

        Returns:
            satstate: New satellite state object
        """

    @property
    def pos(self) -> npt.NDArray[np.float64]:
        """state position in meters in GCRF frame

        Returns:
            npt.ArrayLike[np.float64]: 3-element numpy array representing position in meters in GCRF frame
        """

    @property
    def vel(self) -> npt.NDArray[np.float64]:
        """Return this state velocity in meters / second in GCRF

        Returns:
            npt.ArrayLike[np.float64]: 3-element numpy array representing velocity in meters / second in GCRF frame
        """

    @property
    def qgcrf2lvlh(self) -> quaternion:
        """Quaternion that rotates from the GCRF to the LVLH frame for the current state

        Returns:
            satkit.quaternion: Quaternion that rotates from the GCRF to the LVLH frame for the current state
        """

    @property
    def cov(self) -> npt.NDArray[np.float64] | None:
        """6x6 state covariance matrix in GCRF frame

        Returns:
            npt.ArrayLike[np.float64] | None: 6x6 numpy array representing state covariance in GCRF frame or None if not set
        """

    @property
    def time(self) -> time:
        """Return time of this satellite state

        Returns:
            satkit.time: Time instant of this state
        """

    def propagate(self, time: time | duration, propsettings=None) -> satstate:
        """Propagate this state to a new time, specified by the "time" input, updating the position, the velocity, and the covariance if set

        Args:
            time (satkit.time|satkit.duration): Time or duration from current time to which to propagate the state
            propsettings (satkit.propsettings, optional): object describing settings to use in the propagation.
                If omitted, default is used

        Returns:
            satstate: New satellite state object representing the state at the new time
        """

class propstats:
    """Statistics of a satellite propagation"""

    @property
    def num_eval(self) -> int:
        """Number of function evaluations"""

    @property
    def num_accept(self) -> int:
        """Number of accepted steps in adaptive RK integrator"""

    @property
    def num_reject(self) -> int:
        """Number of rejected steps in adaptive RK integrator"""

class propresult:
    """Results of a satellite propagation

    This class lets the user access results of the satellite propagation

    Notes:

    * If "enable_interp" is set to True in the propagation settings, the propresult object can be used to interpolate solutions at any time between the start and stop times of the propagation via the "interp" method

    """

    @property
    def pos(self) -> npt.NDArray[np.float64]:
        """GCRF position of satellite, meters

        Returns:
            npt.ArrayLike[float]: 3-element numpy array representing GCRF position (meters) at end of propagation

        """

    @property
    def vel(self) -> npt.NDArray[np.float64]:
        """GCRF velocity of satellite, meters/second

        Returns:
            npt.ArrayLike[float]: 3-element numpy array representing GCRF velocity in meters/second at end of propagation
        """

    @property
    def state(self) -> npt.NDArray[np.float64]:
        """6-element end state (pos + vel) of satellite in meters & meters/second

        Returns:
            npt.ArrayLike[float]: 6-element numpy array representing state of satellite in meters & meters/second
        """

    @property
    def state_end(self) -> npt.NDArray[np.float64]:
        """6-element state (pos + vel) of satellite in meters & meters/second at end of propagation

        Notes:
        * This is the same as the "state" property

        Returns:
            npt.ArrayLike[float]: 6-element numpy array representing state of satellite in meters & meters/second
        """

    @property
    def state_start(self) -> npt.NDArray[np.float64]:
        """6-element state (pos + vel) of satellite in meters & meters/second at start of propagation
        Returns:
            npt.NDArray[np.float64]: 6-element numpy array representing state of satellite in meters & meters/second at start of propagation
        """

    @property
    def time(self) -> time:
        """Time at which state is valid

        Returns:
            satkit.time: Time at which state is valid
        """

    @property
    def time_end(self) -> time:
        """Time at which state is valid

        Notes:
        * This is identical to "time" property

        Returns:
            satkit.time: Time at which state is valid
        """

    @property
    def time_start(self) -> time:
        """Time at which state_start is valid


        Returns:
            satkit.time: Time at which state_start is valid
        """

    @property
    def stats(self) -> propstats:
        """Statistics of propagation

        Returns:
            propstats: Object containing statistics of propagation
        """

    @property
    def phi(self) -> npt.NDArray[np.float64] | None:
        """State transition matrix

        Returns:
            npt.ArrayLike[np.float64] | None: 6x6 numpy array representing state transition matrix or None if not computed
        """

    def interp(
        self, time: time, output_phi: bool = False
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """Interpolate state at given time

        Args:
            time (satkit.time): Time at which to interpolate state
            output_phi (bool, optional): Output 6x6 state transition matrix at the interpolated time
                Default is False

        Returns:
            npt.ArrayLike[np.float64] | tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]: 6-element vector representing state at given time. if output_phi, also output 6x6 state transition matrix at given time
        """

class satproperties_static:
    """Satellite properties relevant for drag and radiation pressure

    This class lets the satellite radiation pressure and drag
    paramters be set to static values for duration of propagation

    Attributes:
        cdaoverm (float): Coefficient of drag times area over mass in m^2/kg
        craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg

    """

    def __init__(self, cdaoverm: float = 0, craoverm: float = 0):  # type: ignore
        """Create a satproperties_static object with given craoverm and cdaoverm in m^2/kg

        Args:
            cdaoverm (float, optional): Coefficient of drag times area over mass in m^2/kg
            craoverm (float, optional): Coefficient of radiation pressure times area over mass in m^2/kg


        Notes:

        * The two arguments can be passed as positional arguments or as keyword arguments

        Example:

        >>> properties = satproperties_static(craoverm = 0.5, cdaoverm = 0.4)

        or with same output

        >>> properties = satproperties_static(0.5, 0.4)

        """

        @property
        def cdaoverm(self) -> float:
            """Coeffecient of drag times area over mass.  Units are m^2/kg"""

        @property
        def craoverm(self) -> float:
            """Coefficient of radiation pressure times area over mass.  Units are m^2/kg"""

class propsettings:
    """This class contains settings used in the high-precision orbit propgator part of the "satkit" python toolbox

    Notes:

    * Default settings:
        * abs_error: 1e-8
        * rel_error: 1e-8
        * gravity_order: 4
        * use_spaceweather: True
        * use_jplephem: True
        * enable_interp: True

    * enable_interp enables high-preciion interpolation of state between start and stop times via the returned function,
      it is enabled by default.  There is a small increase in computational efficiency if set to false

    """

    def __init__(**kwargs):
        """Create propagation settings object used to configure high-precision orbit propagator

        Args:
            abs_error (float, optional keyword): Maximum absolute value of error for any element in propagated state following ODE integration. Default is 1e-8
            rel_error (float, optional keyword): Maximum relative error of any element in propagated state following ODE integration. Default is 1e-8
            gravity_order (int, optional keyword): Earth gravity order to use in ODE integration. Default is 4
            use_spaceweather (bool, optional keyword): Use space weather data when computing atmospheric density for drag forces. Default is True
            use_jplephem (bool, optional keyword): Use JPL ephemeris for solar system bodies. Default is True
            enable_interp (bool, optional keyword): Store intermediate data that allows for fast high-precision interpolation of state between start and stop times. Default is True


        Returns:
            propsettings: New propsettings object with default settings
        """

    @property
    def abs_error(self) -> float:
        """Maxmum absolute value of error for any element in propagated state following ODE integration

        Returns:
            float: Maximum absolute value of error for any element in propagated state following ODE integration, default is 1e-8
        """

    @property
    def rel_error(self) -> float:
        """Maximum relative error of any element in propagated state following ODE integration

        Returns:
            float: Maximum relative error of any element in propagated state following ODE integration, default is 1e-8

        """

    @property
    def gravity_order(self) -> int:
        """Earth gravity order to use in ODE integration

        Returns:
            int: Earth gravity order to use in ODE integration, default is 4

        """

    @property
    def use_spaceweather(self) -> bool:
        """Use space weather data when computing atmospheric density for drag forces

        Notes:

        * Space weather data can have a large effect on the density of the atmosphere
        * This can be important for accurate drag force calculations
        * Space weather data is updated every 3 hours.  Most-recent data can be downloaded with ``satkit.utils.update_datafiles()``
        * Default value is True

        Returns:
            bool: Indicate whether or not space weather data should be used when computing atmospheric density for drag forces

        """

    @property
    def enable_interp(self) -> bool:
        """Store intermediate data that allows for fast high-precision interpolation of state between start and stop times
        If not needed, there is a small computational advantage if set to False
        """

    def precompute_terms(
        self, start: time, stop: time, step: Optional[duration] = None
    ):
        """Precompute terms for fast interpolation of state between start and stop times

        This can be used, for example, to compute sun and moon positions only once if propagating many satellites over the same time period

        Args:
            start (satkit.time): Start time of propagation
            stop (satkit.time): Stop time of propagation
            step (satkit.duration, optional): Step size for interpolation.  Default = 60 seconds

        """

def propagate(
    state: npt.NDArray[np.float64],
    start: time,
    stop: time,
    **kwargs,
) -> propresult:
    """High-precision orbit propagator

    Propagate orbits with high-precision force modeling via adaptive Runga-Kutta methods (default is order 9/8).

    Args:
        state (npt.ArrayLike[float], optional): 6-element numpy array representing satellite GCRF position and velocity in meters and meters/second
        start (satkit.time, optional): satkit.time object representing instant at which satellite is at "pos" & "vel"
        stop (satkit.time, optional keyword): satkit.time object representing instant at which new position and velocity will be computed
        duration (satkit.duration, optional keyword): duration from "start" at which new position & velocity will be computed.
        duration_secs (float, optional keyword): duration in seconds from "start" for at which new position and velocity will be computed.
        duration_days (float, optional keyword): duration in days from "start" at which new position and velocity will be computed.
        output_phi (bool, optional keyword): Output 6x6 state transition matrix between "starttime" and "stoptime" (and at intervals, if specified)
        propsettings (propsettings, optional keyword): "propsettings" object with input settings for the propagation. if left out, default will be used.
        satproperties (satproperties_static, optional keyword): "sat_properties_static" object with drag and radiation pressure succeptibility of satellite.

    Returns:
        (propresult): Propagation result object holding state outputs, statistics, and dense output if requested

    Notes:

    * Propagates statellite ephemeris (position, velocity in gcrs & time) to new time and output new position and velocity via Runge-Kutta integration.
    * Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)
    * Propagator uses advanced Runga-Kutta integrators and includes the following forces:
        * Earth gravity with higher-order zonal terms
        * Sun, Moon gravity
        * Radiation pressured
        * Atmospheric drag: NRL-MISE 2000 density model, with option to include space weather effects (which can be large)
    * Stop time must be set by keyword argument, either explicitely or by duration
    * Solid Earth tides are not (yet) included in the model

    """
