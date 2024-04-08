"""
Toolkit containing functions and classes used in satellite dynamics
calculations.
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit
import datetime
from . import jplephem
from . import frametransform
from . import moon
from . import sun
from . import satprop
from . import density

__all__ = [
    "time",
    "duration",
    "timescale",
    "quaternion",
    "sgp4",
    "gravmodel",
    "gravity",
    "nrlmsise00",
    "univ",
    "solarsystem",
    "TLE",
    "itrfcoord",
    "frametransform",
    "moon",
    "sun",
    "satprop",
    "jplephem",
    "utils",
    "density",
]

class TLE:
    """
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
    def from_file(filename: str) -> list[satkit.TLE] | satkit.TLE:
        """
        Return a list of TLES loaded from input text file.

        If the file contains lines only represent a single TLE, the TLE will
        be output, rather than a list with a single TLE element

        # Arguments:

        * `filename` - name of textfile lines for TLE(s) to load

        # Returns:

        * `tle` - a list of TLE objects or a single TLE of lines for
                only 1 are passed in
        """

    @staticmethod
    def from_lines(lines: list[str]) -> list[satkit.TLE] | satkit.TLE:
        """
        Return a list of TLES loaded from input list of lines

        If the file contains lines only represent a single TLE, the TLE will
        be output, rather than a list with a single TLE element

        # Arguments:

        * `lines` - list of strings with lines for TLE(s) to load

        # Returns:

        * `tle` - a list of TLE objects or a single TLE of lines for
                only 1 are passed in
        """

    @property
    def satnum(self) -> int:
        """
        Satellite number, or equivalently the NORAD ID
        """

    @property
    def eccen(self) -> float:
        """
        Satellite eccentricity, in range [0,1]
        """

    @property
    def mean_anomaly(self) -> float:
        """
        Mean anomaly in degrees
        """

    @property
    def mean_motion(self) -> float:
        """
        Mean motion in revs / day
        """

    @property
    def inclination(self) -> float:
        """
        Inclination, in degrees
        """

    @property
    def epoch(self) -> satkit.time:
        """
        TLE epoch
        """

    @property
    def arg_of_perigee(self) -> satkit.time:
        """
        Argument of Perigee, in degrees
        """

    @property
    def mean_motion_dot(self) -> float:
        """
        1/2 of first derivative of mean motion, in revs/day^2

        the "1/2" is because that is how number is stored in the TLE
        """

    @property
    def mean_motion_dot_dot(self) -> float:
        """
        1/6 of 2nd derivative of mean motion, in revs/day^3

        the "1/6" is because that is how number is stored in the TLE
        """

    @property
    def name(self) -> str:
        """
        The name of the satellite
        """

    @property
    def bstar(self) -> str:
        """
        "B Star" or drag of the satellite

        should be rho0 * Cd * A / 2 / m

        Units (which are strange) is multiples of
        1 / Earth radius
        """

def sgp4(
    tle: satkit.TLE | list[satkit.tle],
    tm: satkit.time | list[satkit.time] | npt.ArrayLike[satkit.time],
    **kwargs,
) -> tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]:
    """
    Run Simplified General Perturbations (SGP)-4 propagator on
    Two-Line Element Set to
    output satellite position and velocity at given time
    in the "TEME" coordinate system

    A detailed description is at:
    https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf


    # Arguments

    tle: The TLE (or a list of TLES) on which to operate

    tm: satkit.time object or list of objects or numpy array of
        objects representimg time(s) at which to compute
        position and velocity


    # Optional keyword arguments:

    `gravconst` -  satkit.sgp4_gravconst object indicating gravity constant to use
                   default is gravconst.wgs72

    `opsmode` -  satkit.sgp4_opsmode to use: opsmode.afspc (Air Force Space Command) or opsmode.improved
                 Default is opsmode.afspc

    `errflag` - bool indicating whether or not to output error conditions for each TLE and time output
                Default is false

    # Return

    tuple with the following elements:

    * `0` - Ntle X Ntime X 3 numpy array representing position in meters in the TEME frame at
            each of the "Ntime" input times and each of the "Ntle" tles
            Singleton dimensions (single time or single TLE) are removed

    * `1` - Ntle X Ntime X 3 numpy array representing velocity in meters / second in the TEME
            frame at each of the "Ntime" input times and each of the "Ntle" tles
            Singleton dimensions (single time or single TLE) are removed

    * `2`   Only output if `errflag` keyword is set to `True`:
            Ntle X Ntime numpy array represetnting error codes for each TLE and time
            Error codes are of type `satkit.sgp4_error`
            Singleton dimensions (single time or single TLE) are removed

    Example usage: show Geodetic position of satellite at TLE epoch

    lines = [
        "0 INTELSAT 902",
        "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
        "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981   9300."
    ]


    tle = satkit.TLE.single_from_lines(lines)

    # Compute TEME position & velocity at epoch
    pteme, vteme = satkit.sgp4(tle, tle.epoch)

    # Rotate to ITRF frame
    q = satkit.frametransform.qteme2itrf(tm)
    pitrf = q * pteme
    vitrf = q * vteme - np.cross(np.array([0, 0, satkit.univ.omega_earth]), pitrf)

    # convert to ITRF coordinate object
    coord = satkit.itrfcoord.from_vector(pitrf)
    # Print ITRF coordinate object location
    print(coord)

    Output:

    ITRFCoord(lat:  -0.0363 deg, lon:  -2.2438 deg, hae: 35799.51 km)

    """

class sgp4_gravconst:
    """
    Gravity constant to use for SGP4 propagation
    """

    @property
    def wgs72() -> int:
        """
        WGS-72
        """

    @property
    def wgs72old() -> int:
        """
        WGS-72 Old
        """

    @property
    def wgs84() -> int:
        """
        WGS-84
        """

class sgp4_opsmode:
    """
    Ops Mode for SGP4 Propagation
    """

    @property
    def afspc() -> int:
        """
        afspc (Air Force Space Command), the default
        """

    @property
    def improved() -> int:
        """
        Improved
        """

class gravmodel:
    """
    Earth gravity models available for use

    For details, see: http://icgem.gfz-potsdam.de/
    """

    @property
    def jgm3() -> int:
        """
        The "JGM3" gravity model

        This model is used by default in the orbit propagators
        """

    @property
    def jgm2() -> int:
        """
        The "JGM2" gravity model
        """

    @property
    def egm96() -> int:
        """
        The "EGM96" gravity model
        """

    @property
    def itugrace16() -> int:
        """
        the ITU Grace 16 gravity model
        """

def gravity(
    pos: list[float] | satkit.itrfcoord | npt.ArrayLike[np.float], **kwargs
) -> npt.ArrayLike[np.float]:
    """
    gravity(pos)
    --

    Return acceleration due to Earth gravity at the input position. The
    acceleration does not include the centrifugal force, and is output
    in m/s^2 in the International Terrestrial Reference Frame (ITRF)

    Inputs:

        pos:   Position as ITRF coordinate (satkit.itrfcoord) or numpy
                3-vector representing ITRF position in meters or
                list 3-vector representing ITRF position in meters

    Kwargs:

        model:   The gravity model to use.  Options are:
                    satkit.gravmodel.jgm3
                    satkit.gravmodel.jgm2
                    satkit.gravmodel.egm96
                    satkit.gravmodel.itugrace16

                Default is satkit.gravmodel.jgm3

                For details of models, see:
                http://icgem.gfz-potsdam.de/tom_longtime

        order:    The order of the gravity model to use.
                Default is 6, maximum is 16


                For details of calculation, see Chapter 3.2 of:
                "Satellite Orbits: Models, Methods, Applications",
                O. Montenbruck and B. Gill, Springer, 2012.

    """

def gravity_and_partials(
    pos: satkit.itrfcoord | npt.ArrayLike[np.float], **kwargs
) -> typing.Tuple[npt.ArrayLike[np.float], np.arrayLike[np.float]]:
    """
    gravity_and_partials(pos)
    --

    Return acceleration due to Earth gravity at the input position. The
    acceleration does not include the centrifugal force, and is output
    in m/s^2 in the International Terrestrial Reference Frame (ITRF)

    Also return partial derivative of acceleration with respect to
    ITRF Cartesian coordinate, in m/s^2 / m

    Inputs:

        pos:   Position as ITRF coordinate (satkit.itrfcoord) or numpy
                3-vector representing ITRF position in meters or
                list 3-vector representing ITRF position in meters

    Kwargs:

        model:   The gravity model to use.  Options are:
                    satkit.gravmodel.jgm3
                    satkit.gravmodel.jgm2
                    satkit.gravmodel.egm96
                    satkit.gravmodel.itugrace16

                Default is satkit.gravmodel.jgm3

                For details of models, see:
                http://icgem.gfz-potsdam.de/tom_longtime

        order:    The order of the gravity model to use.
                Default is 6, maximum is 16


                For details of calculation, see Chapter 3.2 of:
                "Satellite Orbits: Models, Methods, Applications",
                O. Montenbruck and B. Gill, Springer, 2012.

    """

class solarsystem:
    """
    Solar system bodies for which high-precision ephemeris can be computed
    """

    @property
    def Mercury() -> int:
        """
        Mercury
        """

    @property
    def Venus() -> int:
        """
        Venus
        """

    @property
    def EMB() -> int:
        """
        Earth-Moon Barycenter
        """

    @property
    def Mars() -> int:
        """
        Mars
        """

    @property
    def Jupiter() -> int:
        """
        Jupter
        """

    @property
    def Saturn() -> int:
        """
        Saturn
        """

    @property
    def Uranus() -> int:
        """
        Uranus
        """

    @property
    def Neptune() -> int:
        """
        Neptune
        """

    @property
    def Pluto() -> int:
        """
        Pluto
        """

    @property
    def Moon() -> int:
        """
        Moon
        """

    @property
    def Sun() -> int:
        """
        Sun
        """

class sgp4error:
    """
    Represent errors from SGP-4 propagation of two-line element sets (TLEs)
    """

    @property
    def success() -> int:
        """
        Success
        """

    @property
    def eccen() -> int:
        """
        Eccentricity < 0 or > 1
        """

    @property
    def mean_motion() -> int:
        """
        Mean motion (revs / day) < 0
        """

    @property
    def perturb_eccen() -> int:
        """
        Perturbed eccentricity < 0 or > 1
        """

    @property
    def semi_latus_rectum() -> int:
        """
        Semi-Latus Rectum < 0
        """

    @property
    def unused() -> int:
        """
        Unused, but in base code, so keeping for completeness
        """

    @property
    def orbit_decay() -> int:
        """
        Orbit decayed
        """

class timescale:
    """
    Specify time scale used to represent or convert between the "satkit.time"
    representation of time

    Most of the time, these are not needed directly, but various time scales
    are needed to compute precise rotations between various inertial and
    Earth-fixed coordinate frames

    For an excellent overview, see:
    https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/MPG%20Book/Release/Chapter2-TimeScales.pdf

    # Options:

    * Invalid: Invalid time scale
    * UTC: Universal Time Coordinate
    * TT: Terrestrial Time
    * UT1: UT1
    * TAI: International Atomic Time
    * GPS: Global Positioning System (GPS) time
    * TDB: Barycentric Dynamical Time    
    """

    @property
    def Invalid() -> int:
        """
        Invalid time scale
        """

    @property
    def UTC() -> int:
        """
        Universal Time Coordinate
        """

    def TT() -> int:
        """
        Terrestrial Time
        """

    def UT1() -> int:
        """
        UT1
        """

    def TAI() -> int:
        """
        International Atomic Time
        (nice because it is monotonically increasing)
        """

    def GPS() -> int:
        """
        Global Positioning System (GPS) time
        """

    def TDB() -> int:
        """
        Barycentric Dynamical Time
        """

class time:
    """
    Representation of an instant in time

    This has functionality similar to the "datetime" object, and in fact has
    the ability to convert to an from the "datetime" object.  However, a separate
    time representation is needed as the "datetime" object does not allow for
    conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)

    Initialization arguments:

    If no arguments are passed in, the created object represents
    the current time, i.e. the time at which the function was called


    If 3 integers are passed in, they represent a UTC date specified
    by the standard Gregorian year, month (1-based), and day of month
    (1-based)

    if 5 integers and a float are passed in, they represent a UTC
    date and time.  The 1st 3 numbers represent the standard
    Gregorian year, month, and day as above.  The last 3 represent the
    hour of the day [0,23], the minute of the hour [0,59], and the
    second (including fractional component) of the minute

    Example 1:
    print(satkit.time(2023, 3, 5, 11, 3,45.453))
    2023-03-05 11:03:45.453Z

    Example 2:
    print(satkit.time(2023, 3, 5))
    2023-03-05 00:00:00.000Z

    """

    def __init__(self, *args):
        """
        Create a "time" object.


        If no arguments are passed in, the created object represents
        the current time, i.e. the time at which the function was called


        If 3 integers are passed in, they represent a UTC date specified
        by the standard Gregorian year, month (1-based), and day of month
        (1-based)

        if 5 integers and a float are passed in, they represent a UTC
        date and time.  The 1st 3 numbers represent the standard
        Gregorian year, month, and day as above.  The last 3 represent the
        hour of the day [0,23], the minute of the hour [0,59], and the
        second (including fractional component) of the minute

        Example 1:
        print(satkit.time(2023, 3, 5, 11, 3,45.453))
        2023-03-05 11:03:45.453Z

        Example 2:
        print(satkit.time(2023, 3, 5))
        2023-03-05 00:00:00.000Z

        """

    @staticmethod
    def now() -> satkit.time:
        """
        Create a "time" object representing the instant of time at the
        calling of the function.
        """

    @staticmethod
    def from_date(year: int, month: int, day: int) -> satkit.time:
        """
        Returns a time object representing the start of the day (midnight)
        on the provided date, specified by standard Gregorian year, month
        (1-based), and day of month (1-based)
        """

    @staticmethod
    def from_jd(jd: float, scale: satkit.timescale) -> satkit.time:
        """
        Return a time object representing input Julian date and time scale
        """

    @staticmethod
    def from_mjd(mjd: float, scale: satkit.timescale) -> satkit.time:
        """
        Return a time object representing input modified Julian date and time scale
        """

    def to_date(self) -> typing.Tuple[int, int, int]:
        """
        Return tuple representing as UTC Gegorian date of the
        time object.  Tuple has 6 elements:
        1 : Gregorian Year
        2 : Gregorian month (1 = January, 2 = February, ...)
        3 : Day of month, beginning with 1

        Fractional day components are neglected
        """

    @staticmethod
    def from_gregorian(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        min: int,
        sec: float,
        scale: satkit.timescale = satkit.timescale.UTC,
    ) -> satkit.time:
        """
        Create time object from 6 input arguments representing
        UTC Gregorian time.

        Inputs are:
        1 : Gregorian Year
        2 : Gregorian month (1 = January, 2 = February, ...)
        3 : Day of month, beginning with 1
        4 : Hour of day, in range [0,23]
        5 : Minute of hour, in range [0,59]
        6 : floating point second of minute, in range [0,60)
        7 : Time scale. Optional, default is satkit.timescale.UTC

        Example:
        print(satkit.time.from_gregorian(2023, 3, 5, 11, 3,45.453))
        2023-03-05 11:03:45.453Z
        """

    def to_gregorian(
        self, scale=satkit.timescale.UTC
    ) -> typing.Tuple[int, int, int, int, int, float]:
        """
        Return tuple representing as UTC Gegorian date and time of the
        time object.  Tuple has 6 elements:
        1 : Gregorian Year
        2 : Gregorian month (1 = January, 2 = February, ...)
        3 : Day of month, beginning with 1
        4 : Hour of day, in range [0,23]
        5 : Minute of hour, in range [0,59]
        6 : floating point second of minute, in range [0,60)

        Optional single input is satkit.timescale representing time scale of gregorian
        output.  Default is satkit.timescale.UTC
        """

    @staticmethod
    def from_datetime(dt: datetime.datetime) -> satkit.time:
        """
        Convert input "datetime.datetime" object to an
        "satkit.time" object represenging the same
        instant in time
        """

    def datetime(self, utc: bool = True) -> datetime.datetime:
        """
        Convert object to "datetime.datetime" object representing
        same instant in time.

        The optional boolean input "utc" specifies wither to make the
        "daettime.datetime" object represent time in the local timezone,
        or whether to have the "datetime.datetime" object be in "UTC" time.
        Default is true

        Example: (from Easterm Standard Time time zone)
        dt = satkit.time(2023, 6, 3, 6, 19, 34).datetime(True)
        print(dt)
        2023-06-03 06:19:34+00:00

        dt = satkit.time(2023, 6, 3, 6, 19, 34).datetime(False)
        print(dt)
        2023-06-03 02:19:34
        """

    def to_mjd(self, scale: satkit.timescale = satkit.timescale.UTC) -> float:
        """
        Represent time instance as a Modified Julian Date
        with the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC
        """

    def to_jd(self, scale: satkit.timescale = satkit.timescale.UTC) -> float:
        """
        Represent time instance as Julian Date with
        the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC
        """

    def to_unixtime(self) -> float:
        """
        Represent time as unixtime

        (seconds since Jan 1, 1970 UTC)

        Includes fractional comopnent of seconds
        """

    def __add__(
        self,
        other: (
            satkit.duration
            | npt.ArrayLike[float]
            | float
            | list[float]
            | npt.ArrayLike[satkit.duration]
        ),
    ) -> satkit.time | npt.ArrayLike[satkit.time]:
        """
        Return an satkit.time object or nunpy array of satkit.time objects
        representing the input "added to" the current object

        # Possible inputs and corresponding outputs:

        *  `float` - return satkit.time object incremented by input number of days
        * `satkit.duration` - return satkit.time object incremented by duration
        * `list[float]`  - return numpy array of satkit.time objects, representing
           an element-wise addition of days to the "self"
        * `list[duration]` -  reuturn numpy array of satkit.time objects, with each
           object representing an element-wise addition of "duration" to the "self"
        * `numpy.array(float)` - return numpy array of satkit.time objects, with each
           object representing an element-wise addition of days to the "self"
        """

    def __sub__(
        self,
        other: (
            satkit.duration
            | satkit.time
            | npt.ArrayLoke[float]
            | npt.ArrayLike[satkit.duration]
            | list[float]
        ),
    ) -> satkit.time | satkit.duration | npt.ArrayLike[satkit.time]:
        """
        Return an satkit.time object or numpy array of satkit.time objects
        representing the input "subtracted from" the current object

        # Possible inputs and corresponding outputs:

        * `satkit.time` - output is duration representing the difference
           between the "other" time and "self"
        * `satkit.duration` - output is satkit.time object representing time minus
           the input duration
        * `list[float]` - return numpy array of satkit.time objects, representing
           an element-wise subtraction of days to the "self"
        * `list[duration]` - return numpy array of satkit.time objects, representing
           an element-wise subtraction of "duration" from the "self"
        * `numpy.array(float)` - return numpy array of satkit.time objects, with
           each object representing an element-wise subtraction of days from
           the "self".

        """

class duration:
    """
    Representation of a time duration
    """

    @staticmethod
    def from_days(d: float) -> duration:
        """
        Create duration object given input number of days
        Note: a day is defined as 86,400 seconds
        """

    @staticmethod
    def from_seconds(d: float) -> duration:
        """
        Create duration object representing input number of seconds
        """

    @staticmethod
    def from_minutes(d: float) -> duration:
        """
        Create duration object representing input number of minutes
        """

    @staticmethod
    def from_hours(d: float) -> duration:
        """
        Create duration object representing input number of hours
        """

    def __add__(self, other: duration | satkit.time) -> duration | satkit.time:
        """
        Add a duration to either another duration or a time

        if "other" is a duration, output is a duration representing the
        sum, or concatenation, of both durations

        if "other" is a time, output is a time representing the input
        time plus the duration

        # Example 1:
        
        print(duration.from_hours(1) + duration.from_minutes(1))
        Duration: 1 hours, 1 minutes, 0.000 seconds

        # Example 2:
        
        print(duration.from_hours(1) + satkit.time(2023, 6, 4, 11,30,0))
        2023-06-04 13:30:00.000Z

        """

    def __sub__(self, other: duration) -> duration:
        """
        Take the difference between two durations
        example:

        print(duration.from_hours(1) - duration.from_minutes(1))
        Duration: 59 minutes, 0.000 seconds

        """

    def __mul__(self, other: float) -> duration:
        """
        Multiply (or scale) duration by given value

        Example:
        print(duration.from_days(1) * 2.5)
        Duration: 2 days, 12 hours, 0 minutes, 0.000 seconds
        """

    def days(self) -> float:
        """
        Floating point number of days represented by duration
        """

    def hours(self) -> float:
        """
        Floating point number of hours represented by duration
        """

    def minutes(self) -> float:
        """
        Floating point number of minutes represented by duration
        """

    def seconds(self) -> float:
        """
        Floating point number of seconds represented by duration
        """

class quaternion:
    """
    Quaternion representing rotation of 3D Cartesian axes

    Quaternion is right-handed rotation of a vector,
    e.g. rotation of +xhat 90 degrees by +zhat give +yhat

    This is different than the convention used in Vallado, but
    it is the way it is commonly used in mathematics and it is
    the way it should be done.

    For the uninitiated: quaternions are a more-compact and
    computationally efficient way of representing 3D rotations.
    They can also be multipled together and easily renormalized to
    avoid problems with floating-point precision eventually causing
    changes in the rotated vecdtor norm.

    For details, see:

    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Under the hood, this is using the "UnitQuaternion" object in the
    rust "nalgebra" crate.
    """

    def __init__(self):
        """
        Return unit quaternion (no rotation)
        """

    @staticmethod
    def from_axis_angle(
        axis: npt.ArrayLike[np.float64], angle: float
    ) -> satkit.quaternion:
        """
        Return quaternion representing right-handed rotation by
        "angle" degrees about the given axis.  The axis does not
        have to be normalized.
        """

    @staticmethod
    def from_rotation_matrix(
        mat: npt.ArrayLike[np.float64],
    ) -> satkit.quaternion:
        """
        Return quaternion representing right-handed rotation
        represented by input 3x3 rotation matrix
        """

    @staticmethod
    def rotx(theta) -> satkit.quaternion:
        """
        Return quaternion representing right-handed rotation of vector
        by "theta" radians about the xhat unit vector

        Equivalent rotation matrix:
        | 1             0            0|
        | 0    cos(theta)   sin(theta)|
        | 0   -sin(theta)   cos(theta)|
        """

    @staticmethod
    def roty(theta) -> satkit.quaternion:
        """
        Return quaternion representing right-handed rotation of vector
        by "theta" radians about the yhat unit vector

        Equivalent rotation matrix:
        | cos(theta)     0   -sin(theta)|
        |          0     1             0|
        | sin(theta)     0    cos(theta)|
        """

    @staticmethod
    def rotz(theta) -> satkit.quaternion:
        """
        Return quaternion representing right-handed rotation of vector
        by "theta" radians about the zhat unit vector

        Equivalent rotation matrix:
        |  cos(theta)      sin(theta)   0|
        | -sin(theta)      cos(theta)   0|
        |           0               0   1|
        """

    @staticmethod
    def rotation_between(
        v1: npt.ArrayLike[np.float64], v2: npt.ArrayLike[np.float64]
    ) -> satkit.quaternion:
        """
        Return quaternion represention rotation from V1 to V2

        # Arguments:

        * `v1` - vector rotating from
        * `v2` - vector rotating to

        # Returns:

        * Quaternion that rotates from v1 to v2
        """

    def to_rotation_matrix(self) -> npt.ArrayLike[np.float64]:
        """
        Return 3x3 rotation matrix representing equivalent rotation
        """

    def to_euler(self) -> typing.Tuple[float, float, float]:
        """
        Return equivalent rotation angle represented as rotation angles:
        ("roll", "pitch", "yaw") in radians:

        * roll = rotation about x axis
        * pitch = rotation about y axis
        * yaw = rotation about z axis
        """

    def angle(self) -> float:
        """
        Return the angle in radians of the rotation
        """

    def axis(self) -> npt.ArrayLike[np.float64]:
        """
        Return the axis of rotation as a unit vector
        """

    def conj(self) -> satkit.quaternion:
        """
        Return conjucate or inverse of the rotation
        """

    def conjugate(self) -> satkit.quaternion:
        """
        Return conjugate or inverse of the rotation
        """

    @typing.overload
    def __mul__(self, other: satkit.quaternion) -> satkit.quaternion:
        """
        Multiply represents concatenation of two rotations representing
        the quaternions.  The left value rotation is applied after
        the right value, per the normal convention
        """

    @typing.overload
    def __mul__(self, other: npt.ArrayLike[np.float64]) -> npt.ArrayLike[np.float64]:
        """
        Multply by a vector to rotate the vector

        The vector is represented as a numpy array

        If the array is 1 demensional it must have 3 elements

        If the array is 2 dimensionsl and the dimensions are Nx3,
        each of the "N" vectors is rotated by the quaternion and a
        Nx3 array is returned
        """

    def slerp(
        self, other: satkit.quaternion, frac: float, epsilon: float = 1.0e-6
    ) -> satkit.quaternion:
        """
        Spherical linear interpolation between self and other

        # Arguments:

        * `other` - Quaternion to perform interpolation to
        * `frac` - fractional amount of interpolation, in range [0,1]
        * `epsilon` - Value below which the sin of the angle separating both
                    quaternions must be to return an error.
                    Default is 1.0e-6

        # Returns

         * Quaternion representing interpolation between self and other

        """

class itrfcoord:
    """
    Representation of a coordinate in the
    International Terrestrial Reference Frame (ITRF)

    This coordinate object can be created from and also
    output to Geodetic coordinates (latitude, longitude,
    height above ellipsoid).

    Functions are also available to provide rotation
    quaternions to the East-North-Up frame
    and North-East-Down frame at this coordinate

    Coordinate from "Cartesian" inputs can be set via folloing:

        1: single 3-element list of floats representing ITRF Cartesian location in meters
        2: single 3-element numpy array of floats representing ITRF Cartesian location in meters
        3. 3 separate input arguments representing x,y,z ITRF Cartesian location in meters

    Input can also be set from geodetic coordinate using kwargs
        Optional kwargs:

        latitude_deg: latitude, degrees
        longitude_deg: longitude, degrees
        latitude_rad: latitude, radians
        longitude_rad: longitude, radians
        altitude: height above ellipsoid, meters
        height: height above ellipsoid, meters

        If altitude is not specified, default is 0

    Output:

        New "ITRF" Coordinate

    Examples:

        1. Create ITRF coord from cartesian

        coord = itrfcoord([ 1523128.63570828 -4461395.28873207  4281865.94218203 ])
        print(coord)

        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)

        2. Create same ITRF coord from geodetic

        coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)

        print(coord)

        ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
    """

    def __init__(self, *args, **kwargs):
        """
        Create a coordinate in the ITRF (International Terrestrial Reference Frame)


        Coordinate from "Cartesian" inputs can be set via folloing:

            1: single 3-element list of floats representing ITRF Cartesian location in meters
            2: single 3-element numpy array of floats representing ITRF Cartesian location in meters
            3. 3 separate input arguments representing x,y,z ITRF Cartesian location in meters

        Input can also be set from geodetic coordinate using kwargs
            Optional kwargs:

            latitude_deg: latitude, degrees
            longitude_deg: longitude, degrees
            latitude_rad: latitude, radians
            longitude_rad: longitude, radians
            altitude: height above ellipsoid, meters
            height: height above ellipsoid, meters

            Note that there are 2 ways to specify latitude, longitude, and altitude
            All 3 must be specified or an error will be triffered

        Output:

            New "ITRF" Coordinate

        Examples:

            1. Create ITRF coord from cartesian

            coord = itrfcoord([ 1523128.63570828 -4461395.28873207  4281865.94218203 ])
            print(coord)

            ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)

            2. Create same ITRF coord from geodetic

            coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)

            print(coord)

            ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
        """

    @property
    def latitude_deg(self) -> float:
        """
        Latitude in degrees
        """

    @property
    def longitude_deg(self) -> float:
        """
        Longitude in degrees
        """

    @property
    def latitude_rad(self) -> float:
        """
        Latitude in radians
        """

    @property
    def longitude_rad(self) -> float:
        """
        Longitude in radians
        """

    @property
    def altitude(self) -> float:
        """
        Altitude above ellipsoid, in meters
        """

    @property
    def geodetic_rad(self) -> typing.Tuple[float, float, float]:
        """
        Tuple with: (latitude_rad, longitude_rad, altitude)
        """

    @property
    def geodetic_deg(self) -> typing.Tuple[float, float, float]:
        """
        Tuple with (latitude_deg, longitude_deg, altitude)
        """

    @property
    def vector(self) -> npt.NDArray[np.float64]:
        """
        Cartesian ITRF coord as numpy array
        """

    @property
    def qned2itrf(self) -> quaternion:
        """
        Quaternion representing rotation from North-East-Down (NED)
        to ITRF at this location
        """

    @property
    def qenu2itrf(self) -> quaternion:
        """
        Quaternion representiong rotation from East-North-Up (ENU)
        to ITRF at this location
        """

    def geodesic_distance(self, other: itrfcoord) -> typing.Tuple[float, float, float]:
        """
        Use Vincenty formula to compute geodesic distance:
        https://en.wikipedia.org/wiki/Vincenty%27s_formulae

        Return a tuple with:

        1: geodesic distance (shortest distance between two points)
        between this coordinate and given coordinate, in meters

        2: initial heading, in radians

        3. heading at destination, in radians
        """

    def move_with_heading(self, distance: float, heading_rad: float) -> itrfcoord:
        """
        Takes two input arguments:

        1) distance (meters)
        2) heading (rad)

        Return itrfcoord representing move along Earth surface by given distance
        in direction given by heading

        Altitude is assumed to be zero

        Use Vincenty formula to compute position:
        https://en.wikipedia.org/wiki/Vincenty%27s_formulae
        """

class consts:
    """
    Some constants that are useful for saetllite dynamics
    """

    @property
    def wgs84_a() -> float:
        """
        WGS-84 semiparameter, in meters
        """

    @property
    def wgs84_f() -> float:
        """
        WGS-84 flattening in meters
        """

    @property
    def earth_radius() -> float:
        """
        Earth radius along major axis, meters
        """

    @property
    def mu_earth() -> float:
        """
        Gravitational parameter of Earth, m^3/s^2
        """

    @property
    def mu_moon() -> float:
        """
        Gravitational parmaeter of Moon, m^3/s^2
        """

    @property
    def mu_sun() -> float:
        """
        Gravitational parameter of sun, m^3/s^2
        """

    @property
    def GM() -> float:
        """
        Gravitational parameter of Earth, m^3/s^2
        """

    @property
    def omega_earth() -> float:
        """
        Scalar Earth rotation rate, rad/s
        """

    @property
    def c() -> float:
        """
        Speed of light, m/s
        """

    @property
    def au() -> float:
        """
        Astronomical Unit, mean Earth-Sun distance, meters
        """

    @property
    def sun_radius() -> float:
        """
        Radius of sun, meters
        """

    @property
    def moon_radius() -> float:
        """
        Radius of moon, meters
        """

    @property
    def earth_moon_mass_ratio() -> float:
        """
        Earth mass over Moon mass, unitless
        """

    @property
    def geo_r() -> float:
        """
        Distance to Geosynchronous orbit from Earth center, meters
        """

    @property
    def jgm3_mu() -> float:
        """
        Earth gravitational parameter from JGM3 gravity model, m^3/s^2
        """

    @property
    def jgm3_a() -> float:
        """
        Earth semiparameter from JGM3 gravity model, m
        """

    @property
    def jgm3_j2() -> float:
        """
        "J2" gravity due oblateness of Earth from JGM3 gravity model,
        unitless
        """
