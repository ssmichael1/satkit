"""
Toolkit containing functions and classes used in satellite dynamics
calculations.
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import datetime

from collections.abc import Sequence
from typing import Any, ClassVar, Optional, Union, overload

class TLE:
    """Two-Line Element Set (TLE) representing a satellite ephemeris

    Structure representing a Two-Line Element Set (TLE), a satellite
    ephemeris format from the 1970s that is still somehow in use
    today and can be used to calculate satellite position and
    velocity in the "TEME" frame (not-quite GCRF) using the
    "Simplified General Perturbations-4" (SGP-4) mathematical
    model that is also included in this package.

    For details, see: <https://en.wikipedia.org/wiki/Two-line_element_set>

    The TLE format is still commonly used to represent satellite
    ephemerides, and satellite ephemerides catalogs in this format
    are publicly available at <https://www.space-track.org> (registration
    required) and <https://celestrak.org> (no registration needed).

    TLEs sometimes have a "line 0" that includes the name of the satellite
    """

    @staticmethod
    def from_file(filename: str) -> list[TLE] | TLE:
        """Load TLEs from input text file
        Return a list of TLES loaded from input text file.

        If the file contains lines only represent a single TLE, the TLE will
        be output, rather than a list with a single TLE element

        Args:
              filename (str): name of text file lines for TLE(s) to load

        Returns:
            a list of TLE objects or a single TLE if lines for
                only 1 are passed in

        Example:
            ```python
            tles = satkit.TLE.from_file("gps-ops.txt")
            for tle in tles:
                print(tle.name, tle.satnum)
            ```
        """
        ...

    @overload
    @staticmethod
    def from_lines(lines: tuple[str, str]) -> TLE: ...
    @overload
    @staticmethod
    def from_lines(lines: tuple[str, str, str]) -> TLE: ...
    @overload
    @staticmethod
    def from_lines(lines: Sequence[str]) -> list[TLE] | TLE: ...
    @staticmethod
    def from_lines(lines: Sequence[str]) -> list[TLE] | TLE:
        """Return a list of TLES loaded from input list of lines

            If the file contains lines only represent a single TLE, the TLE will
            be output, rather than a list with a single TLE element.

            A fixed-length tuple of 2 or 3 strings (the 2-line or name+2-line
            form) is statically known to return a single ``TLE``; any other
            sequence may return either ``TLE`` or ``list[TLE]`` depending on
            how many TLEs the input contains.

        Args:
            lines (Sequence[str]): sequence of strings with lines for TLE(s) to load
                (any sequence type is accepted, e.g. list or tuple)

        Returns:
            a list of TLE objects or a single TLE if lines for
                only 1 are passed in

        Example:
            ```python
            lines = (
                "0 ISS (ZARYA)",
                "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9991",
                "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302878",
            )
            tle = satkit.TLE.from_lines(lines)  # statically typed as TLE
            print(tle.name)
            # ISS (ZARYA)
            ```
        """
        ...

    @staticmethod
    def from_url(url: str) -> list[TLE] | TLE:
        """Load TLE(s) from a URL

        Fetches the content at the given URL and parses it as TLE lines.
        Works with any URL that returns plain-text TLE data.

        Args:
            url (str): URL to fetch TLE data from

        Returns:
            Single TLE or list of TLEs parsed from the response

        Example:
            ```python
            tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
            ```
        """
        ...

    @property
    def satnum(self) -> int:
        """Satellite number, or equivalently the NORAD ID"""
        ...

    @satnum.setter
    def satnum(self, value: int) -> None:
        """Set the satellite number (NORAD ID)"""
        ...

    @property
    def raan(self) -> float:
        """Right Ascension of Ascending Node, in degrees"""
        ...

    @raan.setter
    def raan(self, value: float) -> None:
        """Set the Right Ascension of Ascending Node, in degrees"""
        ...

    @property
    def eccen(self) -> float:
        """Satellite eccentricity, in range [0,1]"""
        ...

    @eccen.setter
    def eccen(self, value: float) -> None:
        """Set the satellite eccentricity"""
        ...

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly in degrees"""
        ...

    @mean_anomaly.setter
    def mean_anomaly(self, value: float) -> None:
        """Set the satellite mean anomaly"""
        ...

    @property
    def mean_motion(self) -> float:
        """Mean motion in revs / day"""
        ...

    @mean_motion.setter
    def mean_motion(self, value: float) -> None:
        """Set the satellite mean motion"""
        ...

    @property
    def inclination(self) -> float:
        """Inclination, in degrees"""
        ...

    @inclination.setter
    def inclination(self, value: float) -> None:
        """Set the satellite inclination, degrees"""
        ...

    @property
    def epoch(self) -> time:
        """TLE epoch"""
        ...

    @epoch.setter
    def epoch(self, value: time) -> None:
        """Set the TLE epoch"""
        ...

    @property
    def arg_of_perigee(self) -> float:
        """Argument of Perigee, in degrees"""
        ...

    @arg_of_perigee.setter
    def arg_of_perigee(self, value: float) -> None:
        """Set the argument of perigee, degrees"""
        ...

    @property
    def mean_motion_dot(self) -> float:
        """1/2 of first derivative of mean motion, in revs/day^2

        Notes:
            The "1/2" is because that is how number is stored in the TLE.
        """
        ...

    @mean_motion_dot.setter
    def mean_motion_dot(self, value: float) -> None:
        """Set the 1/2 of first derivative of mean motion, in revs/day^2"""
        ...

    @property
    def mean_motion_dot_dot(self) -> float:
        """1/6 of 2nd derivative of mean motion, in revs/day^3

        Notes:
            The "1/6" is because that is how number is stored in the TLE.

        """
        ...

    @mean_motion_dot_dot.setter
    def mean_motion_dot_dot(self, value: float) -> None:
        """Set the 1/6 of 2nd derivative of mean motion, in revs/day^3"""
        ...

    @property
    def name(self) -> str:
        """The name of the satellite"""
        ...

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the satellite"""
        ...

    @property
    def bstar(self) -> float:
        """Drag of the satellite

        should be rho0 * Cd * A / 2 / m

        Units (which are strange) is multiples of
        1 / Earth radius
        """
        ...

    @bstar.setter
    def bstar(self, value: float) -> None:
        """Set the drag of the satellite"""
        ...

    def to_2line(self) -> list[str]:
        """
        Output as 2 canonical TLE Lines

        Returns:
            2 canonical TLE Lines

        Example:
            ```python
            lines = tle.to_2line()
            print(lines[0])
            # 1 25544U 98067A  ...
            print(lines[1])
            # 2 25544  51.6442 ...
            ```
        """
        ...

    def to_3line(self) -> list[str]:
        """
        Output as 2 canonical TLE lines preceded by a name line (3-line element set)

        Returns:
            3-line element set, name line then 2 canonical TLE lines

        Example:
            ```python
            lines = tle.to_3line()
            for line in lines:
                print(line)
            ```
        """
        ...

    @staticmethod
    def fit_from_states(
        states: list[np.ndarray],
        times: list[time] | list[datetime.datetime],
        epoch: time | datetime.datetime,
    ) -> tuple[TLE, dict]:
        """
        Perform non-linear least squares fit of TLE parameters to a list of GCRF states

        Args:
            states: List of GCRF states to fit to. Each state is a 6-element vector. The first 3 values are positions in meters. The last 3 values are velocities in meters / second.
            times: List of times corresponding to the states
            epoch: Epoch time for the TLE. Must be within range of times.

        Returns:
            Fitted TLE and fitting results in a dictionary

        Notes:
            SGP4 propagator is used to match TLE to the states.
            Input GCRF states are rotated into TEME frame used by SGP4.
            First and second derivatives of mean motion are ignored, as they are not used by SGP4.

            Non-linear Levenberg-Marquardt optimization is performed to fit
            inclination, eccentricity, RAAN, argument of perigee, mean anomaly,
            mean motion, and drag (bstar) to the provided states. The solver
            is built on top of the ``numeris`` linear algebra crate.

            The results dictionary includes the following keys:
            ``status`` (a :class:`tlefitstatus`), ``converged`` (bool),
            ``orig_norm``, ``best_norm``, ``grad_norm``, ``n_iter``,
            ``n_res_evals``.

        Example:
            ```python
            import numpy as np

            # Given a list of GCRF states and times
            states = [np.array([pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2]])]
            times = [satkit.time(2024, 1, 1)]
            epoch = satkit.time(2024, 1, 1)

            tle, results = satkit.TLE.fit_from_states(states, times, epoch)
            if results["converged"]:
                print("Fit successful")
            ```
        """
        ...

def sgp4(
    tle: TLE | list[TLE] | dict,
    tm: time | list[time] | list[datetime.datetime] | npt.ArrayLike,
    **kwargs,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """SGP-4 propagator for TLE

    Run Simplified General Perturbations (SGP)-4 propagator on Two-Line Element Set to
    output satellite position and velocity at given time
    in the "TEME" coordinate system.

    A detailed description is at:
    <https://celestrak.org/publications/AIAA/2008-6770/AIAA-2008-6770.pdf>

    Args:
        tle (TLE | list[TLE] | dict): TLE or OMM (or list of TLES) on which to operate
        tm (time | list[time] | list[datetime.datetime] | npt.ArrayLike[time] | npt.ArrayLike[datetime.datetime]): time(s) at which to compute position and velocity

    Keyword Args:
        gravconst (satkit.sgp4_gravconst): gravity constant to use.  Default is gravconst.wgs72
        opsmode (satkit.sgp4_opsmode): opsmode.afspc (Air Force Space Command) or opsmode.improved.  Default is opsmode.afspc
        errflag (bool): whether or not to output error conditions for each TLE and time output.  Default is False
                        (this is likely rarely needed, but can be useful for debugging)
                        (this may also flag a typing error ... I can't figure out how to get rid of it)

    Returns:
        position and velocity
            in meters and meters/second, respectively,
            in the TEME frame at each of the "Ntime" input times and each of the "Ntle" tles.
            Additional return value if errflag is True:
            list[sgp4_error] with error conditions for each TLE and time output.

    Notes:
        - Now supports propagation of OMM (Orbital Mean-Element Message) dictionaries.
          The dictionaries must follow the structure used by <https://www.celestrak.org> or
          <https://www.space-track.org.>
        - The "TEME" frame of the SGP4 state vectors is not a truly inertial frame.  It is a "True Equator Mean Equinox"
          frame, which is a non-rotating frame with respect to the mean equator and mean equinox of the epoch of the TLE.
          It is close to a true inertial frame, but can be offset by small amounts due to precession and nutation.

    Example:
        ```python
        lines = [
               "0 INTELSAT 902",
            "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981"
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
        # ITRFCoord(lat:  -0.0363 deg, lon:  -2.2438 deg, hae: 35799.51 km)
        ```


        ```python
        import requests
        import json

        # Query ephemeris for the International Space Station (ISS)
        url = '<https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=json'>
        with requests.get(url) as response:
            omm = response.json()
        # Get a representative time from the output
        epoch = sk.time(omm[0]['EPOCH'])
        # Compute TEME position & velocity at epoch
        pteme, vteme = satkit.sgp4(omm[0], epoch)
        ```

    """
    ...

class sgp4_gravconst:
    """Gravity constant to use for SGP4 propagation"""

    wgs72: ClassVar[sgp4_gravconst]
    """WGS-72"""

    wgs72old: ClassVar[sgp4_gravconst]
    """WGS-72 Old"""

    wgs84: ClassVar[sgp4_gravconst]
    """WGS-84"""

class sgp4_opsmode:
    """Ops Mode for SGP4 Propagation"""

    afspc: ClassVar[int]
    """afspc (Air Force Space Command), the default"""

    @property
    def improved(self) -> int:
        """Improved"""
        ...

class gravmodel:
    """
    Earth gravity models available for use

    For details, see: <http://icgem.gfz-potsdam.de/>
    """

    jgm3: ClassVar[gravmodel]
    """
    The "JGM3" gravity model

    This model is used by default in the orbit propagators
    """

    jgm2: ClassVar[gravmodel]
    """
    The "JGM2" gravity model
    """

    egm96: ClassVar[gravmodel]
    """
    The "EGM96" gravity model
    """

    itugrace16: ClassVar[gravmodel]
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
        model (gravmodel): The gravity model to use.  Default is gravmodel.egm96
        degree (int): Maximum degree of gravity model to use.  Default is 6, maximum is 40
        order (int): Maximum order of gravity model to use.  Default is same as degree

    Returns:
        acceleration in m/s^2 in the International Terrestrial Reference Frame (ITRF)


    Notes:
        - For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.

    Example:
        ```python
        coord = satkit.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=0)
        accel = satkit.gravity(coord)
        print(accel)
        # array with acceleration in m/s^2 in ITRF
        ```
    """
    ...

def gravity_and_partials(
    pos: itrfcoord | npt.NDArray[np.float64], **kwargs
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gravity and partial derivatives of gravity with respect to Cartesian coordinates

    Args:
        pos (itrfcoord | npt.ArrayLike[np.float]): Position as ITRF coordinate or numpy 3-vector representing ITRF position in meters


    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.egm96
        degree (int): Maximum degree of gravity model to use.  Default is 6, maximum is 40
        order (int): Maximum order of gravity model to use.  Default is same as degree

    Returns:
        acceleration in m/s^2 and partial derivative of acceleration with respect to ITRF Cartesian coordinate in m/s^2 / m


    For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and B. Gill, Springer, 2012.

    """
    ...

class solarsystem:
    """Solar system bodies for which high-precision ephemeris can be computed"""

    Mercury: ClassVar[solarsystem]
    """Mercury"""

    Venus: ClassVar[solarsystem]
    """Venus"""

    EMB: ClassVar[solarsystem]
    """Earth-Moon Barycenter"""

    Mars: ClassVar[solarsystem]
    """Mars"""

    Jupiter: ClassVar[solarsystem]
    """Jupiter"""

    Saturn: ClassVar[solarsystem]
    """Saturn"""

    Uranus: ClassVar[solarsystem]
    """Uranus"""

    Neptune: ClassVar[solarsystem]
    """Neptune"""

    Pluto: ClassVar[solarsystem]
    """Pluto"""

    Moon: ClassVar[solarsystem]
    """Moon"""

    Sun: ClassVar[solarsystem]
    """Sun"""

class sgp4_error:
    """Represent errors from SGP-4 propagation of two-line element sets (TLEs)"""

    success: ClassVar[sgp4_error]
    """Success"""

    eccen: ClassVar[sgp4_error]
    """Eccentricity < 0 or > 1"""

    mean_motion: ClassVar[sgp4_error]
    """Mean motion (revs / day) < 0"""

    perturb_eccen: ClassVar[sgp4_error]
    """Perturbed eccentricity < 0 or > 1"""

    semi_latus_rectum: ClassVar[sgp4_error]
    """Semi-Latus Rectum < 0"""

    unused: ClassVar[sgp4_error]
    """Unused, but in base code, so keeping for completeness"""

    orbit_decay: ClassVar[sgp4_error]
    """Orbit decayed"""

class weekday:
    """

    Represent the day of the week

    Values:
    - `Sunday`
    - `Monday`
    - `Tuesday`
    - `Wednesday`
    - `Thursday`
    - `Friday`
    - `Saturday`
    """

    Sunday: ClassVar[weekday]
    """Sunday"""

    Monday: ClassVar[weekday]
    """Monday"""

    Tuesday: ClassVar[weekday]
    """Tuesday"""

    Wednesday: ClassVar[weekday]
    """Wednesday"""

    Thursday: ClassVar[weekday]
    """Thursday"""

    Friday: ClassVar[weekday]
    """Friday"""

    Saturday: ClassVar[weekday]
    """Saturday"""

class tlefitstatus:
    """
    Termination status of the TLE non-linear least-squares fit performed by
    :meth:`TLE.fit_from_states`.

    Values:

    - ``GradientConverged``: converged on gradient norm tolerance
    - ``StepConverged``: converged on relative step size tolerance
    - ``CostConverged``: converged on relative cost change tolerance
    - ``MaxIterations``: maximum number of iterations reached
    - ``DampingSaturated``: Levenberg-Marquardt damping parameter saturated
    """

    GradientConverged: ClassVar[tlefitstatus]
    """Converged on gradient norm tolerance"""

    StepConverged: ClassVar[tlefitstatus]
    """Converged on relative step size tolerance"""

    CostConverged: ClassVar[tlefitstatus]
    """Converged on relative cost change tolerance"""

    MaxIterations: ClassVar[tlefitstatus]
    """Maximum number of iterations reached"""

    DampingSaturated: ClassVar[tlefitstatus]
    """Levenberg-Marquardt damping parameter saturated"""

    def converged(self) -> bool:
        """True if the fit converged successfully."""
        ...

class timescale:
    """
    Specify time scale used to represent or convert between the "satkit.time"
    representation of time

    Most of the time, these are not needed directly, but various time scales
    are needed to compute precise rotations between various inertial and
    Earth-fixed coordinate frames

    For an excellent overview, see:
    <https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/MPG%20Book/Release/Chapter2-TimeScales.pdf>

    Values:

    - `Invalid`: Invalid time scale
    - `UTC`: Universal Time Coordinate
    - `TT`: Terrestrial Time
    - `UT1`: UT1
    - `TAI`: International Atomic Time
    - `GPS`: Global Positioning System (GPS) time
    - `TDB`: Barycentric Dynamical Time
    """

    Invalid: ClassVar[timescale]
    """Invalid time scale"""

    UTC: ClassVar[timescale]
    """Universal Time Coordinate"""

    TT: ClassVar[timescale]
    """Terrestrial Time"""

    UT1: ClassVar[timescale]
    """UT1"""

    TAI: ClassVar[timescale]
    """International Atomic Time
    (nice because it is monotonically increasing)
    """

    GPS: ClassVar[timescale]
    """Global Positioning System (GPS) time"""

    TDB: ClassVar[timescale]
    """Barycentric Dynamical Time"""

class frame:
    """Coordinate reference frame

    Used to specify the frame for thrust vectors and maneuvers.

    Available frames:

    - ``GCRF`` - Geocentric Celestial Reference Frame (inertial)
    - ``ITRF`` - International Terrestrial Reference Frame (Earth-fixed)
    - ``TEME`` - True Equator Mean Equinox (SGP4 output frame)
    - ``CIRS`` - Celestial Intermediate Reference System
    - ``TIRS`` - Terrestrial Intermediate Reference System
    - ``EME2000`` - Earth Mean Equator 2000
    - ``ICRF`` - International Celestial Reference Frame
    - ``LVLH`` - Local Vertical Local Horizontal: z = -r (nadir), y = -h (opposite angular momentum), x completes right-handed system
    - ``RTN`` - Radial / Tangential / Normal (CCSDS OEM convention; also
      exposed as ``RSW`` and ``RIC`` aliases for Vallado / older-NASA naming):
      R = radial (outward), T = tangential (in-track), N = normal (cross-track)
    - ``NTW`` - Normal-to-velocity / Tangent / Cross-track (velocity-aligned):
      T = along velocity, N = in-plane perpendicular to v, W = cross-track

    Example:

    ```python
    import satkit as sk

    # Use RTN frame for in-track thrust (RSW and RIC are aliases and work too)
    t = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)
    ```
    """

    GCRF: ClassVar[frame]
    """Geocentric Celestial Reference Frame (inertial)"""

    ITRF: ClassVar[frame]
    """International Terrestrial Reference Frame (Earth-fixed)"""

    TEME: ClassVar[frame]
    """True Equator Mean Equinox"""

    CIRS: ClassVar[frame]
    """Celestial Intermediate Reference System"""

    TIRS: ClassVar[frame]
    """Terrestrial Intermediate Reference System"""

    EME2000: ClassVar[frame]
    """Earth Mean Equator 2000"""

    ICRF: ClassVar[frame]
    """International Celestial Reference Frame"""

    LVLH: ClassVar[frame]
    """Local Vertical Local Horizontal — the classical crewed-spaceflight
    / GN&C body-pointing frame used on the ISS and most Earth-pointing
    vehicles.

    - z axis: -r (nadir, pointing toward Earth center)
    - y axis: -h (opposite orbital angular momentum, h = r × v)
    - x axis: completes right-handed system (approximately velocity direction for circular orbits)

    Geometrically spans the same orbital plane as ``frame.RTN`` but with
    different labels and sign conventions:

    - LVLH +x = RTN +T (in-track; perpendicular to R, not strictly along v)
    - LVLH -z = RTN +R (radial outward)
    - LVLH -y = RTN +N (cross-track)

    Supported as a maneuver frame — useful when porting GN&C code written
    in LVLH body-frame conventions. For eccentric orbits, note that LVLH
    +x is perpendicular to the position vector, not the velocity vector;
    for strict along-velocity semantics use ``frame.NTW`` instead.
    """

    RTN: ClassVar[frame]
    """Radial / Tangential / Normal — CCSDS OEM/OMM/ODM convention.

    Also known as **RSW** (Vallado) or **RIC** (older NASA / Clohessy-
    Wiltshire literature). The three names refer to the same axes;
    Python-level aliases ``frame.RSW`` and ``frame.RTN`` resolve to the
    same enum value as ``frame.RTN``, so all three compare equal and can
    be used interchangeably.

    - R (radial): unit vector along position (outward from Earth center)
    - T (tangential / in-track): perpendicular to R in the orbit plane,
      in the prograde direction. **Not** strictly along velocity for
      eccentric orbits — for "along velocity" semantics use ``frame.NTW``
      instead.
    - N (normal / cross-track): along angular momentum (h = r × v)

    This is the standard choice for CCSDS OEM/OMM covariance messages,
    for relative-motion (Hill / Clohessy-Wiltshire) equations, and for
    radial/normal burn components whose physical meaning is tied to the
    position vector.
    """

    RSW: ClassVar[frame]
    """Alias for ``frame.RTN`` — Vallado's name for the same orbital
    frame (Radial / S=Ŵ×R̂ / W=ĥ). ``frame.RSW == frame.RTN`` is True.
    See [`RTN`][frame.RTN] for the axis definition.
    """

    RIC: ClassVar[frame]
    """Alias for ``frame.RTN`` — the older NASA / Clohessy-Wiltshire name
    (Radial / In-track / Cross-track). ``frame.RIC == frame.RTN`` is
    True. Kept for backward compatibility with code written against
    earlier satkit versions where ``RIC`` was the canonical name. See
    [`RTN`][frame.RTN] for the axis definition.
    """

    NTW: ClassVar[frame]
    """Velocity-aligned orbital frame (Vallado §3.3).

    - N (in-plane normal to velocity): T̂ × Ŵ. For a circular orbit this
      coincides with the outward radial direction; for eccentric orbits it
      leans off-radial by the flight-path angle.
    - T (tangent): v̂, unit velocity vector
    - W (cross-track): (r × v) / |r × v|, same as RTN's N axis

    The natural frame for prograde/retrograde maneuvers: a pure +T delta-v
    of magnitude Δv adds *exactly* Δv to |v|, regardless of orbit eccentricity.
    """

class time:
    """Representation of an instant in time

    This has functionality similar to the "datetime" object, and in fact has
    the ability to convert to an from the "datetime" object.  However, a separate
    time representation is needed as the "datetime" object does not allow for
    conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)

    Notes:
        - If no arguments are passed in, the created object represents the current time
        - If year is passed in, month and day must also be passed in
        - If hour is passed in, minute and second must also be passed in

    Example:
        ```python
        print(satkit.time(2023, 3, 5, 11, 3, 45.453))
        # 2023-03-05 11:03:45.453Z

        print(satkit.time(2023, 3, 5))
        # 2023-03-05 00:00:00.000Z
        ```

    """

    def __init__(
        self,
        year: int = ...,
        month: int = ...,
        day: int = ...,
        hour: int = 0,
        min: int = 0,
        sec: float = 0.0,
        *,
        scale: timescale = ...,
        str: str = ...,
    ):
        """Create a time object representing input date and time

        This has functionality similar to the "datetime" object, and in fact has
        the ability to convert to an from the "datetime" object.  However, a separate
        time representation is needed as the "datetime" object does not allow for
        conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)

        Notes:
            - If no arguments are passed in, the created object represents the current time
            - If year is passed in, month and day must also be passed in
            - If hour is passed in, minute and second must also be passed in

        Args:
            year: Gregorian year (e.g., 2024)
            month: Gregorian month (1 = January, 2 = February, ...)
            day: Day of month, beginning with 1
            hour: Hour of day, in range [0,23], default is 0
            min: Minute of hour, in range [0,59], default is 0
            sec: Floating point second of minute, in range [0,60), default is 0
            scale: Time scale, default is satkit.timescale.UTC
            str: String representation of time, in format "YYYY-MM-DD HH:MM:SS.sssZ" or if other will try to guess

        Example:
            ```python
            print(satkit.time(2023, 3, 5, 11, 3, 45.453))
            # 2023-03-05 11:03:45.453Z

            print(satkit.time(2023, 3, 5))
            # 2023-03-05 00:00:00.000Z
            ```
        """
        ...

    @staticmethod
    def now() -> time:
        """Create a "time" object representing the instant of time at the
        calling of the function.

        Returns:
            Time object representing the current time
        """
        ...

    @staticmethod
    def from_string(str: str) -> time:
        """
        Create a "time" object from input string

        Args:
            str: String representation of time, in format "YYYY-MM-DD HH:MM:SS.sssZ" or if other will try to intelligently parse, but no guarantees

        Notes:
            - This is probably not what you want.  Use with caution.

        Returns:
            Time object representing input string

        Example:
            ```python
            print(satkit.time.from_string("2023-03-05 11:03:45.453Z"))
            # 2023-03-05 11:03:45.453Z
            ```
        """
        ...

    @staticmethod
    def from_rfc3339(rfc: str) -> time:
        """Create a "time" object from input RFC 3339 string

        Args:
            rfc (str): RFC 3339 string representation of time

        Notes:
            - RFC 3339 is a subset of ISO 8601
            - Only allows a subset of the format: "YYYY-MM-DDTHH:MM:SS.sssZ" or "YYYY-MM-DDTHH:MM:SS.ssssssZ"

        Returns:
            Time object representing input RFC 3339 string

        Example:
            ```python
            print(satkit.time.from_rfctime("2023-03-05T11:03:45.453Z"))
            # 2023-03-05 11:03:45.453Z
            ```
        """
        ...

    @staticmethod
    def strptime(str: str, format: str) -> time:
        """
        Create a "time" object from input string with given formatting

        Args:
            str (str): string representation of time
            format (str): format of the string

        Notes:
            - The format string is a subset of the strptime format string in the Python "datetime" module
            - Format Codes:
                - %Y - year
                - %m - month with leading zeros (01-12)
                - %d - day of month with leading zeros (01-31)
                - %H - hour with leading zeros (00-23)
                - %M - minute with leading zeros (00-59)
                - %S - second with leading zeros (00-59)
                - %f - microsecond, allowing for trailing zeros
                - %b - abbreviated month name (Jan, Feb, ...)
                - %B - full month name (January, February, ...)

        Returns:
            Time object representing input string

        Example:
            ```python
            # Note the microsecond %f actually is represented as milliseconds in the input string
            print(satkit.time.strptime("2023-03-05 11:03:45.453Z", "%Y-%m-%d %H:%M:%S.%fZ"))
            # 2023-03-05 11:03:45.453Z
            ```
        """
        ...

    @staticmethod
    def from_date(year: int, month: int, day: int) -> time:
        """Return a time object representing the start of the input day (midnight)

        Args:
            year (int): Gregorian year (e.g., 2024)
            month (int): Gregorian month (1 = January, 2 = February, ...)
            day (int): Day of month, beginning with 1

        Returns:
            Time object representing the start of the input day (midnight)

        Example:
            ```python
            t = satkit.time.from_date(2023, 6, 15)
            print(t)
            # 2023-06-15 00:00:00.000Z
            ```
        """
        ...

    @staticmethod
    def from_jd(jd: float, scale: timescale = timescale.UTC) -> time:
        """Return a time object representing input Julian date and time scale

        Args:
            jd (float): Julian date
            scale (timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            Time object representing input Julian date and time scale

        Example:
            ```python
            t = satkit.time.from_jd(2460000.5)
            print(t)
            ```
        """
        ...

    @staticmethod
    def from_unixtime(ut: float) -> time:
        """Return a time object representing input unixtime

        Args:
            ut (float): unixtime, UTC seconds since Jan 1, 1970 00:00:00
                        (leap seconds are not included)

        Returns:
            Time object representing input unixtime

        Example:
            ```python
            t = satkit.time.from_unixtime(1700000000)
            print(t)
            # 2023-11-14 22:13:20.000Z
            ```
        """
        ...

    @staticmethod
    def from_gps_week_and_second(week: int, sec: float) -> time:
        """Return a time object representing input GPS week and second

        Args:
            week: GPS week number
            sec: GPS seconds of week

        Returns:
            Time object representing input GPS week and second
        """
        ...

    def weekday(self) -> weekday:
        """
        Return the day of the week

        Returns:
            Day of the week
        """
        ...

    def day_of_year(self) -> int:
        """
        Return the 1-based Gregorian day of the year (1 = January 1, 365 = December 31)

        Returns:
            The 1-based day of the year
        """
        ...

    @staticmethod
    def from_mjd(mjd: float, scale: timescale = timescale.UTC) -> time:
        """Return a time object representing input modified Julian date and time scale

        Args:
            mjd (float): Modified Julian date
            scale (satkit.timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            Time object representing input modified Julian date and time scale

        Example:
            ```python
            t = satkit.time.from_mjd(60000.0)
            print(t)
            ```
        """
        ...

    def as_date(self) -> tuple[int, int, int]:
        """Return tuple representing as UTC Gegorian date of the time object.

        Returns:
            Tuple with 3 elements representing the Gregorian year, month, and day of the time object.
                Fractional component of day are truncated.
                Month is in range [1,12].
                Day is in range [1,31].
        """
        ...

    def as_gregorian(
        self, scale=timescale.UTC
    ) -> tuple[int, int, int, int, int, float]:
        """Return tuple representing as UTC Gegorian date and time of the time object.

        Args:
            scale (timescale, optional): Time scale.  Default is satkit.timescale.UTC

        Returns:
            Tuple with 6 elements representing the Gregorian year, month, day, hour, minute, and second of the time object.
                Month is in range [1,12].
                Day is in range [1,31].
        """
        ...

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
            Time object representing input UTC Gregorian time

        Example:
            ```python
            print(satkit.time.from_gregorian(2023, 3, 5, 11, 3,45.453))
            # 2023-03-05 11:03:45.453Z
            ```
        """
        ...

    @staticmethod
    def from_datetime(dt: datetime.datetime) -> time:
        """Convert input "datetime.datetime" object to an "satkit.time" object representing the same instant in time

        Args:
            dt (datetime.datetime): "datetime.datetime" object to convert

        Returns:
            Time object representing the same instant in time as the input "datetime.datetime" object
        """
        ...

    def as_datetime(self, utc: bool = True) -> datetime.datetime:
        """Convert object to "datetime.datetime" object representing same instant in time.

        Args:
            utc (bool, optional): Whether to make the "datetime.datetime" object represent time in the local timezone or "UTC".  Default is True

        Returns:
            "datetime.datetime" object representing the same instant in time as the "satkit.time" object

        Example:
            ```python
            dt = satkit.time(2023, 6, 3, 6, 19, 34).as_datetime(True)
            print(dt)
            # 2023-06-03 06:19:34+00:00

            dt = satkit.time(2023, 6, 3, 6, 19, 34).as_datetime(False)
            print(dt)
            # 2023-06-03 02:19:34
            ```
        """
        ...

    def datetime(self, utc: bool = True) -> datetime.datetime:
        """Deprecated: use :meth:`satkit.time.as_datetime`.

        Convert object to "datetime.datetime" object representing same instant in time.

        Args:
            utc (bool, optional): Whether to make the "datetime.datetime" object represent time in the local timezone or "UTC".  Default is True

        Returns:
            "datetime.datetime" object representing the same instant in time as the "satkit.time" object

        Example:
            ```python
            dt = satkit.time(2023, 6, 3, 6, 19, 34).datetime(True)
            print(dt)
            # 2023-06-03 06:19:34+00:00

            dt = satkit.time(2023, 6, 3, 6, 19, 34).datetime(False)
            print(dt)
            # 2023-06-03 02:19:34
            ```
        """
        ...

    def as_mjd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as a Modified Julian Date
        with the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC
        """
        ...

    def as_jd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as Julian Date with
        the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC
        """
        ...

    def as_unixtime(self) -> float:
        """
        Represent time as unixtime

        (seconds since Jan 1, 1970 UTC, excluding leap seconds)

        Includes fractional component of seconds
        """
        ...

    def as_iso8601(self) -> str:
        """
        Represent time as ISO 8601 string

        Returns:
            ISO 8601 string representation of time: "YYYY-MM-DDTHH:MM:SS.sssZ"
        """
        ...

    def as_rfc3339(self) -> str:
        """
        Represent time as RFC 3339 string

        Returns:
            RFC 3339 string representation of time: "YYYY-MM-DDTHH:MM:SS.sssZ"
        """
        ...

    def strftime(self, format: str) -> str:
        """
        Represent time as string with given format

        Args:
            format (str): format of the string

        Notes:
            Format Codes:

            - %Y - year
            - %m - month with leading zeros (01-12)
            - %d - day of month with leading zeros (01-31)
            - %H - hour with leading zeros (00-23)
            - %M - minute with leading zeros (00-59)
            - %S - second with leading zeros (00-59)
            - %f - microsecond, allowing for trailing zeros
            - %b - abbreviated month name (Jan, Feb, ...)
            - %B - full month name (January, February, ...)
            - %A - full weekday name (Sunday, Monday, ...)
            - %w - weekday as a decimal number (0=Sunday, 1=Monday, ...)

        Returns:
            string representation of time

        Example:
            ```python
            print(satkit.time(2023, 6, 3, 6, 19, 34).strptime("%Y-%m-%d %H:%M:%S"))
            # 2023-06-03 06:19:34
            ```
        """
        ...

    @typing.overload
    def __add__(self, other: duration) -> time:
        """
        Return a time object representing the input duration added to the current time

        Args:
            other (duration): duration to add to the current time

        Returns:
            Time object representing the input duration added to the current time

        """
        ...

    @typing.overload
    def __add__(self, other: float) -> time:
        """
        Return a time object representing the input number of days added to the current time

        Args:
            other (float): number of days to add to the current time

        Returns:
            Time object representing the input number of days added to the current time

        """
        ...

    @typing.overload
    def __add__(self, other: list[duration]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise addition of days to the "self" time object

        Args:
            other (list[duration]): array-like structure containing days to add to the current time

        Returns:
            Array of time objects representing the element-wise addition of days to the current time
        """
        ...

    def __le__(self, other: time) -> bool:
        """
        Compare two time objects for less than or equal to

        Args:
            other (time): time object to compare with

        Returns:
            True if "self" time is less than or equal to "other" time, False otherwise
        """
        ...

    def __lt__(self, other: time) -> bool:
        """
        Compare two time objects for less than

        Args:
            other (time): time object to compare with

        Returns:
            True if "self" time is less than "other" time, False otherwise
        """
        ...

    def __ge__(self, other: time) -> bool:
        """
        Compare two time objects for greater than or equal to

        Args:
            other (time): time object to compare with

        Returns:
            True if "self" time is greater than or equal to "other" time, False otherwise
        """
        ...

    def __gt__(self, other: time) -> bool:
        """
        Compare two time objects for greater than

        Args:
            other (time): time object to compare with

        Returns:
            True if "self" time is greater than "other" time, False otherwise
        """
        ...

    def __eq__(self, value: object) -> bool:
        """
        Compare two time objects for equality

        Args:
            value (object): object to compare with

        Returns:
            True if "self" time is equal to "value", False otherwise
        """
        ...

    def __ne__(self, value: object) -> bool:
        """
        Compare two time objects for inequality

        Args:
            value (object): object to compare with

        Returns:
            True if "self" time is not equal to "value", False otherwise
        """
        ...

    @typing.overload
    def __add__(self, other: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise addition of duration to the "self" time object

        Args:
            other (npt.ArrayLike[Any]): array-like structure containing durations to add to the current time

        Returns:
            Array of time objects representing the element-wise addition of durations to the current time

        """
        ...

    @typing.overload
    def __sub__(self, other: duration) -> time:
        """
        Return a time object representing the input duration subtracted from the current time

        Args:
            other (duration): duration to subtract from the current time

        Returns:
            Time object representing the input duration subtracted from the current time

        """
        ...

    @typing.overload
    def __sub__(self, other: time) -> duration:
        """
        Return a duration object representing the difference between the two times

        Args:
            other (time): time to subtract from the current time

        Returns:
            Duration object representing the difference between the two times

        """
        ...

    @typing.overload
    def __sub__(self, other: float) -> time:
        """
        Return a time object representing the input number of days subtracted from the current time

        Args:
            other (float): number of days to subtract from the current time

        Returns:
            Time object representing the input number of days subtracted from the current time

        """
        ...

    @typing.overload
    def __sub__(self, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of days from the "self" time object

        Args:
            other (npt.ArrayLike[float]): array-like structure containing days to subtract from the current time

        Returns:
            Array of time objects representing the element-wise subtraction of days from the current time

        """
        ...

    @typing.overload
    def __sub__(self, other: list[duration]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of duration from the "self" time object

        Args:
            other (list[duration]): array-like structure containing durations to subtract from the current time

        Returns:
            Array of time objects representing the element-wise subtraction of durations from the current time
        """
        ...

    @typing.overload
    def __sub__(self, other: list[time]) -> npt.NDArray[Any]:
        """
        Return a numpy array of duration objects, with each object representing an element-wise subtraction of time from the "self" time object

        Args:
            other (list[time]): array-like structure containing times to subtract from the current time

        Returns:
            Array of duration objects representing the element-wise subtraction of times from the current time
        """
        ...

class duration:
    """
    Representation of a duration, or interval of time
    """

    def __init__(
        self,
        *,
        days: float = 0,
        hours: float = 0,
        minutes: float = 0,
        seconds: float = 0.0,
        microseconds: float = 0.0,
    ):
        """Create a duration object representing input time duration

        Args:
            days: Number of days, default is 0
            hours: Number of hours, default is 0
            minutes: Number of minutes, default is 0
            seconds: Number of seconds, default is 0.0
            microseconds: Number of microseconds, default is 0.0

        Notes:
            - If no arguments are passed in, the created object represents a duration of 0 seconds

        Example:
            ```python
            print(satkit.duration(days=1, hours=2, minutes=3, seconds=4.5))
            # Duration: 1 days, 2 hours, 3 minutes, 4.500 seconds
            ```

        """
        ...

    @staticmethod
    def from_days(d: float) -> duration:
        """Create duration object given input number of days. Note: a day is defined as 86,400 seconds

        Args:
            d (float): Number of days

        Returns:
            Duration object representing input number of days

        Example:
            ```python
            d = satkit.duration.from_days(1.5)
            print(d.hours)
            # 36.0
            ```
        """
        ...

    @staticmethod
    def from_seconds(s: float) -> duration:
        """Create duration object representing input number of seconds

        Args:
            s (float): Number of seconds

        Returns:
            Duration object representing input number of seconds

        Example:
            ```python
            d = satkit.duration.from_seconds(3600)
            print(d.hours)
            # 1.0
            ```
        """
        ...

    @staticmethod
    def from_minutes(m: float) -> duration:
        """Create duration object representing input number of minutes

        Args:
            m (float): Number of minutes

        Returns:
            Duration object representing input number of minutes
        """
        ...

    @staticmethod
    def from_hours(h: float) -> duration:
        """Create duration object representing input number of hours

        Args:
            h (float): Number of hours

        Returns:
            Duration object representing input number of hours
        """
        ...

    @typing.overload
    def __add__(self, other: duration) -> duration:
        """Add a duration to another duration

        Args:
            other (duration): duration to add to the current duration

        Returns:
            Duration object representing the sum, or concatenation, of both durations

        Example:
            ```python
            print(duration.from_hours(1) + duration.from_minutes(1))
            # Duration: 1 hours, 1 minutes, 0.000 seconds
            ```
        """
        ...

    @typing.overload
    def __add__(self, other: float) -> duration:
        """Add a number of days to the current duration

        Args:
            other (float): number of days to add to the current duration

        Returns:
            Duration object representing the input number of days added to the current duration

        Example:
            ```python
            print(duration.from_days(1) + 2.5)
            # Duration: 3 days, 0 hours, 0 minutes, 0.000 seconds
            ```
        """
        ...

    @typing.overload
    def __add__(self, other: time) -> time:
        """Add a duration to a time

        Args:
            other (time): time to add the current duration to

        Returns:
            Time object representing the input time plus the duration

        Example:
            ```python
            print(duration.from_hours(1) + satkit.time(2023, 6, 4, 11,30,0))
            # 2023-06-04 13:30:00.000Z
            ```
        """
        ...

    def __sub__(self, other: duration) -> duration:
        """Take the difference between two durations

        Args:
            other (duration): duration to subtract from the current duration

        Returns:
            Duration object representing the difference between the two durations

        Example:
            ```python
            print(duration.from_hours(1) - duration.from_minutes(1))
            # Duration: 59 minutes, 0.000 seconds
            ```
        """
        ...

    def __mul__(self, other: float) -> duration:
        """Multiply (or scale) duration by given value

        Args:
            other (float): value by which to multiply duration

        Returns:
            Duration object representing the input duration scaled by the input value

        Example:
            ```python
            print(duration.from_days(1) * 2.5)
            # Duration: 2 days, 12 hours, 0 minutes, 0.000 seconds
            ```
        """
        ...

    @typing.overload
    def __truediv__(self, other: float) -> duration:
        """Divide (or scale) duration by given value

        Args:
            other (float): value by which to divide duration

        Returns:
            Duration object representing the input duration divided by the input value

        Example:
            ```python
            print(duration.from_days(1) / 2)
            # Duration: 12 hours, 0 minutes, 0.000 seconds
            ```
        """
        ...

    @typing.overload
    def __truediv__(self, other: duration) -> float:
        """Divide (or scale) duration by another duration to get a dimensionless ratio

        Args:
            other (duration): duration by which to divide current duration

        Returns:
            Dimensionless ratio of the two durations

        Example:
            ```python
            print(duration.from_hours(1) / duration.from_minutes(30))
            # 2.0
            ```
        """
        ...

    def __gt__(self, other: duration) -> bool:
        """Compare two durations for greater than

        Args:
            other (duration): duration to compare with
        Returns:
            True if "self" duration is greater than "other" duration, False otherwise

        Example:
            ```python
            print(duration.from_hours(1) > duration.from_minutes(30))
            # True
            ```
        """
        ...

    def __lt__(self, other: duration) -> bool:
        """Compare two durations for less than

        Args:
            other (duration): duration to compare with
        Returns:
            True if "self" duration is less than "other" duration, False otherwise

        Example:
            ```python
            print(duration.from_hours(1) < duration.from_minutes(30))
            # False
            ```
        """
        ...

    def __ge__(self, other: duration) -> bool:
        """Compare two durations for greater than or equal to

        Args:
            other (duration): duration to compare with
        Returns:
            True if "self" duration is greater than or equal to "other" duration, False otherwise

        Example:
            ```python
            print(duration.from_hours(1) >= duration.from_minutes(30))
            # True
            ```
        """
        ...

    def __le__(self, other: duration) -> bool:
        """Compare two durations for less than or equal to

        Args:
            other (duration): duration to compare with
        Returns:
            True if "self" duration is less than or equal to "other" duration, False otherwise

        Example:
            ```python
            print(duration.from_hours(1) <= duration.from_minutes(30))
            # False
            ```
        """
        ...

    @property
    def days(self) -> float:
        """Floating point number of days represented by duration

        Returns:
            Floating point number of days represented by duration

        A day is defined as 86,400 seconds
        """
        ...

    @property
    def hours(self) -> float:
        """Floating point number of hours represented by duration

        Returns:
            Floating point number of hours represented by duration
        """
        ...

    @property
    def minutes(self) -> float:
        """Floating point number of minutes represented by duration

        Returns:
            Floating point number of minutes represented by duration
        """
        ...

    @property
    def seconds(self) -> float:
        """Floating point number of seconds represented by duration

        Returns:
            Floating point number of seconds represented by duration
        """
        ...

class quaternion:
    """Quaternion representing rotation of 3D Cartesian axes

    Quaternions perform right-handed rotation of a vector, e.g. rotation of +xhat 90 degrees by +zhat give +yhat

    This is different than the convention used in Vallado, but it is the way it is commonly used in mathematics and it is the way it should be done.

    For the uninitiated: quaternions are a more-compact and
    computationally efficient way of representing 3D rotations.
    They can also be multiplied together and easily renormalized to
    avoid problems with floating-point precision eventually causing
    changes in the rotated vecdtor norm.

    For details, see:

    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>

    Notes:
        - Under the hood, this is using the "UnitQuaternion" object in the rust "nalgebra" crate.
    """

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Return quaternion with input (w,x,y,z) values

        Args:
            w: Scalar component of the quaternion

        Example:
            ```python
            # Identity quaternion (no rotation)
            q = satkit.quaternion()

            # 90 degree rotation about z-axis
            import math
            q = satkit.quaternion.rotz(math.radians(90))
            ```
            x: X component of the quaternion
            y: Y component of the quaternion
            z: Z component of the quaternion
        """
        ...

    @staticmethod
    def from_axis_angle(axis: npt.NDArray[np.float64], angle: float) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "angle" degrees about the given axis

        Args:
            axis (npt.ArrayLike[np.float64]): 3-element array representing axis of rotation
            angle (float): angle of rotation in radians

        Returns:
            Quaternion representing rotation by "angle" degrees about the given axis
        """
        ...

    @staticmethod
    def from_rotation_matrix(
        mat: npt.NDArray[np.float64],
    ) -> quaternion:
        """Return quaternion representing identical rotation to input 3x3 rotation matrix

        Args:
            mat (npt.ArrayLike[np.float64]): 3x3 rotation matrix

        Returns:
            Quaternion representing identical rotation to input 3x3 rotation matrix
        """
        ...

    @staticmethod
    def rotx(theta) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the xhat unit vector

        Args:
            theta (float): angle of rotation in radians

        Returns:
            Quaternion representing right-handed rotation of vector by "theta" radians about the xhat unit vector

        Notes:
            Equivalent rotation matrix:
            | 1             0            0|
            | 0    cos(theta)  -sin(theta)|
            | 0    sin(theta)   cos(theta)|
        """
        ...

    @staticmethod
    def roty(theta) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the yhat unit vector

        Args:
            theta (float): angle of rotation in radians

        Returns:
            Quaternion representing right-handed rotation of vector by "theta" radians about the yhat unit vector


        Notes:
            Equivalent rotation matrix:
            |  cos(theta)     0    sin(theta)|
            |           0     1             0|
            | -sin(theta)     0    cos(theta)|
        """
        ...

    @staticmethod
    def rotz(theta) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the zhat unit vector

        Args:
            theta (float): angle of rotation in radians

        Returns:
            Quaternion representing right-handed rotation of vector by "theta" radians about the zhat unit vector

        Notes:
            Equivalent rotation matrix:
            |  cos(theta)     -sin(theta)   0|
            |  sin(theta)      cos(theta)   0|
            |           0               0   1|
        """
        ...

    @staticmethod
    def rotation_between(
        v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]
    ) -> quaternion:
        """Quaternion representation rotation between two input vectors

        Args:
            v1 (npt.ArrayLike[np.float64]): vector rotating from
            v2 (npt.ArrayLike[np.float64]): vector rotating to

        Returns:
            Quaternion that rotates from v1 to v2

        Example:
            ```python
            import numpy as np
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, 1, 0])
            q = satkit.quaternion.rotation_between(v1, v2)
            print(q * v1)
            # [0, 1, 0]
            ```
        """
        ...

    def as_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Return 3x3 rotation matrix representing equivalent rotation

        Returns:
            3x3 rotation matrix representing equivalent rotation
        """
        ...

    def as_euler(self) -> tuple[float, float, float]:
        """Return equivalent rotation as intrinsic ZYX Euler angles (yaw, pitch, roll).

        The decomposition follows the aerospace convention (Tait-Bryan angles):
        the rotation is equivalent to first rotating by yaw about Z,
        then pitch about the new Y, then roll about the new X.

        Returns:
            tuple[float, float, float]: ``(roll, pitch, yaw)`` in radians

        Example:
            ```python
            q = satkit.quaternion.rotz(0.1) * satkit.quaternion.roty(0.2)
            roll, pitch, yaw = q.as_euler()
            ```
        """
        ...

    @property
    def angle(self) -> float:
        """Return the angle in radians of the rotation

        Returns:
            Angle in radians of the rotation
        """
        ...

    @property
    def axis(self) -> npt.NDArray[np.float64]:
        """Return the axis of rotation as a unit vector

        Returns:
            3-element array representing the axis of rotation as a unit vector
        """
        ...

    @property
    def conj(self) -> quaternion:
        """Return conjugate or inverse of the rotation

        Returns:
            Conjugate or inverse of the rotation
        """
        ...

    @property
    def conjugate(self) -> quaternion:
        """Return conjugate or inverse of the rotation

        Returns:
            Conjugate or inverse of the rotation
        """
        ...

    @property
    def x(self) -> float:
        """X component of the quaternion

        Returns:
            X component of the quaternion
        """
        ...

    @property
    def y(self) -> float:
        """Y component of the quaternion

        Returns:
            Y component of the quaternion
        """
        ...

    @property
    def z(self) -> float:
        """Z component of the quaternion

        Returns:
            Z component of the quaternion
        """
        ...

    @property
    def w(self) -> float:
        """Scalar component of the quaternion

        Returns:
            Scalar component of the quaternion
        """
        ...

    @typing.overload
    def __mul__(self, other: quaternion) -> quaternion:
        """Multiply by another quaternion to concatenate rotations

        Notes:
            - Multiply represents concatenation of two rotations representing the quaternions.  The left value rotation is applied after the right value, per the normal convention

        Args:
            other (quaternion): quaternion to multiply by

        Returns:
            Quaternion representing concatenation of the two rotations
        """
        ...

    @typing.overload
    def __mul__(self, other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply by a vector to rotate the vector

        Args:
            other (npt.ArrayLike[np.float64]): 3-element array representing vector to rotate or Nx3 array of vectors to rotate

        Returns:
            3-element array representing rotated vector or Nx3 array of rotated vectors

        Example:
            ```python
            xhat = np.array([1,0,0])
            q = satkit.quaternion.rotz(np.pi/2)
            print(q * xhat)
            # [0, 1, 0]
            ```
        """
        ...

    def slerp(
        self, other: quaternion, frac: float, epsilon: float = 1.0e-6
    ) -> quaternion:
        """Spherical linear interpolation between self and other

        Args:
            other (quaternion): Quaternion to perform interpolation to
            frac (float): fractional amount of interpolation, in range [0,1]
            epsilon (float, optional): Value below which the sin of the angle separating both quaternions must be to return an error. Default is 1.0e-6

        Returns:
            Quaternion representing interpolation between self and other

        Example:
            ```python
            import math
            q1 = satkit.quaternion.rotz(math.radians(0))
            q2 = satkit.quaternion.rotz(math.radians(90))
            q_mid = q1.slerp(q2, 0.5)
            print(f"Mid-rotation angle: {math.degrees(q_mid.angle()):.1f} deg")
            # Mid-rotation angle: 45.0 deg
            ```
        """
        ...

class kepler:
    """Represent Keplerian element sets and convert between cartesian


    Notes:
        - This class is used to represent Keplerian elements and convert between Cartesian coordinates
        - The class uses the semi-major axis (a), not the semiparameter
        - All angle units are radians
        - All length units are meters
        - All velocity units are meters / second
    """

    def __init__(
        self,
        a: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        nu: float = ...,
        *,
        true_anomaly: float = ...,
        mean_anomaly: float = ...,
        eccentric_anomaly: float = ...,
    ):
        """Create Keplerian element set object from input elements

        Args:
            a: Semi-major axis, meters
            e: Eccentricity, unitless
            i: Inclination, radians
            raan: Right ascension of ascending node, radians
            argp: Argument of perigee, radians
            nu: True anomaly, radians
            true_anomaly: True anomaly, radians (keyword alternative to nu)
            mean_anomaly: Mean anomaly, radians (keyword alternative to nu)
            eccentric_anomaly: Eccentric anomaly, radians (keyword alternative to nu)

        Notes:
            If "nu" is provided (6th argument), it will be used as the true anomaly.
            Anomaly may also be set via keyword arguments; if so, there should only be
            5 positional input arguments.

        Example:
            ```python
            import math

            # Create a ~400 km circular LEO orbit
            k = satkit.kepler(
                a=6.781e6,        # semi-major axis, meters
                e=0.001,          # near-circular
                i=math.radians(51.6),
                raan=math.radians(0),
                argp=math.radians(0),
                nu=math.radians(0),
            )
            ```
        """
        ...

    def to_pv(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert Keplerian element set to position and velocity vectors

        Returns:
            Tuple with two elements representing the position and velocity vectors

        Example:
            ```python
            pos, vel = k.to_pv()
            print(f"Position: {pos} m")
            print(f"Velocity: {vel} m/s")
            ```
        """
        ...

    def propagate(self, dt: duration | float) -> kepler:
        """Propagate Keplerian element set by input duration

        Args:
            dt (duration | float): Duration by which to propagate the Keplerian element set
                                   If float, value is seconds

        Returns:
            Keplerian element set object after propagation

        Example:
            ```python
            # Propagate orbit by one orbital period
            k2 = k.propagate(k.period)
            ```
        """
        ...

    @property
    def mean_motion(self) -> float:
        """Mean motion, radians / second"""
        ...

    @property
    def true_anomaly(self) -> float:
        """True anomaly, radians"""
        ...

    @property
    def eccentric_anomaly(self) -> float:
        """Eccentric anomaly, radians"""
        ...

    @eccentric_anomaly.setter
    def eccentric_anomaly(self, value: float) -> None: ...
    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly, radians"""
        ...

    @mean_anomaly.setter
    def mean_anomaly(self, value: float) -> None: ...
    @property
    def period(self) -> float:
        """Orbital period, seconds"""
        ...

    @property
    def a(self) -> float:
        """Semi-major axis, meters"""
        ...

    @a.setter
    def a(self, value: float) -> None: ...
    @property
    def eccen(self) -> float:
        """Eccentricity, unitless"""
        ...

    @eccen.setter
    def eccen(self, value: float) -> None: ...
    @property
    def inclination(self) -> float:
        """Inclination, radians"""
        ...

    @inclination.setter
    def inclination(self, value: float) -> None: ...
    @property
    def raan(self) -> float:
        """Right ascension of ascending node, radians"""
        ...

    @raan.setter
    def raan(self, value: float) -> None: ...
    @property
    def nu(self) -> float:
        """True anomaly, radians"""
        ...

    @nu.setter
    def nu(self, value: float) -> None: ...
    @property
    def w(self) -> float:
        """Argument of perigee, radians"""
        ...

    @w.setter
    def w(self, value: float) -> None: ...
    @staticmethod
    def from_pv(pos: npt.NDArray[np.float64], vel: npt.NDArray[np.float64]) -> kepler:
        """Create Keplerian element set from input position and velocity vectors

        Args:
            pos: 3-element array representing position vector
            vel: 3-element array representing velocity vector

        Returns:
            Keplerian element set object

        Example:
            ```python
            import numpy as np
            pos = np.array([6.781e6, 0, 0])  # meters, GCRF
            vel = np.array([0, 7.5e3, 0])    # m/s, GCRF
            k = satkit.kepler.from_pv(pos, vel)
            print(f"Semi-major axis: {k.a/1e3:.1f} km")
            print(f"Eccentricity: {k.e:.6f}")
            ```
        """
        ...

class geodetic:
    """Geodetic coordinates with named fields

    Attributes:
        latitude_rad (float): Latitude in radians
        longitude_rad (float): Longitude in radians
        height_m (float): Height above WGS84 ellipsoid in meters
        latitude_deg (float): Latitude in degrees (computed)
        longitude_deg (float): Longitude in degrees (computed)
    """

    latitude_rad: float
    longitude_rad: float
    height_m: float

    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees"""
        ...

    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees"""
        ...


class itrfcoord:
    """Representation of a coordinate in the International Terrestrial Reference Frame (ITRF)

    This coordinate object can be created from and also output to Geodetic coordinates
    (latitude, longitude, height above ellipsoid). Functions are also available to provide
    rotation quaternions to the East-North-Up frame and North-East-Down frame at this coordinate.

    Example:
        Create ITRF coord from Cartesian:

        ```python
        coord = itrfcoord([ 1523128.63570828, -4461395.28873207,  4281865.94218203 ])
        ```

        Create ITRF coord from Geodetic:

        ```python
        coord = itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
        ```

    """

    def __init__(
        self,
        vec: npt.NDArray[np.float64] | list[float] | None = None,
        *,
        latitude_deg: float = ...,
        longitude_deg: float = ...,
        latitude_rad: float = ...,
        longitude_rad: float = ...,
        altitude: float = ...,
        height: float = ...,
    ):
        """Create ITRF coordinate from Cartesian vector or geodetic parameters.

        Args:
            vec: ITRF Cartesian location in meters (3-element array, list, or tuple)
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            latitude_rad: Latitude in radians
            longitude_rad: Longitude in radians
            altitude: Height above ellipsoid, meters
            height: Height above ellipsoid, meters (alias for altitude)
        """
        ...

    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees"""
        ...

    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees"""
        ...

    @property
    def latitude_rad(self) -> float:
        """Latitude in radians"""
        ...

    @property
    def longitude_rad(self) -> float:
        """Longitude in radians"""
        ...

    @property
    def altitude(self) -> float:
        """Altitude above ellipsoid, in meters"""
        ...

    @property
    def geodetic(self) -> geodetic:
        """Geodetic coordinates as a named struct

        Returns:
            Geodetic struct with latitude_rad, longitude_rad, height_m fields
                and latitude_deg, longitude_deg computed properties
        """
        ...

    @property
    def vector(self) -> npt.NDArray[np.float64]:
        """Cartesian ITRF coord as numpy array

        Returns:
            3-element numpy array representing the ITRF Cartesian coordinate in meters
        """
        ...

    @property
    def qned2itrf(self) -> quaternion:
        """Quaternion representing rotation from North-East-Down (NED) to ITRF at this location

        Returns:
            Quaternion representiong rotation from North-East-Down (NED) to ITRF at this location
        """
        ...

    @property
    def qenu2itrf(self) -> quaternion:
        """Quaternion representiong rotation from East-North-Up (ENU) to ITRF at this location

        Returns:
            Quaternion representiong rotation from East-North-Up (ENU) to ITRF at this location
        """
        ...

    def to_enu(self, origin: itrfcoord) -> npt.NDArray[np.float64]:
        """East-North-Up (ENU) vector from `origin` to `self`, in `origin`'s local-tangent frame.

        The ENU triad has its origin at ``origin``; ``self`` is the point being
        located. The ``Up`` component is positive when ``self`` is above ``origin``
        along ``origin``'s local normal (further from Earth's center) — i.e.
        "what direction is ``self`` from where I'm standing at ``origin``?"

        Args:
            origin (itrfcoord): ITRF coordinate at which the ENU frame is
                anchored (the observer / station / base of the local tangent plane).

        Returns:
            3-element ``[E, N, U]`` vector from ``origin`` to ``self``, in meters.

        Notes:
            - This is equivalent to calling: origin.qenu2itrf.conj * (self - origin)

        Example:
            ```python
            station   = satkit.itrfcoord(latitude_deg=42.466, longitude_deg=-71.1516, altitude=0)
            satellite = satkit.itrfcoord(latitude_deg=42.466, longitude_deg=-71.1516, altitude=400_000)
            enu = satellite.to_enu(station)  # satellite is overhead → Up ≈ +400_000 m
            print(f"East: {enu[0]:.1f} m, North: {enu[1]:.1f} m, Up: {enu[2]:.1f} m")
            ```
        """
        ...

    def to_ned(self, origin: itrfcoord) -> npt.NDArray[np.float64]:
        """North-East-Down (NED) vector from `origin` to `self`, in `origin`'s local-tangent frame.

        The NED triad has its origin at ``origin``; ``self`` is the point being
        located. The ``Down`` component is positive when ``self`` is below
        ``origin`` along ``origin``'s local normal (closer to Earth's center).

        Args:
            origin (itrfcoord): ITRF coordinate at which the NED frame is
                anchored (the observer / station / base of the local tangent plane).

        Returns:
            3-element ``[N, E, D]`` vector from ``origin`` to ``self``, in meters.

        Notes:
            - This is equivalent to calling: origin.qned2itrf.conj * (self - origin)

        """
        ...

    def __sub__(self, other: itrfcoord) -> npt.NDArray[np.float64]:
        """Subtract another ITRF coordinate from this one

        Args:
            other (itrfcoord): Other ITRF coordinate to subtract

        Returns:
            3-element numpy array representing the difference in meters between the two ITRF coordinates
        """
        ...

    def geodesic_distance(self, other: itrfcoord) -> tuple[float, float, float]:
        """Use Vincenty formula to compute geodesic distance:
        <https://en.wikipedia.org/wiki/Vincenty%27s_formulae>

        Returns:
            (distance in meters, initial heading in radians, heading at destination in radians)

        Example:
            ```python
            boston = satkit.itrfcoord(latitude_deg=42.36, longitude_deg=-71.06, altitude=0)
            nyc = satkit.itrfcoord(latitude_deg=40.71, longitude_deg=-74.01, altitude=0)
            dist, heading_start, heading_end = boston.geodesic_distance(nyc)
            print(f"Distance: {dist/1000:.1f} km")
            ```
        """
        ...

    def move_with_heading(self, distance: float, heading_rad: float) -> itrfcoord:
        """Move a distance along the Earth surface with a given initial heading

        Args:
            distance (float): Distance to move in meters
            heading_rad (float): Initial heading in radians

        Notes:
            Altitude is assumed to be zero

            Use Vincenty formula to compute position:
            <https://en.wikipedia.org/wiki/Vincenty%27s_formulae>

        Returns:
            (distance in meters, initial heading in radians, heading at destination in radians)

        Example:
            ```python
            import math
            start = satkit.itrfcoord(latitude_deg=42.36, longitude_deg=-71.06, altitude=0)
            # Move 100 km due north
            dest = start.move_with_heading(100e3, math.radians(0))
            print(f"Destination: {dest.latitude_deg:.2f} deg lat, {dest.longitude_deg:.2f} deg lon")
            ```
        """
        ...

class consts:
    """Some constants that are useful for saetllite dynamics"""

    wgs84_a: ClassVar[float]
    """WGS-84 semiparameter, in meters"""

    wgs84_f: ClassVar[float]
    """WGS-84 flattening in meters"""

    earth_radius: ClassVar[float]
    """Earth radius along major axis, meters"""

    mu_earth: ClassVar[float]
    """Gravitational parameter of Earth, m^3/s^2"""

    mu_moon: ClassVar[float]
    """Gravitational parameter of Moon, m^3/s^2"""

    mu_sun: ClassVar[float]
    """Gravitational parameter of sun, m^3/s^2"""

    GM: ClassVar[float]
    """Gravitational parameter of Earth, m^3/s^2"""

    omega_earth: ClassVar[float]
    """Scalar Earth rotation rate, rad/s"""

    c: ClassVar[float]
    """Speed of light, m/s"""

    au: ClassVar[float]
    """Astronomical Unit, mean Earth-Sun distance, meters"""

    sun_radius: ClassVar[float]
    """Radius of sun, meters"""

    moon_radius: ClassVar[float]
    """Radius of moon, meters"""

    earth_moon_mass_ratio: ClassVar[float]
    """Earth mass over Moon mass, unitless"""

    geo_r: ClassVar[float]
    """Distance to Geosynchronous orbit from Earth center, meters"""

    jgm3_mu: ClassVar[float]
    """Earth gravitational parameter from JGM3 gravity model, m^3/s^2"""

    jgm3_a: ClassVar[float]
    """Earth semiparameter from JGM3 gravity model, m"""

    jgm3_j2: ClassVar[float]
    """ "J2" gravity due oblateness of Earth from JGM3 gravity model, unitless"""

class satstate:
    """Satellite state: position, velocity, optional covariance, and maneuvers

    Bundles a GCRF position/velocity with optional 6x6 covariance and a list
    of impulsive maneuvers into a single propagatable object. Use ``satstate``
    instead of the free :func:`propagate` function when you need:

    - **Covariance propagation** -- attach uncertainty and it propagates
      automatically via the state transition matrix.
    - **Maneuver scheduling** -- add impulsive delta-v events at future times;
      propagation segments around them automatically.
    - **Round-trip propagation** -- propagate forward then backward, recovering
      the original state (maneuvers are reversed).

    For simple state-vector propagation without covariance or maneuvers,
    :func:`propagate` is more direct.

    This class supports ``pickle`` serialization (all fields including
    covariance and maneuvers are preserved).

    Example:
        ```python
        import satkit as sk
        import numpy as np

        # Create state at 500 km altitude
        r = sk.consts.earth_radius + 500e3
        v = np.sqrt(sk.consts.mu_earth / r)
        sat = sk.satstate(sk.time(2024, 1, 1), np.array([r, 0, 0]), np.array([0, v, 0]))

        # Add covariance and maneuver
        sat.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]), frame=sk.frame.LVLH)
        sat.add_prograde(sat.time + sk.duration.from_hours(1), 10.0)

        # Propagate -- covariance and maneuver handled automatically
        new_state = sat.propagate(sat.time + sk.duration.from_hours(3))
        ```
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
            time (satkit.time): Epoch of the state
            pos (npt.NDArray[np.float64]): Position in meters, GCRF frame
            vel (npt.NDArray[np.float64]): Velocity in m/s, GCRF frame
            cov (npt.NDArray[np.float64]|None, optional): 6x6 covariance matrix in GCRF. Defaults to None.

        Example:
            ```python
            t = satkit.time(2024, 1, 1)
            pos = np.array([6.781e6, 0, 0])       # meters, GCRF
            vel = np.array([0, 7.5e3, 0])          # m/s, GCRF
            state = satkit.satstate(t, pos, vel)
            ```
        """
        ...

    @property
    def pos(self) -> npt.NDArray[np.float64]:
        """Position in meters, GCRF frame (alias for pos_gcrf)"""
        ...

    @property
    def vel(self) -> npt.NDArray[np.float64]:
        """Velocity in m/s, GCRF frame (alias for vel_gcrf)"""
        ...

    @property
    def pos_gcrf(self) -> npt.NDArray[np.float64]:
        """Position in meters, GCRF frame"""
        ...

    @property
    def vel_gcrf(self) -> npt.NDArray[np.float64]:
        """Velocity in m/s, GCRF frame"""
        ...

    @property
    def qgcrf2lvlh(self) -> quaternion:
        """Quaternion rotating from GCRF to the LVLH frame for the current state

        LVLH frame:
            - z axis: -r (nadir, pointing toward Earth center)
            - y axis: -h (opposite orbital angular momentum, h = r x v)
            - x axis: completes right-handed system
        """
        ...

    @property
    def cov(self) -> npt.NDArray[np.float64] | None:
        """6x6 state covariance matrix in GCRF, or None if not set

        Upper-left 3x3 is position covariance (m^2), lower-right 3x3 is
        velocity covariance ((m/s)^2), off-diagonal blocks are cross-covariance.
        """
        ...

    @cov.setter
    def cov(self, value: npt.NDArray[np.float64]) -> None:
        """Set the full 6x6 state covariance matrix

        Args:
            value: 6x6 numpy array with state covariance for position (m) and velocity (m/s)
        """
        ...

    @property
    def time(self) -> time:
        """Epoch of this satellite state"""
        ...

    def set_pos_uncertainty(
        self,
        sigma: npt.NDArray[np.float64],
        frame: frame,
    ) -> None:
        """Set 1-sigma position uncertainty in a satellite-local or inertial frame.

        Constructs a diagonal 3x3 covariance from the given 1-sigma values
        (interpreted along the ``frame``'s axes), rotates it into GCRF,
        and stores it as the position block of the 6x6 state covariance.
        Any existing velocity covariance is preserved.

        Args:
            sigma: 3-element numpy array of 1-sigma position components
                along the frame's axes. Units: meters.
            frame: Coordinate frame — **required**, no default (matching
                the Rust API). Supported values:

                - ``frame.GCRF`` — inertial Cartesian
                - ``frame.LVLH`` — Local Vertical / Local Horizontal
                - ``frame.RTN`` — Radial / In-track / Cross-track (= RSW = RTN)
                - ``frame.NTW`` — Normal-to-velocity / Tangent / Cross-track

        Raises:
            RuntimeError: if the frame is not one of the supported frames.

        Example:
            ```python
            # LVLH: 100 m along-track, 200 m cross-track, 50 m nadir
            sat.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]), frame=sk.frame.LVLH)

            # RIC: 10 m radial, 200 m in-track, 30 m cross-track
            sat.set_pos_uncertainty(np.array([10.0, 200.0, 30.0]), frame=sk.frame.RTN)
            ```
        """
        ...

    def set_vel_uncertainty(
        self,
        sigma: npt.NDArray[np.float64],
        frame: frame,
    ) -> None:
        """Set 1-sigma velocity uncertainty in a satellite-local or inertial frame.

        Analogous to :meth:`set_pos_uncertainty`, but for the velocity
        block of the 6x6 state covariance. Any existing position
        covariance is preserved.

        Args:
            sigma: 3-element numpy array of 1-sigma velocity components
                along the frame's axes. Units: m/s.
            frame: Coordinate frame — **required**, no default (matching
                the Rust API). Supported values: ``frame.GCRF``,
                ``frame.LVLH``, ``frame.RTN``, ``frame.NTW``.

        Raises:
            RuntimeError: if the frame is not one of the supported frames.
        """
        ...

    def add_maneuver(
        self,
        time: time,
        delta_v: npt.ArrayLike,
        frame: frame,
    ) -> None:
        """Add an impulsive maneuver (instantaneous delta-v)

        Args:
            time (satkit.time): Time at which to apply the maneuver
            delta_v (array-like): 3-element delta-v vector [m/s]
            frame (satkit.frame): Coordinate frame — **required**, no
                default (matching the Rust API). Supported frames:

                - ``frame.GCRF`` — inertial Cartesian
                - ``frame.RTN`` — radial / in-track / cross-track (a.k.a. RSW, RTN).
                  The I axis is perpendicular to R in the orbit plane — for
                  eccentric orbits this is **not** strictly along velocity.
                - ``frame.NTW`` — normal-to-velocity / tangent / cross-track.
                  The T axis is along velocity, so a pure +T burn of magnitude
                  Δv adds exactly Δv to |v|. Preferred for prograde burns on
                  eccentric orbits.
                - ``frame.LVLH`` — Local Vertical / Local Horizontal (classical
                  crewed-spaceflight frame with z=nadir, y=-h, x=forward).
                  Geometrically equivalent to RIC with relabeled axes; useful
                  when porting GN&C code written in LVLH conventions.

                See the "Theory: Maneuver Coordinate Frames" guide in the satkit
                documentation for a side-by-side comparison.

        See Also:
            :meth:`add_prograde`, :meth:`add_retrograde`, :meth:`add_radial`,
            :meth:`add_normal` for scalar-magnitude helpers that pick the frame
            for you.

        Example:
            ```python
            # Explicit frame selection
            sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.NTW)  # +10 m/s along velocity
            sat.add_maneuver(t_burn, [0, 10, 0], frame=sk.frame.RTN)  # +10 m/s in RIC in-track
            ```
        """
        ...

    def add_prograde(self, time: time, dv_mps: float) -> None:
        """Add a prograde impulsive burn (NTW +T, along velocity).

        A positive ``dv_mps`` adds energy (raises semi-major axis). The burn
        adds exactly ``dv_mps`` to |v| regardless of orbit eccentricity.

        Args:
            time (satkit.time): Time at which to apply the burn
            dv_mps (float): Magnitude along velocity vector [m/s]

        Example:
            ```python
            sat.add_prograde(t_burn, 10.0)  # +10 m/s along velocity
            ```
        """
        ...

    def add_retrograde(self, time: time, dv_mps: float) -> None:
        """Add a retrograde impulsive burn (NTW -T, opposite velocity).

        Equivalent to ``add_prograde`` with a negated magnitude. ``dv_mps``
        should be positive; a positive value removes energy from the orbit.

        Args:
            time (satkit.time): Time at which to apply the burn
            dv_mps (float): Magnitude along anti-velocity vector [m/s]
        """
        ...

    def add_radial(self, time: time, dv_mps: float) -> None:
        """Add a radial-outward impulsive burn (NTW +N axis).

        For circular orbits this is the outward radial direction. For
        eccentric orbits the N axis leans off the radial by the
        flight-path angle.

        Args:
            time (satkit.time): Time at which to apply the burn
            dv_mps (float): Magnitude along in-plane normal-to-velocity [m/s]
        """
        ...

    def add_normal(self, time: time, dv_mps: float) -> None:
        """Add a cross-track ("normal") impulsive burn (NTW +W axis).

        Positive values push in the +angular-momentum direction. Changes
        orbit inclination without altering energy (at apsides).

        Args:
            time (satkit.time): Time at which to apply the burn
            dv_mps (float): Magnitude along angular momentum direction [m/s]
        """
        ...

    @property
    def num_maneuvers(self) -> int:
        """Number of impulsive maneuvers scheduled on this state"""
        ...

    def propagate(
        self,
        time: time | duration,
        *,
        propsettings: propsettings | None = None,
        satproperties: satproperties | None = None,
    ) -> satstate:
        """Propagate this state to a new time

        If covariance is set, it is propagated via the state transition matrix.
        If maneuvers are scheduled between the current and target time, propagation
        automatically segments at each maneuver epoch and applies the delta-v.
        Maneuvers are preserved on the returned state.

        Args:
            time (satkit.time|satkit.duration): Target time, or duration from current time
            propsettings (satkit.propsettings, optional): Propagation settings
            satproperties (satkit.satproperties, optional): Satellite properties (drag, SRP, thrust)

        Returns:
            satstate: New state at the target time

        Example:
            ```python
            sat = sk.satstate(time=t0, pos=r, vel=v)
            sat.add_maneuver(t_burn, [0, 100, 0], frame=sk.frame.RTN)
            new_state = sat.propagate(t_end)
            ```
        """
        ...

class propstats:
    """Statistics of a satellite propagation"""

    @property
    def num_eval(self) -> int:
        """Number of function evaluations"""
        ...

    @property
    def num_accept(self) -> int:
        """Number of accepted steps in adaptive RK integrator"""
        ...

    @property
    def num_reject(self) -> int:
        """Number of rejected steps in adaptive RK integrator"""
        ...

class propresult:
    """Results of a satellite propagation

    This class lets the user access results of the satellite propagation

    Notes:
        If ``enable_interp`` is set to True in the propagation settings,
        the propresult object can be used to interpolate solutions at any
        time between the begin and end times of the propagation via the
        ``interp`` method.
    """

    @property
    def pos(self) -> npt.NDArray[np.float64]:
        """GCRF position of satellite, meters

        Returns:
            3-element numpy array representing GCRF position (meters) at end of propagation

        """
        ...

    @property
    def vel(self) -> npt.NDArray[np.float64]:
        """GCRF velocity of satellite, meters/second

        Returns:
            3-element numpy array representing GCRF velocity in meters/second at end of propagation
        """
        ...

    @property
    def state(self) -> npt.NDArray[np.float64]:
        """6-element end state (pos + vel) of satellite in meters & meters/second

        Returns:
            6-element numpy array representing state of satellite in meters & meters/second
        """
        ...

    @property
    def state_end(self) -> npt.NDArray[np.float64]:
        """6-element state (pos + vel) of satellite in meters & meters/second at end of propagation

        Notes:
        - This is the same as the "state" property

        Returns:
            6-element numpy array representing state of satellite in meters & meters/second
        """
        ...

    @property
    def state_begin(self) -> npt.NDArray[np.float64]:
        """6-element state (pos + vel) of satellite in meters & meters/second at begin of propagation
        Returns:
            6-element numpy array representing state of satellite in meters & meters/second at begin of propagation
        """
        ...

    @property
    def time(self) -> time:
        """Time at which state is valid

        Returns:
            Time at which state is valid
        """
        ...

    @property
    def time_end(self) -> time:
        """Time at which state is valid

        Notes:
        - This is identical to "time" property

        Returns:
            Time at which state is valid
        """
        ...

    @property
    def time_begin(self) -> time:
        """Time at which state_begin is valid


        Returns:
            Time at which state_begin is valid
        """
        ...

    @property
    def stats(self) -> propstats:
        """Statistics of propagation

        Returns:
            propstats: Object containing statistics of propagation
        """
        ...

    @property
    def can_interp(self) -> bool:
        """Whether this result supports interpolation

        Returns:
            True if dense output is available for interpolation
        """
        ...

    @property
    def phi(self) -> npt.NDArray[np.float64] | None:
        """State transition matrix

        Returns:
            6x6 numpy array representing state transition matrix or None if not computed
        """
        ...

    @typing.overload
    def interp(
        self,
        time: time | datetime.datetime,
        output_phi: typing.Literal[False] = False,
    ) -> npt.NDArray[np.float64]:
        """Interpolate state at a single time

        Args:
            time: Time at which to interpolate state
            output_phi: Must be False (default)

        Returns:
            npt.NDArray[np.float64]: 6-element state vector [x, y, z, vx, vy, vz] in meters and m/s
        """
        ...

    @typing.overload
    def interp(
        self,
        time: time | datetime.datetime,
        output_phi: typing.Literal[True] = ...,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate state and state transition matrix at a single time

        Args:
            time: Time at which to interpolate state
            output_phi: Must be True

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: (state, phi) where state is a 6-element
                vector and phi is a 6x6 state transition matrix
        """
        ...

    @typing.overload
    def interp(
        self,
        time: list[time | datetime.datetime],
        output_phi: typing.Literal[False] = False,
    ) -> list[npt.NDArray[np.float64]]:
        """Interpolate state at multiple times

        Args:
            time: List of times at which to interpolate state
            output_phi: Must be False (default)

        Returns:
            list[npt.NDArray[np.float64]]: List of 6-element state vectors
        """
        ...

    @typing.overload
    def interp(
        self,
        time: list[time | datetime.datetime],
        output_phi: typing.Literal[True] = ...,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Interpolate state and state transition matrix at multiple times

        Args:
            time: List of times at which to interpolate state
            output_phi: Must be True

        Returns:
            list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]: List of (state, phi) tuples
        """
        ...

    def interp(
        self,
        time: time | datetime.datetime | list[time | datetime.datetime],
        output_phi: bool = False,
    ):
        """Interpolate state at given time(s)

        Requires ``enable_interp=True`` in propagation settings.

        Args:
            time (time | datetime.datetime | list): Time or list of times at which to interpolate state.
                datetime.datetime objects are interpreted as UTC.
            output_phi (bool): If True, also return the 6x6 state transition matrix. Default is False.

        Returns:
            For a single time: a 6-element state vector, or a ``(state, phi)`` tuple if ``output_phi=True``. For a list of times: a list of state vectors, or a list of ``(state, phi)`` tuples.

        Example:
            ```python
            # After propagation with enable_interp=True
            result = satkit.propagate(state, t0, duration_days=1.0)
            t_mid = t0 + satkit.duration(hours=12)
            mid_state = result.interp(t_mid)
            print(f"Position at 12h: {mid_state[0:3]} m")

            # Interpolate at multiple times
            times = [t0 + satkit.duration(hours=h) for h in range(25)]
            states = result.interp(times)
            ```
        """
        ...

class thrust:
    """Continuous thrust acceleration for orbit maneuvers

    Represents a constant thrust acceleration over a time window,
    specified in GCRF (inertial), RTN (CCSDS-standard orbital frame,
    also known as RSW or RIC), NTW (velocity-aligned), or LVLH.

    RTN components are [R, T, N] where R = radial (outward from Earth
    centre), T = tangential / in-track, N = normal / cross-track (along
    angular momentum, h = r × v).

    Example:

    ```python
    import satkit as sk

    t0 = sk.time(2024, 1, 1)
    t1 = t0 + sk.duration.from_hours(2)

    # In-track thrust in the RTN (a.k.a. RSW, RIC) frame
    t = sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)

    # Fixed direction thrust in GCRF frame
    t = sk.thrust.constant([0, 0, 1e-3], t0, t1, frame=sk.frame.GCRF)
    ```
    """

    @staticmethod
    def constant(
        accel: npt.ArrayLike,
        start: time,
        end: time,
        frame: frame,
    ) -> thrust:
        """Create a constant thrust acceleration

        Args:
            accel (array-like): 3-element acceleration vector [m/s^2]
            start (satkit.time): Start time of thrust arc
            end (satkit.time): End time of thrust arc
            frame (satkit.frame): Coordinate frame — **required**, no
                default (matching the Rust API). Supported values:

                - ``frame.GCRF`` — inertial Cartesian
                - ``frame.RTN`` — radial / in-track / cross-track
                - ``frame.NTW`` — normal-to-velocity / tangent / cross-track
                  (use this for thrust along the velocity vector)
                - ``frame.LVLH`` — Local Vertical / Local Horizontal

        Returns:
            thrust: Thrust object
        """
        ...

    @property
    def accel(self) -> list[float]:
        """Acceleration vector [m/s^2]"""
        ...

    @property
    def frame(self) -> frame:
        """Coordinate frame"""
        ...

    @property
    def start(self) -> time:
        """Start time of thrust arc"""
        ...

    @property
    def end(self) -> time:
        """End time of thrust arc"""
        ...

class satproperties:
    """Satellite properties relevant for drag, radiation pressure, and thrust

    This class lets the satellite radiation pressure, drag,
    and thrust parameters be set for duration of propagation.

    Attributes:
        cdaoverm (float): Coefficient of drag times area over mass in m^2/kg
        craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg
        thrusts (list[thrust]): List of continuous thrust arcs

    """

    def __init__(
        self,
        cdaoverm: float = 0,
        craoverm: float = 0,
        *,
        thrusts: list[thrust] | None = None,
    ) -> None:
        """Create a satproperties object

        Args:
            cdaoverm (float, optional): Coefficient of drag times area over mass in m^2/kg
            craoverm (float, optional): Coefficient of radiation pressure times area over mass in m^2/kg
            thrusts (list[thrust], optional): List of continuous thrust arcs

        Example:

        ```python
        import satkit as sk

        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(2)

        props = sk.satproperties(
            cdaoverm=0.01,
            thrusts=[sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)]
        )
        ```

        """
        ...

    @property
    def cdaoverm(self) -> float:
        """Coefficient of drag times area over mass.  Units are m^2/kg"""
        ...

    @cdaoverm.setter
    def cdaoverm(self, value: float) -> None: ...
    @property
    def craoverm(self) -> float:
        """Coefficient of radiation pressure times area over mass.  Units are m^2/kg"""
        ...

    @craoverm.setter
    def craoverm(self, value: float) -> None: ...

    @property
    def thrusts(self) -> list[thrust]:
        """List of continuous thrust arcs"""
        ...

    @thrusts.setter
    def thrusts(self, value: list[thrust]) -> None: ...

class integrator:
    """Choice of ODE integrator for orbit propagation

    Available integrators, from highest to lowest order:

    - ``rkv98`` - Verner 9(8) with 9th-order dense output, 26 stages (default)
    - ``rkv98_nointerp`` - Verner 9(8) without interpolation, 16 stages
    - ``rkv87`` - Verner 8(7) with 8th-order dense output, 21 stages
    - ``rkv65`` - Verner 6(5), 10 stages
    - ``rkts54`` - Tsitouras 5(4) with FSAL, 7 stages
    - ``rodas4`` - RODAS4 L-stable Rosenbrock 4(3), 6 stages. For stiff problems.
    - ``gauss_jackson8`` - Gauss-Jackson 8, fixed-step multistep predictor-corrector.
      For high-precision long-duration orbit propagation (days to months).

    Higher-order integrators can take larger time steps for the same accuracy,
    so despite having more stages per step, they often require fewer total
    function evaluations. For typical orbit propagation, ``rkv98`` (the default)
    is recommended. For faster but lower-accuracy propagation, ``rkts54`` or
    ``rkv65`` can be used. For stiff problems (re-entry, very low perigee),
    ``rodas4`` is recommended. For long-duration high-precision propagation
    of smooth orbits, ``gauss_jackson8`` typically uses 3-10× fewer force
    evaluations than ``rkv98`` at comparable accuracy — but it requires a
    user-chosen fixed step size (``gj_step_seconds``), does not handle
    discontinuities such as impulsive maneuvers, and needs ≥9 steps of
    startup, so it's unsuitable for very short propagations.
    """

    rkv98: ClassVar[integrator]
    """Verner 9(8) with 9th-order dense output, 26 stages (default)

    Highest accuracy integrator. Recommended for precision orbit propagation.
    """

    rkv98_nointerp: ClassVar[integrator]
    """Verner 9(8) without interpolation, 16 stages

    Same stepping accuracy as ``rkv98`` but skips interpolation stages.
    Slightly faster when dense output is not needed (``enable_interp=False``).
    """

    rkv87: ClassVar[integrator]
    """Verner 8(7) with 8th-order dense output, 21 stages"""

    rkv65: ClassVar[integrator]
    """Verner 6(5), 10 stages"""

    rkts54: ClassVar[integrator]
    """Tsitouras 5(4) with FSAL, 7 stages

    Fastest integrator. Good for quick propagations where high accuracy is not critical.
    """

    rodas4: ClassVar[integrator]
    """RODAS4 — L-stable Rosenbrock 4(3), 6 stages

    Implicit solver for stiff problems such as re-entry or very low perigee orbits.
    Uses analytical Jacobian. Does not support dense output interpolation or
    state transition matrix (``output_phi``) propagation.
    """

    gauss_jackson8: ClassVar[integrator]
    """Gauss-Jackson 8 — 8th-order fixed-step multistep predictor-corrector

    Specialised for 2nd-order ODEs (r'' = f(t, r, v)). The dominant
    integrator in high-precision astrodynamics codes (GMAT, STK, ODTK).
    Typically uses 3-10× fewer force evaluations than ``rkv98`` at
    comparable accuracy on smooth long-duration orbit propagation.

    Uses a fixed step size set via ``propsettings.gj_step_seconds``.
    Supports dense output interpolation (quintic Hermite, 5th-order).
    Does not support state transition matrix (``output_phi``) propagation.
    Not recommended for highly eccentric orbits or integration across
    discontinuities (eclipse boundaries, impulsive maneuvers).
    """

class tidemodel:
    """Solid Earth tide model fidelity for high-precision orbit propagation.

    Solid Earth tides deform the Earth under lunar and solar gravitational
    attraction, perturbing the gravity field. The effect is small (~0.3 m
    position drift over half a day at GEO; ~1 m/day at GPS altitude) but
    matters for sub-meter-class propagation accuracy.

    Implements IERS Conventions 2010, Chapter 6.

    Available models:

    - ``none`` — no solid Earth tide correction
    - ``solid_step1`` — IERS §6.2.1 Step 1, frequency-independent
      Love-number response (default). ≈99% of the total signal.
    - ``solid_full`` — Step 1 + §6.2.2 Step 2 frequency-dependent
      corrections. Step 2 is not yet implemented; currently behaves
      as ``solid_step1``.
    """

    none: ClassVar[tidemodel]
    """No solid Earth tide correction."""

    solid_step1: ClassVar[tidemodel]
    """IERS 2010 §6.2.1 Step 1 — frequency-independent Love-number
    response. Accounts for ≈99% of the solid-tide signal at ~5%
    per-ydot overhead. Default."""

    solid_full: ClassVar[tidemodel]
    """IERS 2010 Step 1 + Step 2 (frequency-dependent corrections).
    Step 2 is not yet implemented; currently behaves as
    ``solid_step1``."""

class propsettings:
    """This class contains settings used in the high-precision orbit propagator part of the "satkit" python toolbox

    Notes:
        - Default settings:
            - abs_error: 1e-8
            - rel_error: 1e-8
            - gravity_degree: 4
            - gravity_order: 4
            - gravity_model: gravmodel.egm96
            - use_spaceweather: True
            - use_sun_gravity: True
            - use_moon_gravity: True
            - tide_model: tidemodel.solid_step1
            - enable_interp: True
            - integrator: integrator.rkv98
            - gj_step_seconds: 60.0
            - max_steps: 1_000_000
        - enable_interp enables high-precision interpolation of state between begin and end times via the returned function,
          it is enabled by default.  There is a small increase in computational efficiency if set to false

    """

    def __init__(
        self,
        *,
        abs_error: float = 1e-8,
        rel_error: float = 1e-8,
        gravity_degree: int = 4,
        gravity_order: int = 4,
        gravity_model: gravmodel = ...,
        use_spaceweather: bool = True,
        use_sun_gravity: bool = True,
        use_moon_gravity: bool = True,
        tide_model: tidemodel = ...,
        enable_interp: bool = True,
        integrator: integrator = ...,
        gj_step_seconds: float = 60.0,
        max_steps: int = 1_000_000,
    ) -> None:
        """Create propagation settings object used to configure high-precision orbit propagator

        Args:
            abs_error: Maximum absolute value of error for any element in propagated state following ODE integration. Default is 1e-8
            rel_error: Maximum relative error of any element in propagated state following ODE integration. Default is 1e-8
            gravity_degree: Maximum degree of spherical harmonic gravity model. Default is 4
            gravity_order: Maximum order of spherical harmonic gravity model. Must be <= gravity_degree. Default is same as gravity_degree
            gravity_model: Gravity model to use. Default is gravmodel.egm96
            use_spaceweather: Use space weather data when computing atmospheric density for drag forces. Default is True
            use_sun_gravity: Include sun third-body gravitational perturbation. Default is True
            use_moon_gravity: Include moon third-body gravitational perturbation. Default is True
            tide_model: Solid Earth tide model. Default is ``tidemodel.solid_step1``
                (IERS 2010 §6.2.1 frequency-independent Love-number response).
                Use ``tidemodel.none`` to disable (e.g., for reproducibility with
                pre-tide releases).
            enable_interp: Store intermediate data that allows for fast high-precision interpolation of state between begin and end times. Default is True
            integrator: ODE integrator to use. Default is integrator.rkv98
            gj_step_seconds: Fixed step size (seconds) used by ``integrator.gauss_jackson8``.
                Ignored by adaptive integrators. Typical values: 30-120 s for LEO, 60-300 s
                for MEO, 300-600 s for GEO. Default is 60.0.
            max_steps: Maximum number of integrator steps before the propagator aborts with
                a max-steps error. Applies to all integrators (adaptive Runge-Kutta, Rosenbrock,
                and Gauss-Jackson 8). Default is 1_000_000, which covers very long propagation
                arcs with plenty of headroom. Lower for a tighter runaway-propagation safeguard.

        Returns:
            propsettings: New propsettings object with default settings

        Example:
            ```python
            settings = satkit.propsettings(
                gravity_degree=16,
                abs_error=1e-10,
                rel_error=1e-10,
                gravity_model=satkit.gravmodel.egm96,
                integrator=satkit.integrator.rkts54,
            )
            ```
        """
        ...

    @property
    def abs_error(self) -> float:
        """Maximum absolute value of error for any element in propagated state following ODE integration

        Returns:
            Maximum absolute value of error for any element in propagated state following ODE integration, default is 1e-8
        """
        ...

    @abs_error.setter
    def abs_error(self, value: float) -> None: ...
    @property
    def rel_error(self) -> float:
        """Maximum relative error of any element in propagated state following ODE integration

        Returns:
            Maximum relative error of any element in propagated state following ODE integration, default is 1e-8

        """
        ...

    @rel_error.setter
    def rel_error(self, value: float) -> None: ...
    @property
    def gravity_degree(self) -> int:
        """Maximum degree of spherical harmonic gravity model

        Returns:
            Maximum degree of spherical harmonic gravity model, default is 4

        """
        ...

    @gravity_degree.setter
    def gravity_degree(self, value: int) -> None: ...
    @property
    def gravity_order(self) -> int:
        """Maximum order of spherical harmonic gravity model

        Returns:
            Maximum order of spherical harmonic gravity model, default is same as gravity_degree

        """
        ...

    @gravity_order.setter
    def gravity_order(self, value: int) -> None: ...
    @property
    def use_sun_gravity(self) -> bool:
        """Include sun third-body gravitational perturbation

        Returns:
            Whether sun gravity is enabled, default is True

        """
        ...

    @use_sun_gravity.setter
    def use_sun_gravity(self, value: bool) -> None: ...
    @property
    def use_moon_gravity(self) -> bool:
        """Include moon third-body gravitational perturbation

        Returns:
            Whether moon gravity is enabled, default is True

        """
        ...

    @use_moon_gravity.setter
    def use_moon_gravity(self, value: bool) -> None: ...
    @property
    def use_spaceweather(self) -> bool:
        """Use space weather data when computing atmospheric density for drag forces

        Notes:

        - Space weather data can have a large effect on the density of the atmosphere
        - This can be important for accurate drag force calculations
        - Space weather data is updated every 3 hours.  Most-recent data can be downloaded with ``satkit.utils.update_datafiles()``
        - Default value is True

        Returns:
            Indicate whether or not space weather data should be used when computing atmospheric density for drag forces

        """
        ...

    @use_spaceweather.setter
    def use_spaceweather(self, value: bool) -> None: ...
    @property
    def tide_model(self) -> tidemodel:
        """Solid Earth tide model fidelity.

        Default is ``tidemodel.solid_step1`` (IERS 2010 §6.2.1
        frequency-independent Love-number response). Set to
        ``tidemodel.none`` to disable.
        """
        ...

    @tide_model.setter
    def tide_model(self, value: tidemodel) -> None: ...
    @property
    def enable_interp(self) -> bool:
        """Store intermediate data that allows for fast high-precision interpolation of state between begin and end times
        If not needed, there is a small computational advantage if set to False
        """
        ...

    @enable_interp.setter
    def enable_interp(self, value: bool) -> None: ...
    @property
    def gravity_model(self) -> gravmodel:
        """Gravity model used for Earth gravity computation

        Returns:
            gravmodel: The gravity model, default is gravmodel.egm96

        """
        ...

    @gravity_model.setter
    def gravity_model(self, value: gravmodel) -> None: ...
    @property
    def integrator(self) -> integrator:
        """ODE integrator used for orbit propagation

        Returns:
            integrator: The integrator, default is integrator.rkv98

        """
        ...

    @integrator.setter
    def integrator(self, value: integrator) -> None: ...
    @property
    def gj_step_seconds(self) -> float:
        """Fixed step size (seconds) used by ``integrator.gauss_jackson8``.

        Ignored by adaptive integrators. Typical values: 30-120 s for LEO,
        60-300 s for MEO, 300-600 s for GEO.

        Returns:
            Fixed step size in seconds, default is 60.0
        """
        ...

    @gj_step_seconds.setter
    def gj_step_seconds(self, value: float) -> None: ...
    @property
    def max_steps(self) -> int:
        """Maximum number of integrator steps before the propagator aborts.

        Applies to all integrators (adaptive Runge-Kutta, Rosenbrock, and
        Gauss-Jackson 8). Increase for very long propagation arcs or tight
        tolerances.

        Returns:
            Maximum steps, default is 1_000_000
        """
        ...

    @max_steps.setter
    def max_steps(self, value: int) -> None: ...
    def precompute_terms(self, begin: time, end: time, step: Optional[Union[duration, float, datetime.timedelta]] = None):
        """Precompute terms for fast interpolation of state between begin and end times

        This can be used, for example, to compute sun and moon positions only once if propagating many satellites over the same time period

        Args:
            begin (satkit.time): Begin time of propagation
            end (satkit.time): End time of propagation
            step (satkit.duration | float | datetime.timedelta, optional): Step size for interpolation.  Default = 60 seconds.  float is interpreted as seconds.

        """
        ...

def lambert(
    r1: npt.NDArray[np.float64],
    r2: npt.NDArray[np.float64],
    tof: float,
    mu: float | None = None,
    prograde: bool | None = None,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Solve Lambert's problem using Izzo's algorithm (2015).

    Given two position vectors and a time of flight, find the velocity vectors
    for transfer orbits connecting them.

    Args:
        r1: 3-element numpy array — departure position (meters)
        r2: 3-element numpy array — arrival position (meters)
        tof: Time of flight in seconds (must be positive)
        mu: Gravitational parameter in m³/s² (default: Earth µ = 3.986e14)
        prograde: If True (default), prograde transfer; if False, retrograde

    Returns:
        List of (v1, v2) tuples. Each v1 and v2 is a 3-element numpy array
        in m/s. The first element is the zero-revolution solution; additional
        elements are multi-revolution solutions if they exist.

    Raises:
        ValueError: If inputs are invalid (negative tof, zero position, etc.)

    Example:
        ```python
        import satkit
        import numpy as np

        r1 = np.array([7000e3, 0, 0])
        r2 = np.array([0, 7000e3, 0])
        solutions = satkit.lambert(r1, r2, 3600.0)
        v1, v2 = solutions[0]
        ```
    """
    ...

def propagate(
    state: npt.NDArray[np.float64],
    begin: time,
    end: time | None = None,
    *,
    duration: duration | None = None,
    duration_secs: float | None = None,
    duration_days: float | None = None,
    output_phi: bool = False,
    propsettings: propsettings | None = None,
    satproperties: satproperties | None = None,
) -> propresult:
    """High-precision orbit propagator

    Propagate orbits with high-precision force modeling via adaptive Runge-Kutta methods (default is order 9/8).

    Args:
        state: 6-element numpy array representing satellite GCRF position and velocity in meters and meters/second
        begin: Time at which satellite is at input state

    Keyword Args:
        end: Time at which new position and velocity will be computed
        duration: Duration from ``begin`` at which new position & velocity will be computed
        duration_secs: Duration in seconds from ``begin`` at which new position and velocity will be computed
        duration_days: Duration in days from ``begin`` at which new position and velocity will be computed
        output_phi: Output 6x6 state transition matrix between begin and end times. Default is False
        propsettings: Settings for the propagation; if omitted, defaults are used
        satproperties: Drag and radiation pressure susceptibility of satellite

    Returns:
        propresult: Propagation result object holding state outputs, statistics,
            and dense output if requested

    Notes:
        Propagates satellite ephemeris (position, velocity in GCRF & time) to new time and
        outputs new position and velocity via Runge-Kutta integration.
        Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF).

        Included forces:

        - Earth gravity with higher-order zonal terms
        - Sun, Moon gravity
        - Radiation pressure
        - Atmospheric drag: NRL-MSISE 2000 density model, with option to include space weather effects

        End time must be set by keyword argument, either explicitly or by duration.
        Solid Earth tides are not (yet) included in the model.

        For future propagation (beyond available data files):

        - Earth orientation parameters use the last available values (constant extrapolation)
        - Space weather uses the NOAA/SWPC solar cycle forecast for predicted F10.7 values;
          Ap defaults to 4. If no forecast is available, F10.7 defaults to 150.

    Example:
        ```python
        import numpy as np

        # Define initial state in GCRF (position in meters, velocity in m/s)
        state = np.array([6.781e6, 0, 0, 0, 7.5e3, 0])
        t0 = satkit.time(2024, 1, 1)

        # Propagate forward by 1 day
        result = satkit.propagate(state, t0, duration_days=1.0)
        print(f"End position: {result.pos} m")
        print(f"End velocity: {result.vel} m/s")

        # Interpolate at intermediate time
        t_mid = t0 + satkit.duration(hours=12)
        mid_state = result.interp(t_mid)
        ```
    """
    ...

def omm_from_url(url: str) -> list[dict]:
    """Load OMM(s) from a URL as a list of dictionaries

    Fetches the content at the given URL and auto-detects JSON vs XML format.
    Returns a list of dictionaries that can be passed directly to :func:`sgp4`.

    Args:
        url (str): URL to fetch OMM data from (e.g. CelesTrak or Space-Track endpoint)

    Returns:
        list[dict]: List of OMM dictionaries with standard CCSDS keys
            (OBJECT_NAME, EPOCH, MEAN_MOTION, ECCENTRICITY, etc.)

    Example:
        ```python
        import satkit as sk

        omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")
        pos, vel = sk.sgp4(omms[0], sk.time(2024, 1, 1))
        ```
    """
    ...
