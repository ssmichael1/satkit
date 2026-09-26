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
from typing import Any, ClassVar, Optional, TypeAlias, Union, overload
from typing_extensions import deprecated

from ._types import OMMDict

# Time inputs are polymorphic: anywhere a ``satkit.time`` is accepted, a
# ``datetime.datetime`` is accepted interchangeably (and, for the vectorized
# functions, a list or numpy array of either). These aliases capture that so
# the individual signatures stay readable.
#
# A ``datetime.datetime`` is converted with Python's own convention
# (``datetime.timestamp()``): a naive datetime is the machine's local time, an
# aware one uses its own offset (see ``time.from_datetime``).
#
# * ``TimeScalar``    — a single time value.
# * ``TimeArrayLike`` — a list or numpy array of time values.
# * ``TimeInput``     — either a scalar or an array of times.
TimeScalar: TypeAlias = "time | datetime.datetime"
TimeArrayLike: TypeAlias = "list[time] | list[datetime.datetime] | npt.ArrayLike"
TimeInput: TypeAlias = "TimeScalar | TimeArrayLike"

class TLE:
    """Two-Line Element Set (TLE) representing a satellite ephemeris

    A Two-Line Element Set is a satellite ephemeris format from the 1970s
    that is still in wide use. Its mean elements are propagated with the
    "Simplified General Perturbations-4" (SGP4) model (``satkit.sgp4``),
    which gives position and velocity in the "TEME" frame (not-quite GCRF).

    For details, see: <https://en.wikipedia.org/wiki/Two-line_element_set>

    Catalogs in this format are publicly available at
    <https://www.space-track.org> (registration required) and
    <https://celestrak.org> (no registration needed).

    TLEs sometimes have a "line 0" that includes the name of the satellite.

    Load TLEs with ``TLE.from_lines``, ``TLE.from_file`` or ``TLE.from_url``;
    each returns a ``list[TLE]``, even for a single element set.

    Example:
        ```python
        tle = satkit.TLE.from_lines([
            "0 ISS (ZARYA)",
            "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9991",
            "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302878",
        ])[0]
        print(tle.name)
        # ISS (ZARYA)
        ```
    """

    @staticmethod
    def from_file(filename: str, *, check_checksum: bool = False) -> list[TLE]:
        """Load TLEs from a text file, parsed as :meth:`TLE.from_lines` does

        Args:
            filename (str): name of the text file holding the TLE lines
            check_checksum (bool, optional): as in :meth:`TLE.from_lines`

        Returns:
            list[TLE]: one TLE per element set in the file

        Raises:
            ValueError: if the file holds no TLEs
            RuntimeError: if a record fails to parse; see :meth:`TLE.from_lines`

        Example:
            ```python
            tles = satkit.TLE.from_file("gps-ops.txt")
            for tle in tles:
                print(tle.name, tle.satnum)
            ```
        """
        ...

    @staticmethod
    def from_lines(lines: Sequence[str], *, check_checksum: bool = False) -> list[TLE]:
        """Load TLEs from a list of lines

        :meth:`TLE.from_file` and :meth:`TLE.from_url` parse their text the
        same way. See the class docstring for an example.

        Args:
            lines (Sequence[str]): the TLE lines (2-line or 3-line format,
                any number of element sets; any sequence type is accepted,
                e.g. list or tuple)
            check_checksum (bool, optional): also verify the checksum digit
                (column 69) of every data line. Default False.

        Returns:
            list[TLE]: one TLE per element set, even if there is only one

        Raises:
            ValueError: if the lines hold no TLEs
            RuntimeError: if a record fails to parse (or, with
                ``check_checksum``, has a wrong checksum); the message gives
                the line the record starts on and its satellite
        """
        ...

    @staticmethod
    def from_url(url: str) -> list[TLE]:
        """Load TLEs from a URL, parsing the response as :meth:`TLE.from_lines` does

        Works with any URL that returns plain-text TLE data.

        Args:
            url (str): URL to fetch TLE data from

        Returns:
            list[TLE]: one TLE per element set in the response

        Raises:
            ValueError: if the response holds no TLEs
            RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
                ``satkit.utils.set_offline(True)``; no connection is opened),
                the request fails, or a record fails to parse

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
    def intl_desig(self) -> str:
        """International designator (e.g. "98067A": launch year, launch number, piece)"""
        ...

    @intl_desig.setter
    def intl_desig(self, value: str) -> None: ...
    @property
    def desig_year(self) -> int:
        """Launch year from the international designator (2-digit, as in the TLE)"""
        ...

    @desig_year.setter
    def desig_year(self, value: int) -> None: ...
    @property
    def desig_launch(self) -> int:
        """Launch number of the year from the international designator"""
        ...

    @desig_launch.setter
    def desig_launch(self, value: int) -> None: ...
    @property
    def desig_piece(self) -> str:
        """Piece of the launch from the international designator (e.g. "A")"""
        ...

    @desig_piece.setter
    def desig_piece(self, value: str) -> None: ...
    @property
    def ephem_type(self) -> int:
        """Ephemeris type (usually 0). A value of 4 marks an SGP4-XP element set, which ``sgp4`` rejects: satkit implements classic SGP4 only."""
        ...

    @ephem_type.setter
    def ephem_type(self, value: int) -> None: ...
    @property
    def element_num(self) -> int:
        """Element set number"""
        ...

    @element_num.setter
    def element_num(self, value: int) -> None: ...
    @property
    def rev_num(self) -> int:
        """Revolution number at epoch"""
        ...

    @rev_num.setter
    def rev_num(self, value: int) -> None: ...

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
        """Satellite eccentricity, unitless, in range [0,1]"""
        ...

    @eccen.setter
    def eccen(self, value: float) -> None:
        """Set the satellite eccentricity, unitless, in range [0,1]"""
        ...

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly in degrees"""
        ...

    @mean_anomaly.setter
    def mean_anomaly(self, value: float) -> None:
        """Set the satellite mean anomaly, degrees"""
        ...

    @property
    def mean_motion(self) -> float:
        """Mean motion in revs / day"""
        ...

    @mean_motion.setter
    def mean_motion(self, value: float) -> None:
        """Set the satellite mean motion, revs / day"""
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
    def epoch(self, value: TimeScalar) -> None:
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
        """Drag term (B*) of the satellite, in units of 1 / Earth radii

        should be rho0 * Cd * A / 2 / m

        Units (which are strange) is multiples of
        1 / Earth radius
        """
        ...

    @bstar.setter
    def bstar(self, value: float) -> None:
        """Set the drag term (B*) of the satellite, in units of 1 / Earth radii"""
        ...

    @staticmethod
    def from_omm(omm: OMMDict) -> TLE:
        """Build a TLE from an OMM (Orbital Mean-Element Message) dictionary

        The dictionary is the same shape :func:`sgp4` accepts: the flat CCSDS
        keys of a Space-Track / CelesTrak JSON record (see :class:`OMMDict`),
        or the nested ``meanElements`` / ``tleParameters`` groups of an
        XML-derived dict. Numbers may be strings.

        The six mean elements, epoch, ``BSTAR``, ``MEAN_MOTION_DOT``,
        ``MEAN_MOTION_DDOT``, ``NORAD_CAT_ID``, ``ELEMENT_SET_NO``,
        ``REV_AT_EPOCH`` and ``EPHEMERIS_TYPE`` carry over; absent optional
        values become zero. ``OBJECT_ID`` in ``YYYY-NNNP`` form becomes the
        international designator. Other metadata is dropped.

        Args:
            omm (OMMDict): OMM dictionary

        Returns:
            TLE: the equivalent two-line element set

        Example:
            ```python
            omm = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=json")[0]
            tle = sk.TLE.from_omm(omm)
            print("\\n".join(tle.to_2line()))
            ```
        """
        ...

    def to_omm(self) -> OMMDict:
        """Render this TLE as an OMM (Orbital Mean-Element Message) dictionary

        The result uses the flat CCSDS keys (see :class:`OMMDict`) with
        ``EPOCH`` as an RFC 3339 string, angles in degrees and mean motion in
        revolutions per day, and can be passed back to :func:`sgp4` or
        serialized with ``json.dumps``. ``OBJECT_ID`` is derived from the
        international designator (``98067A`` becomes ``1998-067A``). The TLE
        carries no classification letter, so ``CLASSIFICATION_TYPE`` is absent.

        Returns:
            OMMDict: OMM dictionary
        """
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
        times: TimeArrayLike,
        epoch: TimeScalar,
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
    tle: TLE | OMMDict | list[TLE | OMMDict],
    time: TimeInput,
    *,
    gravconst: sgp4_gravconst = ...,
    opsmode: sgp4_opsmode = ...,
    errflag: bool = False,
) -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int32]]
):
    """SGP-4 propagator for TLE

    Run Simplified General Perturbations (SGP)-4 propagator on Two-Line Element Set to
    output satellite position and velocity at given time
    in the "TEME" coordinate system.

    A detailed description is in Vallado, Crawford, Hujsak & Kelso,
    "Revisiting Spacetrack Report #3", AIAA 2006-6753:
    <https://doi.org/10.2514/6.2006-6753>
    <https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf>

    Args:
        tle (TLE | OMMDict | list[TLE | OMMDict]): element set(s) to propagate: a
            ``TLE`` object, an OMM dictionary (see :class:`OMMDict`), or a list mixing both
        time (time | list[time] | list[datetime.datetime] | npt.ArrayLike[time] | npt.ArrayLike[datetime.datetime]): time(s) at which to compute position and velocity.
            A naive ``datetime`` is local time, not UTC (see :meth:`time.from_datetime`)

    Keyword Args:
        gravconst (satkit.sgp4_gravconst): gravity constant to use.  Default is gravconst.wgs72
        opsmode (satkit.sgp4_opsmode): opsmode.afspc (Air Force Space Command) or opsmode.improved.  Default is opsmode.afspc
        errflag (bool): whether or not to output error conditions for each TLE and time output.  Default is False
                        (this is likely rarely needed, but can be useful for debugging)

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: position and velocity
            in **meters** and **meters/second**, respectively,
            in the TEME frame at each of the "Ntime" input times and each of the "Ntle" tles.
            Shape is (3,) for a single TLE and single time, (Ntime, 3) for a single TLE
            and multiple times, (Ntle, 3) for a list of TLEs and a single time, and
            (Ntle, Ntime, 3) for a list of TLEs and multiple times.
            A list keeps its axis even with one element ("list in, list out"):
            ``sgp4([tle], t)`` is (1, 3) and ``sgp4([tle], [t])`` is (1, 1, 3).
            Empty lists give empty arrays of the matching shape, e.g.
            ``sgp4(tle, [])`` is (0, 3) and ``sgp4([tle], [])`` is (1, 0, 3).
            If errflag is True, a third element is returned: an ``int32`` numpy
            array of error codes, one per TLE and time — shape ``(1,)`` for a single
            TLE and time, ``(Ntime,)``, ``(Ntle,)`` or ``(Ntle, Ntime)`` otherwise.
            ``0`` is success. The codes are the integer values of
            :class:`sgp4_error`, so comparing against the enum works
            elementwise: ``err == satkit.sgp4_error.success`` is a boolean array.

    Notes:
        - **Units:** the canonical Vallado SGP4 implementation (and most other SGP4
          libraries) return position in kilometers and velocity in kilometers/second.
          satkit converts these to meters and meters/second so that SGP4 output is
          consistent with every other position and velocity in the library.
        - **OMM dictionaries:** any dict with the CCSDS keys ``EPOCH``, ``MEAN_MOTION``,
          ``ECCENTRICITY``, ``INCLINATION``, ``RA_OF_ASC_NODE``, ``ARG_OF_PERICENTER`` and
          ``MEAN_ANOMALY`` (plus optional ``BSTAR``, ``MEAN_MOTION_DOT``, ``MEAN_MOTION_DDOT``)
          is accepted, whether it came from :func:`omm_from_url` / :func:`omm_from_file` /
          :func:`omm_from_text`, from ``json.load`` on a CelesTrak or Space-Track response
          (numbers may be strings), or from ``xmltodict`` on the XML form (the nested
          ``meanElements`` / ``tleParameters`` groups are understood). ``EPOCH`` may be an
          RFC 3339 string, a ``satkit.time`` or a ``datetime`` (naive = local time). Other keys are ignored,
          except that ``MEAN_ELEMENT_THEORY`` must be ``SGP4``, ``TIME_SYSTEM`` must be
          ``UTC`` and ``EPHEMERIS_TYPE`` must not be 4 (SGP4-XP) when present.
        - The "TEME" frame of the SGP4 state vectors is not a truly inertial frame.  It is a "True Equator Mean Equinox"
          frame, which is a non-rotating frame with respect to the mean equator and mean equinox of the epoch of the TLE.
          It is close to a true inertial frame, but can be offset by small amounts due to precession and nutation.

    Example:
        ```python
        import numpy as np
        import satkit

        lines = [
            "0 INTELSAT 902",
            "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290",
            "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981",
        ]

        tle = satkit.TLE.from_lines(lines)[0]  # from_lines always returns a list
        tm = tle.epoch

        # Compute TEME position & velocity at epoch
        pteme, vteme = satkit.sgp4(tle, tm)

        # Rotate to ITRF frame; the velocity also loses the Earth-rotation term
        q = satkit.frametransform.qteme2itrf(tm)
        pitrf = q * pteme
        vitrf = q * vteme - np.cross(np.array([0, 0, satkit.consts.omega_earth]), pitrf)

        # convert to ITRF coordinate object
        coord = satkit.itrfcoord(pitrf)

        # Print ITRF coordinate object location
        print(coord)
        # ITRFCoord(lat:  -0.0362 deg, lon:  62.0172 deg, hae: 35799.52 km)

        # Error codes per output
        pteme, vteme, err = satkit.sgp4(tle, tm, errflag=True)
        assert (err == satkit.sgp4_error.success).all()
        ```


        ```python
        # Query the OMM for the International Space Station (ISS)
        url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=json"
        omm = satkit.omm_from_url(url)[0]
        # Get a representative time from the output
        epoch = satkit.time(omm["EPOCH"])
        # Compute TEME position & velocity at epoch
        pteme, vteme = satkit.sgp4(omm, epoch)
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

    improved: ClassVar[int]
    """improved"""

class gravmodel:
    """
    Earth gravity models available for use

    For details, see: <http://icgem.gfz-potsdam.de/>

    ``egm96``, ``egm2008``, ``jgm2`` and ``jgm3`` are compiled into satkit
    (to degree 70) and need no data directory or network; ``itugrace16``
    is downloaded on first use. Each model's tide system (tide-free or
    zero-tide C20) is read when it is loaded and the propagator's
    ``tidemodel.solid_step1`` correction accounts for it, so any model can
    be combined with any tide model without double-counting the permanent
    tide.
    """

    jgm3: ClassVar[gravmodel]
    """
    The "JGM3" gravity model, Tapley et al. (1996). Zero-tide C20.
    Compiled in.
    """

    jgm2: ClassVar[gravmodel]
    """
    The "JGM2" gravity model, Nerem et al. (1994). Tide-free C20.
    Compiled in.
    """

    egm96: ClassVar[gravmodel]
    """
    The "EGM96" gravity model, Lemoine et al. (1998). Tide-free C20.
    Compiled in. Default for the orbit propagator.
    """

    itugrace16: ClassVar[gravmodel]
    """
    The ITU_GRACE16 gravity model, Akyilmaz et al. (2016), a GRACE-only
    satellite solution. Zero-tide C20. Licensed CC BY 4.0, so it is not
    compiled in: the coefficient file (1.8 MB) is downloaded into the data
    directory on first use, and selecting it offline raises
    ``RuntimeError``. Results derived from it should cite the model.
    """

    egm2008: ClassVar[gravmodel]
    """
    The "EGM2008" gravity model, Pavlis et al. (2012). Tide-free C20.
    Compiled in (truncated to degree 70).
    """

def nrlmsise00(
    alt_km: float,
    *,
    latitude_deg: float = 0.0,
    longitude_deg: float = 0.0,
    time: TimeScalar | None = None,
    use_spaceweather: bool = True,
) -> tuple[float, float]:
    """NRL-MSISE00 atmospheric density model

    Args:
        alt_km (float): Altitude in kilometers

    Keyword Args:
        latitude_deg (float): Latitude in degrees
        longitude_deg (float): Longitude in degrees
        time (satkit.time | datetime.datetime, optional): Time at which to
            evaluate the model; space-weather data at this time is used when
            ``use_spaceweather`` is True
        use_spaceweather (bool): Use the space-weather database in the
            calculation. Default is True

    Returns:
        tuple[float, float]: Density (kg/m^3) and temperature (K)
    """
    ...

def gravity(
    pos: list[float] | itrfcoord | npt.ArrayLike,
    *,
    model: gravmodel = ...,
    degree: int = 6,
    order: int | None = None,
) -> npt.NDArray[np.float64]:
    """Return acceleration due to Earth gravity at the input position

    Args:
        pos (list[float] | satkit.itrfcoord | npt.ArrayLike[np.float]): Position as ITRF coordinate or numpy 3-vector representing ITRF position in meters

    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.egm2008
        degree (int): Maximum degree of gravity model to use.  Default is 6, maximum is 70
        order (int): Maximum order of gravity model to use.  Default is same as degree

    Returns:
        acceleration in m/s^2 in the International Terrestrial Reference Frame (ITRF)


    Notes:
        - For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and E. Gill, Springer, 2000 (https://doi.org/10.1007/978-3-642-58351-3).

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
    pos: itrfcoord | npt.ArrayLike,
    *,
    model: gravmodel = ...,
    degree: int = 6,
    order: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gravity and partial derivatives of gravity with respect to Cartesian coordinates

    Args:
        pos (itrfcoord | npt.ArrayLike[np.float]): Position as ITRF coordinate or numpy 3-vector representing ITRF position in meters


    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.egm2008
        degree (int): Maximum degree of gravity model to use.  Default is 6, maximum is 70
        order (int): Maximum order of gravity model to use.  Default is same as degree

    Returns:
        acceleration in m/s^2 and partial derivative of acceleration with respect to ITRF Cartesian coordinate in m/s^2 / m


    For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and E. Gill, Springer, 2000 (https://doi.org/10.1007/978-3-642-58351-3).

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

    Invalid: ClassVar[weekday]
    """Invalid weekday"""

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

    @property
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

    The scales follow Chapter 10 of the IERS Conventions (2010), IERS Technical
    Note 36 (<https://iers-conventions.obspm.fr/content/tn36.pdf>); UTC and leap
    seconds are defined by ITU-R TF.460-6. For an excellent overview, see:
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
    Python-level aliases ``frame.RSW`` and ``frame.RIC`` resolve to the
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
        - UTC before 1972 follows the "rubber second" model of USNO
          ``tai-utc.dat`` / ERFA ``dat`` from 1961-01-01: TAI - UTC drifts
          linearly and steps by fractions of a second (positive steps are
          labelled ``23:59:60.x``). Before 1961, UTC is taken to equal TAI
          (unlike ERFA, which also models 1960).

    Example:
        ```python
        print(satkit.time(2023, 3, 5, 11, 3, 45.453))
        # 2023-03-05T11:03:45.453000Z

        print(satkit.time(2023, 3, 5))
        # 2023-03-05T00:00:00.000000Z
        ```

    """

    J2000: ClassVar[time]
    """The J2000 epoch: 2000-01-01 12:00:00 TT"""

    GPS_EPOCH: ClassVar[time]
    """The GPS epoch: 1980-01-06 00:00:00 UTC"""

    MJD_EPOCH: ClassVar[time]
    """The Modified Julian Date epoch: 1858-11-17 00:00:00 UTC"""

    UNIX_EPOCH: ClassVar[time]
    """The Unix epoch: 1970-01-01 00:00:00 UTC"""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, string: str, /) -> None: ...
    @overload
    def __init__(self, year: int, month: int, day: int, /, *, scale: timescale = ...) -> None: ...
    @overload
    def __init__(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        min: int,
        sec: float,
        /,
        *,
        scale: timescale = ...,
    ) -> None:
        """Create a time object representing input date and time

        This has functionality similar to the "datetime" object, and in fact has
        the ability to convert to an from the "datetime" object.  However, a separate
        time representation is needed as the "datetime" object does not allow for
        conversion between various time epochs (GPS, TAI, UTC, UT1, etc...)

        Accepted forms (all positional):

        - ``time()``: the current time
        - ``time(string)``: parse a string, RFC 3339 first (e.g.
          ``"2023-03-05T11:03:45.453Z"``), then other common formats
        - ``time(year, month, day)``: midnight at the start of the day
        - ``time(year, month, day, hour, min, sec)``: all six components

        Args:
            string: String representation of time
            year: Gregorian year (e.g., 2024)
            month: Gregorian month (1 = January, 2 = February, ...)
            day: Day of month, beginning with 1
            hour: Hour of day, in range [0,23]
            min: Minute of hour, in range [0,59]
            sec: Floating point second of minute, in range [0,60); up to 61
                within a UTC leap second (e.g. ``23:59:60.5`` on 2016-12-31),
                and within a pre-1972 positive UTC step (e.g. ``23:59:60.05``
                on 1963-10-31, up to 61.422818 on 1960-12-31). Rounded to the
                nearest microsecond, so the seconds from ``to_gregorian()``
                round-trip exactly
            scale: Time scale in which the Gregorian components are
                interpreted, default is satkit.timescale.UTC. Ignored for the
                string and no-argument forms.

        Raises:
            ValueError: If the string form cannot be parsed

        Example:
            ```python
            print(satkit.time(2023, 3, 5, 11, 3, 45.453))
            # 2023-03-05T11:03:45.453000Z

            print(satkit.time(2023, 3, 5))
            # 2023-03-05T00:00:00.000000Z
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
    def from_string(string: str) -> time:
        """
        Create a "time" object from a string, guessing its format

        RFC 3339 is tried first (see :meth:`from_rfc3339`). Otherwise the
        numbers in the string are read in year, month, day, hour, minute,
        second order, so ISO-ordered strings (``"2024-01-04 13:14:12.123"``)
        and month-name strings (``"March 4 2024"``) work, but locale-ordered
        dates such as ``MM/DD/YYYY`` are not supported: use :meth:`strptime`
        for those.

        Args:
            string: String representation of time

        Notes:
            - A number after ``.`` following the seconds is the fraction of a
              second, rounded to the nearest microsecond.
            - A number after ``+`` or ``-`` following the minutes is a UTC
              offset (``±HHMM``, ``±HH:MM`` or ``±HH``; hours 00-23, minutes
              00-59) and is applied: ``"2024-01-04 13:14:12 +0100"`` is
              ``12:14:12Z``. Any other extra number is an error.
            - Seconds default to 0 (``"2024-01-04 13:14"``); an hour without
              minutes is an error, and a date alone is midnight.
            - Words other than month names (weekday names, ``T``, ``Z``,
              ``UTC``, ...) are ignored, so a zone *name* is not applied:
              without a numeric offset the time is UTC.
            - This is probably not what you want. Use with caution, and prefer
              :meth:`from_rfc3339` or :meth:`strptime` when the format is known.

        Returns:
            Time object representing input string

        Raises:
            ValueError: If the string cannot be parsed; the message gives the
                reason

        Example:
            ```python
            print(satkit.time.from_string("2023-03-05 11:03:45.453Z"))
            # 2023-03-05T11:03:45.453000Z
            ```
        """
        ...

    @staticmethod
    def from_rfc3339(rfc3339: str) -> time:
        """Create a "time" object from an RFC 3339 string

        Args:
            rfc3339: RFC 3339 string representation of time

        Notes:
            - Format ``YYYY-MM-DDTHH:MM:SS[.fff...][zone]``. ``T`` may be
              ``t``. The fraction has one or more digits and is rounded to the
              nearest microsecond beyond six. Surrounding whitespace is
              ignored; anything else left over is an error.
            - The zone is ``Z`` / ``z``, or a UTC offset ``±HH:MM`` (RFC 3339),
              ``±HHMM`` or ``±HH`` (ISO 8601 forms, also accepted), with hours
              00-23 and minutes 00-59. The offset is applied:
              ``2024-01-01T12:00:00+01:00`` is ``11:00:00Z``. It shifts the
              calendar label, so it is exact across a leap second.
            - Without a zone the time is taken as UTC (RFC 3339 itself
              requires one).
            - The year is four digits, or a sign and at least four digits
              (ISO 8601 expanded years such as ``-0001`` or ``+10000``, as
              :meth:`to_rfc3339` writes them outside 0000-9999).

        Returns:
            Time object representing input RFC 3339 string

        Raises:
            ValueError: If the string cannot be parsed; the message gives the
                reason

        Example:
            ```python
            print(satkit.time.from_rfc3339("2023-03-05T11:03:45.453Z"))
            # 2023-03-05T11:03:45.453000Z

            print(satkit.time.from_rfc3339("2023-03-05T12:03:45.453+01:00"))
            # 2023-03-05T11:03:45.453000Z
            ```
        """
        ...

    @staticmethod
    def strptime(date_string: str, format: str) -> time:
        """
        Create a "time" object from input string with given formatting

        Args:
            date_string: string representation of time
            format: format of the string

        Notes:
            - The format string is a subset of the strptime format string in
              the Python "datetime" module. Characters other than format codes
              must match literally, and the whole string must be consumed:
              leftover input is an error.
            - Format Codes:
                - %Y - year: exactly four digits, or a sign and at least four
                  digits (ISO 8601 expanded years such as ``-0044`` or
                  ``+10000``, as :meth:`strftime` writes them outside 0000-9999)
                - %m - month, exactly two digits (01-12)
                - %d - day of month, exactly two digits (01-31)
                - %H - hour, exactly two digits (00-23)
                - %M - minute, exactly two digits (00-59)
                - %S - second, exactly two digits (00-59, or 60 in a leap second)
                - %f - fraction of a second: one or more digits (``5`` is
                  500 ms), rounded to the nearest microsecond beyond six
                - %b - abbreviated month name (Jan, Feb, ...)
                - %B - full month name (January, February, ...)
                - %z - UTC offset ``±HH:MM``, ``±HHMM`` or ``±HH`` (exactly two
                  digits per field, hours 00-23, minutes 00-59), or ``Z`` /
                  ``z`` for UTC. ``+HHMM`` means local time is ahead of UTC,
                  so ``12:00:00+0100`` is ``11:00:00Z``; the offset shifts the
                  calendar label, so it is exact across a leap second
                - %% - a literal ``%``

        Returns:
            Time object representing input string

        Raises:
            ValueError: If the string does not match the format; the message
                gives the reason

        Example:
            ```python
            # %f reads any number of fraction digits: ".453" is 453 ms
            print(satkit.time.strptime("2023-03-05 11:03:45.453Z", "%Y-%m-%d %H:%M:%S.%fZ"))
            # 2023-03-05T11:03:45.453000Z
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
            # 2023-06-15T00:00:00.000000Z
            ```
        """
        ...

    @staticmethod
    def from_jd(jd: float, scale: timescale = timescale.UTC) -> time:
        """Return a time object representing input Julian date and time scale

        Args:
            jd (float): Julian date, days
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
    def from_unixtime(unixtime: float) -> time:
        """Return a time object representing input unixtime

        Args:
            ut (float): unixtime, UTC seconds since Jan 1, 1970 00:00:00
                        (leap seconds are not included); rounded to the
                        nearest microsecond

        Returns:
            Time object representing input unixtime

        Example:
            ```python
            t = satkit.time.from_unixtime(1700000000)
            print(t)
            # 2023-11-14T22:13:20.000000Z
            ```
        """
        ...

    @staticmethod
    def from_gps_week_and_second(week: int, seconds: float) -> time:
        """Return a time object representing input GPS week and second

        Args:
            week: GPS week number
            seconds: GPS seconds of week, seconds

        Returns:
            Time object representing input GPS week and second
        """
        ...

    @property
    def weekday(self) -> weekday:
        """
        Day of the week (UTC)
        """
        ...

    @property
    def day_of_year(self) -> int:
        """
        The 1-based Gregorian day of the year (1 = January 1, 365 = December 31)
        """
        ...

    @staticmethod
    def from_mjd(mjd: float, scale: timescale = timescale.UTC) -> time:
        """Return a time object representing input modified Julian date and time scale

        Args:
            mjd (float): Modified Julian date, days
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

    def to_date(self) -> tuple[int, int, int]:
        """Return tuple representing as UTC Gregorian date of the time object.

        Returns:
            Tuple with 3 elements representing the Gregorian year, month, and day of the time object.
                Fractional component of day are truncated.
                Month is in range [1,12].
                Day is in range [1,31].
        """
        ...

    @deprecated("use to_date()")
    def as_date(self) -> tuple[int, int, int]:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_date`."""
        ...

    def to_gregorian(
        self,
    ) -> tuple[int, int, int, int, int, float]:
        """Return tuple representing as UTC Gregorian date and time of the time object.

        Returns:
            Tuple with 6 elements representing the Gregorian year, month, day, hour, minute, and second of the time object.
                Month is in range [1,12].
                Day is in range [1,31].
        """
        ...

    @deprecated("use to_gregorian()")
    def as_gregorian(self) -> tuple[int, int, int, int, int, float]:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_gregorian`."""
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
            # 2023-03-05T11:03:45.453000Z
            ```
        """
        ...

    @staticmethod
    def from_datetime(dt: datetime.datetime) -> time:
        """Convert input "datetime.datetime" object to an "satkit.time" object representing the same instant in time

        Follows Python's own convention (``datetime.timestamp()``):

        - A naive datetime (no ``tzinfo``) is interpreted in the machine's
          **local time zone**, not UTC.
        - An aware datetime uses its own UTC offset.

        For UTC, pass ``tzinfo=datetime.timezone.utc`` or build a
        ``satkit.time`` directly. The conversion is exact to the microsecond
        (it does not go through the float ``timestamp()``), and so is
        ``to_datetime()``.

        Args:
            dt (datetime.datetime): "datetime.datetime" object to convert

        Returns:
            Time object representing the same instant in time as the input "datetime.datetime" object
        """
        ...

    def to_datetime(self, utc: bool = True) -> datetime.datetime:
        """Convert object to "datetime.datetime" object representing same instant in time.

        Args:
            utc (bool, optional): If True (default), return an aware datetime in UTC.
                If False, return a naive datetime in the machine's local time zone,
                which round-trips through :meth:`time.from_datetime`.

        Returns:
            "datetime.datetime" object representing the same instant in time as the "satkit.time" object

        Example:
            ```python
            dt = satkit.time(2023, 6, 3, 6, 19, 34).to_datetime(True)
            print(dt)
            # 2023-06-03 06:19:34+00:00

            dt = satkit.time(2023, 6, 3, 6, 19, 34).to_datetime(False)
            print(dt)
            # 2023-06-03 02:19:34
            ```
        """
        ...

    @deprecated("use to_datetime()")
    def as_datetime(self, utc: bool = True) -> datetime.datetime:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_datetime`."""
        ...

    def datetime(self, utc: bool = True) -> datetime.datetime:
        """Deprecated: use :meth:`satkit.time.to_datetime`.

        Convert object to "datetime.datetime" object representing same instant in time.

        Args:
            utc (bool, optional): If True (default), return an aware datetime in UTC.
                If False, return a naive datetime in the machine's local time zone,
                which round-trips through :meth:`time.from_datetime`.

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

    def add_utc_days(self, days: float) -> time:
        """Return a new time offset by a whole number of UTC days.

        A UTC day is defined as exactly 86400 seconds, avoiding the ambiguity of
        adding a "day" across a leap second.

        Args:
            days (float): Number of UTC days to add

        Returns:
            satkit.time: Time object offset by the given number of UTC days
        """
        ...

    def to_mjd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as a Modified Julian Date
        with the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC

        Returns:
            float: Modified Julian Date, days
        """
        ...

    @deprecated("use to_mjd()")
    def as_mjd(self, scale: timescale = timescale.UTC) -> float:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_mjd`."""
        ...

    def to_jd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as Julian Date with
        the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC

        Returns:
            float: Julian Date, days
        """
        ...

    @deprecated("use to_jd()")
    def as_jd(self, scale: timescale = timescale.UTC) -> float:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_jd`."""
        ...

    def to_unixtime(self) -> float:
        """
        Represent time as unixtime

        (seconds since Jan 1, 1970 UTC, excluding leap seconds)

        Includes fractional component of seconds

        Returns:
            float: Unix time, seconds
        """
        ...

    @deprecated("use to_unixtime()")
    def as_unixtime(self) -> float:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_unixtime`."""
        ...

    def to_iso8601(self) -> str:
        """
        Represent time as ISO 8601 string

        Returns:
            ISO 8601 string representation of time: "YYYY-MM-DDTHH:MM:SS.sssZ"
        """
        ...

    @deprecated("use to_iso8601()")
    def as_iso8601(self) -> str:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_iso8601`."""
        ...

    def to_rfc3339(self) -> str:
        """
        Represent time as RFC 3339 string

        Returns:
            RFC 3339 string representation of time: "YYYY-MM-DDTHH:MM:SS.sssZ"
        """
        ...

    @deprecated("use to_rfc3339()")
    def as_rfc3339(self) -> str:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`time.to_rfc3339`."""
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
            print(satkit.time(2023, 6, 3, 6, 19, 34).strftime("%Y-%m-%d %H:%M:%S"))
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

        Raises:
            ValueError: if ``other`` is NaN or infinite
            OverflowError: if ``other`` is too large for a duration

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

        Raises:
            ValueError: if ``other`` is NaN or infinite
            OverflowError: if ``other`` is too large for a duration

        Returns:
            Time object representing the input number of days subtracted from the current time

        """
        ...

    @typing.overload
    def __sub__(self, other: npt.NDArray[np.float64] | list[float]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of days from the "self" time object

        Args:
            other (npt.NDArray[np.float64] | list[float]): days to subtract from the current time

        Returns:
            Object array of time objects representing the element-wise subtraction of days from the current time

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
            other (list[time]): times to subtract from the current time

        Returns:
            Object array of duration objects, ``self - other[i]`` for each element
        """
        ...

class duration:
    """
    Representation of a duration, or interval of time
    """

    def __init__(
        self,
        *,
        days: float = 0.0,
        hours: float = 0.0,
        minutes: float = 0.0,
        seconds: float = 0.0,
        microseconds: int = 0,
    ) -> None:
        """Create a duration object representing input time duration

        Args:
            days: Number of days, default is 0
            hours: Number of hours, default is 0
            minutes: Number of minutes, default is 0
            seconds: Number of seconds, default is 0.0
            microseconds: Number of microseconds (an integer), default is 0

        Notes:
            - If no arguments are passed in, the created object represents a duration of 0 seconds
            - A duration is an integer number of microseconds; each floating-point
              argument is rounded to the nearest microsecond
            - A NaN or infinite argument raises ``ValueError``, and one too large for
              a duration (beyond about ±292,000 years) raises ``OverflowError``; the
              same holds for the ``from_*`` constructors

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
    def from_seconds(seconds: float) -> duration:
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
    def from_minutes(minutes: float) -> duration:
        """Create duration object representing input number of minutes

        Args:
            m (float): Number of minutes

        Returns:
            Duration object representing input number of minutes
        """
        ...

    @staticmethod
    def from_hours(hours: float) -> duration:
        """Create duration object representing input number of hours

        Args:
            h (float): Number of hours

        Returns:
            Duration object representing input number of hours
        """
        ...

    @staticmethod
    def from_milliseconds(d: float) -> duration:
        """Create duration object representing input number of milliseconds

        Args:
            d (float): Number of milliseconds

        Returns:
            Duration object representing input number of milliseconds
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
            print(satkit.duration.from_hours(1) + satkit.duration.from_minutes(1))
            # Duration: 1 hours, 1 minutes, 0.000 seconds
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
            print(satkit.duration.from_hours(1) + satkit.time(2023, 6, 4, 11,30,0))
            # 2023-06-04T12:30:00.000000Z
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
            print(satkit.duration.from_hours(1) - satkit.duration.from_minutes(1))
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
            print(satkit.duration.from_days(1) * 2.5)
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

        Raises:
            ZeroDivisionError: if ``other`` is zero
            ValueError: if ``other`` is NaN

        Example:
            ```python
            print(satkit.duration.from_days(1) / 2)
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

        Raises:
            ZeroDivisionError: if ``other`` is a zero duration

        Example:
            ```python
            print(satkit.duration.from_hours(1) / satkit.duration.from_minutes(30))
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
            print(satkit.duration.from_hours(1) > satkit.duration.from_minutes(30))
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
            print(satkit.duration.from_hours(1) < satkit.duration.from_minutes(30))
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
            print(satkit.duration.from_hours(1) >= satkit.duration.from_minutes(30))
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
            print(satkit.duration.from_hours(1) <= satkit.duration.from_minutes(30))
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
    def microseconds(self) -> int:
        """Number of whole microseconds represented by duration

        Returns:
            Integer number of microseconds represented by duration
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

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, w: float, x: float, y: float, z: float, /) -> None:
        """Return quaternion with input (w,x,y,z) values

        With no arguments, return the identity quaternion. Otherwise pass all
        four components positionally (keywords are not accepted).

        Args:
            w: Scalar component of the quaternion
            x: X component of the quaternion
            y: Y component of the quaternion
            z: Z component of the quaternion

        Example:
            ```python
            # Identity quaternion (no rotation)
            q = satkit.quaternion()

            # 90 degree rotation about z-axis
            import math
            q = satkit.quaternion.rotz(math.radians(90))
            ```
        """
        ...

    @staticmethod
    def from_axis_angle(axis: npt.ArrayLike, angle: float) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "angle" radians about the given axis

        Args:
            axis (npt.ArrayLike[np.float64]): 3-element array representing axis of rotation (unitless direction; need not be normalized)
            angle (float): angle of rotation in radians

        Returns:
            Quaternion representing rotation by "angle" radians about the given axis
        """
        ...

    @staticmethod
    def from_rotation_matrix(
        dcm: npt.ArrayLike,
    ) -> quaternion:
        """Return quaternion representing identical rotation to input 3x3 rotation matrix

        Args:
            mat (npt.ArrayLike[np.float64]): 3x3 rotation matrix

        Returns:
            Quaternion representing identical rotation to input 3x3 rotation matrix
        """
        ...

    @staticmethod
    def rotx(theta_rad: float) -> quaternion:
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
    def roty(theta_rad: float) -> quaternion:
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
    def rotz(theta_rad: float) -> quaternion:
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
        v1: npt.ArrayLike, v2: npt.ArrayLike
    ) -> quaternion:
        """Quaternion representation rotation between two input vectors

        Args:
            v1 (npt.ArrayLike): 3-element vector rotating from (any real numeric array-like)
            v2 (npt.ArrayLike): 3-element vector rotating to (any real numeric array-like)

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

    def to_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Return 3x3 rotation matrix representing equivalent rotation

        Returns:
            3x3 rotation matrix representing equivalent rotation
        """
        ...

    @deprecated("use to_rotation_matrix()")
    def as_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`quaternion.to_rotation_matrix`."""
        ...

    def to_euler(self) -> tuple[float, float, float]:
        """Return equivalent rotation as intrinsic ZYX Euler angles (yaw, pitch, roll).

        The decomposition follows the aerospace convention (Tait-Bryan angles):
        the rotation is equivalent to first rotating by yaw about Z,
        then pitch about the new Y, then roll about the new X.

        Returns:
            tuple[float, float, float]: ``(roll, pitch, yaw)`` in radians

        Example:
            ```python
            q = satkit.quaternion.rotz(0.1) * satkit.quaternion.roty(0.2)
            roll, pitch, yaw = q.to_euler()
            ```
        """
        ...

    @deprecated("use to_euler()")
    def as_euler(self) -> tuple[float, float, float]:
        """Deprecated since 0.23, removed in 0.25. Use :meth:`quaternion.to_euler`."""
        ...

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> quaternion:
        """Create quaternion from roll, pitch, yaw Euler angles in radians
        (inverse of ``to_euler``)

        Args:
            roll (float): Roll angle in radians
            pitch (float): Pitch angle in radians
            yaw (float): Yaw angle in radians

        Returns:
            quaternion: Quaternion representing the input euler-angle rotation
        """
        ...

    @staticmethod
    def identity() -> quaternion:
        """The identity (no-rotation) quaternion (w=1, x=y=z=0)"""
        ...

    @property
    def norm(self) -> float:
        """Quaternion norm (Euclidean length of the 4 components; 1 for a
        unit rotation quaternion)"""
        ...

    def normalize(self) -> quaternion:
        """Return this quaternion normalized to unit length"""
        ...

    def inverse(self) -> quaternion:
        """Quaternion inverse. Equals the conjugate for a unit (rotation)
        quaternion."""
        ...

    def dot(self, other: quaternion) -> float:
        """Dot product of the 4 quaternion components with another quaternion"""
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

    def conj(self) -> quaternion:
        """Quaternion conjugate, which for a unit (rotation) quaternion is the
        inverse rotation. Same as ``conjugate()`` and ``inverse()``.

        Returns:
            Conjugate or inverse of the rotation
        """
        ...

    def conjugate(self) -> quaternion:
        """Quaternion conjugate, which for a unit (rotation) quaternion is the
        inverse rotation. Same as ``conj()`` and ``inverse()``.

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
    def __mul__(self, other: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Multiply by a vector to rotate the vector

        Args:
            other (npt.ArrayLike): 3-element vector to rotate, or Nx3 array of vectors to rotate (any real numeric array-like; integer arrays and lists are converted to float64)

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

    def slerp(self, other: quaternion, frac: float) -> quaternion:
        """Spherical linear interpolation between self and other

        Args:
            other (quaternion): Quaternion to perform interpolation to
            frac (float): fractional amount of interpolation, in range [0,1]

        Returns:
            Quaternion representing interpolation between self and other

        Example:
            ```python
            import math
            q1 = satkit.quaternion.rotz(math.radians(0))
            q2 = satkit.quaternion.rotz(math.radians(90))
            q_mid = q1.slerp(q2, 0.5)
            print(f"Mid-rotation angle: {math.degrees(q_mid.angle):.1f} deg")
            # Mid-rotation angle: 45.0 deg
            ```
        """
        ...

class kepler:
    """Represent Keplerian element sets and convert between cartesian

    Notes:
        - This class is used to represent Keplerian elements and convert between Cartesian coordinates
        - The class uses the semi-major axis (a), not the semiparameter
        - Elements are osculating and expressed in the frame of the input state
          (normally GCRF); the class does no frame handling
        - Each element set carries the gravitational parameter of its central
          body, ``mu`` (m^3/s^2); Earth's (``satkit.consts.mu_earth``) unless
          given, so lunar or heliocentric elements are supported by passing
          ``mu=satkit.consts.mu_moon`` / ``mu_sun``
        - Only closed orbits are supported (0 <= eccen < 1); the constructor,
          ``from_pv`` and every element setter — the anomaly setters
          included — raise ``ValueError`` for an element outside its domain
          or a non-finite value, so an element set can never hold NaN
        - All angle units are radians
        - All length units are meters
        - All velocity units are meters / second

    See the "Theory: Keplerian Elements" guide page for details.
    """

    def __init__(
        self,
        a: float,
        eccen: float,
        incl: float,
        raan: float,
        argp: float | None = None,
        nu: float | None = None,
        *,
        w: float | None = None,
        true_anomaly: float | None = None,
        eccentric_anomaly: float | None = None,
        mean_anomaly: float | None = None,
        mu: float | None = None,
    ) -> None:
        """Create Keplerian element set object from input elements

        Args:
            a: Semi-major axis, meters (> 0)
            eccen: Eccentricity, unitless (0 <= eccen < 1)
            incl: Inclination, radians (0 <= incl <= pi)
            raan: Right ascension of ascending node, radians
            argp: Argument of periapsis, radians (5th positional argument)
            nu: True anomaly, radians (6th positional argument)
            w: Argument of periapsis, radians — deprecated keyword alias of
                ``argp`` (kept indefinitely); give one or the other, not both
            true_anomaly: True anomaly, radians (keyword alternative to nu)
            eccentric_anomaly: Eccentric anomaly, radians (keyword alternative to nu)
            mean_anomaly: Mean anomaly, radians (keyword alternative to nu)
            mu: Gravitational parameter of the central body, m^3/s^2
                (default ``satkit.consts.mu_earth``)

        Raises:
            ValueError: an element outside its domain (non-finite value,
                ``a <= 0``, ``eccen`` outside [0, 1), ``incl`` outside
                [0, pi], ``mu <= 0``); more or fewer than one anomaly given;
                both ``argp`` and ``w`` given
            TypeError: no argument of periapsis given

        Notes:
            Exactly one of ``nu``, ``true_anomaly``, ``eccentric_anomaly`` or
            ``mean_anomaly`` must be given; anything else raises ValueError.
            All six elements may be passed positionally or by keyword.

        Example:
            ```python
            import math
            import satkit

            # Create a ~400 km circular LEO orbit
            k = satkit.kepler(
                a=6.781e6,  # semi-major axis, meters
                eccen=0.001,  # near-circular
                incl=math.radians(51.6),
                raan=math.radians(0),
                argp=math.radians(0),
                nu=math.radians(0),
            )

            # Same orbit, positional, located by mean anomaly instead
            k2 = satkit.kepler(6.781e6, 0.001, math.radians(51.6), 0, 0, mean_anomaly=1.0)

            # A 100 km circular lunar orbit
            k_moon = satkit.kepler(1837.4e3, 0.0, math.radians(90), 0, 0, 0,
                                   mu=satkit.consts.mu_moon)
            print(f"lunar period: {k_moon.period / 60:.1f} min")
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

    def propagate(self, dt: duration | float | int) -> kepler:
        """Propagate Keplerian element set by input duration

        Two-body (unperturbed) propagation: only the anomaly changes.

        Args:
            dt (duration | float | int): Duration by which to propagate the
                Keplerian element set. A number is interpreted as seconds.

        Returns:
            Keplerian element set object after propagation

        Example:
            ```python
            # Propagate orbit by one orbital period
            k2 = k.propagate(k.period)
            # ... or by ten minutes
            k3 = k.propagate(satkit.duration.from_minutes(10))
            ```

        Raises:
            TypeError: if ``dt`` is neither a duration nor a number.
            ValueError: if ``dt`` is a NaN or infinite number of seconds.
            OverflowError: if ``dt`` is too large for a duration.
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
    def eccentric_anomaly(self, value: float) -> None:
        """Set the in-plane position by eccentric anomaly, radians

        Converted to true anomaly (``nu``) on the spot. A non-finite value
        raises ``ValueError`` and leaves the element set unchanged.
        """
        ...
    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly, radians"""
        ...

    @mean_anomaly.setter
    def mean_anomaly(self, value: float) -> None:
        """Set the in-plane position by mean anomaly, radians

        Kepler's equation is solved for the eccentric anomaly and the result
        stored as true anomaly (``nu``). A non-finite value raises
        ``ValueError`` and leaves the element set unchanged.
        """
        ...
    @property
    def period(self) -> float:
        """Orbital period, seconds"""
        ...

    @property
    def semiparameter(self) -> float:
        """Semiparameter (semi-latus rectum) p = a (1 - e^2), meters"""
        ...

    @property
    def periapsis(self) -> float:
        """Radius of periapsis a (1 - e), meters"""
        ...

    @property
    def apoapsis(self) -> float:
        """Radius of apoapsis a (1 + e), meters"""
        ...

    @property
    def specific_energy(self) -> float:
        """Specific orbital energy -mu / (2 a), J/kg (m^2/s^2)"""
        ...

    @property
    def angular_momentum(self) -> float:
        """Magnitude of the specific angular momentum sqrt(mu p), m^2/s"""
        ...

    @property
    def flight_path_angle(self) -> float:
        """Flight-path angle, radians

        atan2(e sin nu, 1 + e cos nu): the angle of the velocity above the
        local horizontal — zero at periapsis and apoapsis, positive while
        climbing.
        """
        ...

    @property
    def argument_of_latitude(self) -> float:
        """Argument of latitude u = argp + nu, radians in [0, 2 pi)

        Well defined for circular orbits, where ``argp`` and ``nu`` separately
        are not.
        """
        ...

    @property
    def true_longitude(self) -> float:
        """True longitude raan + argp + nu, radians in [0, 2 pi)

        Well defined for circular equatorial orbits, where ``raan``, ``argp``
        and ``nu`` separately are not.
        """
        ...

    @property
    def mu(self) -> float:
        """Gravitational parameter of the central body, m^3/s^2

        Setting it re-targets the dynamics (period, mean motion,
        ``propagate``, ``to_pv``) at another body while keeping the six
        geometric elements. Must be positive and finite (``ValueError``).
        """
        ...

    @mu.setter
    def mu(self, value: float) -> None: ...
    @property
    def a(self) -> float:
        """Semi-major axis, meters (> 0; ``ValueError`` otherwise)"""
        ...

    @a.setter
    def a(self, value: float) -> None: ...
    @property
    def eccen(self) -> float:
        """Eccentricity, unitless (0 <= eccen < 1; ``ValueError`` otherwise)"""
        ...

    @eccen.setter
    def eccen(self, value: float) -> None: ...
    @property
    def inclination(self) -> float:
        """Inclination, radians (0 <= incl <= pi; ``ValueError`` otherwise)"""
        ...

    @inclination.setter
    def inclination(self, value: float) -> None: ...
    @property
    def raan(self) -> float:
        """Right ascension of ascending node, radians (finite; ``ValueError`` otherwise)"""
        ...

    @raan.setter
    def raan(self, value: float) -> None: ...
    @property
    def nu(self) -> float:
        """True anomaly, radians (finite; ``ValueError`` otherwise)"""
        ...

    @nu.setter
    def nu(self, value: float) -> None: ...
    @property
    def argp(self) -> float:
        """Argument of periapsis, radians (finite; ``ValueError`` otherwise)"""
        ...

    @argp.setter
    def argp(self, value: float) -> None: ...
    @property
    def w(self) -> float:
        """Argument of periapsis, radians — deprecated alias of ``argp``

        Kept indefinitely for compatibility; reading or assigning it emits
        ``DeprecationWarning``. Validated like ``argp``.
        """
        ...

    @w.setter
    def w(self, value: float) -> None: ...
    @staticmethod
    def from_pv(
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        *,
        mu: float | None = None,
    ) -> kepler:
        """Create Keplerian element set from input position and velocity vectors

        Args:
            pos: 3-element position vector, meters
            vel: 3-element velocity vector, meters/second
            mu: Gravitational parameter of the central body, m^3/s^2
                (default ``satkit.consts.mu_earth``); the returned element
                set carries it, so its period and ``to_pv`` refer to the same
                body

        Returns:
            Keplerian element set object

        Example:
            ```python
            import numpy as np
            pos = np.array([6.781e6, 0, 0])  # meters, GCRF
            vel = np.array([0, 7.5e3, 0])    # m/s, GCRF
            k = satkit.kepler.from_pv(pos, vel)
            print(f"Semi-major axis: {k.a/1e3:.1f} km")
            print(f"Eccentricity: {k.eccen:.6f}")
            ```

        Raises:
            ValueError: if the state is hyperbolic/parabolic (eccen >= 1) or
                rectilinear (zero angular momentum), if ``pos`` or ``vel``
                holds a NaN or infinity, or if ``mu`` is not positive and
                finite.
            RuntimeError: if the inputs are not 3-element vectors.
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
        coord = satkit.itrfcoord([ 1523128.63570828, -4461395.28873207,  4281865.94218203 ])
        ```

        Create ITRF coord from Geodetic:

        ```python
        coord = satkit.itrfcoord(latitude_deg=42.44, longitude_deg=-71.15, altitude=100)
        ```

    """

    @overload
    def __init__(self, vec: npt.NDArray[np.float64] | list[float], /) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float, /) -> None: ...
    @overload
    def __init__(
        self,
        *,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        latitude_rad: float | None = None,
        longitude_rad: float | None = None,
        altitude: float | None = None,
        height: float | None = None,
    ) -> None:
        """Create ITRF coordinate from Cartesian vector or geodetic parameters.

        Pass the Cartesian position positionally, either as one 3-element
        numpy array or list, or as three floats; or pass the geodetic
        position as keywords (latitude and longitude are required).

        Args:
            vec: ITRF Cartesian location in meters (3-element numpy array or list)
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            latitude_rad: Latitude in radians
            longitude_rad: Longitude in radians
            altitude: Height above ellipsoid, meters. Default is 0
            height: Height above ellipsoid, meters (alias for altitude; wins if both are given)

        Any geodetic keyword selects the geodetic form. ``latitude_rad`` /
        ``longitude_rad`` win over ``latitude_deg`` / ``longitude_deg``.
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
    def height(self) -> float:
        """Height above ellipsoid, in meters (alias of altitude)"""
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
            - This is equivalent to calling: origin.qenu2itrf.conj() * (self - origin)

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
            - This is equivalent to calling: origin.qned2itrf.conj() * (self - origin)

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
        """Use Vincenty's inverse formula to compute geodesic distance
        (T. Vincenty, Survey Review 23(176), 1975, <https://doi.org/10.1179/sre.1975.23.176.88>):

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

    def distance_to(self, other: itrfcoord) -> float:
        """Geodesic distance in meters between this coordinate and another

        Convenience wrapper around ``geodesic_distance`` returning only the
        distance (shortest distance along the Earth's surface).

        Args:
            other (itrfcoord): ITRF coordinate to measure distance to

        Returns:
            float: Distance in meters
        """
        ...

    def move_with_heading(self, distance: float, heading_rad: float) -> itrfcoord:
        """Move a distance along the Earth surface with a given initial heading

        Args:
            distance (float): Distance to move in meters
            heading_rad (float): Initial heading in radians

        Notes:
            Altitude is assumed to be zero

            Use Vincenty's direct formula to compute position
            (T. Vincenty, Survey Review 23(176), 1975, <https://doi.org/10.1179/sre.1975.23.176.88>)

        Returns:
            itrfcoord: New ITRF coordinate after moving ``distance`` meters along the geodesic

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
    """Some constants that are useful for satellite dynamics"""

    wgs84_a: ClassVar[float]
    """WGS-84 semiparameter, in meters"""

    wgs84_f: ClassVar[float]
    """WGS-84 flattening, unitless"""

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

    solar_pressure_1au: ClassVar[float]
    """Solar radiation pressure at 1 AU (1367 W/m^2 / c), N/m^2.

    The cannonball SRP force scales this by (AU / d)^2, with d the
    satellite-Sun distance. The measured solar constant is 1361 W/m^2
    (Kopp & Lean 2011; IAU 2015 Resolution B3); 1367 matches GMAT and STK.
    """

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

# Alias so `time` resolves to the class inside bodies that define a `time` member
_Time = time

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
        sat.set_pos_uncertainty(np.array([50.0, 200.0, 100.0]), frame=sk.frame.RTN)
        sat.add_prograde(sat.time + sk.duration.from_hours(1), 10.0)

        # Propagate -- covariance and maneuver handled automatically
        new_state = sat.propagate(sat.time + sk.duration.from_hours(3))
        ```
    """

    def __init__(
        self,
        time: TimeScalar,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64] | None = None,
    ):
        """Create a new satellite state

        Args:
            time (satkit.time): Epoch of the state
            pos (npt.NDArray[np.float64]): Position in meters, GCRF frame
            vel (npt.NDArray[np.float64]): Velocity in m/s, GCRF frame
            cov (npt.NDArray[np.float64]|None, optional): 6x6 covariance matrix in GCRF
                (position block m^2, velocity block (m/s)^2, cross blocks m^2/s). Defaults to None.

        Example:
            ```python
            t = satkit.time(2024, 1, 1)
            pos = np.array([6.781e6, 0, 0])       # meters, GCRF
            vel = np.array([0, 7.5e3, 0])          # m/s, GCRF
            state = satkit.satstate(t, pos, vel)
            ```
        """
        ...

    @staticmethod
    def from_kepler(time: TimeScalar, kepler: kepler) -> satstate:
        """Create a state from Keplerian elements

        The two-body position (meters) and velocity (m/s) of the elements
        (``kepler.to_pv()``) are taken as GCRF at ``time``; no covariance, no
        maneuvers. The elements are treated as osculating GCRF elements, so
        ``kepler.mu`` should be Earth's.

        Args:
            time (satkit.time): Epoch of the state
            kepler (satkit.kepler): Osculating Keplerian elements

        Returns:
            satstate: state at ``time``

        Example:
            ```python
            import math
            k = satkit.kepler(7000e3, 0.001, math.radians(98), 0, 0, 0)
            state = satkit.satstate.from_kepler(satkit.time(2024, 1, 1), k)
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
    def time(self) -> _Time:
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
                - ``frame.RTN`` — Radial / Transverse / Normal (= RSW = RIC);
                  R exactly along position. CCSDS OEM / CDM convention.
                - ``frame.NTW`` — Normal-to-velocity / Tangent / Cross-track;
                  T exactly along velocity.
                - ``frame.LVLH`` — Local Vertical / Local Horizontal (RTN
                  axes relabeled: x = T, y = -N, z = -R)

        Raises:
            RuntimeError: if the frame is not one of the supported frames.

        Example:
            ```python
            # RTN: 10 m radial, 200 m transverse, 30 m normal
            sat.set_pos_uncertainty(np.array([10.0, 200.0, 30.0]), frame=sk.frame.RTN)

            # NTW: 10 m normal-to-velocity, 200 m along velocity, 30 m cross-track
            sat.set_pos_uncertainty(np.array([10.0, 200.0, 30.0]), frame=sk.frame.NTW)
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
                ``frame.RTN``, ``frame.NTW``, ``frame.LVLH``.

        Raises:
            RuntimeError: if the frame is not one of the supported frames.
        """
        ...

    def add_maneuver(
        self,
        time: TimeScalar,
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
                - ``frame.RTN`` — radial / in-track / cross-track (a.k.a. RSW, RIC).
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

    def add_prograde(self, time: TimeScalar, dv_mps: float) -> None:
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

    def add_retrograde(self, time: TimeScalar, dv_mps: float) -> None:
        """Add a retrograde impulsive burn (NTW -T, opposite velocity).

        Equivalent to ``add_prograde`` with a negated magnitude. ``dv_mps``
        should be positive; a positive value removes energy from the orbit.

        Args:
            time (satkit.time): Time at which to apply the burn
            dv_mps (float): Magnitude along anti-velocity vector [m/s]
        """
        ...

    def add_radial(self, time: TimeScalar, dv_mps: float) -> None:
        """Add a radial-outward impulsive burn (NTW +N axis).

        For circular orbits this is the outward radial direction. For
        eccentric orbits the N axis leans off the radial by the
        flight-path angle.

        Args:
            time (satkit.time): Time at which to apply the burn
            dv_mps (float): Magnitude along in-plane normal-to-velocity [m/s]
        """
        ...

    def add_normal(self, time: TimeScalar, dv_mps: float) -> None:
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

    @property
    def maneuvers(self) -> list[dict]:
        """The scheduled impulsive maneuvers.

        Returns:
            list[dict]: One dict per maneuver with keys ``"time"``
            (satkit.time), ``"delta_v"`` (3-element numpy array, m/s, in
            ``"frame"``), and ``"frame"`` (satkit.frame)
        """
        ...

    def propagate(
        self,
        timedur: TimeScalar | duration,
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
            timedur (satkit.time|datetime.datetime|satkit.duration): Target time, or duration from current time
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
    def time(self) -> _Time:
        """Time at which state is valid

        Returns:
            Time at which state is valid
        """
        ...

    @property
    def time_end(self) -> _Time:
        """Time at which state is valid

        Notes:
        - This is identical to "time" property

        Returns:
            Time at which state is valid
        """
        ...

    @property
    def time_begin(self) -> _Time:
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
    def next_step_secs(self) -> float:
        """Step the integrator would take next, seconds: its working stride at
        ``time_end`` (the controller's last unclamped proposal for the adaptive
        integrators, the fixed step for ``integrator.gauss_jackson8``, 0 for a
        zero-duration propagation). Signed like the propagation direction.
        Pass it as ``propsettings.initial_step_secs`` to continue this arc
        without the start-up ramp.
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
            6x6 numpy array representing state transition matrix or None if not computed.
                Maps a perturbation of the begin state (meters, m/s) to the end state
                (meters, m/s), so blocks are unitless, seconds, 1/seconds, unitless.
        """
        ...

    @typing.overload
    def interp(
        self,
        time: TimeScalar,
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
        time: TimeScalar,
        output_phi: typing.Literal[True] = ...,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Interpolate state and state transition matrix at a single time

        Args:
            time: Time at which to interpolate state
            output_phi: Must be True

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: (state, phi) where state is a 6-element
                vector [x, y, z, vx, vy, vz] in meters and m/s, and phi is a 6x6 state transition matrix

        Raises:
            ValueError: if the propagation did not compute the state transition matrix
                (propagate with ``output_phi=True``)
        """
        ...

    @typing.overload
    def interp(
        self,
        time: list[_Time | datetime.datetime],
        output_phi: typing.Literal[False] = False,
    ) -> npt.NDArray[np.float64]:
        """Interpolate state at multiple times

        Args:
            time: List of times at which to interpolate state
            output_phi: Must be False (default)

        Returns:
            npt.NDArray[np.float64]: (N, 6) array, one row [x, y, z, vx, vy, vz] in meters and m/s
                per input time ((0, 6) for an empty list)
        """
        ...

    @typing.overload
    def interp(
        self,
        time: list[_Time | datetime.datetime],
        output_phi: typing.Literal[True] = ...,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Interpolate state and state transition matrix at multiple times

        Args:
            time: List of times at which to interpolate state
            output_phi: Must be True

        Returns:
            list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]: List of (state, phi) tuples;
                each state is a 6-element vector in meters and m/s, each phi a 6x6 state transition matrix

        Raises:
            ValueError: if the propagation did not compute the state transition matrix
                (propagate with ``output_phi=True``)
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

class ecomparams:
    """Empirical CODE Orbit Model (ECOM) solar-radiation-pressure coefficients

    ECOM expresses the non-gravitational acceleration of a (nominally
    yaw-steering) satellite in a Sun-oriented frame with constant and
    harmonic terms. The coefficients are normally *estimated* in orbit
    determination; satkit propagates with the values you supply.

    Coefficients (all accelerations in m/s² at 1 AU; typical GPS sizes in nm/s²):

        d0        e_D (toward Sun), constant, all models      -80 to -110 (negative)
        y0        e_Y (solar-panel axis), constant, all       ~1  (attitude/thermal Y-bias)
        b0        e_B, constant, all                          ~1-5, varies with beta
        dc, ds    e_D, cos/sin phi, ECOM1 (phi = u)           <~1
        yc, ys    e_Y, cos/sin phi, ECOM1                     <~1
        bc, bs    e_B, cos/sin phi; reduced/ECOM1 (phi = u), ECOM2's B1c/B1s (phi = du)  <~2
        d2c, d2s  e_D, cos/sin 2*du, ECOM2                    few (eclipse seasons)
        d4c, d4s  e_D, cos/sin 4*du, ECOM2                    few (eclipse seasons)

    Experimental: this interface is new and may be reshaped in a minor release;
    the physics and conventions are stable.

    Frame (GCRF), with r the satellite position and s the Sun position:

    - ``e_D = unit(s - r)`` — satellite → Sun
    - ``e_Y = unit(e_D × r̂)`` — solar-panel rotation axis
    - ``e_B = e_D × e_Y``

    Model::

        a = ν · (AU / d)² · [ D(φ)·e_D + Y(φ)·e_Y + B(φ)·e_B ]
        D(φ) = d0 + dc cos φ + ds sin φ + d2c cos 2φ + d2s sin 2φ + d4c cos 4φ + d4s sin 4φ
        Y(φ) = y0 + yc cos φ + ys sin φ
        B(φ) = b0 + bc cos φ + bs sin φ

    where ``ν`` is the Earth-shadow factor applied to all three axes (the
    CODE/Bernese convention: the whole ECOM acceleration is switched off in
    umbra), ``d`` is the satellite-Sun distance, and ``φ`` is the argument of
    latitude ``u`` when ``sun_relative`` is False (ECOM1) or ``Δu``, measured
    from orbit noon, when True (ECOM2).

    Because ``e_D`` points *at* the Sun, the physical ``d0`` is negative:
    about -1e-7 m/s² for a GPS satellite; ``y0`` and the B terms are ~1e-9.
    All coefficients are in m/s² referred to 1 AU and scaled by ``(AU / d)²``
    like the cannonball pressure, so ``d0 = -consts.solar_pressure_1au * craoverm``
    reproduces the cannonball term. Coefficients estimated with software that
    applies them unscaled convert by multiplying by ``(d / AU)²`` at the arc
    epoch. satkit 0.23.1 and earlier applied them unscaled.

    Attach to a propagation via ``satproperties(ecom=...)``. The ECOM
    acceleration is added to the cannonball term, so use ``craoverm=0`` for
    a pure ECOM model.

    Example:

    ```python
    import satkit as sk

    ecom = sk.ecomparams.reduced(d0=-1.0e-7, y0=1e-9, b0=0, bc=2e-9, bs=-1e-9)
    props = sk.satproperties(craoverm=0.0, ecom=ecom)
    res = sk.propagate(state, t0, t1, propsettings=settings, satproperties=props)
    ```
    """

    def __init__(
        self,
        *,
        d0: float = 0.0,
        y0: float = 0.0,
        b0: float = 0.0,
        dc: float = 0.0,
        ds: float = 0.0,
        yc: float = 0.0,
        ys: float = 0.0,
        bc: float = 0.0,
        bs: float = 0.0,
        d2c: float = 0.0,
        d2s: float = 0.0,
        d4c: float = 0.0,
        d4s: float = 0.0,
        sun_relative: bool = False,
    ) -> None:
        """Create ECOM coefficients (m/s² at 1 AU); all default to zero.

        Args:
            d0, y0, b0 (float): constant D, Y, B terms
            dc, ds (float): D cos φ, D sin φ
            yc, ys (float): Y cos φ, Y sin φ
            bc, bs (float): B cos φ, B sin φ
            d2c, d2s, d4c, d4s (float): even D harmonics (ECOM2)
            sun_relative (bool): False → φ = argument of latitude (ECOM1);
                True → φ = Δu from orbit noon (ECOM2)
        """
        ...

    @staticmethod
    def reduced(d0: float, y0: float, b0: float, bc: float, bs: float) -> ecomparams:
        """Reduced ECOM1: D0, Y0, B0, Bc, Bs in argument of latitude (CODE's classic GPS set)"""
        ...

    @staticmethod
    def ecom1(
        d0: float, y0: float, b0: float, dc: float, ds: float, yc: float, ys: float, bc: float, bs: float
    ) -> ecomparams:
        """Full 9-parameter ECOM1 (once-per-revolution terms on D, Y, B in argument of latitude)"""
        ...

    @staticmethod
    def ecom2(
        d0: float, y0: float, b0: float, b1c: float, b1s: float, d2c: float, d2s: float, d4c: float, d4s: float
    ) -> ecomparams:
        """ECOM2 (Arnold et al. 2015): D0, Y0, B0, B1c, B1s, D2c, D2s, D4c, D4s in Δu from orbit noon"""
        ...

    d0: float
    """Constant D (Sun-direction) term, m/s². Physically negative."""
    y0: float
    """Constant Y term, m/s² (along the solar-panel axis)."""
    b0: float
    """Constant B term, m/s²."""
    dc: float
    """D cos φ coefficient, m/s²."""
    ds: float
    """D sin φ coefficient, m/s²."""
    yc: float
    """Y cos φ coefficient, m/s²."""
    ys: float
    """Y sin φ coefficient, m/s²."""
    bc: float
    """B cos φ coefficient, m/s² (B1c in ECOM2)."""
    bs: float
    """B sin φ coefficient, m/s² (B1s in ECOM2)."""
    d2c: float
    """D cos 2φ coefficient, m/s² (ECOM2)."""
    d2s: float
    """D sin 2φ coefficient, m/s² (ECOM2)."""
    d4c: float
    """D cos 4φ coefficient, m/s² (ECOM2)."""
    d4s: float
    """D sin 4φ coefficient, m/s² (ECOM2)."""
    sun_relative: bool
    """True: harmonics in Δu from orbit noon (ECOM2); False: in the argument of latitude u (ECOM1)."""

    def to_dict(self) -> dict[str, float | bool]:
        """Coefficients as a dict (13 floats plus ``sun_relative``)"""
        ...

    @staticmethod
    def from_dict(d: dict[str, float | bool]) -> ecomparams:
        """Build from a dict as produced by :meth:`to_dict`; missing keys default to zero / False"""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class satproperties:
    """Satellite properties relevant for drag, radiation pressure, and thrust

    This class lets the satellite radiation pressure, drag,
    and thrust parameters be set for duration of propagation.

    Attributes:
        cdaoverm (float): Coefficient of drag times area over mass in m^2/kg
        craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg
        thrusts (list[thrust]): List of continuous thrust arcs
        ecom (ecomparams | None): ECOM empirical solar-radiation-pressure
            coefficients, added to the cannonball term (use ``craoverm=0``
            for a pure ECOM model). See :class:`ecomparams` for the
            conventions and the "ECOM Solar Radiation Pressure" tutorial
            for a fit against IGS GPS orbits.

    """

    def __init__(
        self,
        *,
        cdaoverm: float = 0,
        craoverm: float = 0,
        thrusts: list[thrust] | None = None,
        ecom: ecomparams | None = None,
    ) -> None:
        """Create a satproperties object

        All arguments are keyword-only; a positional call raises ``TypeError``.
        (Earlier releases read positional arguments as ``(craoverm, cdaoverm)``,
        the reverse of the documented order, so scripts that passed them
        positionally had drag and radiation pressure swapped.)

        Keyword Args:
            cdaoverm (float, optional): Coefficient of drag times area over mass in m^2/kg
            craoverm (float, optional): Coefficient of radiation pressure times area over mass in m^2/kg
            thrusts (list[thrust], optional): List of continuous thrust arcs
            ecom (ecomparams, optional): ECOM solar-radiation-pressure coefficients

        Example:

        ```python
        import satkit as sk

        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(2)

        props = sk.satproperties(
            cdaoverm=0.01,
            thrusts=[sk.thrust.constant([0, 1e-4, 0], t0, t1, frame=sk.frame.RTN)]
        )

        # GNSS-style empirical SRP instead of the cannonball
        props = sk.satproperties(craoverm=0.0, ecom=sk.ecomparams.reduced(-1e-7, 0, 0, 0, 0))
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

    @property
    def ecom(self) -> ecomparams | None:
        """ECOM solar-radiation-pressure coefficients, or None

        When set, the ECOM acceleration (see :class:`ecomparams` for the
        DYB frame, sign and eclipse conventions) is added to the cannonball
        term ``craoverm``; use ``craoverm=0`` for a pure ECOM model. The
        "ECOM Solar Radiation Pressure" tutorial shows how to fit the
        coefficients to IGS GPS orbits.
        """
        ...

    @ecom.setter
    def ecom(self, value: ecomparams | None) -> None: ...

class integrator:
    """Choice of ODE integrator for orbit propagation

    Available integrators, from highest to lowest order:

    - ``rkv98`` - Verner 9(8) with 8th-degree dense output, 21 stages (default)
    - ``rkv98_nointerp`` - Verner 9(8) without interpolation, 16 stages
    - ``rkv87`` - Verner 8(7) with 7th-degree dense output, 17 stages
    - ``rkv65`` - Verner 6(5) with 6th-degree dense output, 10 stages
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
    """Verner 9(8) with 8th-degree dense output, 21 stages (default)

    Highest accuracy integrator. Recommended for precision orbit propagation.
    """

    rkv98_nointerp: ClassVar[integrator]
    """Verner 9(8) without interpolation, 16 stages

    Same stepping accuracy as ``rkv98`` but skips interpolation stages.
    Slightly faster when dense output is not needed (``enable_interp=False``).
    """

    rkv87: ClassVar[integrator]
    """Verner 8(7) with 7th-degree dense output, 17 stages"""

    rkv65: ClassVar[integrator]
    """Verner 6(5) with 6th-degree dense output, 10 stages"""

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
            - gravity_model: gravmodel.egm2008
            - use_spaceweather: True
            - use_sun_gravity: True
            - use_moon_gravity: True
            - tide_model: tidemodel.solid_step1
            - use_relativistic_correction: True
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
        gravity_order: int | None = None,
        gravity_model: gravmodel = ...,
        use_spaceweather: bool = True,
        use_sun_gravity: bool = True,
        use_moon_gravity: bool = True,
        tide_model: tidemodel = ...,
        use_relativistic_correction: bool = True,
        enable_interp: bool = True,
        integrator: integrator = ...,
        gj_step_seconds: float = 60.0,
        max_steps: int = 1_000_000,
        require_eop_coverage: bool = False,
        initial_step_secs: float | None = None,
    ) -> None:
        """Create propagation settings object used to configure high-precision orbit propagator

        Args:
            abs_error: Maximum absolute value of error for any element in propagated state following ODE integration,
                in the units of the state (meters for position elements, m/s for velocity elements). Default is 1e-8
            rel_error: Maximum relative error of any element in propagated state following ODE integration, unitless. Default is 1e-8
            gravity_degree: Maximum degree of spherical harmonic gravity model, at most 70 (``ValueError`` above that). Default is 4
            gravity_order: Maximum order of spherical harmonic gravity model. Must be <= gravity_degree (and so at most 70). Default is same as gravity_degree
            gravity_model: Gravity model to use. Default is gravmodel.egm2008
            use_spaceweather: Use space weather data when computing atmospheric density for drag forces. Default is True
            use_sun_gravity: Include sun third-body gravitational perturbation. Default is True
            use_moon_gravity: Include moon third-body gravitational perturbation. Default is True
            tide_model: Solid Earth tide model. Default is ``tidemodel.solid_step1``
                (IERS 2010 §6.2.1 frequency-independent Love-number response).
                Use ``tidemodel.none`` to disable (e.g., for reproducibility with
                pre-tide releases).
            use_relativistic_correction: Include the general-relativistic
                acceleration of IERS 2010 §10.3 Eq. 10.12 (β=γ=1): Schwarzschild
                + geodesic (de Sitter) precession + Lense-Thirring. Default is True.
                Its position effect depends on the orbit, propagation arc, and
                fitted parameters (~1 m/day at GPS altitude if omitted; the
                geodesic term dominates beyond ~100,000 km); cost is negligible.
            enable_interp: Store intermediate data that allows for fast high-precision interpolation of state between begin and end times. Default is True.
                When False, no dense output is stored and ``integrator.rkv98`` runs its 16-stage
                no-interpolant tableau (same order and error control, 24% fewer force evaluations per step).
            integrator: ODE integrator to use. Default is integrator.rkv98
            gj_step_seconds: Fixed step size (seconds) used by ``integrator.gauss_jackson8``.
                Ignored by adaptive integrators. Typical values: 30-120 s for LEO, 60-300 s
                for MEO, 300-600 s for GEO. Default is 60.0.
            max_steps: Maximum number of integrator steps before the propagator aborts with
                a max-steps error. Applies to all integrators (adaptive Runge-Kutta, Rosenbrock,
                and Gauss-Jackson 8). Default is 1_000_000, which covers very long propagation
                arcs with plenty of headroom. Lower for a tighter runaway-propagation safeguard.
            require_eop_coverage: Raise ``RuntimeError`` if the propagation span is not inside
                the loaded Earth-orientation-parameter (EOP) table: past its end (instead of
                holding the last EOP row constant) or before its start, 1973-01-02 (instead of
                using zero EOP), each otherwise with a one-time warning. Default is False.
                See ``satkit.frametransform.eop_coverage`` / ``eop_status``.
            initial_step_secs: First step (seconds) the adaptive integrators attempt. Default
                None: derived from the initial state, the tolerances and the integrator order
                as ``1.5 * |r|/|v| * tol**(1/(p+1))`` with ``tol = rel_error + abs_error/|r|``
                (about 170 s for ``rkv98`` at 1e-9 in LEO; within a factor of ~2.5 of the
                settled stride). Set it to a previous ``propresult.next_step_secs`` to warm-start a
                follow-on arc at full stride. Ignored by ``integrator.gauss_jackson8``.

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
            Maximum absolute value of error for any element in propagated state following ODE integration,
                in the units of the state (meters for position elements, m/s for velocity elements); default is 1e-8
        """
        ...

    @abs_error.setter
    def abs_error(self, value: float) -> None: ...
    @property
    def rel_error(self) -> float:
        """Maximum relative error of any element in propagated state following ODE integration

        Returns:
            Maximum relative error of any element in propagated state following ODE integration, unitless; default is 1e-8

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
    def use_relativistic_correction(self) -> bool:
        """Include the general-relativistic acceleration of IERS 2010 §10.3
        Eq. 10.12 (PPN β = γ = 1): the Schwarzschild term, geodesic (de Sitter)
        precession ``2 (Ω × v)`` from the Earth's heliocentric motion, and
        Lense–Thirring frame dragging from the Earth's spin. Default is True.

        The Schwarzschild term dominates below GEO (omitting it costs ~1 m/day
        at GPS altitude); the geodesic term dominates beyond ~100,000 km
        (~1 m over 7 days at 200,000 km). Computational cost is negligible.
        Matches GMAT's ``RelativisticCorrection``.
        """
        ...

    @use_relativistic_correction.setter
    def use_relativistic_correction(self, value: bool) -> None: ...
    @property
    def require_eop_coverage(self) -> bool:
        """Raise ``RuntimeError`` from ``propagate`` if the span is not inside the
        loaded Earth-orientation-parameter (EOP) table: past its end (instead of
        holding the last EOP row constant) or before its start, 1973-01-02 (instead
        of using zero EOP, UT1 = UTC), each otherwise with a one-time warning.
        Default False.

        Polar motion and UT1−UTC drift by ~0.1 arcsec / ~10 ms over a few months —
        metres at LEO — and zero EOP is off by up to ~12 arcsec (hundreds of metres
        at LEO), so set this for precision work and refresh the data files with
        ``satkit.utils.update_datafiles()`` when it trips past the end. See
        ``satkit.frametransform.eop_coverage`` and ``eop_status``.
        """
        ...

    @require_eop_coverage.setter
    def require_eop_coverage(self, value: bool) -> None: ...
    @property
    def initial_step_secs(self) -> float | None:
        """First step (seconds) the adaptive integrators attempt, or None for the
        default, derived from the initial state, the tolerances and the integrator
        order as ``1.5 * |r|/|v| * tol**(1/(p+1))`` with
        ``tol = rel_error + abs_error/|r|`` (about 170 s for ``rkv98`` at 1e-9 in
        LEO; within a factor of ~2.5 of the settled stride, which the step
        controller closes within a step or two).

        Set it to a previous ``propresult.next_step_secs`` to warm-start a
        follow-on arc at full stride. A magnitude: backward propagation applies
        the sign, and a value longer than the arc is clamped to it. Ignored by
        ``integrator.gauss_jackson8``. Zero or non-finite raises ``RuntimeError``
        from ``propagate``.
        """
        ...

    @initial_step_secs.setter
    def initial_step_secs(self, value: float | None) -> None: ...
    @property
    def enable_interp(self) -> bool:
        """Store intermediate data that allows for fast high-precision interpolation of state between begin and end times.

        When False, no dense output is stored and ``integrator.rkv98`` runs its
        16-stage no-interpolant tableau (same order and error control, 24% fewer
        force evaluations per step).
        """
        ...

    @enable_interp.setter
    def enable_interp(self, value: bool) -> None: ...
    @property
    def gravity_model(self) -> gravmodel:
        """Gravity model used for Earth gravity computation

        Returns:
            gravmodel: The gravity model, default is gravmodel.egm2008

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
        prograde: If True (default), prograde transfer (angular momentum
            h_z >= 0); if False, retrograde (h_z <= 0). This picks the short
            or long way around.

    Returns:
        List of (v1, v2) tuples. Each v1 and v2 is a 3-element numpy array
        in m/s. The first element is the zero-revolution solution; additional
        elements are multi-revolution solutions if they exist.

    Raises:
        ValueError: If inputs are invalid (non-finite values, tof or mu not
            positive, zero position, r1 == r2), or if the solver fails to
            converge

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
    state: npt.ArrayLike | None = None,
    begin: time | None = None,
    end: time | None = None,
    *,
    pos: npt.ArrayLike | None = None,
    vel: npt.ArrayLike | None = None,
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
        state: 6-element numpy array representing satellite GCRF position and velocity in meters and meters/second.
            Required unless ``pos`` and ``vel`` are given
        begin: Time at which satellite is at input state. Required (``TypeError`` if omitted); its
            ``None`` default only lets ``state`` be omitted in favour of ``pos`` and ``vel``

    Keyword Args:
        end: Time at which new position and velocity will be computed
        duration: Duration from ``begin`` at which new position & velocity will be computed
        duration_secs: Duration in seconds from ``begin`` at which new position and velocity will be computed
        duration_days: Duration in days from ``begin`` at which new position and velocity will be computed
        output_phi: Output 6x6 state transition matrix between begin and end times. Default is False
        propsettings: Settings for the propagation; if omitted, defaults are used
        satproperties: Drag and radiation pressure susceptibility of satellite
        pos: GCRF position in meters; replaces the first three elements of ``state``,
            or stands in for ``state`` together with ``vel``
        vel: GCRF velocity in meters/second; replaces the last three elements of ``state``

    Returns:
        propresult: Propagation result object holding state outputs, statistics,
            and dense output if requested

    Notes:
        Propagates satellite ephemeris (position, velocity in GCRF & time) to new time and
        outputs new position and velocity via Runge-Kutta integration.
        Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF).

        Included forces:

        - Earth gravity with higher-order spherical-harmonic terms
        - Sun, Moon gravity
        - Solid Earth tides (IERS 2010 Step 1 by default; ``propsettings.tide_model``
          selects ``tidemodel.solid_step1``, ``tidemodel.solid_full`` or ``tidemodel.none``)
        - General relativity (IERS 2010 Eq. 10.12; ``propsettings.use_relativistic_correction``)
        - Radiation pressure
        - Atmospheric drag: NRL-MSISE 2000 density model, with option to include space weather effects

        An end time is required: pass ``end`` or one of ``duration``, ``duration_secs``,
        ``duration_days`` (``TypeError`` if none is given; ``ValueError`` for a NaN or
        infinite ``duration_secs`` / ``duration_days``).

        For future propagation (beyond available data files):

        - Earth orientation parameters use the last available values (constant extrapolation)
        - Space weather past the observed record comes from the NOAA/SWPC 45-day
          forecast and then NASA's MSAFE monthly forecast, which carries a climatological Ap.
          With no space-weather table at all, F10.7 = F10.7A = 150 and Ap = 4.

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

def omm_from_url(url: str) -> list[OMMDict]:
    """Load OMM(s) from a URL as a list of dictionaries

    Fetches the content at the given URL and auto-detects JSON vs XML format.
    Returns a list of dictionaries that can be passed directly to :func:`sgp4`.

    Args:
        url (str): URL to fetch OMM data from (e.g. CelesTrak or Space-Track endpoint)

    Returns:
        list[OMMDict]: one dictionary per message, with every CCSDS field the
            source provided plus its extra keys (see :class:`OMMDict`)

    Raises:
        RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
            ``satkit.utils.set_offline(True)``); no connection is opened

    Example:
        ```python
        import satkit as sk

        omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")
        pos, vel = sk.sgp4(omms[0], sk.time(2024, 1, 1))
        ```
    """
    ...

def omm_from_file(filename: str) -> list[OMMDict]:
    """Load OMM(s) from a JSON or XML file as a list of dictionaries

    The format is detected from the content, not the file extension: a file
    starting with ``[`` or ``{`` is JSON (one message or an array of them), one
    starting with ``<`` is a CCSDS NDM/XML document. KVN is not supported.

    Args:
        filename (str): path to the file

    Returns:
        list[OMMDict]: one dictionary per message (see :class:`OMMDict`)

    Example:
        ```python
        import satkit as sk

        omms = sk.omm_from_file("gp.xml")   # saved from Space-Track or CelesTrak
        pos, vel = sk.sgp4(omms, sk.time(2024, 1, 1))
        ```
    """
    ...

def omm_from_text(text: str) -> list[OMMDict]:
    """Parse OMM(s) from JSON or XML text as a list of dictionaries

    Text starting with ``[`` or ``{`` is parsed as JSON (one message or an
    array of them), text starting with ``<`` as a CCSDS NDM/XML document.
    KVN is not supported.

    Args:
        text (str): the document

    Returns:
        list[OMMDict]: one dictionary per message (see :class:`OMMDict`)

    Example:
        ```python
        import satkit as sk
        import requests

        r = requests.get("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=xml")
        omm = sk.omm_from_text(r.text)[0]
        ```
    """
    ...
