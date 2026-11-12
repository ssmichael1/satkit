"""
Toolkit containing functions and classes used in satellite dynamics
calculations.
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import datetime
import os

from collections.abc import Sequence
from typing import Any, ClassVar, Final, Literal, Optional, TypeAlias, Union, final, overload
from typing_extensions import deprecated

from ._types import OMMDict, TLEFitResult

# Time inputs are polymorphic: a function or method argument annotated with one
# of these aliases takes a ``satkit.time``, a ``datetime.datetime`` or a
# ``numpy.datetime64`` interchangeably (and, for the vectorized functions, a
# list or numpy array of them). These aliases capture that so the individual
# signatures stay readable.
#
# This applies to arguments only where the alias is written. The ``time``
# class itself is narrower: its arithmetic and comparison operators take
# ``satkit.time`` / ``satkit.duration`` (plus ``datetime.timedelta`` and
# day counts for ``+`` and ``-``), not ``datetime.datetime``. Convert first
# with ``time(dt)`` or ``time.from_datetime(dt)``.
#
# A ``datetime.datetime`` is converted with Python's own convention
# (``datetime.timestamp()``): a naive datetime is the machine's local time, an
# aware one uses its own offset (see ``time.from_datetime``).
#
# A ``numpy.datetime64`` (scalar or 1-D array, any unit) is a UTC label:
# numpy has no time zone and no leap seconds. Sub-microsecond values round to
# the nearest microsecond; ``NaT`` raises ``ValueError``.
#
# * ``TimeScalar``    — a single time value.
# * ``TimeArrayLike`` — a list or tuple, a 1-D numpy object array of time
#   values, or a 1-D numpy ``datetime64`` array (not any array-like: a float
#   array or a string is refused).
# * ``TimeInput``     — either a scalar or an array of times.
TimeScalar: TypeAlias = "time | datetime.datetime | np.datetime64"
TimeArrayLike: TypeAlias = (
    "list[time] | list[datetime.datetime] | list[np.datetime64] | list[time | datetime.datetime | np.datetime64] | tuple[time | datetime.datetime | np.datetime64, ...] | npt.NDArray[np.object_] | npt.NDArray[np.datetime64]"
)
TimeInput: TypeAlias = "TimeScalar | TimeArrayLike"

# ``datetime.timedelta``, for use inside ``class time``, whose deprecated
# ``datetime()`` method hides the module name there
_TimeDelta: TypeAlias = datetime.timedelta

class TLEWarning(UserWarning):
    """Warning for a TLE record skipped by a loader called with ``on_error="warn"``

    The message is the one ``on_error="raise"`` raises as ``ValueError``: it gives
    the input line the record starts on, its satellite and what failed to parse.
    With ``on_error="warn"``, input that holds no element sets at all also gets
    one, saying so.
    """

@final
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
            "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9996",
            "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302874",
        ])[0]
        print(tle.name)
        # ISS (ZARYA)
        ```
    """

    def __init__(self) -> None:
        """An empty element set, to be filled in through the properties

        ``TLE()`` takes no arguments. The result has the name ``"none"``,
        satellite number 0, a blank international designator, classification
        ``"U"``, the epoch J2000 (2000-01-01 12:00:00 TT) and every element
        zero, so it cannot be propagated until at least ``mean_motion`` and
        ``epoch`` are set. Element sets are normally loaded with
        ``TLE.from_lines``, ``TLE.from_file``, ``TLE.from_url`` or
        ``TLE.from_omm``, or fitted with ``TLE.fit_from_states``.

        Example:
            ```python
            tle = satkit.TLE()
            tle.satnum = 99999
            tle.epoch = satkit.time(2024, 1, 1)
            tle.inclination = 51.6
            tle.mean_motion = 15.5
            pos, vel = satkit.sgp4(tle, tle.epoch)
            ```
        """
        ...

    def __eq__(self, other: object) -> bool:
        """True if ``other`` is a TLE with the same name and element set

        Every field is compared exactly. The SGP4 initialization that
        ``sgp4`` caches in the object is not part of the comparison, so a TLE
        equals its copy or pickle whether or not it has been propagated.
        A TLE is mutable, so it is not hashable.
        """
        ...

    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __repr__(self) -> str:
        """One line naming the satellite and epoch

        For example ``TLE(name="ISS (ZARYA)", satnum=25544,
        intl_desig="98067A", epoch=2021-09-21T12:25:40.104192Z)``.
        ``str(tle)`` is the multi-line listing of every element.
        """
        ...

    def __str__(self) -> str:
        """Multi-line listing of the name, epoch and every element"""
        ...

    @staticmethod
    def from_file(
        filename: str | os.PathLike[str],
        *,
        check_checksum: bool = False,
        on_error: Literal["raise", "warn", "skip"] = "raise",
    ) -> list[TLE]:
        """Load TLEs from a text file, parsed as ``TLE.from_lines`` does

        Args:
            filename (str | os.PathLike): path of the text file holding the TLE lines
            check_checksum (bool, optional): as in ``TLE.from_lines``
            on_error (str, optional): as in ``TLE.from_lines``

        Returns:
            list[TLE]: one TLE per element set in the file

        Raises:
            FileNotFoundError: if the file does not exist; the exception
                names the path. Another I/O failure raises the matching
                ``OSError`` (``PermissionError``, ``IsADirectoryError``, ...)
            ValueError: with ``on_error="raise"``, if a record fails to parse
                or the file holds no TLEs; see ``TLE.from_lines``

        Example:
            ```python
            tles = satkit.TLE.from_file("gps-ops.txt")
            for tle in tles:
                print(tle.name, tle.satnum)
            ```
        """
        ...

    @staticmethod
    def from_lines(
        lines: Sequence[str],
        *,
        check_checksum: bool = False,
        on_error: Literal["raise", "warn", "skip"] = "raise",
    ) -> list[TLE]:
        """Load TLEs from a list of lines

        ``TLE.from_file`` and ``TLE.from_url`` parse their text the
        same way. See the class docstring for an example.

        Args:
            lines (Sequence[str]): the TLE lines (2-line or 3-line format,
                any number of element sets; any sequence type is accepted,
                e.g. list or tuple)
            check_checksum (bool, optional): also verify the checksum digit
                (column 69) of every data line. Default False.
            on_error (str, optional): what to do with input that is not a
                valid element set. ``"raise"`` (default) raises ``ValueError``
                at the first record that fails to parse, and also if the
                lines hold no element sets at all. ``"warn"`` leaves a bad
                record out and emits a ``TLEWarning`` with the same message
                (one more if the lines hold no element sets at all).
                ``"skip"`` leaves bad records out silently. Only ``"raise"``
                raises for what the lines contain: ``"warn"`` and ``"skip"``
                always return a list

        Returns:
            list[TLE]: one TLE per element set, even if there is only one.
            With ``on_error="warn"`` or ``"skip"``, one per element set that
            parsed: an empty list if none did, or if the lines (empty, or
            text that is not TLE data) held none to begin with.

        Raises:
            ValueError: with ``on_error="raise"``, if a record fails to parse
                (or, with ``check_checksum``, has a wrong checksum), where the
                message gives the line the record starts on and its
                satellite, or if the lines hold no element sets; with any
                ``on_error``, if it is not one of the three values
            TypeError: if ``lines`` is not a sequence of strings

        Example:
            ```python
            import warnings

            lines = [
                "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9996",
                "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302874",
                # A letter O in the epoch day
                "1 49140U 21082L   24O30.39663557  .00000076  00000-0  14180-4 0  9995",
                "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
            ]
            with warnings.catch_warnings(record=True) as skipped:
                warnings.simplefilter("always")
                tles = satkit.TLE.from_lines(lines, on_error="warn")
            print(len(tles), len(skipped))
            # 1 1
            ```
        """
        ...

    @staticmethod
    def from_url(
        url: str, *, on_error: Literal["raise", "warn", "skip"] = "raise"
    ) -> list[TLE]:
        """Load TLEs from a URL, parsing the response as ``TLE.from_lines`` does

        Works with any URL that returns plain-text TLE data.

        Args:
            url (str): URL to fetch TLE data from
            on_error (str, optional): as in ``TLE.from_lines``

        Returns:
            list[TLE]: one TLE per element set in the response

        Raises:
            ValueError: with ``on_error="raise"``, if a record fails to parse
                or the response holds no TLEs; see ``TLE.from_lines``
            RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
                ``satkit.utils.set_offline(True)``; no connection is opened)
                or the request fails

        Example:
            ```python
            tles = sk.TLE.from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle")
            ```
        """
        ...

    @property
    def satnum(self) -> int:
        """Satellite number, or equivalently the NORAD catalog number

        A number above 339999 (the end of the Alpha-5 scheme) is allowed, as
        OMM carries up to nine digits, but cannot be written by ``to_2line``.
        """
        ...

    @satnum.setter
    def satnum(self, value: int) -> None:
        """Set the satellite number; ``ValueError`` unless 0 <= value <= 999999999"""
        ...

    @property
    def intl_desig(self) -> str:
        """International designator as written in the TLE (line 1, columns 10-17)

        For example ``"98067A"``: the last two digits of the launch year,
        the three-digit launch number of that year and the piece of the
        launch (one to three letters). An empty string if the element set
        has none, as for an analyst object or a new ``TLE()``.

        This string is the stored value: ``to_2line`` writes it and
        ``to_omm`` derives ``OBJECT_ID`` from it. ``desig_year``,
        ``desig_launch`` and ``desig_piece`` are read from it, and setting
        any of them rewrites it.
        """
        ...

    @intl_desig.setter
    def intl_desig(self, value: str) -> None:
        """Set the international designator

        ``ValueError`` if it is longer than the 8 columns of the TLE field or
        holds anything but printable ASCII. Surrounding spaces are removed.
        """
        ...

    @property
    def desig_year(self) -> int | None:
        """Launch year from the international designator, two digits as in the TLE

        The first two characters of ``intl_desig`` (98 for ``"98067A"``):
        57-99 are 1957-1999 and 0-56 are 2000-2056. ``None`` if the
        designator is blank or does not start with two digits.
        """
        ...

    @desig_year.setter
    def desig_year(self, value: int) -> None:
        """Set the two-digit launch year, rewriting ``intl_desig``

        ``ValueError`` unless 0 <= value <= 99. A launch number the
        designator does not have yet is written as ``000``.
        """
        ...

    @property
    def desig_launch(self) -> int | None:
        """Launch number of the year from the international designator

        Characters 3-5 of ``intl_desig`` (67 for ``"98067A"``). ``None`` if
        the designator is blank or does not hold three digits there.
        """
        ...

    @desig_launch.setter
    def desig_launch(self, value: int) -> None:
        """Set the launch number of the year, rewriting ``intl_desig``

        ``ValueError`` unless 0 <= value <= 999. A launch year the
        designator does not have yet is written as ``00``.
        """
        ...

    @property
    def desig_piece(self) -> str:
        """Piece of the launch from the international designator (e.g. "A")

        What follows the year and launch number in ``intl_desig``; an empty
        string if the designator is blank or has no piece.
        """
        ...

    @desig_piece.setter
    def desig_piece(self, value: str) -> None:
        """Set the piece of the launch, rewriting ``intl_desig``

        ``ValueError`` unless it is at most three ASCII letters. A launch
        year and number the designator does not have yet are written as zeros.
        """
        ...

    @property
    def classification(self) -> str:
        """Classification (line 1, column 8): "U", "C" or "S" in practice, but any single ASCII letter round-trips"""
        ...

    @classification.setter
    def classification(self, value: str) -> None:
        """Set the classification; must be a single ASCII letter, or raises ValueError"""
        ...

    @property
    def ephem_type(self) -> int:
        """Ephemeris type (usually 0). A value of 4 marks an SGP4-XP element set, which ``sgp4`` rejects: satkit implements classic SGP4 only."""
        ...

    @ephem_type.setter
    def ephem_type(self, value: int) -> None:
        """Set the ephemeris type; ``OverflowError`` unless 0 <= value <= 255 (``to_2line`` writes a value above 9 as 0)"""
        ...

    @property
    def element_num(self) -> int:
        """Element set number"""
        ...

    @element_num.setter
    def element_num(self, value: int) -> None:
        """Set the element set number (``to_2line`` writes it modulo 10,000, a negative value as 0)"""
        ...

    @property
    def rev_num(self) -> int:
        """Revolution number at epoch"""
        ...

    @rev_num.setter
    def rev_num(self, value: int) -> None:
        """Set the revolution number at epoch (``to_2line`` writes it modulo 100,000, a negative value as 0)"""
        ...

    @property
    def raan(self) -> float:
        """Right Ascension of Ascending Node, in degrees"""
        ...

    @raan.setter
    def raan(self, value: float) -> None:
        """Set the Right Ascension of Ascending Node, in degrees; ``ValueError`` if not finite"""
        ...

    @property
    def eccen(self) -> float:
        """Satellite eccentricity, unitless, in range [0, 1)"""
        ...

    @eccen.setter
    def eccen(self, value: float) -> None:
        """Set the satellite eccentricity; ``ValueError`` unless 0 <= value < 1"""
        ...

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly in degrees"""
        ...

    @mean_anomaly.setter
    def mean_anomaly(self, value: float) -> None:
        """Set the satellite mean anomaly, degrees; ``ValueError`` if not finite"""
        ...

    @property
    def mean_motion(self) -> float:
        """Mean motion in revs / day"""
        ...

    @mean_motion.setter
    def mean_motion(self, value: float) -> None:
        """Set the satellite mean motion, revs / day; ``ValueError`` if negative or not finite"""
        ...

    @property
    def inclination(self) -> float:
        """Inclination, in degrees"""
        ...

    @inclination.setter
    def inclination(self, value: float) -> None:
        """Set the satellite inclination, degrees; ``ValueError`` if not finite"""
        ...

    @property
    def epoch(self) -> time:
        """TLE epoch"""
        ...

    @epoch.setter
    def epoch(self, value: TimeScalar) -> None:
        """Set the TLE epoch: a single ``time``, ``datetime.datetime`` or ``numpy.datetime64``

        ``TypeError`` for anything else, including a list of times.
        ``to_2line`` can only write an epoch in the years 1957-2056.
        """
        ...

    @property
    def arg_of_perigee(self) -> float:
        """Argument of Perigee, in degrees"""
        ...

    @arg_of_perigee.setter
    def arg_of_perigee(self, value: float) -> None:
        """Set the argument of perigee, degrees; ``ValueError`` if not finite"""
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
        """Set the 1/2 of first derivative of mean motion, in revs/day^2; ``ValueError`` if not finite"""
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
        """Set the 1/6 of 2nd derivative of mean motion, in revs/day^3; ``ValueError`` if not finite"""
        ...

    @property
    def name(self) -> str:
        """The name of the satellite: the name line of a 3-line element set

        ``"none"`` for an element set that has no name (one read from two
        lines, or a new ``TLE()``).
        """
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
        """Set the drag term (B*) of the satellite, in units of 1 / Earth radii; ``ValueError`` if not finite"""
        ...

    @staticmethod
    def from_omm(omm: OMMDict | dict[str, Any]) -> TLE:
        """Build a TLE from an OMM (Orbital Mean-Element Message) dictionary

        The dictionary is the same shape ``sgp4`` accepts: the flat CCSDS
        keys of a Space-Track / CelesTrak JSON record (see ``OMMDict``),
        or the nested groups of an ``xmltodict`` rendering of the XML form.
        Numbers may be strings.

        The six mean elements, epoch, ``BSTAR``, ``MEAN_MOTION_DOT``,
        ``MEAN_MOTION_DDOT``, ``NORAD_CAT_ID``, ``ELEMENT_SET_NO``,
        ``REV_AT_EPOCH`` and ``EPHEMERIS_TYPE`` carry over; of these, the
        optional ones become zero when absent. ``OBJECT_NAME`` becomes the
        name, and the unnamed-TLE name ``"none"`` when it is absent, empty or
        ``UNKNOWN``. ``OBJECT_ID`` in ``YYYY-NNNP`` form becomes the
        international designator (``1998-067A`` gives ``98067A``); any other
        form leaves the designator empty. ``CLASSIFICATION_TYPE`` becomes the
        classification, ``"U"`` unless it is a single ASCII letter. Other
        metadata is dropped.

        This inverts ``to_omm``: ``TLE.from_omm(tle.to_omm()) == tle`` for a
        TLE read from two or three lines.

        Args:
            omm (OMMDict | dict): OMM dictionary

        Returns:
            TLE: the equivalent two-line element set

        Raises:
            TypeError: if ``omm`` is not a dict
            ValueError: if a required key (``EPOCH`` or one of the six mean
                elements) is missing or a value cannot be read

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

        The result uses the flat CCSDS keys (see ``OMMDict``) with
        ``EPOCH`` as an RFC 3339 string, angles in degrees and mean motion in
        revolutions per day, and can be passed back to ``sgp4`` or
        serialized with ``json.dumps``. ``OBJECT_ID`` is derived from the
        international designator (``98067A`` becomes ``1998-067A``, a blank
        one ``UNKNOWN``), ``OBJECT_NAME`` is the name (``UNKNOWN`` for an
        unnamed TLE), and ``CLASSIFICATION_TYPE`` is the TLE's classification
        letter. A negative ``satnum``, ``element_num`` or ``rev_num`` and an
        ``ephem_type`` above 9 have no OMM form and are left out.

        Returns:
            OMMDict: OMM dictionary
        """
        ...

    def to_2line(self) -> list[str]:
        """
        Output as 2 canonical TLE Lines

        The element set number (4 columns) and revolution number (5 columns) are
        written modulo 10,000 and 100,000, so a larger value wraps around the way
        catalog TLEs roll the revolution counter over.

        Returns:
            2 canonical TLE Lines

        Raises:
            ValueError: if the element set does not fit the format:
                ``satnum`` is 340000 or above (Alpha-5, used for the
                5-character satellite number field, ends at ``Z9999`` =
                339999; use ``to_omm()`` instead) or negative, ``eccen`` is
                outside [0, 1), or the epoch year is outside 1957-2056 (the
                span of the two-digit year)

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
            3-line element set: the name (``"none"`` for an unnamed TLE),
            then the 2 canonical TLE lines of ``to_2line``

        Raises:
            ValueError: as ``to_2line``

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
        states: npt.ArrayLike,
        times: TimeInput,
        epoch: TimeScalar,
        *,
        gravconst: sgp4_gravconst = ...,
        opsmode: sgp4_opsmode = ...,
    ) -> tuple[TLE, TLEFitResult]:
        """
        Perform non-linear least squares fit of TLE parameters to a list of GCRF states

        Args:
            states: GCRF states to fit to: an Nx6 array, or a list of 6-element vectors. The first 3 values are positions in meters. The last 3 values are velocities in meters / second.
            times: Times corresponding to the states, one per state: a list or
                array of times (a single time is accepted for a single state)
            epoch: Epoch time for the TLE: a single time, within the range of ``times``

        Keyword Args:
            gravconst: gravity model SGP4 uses in the fit.  Default is sgp4_gravconst.wgs72
            opsmode: SGP4 ops mode used in the fit.  Default is sgp4_opsmode.afspc

        Returns:
            Fitted TLE and the fit results, a dictionary (see ``TLEFitResult``).
            The fitted TLE has the name ``"none"``, satellite number 0 and a
            blank international designator; set them afterwards if needed.

        Raises:
            TypeError: if ``states`` is not a sequence of 6-element vectors,
                ``times`` or ``epoch`` is not a time (``epoch`` must be a single
                time, not a list), or ``gravconst`` / ``opsmode`` is not the enum
            ValueError: if ``states`` is empty, ``states`` and ``times`` differ in
                length, or ``epoch`` is outside the range of ``times``
            RuntimeError: if the fit itself fails (SGP4 cannot evaluate the
                elements, singular normal equations, or it ends on elements a
                TLE cannot hold)

        Notes:
            SGP4 propagator is used to match TLE to the states. The fitted TLE
            reproduces the states when propagated with ``sgp4`` under the same
            ``gravconst`` and ``opsmode``; the defaults match ``sgp4``'s.
            satkit before 0.24 fitted with WGS84.
            Input GCRF states are rotated into TEME frame used by SGP4, with the
            full IERS 2010 rotation.
            First and second derivatives of mean motion are ignored, as they are not used by SGP4.

            Non-linear Levenberg-Marquardt optimization fits the orbital
            elements, mean motion and drag (bstar) to the provided states. The
            solver is parameterized in equinoctial elements, which are not
            singular for circular or equatorial (e.g. geostationary) orbits,
            and converts to the classical TLE elements at the end. It is
            built on top of the ``numeris`` linear algebra crate.

            The results dictionary (``TLEFitResult``) has the keys
            ``status`` (a ``tlefitstatus``), ``converged`` (bool),
            ``orig_norm``, ``best_norm``, ``grad_norm`` (floats), ``n_iter``
            and ``n_res_evals`` (ints).

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

@overload
def sgp4(
    tle: TLE | OMMDict | dict[str, Any] | list[TLE] | list[OMMDict] | list[dict[str, Any]] | list[TLE | OMMDict | dict[str, Any]],
    time: TimeInput,
    *,
    gravconst: sgp4_gravconst = ...,
    opsmode: sgp4_opsmode = ...,
    errflag: typing.Literal[True],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int32]]: ...
@overload
def sgp4(
    tle: TLE | OMMDict | dict[str, Any] | list[TLE] | list[OMMDict] | list[dict[str, Any]] | list[TLE | OMMDict | dict[str, Any]],
    time: TimeInput,
    *,
    gravconst: sgp4_gravconst = ...,
    opsmode: sgp4_opsmode = ...,
    errflag: typing.Literal[False] = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload
def sgp4(
    tle: TLE | OMMDict | dict[str, Any] | list[TLE] | list[OMMDict] | list[dict[str, Any]] | list[TLE | OMMDict | dict[str, Any]],
    time: TimeInput,
    *,
    gravconst: sgp4_gravconst = ...,
    opsmode: sgp4_opsmode = ...,
    errflag: bool,
) -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int32]]
):
    """SGP-4 propagator for TLE

    Run Simplified General Perturbations (SGP)-4 propagator on Two-Line Element Set to
    output satellite position and velocity at given time
    in the "TEME" coordinate system.

    A detailed description is in Vallado, Crawford, Hujsak and Kelso,
    "Revisiting Spacetrack Report #3", AIAA 2006-6753 (AIAA/AAS Astrodynamics
    Specialist Conference, 2006).

    Args:
        tle (TLE | OMMDict | list[TLE | OMMDict]): element set(s) to propagate: a
            ``TLE`` object, an OMM dictionary (see ``OMMDict``), or a list of either
            (or mixing both). Must be a ``list``; tuples and other sequences are not accepted
        time (time | datetime.datetime | numpy.datetime64 | list | tuple | numpy.ndarray): time(s) at
            which to compute position and velocity: a single time, a list or tuple of times, a 1-D
            numpy object array of times or a 1-D ``datetime64`` array.
            A naive ``datetime`` is local time, not UTC (see ``time.from_datetime``); a
            ``numpy.datetime64`` is UTC

    Keyword Args:
        gravconst (satkit.sgp4_gravconst): gravity constant to use.  Default is sgp4_gravconst.wgs72
        opsmode (satkit.sgp4_opsmode): sgp4_opsmode.afspc (Air Force Space Command) or sgp4_opsmode.improved.  Default is sgp4_opsmode.afspc
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
            ``sgp4_error``, so comparing against the enum works
            elementwise: ``err == satkit.sgp4_error.success`` is a boolean array.

    Notes:
        - **Units:** the canonical Vallado SGP4 implementation (and most other SGP4
          libraries) return position in kilometers and velocity in kilometers/second.
          satkit converts these to meters and meters/second so that SGP4 output is
          consistent with every other position and velocity in the library.
        - **OMM dictionaries:** any dict with the CCSDS keys ``EPOCH``, ``MEAN_MOTION``,
          ``ECCENTRICITY``, ``INCLINATION``, ``RA_OF_ASC_NODE``, ``ARG_OF_PERICENTER`` and
          ``MEAN_ANOMALY`` (plus optional ``BSTAR``, ``MEAN_MOTION_DOT``, ``MEAN_MOTION_DDOT``)
          is accepted, whether it came from ``omm_from_url`` / ``omm_from_file`` /
          ``omm_from_text``, from ``json.load`` on a CelesTrak or Space-Track response
          (numbers may be strings), or from ``xmltodict.parse`` on the XML form: the whole
          result (root key ``omm``, or ``ndm`` holding one ``omm``) or any node below it, the
          nested ``header`` / ``body`` / ``segment`` / ``metadata`` / ``data`` /
          ``meanElements`` / ``tleParameters`` groups being understood. An ``ndm`` holding
          several messages is not one element set: pass ``sgp4`` the list
          ``[{"omm": m} for m in d["ndm"]["omm"]]``. ``EPOCH`` may be an
          RFC 3339 string, a CCSDS day-of-year string (``YYYY-DDDThh:mm:ss``), a ``satkit.time``, a
          ``datetime`` (naive = local time) or a ``numpy.datetime64`` (UTC). Other keys are ignored,
          except that ``MEAN_ELEMENT_THEORY`` must be ``SGP4``, ``TIME_SYSTEM`` must be
          ``UTC``, ``REF_FRAME`` must be ``TEME``, ``CENTER_NAME`` must be ``EARTH`` and
          ``EPHEMERIS_TYPE`` must not be 4 (SGP4-XP) when present.
        - The "TEME" frame of the SGP4 state vectors is not a truly inertial frame.  It is a "True Equator Mean Equinox"
          frame: its axes are the true equator and the mean equinox of date, i.e. of each output time (not of the
          TLE epoch), so the frame itself moves with precession and nutation. Rotate to GCRF with
          ``frametransform.rotation(frame.TEME, frame.GCRF, time)``.
        - **Errors:** a time at which propagation fails (e.g. the orbit has decayed) gives a NaN row,
          with its code in the error array when ``errflag`` is True. An element set that cannot be
          initialized at all (e.g. decayed or eccentricity out of range at epoch, or an SGP4-XP set)
          raises ``RuntimeError`` when ``errflag`` is False; for a list, the message gives its index.
          When ``errflag`` is True it does not raise: its rows are NaN at every time and every time
          carries its init code (``sgp4_error.unsupported`` for an SGP4-XP set or OMM metadata SGP4
          cannot use), so one bad element set does not fail a list.
        - **Leap seconds:** the time since epoch is the physical (SI) time elapsed. Across a leap second
          this is one second more than the difference of the UTC labels that Vallado's reference code
          and python-sgp4 use, so satkit differs from them by 1 s of along-track motion (~7.6 km at
          LEO) per leap second between epoch and time. This is deliberate: the satellite really flies
          86,401 s over such a day, and the SGP4 mean motion is per SI day.

    Raises:
        TypeError: if ``tle`` is not a ``TLE``, a dict or a list of them, ``time`` is not a
            time or a list / array of times, or ``gravconst`` / ``opsmode`` is not a member of
            its enum
        ValueError: if an OMM dictionary lacks a required key or holds a value that cannot be
            read
        RuntimeError: with ``errflag`` False, if an element set cannot be initialized (see
            **Errors** above)

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

@final
class sgp4_gravconst:
    """Gravity constants (Earth radius, mu, J2-J4) used by SGP4 propagation

    Element sets are generated with WGS-72, so ``wgs72`` (the default of
    ``sgp4`` and ``TLE.fit_from_states``) is the one that reproduces the
    orbit a TLE or OMM describes.

    Members compare equal to their integer value (``int(m)``).
    """

    wgs72: ClassVar[sgp4_gravconst]
    """WGS-72, the default (value 0)"""

    wgs72old: ClassVar[sgp4_gravconst]
    """WGS-72 with the older, lower-precision constants of the original Spacetrack Report #3 code (value 1)"""

    wgs84: ClassVar[sgp4_gravconst]
    """WGS-84 (value 2)"""

    def __int__(self) -> int:
        """The member's integer value"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal to the same member, or to its integer value"""
        ...

    def __hash__(self) -> int:
        """Members are hashable and immutable; one hashes like the integer it equals"""
        ...

@final
class sgp4_opsmode:
    """Ops mode for SGP4 propagation

    Selects between the two variants of Vallado's SGP4 code. They differ
    only for deep-space element sets (period of 225 minutes or more), in how
    the ascending node is wrapped in the lunar-solar periodic terms.

    Members compare equal to their integer value (``int(m)``).
    """

    afspc: ClassVar[sgp4_opsmode]
    """Reproduces the Air Force Space Command code that generates TLEs; the default (value 0)"""

    improved: ClassVar[sgp4_opsmode]
    """Vallado's "improved" mode (value 1)"""

    def __int__(self) -> int:
        """The member's integer value"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal to the same member, or to its integer value"""
        ...

    def __hash__(self) -> int:
        """Members are hashable and immutable; one hashes like the integer it equals"""
        ...

@final
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

    Members are hashable, compare equal to their integer value and to each
    other by value, and cannot be constructed.
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
    Compiled in (truncated to degree 70).
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
    Compiled in (truncated to degree 70). Default for the orbit propagator.
    """

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def nrlmsise00(
    alt_km: float,
    *,
    latitude_deg: float = 0.0,
    longitude_deg: float = 0.0,
    time: TimeScalar | None = None,
    use_spaceweather: bool = True,
) -> tuple[float, float]:
    """NRL-MSISE00 atmospheric density model

    ``satkit.density.nrlmsise`` is the same model taking an ``itrfcoord``,
    or meters and radians.

    Args:
        alt_km (float): Altitude above the WGS-84 ellipsoid in kilometers
            (-1 or above; the model runs from the ground to the exosphere)

    Keyword Args:
        latitude_deg (float): Geodetic latitude in degrees, in [-90, 90].
            Default is 0
        longitude_deg (float): Longitude in degrees. Default is 0
        time (satkit.time | datetime.datetime | numpy.datetime64, optional):
            Time at which to evaluate the model. Without a time the model is
            evaluated for 00:00 UT on 1 January, on default indices
            (F10.7 = F10.7A = 150, Ap = 4)
        use_spaceweather (bool): Look up F10.7 and Ap in the space-weather
            data at ``time``. Default is True. It has no effect without a
            ``time``; with False, or for a time the data does not cover, the
            default indices are used

    Returns:
        tuple[float, float]: Mass density (kg/m^3) and temperature (K)

    Raises:
        TypeError: ``time`` is not a single time value, or a number is expected
        ValueError: ``alt_km`` is NaN, infinite or below -1; ``latitude_deg``
            is NaN or outside [-90, 90]; ``longitude_deg`` is NaN or infinite

    Example:
        ```python
        rho, temp = satkit.nrlmsise00(400.0, latitude_deg=30.0, longitude_deg=-70.0)
        print(f"{rho:.2e} kg/m^3, {temp:.0f} K")
        ```
    """
    ...

def gravity(
    pos: itrfcoord | npt.ArrayLike,
    *,
    model: gravmodel = ...,
    degree: int = 6,
    order: int | None = None,
) -> npt.NDArray[np.float64]:
    """Return acceleration due to Earth gravity at the input position

    Args:
        pos (satkit.itrfcoord | npt.ArrayLike): Position as ITRF coordinate, or 3-element ITRF position in meters (numpy array, list or tuple; integers are converted to float)

    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.egm2008
        degree (int): Maximum degree of gravity model to use, 0 to 70.  Default is 6.  0 and 1 both give the point-mass field
        order (int): Maximum order of gravity model to use, 0 to ``degree``.  Default is same as degree

    Returns:
        3-element array (shape ``(3,)``): acceleration in m/s^2 in the International Terrestrial Reference Frame (ITRF). The centrifugal acceleration of the rotating frame is not included

    Raises:
        TypeError: ``pos`` is not an itrfcoord or a real numeric array-like; ``model`` is not a ``gravmodel``; ``degree`` or ``order`` is not an integer
        ValueError: ``pos`` does not have 3 elements, holds a NaN or infinity, or is the origin; ``degree`` is negative or above 70; ``order`` is negative or above ``degree``
        RuntimeError: the coefficient file of ``gravmodel.itugrace16`` is not in the data directory and cannot be downloaded

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
        pos (satkit.itrfcoord | npt.ArrayLike): Position as ITRF coordinate, or 3-element ITRF position in meters (numpy array, list or tuple; integers are converted to float)

    Keyword Args:
        model (gravmodel): The gravity model to use.  Default is gravmodel.egm2008
        degree (int): Maximum degree of gravity model to use, 0 to 70.  Default is 6.  0 and 1 both give the point-mass field
        order (int): Maximum order of gravity model to use, 0 to ``degree``.  Default is same as degree

    Returns:
        Tuple of the acceleration in m/s^2 in the ITRF (shape ``(3,)``) and its partial derivatives with respect to ITRF position (shape ``(3, 3)``, ``partials[i, j] = d a_i / d r_j``) in m/s^2 / m

    Raises:
        TypeError: ``pos`` is not an itrfcoord or a real numeric array-like; ``model`` is not a ``gravmodel``; ``degree`` or ``order`` is not an integer
        ValueError: ``pos`` does not have 3 elements, holds a NaN or infinity, or is the origin; ``degree`` is negative or above 70; ``order`` is negative or above ``degree``
        RuntimeError: the coefficient file of ``gravmodel.itugrace16`` is not in the data directory and cannot be downloaded


    For details of calculation, see Chapter 3.2 of: "Satellite Orbits: Models, Methods, Applications", O. Montenbruck and E. Gill, Springer, 2000 (https://doi.org/10.1007/978-3-642-58351-3).

    """
    ...

@final
class solarsystem:
    """Solar system bodies for which the JPL ephemeris (``satkit.jplephem``)
    and the low-precision planetary ephemeris (``satkit.planets``) can be
    computed

    Notes:
        - ``EMB`` is the Earth-Moon barycenter; there is no ``Earth`` member
          (see ``satkit.jplephem.barycentric_pos`` for the Earth's position)
        - Members compare equal to their integer value
          (``int(solarsystem.Sun)`` is 10), the body's index in the JPL DE files
    """

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

    def __int__(self) -> int:
        """The member's integer value (members also compare equal to it)"""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class sgp4_error:
    """Error codes of SGP4 propagation, as in the error array of ``sgp4(..., errflag=True)``

    That array holds plain ``int32`` codes; a member compares equal to its
    code, so ``err == satkit.sgp4_error.success`` works elementwise, and
    ``satkit.sgp4_error(code)`` turns one code back into a member.

    Example:
        ```python
        print(int(satkit.sgp4_error.orbit_decay))
        # 6
        print(satkit.sgp4_error.orbit_decay == 6)
        # True
        print(satkit.sgp4_error(6))
        # sgp4_error.orbit_decay
        ```
    """

    success: ClassVar[sgp4_error]
    """Success (code 0)"""

    eccen: ClassVar[sgp4_error]
    """Mean eccentricity at the requested time is 1 or more, or below -0.001 (code 1)"""

    mean_motion: ClassVar[sgp4_error]
    """Mean motion is zero or negative (code 2)"""

    perturb_eccen: ClassVar[sgp4_error]
    """Perturbed eccentricity outside [0, 1] (code 3)"""

    semi_latus_rectum: ClassVar[sgp4_error]
    """Semi-latus rectum negative (code 4)"""

    unused: ClassVar[sgp4_error]
    """Not produced (code 5); kept because the reference code numbers it"""

    orbit_decay: ClassVar[sgp4_error]
    """Orbit decayed: the computed radius is less than one Earth radius (code 6)"""

    unsupported: ClassVar[sgp4_error]
    """Only in the error array of ``sgp4(..., errflag=True)``: the element set cannot be
    propagated by classic SGP4 at all (an SGP4-XP set, or OMM metadata naming another theory,
    time system, frame or center) (code 7)"""

    def __new__(cls, code: int) -> sgp4_error:
        """The member with integer code ``code``

        For example ``satkit.sgp4_error(int(err[0]))`` for an element of the
        error array of ``sgp4(..., errflag=True)``.

        Raises:
            ValueError: if ``code`` is not one of the codes 0 to 7
        """
        ...

    def __int__(self) -> int:
        """The member's integer code"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal to the same member, or to its integer code"""
        ...

    def __hash__(self) -> int:
        """Members are hashable and immutable; one hashes like the integer it equals"""
        ...

@final
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
    - `Invalid`: placeholder that is not a day; ``time.weekday`` never returns it
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
    """Placeholder that is not a day of the week; ``time.weekday`` never returns it"""

    def __int__(self) -> int:
        """The integer value of the member (members also compare equal to it)"""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class tlefitstatus:
    """
    Termination status of the TLE non-linear least-squares fit performed by
    ``TLE.fit_from_states``.

    Values:

    - ``GradientConverged``: converged on gradient norm tolerance
    - ``StepConverged``: converged on relative step size tolerance
    - ``CostConverged``: converged on relative cost change tolerance
    - ``MaxIterations``: maximum number of iterations reached
    - ``DampingSaturated``: Levenberg-Marquardt damping parameter saturated

    ``str(status)`` is a one-line description. Members compare equal to
    their integer value (``int(m)``, 0 to 4 in the order above).
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
        """True for the three ``...Converged`` members, False for ``MaxIterations`` and ``DampingSaturated``"""
        ...

    def __int__(self) -> int:
        """The member's integer value"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal to the same member, or to its integer value"""
        ...

    def __hash__(self) -> int:
        """Members are hashable and immutable; one hashes like the integer it equals"""
        ...

@final
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

    - `Invalid`: placeholder that is not a time scale. Passing it as a
      ``scale`` argument raises ``ValueError``
    - `UTC`: Universal Time Coordinate
    - `TT`: Terrestrial Time
    - `UT1`: UT1
    - `TAI`: International Atomic Time
    - `GPS`: Global Positioning System (GPS) time
    - `TDB`: Barycentric Dynamical Time
    """

    Invalid: ClassVar[timescale]
    """Placeholder that is not a time scale

    Every ``satkit.time`` method that takes a ``scale`` raises ``ValueError``
    when given it.
    """

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

    def __int__(self) -> int:
        """The integer value of the member (members also compare equal to it)"""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class frame:
    """Coordinate reference frame

    Names the frame of a vector: for ``satkit.frametransform`` (``to_gcrf``,
    ``from_gcrf``, ``rotation``, ``transform_state`` and the other
    frame-pair functions), for impulsive maneuvers and for continuous
    thrust. Not every function accepts every frame: a thrust or maneuver
    must be in ``GCRF``, ``RTN``, ``NTW`` or ``LVLH``, and each
    ``frametransform`` function lists the frames it supports.

    Members are hashable, compare equal to their integer value and to each
    other by value, and cannot be constructed (``frame(3)`` is a ``TypeError``).

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
    See [`RTN`][satkit.frame.RTN] for the axis definition.
    """

    RIC: ClassVar[frame]
    """Alias for ``frame.RTN`` — the older NASA / Clohessy-Wiltshire name
    (Radial / In-track / Cross-track). ``frame.RIC == frame.RTN`` is
    True. Kept for backward compatibility with code written against
    earlier satkit versions where ``RIC`` was the canonical name. See
    [`RTN`][satkit.frame.RTN] for the axis definition.
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

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
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

    J2000: Final[time]
    """The J2000 epoch: 2000-01-01 12:00:00 TT (class constant, read-only)"""

    GPS_EPOCH: Final[time]
    """The GPS epoch: 1980-01-06 00:00:00 UTC (class constant, read-only)"""

    MJD_EPOCH: Final[time]
    """The Modified Julian Date epoch: 1858-11-17 00:00:00 UTC (class constant, read-only)"""

    UNIX_EPOCH: Final[time]
    """The Unix epoch: 1970-01-01 00:00:00 UTC (class constant, read-only)"""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, string: str, /) -> None: ...
    @overload
    def __init__(self, t: time | datetime.datetime | np.datetime64, /) -> None: ...
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
        - ``time(t)``: the instant of another ``satkit.time``, a
          ``datetime.datetime`` or a ``numpy.datetime64``
        - ``time(year, month, day)``: midnight at the start of the day
        - ``time(year, month, day, hour, min, sec)``: all six components

        Args:
            string: String representation of time
            t: A ``satkit.time`` (copied), a ``datetime.datetime`` (as in
                ``from_datetime``: a naive one is the machine's local time,
                an aware one uses its own offset) or a ``numpy.datetime64``
                (a UTC label; ``NaT`` raises ``ValueError``)
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
                interpreted, default is satkit.timescale.UTC. Only for the
                ``(year, month, day)`` and six-component forms: giving it
                with the string or no-argument form raises ``TypeError``.

        Raises:
            TypeError: If the number of positional arguments is not 0, 1, 3
                or 6; if a single argument is not a string or a time; if a component is
                not a number (the first five must be integers); or if
                ``scale`` is given with the string or no-argument form
            ValueError: If the string form cannot be parsed, if a component
                is out of range (month 13, hour 25, a second of 60 where
                there is no leap second, NaN, ...), if the date is beyond
                about ±292,000 years, or if ``scale`` is
                ``timescale.Invalid``

        Example:
            ```python
            print(satkit.time(2023, 3, 5, 11, 3, 45.453))
            # 2023-03-05T11:03:45.453000Z

            print(satkit.time(2023, 3, 5))
            # 2023-03-05T00:00:00.000000Z

            import datetime
            print(satkit.time(datetime.datetime(2023, 3, 5, tzinfo=datetime.timezone.utc)))
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

        RFC 3339 is tried first (see ``from_rfc3339``). Otherwise the
        numbers in the string are read in year, month, day, hour, minute,
        second order, so ISO-ordered strings (``"2024-01-04 13:14:12.123"``)
        and month-name strings (``"March 4 2024"``) work, but locale-ordered
        dates such as ``MM/DD/YYYY`` are not supported: use ``strptime``
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
              ``from_rfc3339`` or ``strptime`` when the format is known.

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
              ``to_rfc3339`` writes them outside 0000-9999).

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
                  ``+10000``, as ``strftime`` writes them outside 0000-9999)
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

        Raises:
            TypeError: If an argument is not an integer
            ValueError: If the month or day is out of range, or the date is
                beyond about ±292,000 years

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

        Raises:
            ValueError: If ``jd`` is NaN or infinite, or ``scale`` is
                ``timescale.Invalid``
            OverflowError: If the result is beyond about ±292,000 years

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
            unixtime (float): UTC seconds since Jan 1, 1970 00:00:00
                        (leap seconds are not included); rounded to the
                        nearest microsecond

        Returns:
            Time object representing input unixtime

        Raises:
            ValueError: If ``unixtime`` is NaN or infinite
            OverflowError: If the result is beyond about ±292,000 years

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
            week: GPS week number, counted from the GPS epoch
                (1980-01-06) without the 1024-week rollover
            seconds: GPS seconds of week, seconds; rounded to the nearest
                microsecond. Not limited to one week: 604800 is the start
                of the next week

        Returns:
            Time object representing input GPS week and second

        Raises:
            ValueError: If ``seconds`` is NaN or infinite
            OverflowError: If the result is beyond about ±292,000 years
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

        Raises:
            ValueError: If ``mjd`` is NaN or infinite, or ``scale`` is
                ``timescale.Invalid``
            OverflowError: If the result is beyond about ±292,000 years

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
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_date``."""
        ...

    def to_gregorian(
        self,
    ) -> tuple[int, int, int, int, int, float]:
        """Return tuple representing as UTC Gregorian date and time of the time object.

        Returns:
            Tuple with 6 elements representing the Gregorian year, month, day, hour, minute, and second of the time object.
                Month is in range [1,12].
                Day is in range [1,31].
                The second is the float nearest the exact (whole-microsecond)
                value, in [0, 60); inside a leap second it is in [60, 61).
        """
        ...

    @deprecated("use to_gregorian()")
    def as_gregorian(self) -> tuple[int, int, int, int, int, float]:
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_gregorian``."""
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
            sec (float): floating point second of minute, in range [0,60);
                up to 61 within a UTC leap second (e.g. ``23:59:60.5`` on
                2016-12-31) or a pre-1972 positive UTC step. Rounded to the
                nearest microsecond

        Returns:
            Time object representing input UTC Gregorian time

        Raises:
            TypeError: If an argument is not a number (the first five must
                be integers)
            ValueError: If a component is out of range (including a second
                of 60 or more where there is no leap second, and NaN), or
                the date is beyond about ±292,000 years

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
                which round-trips through ``time.from_datetime``.

        Returns:
            "datetime.datetime" object representing the same instant in time as the "satkit.time" object

        Raises:
            OverflowError: If the time is outside the years 1 to 9999 that
                ``datetime.datetime`` can represent

        Notes:
            ``datetime.datetime`` has no leap seconds: a time inside one
            (``23:59:60.5``) comes back as ``23:59:59.5``, as with Unix time.

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
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_datetime``."""
        ...

    @deprecated("use to_datetime()")
    def datetime(self, utc: bool = True) -> datetime.datetime:
        """Deprecated: use ``satkit.time.to_datetime``.

        Convert object to "datetime.datetime" object representing same instant in time.

        Args:
            utc (bool, optional): If True (default), return an aware datetime in UTC.
                If False, return a naive datetime in the machine's local time zone,
                which round-trips through ``time.from_datetime``.

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
        """Return a new time offset by a number of UTC calendar days.

        The offset is applied to the UTC clock label (as with a Modified
        Julian Date in UTC), so the time of day is kept: noon plus one UTC
        day is noon of the next day, which is 86401 elapsed seconds later
        when a leap second lies between the two. ``days`` may be fractional
        (a fraction of 86400 s on the label) or negative.

        To add exactly 86400 elapsed seconds per day instead, use
        ``t + days`` or ``t + satkit.duration(days=days)``.

        Args:
            days (float): Number of UTC days to add; rounded to the nearest
                microsecond

        Returns:
            satkit.time: Time object offset by the given number of UTC days

        Raises:
            ValueError: If ``days`` is NaN or infinite
            OverflowError: If the result is beyond about ±292,000 years

        Example:
            ```python
            t = satkit.time(2016, 12, 31, 12, 0, 0)  # a leap second follows
            print(t.add_utc_days(1))
            # 2017-01-01T12:00:00.000000Z
            print(t + 1)
            # 2017-01-01T11:59:59.000000Z
            ```
        """
        ...

    def to_mjd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as a Modified Julian Date
        with the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC

        Returns:
            float: Modified Julian Date, days

        Raises:
            ValueError: If ``scale`` is ``timescale.Invalid``
        """
        ...

    @deprecated("use to_mjd()")
    def as_mjd(self, scale: timescale = timescale.UTC) -> float:
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_mjd``."""
        ...

    def to_jd(self, scale: timescale = timescale.UTC) -> float:
        """
        Represent time instance as Julian Date with
        the provided time scale

        If no time scale is provided, default is satkit.timescale.UTC

        Returns:
            float: Julian Date, days

        Raises:
            ValueError: If ``scale`` is ``timescale.Invalid``
        """
        ...

    @deprecated("use to_jd()")
    def as_jd(self, scale: timescale = timescale.UTC) -> float:
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_jd``."""
        ...

    def to_unixtime(self) -> float:
        """
        Represent time as unixtime

        (seconds since Jan 1, 1970 UTC, excluding leap seconds)

        Includes fractional component of seconds. The result is the float
        nearest the exact (whole-microsecond) value, so it prints as the
        decimal it was built from: ``time.from_unixtime(1678014225.453)``
        gives back ``1678014225.453``. Inside a leap second it repeats the
        last second of the day.

        Returns:
            float: Unix time, seconds
        """
        ...

    @deprecated("use to_unixtime()")
    def as_unixtime(self) -> float:
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_unixtime``."""
        ...

    def to_iso8601(self) -> str:
        """
        Represent time as ISO 8601 string

        The fraction always has six digits (microseconds). Years outside
        0000-9999 are written with a sign and at least four digits, as in
        ISO 8601 expanded years: ``-0044-03-15T...``, ``+10000-01-01T...``.
        ``to_iso8601`` and ``to_rfc3339`` return the same string.

        Returns:
            ISO 8601 string representation of time: "YYYY-MM-DDTHH:MM:SS.ssssssZ"
        """
        ...

    @deprecated("use to_iso8601()")
    def as_iso8601(self) -> str:
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_iso8601``."""
        ...

    def to_rfc3339(self) -> str:
        """
        Represent time as RFC 3339 string

        The fraction always has six digits (microseconds). Years outside
        0000-9999 are written with a sign and at least four digits, as in
        ISO 8601 expanded years: ``-0044-03-15T...``, ``+10000-01-01T...``.
        ``to_iso8601`` and ``to_rfc3339`` return the same string.

        Returns:
            RFC 3339 string representation of time: "YYYY-MM-DDTHH:MM:SS.ssssssZ"
        """
        ...

    @deprecated("use to_rfc3339()")
    def as_rfc3339(self) -> str:
        """Deprecated since 0.23, removed in 0.25. Use ``time.to_rfc3339``."""
        ...

    def strftime(self, format: str) -> str:
        """
        Represent time as string with given format

        Args:
            format (str): format of the string

        Notes:
            The time is written in UTC. Format codes (a subset of
            Python's; there is no ``%z``, ``%j``, ``%y`` or ``%p``):

            - %Y - year, four digits (0000-9999); outside that range a sign
              and at least four digits (``-0044``, ``+10000``)
            - %m - month with leading zeros (01-12)
            - %d - day of month with leading zeros (01-31)
            - %H - hour with leading zeros (00-23)
            - %M - minute with leading zeros (00-59)
            - %S - second with leading zeros (00-59, or 60 inside a leap second)
            - %f - microsecond, always six digits (000000-999999)
            - %b - abbreviated month name (Jan, Feb, ...)
            - %B - full month name (January, February, ...)
            - %a - abbreviated weekday name (Sun, Mon, ...)
            - %A - full weekday name (Sunday, Monday, ...)
            - %w - weekday as a decimal number (0=Sunday, 1=Monday, ...)
            - %% - a literal ``%``

        Returns:
            string representation of time

        Raises:
            ValueError: If the format has a code not listed above, or ends
                with a lone ``%``

        Example:
            ```python
            print(satkit.time(2023, 6, 3, 6, 19, 34).strftime("%Y-%m-%d %H:%M:%S"))
            # 2023-06-03 06:19:34
            ```
        """
        ...

    @typing.overload
    def __add__(self, other: duration | _TimeDelta) -> time:
        """
        Return a time object representing the input duration added to the current time

        Args:
            other (duration | datetime.timedelta): duration to add to the
                current time (a ``timedelta`` is exact; ``timedelta + time``
                works too)

        Raises:
            OverflowError: if the result is beyond about ±292,000 years

        Returns:
            Time object representing the input duration added to the current time

        """
        ...

    @typing.overload
    def __add__(self, other: float) -> time:
        """
        Return a time object representing the input number of days added to the current time

        Each day is exactly 86400 elapsed seconds, so across a leap second
        the UTC time of day shifts by one second; ``add_utc_days`` keeps the
        UTC time of day instead.

        Args:
            other (float): number of days to add to the current time

        Raises:
            ValueError: if ``other`` is NaN or infinite
            OverflowError: if ``other`` is too large for a duration, or the
                result is beyond about ±292,000 years

        Returns:
            Time object representing the input number of days added to the current time

        """
        ...

    @typing.overload
    def __add__(self, other: list[duration]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise addition of days to the "self" time object

        Args:
            other (list[duration]): list of durations to add to the current time (a list, not any sequence: a tuple is refused)

        Raises:
            OverflowError: if a result is beyond about ±292,000 years

        Returns:
            Object array of time objects, ``self + other[i]`` for each element
        """
        ...

    @typing.overload
    def __add__(
        self,
        other: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | list[float] | list[int],
    ) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise addition of days to the "self" time object

        Args:
            other (npt.NDArray[np.floating] | npt.NDArray[np.integer] | list[float]): 1-D array or list of days to add to the current time (integer and float32 arrays are converted to float64)

        Raises:
            ValueError: if an element is NaN or infinite
            OverflowError: if an element is too large for a duration, or a
                result is beyond about ±292,000 years

        Returns:
            Object array of time objects representing the element-wise addition of days to the current time

        """
        ...

    def __radd__(self, other: _TimeDelta) -> time:
        """
        ``datetime.timedelta + time``: the time ``other`` later

        Args:
            other (datetime.timedelta): time difference to add

        Raises:
            OverflowError: if the result is beyond about ±292,000 years

        Returns:
            Time object representing the current time plus the timedelta
        """
        ...

    def __le__(self, other: time) -> bool:
        """
        Compare two time objects for less than or equal to

        Only another ``satkit.time`` can be compared: ordering against any
        other type raises ``TypeError`` (``==`` is False, ``!=`` True).

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
    def __sub__(self, other: duration | _TimeDelta) -> time:
        """
        Return a time object representing the input duration subtracted from the current time

        Args:
            other (duration | datetime.timedelta): duration to subtract from
                the current time (a ``timedelta`` is exact)

        Raises:
            OverflowError: if the result is beyond about ±292,000 years

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

        Raises:
            OverflowError: if the difference is beyond about ±292,000 years
                (only for times near the two ends of the range)

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
            OverflowError: if ``other`` is too large for a duration, or the
                result is beyond about ±292,000 years

        Returns:
            Time object representing the input number of days subtracted from the current time

        """
        ...

    @typing.overload
    def __sub__(
        self,
        other: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | list[float] | list[int],
    ) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of days from the "self" time object

        Args:
            other (npt.NDArray[np.floating] | npt.NDArray[np.integer] | list[float]): 1-D array or list of days to subtract from the current time (integer and float32 arrays are converted to float64)

        Raises:
            ValueError: if an element is NaN or infinite
            OverflowError: if an element is too large for a duration, or a
                result is beyond about ±292,000 years

        Returns:
            Object array of time objects representing the element-wise subtraction of days from the current time

        """
        ...

    @typing.overload
    def __sub__(self, other: list[duration]) -> npt.NDArray[Any]:
        """
        Return a numpy array of time objects, with each object representing an element-wise subtraction of duration from the "self" time object

        Args:
            other (list[duration]): list of durations to subtract from the current time (a list, not any sequence: a tuple is refused)

        Raises:
            OverflowError: if a result is beyond about ±292,000 years

        Returns:
            Object array of time objects, ``self - other[i]`` for each element
        """
        ...

    @typing.overload
    def __sub__(self, other: list[time]) -> npt.NDArray[Any]:
        """
        Return a numpy array of duration objects, with each object representing an element-wise subtraction of time from the "self" time object

        Args:
            other (list[time]): list of times to subtract from the current time (a list, not any sequence)

        Raises:
            OverflowError: if a difference is beyond about ±292,000 years

        Returns:
            Object array of duration objects, ``self - other[i]`` for each element
        """
        ...

@final
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
            - Adding or subtracting durations and times raises ``OverflowError``
              when the result is beyond about ±292,000 years (it used to wrap around)
            - ``str()`` of a duration of a second or more lists the non-zero
              days, hours and minutes, then the seconds with six decimals;
              a shorter (or negative) one prints as whole microseconds
              (``500000 usec``)

        Raises:
            TypeError: If an argument is not a real number, or
                ``microseconds`` is not an integer (the ``from_*``
                constructors do the same)
            ValueError: If an argument is NaN or infinite
            OverflowError: If an argument is too large for a duration
                (beyond about ±292,000 years), or the arguments fit one by
                one but not summed

        Example:
            ```python
            print(satkit.duration(days=1, hours=2, minutes=3, seconds=4.5))
            # 1 days 2 hours 3 minutes 4.500000 seconds
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
            seconds (float): Number of seconds

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
            minutes (float): Number of minutes

        Returns:
            Duration object representing input number of minutes
        """
        ...

    @staticmethod
    def from_hours(hours: float) -> duration:
        """Create duration object representing input number of hours

        Args:
            hours (float): Number of hours

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

    @staticmethod
    def from_timedelta(td: datetime.timedelta) -> duration:
        """Create duration object from a ``datetime.timedelta``

        Exact: both count whole microseconds.

        Args:
            td (datetime.timedelta): The time difference

        Returns:
            Duration object representing the same interval

        Raises:
            OverflowError: If ``td`` is beyond the range of a duration (about
                ±292,000 years; a ``timedelta`` reaches ±2.7 million)

        Example:
            ```python
            import datetime
            d = satkit.duration.from_timedelta(datetime.timedelta(hours=1, microseconds=1))
            print(d.microseconds)
            # 3600000001
            ```
        """
        ...

    def to_timedelta(self) -> datetime.timedelta:
        """Convert to a ``datetime.timedelta``

        Exact: both count whole microseconds, and every duration fits.

        Returns:
            ``datetime.timedelta`` representing the same interval

        Example:
            ```python
            print(satkit.duration(hours=1, seconds=0.5).to_timedelta())
            # 1:00:00.500000
            ```
        """
        ...

    @typing.overload
    def __add__(self, other: duration) -> duration:
        """Add a duration to another duration

        Args:
            other (duration): duration to add to the current duration

        Raises:
            OverflowError: if the result is beyond about ±292,000 years

        Returns:
            Duration object representing the sum, or concatenation, of both durations

        Example:
            ```python
            print(satkit.duration.from_hours(1) + satkit.duration.from_minutes(1))
            # 1 hours 1 minutes 0.000000 seconds
            ```
        """
        ...

    @typing.overload
    def __add__(self, other: time) -> time:
        """Add a duration to a time

        Args:
            other (time): time to add the current duration to

        Raises:
            OverflowError: if the result is beyond about ±292,000 years

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

        Raises:
            OverflowError: if the result is beyond about ±292,000 years

        Returns:
            Duration object representing the difference between the two durations

        Example:
            ```python
            print(satkit.duration.from_hours(1) - satkit.duration.from_minutes(1))
            # 59 minutes 0.000000 seconds
            ```
        """
        ...

    def __mul__(self, other: float) -> duration:
        """Multiply (or scale) duration by given value

        The product is computed exactly from the duration's integer
        microseconds and rounded once to the nearest microsecond, so an
        integer factor is exact for any duration.

        Args:
            other (float): value by which to multiply duration (any real
                number: ``int``, ``float`` or a numpy scalar)

        Returns:
            Duration object representing the input duration scaled by the input value

        Raises:
            ValueError: if ``other`` is NaN or infinite
            OverflowError: if the result is beyond about ±292,000 years

        Example:
            ```python
            print(satkit.duration.from_days(1) * 2.5)
            # 2 days 12 hours 0.000000 seconds
            ```
        """
        ...

    def __rmul__(self, other: float) -> duration:
        """Multiply (or scale) duration by given value: ``2 * d`` is ``d * 2``

        Args:
            other (float): value by which to multiply duration

        Returns:
            Duration object representing the input duration scaled by the input value

        Raises:
            ValueError: if ``other`` is NaN or infinite
            OverflowError: if the result is beyond about ±292,000 years
        """
        ...

    def __neg__(self) -> duration:
        """The duration with its sign reversed

        Raises:
            OverflowError: for the most negative duration (-2**63 microseconds)

        Example:
            ```python
            print((-satkit.duration(hours=1)).hours)
            # -1.0
            ```
        """
        ...

    def __pos__(self) -> duration:
        """The duration itself"""
        ...

    def __abs__(self) -> duration:
        """The duration without its sign

        Raises:
            OverflowError: for the most negative duration (-2**63 microseconds)
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
            OverflowError: if the result is beyond about ±292,000 years
                (division by a very small number)

        Notes:
            The quotient is computed exactly from the duration's integer
            microseconds and rounded once to the nearest microsecond.
            Dividing by infinity gives a zero duration.

        Example:
            ```python
            print(satkit.duration.from_days(1) / 2)
            # 12 hours 0.000000 seconds
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

@final
class quaternion:
    """Quaternion representing rotation of 3D Cartesian axes

    Quaternions perform right-handed rotation of a vector, e.g. rotation of +xhat 90 degrees by +zhat give +yhat

    This is different than the convention used in Vallado, but it is the way it is commonly used in mathematics and it is the way it should be done.

    For the uninitiated: quaternions are a more-compact and
    computationally efficient way of representing 3D rotations.
    They can also be multiplied together and easily renormalized to
    avoid problems with floating-point precision eventually causing
    changes in the rotated vector norm.

    For details, see:

    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>

    Notes:
        A quaternion is always a unit quaternion (a rotation): every
        constructor normalizes its result, so ``norm`` is 1 to rounding.
        ``q`` and ``-q`` are the same rotation but are different quaternions.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, w: float, x: float, y: float, z: float, /) -> None:
        """Return the unit quaternion along the input (w,x,y,z) values

        With no arguments, return the identity quaternion. Otherwise pass all
        four components positionally (keywords are not accepted); they are
        scaled to unit norm, so ``quaternion(2, 0, 0, 0)`` is the identity.

        Args:
            w: Scalar component of the quaternion
            x: X component of the quaternion
            y: Y component of the quaternion
            z: Z component of the quaternion

        Raises:
            TypeError: the number of arguments is neither 0 nor 4, or a
                component is not a real number
            ValueError: all four components are zero, or one is NaN or
                infinite

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
            axis (npt.ArrayLike): 3-element array representing axis of rotation (unitless direction; need not be normalized)
            angle (float): angle of rotation in radians

        Returns:
            Quaternion representing rotation by "angle" radians about the given axis.
            If the axis has (near-)zero length (norm < 1e-9) the identity is
            returned, whatever the angle

        Raises:
            TypeError: ``axis`` is not a real numeric array-like
            ValueError: ``axis`` does not have 3 elements, or ``axis`` or
                ``angle`` holds a NaN or infinity
        """
        ...

    @staticmethod
    def from_rotation_matrix(
        dcm: npt.ArrayLike,
    ) -> quaternion:
        """Return quaternion representing identical rotation to input 3x3 rotation matrix

        Args:
            dcm (npt.ArrayLike): 3x3 rotation matrix (any real numeric
                array-like): orthonormal, with determinant +1

        Returns:
            Unit quaternion representing identical rotation to input 3x3 rotation matrix

        Raises:
            TypeError: ``dcm`` is not a real numeric array-like
            ValueError: ``dcm`` is not 3x3, holds a NaN or infinity, or is not
                a rotation matrix (``dcm @ dcm.T`` differs from the identity
                by more than 1e-6 in any element, or the determinant is
                negative)
        """
        ...

    @staticmethod
    def rotx(theta_rad: float) -> quaternion:
        """Quaternion representing right-handed rotation of vector by "theta" radians about the xhat unit vector

        Args:
            theta_rad (float): angle of rotation in radians

        Returns:
            Quaternion representing right-handed rotation of vector by "theta" radians about the xhat unit vector

        Raises:
            ValueError: ``theta_rad`` is NaN or infinite

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
            theta_rad (float): angle of rotation in radians

        Returns:
            Quaternion representing right-handed rotation of vector by "theta" radians about the yhat unit vector

        Raises:
            ValueError: ``theta_rad`` is NaN or infinite


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
            theta_rad (float): angle of rotation in radians

        Returns:
            Quaternion representing right-handed rotation of vector by "theta" radians about the zhat unit vector

        Raises:
            ValueError: ``theta_rad`` is NaN or infinite

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
            Quaternion of the shortest rotation from v1 to v2 (about the axis
            ``v1 x v2``; the vectors need not be unit length)

        Raises:
            TypeError: an input is not a real numeric array-like
            ValueError: an input does not have 3 elements or holds a NaN or
                infinity, either vector has (near-)zero length, or the
                vectors point in opposite directions (the rotation axis is
                then not unique)

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
        """Deprecated since 0.23, removed in 0.25. Use ``quaternion.to_rotation_matrix``."""
        ...

    def to_euler(self) -> tuple[float, float, float]:
        """Return equivalent rotation as intrinsic ZYX Euler angles, returned as (roll, pitch, yaw).

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
        """Deprecated since 0.23, removed in 0.25. Use ``quaternion.to_euler``."""
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

        Raises:
            ValueError: an angle is NaN or infinite
        """
        ...

    @staticmethod
    def identity() -> quaternion:
        """The identity (no-rotation) quaternion (w=1, x=y=z=0)"""
        ...

    @property
    def norm(self) -> float:
        """Quaternion norm (Euclidean length of the 4 components).
        Quaternions are kept unit, so this is 1 to rounding."""
        ...

    def normalize(self) -> quaternion:
        """Return this quaternion normalized to unit length

        Quaternions are already unit; this only removes rounding error
        accumulated over a long chain of products.
        """
        ...

    def inverse(self) -> quaternion:
        """Quaternion inverse: the opposite rotation. Same as ``conj()`` and
        ``conjugate()``, quaternions being unit."""
        ...

    def dot(self, other: quaternion) -> float:
        """Dot product of the 4 quaternion components with another quaternion"""
        ...

    @property
    def angle(self) -> float:
        """Return the angle in radians of the rotation, in [0, 2 pi)

        The angle is measured about ``axis``, right-handed. It exceeds pi
        when the scalar component ``w`` is negative; the same rotation is
        then the angle ``2 pi - angle`` about ``-axis``.

        Returns:
            Angle in radians of the rotation
        """
        ...

    @property
    def axis(self) -> npt.NDArray[np.float64]:
        """Return the axis of rotation as a unit vector

        The identity (no rotation) has no axis; ``[1, 0, 0]`` is returned
        for it.

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
        """X component of the vector part"""
        ...

    @property
    def y(self) -> float:
        """Y component of the vector part"""
        ...

    @property
    def z(self) -> float:
        """Z component of the vector part"""
        ...

    @property
    def w(self) -> float:
        """Scalar component (cosine of half the rotation angle)"""
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
    def __mul__(
        self, other: npt.NDArray[Any] | Sequence[float] | Sequence[Sequence[float]]
    ) -> npt.NDArray[np.float64]:
        """Multiply by a vector to rotate the vector

        Args:
            other (npt.NDArray | Sequence[float] | Sequence[Sequence[float]]): 3-element vector to rotate, or Nx3 array of vectors to rotate (any real numeric array, list or tuple; integers are converted to float64). A scalar is not a vector: ``q * 2.0`` raises ``TypeError``

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
            frac (float): fractional distance from self (0) to other (1).
                It is not clamped: a value outside [0, 1] extrapolates along
                the same arc

        Returns:
            Quaternion representing interpolation between self and other,
            along the shorter of the two arcs (``other`` and ``-other`` are
            the same rotation; the nearer one is used)

        Raises:
            ValueError: ``frac`` is NaN or infinite
        """
        ...

    def __eq__(self, other: object) -> bool:
        """True when ``other`` is a quaternion with exactly the same four
        components (no tolerance)

        ``q`` and ``-q`` are the same rotation but compare unequal; so do two
        quaternions that differ by rounding. To compare rotations use
        ``q1.isclose(q2)`` or ``q1.angle_to(q2)``.
        """
        ...

    def angle_to(self, other: quaternion) -> float:
        """Rotation angle between this orientation and another, radians in [0, pi]

        The angle of the rotation ``self.conj() * other`` that takes one to
        the other. ``q`` and ``-q`` are the same rotation, so
        ``q.angle_to(-q)`` is 0.

        Args:
            other (quaternion): The other orientation

        Returns:
            float: Angle between the two rotations, radians in [0, pi]

        Example:
            ```python
            import math
            q1 = satkit.quaternion.rotz(math.radians(10))
            q2 = satkit.quaternion.rotz(math.radians(40))
            print(math.degrees(q1.angle_to(q2)))
            # 30.0 (to rounding)
            ```
        """
        ...

    def isclose(self, other: quaternion, *, atol: float = 1e-12) -> bool:
        """True when ``other`` is the same rotation to within ``atol`` radians

        Compares rotations, not components: ``q`` and ``-q`` are the same
        rotation and are close (unlike ``==``, which compares the four
        components exactly).

        Args:
            other (quaternion): The rotation to compare with
            atol (float): Largest ``angle_to(other)``, radians, still counted
                as the same rotation. Default 1e-12

        Returns:
            bool: ``self.angle_to(other) <= atol``

        Raises:
            ValueError: ``atol`` is negative, NaN or infinite

        Example:
            ```python
            q = satkit.quaternion.rotx(0.3)
            print(q.isclose(satkit.quaternion(-q.w, -q.x, -q.y, -q.z)))
            # True
            ```
        """
        ...

    def to_axis_angle(self) -> tuple[npt.NDArray[np.float64], float]:
        """Rotation axis and angle, the inverse of ``from_axis_angle``

        The same values as the ``axis`` and ``angle`` properties:
        ``quaternion.from_axis_angle(*q.to_axis_angle())`` is ``q``.

        Returns:
            tuple[numpy.ndarray, float]: Unit rotation axis (3-element array;
            ``[1, 0, 0]`` for the identity) and the right-handed angle about
            it in radians, in [0, 2 pi)

        Example:
            ```python
            axis, angle = satkit.quaternion.rotz(0.5).to_axis_angle()
            ```
        """
        ...

    def to_array(self) -> npt.NDArray[np.float64]:
        """The four components as an array in the order ``[w, x, y, z]``
        (scalar first), the inverse of ``from_array``

        Returns:
            numpy.ndarray: 4-element float64 array ``[w, x, y, z]``

        Example:
            ```python
            print(satkit.quaternion().to_array())
            # [1. 0. 0. 0.]
            ```
        """
        ...

    @staticmethod
    def from_array(arr: npt.ArrayLike) -> quaternion:
        """Quaternion from an array of its four components in the order
        ``[w, x, y, z]`` (scalar first), the inverse of ``to_array``

        The same as ``quaternion(w, x, y, z)``: the components are scaled to
        unit norm.

        Args:
            arr (npt.ArrayLike): 4 components ``[w, x, y, z]`` (any real
                numeric array-like: numpy array, list or tuple)

        Returns:
            quaternion: Unit quaternion along ``arr``

        Raises:
            TypeError: ``arr`` is not a real numeric array-like
            ValueError: ``arr`` does not have 4 elements, holds a NaN or
                infinity, or is all zeros

        Example:
            ```python
            q = satkit.quaternion.from_array([1.0, 0.0, 0.0, 0.0])
            ```
        """
        ...

    def __hash__(self) -> int:
        """Hash of the four components, consistent with ``==``

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

@final
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

    def propagate(self, dt: duration | datetime.timedelta | float | int) -> kepler:
        """Propagate Keplerian element set by input duration

        Two-body (unperturbed) propagation: only the anomaly changes.

        Args:
            dt (duration | datetime.timedelta | float | int): Duration by
                which to propagate the Keplerian element set. A number is
                interpreted as seconds.

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
            TypeError: if ``dt`` is neither a duration, a timedelta nor a
                number.
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
        """True anomaly, radians: the same attribute as ``nu``

        Stored as given, not reduced to one revolution. Assigning it sets
        ``nu`` (finite; ``ValueError`` otherwise).
        """
        ...

    @true_anomaly.setter
    def true_anomaly(self, value: float) -> None: ...

    @property
    def eccentric_anomaly(self) -> float:
        """Eccentric anomaly, radians in (-pi, pi]

        Computed from the true anomaly and reduced to (-pi, pi], however
        many revolutions ``nu`` holds.
        """
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
        """Mean anomaly, radians in (-pi, pi]

        Computed from the true anomaly and reduced to (-pi, pi], however
        many revolutions ``nu`` holds.
        """
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
        """Eccentricity, unitless (0 <= eccen < 1; ``ValueError`` otherwise)

        Short alias of ``eccentricity``.
        """
        ...

    @eccen.setter
    def eccen(self, value: float) -> None: ...
    @property
    def eccentricity(self) -> float:
        """Eccentricity, unitless (0 <= eccentricity < 1; ``ValueError``
        otherwise). ``eccen`` is the same attribute under its short name."""
        ...

    @eccentricity.setter
    def eccentricity(self, value: float) -> None: ...
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
        """True anomaly, radians (finite; ``ValueError`` otherwise)

        Stored as given, not reduced to one revolution: the constructor and
        this setter keep the value passed, while the anomaly setters and
        ``propagate`` store a value in (-pi, pi] and ``from_pv`` one in
        [0, 2 pi).
        """
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
    def __eq__(self, other: object) -> bool:
        """True when ``other`` is a ``kepler`` with exactly the same six
        elements and ``mu`` (no tolerance, no reduction of angles)"""
        ...

    __hash__: ClassVar[None]  # type: ignore[assignment]

    @staticmethod
    def from_pv(
        pos: npt.ArrayLike,
        vel: npt.ArrayLike,
        *,
        mu: float | None = None,
    ) -> kepler:
        """Create Keplerian element set from input position and velocity vectors

        Args:
            pos: 3-element position vector, meters (any real numeric
                array-like: numpy array, list or tuple)
            vel: 3-element velocity vector, meters/second (likewise)
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
                holds a NaN or infinity or is not a 3-element vector, or if
                ``mu`` is not positive and finite.
            TypeError: if ``pos`` or ``vel`` is not a real numeric array-like.
        """
        ...

@final
class geodetic:
    """Geodetic coordinates with named fields

    Returned by ``itrfcoord.geodetic``; it has no constructor and its fields
    are read-only.

    Attributes:
        latitude_rad (float): Latitude in radians
        longitude_rad (float): Longitude in radians
        height_m (float): Height above WGS84 ellipsoid in meters
        latitude_deg (float): Latitude in degrees (computed)
        longitude_deg (float): Longitude in degrees (computed)
    """

    @property
    def latitude_rad(self) -> float:
        """Latitude in radians"""
        ...

    @property
    def longitude_rad(self) -> float:
        """Longitude in radians"""
        ...

    @property
    def height_m(self) -> float:
        """Height above WGS84 ellipsoid in meters"""
        ...

    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees"""
        ...

    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees"""
        ...

    def __eq__(self, other: object) -> bool:
        """True when ``other`` is a geodetic with exactly the same latitude,
        longitude and height (no tolerance)"""
        ...

    def __hash__(self) -> int:
        """Hash of the three fields, consistent with ``==``"""
        ...

@final
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
    def __init__(self, vec: npt.ArrayLike, /) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float, /) -> None: ...
    @overload
    def __init__(
        self,
        *,
        latitude_deg: float,
        longitude_deg: float,
        altitude: float | None = None,
        height: float | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        latitude_rad: float,
        longitude_rad: float,
        altitude: float | None = None,
        height: float | None = None,
    ) -> None:
        """Create ITRF coordinate from Cartesian vector or geodetic parameters.

        Pass the Cartesian position positionally, either as one 3-element
        array-like or as three floats; or pass the geodetic position as
        keywords: a latitude and a longitude, both required, each in degrees
        or in radians, and optionally a height. The two forms cannot be mixed.

        Args:
            vec: ITRF Cartesian location in meters (any real numeric
                3-element array-like: numpy array, list or tuple)
            latitude_deg: Geodetic latitude in degrees, in [-90, 90]
            longitude_deg: Longitude in degrees
            latitude_rad: Geodetic latitude in radians, in [-pi/2, pi/2]
            longitude_rad: Longitude in radians
            altitude: Height above ellipsoid, meters. Default is 0
            height: Height above ellipsoid, meters (alias of ``altitude``)

        Raises:
            TypeError: no position given; positional and keyword forms
                mixed; the same quantity given twice (``latitude_deg`` with
                ``latitude_rad``, ``longitude_deg`` with ``longitude_rad``,
                ``altitude`` with ``height``); latitude or longitude missing;
                the positional argument is not a real numeric array-like or
                three numbers; a keyword that is not a real number
            ValueError: latitude outside [-90, 90] degrees; a NaN or infinite
                latitude, longitude, height or Cartesian component; a
                positional array that does not have 3 elements

        Notes:
            The named constructors ``itrfcoord.from_vector`` and
            ``itrfcoord.from_geodetic`` do the same.
        """
        ...

    @overload
    @staticmethod
    def from_geodetic(
        *,
        latitude_deg: float,
        longitude_deg: float,
        height: float | None = None,
    ) -> itrfcoord: ...
    @overload
    @staticmethod
    def from_geodetic(
        *,
        latitude_rad: float,
        longitude_rad: float,
        height: float | None = None,
    ) -> itrfcoord:
        """Create an ITRF coordinate from a geodetic position

        The named-constructor form of ``itrfcoord(latitude_deg=...,
        longitude_deg=..., altitude=...)``, with the same checks. Give a
        latitude and a longitude, each in degrees or in radians, and
        optionally the height; all arguments are keyword-only.

        Args:
            latitude_deg: Geodetic latitude in degrees, in [-90, 90]
            longitude_deg: Longitude in degrees
            latitude_rad: Geodetic latitude in radians, in [-pi/2, pi/2]
            longitude_rad: Longitude in radians
            height: Height above the WGS84 ellipsoid, meters. Default is 0

        Returns:
            itrfcoord: New ITRF coordinate

        Raises:
            TypeError: latitude or longitude missing; the same quantity given
                in both units (``latitude_deg`` with ``latitude_rad``,
                ``longitude_deg`` with ``longitude_rad``); an argument that
                is not a real number
            ValueError: latitude outside [-90, 90] degrees; a NaN or infinite
                latitude, longitude or height

        Example:
            ```python
            coord = satkit.itrfcoord.from_geodetic(latitude_deg=42.44, longitude_deg=-71.15, height=100)
            print(coord)
            # ITRFCoord(lat:  42.4400 deg, lon: -71.1500 deg, hae:  0.10 km)
            ```
        """
        ...

    @staticmethod
    def from_vector(vec: npt.ArrayLike) -> itrfcoord:
        """Create an ITRF coordinate from a Cartesian position

        The named-constructor form of ``itrfcoord(vec)``; the inverse of the
        ``vector`` property.

        Args:
            vec: ITRF Cartesian position in meters: any real numeric
                3-element array-like (numpy array, list or tuple)

        Returns:
            itrfcoord: New ITRF coordinate

        Raises:
            TypeError: ``vec`` is not a real numeric array-like
            ValueError: ``vec`` does not have 3 elements, or holds a NaN or
                infinity

        Example:
            ```python
            coord = satkit.itrfcoord.from_vector([1523128.63570828, -4461395.28873207, 4281865.94218203])
            print(coord.vector)
            ```
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

    def __eq__(self, other: object) -> bool:
        """True when ``other`` is an ``itrfcoord`` with exactly the same
        Cartesian components (no tolerance)"""
        ...

    __hash__: ClassVar[None]  # type: ignore[assignment]

    def geodesic_distance(self, other: itrfcoord) -> tuple[float, float, float]:
        """Geodesic (shortest path on the WGS-84 ellipsoid) from this
        coordinate to another: distance and headings

        Args:
            other (itrfcoord): ITRF coordinate to measure to

        Returns:
            (distance in meters, initial heading in radians, heading at destination in radians)

        Notes:
            - Both points are projected onto the ellipsoid: heights are ignored
            - Vincenty's inverse formula, iterated to convergence
              (T. Vincenty, Survey Review 23(176), 1975, <https://doi.org/10.1179/sre.1975.23.176.88>).
              For nearly antipodal points, where that iteration does not
              converge, the same series are solved for the initial heading
              by bisection. Accurate to better than a millimeter for any
              pair of points
            - Headings are clockwise from north, in (-pi, pi]. The heading
              at the destination is the direction of travel on arrival (add
              pi for the back azimuth)
            - For coincident points all three values are 0. Between exactly
              antipodal points, or from a pole, the shortest path is not
              unique: the distance is well defined, the headings are those
              of one of the shortest paths

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
        distance (shortest distance along the WGS-84 ellipsoid; heights are
        ignored).

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
            The path follows the ellipsoid: the starting height is ignored
            and the result has zero height

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

@final
class consts:
    """Some constants that are useful for satellite dynamics

    The class is a namespace: it cannot be instantiated and its attributes
    cannot be reassigned.
    """

    wgs84_a: Final[float]
    """WGS-84 semi-major axis (equatorial radius), in meters"""

    wgs84_f: Final[float]
    """WGS-84 flattening, unitless"""

    earth_radius: Final[float]
    """WGS-84 equatorial Earth radius, meters (same value as ``wgs84_a``)"""

    mu_earth: Final[float]
    """Gravitational parameter of Earth, m^3/s^2 (WGS-84)"""

    mu_moon: Final[float]
    """Gravitational parameter of Moon, m^3/s^2 (JPL DE440)"""

    mu_sun: Final[float]
    """Gravitational parameter of sun, m^3/s^2 (JPL DE440)"""

    GM: Final[float]
    """Gravitational parameter of Earth, m^3/s^2 (same value as ``mu_earth``)"""

    omega_earth: Final[float]
    """Scalar Earth rotation rate, rad/s"""

    c: Final[float]
    """Speed of light, m/s"""

    au: Final[float]
    """Astronomical unit, meters: the exact defined constant of IAU 2012
    Resolution B1 (149 597 870 700 m), close to the mean Earth-Sun distance"""

    sun_radius: Final[float]
    """Radius of sun, meters"""

    solar_pressure_1au: Final[float]
    """Solar radiation pressure at 1 AU (1367 W/m^2 / c), N/m^2.

    The cannonball SRP force scales this by (AU / d)^2, with d the
    satellite-Sun distance. The measured solar constant is 1361 W/m^2
    (Kopp & Lean 2011; IAU 2015 Resolution B3); 1367 matches GMAT and STK.
    """

    moon_radius: Final[float]
    """Radius of moon, meters"""

    earth_moon_mass_ratio: Final[float]
    """Earth mass over Moon mass, unitless

    The ratio of the JPL DE440 gravitational parameters. It differs from
    ``mu_earth / mu_moon`` in the eighth digit, because ``mu_earth`` is the
    WGS-84 value.
    """

    geo_r: Final[float]
    """Radius of a circular geosynchronous orbit (its semi-major axis),
    meters from the Earth's center: ``(mu_earth / omega_earth^2)^(1/3)``"""

    jgm3_mu: Final[float]
    """Earth gravitational parameter from JGM3 gravity model, m^3/s^2"""

    jgm3_a: Final[float]
    """Earth equatorial radius of the JGM3 gravity model, m"""

    jgm3_j2: Final[float]
    """ "J2" gravity due oblateness of Earth from JGM3 gravity model, unitless"""

# Alias so `time` resolves to the class inside bodies that define a `time` member
_Time = time
# Likewise for `frame`
_Frame = frame

class _ManeuverDict(typing.TypedDict):
    """One entry of ``satstate.maneuvers``: an impulsive maneuver as a plain dict

    A stub-only type (it does not exist at runtime, where the entries are
    ordinary ``dict`` objects).
    """

    time: _Time
    """Time at which the delta-v is applied"""
    delta_v: npt.NDArray[np.float64]
    """3-element delta-v, m/s, along the axes of ``frame``"""
    frame: _Frame
    """Frame of ``delta_v``: ``frame.GCRF``, ``frame.RTN``, ``frame.NTW`` or ``frame.LVLH``"""

@final
class satstate:
    """Satellite state: position, velocity, optional covariance, and maneuvers

    Bundles a GCRF position/velocity with optional 6x6 covariance and a list
    of impulsive maneuvers into a single propagatable object. Use ``satstate``
    instead of the free ``propagate`` function when you need:

    - **Covariance propagation** -- attach uncertainty and it propagates
      automatically via the state transition matrix.
    - **Maneuver scheduling** -- add impulsive delta-v events at future times;
      propagation segments around them automatically.
    - **Round-trip propagation** -- propagate forward then backward, recovering
      the original state (maneuvers are reversed).

    For simple state-vector propagation without covariance or maneuvers,
    ``propagate`` is more direct.

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
        pos: npt.ArrayLike,
        vel: npt.ArrayLike,
        cov: npt.ArrayLike | None = None,
    ):
        """Create a new satellite state

        Args:
            time (satkit.time | datetime.datetime | numpy.datetime64): Epoch of the state
            pos (array-like): 3-element position in meters, GCRF frame
            vel (array-like): 3-element velocity in m/s, GCRF frame
            cov (array-like | None, optional): symmetric 6x6 covariance matrix in GCRF
                (position block m^2, velocity block (m/s)^2, cross blocks m^2/s). Defaults to None.

        Raises:
            TypeError: if ``time`` is not a time, or ``pos``, ``vel`` or ``cov`` is not a
                real numeric array-like
            ValueError: if ``pos`` or ``vel`` does not have 3 elements, or ``cov`` is not
                a symmetric 6x6 matrix

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

        Element ``[i, j]`` is the covariance of elements ``i`` and ``j`` of the
        state ``[x, y, z, vx, vy, vz]``: the upper-left 3x3 block is the position
        covariance (m^2), the lower-right 3x3 the velocity covariance ((m/s)^2),
        and the off-diagonal blocks the position-velocity cross-covariance (m^2/s).

        The getter returns a new array on every access, holding exactly what was
        assigned (or what ``propagate`` computed, ``Phi @ cov @ Phi.T``), so
        modifying the returned array does not change the state: assign a matrix
        to set the covariance, or ``None`` to remove it.

        Raises:
            ValueError: on assignment, if the matrix is not 6x6 or not symmetric
            TypeError: on assignment, if the value is not a real numeric array-like
                or ``None``
        """
        ...

    @cov.setter
    def cov(self, value: npt.ArrayLike | None) -> None: ...

    @property
    def time(self) -> _Time:
        """Epoch of this satellite state"""
        ...

    def set_pos_uncertainty(
        self,
        sigma: npt.ArrayLike,
        frame: frame,
    ) -> None:
        """Set 1-sigma position uncertainty in a satellite-local or inertial frame.

        Constructs a diagonal 3x3 covariance from the given 1-sigma values
        (interpreted along the ``frame``'s axes), rotates it into GCRF,
        and stores it as the position block of the 6x6 state covariance.
        Any existing velocity covariance is preserved; the position-velocity
        cross blocks are cleared.

        Args:
            sigma: 3-element array-like of 1-sigma position components
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
            ValueError: if ``sigma`` does not have 3 elements, or the frame is
                not one of the supported frames.
            TypeError: if ``sigma`` is not a real numeric array-like

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
        sigma: npt.ArrayLike,
        frame: frame,
    ) -> None:
        """Set 1-sigma velocity uncertainty in a satellite-local or inertial frame.

        Analogous to ``set_pos_uncertainty``, but for the velocity
        block of the 6x6 state covariance. Any existing position
        covariance is preserved; the position-velocity cross blocks are
        cleared.

        Args:
            sigma: 3-element array-like of 1-sigma velocity components
                along the frame's axes. Units: m/s.
            frame: Coordinate frame — **required**, no default (matching
                the Rust API). Supported values: ``frame.GCRF``,
                ``frame.RTN``, ``frame.NTW``, ``frame.LVLH``.

        Raises:
            ValueError: if ``sigma`` does not have 3 elements, or the frame is
                not one of the supported frames.
            TypeError: if ``sigma`` is not a real numeric array-like
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
            time (satkit.time | datetime.datetime | numpy.datetime64): Time at which to apply the maneuver
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

        Raises:
            ValueError: if ``delta_v`` does not have 3 elements, or ``frame`` is
                not one of the four above (an Earth-fixed or other inertial
                frame is refused here, not later by ``propagate``)
            TypeError: if ``time`` is not a time, or ``delta_v`` is not numeric

        See Also:
            ``add_prograde``, ``add_retrograde``, ``add_radial``,
            ``add_normal`` for scalar-magnitude helpers that pick the frame
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

    def clear_maneuvers(self) -> None:
        """Remove every scheduled maneuver from this state

        Example:
            ```python
            sat = sk.satstate(sk.time(2024, 1, 1), np.array([6.781e6, 0, 0]), np.array([0, 7.5e3, 0]))
            sat.add_prograde(sat.time + sk.duration(hours=1), 10.0)
            sat.clear_maneuvers()
            assert sat.num_maneuvers == 0
            ```
        """
        ...

    @property
    def num_maneuvers(self) -> int:
        """Number of impulsive maneuvers scheduled on this state"""
        ...

    @property
    def maneuvers(self) -> list[_ManeuverDict]:
        """The scheduled impulsive maneuvers, in the order they were added.

        A new list of new dicts on every access: changing it does not change
        the state. Add maneuvers with the ``add_*`` methods and remove them
        with ``clear_maneuvers``.

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
        automatically segments at each maneuver epoch and applies the delta-v
        (subtracts it when propagating backward). A maneuver at time ``t`` is
        applied when ``start <= t < target`` (``target <= t < start`` backward):
        the state at a maneuver's own time is the state just before the burn, so
        a maneuver at the current time is applied and one exactly at the target
        time is not. Maneuvers are preserved on the returned state.

        Args:
            timedur (satkit.time|datetime.datetime|numpy.datetime64|satkit.duration): Target time, or duration from current time
            propsettings (satkit.propsettings, optional): Propagation settings. Default: ``propsettings()``
            satproperties (satkit.satproperties, optional): Satellite properties (drag, SRP, thrust)

        Returns:
            satstate: New state at the target time

        Raises:
            TypeError: if ``timedur`` is not a time or a ``satkit.duration`` (a
                float or a ``datetime.timedelta`` is refused), or ``propsettings``
                / ``satproperties`` has the wrong type
            ValueError: if the position is all zeros
            RuntimeError: if the propagation fails (see ``satkit.propagate``). A
                covariance needs the state transition matrix, which
                ``integrator.rodas4`` and ``integrator.gauss_jackson8`` cannot
                propagate

        Example:
            ```python
            sat = sk.satstate(time=t0, pos=r, vel=v)
            sat.add_maneuver(t_burn, [0, 100, 0], frame=sk.frame.RTN)
            new_state = sat.propagate(t_end)
            ```
        """
        ...

@final
class propstats:
    """Statistics of a satellite propagation

    The step and force-evaluation counts of the integrator that ran it (an
    adaptive Runge-Kutta pair, RODAS4 or the fixed-step Gauss-Jackson 8).
    Returned by ``propresult.stats``; not constructed directly.
    """

    @property
    def num_eval(self) -> int:
        """Number of force-model (derivative) evaluations"""
        ...

    @property
    def num_accept(self) -> int:
        """Number of accepted integrator steps (for the fixed-step
        ``integrator.gauss_jackson8``, the number of steps taken)"""
        ...

    @property
    def num_reject(self) -> int:
        """Number of steps the adaptive step-size control rejected and retried
        (always 0 for the fixed-step ``integrator.gauss_jackson8``)"""
        ...

    def __repr__(self) -> str:
        """One line: ``propstats(num_eval=48, num_accept=3, num_reject=0)``"""
        ...

    def __str__(self) -> str: ...

@final
class propresult:
    """Results of a satellite propagation

    This class lets the user access results of the satellite propagation.
    It is returned by ``satkit.propagate`` and is not constructed directly.

    Notes:
        Unless ``enable_interp`` is set to False in the propagation settings
        (it is True by default), the propresult object can be used to
        interpolate solutions at any time between the begin and end times of
        the propagation via the ``interp`` method. ``can_interp`` tells
        whether it can: ``integrator.rodas4`` and ``integrator.rkv98_nointerp``
        never store the dense output this needs.
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
        """Time of the end state, at which ``state``, ``pos`` and ``vel`` are valid

        Notes:
        - This is identical to the "time_end" property

        Returns:
            Time at which the end state is valid
        """
        ...

    @property
    def time_end(self) -> _Time:
        """Time of the end state, at which ``state_end`` is valid

        Notes:
        - This is identical to "time" property

        Returns:
            Time at which the end state is valid
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
        """Whether ``interp`` works on this result

        True when the propagation stored dense output. False when
        ``propsettings.enable_interp`` was False, for ``integrator.rodas4``
        and ``integrator.rkv98_nointerp`` (which have no interpolant) and for
        a zero-duration propagation; ``interp`` then raises ``ValueError``.

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
        output_phi: typing.Literal[True],
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
        time: TimeArrayLike,
        output_phi: typing.Literal[False] = False,
    ) -> npt.NDArray[np.float64]:
        """Interpolate state at multiple times

        Args:
            time: List or 1-D array of times at which to interpolate state
            output_phi: Must be False (default)

        Returns:
            npt.NDArray[np.float64]: (N, 6) array, one row [x, y, z, vx, vy, vz] in meters and m/s
                per input time ((0, 6) for an empty list)
        """
        ...

    @typing.overload
    def interp(
        self,
        time: TimeArrayLike,
        output_phi: typing.Literal[True],
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Interpolate state and state transition matrix at multiple times

        Args:
            time: List or 1-D array of times at which to interpolate state
            output_phi: Must be True

        Returns:
            list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]: List of (state, phi) tuples;
                each state is a 6-element vector in meters and m/s, each phi a 6x6 state transition matrix

        Raises:
            ValueError: if the propagation did not compute the state transition matrix
                (propagate with ``output_phi=True``)
        """
        ...

    @typing.overload
    def interp(
        self,
        time: TimeInput,
        output_phi: bool,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    ):
        """Interpolate the GCRF state at one or more times between the begin and end times

        Args:
            time: time(s) at which to interpolate: a single time, or a list or
                1-D array of times
            output_phi: also return the 6x6 state transition matrix. Default False

        Returns:
            npt.NDArray[np.float64] | tuple | list[tuple]: 6-element state
                [x, y, z, vx, vy, vz] in meters and m/s for a single time; for a
                list or array of N times, one (N, 6) array ((0, 6) for an empty
                list). With ``output_phi=True``, a (state, phi) tuple, where phi
                is the 6x6 state transition matrix, or a list of them for a list
                of times

        Raises:
            ValueError: if the result has no dense output (``can_interp`` is
                False), a time is outside the begin-to-end range, or
                ``output_phi`` is True but the propagation did not compute the
                state transition matrix (propagate with ``output_phi=True``)
            TypeError: if ``time`` is not a time or a list / array of times (a
                ``RuntimeError`` for a tuple, a number or a string)
        """
        ...

    def __repr__(self) -> str:
        """One line with the begin and end times, the end position (m) and
        velocity (m/s), whether there is a state transition matrix, and
        ``can_interp``; ``str()`` gives the multi-line summary"""
        ...

    def __str__(self) -> str: ...

@final
class thrust:
    """Continuous thrust acceleration for orbit maneuvers

    Represents a constant thrust acceleration over a time window,
    specified in GCRF (inertial), RTN (CCSDS-standard orbital frame,
    also known as RSW or RIC), NTW (velocity-aligned), or LVLH. The
    thrust is on from ``start`` to ``end``, both inclusive. Build one
    with ``thrust.constant`` and pass a list of them as
    ``satproperties(thrusts=[...])``; a thrust object is immutable.

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
        start: TimeScalar,
        end: TimeScalar,
        frame: frame,
    ) -> thrust:
        """Create a constant thrust acceleration

        Args:
            accel (array-like): 3-element acceleration vector [m/s^2]
            start (satkit.time | datetime.datetime | numpy.datetime64): Start time of thrust arc
            end (satkit.time | datetime.datetime | numpy.datetime64): End time of thrust arc; not before ``start``
            frame (satkit.frame): Coordinate frame — **required**, no
                default (matching the Rust API). Supported values:

                - ``frame.GCRF`` — inertial Cartesian
                - ``frame.RTN`` — radial / in-track / cross-track
                - ``frame.NTW`` — normal-to-velocity / tangent / cross-track
                  (use this for thrust along the velocity vector)
                - ``frame.LVLH`` — Local Vertical / Local Horizontal

        Returns:
            thrust: Thrust object

        Raises:
            ValueError: if ``accel`` does not have 3 elements or has a NaN or
                infinite element, ``end`` is before ``start``, or ``frame`` is
                not one of the four above
            TypeError: if ``accel`` is not numeric, or ``start`` or ``end`` is
                not a time
        """
        ...

    @property
    def accel(self) -> npt.NDArray[np.float64]:
        """Acceleration vector in ``frame`` [m/s^2]: a new 3-element numpy array on each access"""
        ...

    @property
    def frame(self) -> frame:
        """Coordinate frame of the acceleration vector"""
        ...

    @property
    def start(self) -> time:
        """Start time of thrust arc (the thrust is on at this instant)"""
        ...

    @property
    def end(self) -> time:
        """End time of thrust arc (the thrust is still on at this instant)"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal when acceleration, frame, start and end are all equal"""
        ...

    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
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
            d0 (float): constant D term
            y0 (float): constant Y term
            b0 (float): constant B term
            dc (float): D cos φ term
            ds (float): D sin φ term
            yc (float): Y cos φ term
            ys (float): Y sin φ term
            bc (float): B cos φ term
            bs (float): B sin φ term
            d2c (float): D cos 2φ term (ECOM2)
            d2s (float): D sin 2φ term (ECOM2)
            d4c (float): D cos 4φ term (ECOM2)
            d4s (float): D sin 4φ term (ECOM2)
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
        """Build from a dict as produced by ``to_dict``; missing keys default to zero / False

        Args:
            d: coefficient names (``d0``, ``y0``, ``b0``, ``dc``, ``ds``, ``yc``,
                ``ys``, ``bc``, ``bs``, ``d2c``, ``d2s``, ``d4c``, ``d4s``) to
                floats, and optionally ``sun_relative`` to a bool

        Returns:
            ecomparams: the coefficients

        Raises:
            ValueError: if the dict has a key that is not one of the above (a
                misspelt coefficient would otherwise be silently zero)
            TypeError: if ``d`` is not a dict, or a value has the wrong type
        """
        ...

    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

@final
class satproperties:
    """Satellite properties relevant for drag, radiation pressure, and thrust

    This class lets the satellite radiation pressure, drag,
    and thrust parameters be set for duration of propagation.

    Attributes:
        cdaoverm (float): Coefficient of drag times area over mass in m^2/kg; not negative
        craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg
        thrusts (list[thrust]): List of continuous thrust arcs (a copy; assign a list to change them)
        ecom (ecomparams | None): ECOM empirical solar-radiation-pressure
            coefficients, added to the cannonball term (use ``craoverm=0``
            for a pure ECOM model). See ``ecomparams`` for the
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

        Raises:
            TypeError: if an argument is passed positionally or has the wrong type
            ValueError: if ``cdaoverm`` is negative or NaN

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
        """Coefficient of drag times area over mass.  Units are m^2/kg

        Assigning a negative or NaN value raises ``ValueError``.
        """
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
        """List of continuous thrust arcs

        A new list on every access: appending to it or removing from it does
        not change the properties. Assign a whole list to change the arcs
        (``props.thrusts = props.thrusts + [arc]``).
        """
        ...

    @thrusts.setter
    def thrusts(self, value: list[thrust]) -> None: ...

    @property
    def ecom(self) -> ecomparams | None:
        """ECOM solar-radiation-pressure coefficients, or None

        When set, the ECOM acceleration (see ``ecomparams`` for the
        DYB frame, sign and eclipse conventions) is added to the cannonball
        term ``craoverm``; use ``craoverm=0`` for a pure ECOM model. The
        "ECOM Solar Radiation Pressure" tutorial shows how to fit the
        coefficients to IGS GPS orbits.

        A copy on every access: setting a coefficient on it does not change
        the properties; assign the modified ``ecomparams`` back.
        """
        ...

    @ecom.setter
    def ecom(self, value: ecomparams | None) -> None: ...
    def __eq__(self, other: object) -> bool:
        """Equal when ``cdaoverm``, ``craoverm``, the thrust arcs (in order) and
        ``ecom`` are all equal. Being mutable and comparable by value, the
        object is not hashable."""
        ...

    __hash__: ClassVar[None]  # type: ignore[assignment]
    def __repr__(self) -> str:
        """One line: ``satproperties(cdaoverm=..., craoverm=..., thrusts=[...], ecom=...)``"""
        ...

    def __str__(self) -> str: ...

@final
class integrator:
    """Choice of ODE integrator for orbit propagation

    Available integrators, from highest to lowest order:

    - ``rkv98`` - Verner 9(8) with 8th-degree dense output, 21 stages (default)
    - ``rkv98_nointerp`` - Verner 9(8) without interpolation, 16 stages. The same as
      ``rkv98`` with ``propsettings.enable_interp=False``
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
    user-chosen fixed step size (``gj_step_seconds``), cannot step across a
    discontinuity, and needs a span of at least 8 steps (``RuntimeError``
    below that), so it's unsuitable for very short propagations.
    ``satstate.propagate`` still handles impulsive maneuvers with it, by
    restarting the integrator at each one; every segment between maneuvers
    must then cover 8 steps.
    """

    rkv98: ClassVar[integrator]
    """Verner 9(8) with 8th-degree dense output, 21 stages (default)

    Highest accuracy integrator. Recommended for precision orbit propagation.
    """

    rkv98_nointerp: ClassVar[integrator]
    """Verner 9(8) without interpolation, 16 stages

    The 16-stage tableau behind ``rkv98``, without the five extra stages that
    only build the interpolant: the same order and error control, and a result
    that never supports ``propresult.interp``, whatever ``enable_interp`` says.

    ``rkv98`` with ``enable_interp=False`` already runs this tableau, so
    choosing this member gains nothing over it (the two take the same number
    of force evaluations); it remains for scripts that name it.
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

    Uses a fixed step size set via ``propsettings.gj_step_seconds`` (its
    magnitude; the direction of propagation sets the sign). The span must
    cover at least 8 steps (``RuntimeError`` otherwise).
    Supports dense output interpolation (quintic Hermite, 5th-order).
    Does not support state transition matrix (``output_phi``) propagation,
    and so cannot propagate a ``satstate`` that has a covariance.
    Not recommended for highly eccentric orbits or integration across
    discontinuities (eclipse boundaries). ``satstate.propagate`` restarts
    it at each impulsive maneuver, so maneuvers work as long as every
    segment between them covers 8 steps.
    """

    def __int__(self) -> int:
        """The member's integer value (``rkv98`` is 0, ... ``gauss_jackson8`` is 6)"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal to the same member, and to its integer value"""
        ...

    def __hash__(self) -> int: ...

@final
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

    def __int__(self) -> int:
        """The member's integer value (``none`` is 0, ``solid_step1`` 1, ``solid_full`` 2)"""
        ...

    def __eq__(self, other: object) -> bool:
        """Equal to the same member, and to its integer value"""
        ...

    def __hash__(self) -> int: ...

@final
class propsettings:
    """This class contains settings used in the high-precision orbit propagator part of the "satkit" python toolbox

    Notes:
        - Default settings:
            - abs_error: 1e-8
            - rel_error: 1e-8
            - gravity_degree: 4
            - gravity_order: same as gravity_degree
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
            - require_eop_coverage: False
            - initial_step_secs: None (derived from the state and tolerances)
        - enable_interp stores the dense output that ``propresult.interp`` needs to interpolate the state between the begin
          and end times; it is enabled by default.  Setting it to False saves the memory and, with the default
          ``integrator.rkv98``, 24% of the force evaluations per step; ``propresult.can_interp`` is then False
        - The object compares equal (``==``) to another with the same settings

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
                in the units of the state (meters for position elements, m/s for velocity elements). Not negative
                (``ValueError``). Default is 1e-8
            rel_error: Maximum relative error of any element in propagated state following ODE integration, unitless.
                Not negative (``ValueError``). Default is 1e-8
            gravity_degree: Maximum degree of spherical harmonic gravity model, at most 70 (``ValueError`` above that). Default is 4
            gravity_order: Maximum order of spherical harmonic gravity model. Must be <= gravity_degree (and so at most 70;
                ``ValueError`` otherwise). Default is None: the order is the same as gravity_degree, and keeps following it
                when ``gravity_degree`` is assigned later, until ``gravity_order`` is assigned
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
            enable_interp: Store the dense output that lets ``propresult.interp`` interpolate the state between begin and end times. Default is True.
                When False, no dense output is stored (``propresult.can_interp`` is False) and ``integrator.rkv98`` runs its 16-stage
                no-interpolant tableau (same order and error control, 24% fewer force evaluations per step).
            integrator: ODE integrator to use. Default is integrator.rkv98
            gj_step_seconds: Fixed step size (seconds) used by ``integrator.gauss_jackson8``; its
                magnitude is used, and the span must cover at least 8 steps.
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

        Raises:
            TypeError: if an argument is passed positionally or has the wrong type
            ValueError: if ``gravity_degree`` is above 70, ``gravity_order`` is above
                ``gravity_degree``, or ``abs_error`` / ``rel_error`` is negative or NaN

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

        Assigning a negative or NaN value raises ``ValueError``.

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

        Assigning a negative or NaN value raises ``ValueError``.

        Returns:
            Maximum relative error of any element in propagated state following ODE integration, unitless; default is 1e-8

        """
        ...

    @rel_error.setter
    def rel_error(self, value: float) -> None: ...
    @property
    def gravity_degree(self) -> int:
        """Maximum degree of spherical harmonic gravity model, at most 70 (``ValueError`` above that)

        Assigning it also sets ``gravity_order`` to the same value, unless
        ``gravity_order`` was given to the constructor or has been assigned; the
        order is then kept, lowered to the new degree if it exceeds it.

        Returns:
            Maximum degree of spherical harmonic gravity model, default is 4

        """
        ...

    @gravity_degree.setter
    def gravity_degree(self, value: int) -> None: ...
    @property
    def gravity_order(self) -> int:
        """Maximum order of spherical harmonic gravity model

        Must be <= ``gravity_degree`` (``ValueError`` otherwise). Until it is
        assigned (or given to the constructor) it equals ``gravity_degree`` and
        follows it: ``ps.gravity_degree = 20`` alone gives a 20x20 field.

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
        """Store the dense output that lets ``propresult.interp`` interpolate the state between begin and end times. Default True.

        When False, no dense output is stored (``propresult.can_interp`` is
        False and ``interp`` raises) and ``integrator.rkv98`` runs its 16-stage
        no-interpolant tableau (same order and error control, 24% fewer force
        evaluations per step). ``integrator.rodas4`` and
        ``integrator.rkv98_nointerp`` never store dense output.
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

        Its magnitude is used (the direction of propagation sets the sign), and
        the span must cover at least 8 steps (``RuntimeError`` from
        ``propagate`` otherwise).
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
    def __eq__(self, other: object) -> bool:
        """Equal when every setting is equal (the table stored by
        ``precompute_terms`` is a cache and is not compared). Being mutable and
        comparable by value, the object is not hashable."""
        ...

    __hash__: ClassVar[None]  # type: ignore[assignment]
    def precompute_terms(
        self,
        begin: TimeScalar,
        end: TimeScalar,
        step: Optional[Union[duration, float, datetime.timedelta]] = None,
    ) -> None:
        """Tabulate the time-only terms of the force model once, for propagating many satellites over the same span

        Stores in these settings a table, at ``step`` intervals from ``begin`` to
        ``end`` (in either order, padded by a few minutes), of the Sun and Moon
        positions and the GCRF-to-ITRF rotation used for Earth gravity.
        ``propagate`` with these settings then interpolates the table instead of
        building its own on every call, whenever its span lies inside the
        tabulated one; otherwise the table is ignored. It is unrelated to
        ``enable_interp`` and ``propresult.interp``. A new call replaces the
        table, and a pickle of the settings does not carry it.

        Args:
            begin (satkit.time | datetime.datetime | numpy.datetime64): One end of the span
            end (satkit.time | datetime.datetime | numpy.datetime64): The other end of the span
            step (satkit.duration | float | datetime.timedelta, optional): Table step.  Default = 60 seconds.  float is interpreted as seconds.

        Raises:
            ValueError: if ``step`` is zero, negative or not finite
            TypeError: if ``begin`` or ``end`` is not a time, or ``step`` is not a
                duration, a float or a timedelta
            RuntimeError: if the table cannot be built (missing JPL ephemeris or
                Earth-orientation data, or a span and step needing too many entries)

        Example:
            ```python
            t0 = satkit.time(2024, 1, 1)
            settings = satkit.propsettings()
            settings.precompute_terms(t0, t0 + satkit.duration(days=1))
            for state in (
                np.array([6.781e6, 0, 0, 0, 7.5e3, 0]),
                np.array([7.0e6, 0, 0, 0, 0, 7.5e3]),
            ):
                result = satkit.propagate(state, t0, duration_days=1, propsettings=settings)
            ```
        """
        ...

def lambert(
    r1: npt.ArrayLike,
    r2: npt.ArrayLike,
    tof: float | duration | datetime.timedelta,
    mu: float | None = None,
    prograde: bool | None = None,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Solve Lambert's problem using Izzo's algorithm (2015).

    Given two position vectors and a time of flight, find the velocity vectors
    for transfer orbits connecting them.

    Args:
        r1: Departure position, meters: any real numeric 3-element
            array-like (numpy array, list or tuple)
        r2: Arrival position, meters (likewise)
        tof: Time of flight, must be positive: a number of seconds, a
            ``satkit.duration`` or a ``datetime.timedelta``
        mu: Gravitational parameter in m³/s² (default: Earth µ = 3.986e14)
        prograde: If True (default), prograde transfer (angular momentum
            h_z >= 0); if False, retrograde (h_z <= 0). This picks the short
            or long way around.

    Returns:
        List of (v1, v2) tuples. Each v1 and v2 is a 3-element numpy array
        in m/s. The first element is the zero-revolution solution; additional
        elements are multi-revolution solutions if they exist.

    Raises:
        TypeError: If ``r1`` or ``r2`` is not a real numeric array-like,
            ``tof`` is not a number, duration or timedelta, or ``mu`` is not
            a number
        ValueError: If inputs are invalid (``r1`` or ``r2`` not 3 elements,
            non-finite values, tof or mu not positive, zero position,
            r1 == r2), or if the solver fails to converge

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
    begin: TimeScalar | None = None,
    end: TimeScalar | None = None,
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

    Propagate orbits with high-precision force modeling. The default integrator is the adaptive
    Verner 9(8) Runge-Kutta pair; ``propsettings.integrator`` selects the others (lower-order
    Runge-Kutta pairs, the RODAS4 Rosenbrock method, fixed-step Gauss-Jackson 8).

    Args:
        state: 6-element array-like representing satellite GCRF position and velocity in meters and meters/second.
            Required unless ``pos`` and ``vel`` are given
        begin: Time at which satellite is at input state. Required (``TypeError`` if omitted); its
            ``None`` default only lets ``state`` be omitted in favour of ``pos`` and ``vel``
        end: Time at which new position and velocity will be computed. May be before ``begin``
            (backward propagation)

    Keyword Args:
        duration: Duration from ``begin`` at which new position & velocity will be computed
        duration_secs: Duration in seconds from ``begin`` at which new position and velocity will be computed
        duration_days: Duration in days from ``begin`` at which new position and velocity will be computed
        output_phi: Output 6x6 state transition matrix between begin and end times. Default is False
        propsettings: Settings for the propagation; if omitted, defaults are used
        satproperties: Drag, radiation pressure and thrust properties of the satellite; if omitted,
            none of the three is modeled
        pos: 3-element GCRF position in meters. Give it together with ``vel``, in place of ``state``
        vel: 3-element GCRF velocity in meters/second. Give it together with ``pos``

    Returns:
        propresult: Propagation result object holding state outputs, statistics,
            and dense output for ``propresult.interp`` unless ``propsettings.enable_interp``
            is False

    Raises:
        TypeError: if ``state`` (or ``pos`` and ``vel``) or ``begin`` is missing, ``pos`` is
            given without ``vel`` or the reverse, none or more than one of ``end``,
            ``duration``, ``duration_secs``, ``duration_days`` is given, an argument has the
            wrong type, or an unknown keyword is passed
        ValueError: if ``state`` does not have 6 elements (``pos`` / ``vel`` 3), the position
            is all zeros, or ``duration_secs`` / ``duration_days`` is NaN or infinite
        RuntimeError: if the propagation fails: a required data file is missing, the
            integrator does not converge or exceeds ``max_steps``, ``output_phi`` is used
            with ``integrator.rodas4`` or ``integrator.gauss_jackson8``, the span is shorter
            than 8 steps of ``integrator.gauss_jackson8``, or ``require_eop_coverage`` is set
            and the span is outside the Earth-orientation table

    Notes:
        Propagates satellite ephemeris (position, velocity in GCRF & time) to new time and
        outputs new position and velocity via numerical integration.
        Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF).

        Included forces:

        - Earth gravity with higher-order spherical-harmonic terms
        - Sun, Moon gravity
        - Solid Earth tides (IERS 2010 Step 1 by default; ``propsettings.tide_model``
          selects ``tidemodel.solid_step1``, ``tidemodel.solid_full`` or ``tidemodel.none``)
        - General relativity (IERS 2010 Eq. 10.12; ``propsettings.use_relativistic_correction``)
        - Radiation pressure
        - Atmospheric drag: NRL-MSISE 2000 density model, with option to include space weather effects

        An end time is required: pass exactly one of ``end``, ``duration``, ``duration_secs``
        and ``duration_days`` (``TypeError`` if none or more than one is given; ``ValueError``
        for a NaN or infinite ``duration_secs`` / ``duration_days``).

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

def omm_from_url(url: str) -> list[dict[str, Any]]:
    """Load OMM(s) from a URL as a list of dictionaries

    Fetches the content at the given URL and auto-detects the format (JSON,
    XML or CSV), as ``omm_from_text`` does. A CelesTrak GP query with no
    ``FORMAT`` answers in CSV.
    Returns a list of dictionaries that can be passed directly to ``sgp4``.

    Args:
        url (str): URL to fetch OMM data from (e.g. CelesTrak or Space-Track endpoint)

    Returns:
        list[dict[str, Any]]: one dictionary per message, with every CCSDS field the
            source provided (the keys of ``OMMDict``) plus the source's extra keys

    Raises:
        RuntimeError: if offline mode is on (``SATKIT_OFFLINE=1`` or
            ``satkit.utils.set_offline(True)``; no connection is opened) or the
            request fails
        ValueError: if the response is not JSON, XML or CSV, or is malformed

    Example:
        ```python
        import satkit as sk

        omms = sk.omm_from_url("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json")
        pos, vel = sk.sgp4(omms[0], sk.time(2024, 1, 1))
        ```
    """
    ...

def omm_from_file(filename: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load OMM(s) from a JSON, XML or CSV file as a list of dictionaries

    The format is detected from the content, not the file extension, as
    ``omm_from_text`` does: a file starting with ``[`` or ``{`` is JSON
    (one message or an array of them), one starting with ``<`` is a CCSDS
    NDM/XML document, and one whose first line holds a comma is a CSV table.
    KVN is not supported.

    Args:
        filename (str | os.PathLike): path to the file

    Returns:
        list[dict[str, Any]]: one dictionary per message, as ``omm_from_text`` returns

    Raises:
        FileNotFoundError: if the file does not exist; the exception names
            the path. Another I/O failure raises the matching ``OSError``
        ValueError: if the content is none of the three formats or is malformed

    Example:
        ```python
        import satkit as sk

        omms = sk.omm_from_file("gp.xml")   # saved from Space-Track or CelesTrak
        pos, vel = sk.sgp4(omms, sk.time(2024, 1, 1))
        ```
    """
    ...

def omm_from_text(text: str) -> list[dict[str, Any]]:
    """Parse OMM(s) from JSON, XML or CSV text as a list of dictionaries

    Text starting with ``[`` or ``{`` is parsed as JSON (one message or an
    array of them), text starting with ``<`` as a CCSDS NDM/XML document, and
    text whose first line holds a comma as CSV. KVN is not supported.

    The CSV form is a header row of OMM keywords and one row per message, as
    CelesTrak (its default for GP queries) and Space-Track serve it. Fields
    may be quoted and columns may come in any order. An empty cell is an
    absent field, except ``OBJECT_NAME`` and ``OBJECT_ID``, which stay empty
    strings; columns that are not OMM fields are kept as strings (``None``
    where empty).

    Args:
        text (str): the document

    Returns:
        list[dict[str, Any]]: one dictionary per message. The CCSDS fields
            have the keys and value types of ``OMMDict``, with ``EPOCH`` an
            RFC 3339 string; keys that are not CCSDS fields (Space-Track's
            ``OBJECT_TYPE``, ``TLE_LINE1``, ...; XML ``USER_DEFINED``
            parameters; extra CSV columns) are kept as the source gave them,
            which is why the type is a plain ``dict`` and not ``OMMDict``

    Raises:
        ValueError: if the text is none of the three formats or is
            malformed; for CSV the message gives the line of the bad row, and
            a table with no ``EPOCH`` or ``MEAN_MOTION`` column is refused

    Example:
        ```python
        import satkit as sk
        import requests

        r = requests.get("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=xml")
        omm = sk.omm_from_text(r.text)[0]
        ```
    """
    ...
