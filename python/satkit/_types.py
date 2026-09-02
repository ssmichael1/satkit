"""Pure-Python types shared by the stubs and the runtime package."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .satkit import time

__all__ = ["OMMDict"]


class OMMDict(TypedDict, total=False):
    """An Orbital Mean-Element Message (OMM) as a plain dictionary

    This is the shape returned by :func:`omm_from_url`, :func:`omm_from_file`,
    :func:`omm_from_text` and :meth:`TLE.to_omm`, and accepted by :func:`sgp4`
    and :meth:`TLE.from_omm`. It is an ordinary ``dict`` (this class exists for
    type checking only): the keys are the CCSDS 502.0-B-3 field names, so
    ``json.load`` on a Space-Track or CelesTrak JSON response produces one
    directly. Every key is optional in the type; :func:`sgp4` needs ``EPOCH``
    and the six mean elements.

    Units follow the TLE convention: angles in degrees, ``MEAN_MOTION`` in
    revolutions per day, ``MEAN_MOTION_DOT`` in rev/day² and
    ``MEAN_MOTION_DDOT`` in rev/day³ (the TLE fields, i.e. ṅ/2 and n̈/6),
    ``BSTAR`` in inverse Earth radii, ``GM`` in km³/s², ``MASS`` in kg, areas in
    m², ``BTERM`` and ``AGOM`` in m²/kg.

    The loaders return numbers as ``float`` / ``int`` (Space-Track quotes them
    as strings in its JSON; :func:`sgp4` accepts either), ``EPOCH`` as an
    RFC 3339 string, absent optional fields omitted, and every key that is not
    a CCSDS field (Space-Track's ``OBJECT_TYPE``, ``RCS_SIZE``, ``LAUNCH_DATE``,
    ``TLE_LINE1``, ...; XML ``USER_DEFINED`` parameters) kept verbatim.
    """

    CCSDS_OMM_VERS: str
    COMMENT: str
    ORIGINATOR: str
    CLASSIFICATION: str
    MESSAGE_ID: str
    OBJECT_NAME: str
    OBJECT_ID: str
    CENTER_NAME: str
    REF_FRAME: str
    REF_FRAME_EPOCH: str
    TIME_SYSTEM: str
    MEAN_ELEMENT_THEORY: str
    EPOCH: str | time | datetime.datetime
    MEAN_MOTION: float | str
    ECCENTRICITY: float | str
    INCLINATION: float | str
    RA_OF_ASC_NODE: float | str
    ARG_OF_PERICENTER: float | str
    MEAN_ANOMALY: float | str
    GM: float | str | None
    MASS: float | str | None
    SOLAR_RAD_AREA: float | str | None
    DRAG_AREA: float | str | None
    SOLAR_RAD_COEFF: float | str | None
    DRAG_COEFF: float | str | None
    EPHEMERIS_TYPE: int | str | None
    CLASSIFICATION_TYPE: str
    NORAD_CAT_ID: int | str | None
    ELEMENT_SET_NO: int | str | None
    REV_AT_EPOCH: int | str | None
    BSTAR: float | str | None
    BTERM: float | str | None
    MEAN_MOTION_DOT: float | str | None
    MEAN_MOTION_DDOT: float | str | None
    AGOM: float | str | None
