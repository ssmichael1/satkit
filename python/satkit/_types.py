"""Pure-Python types shared by the stubs and the runtime package."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np

    from .satkit import time, tlefitstatus

__all__ = ["OMMDict", "TLEFitResult"]


class OMMDict(TypedDict, total=False):
    """An Orbital Mean-Element Message (OMM) as a plain dictionary

    This is the shape returned by ``TLE.to_omm`` and accepted by ``sgp4``
    and ``TLE.from_omm``. It is an ordinary ``dict`` (this class exists for
    type checking only): the keys are the CCSDS 502.0-B-3 field names, so
    ``json.load`` on a Space-Track or CelesTrak JSON response produces one
    directly. Every key is optional in the type; ``sgp4`` needs ``EPOCH``
    and the six mean elements.

    ``omm_from_url``, ``omm_from_file`` and ``omm_from_text`` return
    dictionaries with these keys and whatever other keys their source had,
    so they are typed ``dict[str, Any]``, which ``sgp4`` and ``TLE.from_omm``
    accept as well. To have a type checker verify the CCSDS keys of one, use
    ``typing.cast(OMMDict, omm)``.

    Units follow the TLE convention: angles in degrees, ``MEAN_MOTION`` in
    revolutions per day, ``MEAN_MOTION_DOT`` in rev/day² and
    ``MEAN_MOTION_DDOT`` in rev/day³ (the TLE fields, i.e. ṅ/2 and n̈/6),
    ``BSTAR`` in inverse Earth radii, ``GM`` in km³/s², ``MASS`` in kg, areas in
    m², ``BTERM`` and ``AGOM`` in m²/kg.

    The loaders return numbers as ``float`` / ``int`` (Space-Track quotes them
    as strings in its JSON; ``sgp4`` accepts either), ``EPOCH`` as an
    RFC 3339 string, absent optional fields omitted, and every key that is not
    one of the fields below (Space-Track's ``OBJECT_TYPE``, ``RCS_SIZE``,
    ``LAUNCH_DATE``, ``TLE_LINE1``, ...; XML ``USER_DEFINED`` parameters; extra
    CSV columns, as strings) kept verbatim.

    As input, ``EPOCH`` may also be a ``satkit.time``, a
    ``datetime.datetime`` (naive = local time) or a ``numpy.datetime64``
    (UTC).
    """

    CCSDS_OMM_VERS: str
    COMMENT: str
    CREATION_DATE: str
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
    EPOCH: str | time | datetime.datetime | np.datetime64
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


class TLEFitResult(TypedDict):
    """Results of ``TLE.fit_from_states``, returned next to the fitted TLE

    An ordinary ``dict`` (this class exists for type checking only). The
    norms are those of the residual vector the Levenberg-Marquardt solver
    minimizes.
    """

    status: tlefitstatus
    """Why the solver stopped"""
    converged: bool
    """``status.converged``: True if it stopped on a convergence tolerance"""
    orig_norm: float
    """Residual norm at the initial guess"""
    best_norm: float
    """Residual norm at the fitted elements"""
    grad_norm: float
    """Norm of the gradient at the fitted elements"""
    n_iter: int
    """Number of iterations taken"""
    n_res_evals: int
    """Number of residual evaluations (SGP4 runs over all the states)"""
