"""
High-precision satellite propagation using Runga-Kutta methods for
differential equation solving

Force models are mostly pulled from Montenbruck & Gill:
https://link.springer.com/book/10.1007/978-3-642-58351-3

The propagator can also compute the state transition matrix, meaning
position and velocity covariances can be propagated as well.

The default propagator uses a Runga-Kutta 9(8) integrator 
with coefficient computed by Verner:
https://www.sfu.ca/~jverner/

This works much better than lower-order Runga-Kutta solvers
such as Dormund-Prince, and I don't know why it isn't more
popular in numerical packages

This module includes a function to propagate a position and time directly,
and a convenience "satstate" object that represents satellite position, velocity,
and optionally covariance and can propagate itself to different times 

Forces included in the propagator:

1. Earth gravity with higher-order zonal terms
2. Sun, Moon gravity
3. Radiation pressure
4. Atmospheric drag: NRL-MISE 2000 density model, with option
   to include space weather effects (can be large)

"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit
import datetime

class PropStats(typing.TypedDict):
    """PropStats is a dictionary containing statistics for the propagation.
    """
    
    num_eval: int
    """int: Number of function evaluations of the force model computed during propagation function call
    """

    accepted_steps: int
    """int: Accepted steps in the adaptive Runga-Kutta solver
    """
    
    rejected_steps: int
    """int: Rejected steps in the adaptive Runga-Kutta solver
    """

class PropResult(typing.TypedDict):
    """
    PropResult is a dictionary containing the results of a high-precision orbit propagation that is returned by the "propagate" function
    """

    time: npt.ArrayLike[satkit.time]
    """npt.ArrayLike[satkit.time]: List of satkit.time objects at which time is computed
    """
    
    pos: npt.ArrayLike[np.float64]
    """npt.ArrayLike[float]: GCRF position in meters at output times.  Output is Nx3 numpy matrix, where N is the number of times
    """

    vel: npt.ArrayLike[np.float64]
    """npt.ArrayLike[float]: GCRF velocity in meters per second at output times.  Output is Nx3 numpy matrix, where N is the number of times
    """

    Phi: typing.NotRequired[npt.ArrayLike[np.float64]]
    """6x6 State transition matrix corresponding to each time. Output is Nx6x6 numpy matrix, where N is the lenght of the output "time" list. Not included if output_phi kwarg is set to false (the default)
    """

    stats: PropStats
    """
    (PropStats): Python dictionary with statistics for the propagation. This includes:

            * num_eval: Number of function evaluations of the force model required to get solution with desired accuracy
            * accepted_steps: Accepted steps in the adpative Runga-Kutta solver
            * rejected_steps: Rejected steps in the adaptive Runga-Kutta solver
    """

class satstate:
    """
    A convenience class representing a satellite position and velocity, and
    optionally 6x6 position/velocity covariance at a particular instant in time

    This class can be used to propagate the position, velocity, and optional
    covariance to different points in time.
    """

    def __init__(
        self,
        time: satkit.time,
        pos: npt.ArrayLike[np.float64],
        vel: npt.ArrayLike[np.float64],
        cov: npt.ArrayLike[np.float64] | None = None,
    ):
        """Create a new satellite state

        Args:
            time (satkit.time): Time instant of this state
            pos (npt.ArrayLike[np.float64]): Position in meters in GCRF frame
            vel (npt.ArrayLike[np.float64]): Velocity in meters / second in GCRF frame
            cov (npt.ArrayLike[np.float64]|None, optional): Covariance in GCRF frame. Defaults to None.  If input, should be a 6x6 numpy array
        
        Returns:
            satstate: New satellite state object
        """

    @property
    def pos(self) -> npt.ArrayLike[np.float64]:
        """state position in meters in GCRF frame
        
        Returns:
            npt.ArrayLike[np.float64]: 3-element numpy array representing position in meters in GCRF frame
        """

    @property
    def vel(self) -> npt.ArrayLike[np.float64]:
        """Return this state velocity in meters / second in GCRF

        Returns:
            npt.ArrayLike[np.float64]: 3-element numpy array representing velocity in meters / second in GCRF frame
        """

    @property
    def qgcrf2lvlh(self) -> satkit.quaternion:
        """ Quaternion that rotates from the GCRF to the LVLH frame for the current state

        Returns:
            satkit.quaternion: Quaternion that rotates from the GCRF to the LVLH frame for the current state
        """


    @property
    def cov(self) -> npt.ArrayLike[np.float64] | None:
        """6x6 state covariance matrix in GCRF frame
        
        Returns:
            npt.ArrayLike[np.float64] | None: 6x6 numpy array representing state covariance in GCRF frame or None if not set
        """


    @property
    def time(self) -> satkit.time:
        """Return time of this satellite state
        
        Returns:
            satkit.time: Time instant of this state
        """

    def propagate(self, time: satkit.time, propsettings=None) -> satstate:
        """Propagate this state to a new time, specified by the "time" input, updating the position, the velocity, and the covariance if set

        Keyword Arguments:
            propsettings: satkit.satprop.propsettings object describing settings to use in the propagation. If omitted, default is used 

        Returns:
            satstate: New satellite state object representing the state at the new time
        """

class satproperties_static:
    """Satellite properties relevant for drag and radiation pressure

    This class lets the satellite radiation pressure and drag
    paramters be set to static values for duration of propagation
    """

    def __init__(self, *args, **kwargs):
        """Create a satproperties_static object with given craoverm and cdaoverm in m^2/kg

        Args:
            cdaroverm (float): Coefficient of drag times area over mass in m^2/kg
            craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg
        
        Keyword Arguments:
            craoverm (float): Coefficient of radiation pressure times area over mass in m^2/kg
            cdaoverm (float): Coefficient of drag times area over mass in m^2/kg


        Notes:
        The two arguments can be passed as positional arguments or as keyword arguments
            
        Example:

        >>> properties = satproperties_static(craoverm = 0.5, cdaoverm = 0.4)
        
        or with same output
        
        >>> properties = satproperties_static(0.5, 0.4)

        """

        @property
        def cdaoverm() -> float:
            """Coeffecient of drag times area over mass.  Units are m^2/kg
            """

        @property
        def craoverm() -> float:
            """Coefficient of radiation pressure times area over mass.  Units are m^2/kg
            """


class propsettings:
    """This class contains settings used in the high-precision orbit propgator part of the "satkit" python toolbox
    """

    def __init__(self):
        """ Create default propsetting object

        Notes:
            * Default settings:
                * abs_error: 1e-8
                * rel_error: 1e-8
                * gravity_order: 4
                * use_spaceweather: True
                * use_jplephem: True


        Returns:
            propsettings: New propsettings object with default settings
        """

    @property
    def abs_error() -> float:
        """Maxmum absolute value of error for any element in propagated state following ODE integration

        Default: 1e-8
        """

    @property
    def rel_error() -> float:
        """Maximum relative error of any element in propagated state following ODE integration

        Default: 1e-8
        """

    @property
    def gravity_order() -> int:
        """Earth gravity order to use in ODE integration

        Default: 4
        """

    @property
    def use_spaceweather() -> bool:
        """Use space weather data when computing atmospheric density for drag forces

        Default: true

        Notes:
            * Space weather data can have a large effect on the density of the atmosphere
            * This can be important for accurate drag force calculations
            * Space weather data is updated every 3 hours.  Most-recent data can be downloaded with ``satkit.utils.update_datafiles()``

        """

    @property
    def use_jplephem() -> bool:
        """Use high-precision but computationally expensive JPL ephemerides for sun and mun when computing their gravitational force
        """

def propagate(
    pos: npt.ArrayLike[float],
    vel: npt.ArrayLike[float],
    start: satkit.time,
    **kwargs,
) -> PropResult:
    """High-precision orbit propagator

    Notes:
        * Propagator uses advanced Runga-Kutta integrators and includes the following forces:
            * Earth gravity with higher-order zonal terms
            * Sun, Moon gravity
            * Radiation pressure
            * Atmospheric drag: NRL-MISE 2000 density model, with option to include space weather effects (can be large)
        * Stop time must be set by keyword argument, either explicitely or by duration
        * Solid Earth tides are not (yet) included in the model


    Propagate statellite ephemeris (position, velocity in gcrs & time) to new time
    and output new position and velocity via Runge-Kutta integration.

    Inputs and outputs are all in the Geocentric Celestial Reference Frame (GCRF)

    Args:
        pos (npt.ArrayLike[float]): 3-element numpy array representing satellite GCRF position in meters
        vel (npt.ArrayLike[float]): 3-element numpy array representing satellite GCRF velocity in m/s
        tm (satkit.time): satkit.time object representing instant at which satellite is at "pos" & "vel"

    Keyword Arguments:
        stop_time (satkit.time): satkit.time object representing instant at which new position and velocity will be computed
        duration_secs (float): duration in seconds from "tm" for at which new position and velocity will be computed.
        duration_days (float): duration in days from "tm" at which new position and velocity will be computed.
        duration (satkit.duration): duration from "tm" at which new position & velocity will be computed.
        dt_secs (float): Interval in seconds between "starttime" and "stoptime" at which solution will also be computed
        dt_days (float): Interval in days between "starttime" and "stoptime" at which solution will also be computed
        dt (satkit.duration): Interval over which new position & velocity will be computed
        output_phi (bool): Output 6x6 state transition matrix between "starttime" and "stoptime" (and at intervals, if specified)
        propsettings (propsettings): "propsettings" object with input settings for the propagation. if left out, default will be used.
        satproperties (satproperties_static): "SatPropertiesStatic" object with drag and radiation pressure succeptibility of satellite. If left out, drag and radiation pressure are neglected

    Returns:
        (PropResult): Python dictionary with the following elements:
            * "time": list of satkit.time objects at which solution is computed
            * "pos": GCRF position in meters at "time".  Output is a Nx3 numpy matrix, where N is the length of the output "time" list
            * "vel": GCRF velocity in meters / second at "time".  Output is a Nx3 numpy matrix, where N is the length of the output "time" list
            *  "Phi": 6x6 State transition matrix corresponding to each time. Output is Nx6x6 numpy matrix, where N is the lenght of the output "time" list. Not included if output_phi kwarg is set to false (the default)
            * "stats": Python dictionary with statistics for the propagation. This includes:
                * "num_eval": Number of function evaluations of the force model required to get solution with desired accuracy
                * "accepted_steps": Accepted steps in the adpative Runga-Kutta solver
                * "rejected_steps": Rejected steps in the adaptive Runga-Kutta solver
    """
