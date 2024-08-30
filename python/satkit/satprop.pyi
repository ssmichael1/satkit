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

class propstats:
    @property
    def num_eval() -> int:
        """Number of function evaluations"""

    @property
    def num_accept() -> int:
        """Number of accepted steps in adaptive RK integrator"""

    @property
    def num_reject() -> int:
        """Number of rejected steps in adaptive RK integrator"""

class propresult:
    """Results of a satellite propagation
    
    This class lets the user access results of the satellite propagation
    """

    @property
    def pos() -> npt.ArrayLike[float]:
        """GCRF position of satellite, meters"""

    @property
    def vel() -> npt.ArrayLike[float]:
        """GCRF velocity of satellite, meters/second"""

    @property
    def state() -> npt.ArrayLike[float]:
        """6-element state (pos + vel) of satellite in meters & meters/second"""

    @property
    def time() -> satkit.time:
        """Time at which state is valid"""

    @property
    def stats() -> propstats:
        """Statistics of propagation"""

    @property
    def phi() -> npt.ArrayLike[np.float64]|None:
        """6x6 State transition matrix 
        or None if not computed
        """        

    def interp(time: satkit.time, output_phi: bool=False) -> npt.ArrayLike[np.float64]|typing.Tuple[npt.ArrayLike[np.float64], npt.ArrayLike[np.float64]]:
        """Interpolate state at given time

        Args:
            time (satkit.time): Time at which to interpolate state

        Keyword Arguments:
            output_phi (bool): Output 6x6 state transition matrix at the interpolated time

        Returns:
            6-element vector representing state at given time
            if output_phi, also output 6x6 state transition matrix at given time
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
) -> propresult:
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

    Keyword Args: 
        stop_time (satkit.time): satkit.time object representing instant at which new position and velocity will be computed
        duration_secs (float): duration in seconds from "tm" for at which new position and velocity will be computed.
        duration_days (float): duration in days from "tm" at which new position and velocity will be computed.
        duration (satkit.duration): duration from "tm" at which new position & velocity will be computed.
        output_phi (bool): Output 6x6 state transition matrix between "starttime" and "stoptime" (and at intervals, if specified)
        propsettings (propsettings): "propsettings" object with input settings for the propagation. if left out, default will be used.
        satproperties (satproperties_static): "SatPropertiesStatic" object with drag and radiation pressure succeptibility of satellite. If left out, drag and radiation pressure are neglected
        output_dense: boolean indicting whether or not dense output should be recorded.  Default is false.  If true, this will allow for calling the "interp" function to query states at arbitrary times between the start time and the stop time

    Returns:
        (propresult): Propagation result object holding state outputs, statistics, and dense output if requested
    """
