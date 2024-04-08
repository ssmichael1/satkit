"""
Utility functions for SatKit
"""

from __future__ import annotations
import typing
import numpy.typing as npt
import numpy as np

import satkit

def update_datafiles(**kwargs):
    """
    Download & store data files needed for "satkit" computations

    Keyword Arguments:

      overwrite:  <bool>  :: Download and overwrite files if they already exist
            dir: <string> :: Target directory for files.  Uses existing
                             data directory if not specified
                             (see "datadir" function)

    Files include:

            EGM96.gfc :: EGM-96 Gravity Model Coefficients
                JGM3.gfc :: JGM-3 Gravity Model Coefficients
                JGM2.gfc :: JGM-2 Gravity Model Coefficients
        ITU_GRACE16.gfc :: ITU Grace 16 Gravity Model Coefficients
            tab5.2a.txt :: Coefficients for GCRS to GCRF conversion
            tab5.2b.txt :: Coefficients for GCRS to GCRF conversion
            tab5.2d.txt :: Coefficients for GCRS to GCRF conversion
        sw19571001.txt :: Space weather data, updated daily
        leap-seconds.txt :: Leap seconds (UTC vs TAI)
        finals2000A.all :: Earth orientation parameters,  updated daily
    linux_p1550p2650.440 :: JPL Ephemeris version 440 (~ 100 MB)

    Note that files update daily will always be downloaded independed of
    overwrite flag

    """

def datadir() -> str:
    """
    Return directory currently used to hold
    necessary data files for the directory

    e.g., Earth Orientation Parameters, gravity coefficients,
    JPL Ephemeris, etc..

    Data directory is 1st of following directories search that contains
    the data files listed in "update_datafiles"

    MacOS:

    1. Directory pointed to by "SATKIT_DATA" environment variable
    2. $HOME/LIBRARY/Application\ Support/astro-data
    3. $HOME/.astro-data
    4. $HOME
    5. /usr/share/astro-data
    6. /Library/Application\ Support/astro-data

    Linux:

    1. Directory pointed to by "ASTRO_DATA" directory
    2. $HOME/.astro-data
    3. $HOME
    4. /usr/share/astro-data

    Windows:
    1. Directory pointed to by "ASTRO_DATA" directory
    2. $HOME/.astro-data
    3. $HOME


    """

def githash() -> str:
    """
    Return git hash of this satkit build
    """

def builddate() -> str:
    """
    Return build date of this satkit library as a string
    """
