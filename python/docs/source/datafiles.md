# Data Files

The ``satkit`` package relies upon a number of data files for certain calculations: 

* **leap-seconds.list** <br/>A list of the UTC leap seconds since 1972.  This is a common file in *nix platforms and is used to keep track of the number of seconds (currently 37) that UTC lags TAI
<br/>

* **linux_p1550p2650.440**<br>File containing the precice ephemerides of the planets and 400 large asteroids between the years 1550 and 2650, as modelled by the Jet Propulsion Laboraotry (JPL).  Note: this file is large -- approx. 100 MB -- and may take a long time to download


* **tab5.2a.txt**, **tab5.2b.txt**, **tab5.2d.txt**<br>Tables from IERS Conventions Technical Note 36, containng coefficients used in the precice rotation between the inertial International Celestial Reference Frame and the Earth-fixed International Terrestrial Reference Frame.


* **EGM96.gfc**, **JGM2.gfc**, **JGM3.gfc**, **ITU-GRACE16.gfc**<br/>Files containing gravity coefficients for various gravity models.  These are used to compute the precise acceleration due to Earth gravity as a function of position in the Earth-fixed ITRF frame.


* **SW-All.csv**<br/>Space Weather.  The solar flux at $\lambda~=~10.7cm$ (2800 Mhz) is an indication of solar activity, which in turn is an important predictor of air density at altitudes relevant for low-Earth orbits.
This file is updated at [celestrack.org](https://www.celestrak.org) every 3 hours with the most-recent space weather information. 


* **EOP-All.csv**<br/>Earth orientation parameters.  This includes $\Delta UT1$, the difference between $UT1$ and $UTC$, as well as $x_p$ and $y_p$, the polar "wander" of the Earth rotation axis.  This file is updated daily with most-recent values at [celestrak.org](https://www.celestrak.org)

## Acquiring the data files

The data files are downloaded on-demand if they are needed but do not exist.  The data files can be manually downloaded with the following command:

```python
satkit.utils.update_datafiles()
```

If the files alaready exist, they will *not* be downloaded, with the exception of the space weather and earth orientation paramters files, as these are regularly updated.

## Download location

The data files are all downloaded into a common directory.  This directory can be queried via python:

```python
satkit.utils.datadir()
```

The ``satkit`` package will search for the data files in the following locations, in order, stopping when the files are found:

* Directory pointed to by the ``SATKIT_DATA`` environment variable

* ``$HOME/.satkit-data``<br/>

* ``$DYLIB_PATH/share/satkit-data`` where ``$DYLIB_PATH`` is the location of the satkit shared library called by python.
<br/>
* *For Mac users only* : ``$HOME/Libary/Application Support/satkit-data``
<br/>
* ``$HOME/satkit-data``
<br/>
* ``$HOME``
<br/>
* ``/usr/share/satkit-data``
<br/>
* *For Mac users only* : ``/Library/Application Support/satkit-data``
<br/>

If no files are found, the ``satkit`` package will go through the above list of directories in order, stopping when a directory either exists and is writable, or can be created and is writable.  The files will then be downloaded to that location.



