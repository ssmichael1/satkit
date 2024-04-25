# Coordinate Frame Transforms

## Introduction

The ``satkit.frametransform`` module provides functions for transforming between various coordinate
frames used in satellite tracking and orbit determination.  These include multiple variations of "inertial"
coordinate frames, and multiple verisons of "Earth-fixed" coordinate frames. 

Some notes:

* Most of the algorithms in this module are from the book 
  `"Fundamentals of Astrodynamics and Applications"`` by David Vallado.

* The frame transforms are defined as arbitrary rotations in a 3-dimensional space.
  The rotations are a function of time, and are represented as [Quaternions](quaternion-api)
* The rotation from the Geocentric Celestial Reference Frame (GCRF) to the Earth-Centered Inertial (ECI) frame
  is defined by the International Astronomical Union (IAU), available [here](https://www.iers.org/).  See IERS
  Technical Note 36 for the latest values. 

## API Reference

```{eval-rst}
.. autoapimodule:: satkit.frametransform
   :members:
```