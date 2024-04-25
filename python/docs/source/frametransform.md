# Frame Transforms

The ```satkit.frametransform``` module includes functions to rotate between multiple inertial and Earth-fixed coordinate frames.

Frame transforms are generally output as **Quaternions**.  See related documentation [here](quaternion.md)


## Coordinate frames

The `satkit.frametransform` module supports transforms between the following frames:

* `GCRF` - Geocentric Celestial Reference Frame.  
* `ITRF` - International Terrestrial Reference Frame
* `TEME` - True Equator, Mean Equinox Frame
* `MOD` - Mean of Date Frame
* `TIRS` - Terrestrial Intermediate Reference System
* `CIRS` - Celestial Intermediate Reference System

## Calculation Notes:

Transformation between the `GCRF` and the `ITRF` is performed via the detailed calculations described by the International Earth Rotation and Reference System Service in their Technical Note 36: <https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html>.  This is a computationally expensive calculation.