/* -----------------------------------------------------------------------------
*
*                           function getgravconst
*
*  this function gets constants for the propagator. note that mu is identified to
*    facilitiate comparisons with newer models. the common useage is wgs72.
*
*  author        : david vallado                  719-573-2600   21 jul 2006
*
*  inputs        :
*    whichconst  - which set of constants to use  wgs72old, wgs72, wgs84
*
*  outputs       :
*    tumin       - minutes in one time unit
*    mu          - earth gravitational parameter
*    radiusearthkm - radius of the earth in km
*    xke         - reciprocal of tumin
*    j2, j3, j4  - un-normalized zonal harmonic values
*    j3oj2       - j3 divided by j2
*
*  locals        :
*
*  coupling      :
*    none
*
*  references    :
*    norad spacetrack report #3
*    vallado, crawford, hujsak, kelso  2006
--------------------------------------------------------------------------- */
use super::GravConst;

#[allow(clippy::too_many_arguments)]
pub fn getgravconst(
    whichconst: GravConst,
    tumin: &mut f64,
    mus: &mut f64,
    radiusearthkm: &mut f64,
    xke: &mut f64,
    j2: &mut f64,
    j3: &mut f64,
    j4: &mut f64,
    j3oj2: &mut f64,
) {
    match whichconst {
        // -- wgs-72 low precision str#3 constants --
        GravConst::WGS72OLD => {
            *mus = 398600.79964; // in km3 / s2
            *radiusearthkm = 6378.135; // km
            *xke = 0.0743669161; // reciprocal of tumin
            *tumin = 1.0 / *xke;
            *j2 = 0.001082616;
            *j3 = -0.00000253881;
            *j4 = -0.00000165597;
            *j3oj2 = *j3 / *j2;
        }
        // ------------ wgs-72 constants ------------
        GravConst::WGS72 => {
            *mus = 398600.8; // in km3 / s2
            *radiusearthkm = 6378.135; // km
            *xke = 60.0 / f64::sqrt(*radiusearthkm * *radiusearthkm * *radiusearthkm / *mus);
            *tumin = 1.0 / *xke;
            *j2 = 0.001082616;
            *j3 = -0.00000253881;
            *j4 = -0.00000165597;
            *j3oj2 = *j3 / *j2;
        }
        GravConst::WGS84 => {
            // ------------ wgs-84 constants ------------
            *mus = 398600.5; // in km3 / s2
            *radiusearthkm = 6378.137; // km
            *xke = 60.0 / f64::sqrt(*radiusearthkm * *radiusearthkm * *radiusearthkm / *mus);
            *tumin = 1.0 / *xke;
            *j2 = 0.00108262998905;
            *j3 = -0.00000253215306;
            *j4 = -0.00000161098761;
            *j3oj2 = *j3 / *j2;
        }
    }
} // getgravconst
