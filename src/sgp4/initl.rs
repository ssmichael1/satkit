/* -----------------------------------------------------------------------------
*
*                           function gstime_SGP4
*
*  this function finds the greenwich sidereal time.
*
*  author        : david vallado                  719-573-2600    1 mar 2001
*
*  inputs          description                    range / units
*    jdut1       - julian date in ut1             days from 4713 bc
*
*  outputs       :
*    gstime      - greenwich sidereal time        0 to 2pi rad
*
*  locals        :
*    temp        - temporary variable for doubles   rad
*    tut1        - julian centuries from the
*                  jan 1, 2000 12 h epoch (ut1)
*
*  coupling      :
*    none
*
*  references    :
*    vallado       2013, 187, eq 3-45
* --------------------------------------------------------------------------- */

/* c++ comment out
double  gstime_SGP4
(
double jdut1
)
*/
const TWOPI: f64 = std::f64::consts::PI * 2.0;
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

fn gstime_sgp4(jdut1: f64) -> f64 {
    let tut1 = (jdut1 - 2451545.0) / 36525.0;
    let mut temp = 876600.0f64.mul_add(3600.0, 8640184.812866).mul_add(tut1, (-6.2e-6 * tut1 * tut1).mul_add(tut1, 0.093104 * tut1 * tut1))
        + 67310.54841; // sec
    temp = (temp * DEG2RAD / 240.0) % TWOPI; //360/86400 = 1/240, to deg, to rad

    // ------------------------ check quadrants ---------------------
    if temp < 0.0 {
        temp += TWOPI;
    }

    temp
} // gstime

/*-----------------------------------------------------------------------------
*
*                           procedure initl
*
*  this procedure initializes the spg4 propagator. all the initialization is
*    consolidated here instead of having multiple loops inside other routines.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    satn        - satellite number - not needed, placed in satrec
*    xke         - reciprocal of tumin
*    j2          - j2 zonal harmonic
*    ecco        - eccentricity                           0.0 - 1.0
*    epoch       - epoch time in days from jan 0, 1950. 0 hr
*    inclo       - inclination of satellite
*    no          - mean motion of satellite
*
*  outputs       :
*    ainv        - 1.0 / a
*    ao          - semi major axis
*    con41       -
*    con42       - 1.0 - 5.0 cos(i)
*    cosio       - cosine of inclination
*    cosio2      - cosio squared
*    eccsq       - eccentricity squared
*    method      - flag for deep space                    'd', 'n'
*    omeosq      - 1.0 - ecco * ecco
*    posq        - semi-parameter squared
*    rp          - radius of perigee
*    rteosq      - square root of (1.0 - ecco*ecco)
*    sinio       - sine of inclination
*    gsto        - gst at time of observation               rad
*    no          - mean motion of satellite
*
*  locals        :
*    ak          -
*    d1          -
*    del         -
*    adel        -
*    po          -
*
*  coupling      :
*    getgravconst- no longer used
*    gstime      - find greenwich sidereal time from the julian date
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
----------------------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn initl(
    // sgp4fix satn not needed. include in satrec in case needed later
    // int satn,
    // sgp4fix just pass in xke and j2
    // gravconsttype whichconst,
    xke: f64,
    j2: f64,
    ecco: f64,
    epoch: f64,
    inclo: f64,
    no_kozai: f64,
    _opsmode: super::OpsMode,
    method: &mut char,
    ainv: &mut f64,
    ao: &mut f64,
    con41: &mut f64,
    con42: &mut f64,
    cosio: &mut f64,
    cosio2: &mut f64,
    eccsq: &mut f64,
    omeosq: &mut f64,
    posq: &mut f64,
    rp: &mut f64,
    rteosq: &mut f64,
    sinio: &mut f64,
    gsto: &mut f64,
    no_unkozai: &mut f64,
) {
    /* --------------------- local variables ------------------------ */
    //double ak, d1, del, adel, po, x2o3;

    let mut del: f64;

    /*
    // sgp4fix use old way of finding gst
    let ds70: f64;
    //double ts70, tfrac, c1, thgr70, fk5r, c1p2p;
    let ts70: f64;
    let tfrac: f64;
    let c1: f64;
    let thgr70: f64;
    let fk5r: f64;
    let c1p2p: f64;
    */

    //const TWOPI: f64 = std::f64::consts::PI * 2.0;

    /* ----------------------- earth constants ---------------------- */
    // sgp4fix identify constants and allow alternate values
    // only xke and j2 are used here so pass them in directly
    // getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
    let x2o3: f64 = 2.0 / 3.0;

    /* ------------- calculate auxillary epoch quantities ---------- */
    *eccsq = ecco * ecco;
    *omeosq = 1.0 - *eccsq;
    *rteosq = f64::sqrt(*omeosq);
    *cosio = f64::cos(inclo);
    *cosio2 = *cosio * *cosio;

    /* ------------------ un-kozai the mean motion ----------------- */
    let ak: f64 = f64::powf(xke / no_kozai, x2o3);
    let d1: f64 = 0.75 * j2 * 3.0f64.mul_add(*cosio2, -1.0) / (*rteosq * *omeosq);
    del = d1 / (ak * ak);
    let adel: f64 = ak * del.mul_add(-(1.0 / 3.0 + 134.0 * del * del / 81.0), del.mul_add(-del, 1.0));
    del = d1 / (adel * adel);
    *no_unkozai = no_kozai / (1.0 + del);

    *ao = f64::powf(xke / *no_unkozai, x2o3);
    *sinio = f64::sin(inclo);
    let po: f64 = *ao * *omeosq;
    *con42 = 5.0f64.mul_add(-(*cosio2), 1.0);
    *con41 = -*con42 - *cosio2 - *cosio2;
    *ainv = 1.0 / *ao;
    *posq = po * po;
    *rp = *ao * (1.0 - ecco);
    *method = 'n';

    // sgp4fix modern approach to finding sidereal time
    //   if (opsmode == 'a')
    //      {
    // sgp4fix use old way of finding gst
    // count integer number of days from 0 jan 1970
    //ts70 = epoch - 7305.0;
    //ds70 = f64::floor(ts70 + 1.0e-8);
    //tfrac = ts70 - ds70;
    // find greenwich location at epoch
    //c1 = 1.72027916940703639e-2;
    //thgr70 = 1.7321343856509374;
    //fk5r = 5.07551419432269442e-15;
    //c1p2p = c1 + TWOPI;
    //let mut gsto1: f64 = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % TWOPI;
    //if gsto1 < 0.0 {
    //    gsto1 = gsto1 + TWOPI;
    //}
    //    else
    *gsto = gstime_sgp4(epoch + 2433281.5);

    //#include "debug5.cpp"
} // initl
