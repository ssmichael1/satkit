use super::OpsMode;
/* -----------------------------------------------------------------------------
*
*                           procedure dpper
*
*  this procedure provides deep space long period periodic contributions
*    to the mean elements.  by design, these periodics are zero at epoch.
*    this used to be dscom which included initialization, but it's really a
*    recurring function.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    e3          -
*    ee2         -
*    peo         -
*    pgho        -
*    pho         -
*    pinco       -
*    plo         -
*    se2 , se3 , sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4 -
*    t           -
*    xh2, xh3, xi2, xi3, xl2, xl3, xl4 -
*    zmol        -
*    zmos        -
*    ep          - eccentricity                           0.0 - 1.0
*    inclo       - inclination - needed for lyddane modification
*    nodep       - right ascension of ascending node
*    argpp       - argument of perigee
*    mp          - mean anomaly
*
*  outputs       :
*    ep          - eccentricity                           0.0 - 1.0
*    inclp       - inclination
*    nodep        - right ascension of ascending node
*    argpp       - argument of perigee
*    mp          - mean anomaly
*
*  locals        :
*    alfdp       -
*    betdp       -
*    cosip  , sinip  , cosop  , sinop  ,
*    dalf        -
*    dbet        -
*    dls         -
*    f2, f3      -
*    pe          -
*    pgh         -
*    ph          -
*    pinc        -
*    pl          -
*    sel   , ses   , sghl  , sghs  , shl   , shs   , sil   , sinzf , sis   ,
*    sll   , sls
*    xls         -
*    xnoh        -
*    zf          -
*    zm          -
*
*  coupling      :
*    none.
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
----------------------------------------------------------------------------*/

pub fn dpper(
    e3: f64,
    ee2: f64,
    peo: f64,
    pgho: f64,
    pho: f64,
    pinco: f64,
    plo: f64,
    se2: f64,
    se3: f64,
    sgh2: f64,
    sgh3: f64,
    sgh4: f64,
    sh2: f64,
    sh3: f64,
    si2: f64,
    si3: f64,
    sl2: f64,
    sl3: f64,
    sl4: f64,
    t: f64,
    xgh2: f64,
    xgh3: f64,
    xgh4: f64,
    xh2: f64,
    xh3: f64,
    xi2: f64,
    xi3: f64,
    xl2: f64,
    xl3: f64,
    xl4: f64,
    zmol: f64,
    zmos: f64,
    _inclo: f64,
    init: char,
    ep: &mut f64,
    inclp: &mut f64,
    nodep: &mut f64,
    argpp: &mut f64,
    mp: &mut f64,
    opsmode: OpsMode,
) {
    /* --------------------- local variables ------------------------ */
    use std::f64::consts::PI;
    const TWOPI: f64 = PI * 2.0;
    /*
    double alfdp, betdp, cosip, cosop, dalf, dbet, dls,
        f2, f3, pe, pgh, ph, pinc, pl,
        sel, ses, sghl, sghs, shll, shs, sil,
        sinip, sinop, sinzf, sis, sll, sls, xls,
        xnoh, zf, zm, zel, zes, znl, zns;
    */
    let mut alfdp: f64;
    let mut betdp: f64;
    let cosip: f64;
    let cosop: f64;
    let dalf: f64;
    let dbet: f64;
    let dls: f64;
    let mut f2: f64;
    let mut f3: f64;
    let mut pe: f64;
    let mut pgh: f64;
    let mut ph: f64;
    let mut pinc: f64;
    let mut pl: f64;
    let sel: f64;
    let ses: f64;
    let sghl: f64;
    let sghs: f64;
    let shll: f64;
    let shs: f64;
    let sil: f64;
    let sinip: f64;
    let sinop: f64;
    let mut sinzf: f64;
    let sis: f64;
    let sll: f64;
    let sls: f64;
    let mut xls: f64;
    let xnoh: f64;
    let mut zf: f64;
    let mut zm: f64;

    /* ---------------------- constants ----------------------------- */
    const ZNS: f64 = 1.19459e-5;
    const ZES: f64 = 0.01675;
    const ZNL: f64 = 1.5835218e-4;
    const ZEL: f64 = 0.05490;

    /* --------------- calculate time varying periodics ----------- */
    zm = zmos + ZNS * t;
    // be sure that the initial call has time set to zero
    if init == 'y' {
        zm = zmos;
    }
    zf = zm + 2.0 * ZES * f64::sin(zm);
    sinzf = f64::sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * f64::cos(zf);
    ses = se2 * f2 + se3 * f3;
    sis = si2 * f2 + si3 * f3;
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf;
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf;
    shs = sh2 * f2 + sh3 * f3;
    zm = zmol + ZNL * t;
    if init == 'y' {
        zm = zmol;
    }
    zf = zm + 2.0 * ZEL * f64::sin(zm);
    sinzf = f64::sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * f64::cos(zf);
    sel = ee2 * f2 + e3 * f3;
    sil = xi2 * f2 + xi3 * f3;
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf;
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf;
    shll = xh2 * f2 + xh3 * f3;
    pe = ses + sel;
    pinc = sis + sil;
    pl = sls + sll;
    pgh = sghs + sghl;
    ph = shs + shll;

    if init == 'n' {
        pe = pe - peo;
        pinc = pinc - pinco;
        pl = pl - plo;
        pgh = pgh - pgho;
        ph = ph - pho;
        *inclp = *inclp + pinc;
        *ep = *ep + pe;
        sinip = f64::sin(*inclp);
        cosip = f64::cos(*inclp);

        /* ----------------- apply periodics directly ------------ */
        //  sgp4fix for lyddane choice
        //  strn3 used original inclination - this is technically feasible
        //  gsfc used perturbed inclination - also technically feasible
        //  probably best to readjust the 0.2 limit value and limit discontinuity
        //  0.2 rad = 11.45916 deg
        //  use next line for original strn3 approach and original inclination
        //  if (inclo >= 0.2)
        //  use next line for gsfc version and perturbed inclination
        if *inclp >= 0.2 {
            ph = ph / sinip;
            pgh = pgh - cosip * ph;
            *argpp = *argpp + pgh;
            *nodep = *nodep + ph;
            *mp = *mp + pl;
        } else {
            /* ---- apply periodics with lyddane modification ---- */
            sinop = f64::sin(*nodep);
            cosop = f64::cos(*nodep);
            alfdp = sinip * sinop;
            betdp = sinip * cosop;
            dalf = ph * cosop + pinc * cosip * sinop;
            dbet = -ph * sinop + pinc * cosip * cosop;
            alfdp = alfdp + dalf;
            betdp = betdp + dbet;
            *nodep = *nodep % TWOPI;
            //  sgp4fix for afspc written intrinsic functions
            // nodep used without a trigonometric function ahead
            if *nodep < 0.0 && opsmode == OpsMode::AFSPC {
                *nodep = *nodep + TWOPI;
            }
            xls = *mp + *argpp + cosip * *nodep;
            dls = pl + pgh - pinc * *nodep * sinip;
            xls = xls + dls;
            xnoh = *nodep;
            *nodep = f64::atan2(alfdp, betdp);
            //  sgp4fix for afspc written intrinsic functions
            // nodep used without a trigonometric function ahead
            if (*nodep < 0.0) && (opsmode == OpsMode::AFSPC) {
                *nodep = *nodep + TWOPI;
            }
            if f64::abs(xnoh - *nodep) > PI {
                if *nodep < xnoh {
                    *nodep = *nodep + TWOPI;
                } else {
                    *nodep = *nodep - TWOPI;
                }
            }
            *mp = *mp + pl;
            *argpp = xls - *mp - cosip * *nodep;
        }
    } // if init == 'n'

    //#include "debug1.cpp"
} // dpper
