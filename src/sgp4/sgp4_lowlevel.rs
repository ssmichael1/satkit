/*-----------------------------------------------------------------------------
*
*                             procedure sgp4
*
*  this procedure is the sgp4 prediction model from space command. this is an
*    updated and combined version of sgp4 and sdp4, which were originally
*    published separately in spacetrack report #3. this version follows the
*    methodology from the aiaa paper (2006) describing the history and
*    development of the code.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    satrec	 - initialised structure from sgp4init() call.
*    tsince	 - time since epoch (minutes)
*
*  outputs       :
*    r           - position vector                     km
*    v           - velocity                            km/sec
*  return code - non-zero on error.
*                   1 - mean elements, ecc >= 1.0 or ecc < -0.001 or a < 0.95 er
*                   2 - mean motion less than 0.0
*                   3 - pert elements, ecc < 0.0  or  ecc > 1.0
*                   4 - semi-latus rectum < 0.0
*                   5 - epoch elements are sub-orbital
*                   6 - satellite has decayed
*
*  locals        :
*    am          -
*    axnl, aynl        -
*    betal       -
*    cosim   , sinim   , cosomm  , sinomm  , cnod    , snod    , cos2u   ,
*    sin2u   , coseo1  , sineo1  , cosi    , sini    , cosip   , sinip   ,
*    cosisq  , cossu   , sinsu   , cosu    , sinu
*    delm        -
*    delomg      -
*    dndt        -
*    eccm        -
*    emsq        -
*    ecose       -
*    el2         -
*    eo1         -
*    eccp        -
*    esine       -
*    argpm       -
*    argpp       -
*    omgadf      -c
*    pl          -
*    r           -
*    rtemsq      -
*    rdotl       -
*    rl          -
*    rvdot       -
*    rvdotl      -
*    su          -
*    t2  , t3   , t4    , tc
*    tem5, temp , temp1 , temp2  , tempa  , tempe  , templ
*    u   , ux   , uy    , uz     , vx     , vy     , vz
*    inclm       - inclination
*    mm          - mean anomaly
*    nm          - mean motion
*    nodem       - right asc of ascending node
*    xinc        -
*    xincp       -
*    xl          -
*    xlm         -
*    mp          -
*    xmdf        -
*    xmx         -
*    xmy         -
*    nodedf      -
*    xnode       -
*    nodep       -
*    np          -
*
*  coupling      :
*    getgravconst- no longer used. Variables are conatined within satrec
*    dpper
*    dpspace
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
----------------------------------------------------------------------------*/
use super::dpper::dpper;
use super::dspace::dspace;
use super::satrec::SatRec;

use std::f64::consts::PI;

pub fn sgp4_lowlevel(satrec: &mut SatRec, tsince: f64) -> Result<([f64; 3], [f64; 3]), i32> {
    
    
    
    
    
    
    
    let mut coseo1: f64 = 0.0;
    
    let mut cosip: f64;
    let cosisq: f64;
    
    
    let delm: f64;
    let delomg: f64;
    let mut em: f64;
    
    
    
    let mut eo1: f64;
    let mut ep: f64;
    
    let mut argpm: f64;
    let mut argpp: f64;
    
    
    
    
    
    
    
    
    
    
    let mut sineo1: f64 = 0.0;
    
    let mut sinip: f64;
    
    
    
    let mut su: f64;
    
    let t3: f64;
    let t4: f64;
    let mut tem5: f64;
    let mut temp: f64;
    
    
    let mut tempa: f64;
    let mut tempe: f64;
    let mut templ: f64;
    
    
    
    
    
    
    
    let mut inclm: f64;
    let mut mm: f64;
    let mut nm: f64;
    let mut nodem: f64;
    
    let mut xincp: f64;
    
    let mut xlm: f64;
    let mut mp: f64;
    
    
    
    
    
    let mut nodep: f64;
    let tc: f64;
    let mut dndt: f64 = 0.0;
    
    let delmtemp: f64;

    let mut ktr: i32;

    /*
    double am, axnl, aynl, betal, cosim, cnod,
        cos2u, coseo1, cosi, cosip, cosisq, cossu, cosu,
        delm, delomg, em, emsq, ecose, el2, eo1,
        ep, esine, argpm, argpp, argpdf, pl, mrt = 0.0,
        mvt, rdotl, rl, rvdot, rvdotl, sinim,
        sin2u, sineo1, sini, sinip, sinsu, sinu,
        snod, su, t2, t3, t4, tem5, temp,
        temp1, temp2, tempa, tempe, templ, u, ux,
        uy, uz, vx, vy, vz, inclm, mm,
        nm, nodem, xinc, xincp, xl, xlm, mp,
        xmdf, xmx, xmy, nodedf, xnode, nodep, tc, dndt,
        twopi, x2o3, vkmpersec, delmtemp;
    int ktr;
    */

    /* ------------------ set mathematical constants --------------- */
    // sgp4fix divisor for divide by zero check on inclination
    // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    const TEMP4: f64 = 1.5e-12;
    const TWOPI: f64 = 2.0 * std::f64::consts::PI;
    const X2O3: f64 = 2.0 / 3.0;
    // sgp4fix identify constants and allow alternate values
    // getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
    let vkmpersec: f64 = satrec.radiusearthkm * satrec.xke / 60.0;

    /* --------------------- clear sgp4 error flag ----------------- */
    satrec.t = tsince;
    satrec.error = 0;

    /* ------- update for secular gravity and atmospheric drag ----- */
    let xmdf: f64 = satrec.mo + satrec.mdot * satrec.t;
    let argpdf: f64 = satrec.argpo + satrec.argpdot * satrec.t;
    let nodedf: f64 = satrec.nodeo + satrec.nodedot * satrec.t;
    argpm = argpdf;
    mm = xmdf;
    let t2: f64 = satrec.t * satrec.t;
    nodem = nodedf + satrec.nodecf * t2;
    tempa = 1.0 - satrec.cc1 * satrec.t;
    tempe = satrec.bstar * satrec.cc4 * satrec.t;
    templ = satrec.t2cof * t2;

    if satrec.isimp != 1 {
        delomg = satrec.omgcof * satrec.t;
        // sgp4fix use mutliply for speed instead of pow
        delmtemp = 1.0 + satrec.eta * f64::cos(xmdf);
        delm = satrec.xmcof * (delmtemp * delmtemp * delmtemp - satrec.delmo);
        temp = delomg + delm;
        mm = xmdf + temp;
        argpm = argpdf - temp;
        t3 = t2 * satrec.t;
        t4 = t3 * satrec.t;
        tempa = tempa - satrec.d2 * t2 - satrec.d3 * t3 - satrec.d4 * t4;
        tempe += satrec.bstar * satrec.cc5 * (f64::sin(mm) - satrec.sinmao);
        templ = templ + satrec.t3cof * t3 + t4 * (satrec.t4cof + satrec.t * satrec.t5cof);
    }

    nm = satrec.no_unkozai;
    em = satrec.ecco;
    inclm = satrec.inclo;
    if satrec.method == 'd' {
        tc = satrec.t;
        dspace(
            satrec.irez,
            satrec.d2201,
            satrec.d2211,
            satrec.d3210,
            satrec.d3222,
            satrec.d4410,
            satrec.d4422,
            satrec.d5220,
            satrec.d5232,
            satrec.d5421,
            satrec.d5433,
            satrec.dedt,
            satrec.del1,
            satrec.del2,
            satrec.del3,
            satrec.didt,
            satrec.dmdt,
            satrec.dnodt,
            satrec.domdt,
            satrec.argpo,
            satrec.argpdot,
            satrec.t,
            tc,
            satrec.gsto,
            satrec.xfact,
            satrec.xlamo,
            satrec.no_unkozai,
            &mut satrec.atime,
            &mut em,
            &mut argpm,
            &mut inclm,
            &mut satrec.xli,
            &mut mm,
            &mut satrec.xni,
            &mut nodem,
            &mut dndt,
            &mut nm,
        );
    } // if method = d

    if nm <= 0.0 {
        //         printf("# error nm %f\n", nm);
        satrec.error = 2;
        // sgp4fix add return
        return Err(satrec.error);
    }
    let am: f64 = f64::powf(satrec.xke / nm, X2O3) * tempa * tempa;
    nm = satrec.xke / f64::powf(am, 1.5);
    em -= tempe;

    // fix tolerance for error recognition
    // sgp4fix am is fixed from the previous nm check
    if !(-0.001..1.0).contains(&em) {
        //         printf("# error em %f\n", em);
        satrec.error = 1;
        // sgp4fix to return if there is an error in eccentricity
        return Err(satrec.error);
    }
    // sgp4fix fix tolerance to avoid a divide by zero
    if em < 1.0e-6 {
        em = 1.0e-6;
    }
    mm += satrec.no_unkozai * templ;
    xlm = mm + argpm + nodem;
    let _emsq: f64 = em * em;
    //temp = 1.0 - emsq;

    nodem %= TWOPI; //fmod(nodem, twopi);
    argpm %= TWOPI; //fmod(argpm, twopi);
    xlm %= TWOPI; //fmod(xlm, twopi);
    mm = (xlm - argpm - nodem) % TWOPI;

    // sgp4fix recover singly averaged mean elements
    satrec.am = am;
    satrec.em = em;
    satrec.im = inclm;
    satrec.om = nodem;
    satrec.om = argpm;
    satrec.mm = mm;
    satrec.nm = nm;

    /* ----------------- compute extra mean quantities ------------- */
    let sinim: f64 = f64::sin(inclm);
    let cosim: f64 = f64::cos(inclm);

    /* -------------------- add lunar-solar periodics -------------- */
    ep = em;
    xincp = inclm;
    argpp = argpm;
    nodep = nodem;
    mp = mm;
    sinip = sinim;
    cosip = cosim;
    if satrec.method == 'd' {
        dpper(
            satrec.e3,
            satrec.ee2,
            satrec.peo,
            satrec.pgho,
            satrec.pho,
            satrec.pinco,
            satrec.plo,
            satrec.se2,
            satrec.se3,
            satrec.sgh2,
            satrec.sgh3,
            satrec.sgh4,
            satrec.sh2,
            satrec.sh3,
            satrec.si2,
            satrec.si3,
            satrec.sl2,
            satrec.sl3,
            satrec.sl4,
            satrec.t,
            satrec.xgh2,
            satrec.xgh3,
            satrec.xgh4,
            satrec.xh2,
            satrec.xh3,
            satrec.xi2,
            satrec.xi3,
            satrec.xl2,
            satrec.xl3,
            satrec.xl4,
            satrec.zmol,
            satrec.zmos,
            satrec.inclo,
            'n',
            &mut ep,
            &mut xincp,
            &mut nodep,
            &mut argpp,
            &mut mp,
            satrec.operationmode,
        );
        if xincp < 0.0 {
            xincp = -xincp;
            nodep += PI;
            argpp -= PI;
        }
        if !(0.0..=1.0).contains(&ep) {
            //            printf("# error ep %f\n", ep);
            satrec.error = 3;
            // sgp4fix add return
            return Err(satrec.error);
        }
    } // if method = d

    /* -------------------- long period periodics ------------------ */
    if satrec.method == 'd' {
        sinip = f64::sin(xincp);
        cosip = f64::cos(xincp);
        satrec.aycof = -0.5 * satrec.j3oj2 * sinip;
        // sgp4fix for divide by zero for xincp = 180 deg
        if f64::abs(cosip + 1.0) > 1.5e-12 {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
        } else {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / TEMP4;
        }
    }
    let axnl: f64 = ep * f64::cos(argpp);
    temp = 1.0 / (am * (1.0 - ep * ep));
    let aynl: f64 = ep * f64::sin(argpp) + temp * satrec.aycof;
    let xl: f64 = mp + argpp + nodep + temp * satrec.xlcof * axnl;

    /* --------------------- solve kepler's equation --------------- */
    let u: f64 = (xl - nodep) % TWOPI;
    eo1 = u;
    tem5 = 9999.9;
    ktr = 1;
    //   sgp4fix for kepler iteration
    //   the following iteration needs better limits on corrections
    while (f64::abs(tem5) >= 1.0e-12) && (ktr <= 10) {
        sineo1 = f64::sin(eo1);
        coseo1 = f64::cos(eo1);
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
        if f64::abs(tem5) >= 0.95 {
            if tem5 > 0.0 {
                tem5 = 0.95;
            } else {
                tem5 = -0.95;
            }
            //tem5 = tem5 > 0.0 ? 0.95 : -0.95;
        }
        eo1 += tem5;
        ktr += 1;
    }

    /* ------------- short period preliminary quantities ----------- */
    let ecose: f64 = axnl * coseo1 + aynl * sineo1;
    let esine: f64 = axnl * sineo1 - aynl * coseo1;
    let el2: f64 = axnl * axnl + aynl * aynl;
    let pl: f64 = am * (1.0 - el2);

    if pl < 0.0 {
        //         printf("# error pl %f\n", pl);
        satrec.error = 4;
        // sgp4fix add return
        return Err(satrec.error);
    }

    let rl: f64 = am * (1.0 - ecose);
    let rdotl: f64 = f64::sqrt(am) * esine / rl;
    let rvdotl: f64 = f64::sqrt(pl) / rl;
    let betal: f64 = f64::sqrt(1.0 - el2);
    temp = esine / (1.0 + betal);
    let sinu: f64 = am / rl * (sineo1 - aynl - axnl * temp);
    let cosu: f64 = am / rl * (coseo1 - axnl + aynl * temp);
    su = f64::atan2(sinu, cosu);
    let sin2u: f64 = (cosu + cosu) * sinu;
    let cos2u: f64 = 1.0 - 2.0 * sinu * sinu;
    temp = 1.0 / pl;
    let temp1: f64 = 0.5 * satrec.j2 * temp;
    let temp2: f64 = temp1 * temp;

    /* -------------- update for short period periodics ------------ */
    if satrec.method == 'd' {
        cosisq = cosip * cosip;
        satrec.con41 = 3.0 * cosisq - 1.0;
        satrec.x1mth2 = 1.0 - cosisq;
        satrec.x7thm1 = 7.0 * cosisq - 1.0;
    }
    let mrt: f64 = rl * (1.0 - 1.5 * temp2 * betal * satrec.con41) + 0.5 * temp1 * satrec.x1mth2 * cos2u;
    su -= 0.25 * temp2 * satrec.x7thm1 * sin2u;
    let xnode: f64 = nodep + 1.5 * temp2 * cosip * sin2u;
    let xinc: f64 = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    let mvt: f64 = rdotl - nm * temp1 * satrec.x1mth2 * sin2u / satrec.xke;
    let rvdot: f64 = rvdotl + nm * temp1 * (satrec.x1mth2 * cos2u + 1.5 * satrec.con41) / satrec.xke;

    /* --------------------- orientation vectors ------------------- */
    let sinsu: f64 = f64::sin(su);
    let cossu: f64 = f64::cos(su);
    let snod: f64 = f64::sin(xnode);
    let cnod: f64 = f64::cos(xnode);
    let sini: f64 = f64::sin(xinc);
    let cosi: f64 = f64::cos(xinc);
    let xmx: f64 = -snod * cosi;
    let xmy: f64 = cnod * cosi;
    let ux: f64 = xmx * sinsu + cnod * cossu;
    let uy: f64 = xmy * sinsu + snod * cossu;
    let uz: f64 = sini * sinsu;
    let vx: f64 = xmx * cossu - cnod * sinsu;
    let vy: f64 = xmy * cossu - snod * sinsu;
    let vz: f64 = sini * cossu;

    // sgp4fix for decaying satellites
    if mrt < 1.0 {
        //         printf("# decay condition %11.6f \n",mrt);
        satrec.error = 6;
        return Err(satrec.error);
    }

    /* --------- position and velocity (in km and km/sec) ---------- */
    //r[0] = (mrt * ux) * satrec.radiusearthkm;
    //r[1] = (mrt * uy) * satrec.radiusearthkm;
    //r[2] = (mrt * uz) * satrec.radiusearthkm;
    //v[0] = (mvt * ux + rvdot * vx) * vkmpersec;
    //v[1] = (mvt * uy + rvdot * vy) * vkmpersec;
    //v[2] = (mvt * uz + rvdot * vz) * vkmpersec;

    Ok((
        [
            (mrt * ux) * satrec.radiusearthkm,
            (mrt * uy) * satrec.radiusearthkm,
            (mrt * uz) * satrec.radiusearthkm,
        ],
        [
            (mvt * ux + rvdot * vx) * vkmpersec,
            (mvt * uy + rvdot * vy) * vkmpersec,
            (mvt * uz + rvdot * vz) * vkmpersec,
        ],
    ))

    //#include "debug7.cpp"
} // sgp4
