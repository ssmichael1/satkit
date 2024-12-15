use super::dpper::dpper;
use super::dscom::dscom;
use super::dsinit::dsinit;
use super::getgravconst::getgravconst;
use super::initl::initl;
use super::sgp4_lowlevel::sgp4_lowlevel;
use super::SatRec;
use super::{GravConst, OpsMode};

use std::f64::consts::PI;

/*-----------------------------------------------------------------------------
*
*                             procedure sgp4init
*
*  this procedure initializes variables for sgp4.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    opsmode     - mode of operation afspc or improved 'a', 'i'
*    whichconst  - which set of constants to use  72, 84
*    satn        - satellite number
*    bstar       - sgp4 type drag coefficient              kg/m2er
*    ecco        - eccentricity
*    epoch       - epoch time in days from jan 0, 1950. 0 hr
*    argpo       - argument of perigee (output if ds)
*    inclo       - inclination
*    mo          - mean anomaly (output if ds)
*    no          - mean motion
*    nodeo       - right ascension of ascending node
*
*  outputs       :
*    satrec      - common values for subsequent calls
*    return code - non-zero on error.
*                   1 - mean elements, ecc >= 1.0 or ecc < -0.001 or a < 0.95 er
*                   2 - mean motion less than 0.0
*                   3 - pert elements, ecc < 0.0  or  ecc > 1.0
*                   4 - semi-latus rectum < 0.0
*                   5 - epoch elements are sub-orbital
*                   6 - satellite has decayed
*
*  locals        :
*    cnodm  , snodm  , cosim  , sinim  , cosomm , sinomm
*    cc1sq  , cc2    , cc3
*    coef   , coef1
*    cosio4      -
*    day         -
*    dndt        -
*    em          - eccentricity
*    emsq        - eccentricity squared
*    eeta        -
*    etasq       -
*    gam         -
*    argpm       - argument of perigee
*    nodem       -
*    inclm       - inclination
*    mm          - mean anomaly
*    nm          - mean motion
*    perige      - perigee
*    pinvsq      -
*    psisq       -
*    qzms24      -
*    rtemsq      -
*    s1, s2, s3, s4, s5, s6, s7          -
*    sfour       -
*    ss1, ss2, ss3, ss4, ss5, ss6, ss7         -
*    sz1, sz2, sz3
*    sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33        -
*    tc          -
*    temp        -
*    temp1, temp2, temp3       -
*    tsi         -
*    xpidot      -
*    xhdot1      -
*    z1, z2, z3          -
*    z11, z12, z13, z21, z22, z23, z31, z32, z33         -
*
*  coupling      :
*    getgravconst-
*    initl       -
*    dscom       -
*    dpper       -
*    dsinit      -
*    sgp4        -
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
----------------------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn sgp4init(
    gravconst: GravConst,
    opsmode: OpsMode,
    _satn: &str,
    epoch: f64,
    xbstar: f64,
    xndot: f64,
    xnddot: f64,
    xecco: f64,
    xargpo: f64,
    xinclo: f64,
    xmo: f64,
    xno_kozai: f64,
    xnodeo: f64,
) -> Result<SatRec, i32> {
    /* --------------------- local variables ------------------------ */
    let mut ao: f64 = 0.0;
    let mut ainv: f64 = 0.0;
    let mut con42: f64 = 0.0;
    let mut cosio: f64 = 0.0;
    let mut sinio: f64 = 0.0;
    let mut cosio2: f64 = 0.0;
    let mut eccsq: f64 = 0.0;
    let mut omeosq: f64 = 0.0;
    let mut posq: f64 = 0.0;
    let mut rp: f64 = 0.0;
    let mut rteosq: f64 = 0.0;
    let mut cnodm: f64 = 0.0;
    let mut snodm: f64 = 0.0;
    let mut cosim: f64 = 0.0;
    let mut sinim: f64 = 0.0;
    let mut cosomm: f64 = 0.0;
    let mut sinomm: f64 = 0.0;
    let cc1sq: f64;
    let cc2: f64;
    let mut cc3: f64;
    let coef: f64;
    let coef1: f64;
    let cosio4: f64;
    let mut day: f64 = 0.0;
    let mut dndt: f64 = 0.0;
    let mut em: f64 = 0.0;
    let mut emsq: f64 = 0.0;
    let eeta: f64;
    let etasq: f64;
    let mut gam: f64 = 0.0;
    let mut argpm: f64;
    let mut nodem: f64;
    let mut inclm: f64;
    let mut mm: f64;
    let mut nm: f64 = 0.0;
    let perige: f64;
    let pinvsq: f64;
    let psisq: f64;
    let mut qzms24: f64;
    let mut rtemsq: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut s3: f64 = 0.0;
    let mut s4: f64 = 0.0;
    let mut s5: f64 = 0.0;
    let mut s6: f64 = 0.0;
    let mut s7: f64 = 0.0;
    let mut sfour: f64;
    let mut ss1: f64 = 0.0;
    let mut ss2: f64 = 0.0;
    let mut ss3: f64 = 0.0;
    let mut ss4: f64 = 0.0;
    let mut ss5: f64 = 0.0;
    let mut ss6: f64 = 0.0;
    let mut ss7: f64 = 0.0;
    let mut sz1: f64 = 0.0;
    let mut sz2: f64 = 0.0;
    let mut sz3: f64 = 0.0;
    let mut sz11: f64 = 0.0;
    let mut sz12: f64 = 0.0;
    let mut sz13: f64 = 0.0;
    let mut sz21: f64 = 0.0;
    let mut sz22: f64 = 0.0;
    let mut sz23: f64 = 0.0;
    let mut sz31: f64 = 0.0;
    let mut sz32: f64 = 0.0;
    let mut sz33: f64 = 0.0;
    let tc: f64;
    let temp: f64;
    let temp1: f64;
    let temp2: f64;
    let temp3: f64;
    let tsi: f64;
    let xpidot: f64;
    let xhdot1: f64;
    let mut z1: f64 = 0.0;
    let mut z2: f64 = 0.0;
    let mut z3: f64 = 0.0;
    let mut z11: f64 = 0.0;
    let mut z12: f64 = 0.0;
    let mut z13: f64 = 0.0;
    let mut z21: f64 = 0.0;
    let mut z22: f64 = 0.0;
    let mut z23: f64 = 0.0;
    let mut z31: f64 = 0.0;
    let mut z32: f64 = 0.0;
    let mut z33: f64 = 0.0;

    let delmotemp: f64;

    let qzms24temp: f64;

    /*
    double ao, ainv, con42, cosio, sinio, cosio2, eccsq,
        omeosq, posq, rp, rteosq,
        cnodm, snodm, cosim, sinim, cosomm, sinomm, cc1sq,
        cc2, cc3, coef, coef1, cosio4, day, dndt,
        em, emsq, eeta, etasq, gam, argpm, nodem,
        inclm, mm, nm, perige, pinvsq, psisq, qzms24,
        rtemsq, s1, s2, s3, s4, s5, s6,
        s7, sfour, ss1, ss2, ss3, ss4, ss5,
        ss6, ss7, sz1, sz2, sz3, sz11, sz12,
        sz13, sz21, sz22, sz23, sz31, sz32, sz33,
        tc, temp, temp1, temp2, temp3, tsi, xpidot,
        xhdot1, z1, z2, z3, z11, z12, z13,
        z21, z22, z23, z31, z32, z33,
        qzms2t, ss, x2o3, r[3], v[3],
        delmotemp, qzms2ttemp, qzms24temp;
    */
    /* ------------------------ initialization --------------------- */
    // sgp4fix divisor for divide by zero check on inclination
    // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    const TEMP4: f64 = 1.5e-12;

    let mut satrec = SatRec::new();

    /* ------------------------ earth constants ----------------------- */
    // sgp4fix identify constants and allow alternate values
    // this is now the only call for the constants
    getgravconst(
        gravconst,
        &mut satrec.tumin,
        &mut satrec.mus,
        &mut satrec.radiusearthkm,
        &mut satrec.xke,
        &mut satrec.j2,
        &mut satrec.j3,
        &mut satrec.j4,
        &mut satrec.j3oj2,
    );

    //-------------------------------------------------------------------------

    satrec.error = 0;
    satrec.operationmode = opsmode;
    // new alpha5 or 9-digit number
    /*
    #ifdef _MSC_VER
                        strcpy_s(satrec.satnum, 6 * sizeof(char), satn);
    #else
                        strcpy(satrec.satnum, satn);
    #endif
    */

    // sgp4fix - note the following variables are also passed directly via satrec.
    // it is possible to streamline the sgp4init call by deleting the "x"
    // variables, but the user would need to set the satrec.* values first. we
    // include the additional assignments in case twoline2rv is not used.
    satrec.bstar = xbstar;
    // sgp4fix allow additional parameters in the struct
    satrec.ndot = xndot;
    satrec.nddot = xnddot;
    satrec.ecco = xecco;
    satrec.argpo = xargpo;
    satrec.inclo = xinclo;
    satrec.mo = xmo;
    // sgp4fix rename variables to clarify which mean motion is intended
    satrec.no_kozai = xno_kozai;
    satrec.nodeo = xnodeo;

    // single averaged mean elements
    // satrec.am = satrec.em = satrec.im = satrec.om = satrec.mm = satrec.nm = 0.0;

    /* ------------------------ earth constants ----------------------- */
    // sgp4fix identify constants and allow alternate values no longer needed
    // getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );
    let ss: f64 = 78.0 / satrec.radiusearthkm + 1.0;
    // sgp4fix use multiply for speed instead of pow
    let qzms2ttemp: f64 = (120.0 - 78.0) / satrec.radiusearthkm;
    let qzms2t: f64 = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
    let x2o3: f64 = 2.0 / 3.0;

    satrec.init = 'y';
    satrec.t = 0.0;

    // sgp4fix remove satn as it is not needed in initl
    initl(
        satrec.xke,
        satrec.j2,
        satrec.ecco,
        epoch,
        satrec.inclo,
        satrec.no_kozai,
        satrec.operationmode,
        &mut satrec.method,
        &mut ainv,
        &mut ao,
        &mut satrec.con41,
        &mut con42,
        &mut cosio,
        &mut cosio2,
        &mut eccsq,
        &mut omeosq,
        &mut posq,
        &mut rp,
        &mut rteosq,
        &mut sinio,
        &mut satrec.gsto,
        &mut satrec.no_unkozai,
    );
    satrec.a = f64::powf(satrec.no_unkozai * satrec.tumin, -2.0 / 3.0);
    satrec.alta = satrec.a.mul_add(1.0 + satrec.ecco, -1.0);
    satrec.altp = satrec.a.mul_add(1.0 - satrec.ecco, -1.0);
    satrec.error = 0;

    // sgp4fix remove this check as it is unnecessary
    // the mrt check in sgp4 handles decaying satellite cases even if the starting
    // condition is below the surface of te earth
    //     if (rp < 1.0)
    //       {
    //         printf("# *** satn%d epoch elts sub-orbital ***\n", satn);
    //         satrec.error = 5;
    //       }

    if (omeosq >= 0.0) || (satrec.no_unkozai >= 0.0) {
        satrec.isimp = 0;
        if rp < (220.0 / satrec.radiusearthkm + 1.0) {
            satrec.isimp = 1;
        }
        sfour = ss;
        qzms24 = qzms2t;
        perige = (rp - 1.0) * satrec.radiusearthkm;

        /* - for perigees below 156 km, s and qoms2t are altered - */
        if perige < 156.0 {
            sfour = perige - 78.0;
            if perige < 98.0 {
                sfour = 20.0;
            }
            // sgp4fix use multiply for speed instead of pow
            qzms24temp = (120.0 - sfour) / satrec.radiusearthkm;
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
            sfour = sfour / satrec.radiusearthkm + 1.0;
        }
        pinvsq = 1.0 / posq;

        tsi = 1.0 / (ao - sfour);
        satrec.eta = ao * satrec.ecco * tsi;
        etasq = satrec.eta * satrec.eta;
        eeta = satrec.ecco * satrec.eta;
        psisq = f64::abs(1.0 - etasq);
        coef = qzms24 * f64::powf(tsi, 4.0);
        coef1 = coef / f64::powf(psisq, 3.5);
        cc2 = coef1
            * satrec.no_unkozai
            * ao.mul_add(eeta.mul_add(4.0 + etasq, 1.5f64.mul_add(etasq, 1.0)), 0.375 * satrec.j2 * tsi / psisq
                    * satrec.con41 * (3.0 * etasq).mul_add(8.0 + etasq, 8.0));
        satrec.cc1 = satrec.bstar * cc2;
        cc3 = 0.0;
        if satrec.ecco > 1.0e-4 {
            cc3 = -2.0 * coef * tsi * satrec.j3oj2 * satrec.no_unkozai * sinio / satrec.ecco;
        }
        satrec.x1mth2 = 1.0 - cosio2;
        satrec.cc4 = 2.0
            * satrec.no_unkozai
            * coef1
            * ao
            * omeosq
            * (satrec.j2 * tsi / (ao * psisq)).mul_add(-(-3.0 * satrec.con41).mul_add(etasq.mul_add(0.5f64.mul_add(-eeta, 1.5), 2.0f64.mul_add(-eeta, 1.0)), 0.75
                            * satrec.x1mth2
                            * 2.0f64.mul_add(etasq, -(eeta * (1.0 + etasq))) * f64::cos(2.0 * satrec.argpo)), satrec.eta.mul_add(0.5f64.mul_add(etasq, 2.0), satrec.ecco * 2.0f64.mul_add(etasq, 0.5)));
        satrec.cc5 = 2.0 * coef1 * ao * omeosq * eeta.mul_add(etasq, 2.75f64.mul_add(etasq + eeta, 1.0));
        cosio4 = cosio2 * cosio2;
        temp1 = 1.5 * satrec.j2 * pinvsq * satrec.no_unkozai;
        temp2 = 0.5 * temp1 * satrec.j2 * pinvsq;
        temp3 = -0.46875 * satrec.j4 * pinvsq * pinvsq * satrec.no_unkozai;
        satrec.mdot = (0.0625 * temp2 * rteosq).mul_add(137.0f64.mul_add(cosio4, 78.0f64.mul_add(-cosio2, 13.0)), (0.5 * temp1 * rteosq).mul_add(satrec.con41, satrec.no_unkozai));
        satrec.argpdot = temp3.mul_add(49.0f64.mul_add(cosio4, 36.0f64.mul_add(-cosio2, 3.0)), (-0.5 * temp1).mul_add(con42, 0.0625 * temp2 * 395.0f64.mul_add(cosio4, 114.0f64.mul_add(-cosio2, 7.0))));
        xhdot1 = -temp1 * cosio;
        satrec.nodedot = (0.5 * temp2).mul_add(19.0f64.mul_add(-cosio2, 4.0), 2.0 * temp3 * 7.0f64.mul_add(-cosio2, 3.0)).mul_add(cosio, xhdot1);
        xpidot = satrec.argpdot + satrec.nodedot;
        satrec.omgcof = satrec.bstar * cc3 * f64::cos(satrec.argpo);
        satrec.xmcof = 0.0;
        if satrec.ecco > 1.0e-4 {
            satrec.xmcof = -x2o3 * coef * satrec.bstar / eeta;
        }
        satrec.nodecf = 3.5 * omeosq * xhdot1 * satrec.cc1;
        satrec.t2cof = 1.5 * satrec.cc1;
        // sgp4fix for divide by zero with xinco = 180 deg
        if f64::abs(cosio + 1.0) > 1.5e-12 {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * 5.0f64.mul_add(cosio, 3.0) / (1.0 + cosio);
        } else {
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * 5.0f64.mul_add(cosio, 3.0) / TEMP4;
        }
        satrec.aycof = -0.5 * satrec.j3oj2 * sinio;
        // sgp4fix use multiply for speed instead of pow
        delmotemp = satrec.eta.mul_add(f64::cos(satrec.mo), 1.0);
        satrec.delmo = delmotemp * delmotemp * delmotemp;
        satrec.sinmao = f64::sin(satrec.mo);
        satrec.x7thm1 = 7.0f64.mul_add(cosio2, -1.0);

        /* --------------- deep space initialization ------------- */
        if (2.0 * PI / satrec.no_unkozai) >= 225.0 {
            satrec.method = 'd';
            satrec.isimp = 1;
            tc = 0.0;
            inclm = satrec.inclo;

            dscom(
                epoch,
                satrec.ecco,
                satrec.argpo,
                tc,
                satrec.inclo,
                satrec.nodeo,
                satrec.no_unkozai,
                &mut snodm,
                &mut cnodm,
                &mut sinim,
                &mut cosim,
                &mut sinomm,
                &mut cosomm,
                &mut day,
                &mut satrec.e3,
                &mut satrec.ee2,
                &mut em,
                &mut emsq,
                &mut gam,
                &mut satrec.peo,
                &mut satrec.pgho,
                &mut satrec.pho,
                &mut satrec.pinco,
                &mut satrec.plo,
                &mut rtemsq,
                &mut satrec.se2,
                &mut satrec.se3,
                &mut satrec.sgh2,
                &mut satrec.sgh3,
                &mut satrec.sgh4,
                &mut satrec.sh2,
                &mut satrec.sh3,
                &mut satrec.si2,
                &mut satrec.si3,
                &mut satrec.sl2,
                &mut satrec.sl3,
                &mut satrec.sl4,
                &mut s1,
                &mut s2,
                &mut s3,
                &mut s4,
                &mut s5,
                &mut s6,
                &mut s7,
                &mut ss1,
                &mut ss2,
                &mut ss3,
                &mut ss4,
                &mut ss5,
                &mut ss6,
                &mut ss7,
                &mut sz1,
                &mut sz2,
                &mut sz3,
                &mut sz11,
                &mut sz12,
                &mut sz13,
                &mut sz21,
                &mut sz22,
                &mut sz23,
                &mut sz31,
                &mut sz32,
                &mut sz33,
                &mut satrec.xgh2,
                &mut satrec.xgh3,
                &mut satrec.xgh4,
                &mut satrec.xh2,
                &mut satrec.xh3,
                &mut satrec.xi2,
                &mut satrec.xi3,
                &mut satrec.xl2,
                &mut satrec.xl3,
                &mut satrec.xl4,
                &mut nm,
                &mut z1,
                &mut z2,
                &mut z3,
                &mut z11,
                &mut z12,
                &mut z13,
                &mut z21,
                &mut z22,
                &mut z23,
                &mut z31,
                &mut z32,
                &mut z33,
                &mut satrec.zmol,
                &mut satrec.zmos,
            );
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
                inclm,
                satrec.init,
                &mut satrec.ecco,
                &mut satrec.inclo,
                &mut satrec.nodeo,
                &mut satrec.argpo,
                &mut satrec.mo,
                satrec.operationmode,
            );

            argpm = 0.0;
            nodem = 0.0;
            mm = 0.0;

            dsinit(
                satrec.xke,
                cosim,
                emsq,
                satrec.argpo,
                s1,
                s2,
                s3,
                s4,
                s5,
                sinim,
                ss1,
                ss2,
                ss3,
                ss4,
                ss5,
                sz1,
                sz3,
                sz11,
                sz13,
                sz21,
                sz23,
                sz31,
                sz33,
                satrec.t,
                tc,
                satrec.gsto,
                satrec.mo,
                satrec.mdot,
                satrec.no_unkozai,
                satrec.nodeo,
                satrec.nodedot,
                xpidot,
                z1,
                z3,
                z11,
                z13,
                z21,
                z23,
                z31,
                z33,
                satrec.ecco,
                eccsq,
                &mut em,
                &mut argpm,
                &mut inclm,
                &mut mm,
                &mut nm,
                &mut nodem,
                &mut satrec.irez,
                &mut satrec.atime,
                &mut satrec.d2201,
                &mut satrec.d2211,
                &mut satrec.d3210,
                &mut satrec.d3222,
                &mut satrec.d4410,
                &mut satrec.d4422,
                &mut satrec.d5220,
                &mut satrec.d5232,
                &mut satrec.d5421,
                &mut satrec.d5433,
                &mut satrec.dedt,
                &mut satrec.didt,
                &mut satrec.dmdt,
                &mut dndt,
                &mut satrec.dnodt,
                &mut satrec.domdt,
                &mut satrec.del1,
                &mut satrec.del2,
                &mut satrec.del3,
                &mut satrec.xfact,
                &mut satrec.xlamo,
                &mut satrec.xli,
                &mut satrec.xni,
            );
        }

        /* ----------- set variables if not deep space ----------- */
        if satrec.isimp != 1 {
            cc1sq = satrec.cc1 * satrec.cc1;
            satrec.d2 = 4.0 * ao * tsi * cc1sq;
            temp = satrec.d2 * tsi * satrec.cc1 / 3.0;
            satrec.d3 = 17.0f64.mul_add(ao, sfour) * temp;
            satrec.d4 = 0.5 * temp * ao * tsi * 221.0f64.mul_add(ao, 31.0 * sfour) * satrec.cc1;
            satrec.t3cof = 2.0f64.mul_add(cc1sq, satrec.d2);
            satrec.t4cof =
                0.25 * 3.0f64.mul_add(satrec.d3, satrec.cc1 * 12.0f64.mul_add(satrec.d2, 10.0 * cc1sq));
            satrec.t5cof = 0.2
                * (15.0 * cc1sq).mul_add(2.0f64.mul_add(satrec.d2, cc1sq), (6.0 * satrec.d2).mul_add(satrec.d2, 3.0f64.mul_add(satrec.d4, 12.0 * satrec.cc1 * satrec.d3)));
        }
    } // if omeosq = 0 ...

    /* finally propogate to zero epoch to initialize all others. */
    // sgp4fix take out check to let satellites process until they are actually below earth surface
    //       if(satrec.error == 0)
    //sgp4(satrec, 0.0, r, v);

    sgp4_lowlevel(&mut satrec, 0.0)?;

    satrec.init = 'n';

    //#include "debug6.cpp"
    //sgp4fix return boolean. satrec.error contains any error codes
    Ok(satrec)
} // sgp4init
