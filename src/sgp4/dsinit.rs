use std::f64::consts::PI;

/*-----------------------------------------------------------------------------
*
*                           procedure dsinit
*
*  this procedure provides deep space contributions to mean motion dot due
*    to geopotential resonance with half day and one day orbits.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    xke         - reciprocal of tumin
*    cosim, sinim-
*    emsq        - eccentricity squared
*    argpo       - argument of perigee
*    s1, s2, s3, s4, s5      -
*    ss1, ss2, ss3, ss4, ss5 -
*    sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33 -
*    t           - time
*    tc          -
*    gsto        - greenwich sidereal time                   rad
*    mo          - mean anomaly
*    mdot        - mean anomaly dot (rate)
*    no          - mean motion
*    nodeo       - right ascension of ascending node
*    nodedot     - right ascension of ascending node dot (rate)
*    xpidot      -
*    z1, z3, z11, z13, z21, z23, z31, z33 -
*    eccm        - eccentricity
*    argpm       - argument of perigee
*    inclm       - inclination
*    mm          - mean anomaly
*    xn          - mean motion
*    nodem       - right ascension of ascending node
*
*  outputs       :
*    em          - eccentricity
*    argpm       - argument of perigee
*    inclm       - inclination
*    mm          - mean anomaly
*    nm          - mean motion
*    nodem       - right ascension of ascending node
*    irez        - flag for resonance           0-none, 1-one day, 2-half day
*    atime       -
*    d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433    -
*    dedt        -
*    didt        -
*    dmdt        -
*    dndt        -
*    dnodt       -
*    domdt       -
*    del1, del2, del3        -
*    ses  , sghl , sghs , sgs  , shl  , shs  , sis  , sls
*    theta       -
*    xfact       -
*    xlamo       -
*    xli         -
*    xni
*
*  locals        :
*    ainv2       -
*    aonv        -
*    cosisq      -
*    eoc         -
*    f220, f221, f311, f321, f322, f330, f441, f442, f522, f523, f542, f543  -
*    g200, g201, g211, g300, g310, g322, g410, g422, g520, g521, g532, g533  -
*    sini2       -
*    temp        -
*    temp1       -
*    theta       -
*    xno2        -
*
*  coupling      :
*    getgravconst- no longer used
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
----------------------------------------------------------------------------*/

#[allow(clippy::too_many_arguments)]
pub fn dsinit(
    // sgp4fix just send in xke as a constant and eliminate getgravconst call
    // gravconsttype whichconst,
    xke: f64,
    cosim: f64,
    mut emsq: f64,
    argpo: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
    sinim: f64,
    ss1: f64,
    ss2: f64,
    ss3: f64,
    ss4: f64,
    ss5: f64,
    sz1: f64,
    sz3: f64,
    sz11: f64,
    sz13: f64,
    sz21: f64,
    sz23: f64,
    sz31: f64,
    sz33: f64,
    t: f64,
    tc: f64,
    gsto: f64,
    mo: f64,
    mdot: f64,
    no: f64,
    nodeo: f64,
    nodedot: f64,
    xpidot: f64,
    z1: f64,
    z3: f64,
    z11: f64,
    z13: f64,
    z21: f64,
    z23: f64,
    z31: f64,
    z33: f64,
    ecco: f64,
    eccsq: f64,
    em: &mut f64,
    argpm: &mut f64,
    inclm: &mut f64,
    mm: &mut f64,
    nm: &mut f64,
    nodem: &mut f64,
    irez: &mut i32,
    atime: &mut f64,
    d2201: &mut f64,
    d2211: &mut f64,
    d3210: &mut f64,
    d3222: &mut f64,
    d4410: &mut f64,
    d4422: &mut f64,
    d5220: &mut f64,
    d5232: &mut f64,
    d5421: &mut f64,
    d5433: &mut f64,
    dedt: &mut f64,
    didt: &mut f64,
    dmdt: &mut f64,
    dndt: &mut f64,
    dnodt: &mut f64,
    domdt: &mut f64,
    del1: &mut f64,
    del2: &mut f64,
    del3: &mut f64,
    xfact: &mut f64,
    xlamo: &mut f64,
    xli: &mut f64,
    xni: &mut f64,
) {
    /* --------------------- local variables ------------------------ */
    const TWOPI: f64 = 2.0 * PI;

    let mut f220: f64;
    let f221: f64;
    let f311: f64;
    let f321: f64;
    let f322: f64;
    let mut f330: f64;
    let f441: f64;
    let f442: f64;
    let f522: f64;
    let f523: f64;
    let f542: f64;
    let f543: f64;
    let g200: f64;
    let g201: f64;
    let g211: f64;
    let g300: f64;
    let mut g310: f64;
    let g322: f64;
    let g410: f64;
    let g422: f64;
    let g520: f64;
    let g521: f64;
    let g532: f64;
    let g533: f64;
    let cosisq: f64;
    let emo: f64;
    let emsqo: f64;

    /*
    double ainv2, aonv = 0.0, cosisq, eoc, f220, f221, f311,
        f321, f322, f330, f441, f442, f522, f523,
        f542, f543, g200, g201, g211, g300, g310,
        g322, g410, g422, g520, g521, g532, g533,
        ses, sgs, sghl, sghs, shs, shll, sis,
        sini2, sls, temp, temp1, theta, xno2, q22,
        q31, q33, root22, root44, root54, rptim, root32,
        root52, x2o3, znl, emo, zns, emsqo;
    */

    const Q22: f64 = 1.7891679e-6;
    const Q31: f64 = 2.1460748e-6;
    const Q33: f64 = 2.2123015e-7;
    const ROOT22: f64 = 1.7891679e-6;
    const ROOT44: f64 = 7.3636953e-9;
    const ROOT54: f64 = 2.1765803e-9;
    const RPTIM: f64 = 4.375_269_088_011_3e-3; // this equates to 7.29211514668855e-5 rad/sec
    const ROOT32: f64 = 3.7393792e-7;
    const ROOT52: f64 = 1.1428639e-7;
    const X2O3: f64 = 2.0 / 3.0;
    const ZNL: f64 = 1.5835218e-4;
    const ZNS: f64 = 1.19459e-5;

    // sgp4fix identify constants and allow alternate values
    // just xke is used here so pass it in rather than have multiple calls
    // getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );

    /* -------------------- deep space initialization ------------ */
    *irez = 0;
    if (*nm < 0.0052359877) && (*nm > 0.0034906585) {
        *irez = 1;
    }
    if (*nm >= 8.26e-3) && (*nm <= 9.24e-3) && (*em >= 0.5) {
        *irez = 2;
    }

    /* ------------------------ do solar terms ------------------- */
    let ses = ss1 * ZNS * ss5;
    let sis = ss2 * ZNS * (sz11 + sz13);
    let sls = -ZNS * ss3 * 6.0f64.mul_add(-emsq, sz1 + sz3 - 14.0);
    let sghs = ss4 * ZNS * (sz31 + sz33 - 6.0);
    let mut shs = -ZNS * ss2 * (sz21 + sz23);
    // sgp4fix for 180 deg incl
    if (*inclm < 5.2359877e-2) || (*inclm > PI - 5.2359877e-2) {
        shs = 0.0;
    }
    if sinim != 0.0 {
        shs /= sinim;
    }
    let sgs = cosim.mul_add(-shs, sghs);

    /* ------------------------- do lunar terms ------------------ */
    *dedt = (s1 * ZNL).mul_add(s5, ses);
    *didt = (s2 * ZNL).mul_add(z11 + z13, sis);
    *dmdt = (ZNL * s3).mul_add(-6.0f64.mul_add(-emsq, z1 + z3 - 14.0), sls);
    let sghl = s4 * ZNL * (z31 + z33 - 6.0);
    let mut shll = -ZNL * s2 * (z21 + z23);
    // sgp4fix for 180 deg incl
    if (*inclm < 5.2359877e-2) || (*inclm > PI - 5.2359877e-2) {
        shll = 0.0;
    }
    *domdt = sgs + sghl;
    *dnodt = shs;
    if sinim != 0.0 {
        *domdt -= cosim / sinim * shll;
        *dnodt += shll / sinim;
    }

    /* ----------- calculate deep space resonance effects -------- */
    *dndt = 0.0;
    let theta = tc.mul_add(RPTIM, gsto) % TWOPI;
    *em += *dedt * t;
    *inclm += *didt * t;
    *argpm += *domdt * t;
    *nodem += *dnodt * t;
    *mm += *dmdt * t;
    //   sgp4fix for negative inclinations
    //   the following if statement should be commented out
    //if (inclm < 0.0)
    //  {
    //    inclm  = -inclm;
    //    argpm  = argpm - pi;
    //    nodem = nodem + pi;
    //  }

    /* -------------- initialize the resonance terms ------------- */
    if *irez != 0 {
        let aonv = f64::powf(*nm / xke, X2O3);

        /* ---------- geopotential resonance for 12 hour orbits ------ */
        if *irez == 2 {
            cosisq = cosim * cosim;
            emo = *em;
            *em = ecco;
            emsqo = emsq;
            emsq = eccsq;
            let eoc = *em * emsq;
            g201 = (*em - 0.64).mul_add(-0.440, -0.306);

            if *em <= 0.65 {
                g211 = 16.2900f64.mul_add(emsq, 13.2470f64.mul_add(-(*em), 3.616));
                g310 = 156.5910f64.mul_add(
                    eoc,
                    228.4190f64.mul_add(-emsq, 117.3900f64.mul_add(*em, -19.302)),
                );
                g322 = 146.5816f64.mul_add(
                    eoc,
                    214.6334f64.mul_add(-emsq, 109.7927f64.mul_add(*em, -18.9068)),
                );
                g410 = 313.9530f64.mul_add(
                    eoc,
                    471.0940f64.mul_add(-emsq, 242.6940f64.mul_add(*em, -41.122)),
                );
                g422 = 1083.4350f64.mul_add(
                    eoc,
                    1629.014f64.mul_add(-emsq, 841.8800f64.mul_add(*em, -146.407)),
                );
                g520 = 3708.2760f64.mul_add(
                    eoc,
                    5740.032f64.mul_add(-emsq, 3017.977f64.mul_add(*em, -532.114)),
                );
            } else {
                g211 = 266.724f64.mul_add(
                    eoc,
                    508.738f64.mul_add(-emsq, 331.819f64.mul_add(*em, -72.099)),
                );
                g310 = 1246.113f64.mul_add(
                    eoc,
                    2415.925f64.mul_add(-emsq, 1582.851f64.mul_add(*em, -346.844)),
                );
                g322 = 1215.972f64.mul_add(
                    eoc,
                    2366.899f64.mul_add(-emsq, 1554.908f64.mul_add(*em, -342.585)),
                );
                g410 = 3651.957f64.mul_add(
                    eoc,
                    7193.992f64.mul_add(-emsq, 4758.686f64.mul_add(*em, -1052.797)),
                );
                g422 = 12422.520f64.mul_add(
                    eoc,
                    24462.770f64.mul_add(-emsq, 16178.110f64.mul_add(*em, -3581.690)),
                );
                if *em > 0.715 {
                    g520 = 31324.56f64.mul_add(
                        eoc,
                        54087.36f64.mul_add(-emsq, 29936.92f64.mul_add(*em, -5149.66)),
                    );
                } else {
                    g520 = 3763.64f64.mul_add(emsq, 4664.75f64.mul_add(-(*em), 1464.74));
                }
            }
            if *em < 0.7 {
                g533 = 5542.21f64.mul_add(
                    eoc,
                    9064.7700f64.mul_add(-emsq, 4988.6100f64.mul_add(*em, -919.22770)),
                );
                g521 = 5337.524f64.mul_add(
                    eoc,
                    8491.4146f64.mul_add(-emsq, 4568.6173f64.mul_add(*em, -822.71072)),
                );
                g532 = 5341.4f64.mul_add(
                    eoc,
                    8624.7700f64.mul_add(-emsq, 4690.2500f64.mul_add(*em, -853.66600)),
                );
            } else {
                g533 = 109377.94f64.mul_add(
                    eoc,
                    229838.20f64.mul_add(-emsq, 161616.52f64.mul_add(*em, -37995.780)),
                );
                g521 = 146349.42f64.mul_add(
                    eoc,
                    309468.16f64.mul_add(-emsq, 218913.95f64.mul_add(*em, -51752.104)),
                );
                g532 = 115605.82f64.mul_add(
                    eoc,
                    242699.48f64.mul_add(-emsq, 170470.89f64.mul_add(*em, -40023.880)),
                );
            }

            let sini2 = sinim * sinim;
            f220 = 0.75 * (2.0f64.mul_add(cosim, 1.0) + cosisq);
            f221 = 1.5 * sini2;
            f321 = 1.875 * sinim * 3.0f64.mul_add(-cosisq, 2.0f64.mul_add(-cosim, 1.0));
            f322 = -1.875 * sinim * 3.0f64.mul_add(-cosisq, 2.0f64.mul_add(cosim, 1.0));
            f441 = 35.0 * sini2 * f220;
            f442 = 39.3750 * sini2 * sini2;
            f522 = 9.84375
                * sinim
                * sini2.mul_add(
                    5.0f64.mul_add(-cosisq, 2.0f64.mul_add(-cosim, 1.0)),
                    0.33333333 * 6.0f64.mul_add(cosisq, 4.0f64.mul_add(cosim, -2.0)),
                );
            f523 = sinim
                * (4.92187512 * sini2).mul_add(
                    10.0f64.mul_add(cosisq, 4.0f64.mul_add(-cosim, -2.0)),
                    6.56250012 * 3.0f64.mul_add(-cosisq, 2.0f64.mul_add(cosim, 1.0)),
                );
            f542 = 29.53125
                * sinim
                * cosisq.mul_add(
                    10.0f64.mul_add(cosisq, 8.0f64.mul_add(cosim, -12.0)),
                    8.0f64.mul_add(-cosim, 2.0),
                );
            f543 = 29.53125
                * sinim
                * cosisq.mul_add(
                    10.0f64.mul_add(-cosisq, 8.0f64.mul_add(cosim, 12.0)),
                    8.0f64.mul_add(-cosim, -2.0),
                );
            let xno2 = *nm * *nm;
            let ainv2 = aonv * aonv;
            let mut temp1 = 3.0 * xno2 * ainv2;
            let mut temp = temp1 * ROOT22;
            *d2201 = temp * f220 * g201;
            *d2211 = temp * f221 * g211;
            temp1 *= aonv;
            temp = temp1 * ROOT32;
            *d3210 = temp * f321 * g310;
            *d3222 = temp * f322 * g322;
            temp1 *= aonv;
            temp = 2.0 * temp1 * ROOT44;
            *d4410 = temp * f441 * g410;
            *d4422 = temp * f442 * g422;
            temp1 *= aonv;
            temp = temp1 * ROOT52;
            *d5220 = temp * f522 * g520;
            *d5232 = temp * f523 * g532;
            temp = 2.0 * temp1 * ROOT54;
            *d5421 = temp * f542 * g521;
            *d5433 = temp * f543 * g533;
            *xlamo = (mo + nodeo + nodeo - theta - theta) % TWOPI;
            *xfact = 2.0f64.mul_add(nodedot + *dnodt - RPTIM, mdot + *dmdt) - no;
            *em = emo;
            emsq = emsqo;
        }

        /* ---------------- synchronous resonance terms -------------- */
        if *irez == 1 {
            g200 = emsq.mul_add(0.8125f64.mul_add(emsq, -2.5), 1.0);
            g310 = 2.0f64.mul_add(emsq, 1.0);
            g300 = emsq.mul_add(6.60937f64.mul_add(emsq, -6.0), 1.0);
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
            f311 = (0.9375 * sinim * sinim)
                .mul_add(3.0f64.mul_add(cosim, 1.0), -(0.75 * (1.0 + cosim)));
            f330 = 1.0 + cosim;
            f330 = 1.875 * f330 * f330 * f330;
            *del1 = 3.0 * *nm * *nm * aonv * aonv;
            *del2 = 2.0 * *del1 * f220 * g200 * Q22;
            *del3 = 3.0 * *del1 * f330 * g300 * Q33 * aonv;
            *del1 = *del1 * f311 * g310 * Q31 * aonv;
            *xlamo = (mo + nodeo + argpo - theta) % TWOPI;
            *xfact = mdot + xpidot - RPTIM + *dmdt + *domdt + *dnodt - no;
        }

        /* ------------ for sgp4, initialize the integrator ---------- */
        *xli = *xlamo;
        *xni = no;
        *atime = 0.0;
        *nm = no + *dndt;
    }

    //#include "debug3.cpp"
} // dsinit
