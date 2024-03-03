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
    const RPTIM: f64 = 4.37526908801129966e-3; // this equates to 7.29211514668855e-5 rad/sec
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
    let sls = -ZNS * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq);
    let sghs = ss4 * ZNS * (sz31 + sz33 - 6.0);
    let mut shs = -ZNS * ss2 * (sz21 + sz23);
    // sgp4fix for 180 deg incl
    if (*inclm < 5.2359877e-2) || (*inclm > PI - 5.2359877e-2) {
        shs = 0.0;
    }
    if sinim != 0.0 {
        shs = shs / sinim;
    }
    let sgs = sghs - cosim * shs;

    /* ------------------------- do lunar terms ------------------ */
    *dedt = ses + s1 * ZNL * s5;
    *didt = sis + s2 * ZNL * (z11 + z13);
    *dmdt = sls - ZNL * s3 * (z1 + z3 - 14.0 - 6.0 * emsq);
    let sghl = s4 * ZNL * (z31 + z33 - 6.0);
    let mut shll = -ZNL * s2 * (z21 + z23);
    // sgp4fix for 180 deg incl
    if (*inclm < 5.2359877e-2) || (*inclm > PI - 5.2359877e-2) {
        shll = 0.0;
    }
    *domdt = sgs + sghl;
    *dnodt = shs;
    if sinim != 0.0 {
        *domdt = *domdt - cosim / sinim * shll;
        *dnodt = *dnodt + shll / sinim;
    }

    /* ----------- calculate deep space resonance effects -------- */
    *dndt = 0.0;
    let theta = (gsto + tc * RPTIM) % TWOPI;
    *em = *em + *dedt * t;
    *inclm = *inclm + *didt * t;
    *argpm = *argpm + *domdt * t;
    *nodem = *nodem + *dnodt * t;
    *mm = *mm + *dmdt * t;
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
            g201 = -0.306 - (*em - 0.64) * 0.440;

            if *em <= 0.65 {
                g211 = 3.616 - 13.2470 * *em + 16.2900 * emsq;
                g310 = -19.302 + 117.3900 * *em - 228.4190 * emsq + 156.5910 * eoc;
                g322 = -18.9068 + 109.7927 * *em - 214.6334 * emsq + 146.5816 * eoc;
                g410 = -41.122 + 242.6940 * *em - 471.0940 * emsq + 313.9530 * eoc;
                g422 = -146.407 + 841.8800 * *em - 1629.014 * emsq + 1083.4350 * eoc;
                g520 = -532.114 + 3017.977 * *em - 5740.032 * emsq + 3708.2760 * eoc;
            } else {
                g211 = -72.099 + 331.819 * *em - 508.738 * emsq + 266.724 * eoc;
                g310 = -346.844 + 1582.851 * *em - 2415.925 * emsq + 1246.113 * eoc;
                g322 = -342.585 + 1554.908 * *em - 2366.899 * emsq + 1215.972 * eoc;
                g410 = -1052.797 + 4758.686 * *em - 7193.992 * emsq + 3651.957 * eoc;
                g422 = -3581.690 + 16178.110 * *em - 24462.770 * emsq + 12422.520 * eoc;
                if *em > 0.715 {
                    g520 = -5149.66 + 29936.92 * *em - 54087.36 * emsq + 31324.56 * eoc;
                } else {
                    g520 = 1464.74 - 4664.75 * *em + 3763.64 * emsq;
                }
            }
            if *em < 0.7 {
                g533 = -919.22770 + 4988.6100 * *em - 9064.7700 * emsq + 5542.21 * eoc;
                g521 = -822.71072 + 4568.6173 * *em - 8491.4146 * emsq + 5337.524 * eoc;
                g532 = -853.66600 + 4690.2500 * *em - 8624.7700 * emsq + 5341.4 * eoc;
            } else {
                g533 = -37995.780 + 161616.52 * *em - 229838.20 * emsq + 109377.94 * eoc;
                g521 = -51752.104 + 218913.95 * *em - 309468.16 * emsq + 146349.42 * eoc;
                g532 = -40023.880 + 170470.89 * *em - 242699.48 * emsq + 115605.82 * eoc;
            }

            let sini2 = sinim * sinim;
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq);
            f221 = 1.5 * sini2;
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq);
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq);
            f441 = 35.0 * sini2 * f220;
            f442 = 39.3750 * sini2 * sini2;
            f522 = 9.84375
                * sinim
                * (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                    + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq));
            f523 = sinim
                * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                    + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq));
            f542 = 29.53125
                * sinim
                * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq));
            f543 = 29.53125
                * sinim
                * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq));
            let xno2 = *nm * *nm;
            let ainv2 = aonv * aonv;
            let mut temp1 = 3.0 * xno2 * ainv2;
            let mut temp = temp1 * ROOT22;
            *d2201 = temp * f220 * g201;
            *d2211 = temp * f221 * g211;
            temp1 = temp1 * aonv;
            temp = temp1 * ROOT32;
            *d3210 = temp * f321 * g310;
            *d3222 = temp * f322 * g322;
            temp1 = temp1 * aonv;
            temp = 2.0 * temp1 * ROOT44;
            *d4410 = temp * f441 * g410;
            *d4422 = temp * f442 * g422;
            temp1 = temp1 * aonv;
            temp = temp1 * ROOT52;
            *d5220 = temp * f522 * g520;
            *d5232 = temp * f523 * g532;
            temp = 2.0 * temp1 * ROOT54;
            *d5421 = temp * f542 * g521;
            *d5433 = temp * f543 * g533;
            *xlamo = (mo + nodeo + nodeo - theta - theta) % TWOPI;
            *xfact = mdot + *dmdt + 2.0 * (nodedot + *dnodt - RPTIM) - no;
            *em = emo;
            emsq = emsqo;
        }

        /* ---------------- synchronous resonance terms -------------- */
        if *irez == 1 {
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq);
            g310 = 1.0 + 2.0 * emsq;
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq);
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim);
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
