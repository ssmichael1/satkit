use std::f64::consts::PI;

/*-----------------------------------------------------------------------------
*
*                           procedure dspace
*
*  this procedure provides deep space contributions to mean elements for
*    perturbing third body.  these effects have been averaged over one
*    revolution of the sun and moon.  for earth resonance effects, the
*    effects have been averaged over no revolutions of the satellite.
*    (mean motion)
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433 -
*    dedt        -
*    del1, del2, del3  -
*    didt        -
*    dmdt        -
*    dnodt       -
*    domdt       -
*    irez        - flag for resonance           0-none, 1-one day, 2-half day
*    argpo       - argument of perigee
*    argpdot     - argument of perigee dot (rate)
*    t           - time
*    tc          -
*    gsto        - gst
*    xfact       -
*    xlamo       -
*    no          - mean motion
*    atime       -
*    em          - eccentricity
*    ft          -
*    argpm       - argument of perigee
*    inclm       - inclination
*    xli         -
*    mm          - mean anomaly
*    xni         - mean motion
*    nodem       - right ascension of ascending node
*
*  outputs       :
*    atime       -
*    em          - eccentricity
*    argpm       - argument of perigee
*    inclm       - inclination
*    xli         -
*    mm          - mean anomaly
*    xni         -
*    nodem       - right ascension of ascending node
*    dndt        -
*    nm          - mean motion
*
*  locals        :
*    delt        -
*    ft          -
*    theta       -
*    x2li        -
*    x2omi       -
*    xl          -
*    xldot       -
*    xnddt       -
*    xndt        -
*    xomi        -
*
*  coupling      :
*    none        -
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
a	----------------------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn dspace(
    irez: i32,
    d2201: f64,
    d2211: f64,
    d3210: f64,
    d3222: f64,
    d4410: f64,
    d4422: f64,
    d5220: f64,
    d5232: f64,
    d5421: f64,
    d5433: f64,
    dedt: f64,
    del1: f64,
    del2: f64,
    del3: f64,
    didt: f64,
    dmdt: f64,
    dnodt: f64,
    domdt: f64,
    argpo: f64,
    argpdot: f64,
    t: f64,
    tc: f64,
    gsto: f64,
    xfact: f64,
    xlamo: f64,
    no: f64,
    atime: &mut f64,
    em: &mut f64,
    argpm: &mut f64,
    inclm: &mut f64,
    xli: &mut f64,
    mm: &mut f64,
    xni: &mut f64,
    nodem: &mut f64,
    dndt: &mut f64,
    nm: &mut f64,
) {
    const TWOPI: f64 = 2.0 * PI;
    let mut iretn: i32;
    //let mut iret: i32;

    let delt: f64;
    let mut ft: f64;

    let mut x2li: f64;
    let mut x2omi: f64;
    let xl: f64;
    let mut xldot: f64 = 0.0;
    let mut xnddt: f64 = 0.0;
    let mut xndt: f64 = 0.0;
    let mut xomi: f64;

    /*
        int iretn, iret;
        double delt, ft, theta, x2li, x2omi, xl, xldot, xnddt, xndt, xomi, g22, g32,
            g44, g52, g54, fasx2, fasx4, fasx6, rptim, step2, stepn, stepp;
    */

    const FASX2: f64 = 0.13130908;
    const FASX4: f64 = 2.8843198;
    const FASX6: f64 = 0.37448087;
    const G22: f64 = 5.7686396;
    const G32: f64 = 0.95240898;
    const G44: f64 = 1.8014998;
    const G52: f64 = 1.0508330;
    const G54: f64 = 4.4108898;
    const RPTIM: f64 = 4.375_269_088_011_3e-3; // this equates to 7.29211514668855e-5 rad/sec
    const STEPP: f64 = 720.0;
    const STEPN: f64 = -720.0;
    const STEP2: f64 = 259200.0;

    /* ----------- calculate deep space resonance effects ----------- */
    *dndt = 0.0;
    let theta: f64 = tc.mul_add(RPTIM, gsto) % TWOPI;
    *em += dedt * t;

    *inclm += didt * t;
    *argpm += domdt * t;
    *nodem += dnodt * t;
    *mm += dmdt * t;

    //   sgp4fix for negative inclinations
    //   the following if statement should be commented out
    //  if (inclm < 0.0)
    // {
    //    inclm = -inclm;
    //    argpm = argpm - pi;
    //    nodem = nodem + pi;
    //  }

    /* - update resonances : numerical (euler-maclaurin) integration - */
    /* ------------------------- epoch restart ----------------------  */
    //   sgp4fix for propagator problems
    //   the following integration works for negative time steps and periods
    //   the specific changes are unknown because the original code was so convoluted

    // sgp4fix take out atime = 0.0 and fix for faster operation
    ft = 0.0;
    if irez != 0 {
        // sgp4fix streamline check
        if (*atime == 0.0) || (t * *atime <= 0.0) || (f64::abs(t) < f64::abs(*atime)) {
            *atime = 0.0;
            *xni = no;
            *xli = xlamo;
        }
        // sgp4fix move check outside loop
        if t > 0.0 {
            delt = STEPP;
        } else {
            delt = STEPN;
        }

        iretn = 381; // added for do loop
                     //iret = 0; // added for loop
        while iretn == 381 {
            /* ------------------- dot terms calculated ------------- */
            /* ----------- near - synchronous resonance terms ------- */
            if irez != 2 {
                xndt = del3.mul_add(f64::sin(3.0 * (*xli - FASX6)), del1.mul_add(f64::sin(*xli - FASX2), del2 * f64::sin(2.0 * (*xli - FASX4))));
                xldot = *xni + xfact;
                xnddt = (3.0 * del3).mul_add(f64::cos(3.0 * (*xli - FASX6)), del1.mul_add(f64::cos(*xli - FASX2), 2.0 * del2 * f64::cos(2.0 * (*xli - FASX4))));
                xnddt *= xldot;
            } else {
                /* --------- near - half-day resonance terms -------- */
                xomi = argpdot.mul_add(*atime, argpo);
                x2omi = xomi + xomi;
                x2li = *xli + *xli;
                xndt = d5433.mul_add(f64::sin(-xomi + x2li - G54), d5421.mul_add(f64::sin(xomi + x2li - G54), d5232.mul_add(f64::sin(-xomi + *xli - G52), d5220.mul_add(f64::sin(xomi + *xli - G52), d4422.mul_add(f64::sin(x2li - G44), d4410.mul_add(f64::sin(x2omi + x2li - G44), d3222.mul_add(f64::sin(-xomi + *xli - G32), d3210.mul_add(f64::sin(xomi + *xli - G32), d2201.mul_add(f64::sin(x2omi + *xli - G22), d2211 * f64::sin(*xli - G22))))))))));
                xldot = *xni + xfact;
                xnddt = 2.0f64.mul_add(d5433.mul_add(f64::cos(-xomi + x2li - G54), d5421.mul_add(f64::cos(xomi + x2li - G54), d4410.mul_add(f64::cos(x2omi + x2li - G44), d4422 * f64::cos(x2li - G44)))), d5232.mul_add(f64::cos(-xomi + *xli - G52), d5220.mul_add(f64::cos(xomi + *xli - G52), d3222.mul_add(f64::cos(-xomi + *xli - G32), d3210.mul_add(f64::cos(xomi + *xli - G32), d2201.mul_add(f64::cos(x2omi + *xli - G22), d2211 * f64::cos(*xli - G22)))))));
                xnddt *= xldot;
            }

            /* ----------------------- integrator ------------------- */
            // sgp4fix move end checks to end of routine
            if f64::abs(t - *atime) >= STEPP {
                //iret = 0;
                iretn = 381;
            } else
            // exit here
            {
                ft = t - *atime;
                iretn = 0;
            }

            if iretn == 381 {
                *xli = xndt.mul_add(STEP2, xldot.mul_add(delt, *xli));
                *xni = xnddt.mul_add(STEP2, xndt.mul_add(delt, *xni));
                *atime += delt;
            }
        } // while iretn = 381

        *nm = (xnddt * ft * ft).mul_add(0.5, xndt.mul_add(ft, *xni));
        xl = (xndt * ft * ft).mul_add(0.5, xldot.mul_add(ft, *xli));
        if irez != 1 {
            *mm = 2.0f64.mul_add(theta, 2.0f64.mul_add(-(*nodem), xl));
            *dndt = *nm - no;
        } else {
            *mm = xl - *nodem - *argpm + theta;
            *dndt = *nm - no;
        }
        *nm = no + *dndt;
    }

    //#include "debug4.cpp"
} // dsspace
