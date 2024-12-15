/*-----------------------------------------------------------------------------
*
*                           procedure dscom
*
*  this procedure provides deep space common items used by both the secular
*    and periodics subroutines.  input is provided as shown. this routine
*    used to be called dpper, but the functions inside weren't well organized.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    epoch       -
*    ep          - eccentricity
*    argpp       - argument of perigee
*    tc          -
*    inclp       - inclination
*    nodep       - right ascension of ascending node
*    np          - mean motion
*
*  outputs       :
*    sinim  , cosim  , sinomm , cosomm , snodm  , cnodm
*    day         -
*    e3          -
*    ee2         -
*    em          - eccentricity
*    emsq        - eccentricity squared
*    gam         -
*    peo         -
*    pgho        -
*    pho         -
*    pinco       -
*    plo         -
*    rtemsq      -
*    se2, se3         -
*    sgh2, sgh3, sgh4        -
*    sh2, sh3, si2, si3, sl2, sl3, sl4         -
*    s1, s2, s3, s4, s5, s6, s7          -
*    ss1, ss2, ss3, ss4, ss5, ss6, ss7, sz1, sz2, sz3         -
*    sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33        -
*    xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4         -
*    nm          - mean motion
*    z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33         -
*    zmol        -
*    zmos        -
*
*  locals        :
*    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10         -
*    betasq      -
*    cc          -
*    ctem, stem        -
*    x1, x2, x3, x4, x5, x6, x7, x8          -
*    xnodce      -
*    xnoi        -
*    zcosg  , zsing  , zcosgl , zsingl , zcosh  , zsinh  , zcoshl , zsinhl ,
*    zcosi  , zsini  , zcosil , zsinil ,
*    zx          -
*    zy          -
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
#[allow(clippy::too_many_arguments)]
pub fn dscom(
    epoch: f64,
    ep: f64,
    argpp: f64,
    tc: f64,
    inclp: f64,
    nodep: f64,
    np: f64,
    snodm: &mut f64,
    cnodm: &mut f64,
    sinim: &mut f64,
    cosim: &mut f64,
    sinomm: &mut f64,
    cosomm: &mut f64,
    day: &mut f64,
    e3: &mut f64,
    ee2: &mut f64,
    em: &mut f64,
    emsq: &mut f64,
    gam: &mut f64,
    peo: &mut f64,
    pgho: &mut f64,
    pho: &mut f64,
    pinco: &mut f64,
    plo: &mut f64,
    rtemsq: &mut f64,
    se2: &mut f64,
    se3: &mut f64,
    sgh2: &mut f64,
    sgh3: &mut f64,
    sgh4: &mut f64,
    sh2: &mut f64,
    sh3: &mut f64,
    si2: &mut f64,
    si3: &mut f64,
    sl2: &mut f64,
    sl3: &mut f64,
    sl4: &mut f64,
    s1: &mut f64,
    s2: &mut f64,
    s3: &mut f64,
    s4: &mut f64,
    s5: &mut f64,
    s6: &mut f64,
    s7: &mut f64,
    ss1: &mut f64,
    ss2: &mut f64,
    ss3: &mut f64,
    ss4: &mut f64,
    ss5: &mut f64,
    ss6: &mut f64,
    ss7: &mut f64,
    sz1: &mut f64,
    sz2: &mut f64,
    sz3: &mut f64,
    sz11: &mut f64,
    sz12: &mut f64,
    sz13: &mut f64,
    sz21: &mut f64,
    sz22: &mut f64,
    sz23: &mut f64,
    sz31: &mut f64,
    sz32: &mut f64,
    sz33: &mut f64,
    xgh2: &mut f64,
    xgh3: &mut f64,
    xgh4: &mut f64,
    xh2: &mut f64,
    xh3: &mut f64,
    xi2: &mut f64,
    xi3: &mut f64,
    xl2: &mut f64,
    xl3: &mut f64,
    xl4: &mut f64,
    nm: &mut f64,
    z1: &mut f64,
    z2: &mut f64,
    z3: &mut f64,
    z11: &mut f64,
    z12: &mut f64,
    z13: &mut f64,
    z21: &mut f64,
    z22: &mut f64,
    z23: &mut f64,
    z31: &mut f64,
    z32: &mut f64,
    z33: &mut f64,
    zmol: &mut f64,
    zmos: &mut f64,
) {
    /* -------------------------- constants ------------------------- */
    const ZES: f64 = 0.01675;
    const ZEL: f64 = 0.05490;
    const C1SS: f64 = 2.9864797e-6;
    const C1L: f64 = 4.7968065e-7;
    const ZSINIS: f64 = 0.39785416;
    const ZCOSIS: f64 = 0.91744867;
    const ZCOSGS: f64 = 0.1945905;
    const ZSINGS: f64 = -0.98088458;
    const TWOPI: f64 = 2.0 * std::f64::consts::PI;

    /* --------------------- local variables ------------------------ */
    //let lsflg: i32;
    /*
    int lsflg;
    double a1, a2, a3, a4, a5, a6, a7,
        a8, a9, a10, betasq, cc, ctem, stem,
        x1, x2, x3, x4, x5, x6, x7,
        x8, xnodce, xnoi, zcosg, zcosgl, zcosh, zcoshl,
        zcosi, zcosil, zsing, zsingl, zsinh, zsinhl, zsini,
        zsinil, zx, zy;
    */
    let mut a1: f64;
    let mut a2: f64;
    let mut a3: f64;
    let mut a4: f64;
    let mut a5: f64;
    let mut a6: f64;
    let mut a7: f64;
    let mut a8: f64;
    let mut a9: f64;
    let mut a10: f64;

    let mut cc: f64;

    let mut x1: f64;
    let mut x2: f64;
    let mut x3: f64;
    let mut x4: f64;
    let mut x5: f64;
    let mut x6: f64;
    let mut x7: f64;
    let mut x8: f64;

    let mut zcosg: f64;

    let mut zcosh: f64;

    let mut zcosi: f64;

    let mut zsing: f64;

    let mut zsinh: f64;

    let mut zsini: f64;

    let mut zx: f64;

    *nm = np;
    *em = ep;
    *snodm = f64::sin(nodep);
    *cnodm = f64::cos(nodep);
    *sinomm = f64::sin(argpp);
    *cosomm = f64::cos(argpp);
    *sinim = f64::sin(inclp);
    *cosim = f64::cos(inclp);
    *emsq = *em * *em;
    let betasq: f64 = 1.0 - *emsq;
    *rtemsq = f64::sqrt(betasq);

    /* ----------------- initialize lunar solar terms --------------- */
    *peo = 0.0;
    *pinco = 0.0;
    *plo = 0.0;
    *pgho = 0.0;
    *pho = 0.0;
    *day = epoch + 18261.5 + tc / 1440.0;
    let xnodce: f64 = 9.2422029e-4f64.mul_add(-(*day), 4.5236020) % TWOPI;
    let stem: f64 = f64::sin(xnodce);
    let ctem: f64 = f64::cos(xnodce);
    let zcosil: f64 = 0.03568096f64.mul_add(-ctem, 0.91375164);
    let zsinil: f64 = f64::sqrt(zcosil.mul_add(-zcosil, 1.0));
    let zsinhl: f64 = 0.089683511 * stem / zsinil;
    let zcoshl: f64 = f64::sqrt(zsinhl.mul_add(-zsinhl, 1.0));
    *gam = 0.0019443680f64.mul_add(*day, 5.8351514);
    zx = 0.39785416 * stem / zsinil;
    let zy: f64 = zcoshl.mul_add(ctem, 0.91744867 * zsinhl * stem);
    zx = f64::atan2(zx, zy);
    zx = *gam + zx - xnodce;
    let zcosgl: f64 = f64::cos(zx);
    let zsingl: f64 = f64::sin(zx);

    /* ------------------------- do solar terms --------------------- */
    zcosg = ZCOSGS;
    zsing = ZSINGS;
    zcosi = ZCOSIS;
    zsini = ZSINIS;
    zcosh = *cnodm;
    zsinh = *snodm;
    cc = C1SS;
    let xnoi: f64 = 1.0 / *nm;

    //for (lsflg = 1; lsflg <= 2; lsflg++)
    for lsflg in 1..3 {
        a1 = zcosg.mul_add(zcosh, zsing * zcosi * zsinh);
        a3 = (-zsing).mul_add(zcosh, zcosg * zcosi * zsinh);
        a7 = (-zcosg).mul_add(zsinh, zsing * zcosi * zcosh);
        a8 = zsing * zsini;
        a9 = zsing.mul_add(zsinh, zcosg * zcosi * zcosh);
        a10 = zcosg * zsini;
        a2 = (*cosim).mul_add(a7, *sinim * a8);
        a4 = (*cosim).mul_add(a9, *sinim * a10);
        a5 = (-*sinim).mul_add(a7, *cosim * a8);
        a6 = (-*sinim).mul_add(a9, *cosim * a10);

        x1 = a1.mul_add(*cosomm, a2 * *sinomm);
        x2 = a3.mul_add(*cosomm, a4 * *sinomm);
        x3 = (-a1).mul_add(*sinomm, a2 * *cosomm);
        x4 = (-a3).mul_add(*sinomm, a4 * *cosomm);
        x5 = a5 * *sinomm;
        x6 = a6 * *sinomm;
        x7 = a5 * *cosomm;
        x8 = a6 * *cosomm;

        *z31 = (12.0 * x1).mul_add(x1, -(3.0 * x3 * x3));
        *z32 = (24.0 * x1).mul_add(x2, -(6.0 * x3 * x4));
        *z33 = (12.0 * x2).mul_add(x2, -(3.0 * x4 * x4));
        *z1 = 3.0f64.mul_add(a1.mul_add(a1, a2 * a2), *z31 * *emsq);
        *z2 = 6.0f64.mul_add(a1.mul_add(a3, a2 * a4), *z32 * *emsq);
        *z3 = 3.0f64.mul_add(a3.mul_add(a3, a4 * a4), *z33 * *emsq);
        *z11 = (-6.0 * a1).mul_add(a5, *emsq * (-24.0 * x1).mul_add(x7, -(6.0 * x3 * x5)));
        *z12 = (-6.0f64).mul_add(a1.mul_add(a6, a3 * a5), *emsq * (-24.0f64).mul_add(x2.mul_add(x7, x1 * x8), -(6.0 * x3.mul_add(x6, x4 * x5))));
        *z13 = (-6.0 * a3).mul_add(a6, *emsq * (-24.0 * x2).mul_add(x8, -(6.0 * x4 * x6)));
        *z21 = (6.0 * a2).mul_add(a5, *emsq * (24.0 * x1).mul_add(x5, -(6.0 * x3 * x7)));
        *z22 = 6.0f64.mul_add(a4.mul_add(a5, a2 * a6), *emsq * 24.0f64.mul_add(x2.mul_add(x5, x1 * x6), -(6.0 * x4.mul_add(x7, x3 * x8))));
        *z23 = (6.0 * a4).mul_add(a6, *emsq * (24.0 * x2).mul_add(x6, -(6.0 * x4 * x8)));
        *z1 = betasq.mul_add(*z31, *z1 + *z1);
        *z2 = betasq.mul_add(*z32, *z2 + *z2);
        *z3 = betasq.mul_add(*z33, *z3 + *z3);
        *s3 = cc * xnoi;
        *s2 = -0.5 * *s3 / *rtemsq;
        *s4 = *s3 * *rtemsq;
        *s1 = -15.0 * *em * *s4;
        *s5 = x1.mul_add(x3, x2 * x4);
        *s6 = x2.mul_add(x3, x1 * x4);
        *s7 = x2.mul_add(x4, -(x1 * x3));

        /* ----------------------- do lunar terms ------------------- */
        if lsflg == 1 {
            *ss1 = *s1;
            *ss2 = *s2;
            *ss3 = *s3;
            *ss4 = *s4;
            *ss5 = *s5;
            *ss6 = *s6;
            *ss7 = *s7;
            *sz1 = *z1;
            *sz2 = *z2;
            *sz3 = *z3;
            *sz11 = *z11;
            *sz12 = *z12;
            *sz13 = *z13;
            *sz21 = *z21;
            *sz22 = *z22;
            *sz23 = *z23;
            *sz31 = *z31;
            *sz32 = *z32;
            *sz33 = *z33;
            zcosg = zcosgl;
            zsing = zsingl;
            zcosi = zcosil;
            zsini = zsinil;
            zcosh = zcoshl.mul_add(*cnodm, zsinhl * *snodm);
            zsinh = (*snodm).mul_add(zcoshl, -(*cnodm * zsinhl));
            cc = C1L;
        }
    }

    *zmol = (0.22997150f64.mul_add(*day, 4.7199672) - *gam) % TWOPI;
    *zmos = 0.017201977f64.mul_add(*day, 6.2565837) % TWOPI;

    /* ------------------------ do solar terms ---------------------- */
    *se2 = 2.0 * *ss1 * *ss6;
    *se3 = 2.0 * *ss1 * *ss7;
    *si2 = 2.0 * *ss2 * *sz12;
    *si3 = 2.0 * *ss2 * (*sz13 - *sz11);
    *sl2 = -2.0 * *ss3 * *sz2;
    *sl3 = -2.0 * *ss3 * (*sz3 - *sz1);
    *sl4 = -2.0 * *ss3 * 9.0f64.mul_add(-(*emsq), -21.0) * ZES;
    *sgh2 = 2.0 * *ss4 * *sz32;
    *sgh3 = 2.0 * *ss4 * (*sz33 - *sz31);
    *sgh4 = -18.0 * *ss4 * ZES;
    *sh2 = -2.0 * *ss2 * *sz22;
    *sh3 = -2.0 * *ss2 * (*sz23 - *sz21);

    /* ------------------------ do lunar terms ---------------------- */
    *ee2 = 2.0 * *s1 * *s6;
    *e3 = 2.0 * *s1 * *s7;
    *xi2 = 2.0 * *s2 * *z12;
    *xi3 = 2.0 * *s2 * (*z13 - *z11);
    *xl2 = -2.0 * *s3 * *z2;
    *xl3 = -2.0 * *s3 * (*z3 - *z1);
    *xl4 = -2.0 * *s3 * 9.0f64.mul_add(-(*emsq), -21.0) * ZEL;
    *xgh2 = 2.0 * *s4 * *z32;
    *xgh3 = 2.0 * *s4 * (*z33 - *z31);
    *xgh4 = -18.0 * *s4 * ZEL;
    *xh2 = -2.0 * *s2 * *z22;
    *xh3 = -2.0 * *s2 * (*z23 - *z21);

    //#include "debug2.cpp"
} // dscom
