use nalgebra as na;

use super::ierstable::IERSTable;
use crate::frametransform::{qrot_ycoord, qrot_zcoord};
use crate::{Instant, TimeScale};

type Quat = na::UnitQuaternion<f64>;
type Delaunay = na::SVector<f64, 14>;

use std::f64::consts::PI;

use once_cell::sync::OnceCell;

fn table5a_singleton() -> &'static IERSTable {
    static INSTANCE: OnceCell<IERSTable> = OnceCell::new();
    INSTANCE.get_or_init(|| IERSTable::from_file("tab5.2a.txt").unwrap())
}

fn table5b_singleton() -> &'static IERSTable {
    static INSTANCE: OnceCell<IERSTable> = OnceCell::new();
    INSTANCE.get_or_init(|| IERSTable::from_file("tab5.2b.txt").unwrap())
}

fn table5d_singleton() -> &'static IERSTable {
    static INSTANCE: OnceCell<IERSTable> = OnceCell::new();
    INSTANCE.get_or_init(|| IERSTable::from_file("tab5.2d.txt").unwrap())
}

pub fn qcirs2gcrs_dxdy(tm: &Instant, dxdy: Option<(f64, f64)>) -> Quat {
    let t_tt = (tm.as_mjd_with_scale(TimeScale::TT) - 51544.5) / 36525.0;
    const ASEC2RAD: f64 = PI / 180.0 / 3600.0;

    let mut delaunay = Delaunay::zeros();

    // Arguments for lunisolar nutation
    // Equation 5.43 in IERS technical note 36

    // Mean anomaly of the Moon
    delaunay[0] = ASEC2RAD
        * 3600.0f64.mul_add(
            134.96340251,
            t_tt * t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(-0.00024470, 0.051635), 31.8792),
                1717915923.2178,
            ),
        );

    // Mean anomaly of the sun
    delaunay[1] = ASEC2RAD
        * 3600.0f64.mul_add(
            357.52910918,
            t_tt * t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(-0.00001149, 0.000136), -0.5532),
                129596581.0481,
            ),
        );

    // F = L-Omega
    delaunay[2] = ASEC2RAD
        * 3600.0f64.mul_add(
            93.27209062,
            t_tt * t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(0.00000417, -0.001037), -12.7512),
                1739527262.8478,
            ),
        );

    // D = Mean elongation of the Moon from the Sun
    delaunay[3] = ASEC2RAD
        * 3600.0f64.mul_add(
            297.85019547,
            t_tt * t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(-0.00003169, 0.006593), -6.37006),
                1602961601.2090,
            ),
        );

    // Omega = mean longitude of ascending node of the Moon
    delaunay[4] = ASEC2RAD
        * 3600.0f64.mul_add(
            125.04455501,
            t_tt * t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(-0.00005939, 0.007702), 7.4722),
                -6962890.5431,
            ),
        );

    // Planetary nutation
    // Equations 5.44 in IERS technical note 36
    delaunay[5] = 2608.7903141574f64.mul_add(t_tt, 4.402608842);
    delaunay[6] = 1021.3285546211f64.mul_add(t_tt, 3.176146697);
    delaunay[7] = 628.3075849991f64.mul_add(t_tt, 1.753470314);
    delaunay[8] = 334.0612426700f64.mul_add(t_tt, 6.203480913);
    delaunay[9] = 52.9690962641f64.mul_add(t_tt, 0.599546497);
    delaunay[10] = 21.3299104960f64.mul_add(t_tt, 0.874016757);
    delaunay[11] = 7.4781598567f64.mul_add(t_tt, 5.481293872);
    delaunay[12] = 3.8133035638f64.mul_add(t_tt, 5.311886287);
    delaunay[13] = t_tt * t_tt.mul_add(0.00000538691, 0.02438175);

    // Polynomial part of X & Y, values in arcseconds
    // Equations 5.16 in IERS technical note 36
    let x0 = t_tt.mul_add(
        t_tt.mul_add(
            t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(0.0000059285, 0.000007578), -0.19861834),
                -0.4297829,
            ),
            2004.191898,
        ),
        -0.016617,
    );
    let y0 = t_tt.mul_add(
        t_tt.mul_add(
            t_tt.mul_add(
                t_tt.mul_add(t_tt.mul_add(0.0000001358, 0.001112526), 0.00190059),
                -22.4072747,
            ),
            -0.025896,
        ),
        -0.006951,
    );

    // Polynomial part of CIO locator s, values in microarcseconds
    // Described in table 5.2d of IERS technical note 36
    let s0 = t_tt.mul_add(
        t_tt.mul_add(
            t_tt.mul_add(t_tt.mul_add(t_tt.mul_add(15.62, 27.98), -72574.11), -122.68),
            3808.65,
        ),
        94.0,
    );

    let xsums = table5a_singleton().compute(t_tt, &delaunay);
    let ysums = table5b_singleton().compute(t_tt, &delaunay);
    let ssums = table5d_singleton().compute(t_tt, &delaunay);
    let mut x = xsums.mul_add(1.0e-6, x0) * ASEC2RAD;
    let mut y = ysums.mul_add(1.0e-6, y0) * ASEC2RAD;
    // If dX and dY are passed in, they are in milli-arcsecs
    if dxdy.is_some() {
        let (dx, dy) = dxdy.unwrap();
        x += dx * 1e-3 * ASEC2RAD;
        y += dy * 1e-3 * ASEC2RAD;
    }

    let s = ((s0 + ssums) * 1.0e-6).mul_add(ASEC2RAD, -(x * y / 2.0));

    // Compute expression for the celestial motion of the
    // celestial intermediate pole (CIP)
    // Equations 5.6 & 5.7 of IERS technical note 36
    let e = f64::atan2(y, x);
    let d = f64::asin(f64::sqrt(x.mul_add(x, y * y)));
    qrot_zcoord(-e) * qrot_ycoord(-d) * qrot_zcoord(e + s)
}

///
/// Return quatnerion represention rotation
/// from the CIRS (Celestial Intermediate Reference System) to the
/// GCRS (Geocentric Celestial Reference Frame) at given instant
///
/// # Arguments:
///
/// * `time` - The time instance at which to compute the rotation
///
/// # Reference:
///
/// * See Vallado Ch. 3.7
/// * Also see [IERS Technical Note 36, Chapter 5](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_043.pdf)
///    
pub fn qcirs2gcrs(tm: &Instant) -> Quat {
    let dxdy: Option<(f64, f64)> = crate::earth_orientation_params::get(tm).map(|v| (v[4], v[5]));
    qcirs2gcrs_dxdy(tm, dxdy)
}
