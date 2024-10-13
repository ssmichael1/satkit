//! Tsitouras Order 5(4) Runge-Kutta integrator
//!
//! See:
//! <https://dblp.org/rec/journals/cma/Tsitouras11>
//!
//! Note: in paper, sign is reversed on Bhat[7] ...
//! should be -1.0/66.0, not 1.0/66.0
//!
//! Note also: bhat values are actually error values
//! The nomenclature is confusing
//!

use super::rk_adaptive::RKAdaptive;

const A32: f64 = 0.3354806554923570;
const A42: f64 = -6.359448489975075;
const A52: f64 = -11.74888356406283;
const A43: f64 = 4.362295432869581;
const A53: f64 = 7.495539342889836;
const A54: f64 = -0.09249506636175525;
const A62: f64 = -12.92096931784711;
const A63: f64 = 8.159367898576159;
const A64: f64 = -0.07158497328140100;
const A65: f64 = -0.02826905039406838;

const BI11: f64 = -1.0530884977290216;
const BI12: f64 = -1.3299890189751412;
const BI13: f64 = -1.4364028541716351;
const BI14: f64 = 0.7139816917074209;

const BI21: f64 = 0.1017;
const BI22: f64 = -2.1966568338249754;
const BI23: f64 = 1.2949852507374631;

const BI31: f64 = 2.490627285651252793;
const BI32: f64 = -2.38535645472061657;
const BI33: f64 = 1.57803468208092486;

const BI41: f64 = -16.54810288924490272;
const BI42: f64 = -1.21712927295533244;
const BI43: f64 = -0.61620406037800089;

const BI51: f64 = 47.37952196281928122;
const BI52: f64 = -1.203071208372362603;
const BI53: f64 = -0.658047292653547382;

const BI61: f64 = -34.87065786149660974;
const BI62: f64 = -1.2;
const BI63: f64 = -2.0 / 3.0;

const BI71: f64 = 2.5;
const BI72: f64 = -1.0;
const BI73: f64 = -0.6;

pub struct RKTS54 {}

impl RKTS54 {}

impl RKAdaptive<7, 4> for RKTS54 {
    const C: [f64; 7] = [0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0];

    const B: [f64; 7] = [
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ];

    const BERR: [f64; 7] = [
        0.001780011052226,
        0.000816434459657,
        -0.007880878010262,
        0.144711007173263,
        -0.582357165452555,
        0.458082105929187,
        -1.0 / 66.0,
    ];

    const A: [[f64; 7]; 7] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [Self::C[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [Self::C[2] - A32, A32, 0.0, 0.0, 0.0, 0.0, 0.0],
        [Self::C[3] - A42 - A43, A42, A43, 0.0, 0.0, 0.0, 0.0],
        [Self::C[4] - A52 - A53 - A54, A52, A53, A54, 0.0, 0.0, 0.0],
        [
            Self::C[5] - A62 - A63 - A64 - A65,
            A62,
            A63,
            A64,
            A65,
            0.0,
            0.0,
        ],
        Self::B,
    ];

    const ORDER: usize = 5;

    const FSAL: bool = false;

    // From expanding expressions in Tsitorous paper...
    const BI: [[f64; 4]; 7] = [
        [
            BI11 * BI12 * BI14,
            BI11 * (BI14 + BI12 * BI13),
            BI11 * (BI13 + BI12),
            BI11,
        ],
        [0.0, BI21 * BI23, BI21 * BI22, BI21],
        [0.0, BI31 * BI33, BI31 * BI32, BI31],
        [0.0, BI41 * BI42 * BI43, BI41 * (BI42 + BI43), BI41],
        [0.0, BI51 * BI52 * BI53, BI51 * (BI52 + BI53), BI51],
        [0.0, BI61 * BI62 * BI63, BI61 * (BI62 + BI63), BI61],
        [0.0, BI71 * BI72 * BI73, BI71 * (BI72 + BI73), BI71],
    ];
}

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::super::RKAdaptive;
    use super::super::RKAdaptiveSettings;

    use super::*;
    type State = nalgebra::Vector2<f64>;

    #[test]
    fn test_nointerp() -> ODEResult<()> {
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        let mut settings = RKAdaptiveSettings::default();
        settings.dense_output = false;
        settings.abserror = 1e-8;
        settings.relerror = 1e-8;

        let _res = RKTS54::integrate(
            0.0,
            2.0 * PI,
            &y0,
            |_t, y| Ok([y[1], -y[0]].into()),
            &settings,
        );

        Ok(())
    }

    #[test]
    fn testit() -> ODEResult<()> {
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        let mut settings = RKAdaptiveSettings::default();
        settings.abserror = 1e-14;
        settings.relerror = 1e-14;
        settings.dense_output = true;

        let sol = RKTS54::integrate(0.0, PI, &y0, |_t, &y| Ok([y[1], -y[0]].into()), &settings)?;

        let testcount = 100;
        (0..100).for_each(|idx| {
            let x = idx as f64 * PI / testcount as f64;
            let interp = RKTS54::interpolate(x, &sol).unwrap();
            assert!((interp[0] - x.cos()).abs() < 1e-11);
            assert!((interp[1] + x.sin()).abs() < 1e-11);
        });

        Ok(())
    }
}
