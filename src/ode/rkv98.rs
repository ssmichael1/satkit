use super::rk_adaptive::RKAdaptive;

// File below is auto-generated via python script that parses
// data available on web at:
// <https://www.sfu.ca/~jverner/RKV98.IIa.Robust.000000351.081209.CoeffsOnlyFLOAT6040>

use super::rkv98_table as bt;
const N: usize = 21;
const NI: usize = 8;

pub struct RKV98 {}

impl RKAdaptive<N, NI> for RKV98 {
    const ORDER: usize = 9;

    const FSAL: bool = false;

    const B: [f64; N] = bt::B;

    const C: [f64; N] = bt::C;

    const A: [[f64; N]; N] = bt::A;

    const BI: [[f64; 8]; N] = bt::BI;

    const BERR: [f64; N] = {
        let mut berr = [0.0; N];
        let mut ix: usize = 0;
        while ix < N {
            berr[ix] = Self::B[ix] - bt::BHAT[ix];
            ix += 1;
        }
        berr
    };
}

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::super::RKAdaptive;
    use super::super::RKAdaptiveSettings;
    use super::RKV98;
    use crate::types::*;
    use std::f64::consts::PI;

    type State = Vector<2>;
    #[test]
    fn test_nointerp() -> ODEResult<()> {
        let y0 = State::new(1.0, 0.0);

        let mut settings = RKAdaptiveSettings::default();
        settings.dense_output = false;
        settings.abserror = 1e-8;
        settings.relerror = 1e-8;

        let _res = RKV98::integrate(
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
        let y0 = Vector::<2>::new(1.0, 0.0);
        let mut settings = RKAdaptiveSettings::default();
        settings.abserror = 1e-14;
        settings.relerror = 1e-14;
        settings.dense_output = true;

        let sol = RKV98::integrate(
            0.0,
            PI,
            &y0,
            |_t, &y| Ok(Vector::<2>::new(y[1], -y[0])),
            &settings,
        )?;
        let testcount = 100;
        (0..100).for_each(|idx| {
            let x = idx as f64 * PI / testcount as f64;
            let interp = RKV98::interpolate(x, &sol).unwrap();
            assert!((interp[0] - x.cos()).abs() < 1e-12);
            assert!((interp[1] + x.sin()).abs() < 1e-12);
        });

        Ok(())
    }
}
