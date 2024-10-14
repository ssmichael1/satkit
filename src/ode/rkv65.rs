use super::rk_adaptive::RKAdaptive;

// File below is auto-generated via python script that parses
// data available on web at:
// <https://www.sfu.ca/~jverner/RKV87.IIa.Robust.00000754677.081208.CoeffsOnlyFLOAT>

use super::rkv65_table;

pub struct RKV65 {}

impl RKAdaptive<10, 6> for RKV65 {
    const ORDER: usize = 6;

    const FSAL: bool = false;

    const B: [f64; 10] = rkv65_table::B;

    const C: [f64; 10] = rkv65_table::C;

    const A: [[f64; 10]; 10] = rkv65_table::A;

    const BI: [[f64; 6]; 10] = rkv65_table::BI;

    const BERR: [f64; 10] = {
        let mut berr = [0.0; 10];
        let mut ix: usize = 0;
        while ix < 10 {
            berr[ix] = Self::B[ix] - rkv65_table::BHAT[ix];
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
    use super::RKV65;

    type State = nalgebra::Vector2<f64>;
    #[test]
    fn test_nointerp() -> ODEResult<()> {
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        let mut settings = RKAdaptiveSettings::default();
        settings.dense_output = false;
        settings.abserror = 1e-8;
        settings.relerror = 1e-8;

        let _res = RKV65::integrate(
            0.0,
            2.0 * PI,
            &y0,
            |_t, y| Ok([y[1], -y[0]].into()),
            &settings,
        )?;

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

        let sol = RKV65::integrate(0.0, PI, &y0, |_t, y| Ok([y[1], -y[0]].into()), &settings)?;

        let testcount = 100;
        (0..100).for_each(|idx| {
            let x = idx as f64 * PI / testcount as f64;
            let interp = RKV65::interpolate(x, &sol).unwrap();
            assert!((interp[0] - x.cos()).abs() < 1e-11);
            assert!((interp[1] + x.sin()).abs() < 1e-11);
        });

        Ok(())
    }
}
