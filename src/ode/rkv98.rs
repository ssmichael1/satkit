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
    use super::super::HarmonicOscillator;
    use super::super::RKAdaptive;
    use super::super::RKAdaptiveSettings;
    use super::RKV98;

    type State = nalgebra::Vector2<f64>;
    #[test]
    fn test_noinetrp() -> ODEResult<()> {
        let mut system = HarmonicOscillator::new(1.0);
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        let mut settings = RKAdaptiveSettings::default();
        settings.dense_output = false;
        settings.abserror = 1e-8;
        settings.relerror = 1e-8;

        let res = RKV98::integrate(0.0, 2.0 * PI, &y0, &mut system, &settings)?;
        println!("res = {:?}", res);
        //let res2 = RKV98::integrate(0.0, -2.0 * PI, &y0, &mut system, &settings)?;
        //println!("res2 = {:?}", res2);

        //assert!((res.y[0] - 1.0).abs() < 1.0e-11);
        //assert!(res.y[1].abs() < 1.0e-11);

        Ok(())
    }

    #[test]
    fn testit() -> ODEResult<()> {
        let mut system = HarmonicOscillator::new(1.0);
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        let mut settings = RKAdaptiveSettings::default();
        settings.dense_output = true;
        settings.abserror = 1e-12;
        settings.relerror = 1e-12;

        let (sol, interp) =
            RKV98::integrate_dense(0.0, PI, PI / 2.0 * 0.05, &y0, &mut system, &settings)?;

        println!("sol evals = {}", sol.nevals);
        interp.x.iter().enumerate().for_each(|(idx, x)| {
            // We know the exact solution for the harmonic oscillator
            let exact = x.cos();
            let exact_v = -x.sin();
            // Compare with the interpolated result
            let diff = exact - interp.y[idx][0];
            let diff_v = exact_v - interp.y[idx][1];
            // we set abs and rel error to 1e-12, so lets check!
            println!("{:+e} {:+e}", diff.abs(), diff_v.abs());
            assert!(diff.abs() < 1e-11);
            assert!(diff_v.abs() < 1e-11);
        });

        Ok(())
    }
}
