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
    use super::super::HarmonicOscillator;
    use super::super::RKAdaptive;
    use super::super::RKAdaptiveSettings;
    use super::RKV65;

    type State = nalgebra::Vector2<f64>;

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
            RKV65::integrate_dense(0.0, PI, PI / 2.0 * 0.05, &y0, &mut system, &settings)?;

        println!("sol evals = {}", sol.nevals);
        interp.x.iter().enumerate().for_each(|(idx, x)| {
            // We know the exact solution for the harmonic oscillator
            let exact = x.cos();
            let exact_v = -x.sin();
            // Compare with the interpolated result
            let diff = exact - interp.y[idx][0];
            let diff_v = exact_v - interp.y[idx][1];
            // we set abs and rel error to 1e-12, so lets check!
            //println!("{:+e} {:+e}", diff.abs(), diff_v.abs());
            assert!(diff.abs() < 1e-11);
            assert!(diff_v.abs() < 1e-11);
        });

        Ok(())
    }
}
