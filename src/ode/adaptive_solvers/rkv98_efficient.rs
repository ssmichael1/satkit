use super::RKAdaptive;

// Coefficients from:
// RKV98.IIa.Efficient.00000034399.240714.CoeffsOnlyFLOAT6040.txt
// Efficient 26-stage 9(8) Runge-Kutta pair with order 9 interpolant

use super::rkv98_efficient_table as bt;
const N: usize = 26;
const NI: usize = 9;

pub struct RKV98Efficient {}

impl RKAdaptive<N, NI> for RKV98Efficient {
    const ORDER: usize = 9;

    const FSAL: bool = false;

    const B: [f64; N] = bt::B;

    const C: [f64; N] = bt::C;

    const A: [[f64; N]; N] = bt::A;

    const BI: [[f64; NI]; N] = bt::BI;

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
