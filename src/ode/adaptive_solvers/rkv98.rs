use super::RKAdaptive;

// Coefficients for the "efficient" Verner RK 9(8) pair
// with 9th-order interpolation
//
// Source:
// <https://www.sfu.ca/~jverner/RKV98.IIa.Efficient.00000034399.240714.CoeffsOnlyFLOAT6040>

use super::rkv98_efficient_table as bt;

// 17 stepping stages + 9 interpolation stages = 26 total
const N: usize = 26;
const NI: usize = 9;

pub struct RKV98 {}

impl RKAdaptive<N, NI> for RKV98 {
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
