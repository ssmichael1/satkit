use super::RKAdaptive;

// Coefficients for the "efficient" Verner RK 8(7) pair
// with 8th-order interpolation
//
// Source:
// <https://www.sfu.ca/~jverner/RKV87.IIa.Efficient.000000282866.081208.CoeffsOnlyFLOAT>

use super::rkv87_table as bt;

// 13 stepping stages + 8 interpolation stages = 21 total
const N: usize = 21;
const NI: usize = 8;

pub struct RKV87 {}

impl RKAdaptive<N, NI> for RKV87 {
    const ORDER: usize = 8;

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
