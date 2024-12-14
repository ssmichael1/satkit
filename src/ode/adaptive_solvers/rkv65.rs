use super::RKAdaptive;

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
