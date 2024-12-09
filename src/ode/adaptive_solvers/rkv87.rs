use super::RKAdaptive;

// File below is auto-generated via python script that parses
// data available on web at:
// <https://www.sfu.ca/~jverner/RKV87.IIa.Robust.00000754677.081208.CoeffsOnlyFLOAT>

use super::rkv87_table;

pub struct RKV87 {}

impl RKAdaptive<17, 7> for RKV87 {
    const ORDER: usize = 8;

    const FSAL: bool = false;

    const B: [f64; 17] = rkv87_table::B;

    const C: [f64; 17] = rkv87_table::C;

    const A: [[f64; 17]; 17] = rkv87_table::A;

    const BI: [[f64; 7]; 17] = rkv87_table::BI;

    const BERR: [f64; 17] = {
        let mut berr = [0.0; 17];
        let mut ix: usize = 0;
        while ix < 17 {
            berr[ix] = Self::B[ix] - rkv87_table::BHAT[ix];
            ix += 1;
        }
        berr
    };
}
