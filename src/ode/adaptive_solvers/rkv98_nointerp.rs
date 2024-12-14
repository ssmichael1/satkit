use super::RKAdaptive;

// File below is auto-generated via python script that parses
// data available on web at:
// <https://www.sfu.ca/~jverner/RKV98.IIa.Robust.000000351.081209.CoeffsOnlyFLOAT6040>

use super::rkv98_nointerp_table as bt;
use crate::ode::types::{ODEError, ODEResult, ODESolution, ODEState};

pub struct RKV98NoInterp {}

const N: usize = 16;

impl RKAdaptive<N, 1> for RKV98NoInterp {
    const ORDER: usize = 9;

    const FSAL: bool = false;

    const B: [f64; N] = bt::B;

    const C: [f64; N] = bt::C;

    const A: [[f64; N]; N] = bt::A;

    const BI: [[f64; 1]; N] = bt::BI;

    const BERR: [f64; N] = {
        let mut berr = [0.0; N];
        let mut ix: usize = 0;
        while ix < N {
            berr[ix] = Self::B[ix] - bt::BHAT[ix];
            ix += 1;
        }
        berr
    };

    fn interpolate<S: ODEState>(_xinterp: f64, _sol: &ODESolution<S>) -> ODEResult<S> {
        ODEError::InterpNotImplemented.into()
    }
}
