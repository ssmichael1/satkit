//! Tsitouras Order 5(4) Runge-Kutta integrator
//!
//! See:
//! <https://dblp.org/rec/journals/cma/Tsitouras11>
//!
//! Note: in paper, sign is reversed on Bhat[7] ...
//! should be -1.0/66.0, not 1.0/66.0
//!
//! Note also: bhat values are actually error values
//! The nomenclature is confusing
//!

use super::RKAdaptive;

const A32: f64 = 0.335_480_655_492_357;
const A42: f64 = -6.359448489975075;
const A52: f64 = -11.74888356406283;
const A43: f64 = 4.362295432869581;
const A53: f64 = 7.495539342889836;
const A54: f64 = -0.09249506636175525;
const A62: f64 = -12.92096931784711;
const A63: f64 = 8.159367898576159;
const A64: f64 = -0.071_584_973_281_401;
const A65: f64 = -0.02826905039406838;

const BI11: f64 = -1.0530884977290216;
const BI12: f64 = -1.329_989_018_975_141;
const BI13: f64 = -1.4364028541716351;
const BI14: f64 = 0.7139816917074209;

const BI21: f64 = 0.1017;
const BI22: f64 = -2.1966568338249754;
const BI23: f64 = 1.294_985_250_737_463;

const BI31: f64 = 2.490_627_285_651_253;
const BI32: f64 = -2.385_356_454_720_616_5;
const BI33: f64 = 1.578_034_682_080_924_8;

const BI41: f64 = -16.548_102_889_244_902;
const BI42: f64 = -1.217_129_272_955_332_5;
const BI43: f64 = -0.616_204_060_378_000_9;

const BI51: f64 = 47.379_521_962_819_28;
const BI52: f64 = -1.203_071_208_372_362_7;
const BI53: f64 = -0.658_047_292_653_547_3;

const BI61: f64 = -34.870_657_861_496_61;
const BI62: f64 = -1.2;
const BI63: f64 = -2.0 / 3.0;

const BI71: f64 = 2.5;
const BI72: f64 = -1.0;
const BI73: f64 = -0.6;

pub struct RKTS54 {}

impl RKTS54 {}

impl RKAdaptive<7, 4> for RKTS54 {
    const C: [f64; 7] = [0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0];

    const B: [f64; 7] = [
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ];

    const BERR: [f64; 7] = [
        0.001780011052226,
        0.000816434459657,
        -0.007880878010262,
        0.144711007173263,
        -0.582357165452555,
        0.458082105929187,
        -1.0 / 66.0,
    ];

    const A: [[f64; 7]; 7] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [Self::C[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [Self::C[2] - A32, A32, 0.0, 0.0, 0.0, 0.0, 0.0],
        [Self::C[3] - A42 - A43, A42, A43, 0.0, 0.0, 0.0, 0.0],
        [Self::C[4] - A52 - A53 - A54, A52, A53, A54, 0.0, 0.0, 0.0],
        [
            Self::C[5] - A62 - A63 - A64 - A65,
            A62,
            A63,
            A64,
            A65,
            0.0,
            0.0,
        ],
        Self::B,
    ];

    const ORDER: usize = 5;

    const FSAL: bool = false;

    // From expanding expressions in Tsitorous paper...
    const BI: [[f64; 4]; 7] = [
        [
            BI11 * BI12 * BI14,
            BI11 * (BI14 + BI12 * BI13),
            BI11 * (BI13 + BI12),
            BI11,
        ],
        [0.0, BI21 * BI23, BI21 * BI22, BI21],
        [0.0, BI31 * BI33, BI31 * BI32, BI31],
        [0.0, BI41 * BI42 * BI43, BI41 * (BI42 + BI43), BI41],
        [0.0, BI51 * BI52 * BI53, BI51 * (BI52 + BI53), BI51],
        [0.0, BI61 * BI62 * BI63, BI61 * (BI62 + BI63), BI61],
        [0.0, BI71 * BI72 * BI73, BI71 * (BI72 + BI73), BI71],
    ];
}
