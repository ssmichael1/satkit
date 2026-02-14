//! Native Rust implementation of the NRLMSIS 2.1 neutral atmosphere model.
//!
//! This module provides:
//! - A modern SI/MKS interface via [`msiscalc`]
//! - Legacy-compatible wrappers via [`gtd8d`] / [`gtd8d_legacy`]
//! - Geometric/geopotential conversion helpers via [`alt2gph`] and [`gph2alt`]
//!
//! Unless otherwise noted, units follow the original NRLMSIS 2.1 SI formulation.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const DMISSING: f64 = 9.999e-38;
const PI: f64 = std::f64::consts::PI;
const DEG2RAD: f64 = PI / 180.0;
const DOY2RAD: f64 = 2.0 * PI / 365.0;
const LST2RAD: f64 = PI / 12.0;
const TANH1: f64 = 0.761_594_155_955_765_5;
const KB: f64 = 1.380_649e-23;
const NA: f64 = 6.022_140_76e23;
const G0: f64 = 9.806_65;

const NSPEC: usize = 11;
const ND: usize = 27;
const P: usize = 4;
const NL: usize = ND - P;
const NLS: usize = 9;
const BWALT: f64 = 122.5;
const ZETA_F: f64 = 70.0;
const ZETA_B: f64 = BWALT;
const ZETA_A: f64 = 85.0;
const ZETA_GAMMA: f64 = 100.0;
const HGAMMA: f64 = 1.0 / 30.0;
const IZFMX: usize = 13;
const IZFX: usize = 14;
const IZAX: usize = 17;
const ITEX: usize = NL;
const ITGB0: usize = NL - 1;
const ITB0: usize = NL - 2;

const NDO1: usize = 13;
const NSPLO1: usize = NDO1 - 5;
const NDNO: usize = 13;
const NSPLNO: usize = NDNO - 5;

const MAXNBF: usize = 512;
const AMAXN: usize = 6;
const AMAXS: usize = 2;
const TMAXL: usize = 3;
const TMAXN: usize = 6;
const TMAXS: usize = 2;
const PMAXM: usize = 2;
const PMAXN: usize = 6;
const PMAXS: usize = 2;
const NSFX: usize = 5;
const NSFXMOD: usize = 5;
const NMAG: usize = 54;
const NUT: usize = 12;
const CTIMEIND: usize = 0;
const CINTANN: usize = CTIMEIND + (AMAXN + 1);
const CTIDE: usize = CINTANN + ((AMAXN + 1) * 2 * AMAXS);
const CSPW: usize = CTIDE + (4 * TMAXS + 2) * (TMAXL * (TMAXN + 1) - (TMAXL * (TMAXL + 1)) / 2);
const CSFX: usize = CSPW + (4 * PMAXS + 2) * (PMAXM * (PMAXN + 1) - (PMAXM * (PMAXM + 1)) / 2);
const MBF: usize = 383;
const CNONLIN: usize = MBF + 1;
const CSFXMOD: usize = CNONLIN;
const CMAG: usize = CSFXMOD + NSFXMOD;
const CUT: usize = CMAG + NMAG;

const G0DIVKB: f64 = G0 / KB * 1.0e3;
const MBAR: f64 = 28.96546 / (1.0e3 * NA);
const MBARG0DIVKB: f64 = MBAR * G0 / KB * 1.0e3;
const LNP0: f64 = 11.515_614;
const ZETAREF_O1: f64 = ZETA_A;
const ZETAREF_NO: f64 = ZETA_B;
const ZETAREF_OA: f64 = ZETA_B;
const TOA: f64 = 4000.0;
const HOA: f64 = (KB * TOA) / ((16.0 / (1.0e3 * NA)) * G0) * 1.0e-3;

const SPECMASS: [f64; 11] = [
    0.0,
    0.0,
    28.0134 / (1.0e3 * NA),
    31.9988 / (1.0e3 * NA),
    (31.9988 / 2.0) / (1.0e3 * NA),
    4.0 / (1.0e3 * NA),
    1.0 / (1.0e3 * NA),
    39.948 / (1.0e3 * NA),
    (28.0134 / 2.0) / (1.0e3 * NA),
    (31.9988 / 2.0) / (1.0e3 * NA),
    ((28.0134 + 31.9988) / 2.0) / (1.0e3 * NA),
];

const LNVMR: [f64; 11] = [
    0.0,
    0.0,
    -0.247_374_770_362_953_83,
    -1.563_556_737_177_912,
    0.0,
    -12.166_851_932_376_892,
    0.0,
    -4.674_305_924_822_954,
    0.0,
    0.0,
    0.0,
];

const NODES_TN: [f64; ND + 3] = [
    -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
    65.0, 70.0, 75.0, 80.0, 85.0, 92.5, 102.5, 112.5, 122.5, 132.5, 142.5, 152.5, 162.5, 172.5,
];
const NODES_O1: [f64; NDO1 + 1] = [
    35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 92.5, 102.5, 112.5,
];
const NODES_NO: [f64; NDNO + 1] = [
    47.5, 55.0, 62.5, 70.0, 77.5, 85.0, 92.5, 100.0, 107.5, 115.0, 122.5, 130.0, 137.5, 145.0,
];

const C2TN: [[f64; 3]; 3] = [
    [1.0, 1.0, 1.0],
    [-10.0, 0.0, 10.0],
    [
        33.333_333_333_333_336,
        -16.666_666_666_668,
        33.333_333_333_333_336,
    ],
];
const C1O1: [[f64; 2]; 2] = [
    [1.75, -1.624_999_900_076_852],
    [-2.916_666_573_405_061, 21.458_332_647_194_382],
];
const C1O1ADJ: [f64; 2] = [0.257_142_857_142_857, -0.102_857_142_686_844];
const C1NO: [[f64; 2]; 2] = [[1.5, 0.0], [-3.75, 15.0]];
const C1NOADJ: [f64; 2] = [0.166_666_666_666_667, -0.066_666_666_666_667];

const GWHT: [f64; 4] = [5.0 / 24.0, 55.0 / 24.0, 55.0 / 24.0, 5.0 / 24.0];
const WBETA: [f64; NL + 1] = {
    let mut arr = [0.0; NL + 1];
    let mut j = 0;
    while j <= NL {
        arr[j] = (NODES_TN[j + 4] - NODES_TN[j]) / 4.0;
        j += 1;
    }
    arr
};
const WGAMMA: [f64; NL + 1] = {
    let mut arr = [0.0; NL + 1];
    let mut j = 0;
    while j <= NL {
        arr[j] = (NODES_TN[j + 5] - NODES_TN[j]) / 5.0;
        j += 1;
    }
    arr
};

const S5_ZETA_B: [f64; 4] = [
    0.041_666_666_666_667,
    0.458_333_333_333_333,
    0.458_333_333_333_333,
    0.041_666_666_666_667,
];
const S6_ZETA_B: [f64; 5] = [
    0.008_771_929_824_561,
    0.216_228_070_175_439,
    0.55,
    0.216_666_666_666_667,
    0.008_333_333_333_333,
];
const WGHT_AX_DZ: [f64; 3] = [-0.102_857_142_857, 0.049_523_809_523_8, 0.053_333_333_333];
const S4_ZETA_A: [f64; 3] = [
    0.257_142_857_142_857,
    0.653_968_253_968_254,
    0.088_888_888_888_889,
];
const S5_ZETA_A: [f64; 4] = [
    0.085_714_285_714_286,
    0.587_590_187_590_188,
    0.313_020_313_020_313,
    0.013_675_213_675_214,
];
const S6_ZETA_A: [f64; 5] = [
    0.023_376_623_376_623,
    0.378_732_378_732_379,
    0.500_743_700_743_701,
    0.095_538_448_479_625,
    0.001_608_848_667_672,
];
const S4_ZETA_F: [f64; 3] = [
    0.166_666_666_666_667,
    0.666_666_666_666_667,
    0.166_666_666_666_667,
];
const S5_ZETA_F: [f64; 4] = [
    0.041_666_666_666_667,
    0.458_333_333_333_333,
    0.458_333_333_333_333,
    0.041_666_666_666_667,
];
const S5_ZETA_0: [f64; 3] = [
    0.458_333_333_333_333,
    0.458_333_333_333_333,
    0.041_666_666_666_667,
];

#[derive(Clone)]
struct Subset {
    nlev: usize,
    beta: Vec<f64>,
}

impl Subset {
    fn new(nlev: usize) -> Self {
        Self {
            nlev,
            beta: vec![0.0; MAXNBF * nlev],
        }
    }

    fn idx(&self, bf: usize, lev: usize) -> usize {
        bf + MAXNBF * lev
    }

    fn get(&self, bf: usize, lev: usize) -> f64 {
        self.beta[self.idx(bf, lev)]
    }

    fn set(&mut self, bf: usize, lev: usize, value: f64) {
        let index = self.idx(bf, lev);
        self.beta[index] = value;
    }

    fn fill_from_parm(&mut self, raw: &[f64], start_col: usize) {
        for lev in 0..self.nlev {
            let src_col = start_col + lev;
            let src0 = src_col * MAXNBF;
            let dst0 = lev * MAXNBF;
            self.beta[dst0..dst0 + MAXNBF].copy_from_slice(&raw[src0..src0 + MAXNBF]);
        }
    }
}

#[derive(Clone)]
struct Model {
    tn: Subset,
    pr: Subset,
    n2: Subset,
    o2: Subset,
    o1: Subset,
    he: Subset,
    h1: Subset,
    ar: Subset,
    n1: Subset,
    oa: Subset,
    no: Subset,
    swg: [bool; MAXNBF],
    smod: [bool; NL + 1],
    zsfx: [bool; MBF + 1],
    tsfx: [bool; MBF + 1],
    psfx: [bool; MBF + 1],
    eta_tn: [[f64; 7]; 31],
    eta_o1: [[f64; 7]; 31],
    eta_no: [[f64; 7]; 31],
    hr_fact_o1_ref: f64,
    dhr_fact_o1_ref: f64,
    hr_fact_no_ref: f64,
    dhr_fact_no_ref: f64,
    masswgt: [f64; NSPEC],
    n2r_flag: bool,
}

fn default_parm_path() -> Option<PathBuf> {
    let cands = [
        crate::utils::datadir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("msis21.parm"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("nrlmsis2/msis21.parm"),
        PathBuf::from("nrlmsis2/msis21.parm"),
        PathBuf::from("msis21.parm"),
    ];
    cands.into_iter().find(|path| path.exists())
}

fn read_parm_file(path: &Path) -> Result<Vec<f64>, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err("invalid parameter file length".to_string());
    }
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut array = [0_u8; 8];
        array.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(array));
    }
    Ok(values)
}

impl Model {
    fn new(path: &Path) -> Result<Self, String> {
        let raw = read_parm_file(path)?;
        let expected = MAXNBF * 131;
        if raw.len() != expected {
            return Err(format!(
                "unexpected parameter count: got {}, expected {}",
                raw.len(),
                expected
            ));
        }

        let mut model = Self {
            tn: Subset::new(NL + 1),
            pr: Subset::new(NL + 1),
            n2: Subset::new(NLS + 1),
            o2: Subset::new(NLS + 1),
            o1: Subset::new(NLS + NSPLO1 + 1),
            he: Subset::new(NLS + 1),
            h1: Subset::new(NLS + 1),
            ar: Subset::new(NLS + 1),
            n1: Subset::new(NLS + 1),
            oa: Subset::new(NLS + 1),
            no: Subset::new(NLS + NSPLNO + 1),
            swg: [true; MAXNBF],
            smod: [false; NL + 1],
            zsfx: [false; MBF + 1],
            tsfx: [false; MBF + 1],
            psfx: [false; MBF + 1],
            eta_tn: [[0.0; 7]; 31],
            eta_o1: [[0.0; 7]; 31],
            eta_no: [[0.0; 7]; 31],
            hr_fact_o1_ref: 0.0,
            dhr_fact_o1_ref: 0.0,
            hr_fact_no_ref: 0.0,
            dhr_fact_no_ref: 0.0,
            masswgt: [0.0; NSPEC],
            n2r_flag: false,
        };

        let mut c = 0;
        model.tn.fill_from_parm(&raw, c);
        c += model.tn.nlev;

        for bf in 0..MAXNBF {
            model.pr.set(bf, 0, raw[bf + MAXNBF * c]);
        }
        c += 1;

        model.n2.fill_from_parm(&raw, c);
        c += model.n2.nlev;
        model.o2.fill_from_parm(&raw, c);
        c += model.o2.nlev;
        model.o1.fill_from_parm(&raw, c);
        c += model.o1.nlev;
        model.he.fill_from_parm(&raw, c);
        c += model.he.nlev;
        model.h1.fill_from_parm(&raw, c);
        c += model.h1.nlev;
        model.ar.fill_from_parm(&raw, c);
        c += model.ar.nlev;
        model.n1.fill_from_parm(&raw, c);
        c += model.n1.nlev;
        model.oa.fill_from_parm(&raw, c);
        c += model.oa.nlev;
        model.no.fill_from_parm(&raw, c);

        model.zsfx[9] = true;
        model.zsfx[10] = true;
        model.zsfx[13] = true;
        model.zsfx[14] = true;
        model.zsfx[17] = true;
        model.zsfx[18] = true;
        for j in CTIDE..CSPW {
            model.tsfx[j] = true;
        }
        for j in CSPW..=(CSPW + 59) {
            model.psfx[j] = true;
        }

        for k in 2..=6 {
            for j in 0..=NL {
                model.eta_tn[j][k] = 1.0 / (NODES_TN[j + k - 1] - NODES_TN[j]);
            }
        }
        for k in 2..=4 {
            for j in 0..=(NDO1 - k + 1) {
                model.eta_o1[j][k] = 1.0 / (NODES_O1[j + k - 1] - NODES_O1[j]);
            }
            for j in 0..=(NDNO - k + 1) {
                model.eta_no[j][k] = 1.0 / (NODES_NO[j + k - 1] - NODES_NO[j]);
            }
        }

        let gammaterm_o1 = ((ZETAREF_O1 - ZETA_GAMMA) * HGAMMA).tanh();
        model.hr_fact_o1_ref = 0.5 * (1.0 + gammaterm_o1);
        model.dhr_fact_o1_ref = (1.0 - (ZETAREF_O1 - ZETA_GAMMA) * (1.0 - gammaterm_o1) * HGAMMA)
            / model.hr_fact_o1_ref;

        let gammaterm_no = ((ZETAREF_NO - ZETA_GAMMA) * HGAMMA).tanh();
        model.hr_fact_no_ref = 0.5 * (1.0 + gammaterm_no);
        model.dhr_fact_no_ref = (1.0 - (ZETAREF_NO - ZETA_GAMMA) * (1.0 - gammaterm_no) * HGAMMA)
            / model.hr_fact_no_ref;

        for iz in 0..=NL {
            model.smod[iz] = model.tn.get(CSFXMOD, iz) != 0.0
                || model.tn.get(CSFXMOD + 1, iz) != 0.0
                || model.tn.get(CSFXMOD + 2, iz) != 0.0;
        }

        model.press_parm();

        for ispec in 1..=10 {
            model.masswgt[ispec] = SPECMASS[ispec];
        }
        model.masswgt[1] = 0.0;
        model.masswgt[10] = 0.0;

        Ok(model)
    }

    fn press_parm(&mut self) {
        for j in 0..=MBF {
            let mut lnz = 0.0;
            for (b, w) in GWHT.iter().enumerate() {
                lnz += self.tn.get(j, b) * *w * MBARG0DIVKB;
            }
            self.pr.set(j, 1, -lnz);
            for iz in 1..=IZFMX {
                let mut lnz_iz = 0.0;
                for (b, w) in GWHT.iter().enumerate() {
                    lnz_iz += self.tn.get(j, iz + b) * *w * MBARG0DIVKB;
                }
                let val = self.pr.get(j, iz) - lnz_iz;
                self.pr.set(j, iz + 1, val);
            }
        }
    }

    fn with_legacy_switches(&self, switch_legacy: [f64; 25]) -> Self {
        let mut model = self.clone();
        model.swg = legacy_switches_to_swg(switch_legacy);
        model
    }
}

fn set_range(swg: &mut [bool; MAXNBF], start: usize, end: usize, value: bool) {
    for slot in swg.iter_mut().take(end + 1).skip(start) {
        *slot = value;
    }
}

fn set_indices(swg: &mut [bool; MAXNBF], indices: &[usize], value: bool) {
    for &index in indices {
        swg[index] = value;
    }
}

fn legacy_switches_to_swg(sv: [f64; 25]) -> [bool; MAXNBF] {
    let mut swg = [true; MAXNBF];
    let mut swleg = [0.0_f64; 25];
    let mut swc = [0.0_f64; 25];

    for i in 0..25 {
        swleg[i] = sv[i] % 2.0;
        let a = sv[i].abs();
        swc[i] = if (a - 1.0).abs() < f64::EPSILON || (a - 2.0).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }

    swg[0] = true;
    set_range(
        &mut swg,
        CSFX,
        CSFX + NSFX - 1,
        (swleg[0] - 1.0).abs() < f64::EPSILON,
    );
    swg[310] = (swleg[0] - 1.0).abs() < f64::EPSILON;

    set_range(&mut swg, 1, 6, (swleg[1] - 1.0).abs() < f64::EPSILON);
    set_range(&mut swg, 304, 305, (swleg[1] - 1.0).abs() < f64::EPSILON);
    set_range(&mut swg, 311, 312, (swleg[1] - 1.0).abs() < f64::EPSILON);
    set_range(&mut swg, 313, 314, (swleg[1] - 1.0).abs() < f64::EPSILON);

    set_indices(
        &mut swg,
        &[7, 8, 11, 12, 15, 16, 19, 20],
        (swleg[2] - 1.0).abs() < f64::EPSILON,
    );
    set_range(&mut swg, 306, 307, (swleg[2] - 1.0).abs() < f64::EPSILON);

    set_indices(
        &mut swg,
        &[21, 22, 25, 26, 29, 30, 33, 34],
        (swleg[3] - 1.0).abs() < f64::EPSILON,
    );
    set_range(&mut swg, 308, 309, (swleg[3] - 1.0).abs() < f64::EPSILON);

    set_indices(
        &mut swg,
        &[9, 10, 13, 14, 17, 18],
        (swleg[4] - 1.0).abs() < f64::EPSILON,
    );
    set_indices(
        &mut swg,
        &[23, 24, 27, 28, 31, 32],
        (swleg[5] - 1.0).abs() < f64::EPSILON,
    );

    set_range(&mut swg, 35, 94, (swleg[6] - 1.0).abs() < f64::EPSILON);
    set_range(&mut swg, 300, 303, (swleg[6] - 1.0).abs() < f64::EPSILON);

    set_range(&mut swg, 95, 144, (swleg[7] - 1.0).abs() < f64::EPSILON);
    set_range(&mut swg, 145, 184, (swleg[13] - 1.0).abs() < f64::EPSILON);

    swg[CMAG] = false;
    swg[CMAG + 1] = false;
    if swleg[8] > 0.0 || (swleg[12] - 1.0).abs() < f64::EPSILON {
        swg[CMAG] = true;
        swg[CMAG + 1] = true;
    }
    if swleg[8] < 0.0 {
        swg[CMAG] = false;
        swg[CMAG + 1] = true;
    }
    set_range(
        &mut swg,
        CMAG + 2,
        CMAG + 12,
        (swleg[8] - 1.0).abs() < f64::EPSILON,
    );
    set_range(
        &mut swg,
        CMAG + 28,
        CMAG + 40,
        (swleg[8] + 1.0).abs() < f64::EPSILON,
    );

    set_range(
        &mut swg,
        CSPW,
        CSFX - 1,
        (swleg[10] - 1.0).abs() < f64::EPSILON && (swleg[9] - 1.0).abs() < f64::EPSILON,
    );
    set_range(
        &mut swg,
        CUT,
        CUT + NUT - 1,
        (swleg[11] - 1.0).abs() < f64::EPSILON && (swleg[9] - 1.0).abs() < f64::EPSILON,
    );
    set_range(
        &mut swg,
        CMAG + 13,
        CMAG + 25,
        (swleg[12] - 1.0).abs() < f64::EPSILON && (swleg[9] - 1.0).abs() < f64::EPSILON,
    );
    set_range(
        &mut swg,
        CMAG + 41,
        CMAG + 53,
        (swleg[12] - 1.0).abs() < f64::EPSILON && (swleg[9] - 1.0).abs() < f64::EPSILON,
    );

    if swc[0] == 0.0 {
        set_range(&mut swg, CSFXMOD, CSFXMOD + NSFXMOD - 1, false);
        set_range(&mut swg, 302, 303, false);
        set_range(&mut swg, 304, 305, false);
        set_range(&mut swg, 306, 307, false);
        set_range(&mut swg, 308, 309, false);
        set_range(&mut swg, 311, 314, false);
        swg[447] = false;
        swg[454] = false;
    } else {
        set_range(&mut swg, CSFXMOD, CSFXMOD + NSFXMOD - 1, true);
    }

    if swc[1] == 0.0 {
        set_range(&mut swg, 9, 20, false);
        set_range(&mut swg, 23, 34, false);
        set_range(&mut swg, 35, 184, false);
        set_range(&mut swg, 185, 294, false);
        set_range(&mut swg, 392, 414, false);
        set_range(&mut swg, 420, 442, false);
        set_range(&mut swg, 449, 453, false);
    }
    if swc[2] == 0.0 {
        set_range(&mut swg, 201, 204, false);
        set_range(&mut swg, 209, 212, false);
        set_range(&mut swg, 217, 220, false);
        set_range(&mut swg, 255, 258, false);
        set_range(&mut swg, 263, 266, false);
        set_range(&mut swg, 271, 274, false);
        set_range(&mut swg, 306, 307, false);
    }
    if swc[3] == 0.0 {
        set_range(&mut swg, 225, 228, false);
        set_range(&mut swg, 233, 236, false);
        set_range(&mut swg, 241, 244, false);
        set_range(&mut swg, 275, 278, false);
        set_range(&mut swg, 283, 286, false);
        set_range(&mut swg, 291, 294, false);
        set_range(&mut swg, 308, 309, false);
    }
    if swc[4] == 0.0 {
        set_range(&mut swg, 47, 50, false);
        set_range(&mut swg, 51, 54, false);
        set_range(&mut swg, 55, 58, false);
        set_range(&mut swg, 59, 62, false);
        set_range(&mut swg, 63, 66, false);
        set_range(&mut swg, 67, 70, false);
        set_range(&mut swg, 105, 108, false);
        set_range(&mut swg, 109, 112, false);
        set_range(&mut swg, 113, 116, false);
        set_range(&mut swg, 117, 120, false);
        set_range(&mut swg, 121, 124, false);
        set_range(&mut swg, 153, 156, false);
        set_range(&mut swg, 157, 160, false);
        set_range(&mut swg, 161, 164, false);
        set_range(&mut swg, 165, 168, false);
        set_range(&mut swg, 197, 200, false);
        set_range(&mut swg, 205, 208, false);
        set_range(&mut swg, 213, 216, false);
        set_range(&mut swg, 259, 262, false);
        set_range(&mut swg, 267, 270, false);
        set_range(&mut swg, 394, 397, false);
        set_range(&mut swg, 407, 410, false);
        set_range(&mut swg, 422, 425, false);
        set_range(&mut swg, 435, 438, false);
        swg[446] = false;
    }
    if swc[5] == 0.0 {
        set_range(&mut swg, 221, 224, false);
        set_range(&mut swg, 229, 232, false);
        set_range(&mut swg, 237, 240, false);
        set_range(&mut swg, 279, 282, false);
        set_range(&mut swg, 287, 290, false);
    }
    if swc[6] == 0.0 {
        set_range(&mut swg, 398, 401, false);
        set_range(&mut swg, 426, 429, false);
    }
    if swc[10] == 0.0 {
        set_range(&mut swg, 402, 410, false);
        set_range(&mut swg, 430, 438, false);
        set_range(&mut swg, 452, 453, false);
    }
    if swc[11] == 0.0 {
        set_range(&mut swg, 411, 414, false);
        set_range(&mut swg, 439, 440, false);
    }

    swg
}

static MODEL: OnceLock<Result<Model, String>> = OnceLock::new();

fn model() -> Result<&'static Model, String> {
    let res = MODEL.get_or_init(|| {
        let path = default_parm_path().ok_or_else(|| "msis21.parm not found".to_string())?;
        Model::new(&path)
    });
    match res {
        Ok(model) => Ok(model),
        Err(error) => Err(error.clone()),
    }
}

fn dot_subset_level(subset: &Subset, level: usize, gf: &[f64; MAXNBF], imax: usize) -> f64 {
    let mut sum = 0.0;
    for (i, gfi) in gf.iter().enumerate().take(imax + 1) {
        sum += subset.get(i, level) * *gfi;
    }
    sum
}

fn bspline(
    x: f64,
    nodes: &[f64],
    nd: usize,
    kmax: usize,
    eta: &[[f64; 7]; 31],
) -> ([[f64; 7]; 6], i32) {
    let mut s = [[0.0; 7]; 6];

    if x >= nodes[nd] {
        return (s, nd as i32);
    }
    if x <= nodes[0] {
        return (s, -1);
    }

    let mut low = 0usize;
    let mut high = nd;
    let mut i = (low + high) / 2;
    while x < nodes[i] || x >= nodes[i + 1] {
        if x < nodes[i] {
            high = i;
        } else {
            low = i;
        }
        i = (low + high) / 2;
    }

    let lidx = |l: i32| -> usize { (l + 5) as usize };

    s[lidx(0)][2] = (x - nodes[i]) * eta[i][2];
    if i > 0 {
        s[lidx(-1)][2] = 1.0 - s[lidx(0)][2];
    }
    if i >= nd - 1 {
        s[lidx(0)][2] = 0.0;
    }

    let mut w = [0.0; 5];
    let widx = |l: i32| -> usize { (l + 4) as usize };

    w[widx(0)] = (x - nodes[i]) * eta[i][3];
    if i != 0 {
        w[widx(-1)] = (x - nodes[i - 1]) * eta[i - 1][3];
    }
    if i < (nd - 2) {
        s[lidx(0)][3] = w[widx(0)] * s[lidx(0)][2];
    }
    if (i as i32 - 1) >= 0 && (i - 1) < (nd - 2) {
        s[lidx(-1)][3] = w[widx(-1)] * s[lidx(-1)][2] + (1.0 - w[widx(0)]) * s[lidx(0)][2];
    }
    if (i as i32 - 2) >= 0 {
        s[lidx(-2)][3] = (1.0 - w[widx(-1)]) * s[lidx(-1)][2];
    }

    for l in [-0_i32, -1, -2] {
        let j = i as i32 + l;
        if j < 0 {
            break;
        }
        w[widx(l)] = (x - nodes[j as usize]) * eta[j as usize][4];
    }
    if i < (nd - 3) {
        s[lidx(0)][4] = w[widx(0)] * s[lidx(0)][3];
    }
    for l in [-1_i32, -2] {
        let j = i as i32 + l;
        if j >= 0 && (j as usize) < (nd - 3) {
            s[lidx(l)][4] = w[widx(l)] * s[lidx(l)][3] + (1.0 - w[widx(l + 1)]) * s[lidx(l + 1)][3];
        }
    }
    if (i as i32 - 3) >= 0 {
        s[lidx(-3)][4] = (1.0 - w[widx(-2)]) * s[lidx(-2)][3];
    }

    for l in [-0_i32, -1, -2, -3] {
        let j = i as i32 + l;
        if j < 0 {
            break;
        }
        w[widx(l)] = (x - nodes[j as usize]) * eta[j as usize][5];
    }
    if i < (nd - 4) {
        s[lidx(0)][5] = w[widx(0)] * s[lidx(0)][4];
    }
    for l in [-1_i32, -2, -3] {
        let j = i as i32 + l;
        if j >= 0 && (j as usize) < (nd - 4) {
            s[lidx(l)][5] = w[widx(l)] * s[lidx(l)][4] + (1.0 - w[widx(l + 1)]) * s[lidx(l + 1)][4];
        }
    }
    if (i as i32 - 4) >= 0 {
        s[lidx(-4)][5] = (1.0 - w[widx(-3)]) * s[lidx(-3)][4];
    }
    if kmax == 5 {
        return (s, i as i32);
    }

    for l in [-0_i32, -1, -2, -3, -4] {
        let j = i as i32 + l;
        if j < 0 {
            break;
        }
        w[widx(l)] = (x - nodes[j as usize]) * eta[j as usize][6];
    }
    if i < (nd - 5) {
        s[lidx(0)][6] = w[widx(0)] * s[lidx(0)][5];
    }
    for l in [-1_i32, -2, -3, -4] {
        let j = i as i32 + l;
        if j >= 0 && (j as usize) < (nd - 5) {
            s[lidx(l)][6] = w[widx(l)] * s[lidx(l)][5] + (1.0 - w[widx(l + 1)]) * s[lidx(l + 1)][5];
        }
    }
    if (i as i32 - 5) >= 0 {
        s[lidx(-5)][6] = (1.0 - w[widx(-4)]) * s[lidx(-4)][5];
    }

    (s, i as i32)
}

fn dilog(x0: f64) -> f64 {
    let pi2_6 = PI * PI / 6.0;
    let mut x = x0;
    if x > 0.5 {
        let lnx = x.ln();
        x = 1.0 - x;
        let xx = x * x;
        let x4 = 4.0 * x;
        pi2_6
            - lnx * x.ln()
            - (4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
                + x4
                + 3.0 * (1.0 - xx) * lnx)
                / (1.0 + x4 + xx)
    } else {
        let xx = x * x;
        let x4 = 4.0 * x;
        (4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
            + x4
            + 3.0 * (1.0 - xx) * (1.0 - x).ln())
            / (1.0 + x4 + xx)
    }
}

/// Convert geodetic altitude to geopotential height.
///
/// # Arguments
/// - `lat`: geodetic latitude in degrees
/// - `alt`: geodetic altitude in kilometers
///
/// # Returns
/// Geopotential height in kilometers.
pub fn alt2gph(lat: f64, alt: f64) -> f64 {
    let a: f64 = 6_378.137_0e3;
    let finv: f64 = 298.257_223_563;
    let w: f64 = 7_292_115e-11;
    let gm: f64 = 398_600.4418e9;

    let asq = a * a;
    let wsq = w * w;
    let f = 1.0 / finv;
    let esq = 2.0 * f - f * f;
    let e = esq.sqrt();
    let elin = a * e;
    let elinsq = elin * elin;
    let epr = e / (1.0 - f);
    let q0 = ((1.0 + 3.0 / (epr * epr)) * epr.atan() - 3.0 / epr) / 2.0;
    let u0 = -gm * epr.atan() / elin - wsq * asq / 3.0;
    let g0 = 9.80665;
    let gm_div_elin = gm / elin;

    let x0sq = 2e7_f64.powi(2);
    let hsq = 1.2e7_f64.powi(2);

    let altm = alt * 1000.0;
    let sinsqlat = (lat * DEG2RAD).sin().powi(2);
    let v = a / (1.0 - esq * sinsqlat).sqrt();
    let xsq = (v + altm).powi(2) * (1.0 - sinsqlat);
    let zsq = (v * (1.0 - esq) + altm).powi(2) * sinsqlat;
    let rsq_min_elinsq = xsq + zsq - elinsq;
    let usq = rsq_min_elinsq / 2.0 + (rsq_min_elinsq.powi(2) / 4.0 + elinsq * zsq).sqrt();
    let cossqdelta = zsq / usq;

    let epru = elin / usq.sqrt();
    let atanepru = epru.atan();
    let q = ((1.0 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0;
    let mut u = -gm_div_elin * atanepru - wsq * (asq * q * (cossqdelta - 1.0 / 3.0) / q0) / 2.0;

    let vc = if xsq <= x0sq {
        (wsq / 2.0) * xsq
    } else {
        (wsq / 2.0) * (hsq * ((xsq - x0sq) / hsq).tanh() + x0sq)
    };
    u -= vc;

    (u - u0) / g0 / 1000.0
}

/// Convert geopotential height to geodetic altitude.
///
/// Uses Newton iteration around [`alt2gph`].
///
/// # Arguments
/// - `theta`: geodetic latitude in degrees
/// - `gph`: geopotential height in kilometers
///
/// # Returns
/// Geodetic altitude in kilometers.
pub fn gph2alt(theta: f64, gph: f64) -> f64 {
    let epsilon: f64 = 0.0005;
    let mut x = gph;
    let mut n = 0;
    let mut dx: f64 = 2.0 * epsilon;
    while dx.abs() > epsilon && n < 10 {
        let y = alt2gph(theta, x);
        let dydz = (alt2gph(theta, x + dx) - y) / dx;
        dx = (gph - y) / dydz;
        x += dx;
        n += 1;
    }
    x
}

fn solzen(ddd: f64, lst: f64, lat: f64, lon: f64) -> f64 {
    let wlon = 360.0 - lon;
    let _ = ddd + (lst + wlon / 15.0) / 24.0 + 0.9369;
    let teqnx = ddd + 0.9369;
    let p = [
        0.017_203_534,
        0.034_407_068,
        0.051_610_602,
        0.068_814_136,
        0.103_221_204,
    ];

    let mut dec = 23.256 * (p[0] * (teqnx - 82.242)).sin()
        + 0.381 * (p[1] * (teqnx - 44.855)).sin()
        + 0.167 * (p[2] * (teqnx - 23.355)).sin()
        - 0.013 * (p[3] * (teqnx + 11.97)).sin()
        + 0.011 * (p[4] * (teqnx - 10.410)).sin()
        + 0.339_137;
    dec *= DEG2RAD;

    let tf = teqnx - 0.5;
    let teqt = -7.38 * (p[0] * (tf - 4.0)).sin() - 9.87 * (p[1] * (tf + 9.0)).sin()
        + 0.27 * (p[2] * (tf - 53.0)).sin()
        - 0.2 * (p[3] * (tf - 17.0)).cos();

    let phi = (PI / 12.0) * (lst - 12.0) + teqt * DEG2RAD / 4.0;
    let rlat = lat * DEG2RAD;

    let mut cosx = rlat.sin() * dec.sin() + rlat.cos() * dec.cos() * phi.cos();
    if cosx.abs() > 1.0 {
        cosx = cosx.signum();
    }
    cosx.acos() / DEG2RAD
}

fn globe(
    model: &Model,
    doy: f64,
    utsec: f64,
    lat: f64,
    lon: f64,
    sfluxavg: f64,
    sflux: f64,
    ap: [f64; 7],
) -> [f64; MAXNBF] {
    let mut bf = [0.0; MAXNBF];
    let mut plg = [[0.0; 7]; 7];

    let clat = (lat * DEG2RAD).sin();
    let slat = (lat * DEG2RAD).cos();
    let clat2 = clat * clat;
    let clat4 = clat2 * clat2;
    let slat2 = slat * slat;

    plg[0][0] = 1.0;
    plg[1][0] = clat;
    plg[2][0] = 0.5 * (3.0 * clat2 - 1.0);
    plg[3][0] = 0.5 * (5.0 * clat * clat2 - 3.0 * clat);
    plg[4][0] = (35.0 * clat4 - 30.0 * clat2 + 3.0) / 8.0;
    plg[5][0] = (63.0 * clat2 * clat2 * clat - 70.0 * clat2 * clat + 15.0 * clat) / 8.0;
    plg[6][0] = (11.0 * clat * plg[5][0] - 5.0 * plg[4][0]) / 6.0;

    plg[1][1] = slat;
    plg[2][1] = 3.0 * clat * slat;
    plg[3][1] = 1.5 * (5.0 * clat2 - 1.0) * slat;
    plg[4][1] = 2.5 * (7.0 * clat2 * clat - 3.0 * clat) * slat;
    plg[5][1] = 1.875 * (21.0 * clat4 - 14.0 * clat2 + 1.0) * slat;
    plg[6][1] = (11.0 * clat * plg[5][1] - 6.0 * plg[4][1]) / 5.0;

    plg[2][2] = 3.0 * slat2;
    plg[3][2] = 15.0 * slat2 * clat;
    plg[4][2] = 7.5 * (7.0 * clat2 - 1.0) * slat2;
    plg[5][2] = 3.0 * clat * plg[4][2] - 2.0 * plg[3][2];
    plg[6][2] = (11.0 * clat * plg[5][2] - 7.0 * plg[4][2]) / 4.0;

    plg[3][3] = 15.0 * slat2 * slat;
    plg[4][3] = 105.0 * slat2 * slat * clat;
    plg[5][3] = (9.0 * clat * plg[4][3] - 7.0 * plg[3][3]) / 2.0;
    plg[6][3] = (11.0 * clat * plg[5][3] - 8.0 * plg[4][3]) / 3.0;

    let cdoy = [(DOY2RAD * doy).cos(), (DOY2RAD * doy * 2.0).cos()];
    let sdoy = [(DOY2RAD * doy).sin(), (DOY2RAD * doy * 2.0).sin()];

    let lst = (utsec / 3600.0 + lon / 15.0 + 24.0).rem_euclid(24.0);
    let clst = [
        (LST2RAD * lst).cos(),
        (LST2RAD * lst * 2.0).cos(),
        (LST2RAD * lst * 3.0).cos(),
    ];
    let slst = [
        (LST2RAD * lst).sin(),
        (LST2RAD * lst * 2.0).sin(),
        (LST2RAD * lst * 3.0).sin(),
    ];

    let clon = [(DEG2RAD * lon).cos(), (DEG2RAD * lon * 2.0).cos()];
    let slon = [(DEG2RAD * lon).sin(), (DEG2RAD * lon * 2.0).sin()];

    let mut c = CTIMEIND;
    for row in plg.iter().take(AMAXN + 1) {
        bf[c] = row[0];
        c += 1;
    }

    for s in 1..=AMAXS {
        let cosdoy = cdoy[s - 1];
        let sindoy = sdoy[s - 1];
        for row in plg.iter().take(AMAXN + 1) {
            let pl = row[0];
            bf[c] = pl * cosdoy;
            bf[c + 1] = pl * sindoy;
            c += 2;
        }
    }

    for l in 1..=TMAXL {
        let coslst = clst[l - 1];
        let sinlst = slst[l - 1];
        for (n, row) in plg.iter().enumerate().take(TMAXN + 1).skip(l) {
            let _ = n;
            let pl = row[l];
            bf[c] = pl * coslst;
            bf[c + 1] = pl * sinlst;
            c += 2;
        }
        for s in 1..=TMAXS {
            let cosdoy = cdoy[s - 1];
            let sindoy = sdoy[s - 1];
            for row in plg.iter().take(TMAXN + 1).skip(l) {
                let pl = row[l];
                bf[c] = pl * coslst * cosdoy;
                bf[c + 1] = pl * sinlst * cosdoy;
                bf[c + 2] = pl * coslst * sindoy;
                bf[c + 3] = pl * sinlst * sindoy;
                c += 4;
            }
        }
    }

    for m in 1..=PMAXM {
        let coslon = clon[m - 1];
        let sinlon = slon[m - 1];
        for row in plg.iter().take(PMAXN + 1).skip(m) {
            let pl = row[m];
            bf[c] = pl * coslon;
            bf[c + 1] = pl * sinlon;
            c += 2;
        }
        for s in 1..=PMAXS {
            let cosdoy = cdoy[s - 1];
            let sindoy = sdoy[s - 1];
            for row in plg.iter().take(PMAXN + 1).skip(m) {
                let pl = row[m];
                bf[c] = pl * coslon * cosdoy;
                bf[c + 1] = pl * sinlon * cosdoy;
                bf[c + 2] = pl * coslon * sindoy;
                bf[c + 3] = pl * sinlon * sindoy;
                c += 4;
            }
        }
    }

    let dfa = sfluxavg - 150.0;
    let df = sflux - sfluxavg;
    bf[c] = dfa;
    bf[c + 1] = dfa * dfa;
    bf[c + 2] = df;
    bf[c + 3] = df * df;
    bf[c + 4] = df * dfa;
    c += NSFX;

    let sza = solzen(doy, lst, lat, lon);
    bf[c] = -0.5 * ((sza - 98.0) / 6.0).tanh();
    bf[c + 1] = -0.5 * ((sza - 101.5) / 20.0).tanh();
    bf[c + 2] = dfa * bf[c];
    bf[c + 3] = dfa * bf[c + 1];
    bf[c + 4] = dfa * plg[2][0];
    bf[c + 5] = dfa * plg[4][0];
    bf[c + 6] = dfa * plg[0][0] * cdoy[0];
    bf[c + 7] = dfa * plg[0][0] * sdoy[0];
    bf[c + 8] = dfa * plg[0][0] * cdoy[1];
    bf[c + 9] = dfa * plg[0][0] * sdoy[1];
    if sfluxavg <= 150.0 {
        bf[c + 10] = dfa * dfa;
    } else {
        bf[c + 10] = (150.0 - 150.0) * (2.0 * dfa - (150.0 - 150.0));
    }
    bf[c + 11] = bf[c + 10] * plg[2][0];
    bf[c + 12] = bf[c + 10] * plg[4][0];
    bf[c + 13] = df * plg[2][0];
    bf[c + 14] = df * plg[4][0];

    let mut c = CNONLIN;
    bf[c] = dfa;
    bf[c + 1] = dfa * dfa;
    bf[c + 2] = df;
    bf[c + 3] = df * df;
    bf[c + 4] = df * dfa;
    c += NSFXMOD;

    for i in 0..7 {
        bf[c + i] = ap[i] - 4.0;
    }
    bf[c + 8] = DOY2RAD * doy;
    bf[c + 9] = LST2RAD * lst;
    bf[c + 10] = DEG2RAD * lon;
    bf[c + 11] = LST2RAD * utsec / 3600.0;
    bf[c + 12] = lat.abs();
    c += 13;
    for m in 0..=1 {
        for row in plg.iter().take(AMAXN + 1) {
            bf[c] = row[m];
            c += 1;
        }
    }

    let c = CUT;
    bf[c] = LST2RAD * utsec / 3600.0;
    bf[c + 1] = DOY2RAD * doy;
    bf[c + 2] = dfa;
    bf[c + 3] = DEG2RAD * lon;
    bf[c + 4] = plg[1][0];
    bf[c + 5] = plg[3][0];
    bf[c + 6] = plg[5][0];
    bf[c + 7] = plg[3][2];
    bf[c + 8] = plg[5][2];

    for (j, bfj) in bf.iter_mut().enumerate().take(MBF + 1) {
        if !model.swg[j] {
            *bfj = 0.0;
        }
    }

    bf
}

fn sfluxmod(model: &Model, iz: usize, gf: &[f64; MAXNBF], parmset: &Subset, dffact: f64) -> f64 {
    let f1 = if model.swg[CSFXMOD] {
        parmset.get(CSFXMOD, iz) * gf[CSFXMOD]
            + (parmset.get(CSFX + 2, iz) * gf[CSFXMOD + 2]
                + parmset.get(CSFX + 3, iz) * gf[CSFXMOD + 3])
                * dffact
    } else {
        0.0
    };

    let f2 = if model.swg[CSFXMOD + 1] {
        parmset.get(CSFXMOD + 1, iz) * gf[CSFXMOD]
            + (parmset.get(CSFX + 2, iz) * gf[CSFXMOD + 2]
                + parmset.get(CSFX + 3, iz) * gf[CSFXMOD + 3])
                * dffact
    } else {
        0.0
    };

    let f3 = if model.swg[CSFXMOD + 2] {
        parmset.get(CSFXMOD + 2, iz) * gf[CSFXMOD]
    } else {
        0.0
    };

    let mut sum = 0.0;
    for j in 0..=MBF {
        if model.zsfx[j] {
            sum += parmset.get(j, iz) * gf[j] * f1;
            continue;
        }
        if model.tsfx[j] {
            sum += parmset.get(j, iz) * gf[j] * f2;
            continue;
        }
        if model.psfx[j] {
            sum += parmset.get(j, iz) * gf[j] * f3;
        }
    }
    sum
}

fn g0fn(a: f64, k00r: f64, k00s: f64) -> f64 {
    a + (k00r - 1.0) * (a + ((-a * k00s).exp() - 1.0) / k00s)
}

fn geomag(model: &Model, p0: &[f64], bf: &[f64], plg: &[f64; 14]) -> f64 {
    if !(model.swg[CMAG] || model.swg[CMAG + 1]) {
        return 0.0;
    }

    let mut p = p0.to_vec();
    let swg1 = &model.swg[CMAG..(CMAG + NMAG)];

    if swg1[0] == swg1[1] {
        if p[1] == 0.0 {
            return 0.0;
        }
        for i in 2..=25 {
            if !swg1[i] {
                p[i] = 0.0;
            }
        }
        p[8] = p0[8];
        let dela = g0fn(bf[0], p[0], p[1]);
        (p[2] * plg[0]
            + p[3] * plg[2]
            + p[4] * plg[4]
            + (p[5] * plg[1] + p[6] * plg[3] + p[7] * plg[5]) * (bf[8] - p[8]).cos()
            + (p[9] * plg[8] + p[10] * plg[10] + p[11] * plg[12]) * (bf[9] - p[12]).cos()
            + (1.0 + p[13] * plg[1])
                * (p[14] * plg[9] + p[15] * plg[11] + p[16] * plg[13])
                * (bf[10] - p[17]).cos()
            + (p[18] * plg[8] + p[19] * plg[10] + p[20] * plg[12])
                * (bf[10] - p[21]).cos()
                * (bf[8] - p[8]).cos()
            + (p[22] * plg[1] + p[23] * plg[3] + p[24] * plg[5]) * (bf[11] - p[25]).cos())
            * dela
    } else {
        if p[28] == 0.0 {
            return 0.0;
        }
        for i in 30..NMAG {
            if !swg1[i] {
                p[i] = 0.0;
            }
        }
        p[36] = p0[36];
        let gbeta = p[28] / (1.0 + p[29] * (45.0 - bf[12]));
        let ex = (-10_800.0 * gbeta).exp();
        let sumex = 1.0 + (1.0 - ex.powf(19.0)) * ex.sqrt() / (1.0 - ex);
        let mut g = [0.0; 6];
        for i in 1..=6 {
            g[i - 1] = g0fn(bf[i], p[26], p[27]);
        }
        let dela = (g[0]
            + (g[1] * ex
                + g[2] * ex * ex
                + g[3] * ex.powf(3.0)
                + (g[4] * ex.powf(4.0) + g[5] * ex.powf(12.0)) * (1.0 - ex.powf(8.0))
                    / (1.0 - ex)))
            / sumex;

        (p[30] * plg[0]
            + p[31] * plg[2]
            + p[32] * plg[4]
            + (p[33] * plg[1] + p[34] * plg[3] + p[35] * plg[5]) * (bf[8] - p[36]).cos()
            + (p[37] * plg[8] + p[38] * plg[10] + p[39] * plg[12]) * (bf[9] - p[40]).cos()
            + (1.0 + p[41] * plg[1])
                * (p[42] * plg[9] + p[43] * plg[11] + p[44] * plg[13])
                * (bf[10] - p[45]).cos()
            + (p[46] * plg[8] + p[47] * plg[10] + p[48] * plg[12])
                * (bf[10] - p[49]).cos()
                * (bf[8] - p[36]).cos()
            + (p[50] * plg[1] + p[51] * plg[3] + p[52] * plg[5]) * (bf[11] - p[53]).cos())
            * dela
    }
}

fn utdep(model: &Model, p0: &[f64], bf: &[f64]) -> f64 {
    let mut p = p0.to_vec();
    for i in 3..NUT {
        if !model.swg[CUT + i] {
            p[i] = 0.0;
        }
    }

    (bf[0] - p[0]).cos()
        * (1.0 + p[3] * bf[4] * (bf[1] - p[1]).cos())
        * (1.0 + p[4] * bf[2])
        * (1.0 + p[5] * bf[4])
        * (p[6] * bf[4] + p[7] * bf[5] + p[8] * bf[6])
        + (bf[0] - p[2] + 2.0 * bf[3]).cos()
            * (p[9] * bf[7] + p[10] * bf[8])
            * (1.0 + p[11] * bf[2])
}

#[derive(Clone, Default)]
struct TnParm {
    cf: [f64; NL + 1],
    tzeta_f: f64,
    tzeta_a: f64,
    dlntdz_a: f64,
    lndtot_f: f64,
    tex: f64,
    tgb0: f64,
    tb0: f64,
    sigma: f64,
    sigmasq: f64,
    b: f64,
    beta: [f64; NL + 1],
    gamma: [f64; NL + 1],
    cvs: f64,
    cvb: f64,
    cws: f64,
    cwb: f64,
    vzeta_f: f64,
    vzeta_a: f64,
    wzeta_a: f64,
    vzeta_0: f64,
}

fn tfnparm(model: &Model, gf: &[f64; MAXNBF]) -> TnParm {
    let mut tpro = TnParm::default();

    for ix in 0..ITB0 {
        tpro.cf[ix] = dot_subset_level(&model.tn, ix, gf, MBF);
    }
    for ix in 0..ITB0 {
        if model.smod[ix] {
            tpro.cf[ix] += sfluxmod(model, ix, gf, &model.tn, 1.0 / model.tn.get(0, ix));
        }
    }

    tpro.tex = dot_subset_level(&model.tn, ITEX, gf, MBF);
    tpro.tex += sfluxmod(model, ITEX, gf, &model.tn, 1.0 / model.tn.get(0, ITEX));

    let mut pgeom = [0.0; NMAG];
    for i in 0..NMAG {
        pgeom[i] = model.tn.get(CMAG + i, ITEX);
    }
    let mut bfm = [0.0; 13];
    for i in 0..13 {
        bfm[i] = gf[CMAG + i];
    }
    let mut plgm = [0.0; 14];
    for i in 0..14 {
        plgm[i] = gf[CMAG + 13 + i];
    }
    tpro.tex += geomag(model, &pgeom, &bfm, &plgm);

    let mut put = [0.0; NUT];
    for i in 0..NUT {
        put[i] = model.tn.get(CUT + i, ITEX);
    }
    let mut bfut = [0.0; 9];
    for i in 0..9 {
        bfut[i] = gf[CUT + i];
    }
    tpro.tex += utdep(model, &put, &bfut);

    tpro.tgb0 = dot_subset_level(&model.tn, ITGB0, gf, MBF);
    if model.smod[ITGB0] {
        tpro.tgb0 += sfluxmod(model, ITGB0, gf, &model.tn, 1.0 / model.tn.get(0, ITGB0));
    }
    for i in 0..NMAG {
        pgeom[i] = model.tn.get(CMAG + i, ITGB0);
    }
    tpro.tgb0 += geomag(model, &pgeom, &bfm, &plgm);

    tpro.tb0 = dot_subset_level(&model.tn, ITB0, gf, MBF);
    if model.smod[ITB0] {
        tpro.tb0 += sfluxmod(model, ITB0, gf, &model.tn, 1.0 / model.tn.get(0, ITB0));
    }
    for i in 0..NMAG {
        pgeom[i] = model.tn.get(CMAG + i, ITB0);
    }
    tpro.tb0 += geomag(model, &pgeom, &bfm, &plgm);

    tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0);

    let bc = [
        1.0 / tpro.tb0,
        -tpro.tgb0 / (tpro.tb0 * tpro.tb0),
        (tpro.tgb0 / (tpro.tb0 * tpro.tb0)) * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0),
    ];
    for ix in ITB0..=ITEX {
        let j = ix - ITB0;
        tpro.cf[ix] = bc[0] * C2TN[0][j] + bc[1] * C2TN[1][j] + bc[2] * C2TN[2][j];
    }

    tpro.tzeta_f = 1.0
        / (tpro.cf[IZFX] * S4_ZETA_F[0]
            + tpro.cf[IZFX + 1] * S4_ZETA_F[1]
            + tpro.cf[IZFX + 2] * S4_ZETA_F[2]);
    tpro.tzeta_a = 1.0
        / (tpro.cf[IZAX] * S4_ZETA_A[0]
            + tpro.cf[IZAX + 1] * S4_ZETA_A[1]
            + tpro.cf[IZAX + 2] * S4_ZETA_A[2]);
    tpro.dlntdz_a = -(tpro.cf[IZAX] * WGHT_AX_DZ[0]
        + tpro.cf[IZAX + 1] * WGHT_AX_DZ[1]
        + tpro.cf[IZAX + 2] * WGHT_AX_DZ[2])
        * tpro.tzeta_a;

    tpro.beta[0] = tpro.cf[0] * WBETA[0];
    for ix in 1..=NL {
        tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * WBETA[ix];
    }
    tpro.gamma[0] = tpro.beta[0] * WGAMMA[0];
    for ix in 1..=NL {
        tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * WGAMMA[ix];
    }

    tpro.b = 1.0 - tpro.tb0 / tpro.tex;
    tpro.sigmasq = tpro.sigma * tpro.sigma;
    tpro.cvs = -(tpro.beta[ITB0 - 1] * S5_ZETA_B[0]
        + tpro.beta[ITB0] * S5_ZETA_B[1]
        + tpro.beta[ITB0 + 1] * S5_ZETA_B[2]
        + tpro.beta[ITB0 + 2] * S5_ZETA_B[3]);
    tpro.cws = -(tpro.gamma[ITB0 - 2] * S6_ZETA_B[0]
        + tpro.gamma[ITB0 - 1] * S6_ZETA_B[1]
        + tpro.gamma[ITB0] * S6_ZETA_B[2]
        + tpro.gamma[ITB0 + 1] * S6_ZETA_B[3]
        + tpro.gamma[ITB0 + 2] * S6_ZETA_B[4]);
    tpro.cvb = -((1.0 - tpro.b).ln()) / (tpro.sigma * tpro.tex);
    tpro.cwb = -dilog(tpro.b) / (tpro.sigmasq * tpro.tex);
    tpro.vzeta_f = tpro.beta[IZFX - 1] * S5_ZETA_F[0]
        + tpro.beta[IZFX] * S5_ZETA_F[1]
        + tpro.beta[IZFX + 1] * S5_ZETA_F[2]
        + tpro.beta[IZFX + 2] * S5_ZETA_F[3]
        + tpro.cvs;
    tpro.vzeta_a = tpro.beta[IZAX - 1] * S5_ZETA_A[0]
        + tpro.beta[IZAX] * S5_ZETA_A[1]
        + tpro.beta[IZAX + 1] * S5_ZETA_A[2]
        + tpro.beta[IZAX + 2] * S5_ZETA_A[3]
        + tpro.cvs;
    tpro.wzeta_a = tpro.gamma[IZAX - 2] * S6_ZETA_A[0]
        + tpro.gamma[IZAX - 1] * S6_ZETA_A[1]
        + tpro.gamma[IZAX] * S6_ZETA_A[2]
        + tpro.gamma[IZAX + 1] * S6_ZETA_A[3]
        + tpro.gamma[IZAX + 2] * S6_ZETA_A[4]
        + tpro.cvs * (ZETA_A - ZETA_B)
        + tpro.cws;
    tpro.vzeta_0 = tpro.beta[0] * S5_ZETA_0[0]
        + tpro.beta[1] * S5_ZETA_0[1]
        + tpro.beta[2] * S5_ZETA_0[2]
        + tpro.cvs;

    tpro.lndtot_f = LNP0 - MBARG0DIVKB * (tpro.vzeta_f - tpro.vzeta_0) - (KB * tpro.tzeta_f).ln();

    tpro
}

fn tfnx(z: f64, iz: i32, s: &[[f64; 7]; 6], tpro: &TnParm) -> f64 {
    if z < ZETA_B {
        let izu = iz as usize;
        let i = izu.saturating_sub(3);
        let j0 = if izu < 3 { 5 - izu } else { 2 };
        let mut denom = 0.0;
        for off in 0..=(izu - i) {
            denom += tpro.cf[i + off] * s[j0 + off][4];
        }
        1.0 / denom
    } else {
        tpro.tex - (tpro.tex - tpro.tb0) * (-tpro.sigma * (z - ZETA_B)).exp()
    }
}

#[derive(Clone, Default)]
struct DnParm {
    ln_phi_f: f64,
    lnd_ref: f64,
    zeta_m: f64,
    hml: f64,
    hmu: f64,
    c: f64,
    zeta_c: f64,
    hc: f64,
    r: f64,
    zeta_r: f64,
    hr: f64,
    cf: [f64; NSPLO1 + 2],
    zref: f64,
    mi: [f64; 5],
    zeta_mi: [f64; 5],
    ami: [f64; 5],
    wmi: [f64; 5],
    xmi: [f64; 5],
    iz_ref: f64,
    tref: f64,
    zmin: f64,
    zhyd: f64,
    ispec: usize,
}

fn pwmp(z: f64, zm: &[f64; 5], m: &[f64; 5], dmdz: &[f64; 5]) -> f64 {
    if z >= zm[4] {
        return m[4];
    }
    if z <= zm[0] {
        return m[0];
    }
    for inode in 0..4 {
        if z < zm[inode + 1] {
            return m[inode] + dmdz[inode] * (z - zm[inode]);
        }
    }
    m[4]
}

fn dfnparm(model: &Model, ispec: usize, gf: &[f64; MAXNBF], tpro: &TnParm) -> DnParm {
    let mut dpro = DnParm {
        ispec,
        ..Default::default()
    };

    let mut pgeom = [0.0; NMAG];
    let mut bfm = [0.0; 13];
    let mut plgm = [0.0; 14];
    for i in 0..13 {
        bfm[i] = gf[CMAG + i];
    }
    for i in 0..14 {
        plgm[i] = gf[CMAG + 13 + i];
    }
    let mut put = [0.0; NUT];
    let mut bfut = [0.0; 9];
    for i in 0..9 {
        bfut[i] = gf[CUT + i];
    }

    match ispec {
        2 => {
            dpro.ln_phi_f = LNVMR[ispec];
            dpro.lnd_ref = tpro.lndtot_f + dpro.ln_phi_f;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zeta_m = dot_subset_level(&model.n2, 1, gf, MBF);
            dpro.hml = model.n2.get(0, 2);
            dpro.hmu = model.n2.get(0, 3);
            if model.n2r_flag {
                dpro.r = dot_subset_level(&model.n2, 7, gf, MBF);
            }
            dpro.zeta_r = model.n2.get(0, 8);
            dpro.hr = model.n2.get(0, 9);
        }
        3 => {
            dpro.ln_phi_f = LNVMR[ispec];
            dpro.lnd_ref = tpro.lndtot_f + dpro.ln_phi_f;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zeta_m = model.o2.get(0, 1);
            dpro.hml = model.o2.get(0, 2);
            dpro.hmu = model.o2.get(0, 3);
            dpro.r = dot_subset_level(&model.o2, 7, gf, MBF);
            for i in 0..NMAG {
                pgeom[i] = model.o2.get(CMAG + i, 7);
            }
            dpro.r += geomag(model, &pgeom, &bfm, &plgm);
            dpro.zeta_r = model.o2.get(0, 8);
            dpro.hr = model.o2.get(0, 9);
        }
        4 => {
            dpro.lnd_ref = dot_subset_level(&model.o1, 0, gf, MBF);
            dpro.zref = ZETAREF_O1;
            dpro.zmin = NODES_O1[3];
            dpro.zhyd = ZETAREF_O1;
            dpro.zeta_m = model.o1.get(0, 1);
            dpro.hml = model.o1.get(0, 2);
            dpro.hmu = model.o1.get(0, 3);
            dpro.c = dot_subset_level(&model.o1, 4, gf, MBF);
            dpro.zeta_c = model.o1.get(0, 5);
            dpro.hc = model.o1.get(0, 6);
            dpro.r = dot_subset_level(&model.o1, 7, gf, MBF);
            dpro.r += sfluxmod(model, 7, gf, &model.o1, 0.0);
            for i in 0..NMAG {
                pgeom[i] = model.o1.get(CMAG + i, 7);
            }
            dpro.r += geomag(model, &pgeom, &bfm, &plgm);
            for i in 0..NUT {
                put[i] = model.o1.get(CUT + i, 7);
            }
            dpro.r += utdep(model, &put, &bfut);
            dpro.zeta_r = model.o1.get(0, 8);
            dpro.hr = model.o1.get(0, 9);
            for izf in 0..NSPLO1 {
                dpro.cf[izf] = dot_subset_level(&model.o1, izf + 10, gf, MBF);
            }
        }
        5 => {
            dpro.ln_phi_f = LNVMR[ispec];
            dpro.lnd_ref = tpro.lndtot_f + dpro.ln_phi_f;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zeta_m = model.he.get(0, 1);
            dpro.hml = model.he.get(0, 2);
            dpro.hmu = model.he.get(0, 3);
            dpro.r = dot_subset_level(&model.he, 7, gf, MBF);
            dpro.r += sfluxmod(model, 7, gf, &model.he, 1.0);
            for i in 0..NMAG {
                pgeom[i] = model.he.get(CMAG + i, 7);
            }
            dpro.r += geomag(model, &pgeom, &bfm, &plgm);
            for i in 0..NUT {
                put[i] = model.he.get(CUT + i, 7);
            }
            dpro.r += utdep(model, &put, &bfut);
            dpro.zeta_r = model.he.get(0, 8);
            dpro.hr = model.he.get(0, 9);
        }
        6 => {
            dpro.lnd_ref = dot_subset_level(&model.h1, 0, gf, MBF);
            dpro.zref = ZETA_A;
            dpro.zmin = 75.0;
            dpro.zhyd = ZETA_F;
            dpro.zeta_m = model.h1.get(0, 1);
            dpro.hml = model.h1.get(0, 2);
            dpro.hmu = model.h1.get(0, 3);
            dpro.c = dot_subset_level(&model.h1, 4, gf, MBF);
            dpro.zeta_c = dot_subset_level(&model.h1, 5, gf, MBF);
            dpro.hc = model.h1.get(0, 6);
            dpro.r = dot_subset_level(&model.h1, 7, gf, MBF);
            dpro.r += sfluxmod(model, 7, gf, &model.h1, 0.0);
            for i in 0..NMAG {
                pgeom[i] = model.h1.get(CMAG + i, 7);
            }
            dpro.r += geomag(model, &pgeom, &bfm, &plgm);
            for i in 0..NUT {
                put[i] = model.h1.get(CUT + i, 7);
            }
            dpro.r += utdep(model, &put, &bfut);
            dpro.zeta_r = model.h1.get(0, 8);
            dpro.hr = model.h1.get(0, 9);
        }
        7 => {
            dpro.ln_phi_f = LNVMR[ispec];
            dpro.lnd_ref = tpro.lndtot_f + dpro.ln_phi_f;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zeta_m = model.ar.get(0, 1);
            dpro.hml = model.ar.get(0, 2);
            dpro.hmu = model.ar.get(0, 3);
            dpro.r = dot_subset_level(&model.ar, 7, gf, MBF);
            for i in 0..NMAG {
                pgeom[i] = model.ar.get(CMAG + i, 7);
            }
            dpro.r += geomag(model, &pgeom, &bfm, &plgm);
            for i in 0..NUT {
                put[i] = model.ar.get(CUT + i, 7);
            }
            dpro.r += utdep(model, &put, &bfut);
            dpro.zeta_r = model.ar.get(0, 8);
            dpro.hr = model.ar.get(0, 9);
        }
        8 => {
            dpro.lnd_ref = dot_subset_level(&model.n1, 0, gf, MBF);
            dpro.lnd_ref += sfluxmod(model, 0, gf, &model.n1, 0.0);
            for i in 0..NMAG {
                pgeom[i] = model.n1.get(CMAG + i, 0);
            }
            dpro.lnd_ref += geomag(model, &pgeom, &bfm, &plgm);
            for i in 0..NUT {
                put[i] = model.n1.get(CUT + i, 0);
            }
            dpro.lnd_ref += utdep(model, &put, &bfut);
            dpro.zref = ZETA_B;
            dpro.zmin = 90.0;
            dpro.zhyd = ZETA_F;
            dpro.zeta_m = model.n1.get(0, 1);
            dpro.hml = model.n1.get(0, 2);
            dpro.hmu = model.n1.get(0, 3);
            dpro.c = model.n1.get(0, 4);
            dpro.zeta_c = model.n1.get(0, 5);
            dpro.hc = model.n1.get(0, 6);
            dpro.r = dot_subset_level(&model.n1, 7, gf, MBF);
            dpro.zeta_r = model.n1.get(0, 8);
            dpro.hr = model.n1.get(0, 9);
        }
        9 => {
            dpro.lnd_ref = dot_subset_level(&model.oa, 0, gf, MBF);
            for i in 0..NMAG {
                pgeom[i] = model.oa.get(CMAG + i, 0);
            }
            dpro.lnd_ref += geomag(model, &pgeom, &bfm, &plgm);
            dpro.zref = ZETAREF_OA;
            dpro.zmin = 120.0;
            dpro.zhyd = 0.0;
            dpro.c = model.oa.get(0, 4);
            dpro.zeta_c = model.oa.get(0, 5);
            dpro.hc = model.oa.get(0, 6);
            return dpro;
        }
        10 => {
            if model.no.get(0, 0) == 0.0 {
                return dpro;
            }
            dpro.lnd_ref = dot_subset_level(&model.no, 0, gf, MBF);
            for i in 0..NMAG {
                pgeom[i] = model.no.get(CMAG + i, 0);
            }
            dpro.lnd_ref += geomag(model, &pgeom, &bfm, &plgm);
            dpro.zref = ZETAREF_NO;
            dpro.zmin = 72.5;
            dpro.zhyd = ZETAREF_NO;
            dpro.zeta_m = dot_subset_level(&model.no, 1, gf, MBF);
            dpro.hml = dot_subset_level(&model.no, 2, gf, MBF);
            dpro.hmu = dot_subset_level(&model.no, 3, gf, MBF);
            dpro.c = dot_subset_level(&model.no, 4, gf, MBF);
            for i in 0..NMAG {
                pgeom[i] = model.no.get(CMAG + i, 4);
            }
            dpro.c += geomag(model, &pgeom, &bfm, &plgm);
            dpro.zeta_c = dot_subset_level(&model.no, 5, gf, MBF);
            dpro.hc = dot_subset_level(&model.no, 6, gf, MBF);
            dpro.r = dot_subset_level(&model.no, 7, gf, MBF);
            dpro.zeta_r = dot_subset_level(&model.no, 8, gf, MBF);
            dpro.hr = dot_subset_level(&model.no, 9, gf, MBF);
            for izf in 0..NSPLNO {
                dpro.cf[izf] = dot_subset_level(&model.no, izf + 10, gf, MBF);
                for i in 0..NMAG {
                    pgeom[i] = model.no.get(CMAG + i, izf + 10);
                }
                dpro.cf[izf] += geomag(model, &pgeom, &bfm, &plgm);
            }
        }
        _ => return dpro,
    }

    dpro.zeta_mi[0] = dpro.zeta_m - 2.0 * dpro.hml;
    dpro.zeta_mi[1] = dpro.zeta_m - dpro.hml;
    dpro.zeta_mi[2] = dpro.zeta_m;
    dpro.zeta_mi[3] = dpro.zeta_m + dpro.hmu;
    dpro.zeta_mi[4] = dpro.zeta_m + 2.0 * dpro.hmu;
    dpro.mi[0] = MBAR;
    dpro.mi[4] = SPECMASS[ispec];
    dpro.mi[2] = (dpro.mi[0] + dpro.mi[4]) / 2.0;
    let delm = TANH1 * (dpro.mi[4] - dpro.mi[0]) / 2.0;
    dpro.mi[1] = dpro.mi[2] - delm;
    dpro.mi[3] = dpro.mi[2] + delm;
    for i in 0..4 {
        dpro.ami[i] = (dpro.mi[i + 1] - dpro.mi[i]) / (dpro.zeta_mi[i + 1] - dpro.zeta_mi[i]);
    }

    for i in 0..5 {
        let delz = dpro.zeta_mi[i] - ZETA_B;
        if dpro.zeta_mi[i] < ZETA_B {
            let (si, iz) = bspline(dpro.zeta_mi[i], &NODES_TN, ND + 2, 6, &model.eta_tn);
            let izu = iz as usize;
            let mut sum = 0.0;
            for k in 0..=5 {
                sum += tpro.gamma[izu - 5 + k] * si[k][6];
            }
            dpro.wmi[i] = sum + tpro.cvs * delz + tpro.cws;
        } else {
            dpro.wmi[i] = (0.5 * delz * delz
                + dilog(tpro.b * (-tpro.sigma * delz).exp()) / tpro.sigmasq)
                / tpro.tex
                + tpro.cvb * delz
                + tpro.cwb;
        }
    }

    dpro.xmi[0] = -dpro.ami[0] * dpro.wmi[0];
    for i in 1..4 {
        dpro.xmi[i] = dpro.xmi[i - 1] - dpro.wmi[i] * (dpro.ami[i] - dpro.ami[i - 1]);
    }
    dpro.xmi[4] = dpro.xmi[3] + dpro.wmi[4] * dpro.ami[3];

    if (dpro.zref - ZETA_F).abs() < 1e-12 {
        let mzref = MBAR;
        dpro.tref = tpro.tzeta_f;
        dpro.iz_ref = MBAR * tpro.vzeta_f;
        let _ = mzref;
    } else if (dpro.zref - ZETA_B).abs() < 1e-12 {
        let mzref = pwmp(dpro.zref, &dpro.zeta_mi, &dpro.mi, &dpro.ami);
        dpro.tref = tpro.tb0;
        dpro.iz_ref = 0.0;
        if ZETA_B > dpro.zeta_mi[0] && ZETA_B < dpro.zeta_mi[4] {
            let mut i = 0usize;
            for i1 in 1..=3 {
                if ZETA_B < dpro.zeta_mi[i1] {
                    break;
                }
                i = i1;
            }
            dpro.iz_ref -= dpro.xmi[i];
        } else {
            dpro.iz_ref -= dpro.xmi[4];
        }
        let _ = mzref;
    } else if (dpro.zref - ZETA_A).abs() < 1e-12 {
        let mzref = pwmp(dpro.zref, &dpro.zeta_mi, &dpro.mi, &dpro.ami);
        dpro.tref = tpro.tzeta_a;
        dpro.iz_ref = mzref * tpro.vzeta_a;
        if ZETA_A > dpro.zeta_mi[0] && ZETA_A < dpro.zeta_mi[4] {
            let mut i = 0usize;
            for i1 in 1..=3 {
                if ZETA_A < dpro.zeta_mi[i1] {
                    break;
                }
                i = i1;
            }
            dpro.iz_ref -= dpro.ami[i] * tpro.wzeta_a + dpro.xmi[i];
        } else {
            dpro.iz_ref -= dpro.xmi[4];
        }
    }

    if ispec == 4 {
        let cterm = dpro.c * (-(dpro.zref - dpro.zeta_c) / dpro.hc).exp();
        let rterm0 = ((dpro.zref - dpro.zeta_r) / (model.hr_fact_o1_ref * dpro.hr)).tanh();
        let rterm = dpro.r * (1.0 + rterm0);
        let bc0 = dpro.lnd_ref - cterm + rterm - dpro.cf[7] * C1O1ADJ[0];
        let bc1 = -pwmp(dpro.zref, &dpro.zeta_mi, &dpro.mi, &dpro.ami) * G0DIVKB / tpro.tzeta_a
            - tpro.dlntdz_a
            + cterm / dpro.hc
            + rterm * (1.0 - rterm0) / dpro.hr * model.dhr_fact_o1_ref
            - dpro.cf[7] * C1O1ADJ[1];
        dpro.cf[8] = bc0 * C1O1[0][0] + bc1 * C1O1[1][0];
        dpro.cf[9] = bc0 * C1O1[0][1] + bc1 * C1O1[1][1];
    }

    if ispec == 10 {
        let cterm = dpro.c * (-(dpro.zref - dpro.zeta_c) / dpro.hc).exp();
        let rterm0 = ((dpro.zref - dpro.zeta_r) / (model.hr_fact_no_ref * dpro.hr)).tanh();
        let rterm = dpro.r * (1.0 + rterm0);
        let bc0 = dpro.lnd_ref - cterm + rterm - dpro.cf[7] * C1NOADJ[0];
        let bc1 = -pwmp(dpro.zref, &dpro.zeta_mi, &dpro.mi, &dpro.ami) * G0DIVKB / tpro.tb0
            - tpro.tgb0 / tpro.tb0
            + cterm / dpro.hc
            + rterm * (1.0 - rterm0) / dpro.hr * model.dhr_fact_no_ref
            - dpro.cf[7] * C1NOADJ[1];
        dpro.cf[8] = bc0 * C1NO[0][0] + bc1 * C1NO[1][0];
        dpro.cf[9] = bc0 * C1NO[0][1] + bc1 * C1NO[1][1];
    }

    dpro
}

fn dfnx(
    model: &Model,
    z: f64,
    tnz: f64,
    lndtotz: f64,
    vz: f64,
    wz: f64,
    hrfact: f64,
    dpro: &DnParm,
) -> f64 {
    if z < dpro.zmin {
        return DMISSING;
    }

    if dpro.ispec == 9 {
        return (dpro.lnd_ref
            - (z - dpro.zref) / HOA
            - dpro.c * (-(z - dpro.zeta_c) / dpro.hc).exp())
        .exp();
    }

    if dpro.ispec == 10 && dpro.lnd_ref == 0.0 {
        return DMISSING;
    }

    let ccor = match dpro.ispec {
        2 | 3 | 5 | 7 => dpro.r * (1.0 + ((z - dpro.zeta_r) / (hrfact * dpro.hr)).tanh()),
        _ => {
            -dpro.c * (-(z - dpro.zeta_c) / dpro.hc).exp()
                + dpro.r * (1.0 + ((z - dpro.zeta_r) / (hrfact * dpro.hr)).tanh())
        }
    };

    if z < dpro.zhyd {
        match dpro.ispec {
            2 | 3 | 5 | 7 => return (lndtotz + dpro.ln_phi_f + ccor).exp(),
            4 => {
                let (s, iz) = bspline(z, &NODES_O1, NDO1, 4, &model.eta_o1);
                let izu = iz as usize;
                let mut sum = 0.0;
                for k in 0..=3 {
                    sum += dpro.cf[izu - 3 + k] * s[k + 2][4];
                }
                return sum.exp();
            }
            10 => {
                let (s, iz) = bspline(z, &NODES_NO, NDNO, 4, &model.eta_no);
                let izu = iz as usize;
                let mut sum = 0.0;
                for k in 0..=3 {
                    sum += dpro.cf[izu - 3 + k] * s[k + 2][4];
                }
                return sum.exp();
            }
            _ => {}
        }
    }

    let mz = pwmp(z, &dpro.zeta_mi, &dpro.mi, &dpro.ami);
    let mut ihyd = mz * vz - dpro.iz_ref;
    if z > dpro.zeta_mi[0] && z < dpro.zeta_mi[4] {
        let mut i = 0usize;
        for i1 in 1..=3 {
            if z < dpro.zeta_mi[i1] {
                break;
            }
            i = i1;
        }
        ihyd -= dpro.ami[i] * wz + dpro.xmi[i];
    } else if z >= dpro.zeta_mi[4] {
        ihyd -= dpro.xmi[4];
    }
    let mut dn = dpro.lnd_ref - ihyd * G0DIVKB + ccor;
    dn = dn.exp() * dpro.tref / tnz;
    dn
}

#[derive(Debug, Clone)]
/// Output of [`msiscalc`].
pub struct MsisOutput {
    /// Temperature at input altitude in K.
    pub tn: f64,
    /// Exospheric temperature in K.
    pub tex: f64,
    /// Densities in SI order and units:
    /// 1) mass density (kg/m^3),
    /// 2) N2 (m^-3), 3) O2 (m^-3), 4) O (m^-3), 5) He (m^-3),
    /// 6) H (m^-3), 7) Ar (m^-3), 8) N (m^-3), 9) anomalous O (m^-3), 10) NO (m^-3).
    pub dn: [f64; 10],
}

fn msiscalc_with_model(
    model: &Model,
    day: f64,
    utsec: f64,
    z: f64,
    lat: f64,
    lon: f64,
    sfluxavg: f64,
    sflux: f64,
    ap: [f64; 7],
) -> Result<MsisOutput, String> {
    let zeta = alt2gph(lat, z);

    let kmax = if zeta < ZETA_F { 5 } else { 6 };
    let (sz, iz) = if zeta < ZETA_B {
        bspline(zeta, &NODES_TN, ND + 2, kmax, &model.eta_tn)
    } else {
        ([[0.0; 7]; 6], 0)
    };

    let gf = globe(model, day, utsec, lat, lon, sfluxavg, sflux, ap);
    let tpro = tfnparm(model, &gf);

    let tn = tfnx(zeta, iz, &sz, &tpro);

    let delz = zeta - ZETA_B;
    let (vz, wz, lndtotz) = if zeta < ZETA_F {
        let izu = iz as usize;
        let i = izu.saturating_sub(4);
        let j0 = if izu < 4 { 5 - izu } else { 1 };

        let mut vz = tpro.cvs;
        for off in 0..=(izu - i) {
            vz += tpro.beta[i + off] * sz[j0 + off][5];
        }
        let ln_pz = LNP0 - MBARG0DIVKB * (vz - tpro.vzeta_0);
        let lndtotz = ln_pz - (KB * tn).ln();
        (vz, 0.0, lndtotz)
    } else if zeta < ZETA_B {
        let izu = iz as usize;
        let mut vz = tpro.cvs;
        let mut wz = tpro.cvs * delz + tpro.cws;
        for off in 0..=4 {
            vz += tpro.beta[izu - 4 + off] * sz[off + 1][5];
        }
        for off in 0..=5 {
            wz += tpro.gamma[izu - 5 + off] * sz[off][6];
        }
        (vz, wz, 0.0)
    } else {
        let vz = (delz + (tn / tpro.tex).ln() / tpro.sigma) / tpro.tex + tpro.cvb;
        let wz = (0.5 * delz * delz + dilog(tpro.b * (-tpro.sigma * delz).exp()) / tpro.sigmasq)
            / tpro.tex
            + tpro.cvb * delz
            + tpro.cwb;
        (vz, wz, 0.0)
    };

    let hrfact = 0.5 * (1.0 + (HGAMMA * (zeta - ZETA_GAMMA)).tanh());
    let mut dn1 = [DMISSING; NSPEC];
    for (ispec, dn_slot) in dn1.iter_mut().enumerate().take(10 + 1).skip(2) {
        let dpro = dfnparm(model, ispec, &gf, &tpro);
        *dn_slot = dfnx(model, zeta, tn, lndtotz, vz, wz, hrfact, &dpro);
    }

    let mut mass = 0.0;
    for (ispec, dn_val) in dn1.iter().enumerate().take(10 + 1).skip(1) {
        mass += *dn_val * model.masswgt[ispec];
    }
    dn1[1] = mass;

    let mut out = [0.0; 10];
    out.copy_from_slice(&dn1[1..11]);

    Ok(MsisOutput {
        tn,
        tex: tpro.tex,
        dn: out,
    })
}

/// Evaluate NRLMSIS 2.1 using the SI/MKS interface.
///
/// # Arguments
/// - `day`: day of year (may include fractional part)
/// - `utsec`: universal time in seconds
/// - `z`: geodetic altitude in kilometers
/// - `lat`: geodetic latitude in degrees
/// - `lon`: geodetic longitude in degrees
/// - `sfluxavg`: centered 81-day average F10.7
/// - `sflux`: previous-day F10.7
/// - `ap`: geomagnetic index array `[Ap_daily, ap3h_now, ap3h_3h_ago, ..., ap3h_57h_ago]`
///
/// # Returns
/// [`MsisOutput`] with SI units.
pub fn msiscalc(
    day: f64,
    utsec: f64,
    z: f64,
    lat: f64,
    lon: f64,
    sfluxavg: f64,
    sflux: f64,
    ap: [f64; 7],
) -> Result<MsisOutput, String> {
    let model = model()?;
    msiscalc_with_model(model, day, utsec, z, lat, lon, sfluxavg, sflux, ap)
}

#[derive(Debug, Clone)]
/// Output of legacy-compatible wrappers [`gtd8d`], [`gtd8d_legacy`], and [`gtd8d_legacy_with_switches`].
pub struct Gtd8dOutput {
    /// Legacy density order/units:
    /// `[He, O, N2, O2, Ar, rho, H, N, O*, NO]`
    /// where number densities are in cm^-3 and `rho` is in g/cm^3.
    pub d: [f64; 10],
    /// Temperatures `[Tex, Tn]` in K.
    pub t: [f64; 2],
}

/// Legacy `gtd8d` wrapper with explicit NRLMSISE-style legacy switch control.
///
/// `switch_legacy` corresponds to the 25-element legacy switch array (as in `TSELEC`).
/// Values typically use `0` (off), `1` (on), or `2` (cross-terms only behavior depending on switch).
///
/// Notes:
/// - `stl` and `mass` are accepted for legacy signature compatibility.
/// - `stl` is ignored, matching the Fortran 2.1 wrapper behavior.
/// - Outputs are in legacy CGS layout/units (see [`Gtd8dOutput`]).
pub fn gtd8d_legacy_with_switches(
    iyd: i32,
    sec: f64,
    alt: f64,
    glat: f64,
    glong: f64,
    _stl: f64,
    f107a: f64,
    f107: f64,
    ap: [f64; 7],
    _mass: i32,
    switch_legacy: [f64; 25],
) -> Result<Gtd8dOutput, String> {
    let day = (iyd % 1000) as f64;
    let base = model()?;
    let switched = base.with_legacy_switches(switch_legacy);
    let out = msiscalc_with_model(&switched, day, sec, alt, glat, glong, f107a, f107, ap)?;

    let mut xdn = out.dn;
    for value in &mut xdn {
        if *value != DMISSING {
            *value *= 1.0e-6;
        }
    }
    if xdn[0] != DMISSING {
        xdn[0] *= 1.0e3;
    }

    let d = [
        xdn[4], xdn[3], xdn[1], xdn[2], xdn[6], xdn[0], xdn[5], xdn[7], xdn[8], xdn[9],
    ];

    Ok(Gtd8dOutput {
        d,
        t: [out.tex, out.tn],
    })
}

/// Legacy-compatible `gtd8d` wrapper using default legacy switches (`[1.0; 25]`).
///
/// Keeps the original argument shape for compatibility with existing callers.
pub fn gtd8d_legacy(
    iyd: i32,
    sec: f64,
    alt: f64,
    glat: f64,
    glong: f64,
    _stl: f64,
    f107a: f64,
    f107: f64,
    ap: [f64; 7],
    _mass: i32,
) -> Result<Gtd8dOutput, String> {
    gtd8d_legacy_with_switches(
        iyd, sec, alt, glat, glong, _stl, f107a, f107, ap, _mass, [1.0; 25],
    )
}

/// Convenience wrapper equivalent to [`gtd8d_legacy`] with `stl=0` and `mass=0`.
pub fn gtd8d(
    iyd: i32,
    sec: f64,
    alt: f64,
    glat: f64,
    glong: f64,
    f107a: f64,
    f107: f64,
    ap: [f64; 7],
) -> Result<Gtd8dOutput, String> {
    gtd8d_legacy(iyd, sec, alt, glat, glong, 0.0, f107a, f107, ap, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test;
    use std::fs;

    #[test]
    fn test_nrlmsis2_against_reference_row() {
        let ap = [35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0];
        let out = gtd8d(70178, 64800.0, 0.2, 50.0, 55.0, 153.3, 146.5, ap).expect("gtd8d eval");

        let ref_rho = 0.1160e-02;
        let ref_tn = 294.10;

        let rel_rho = (out.d[5] - ref_rho).abs() / ref_rho.abs();
        let rel_tn = (out.t[1] - ref_tn).abs() / ref_tn.abs();

        assert!(rel_rho < 3.0, "rho rel error too high: {rel_rho}");
        assert!(rel_tn < 0.2, "temperature rel error too high: {rel_tn}");
    }

    fn rel_err(pred: f64, truth: f64) -> f64 {
        if truth.abs() < 1.0e-30 {
            if pred.abs() < 1.0e-30 {
                0.0
            } else {
                pred.abs()
            }
        } else {
            (pred - truth).abs() / truth.abs()
        }
    }

    #[test]
    fn test_nrlmsis2_regression_harness_200_rows() {
        let testinput = test::get_testvec_dir()
            .unwrap()
            .join("nrlmsis2")
            .join("msis2.1_test_in.txt");

        let input_text = fs::read_to_string(&testinput).expect("read msis2.1_test_in.txt");
        let testref = test::get_testvec_dir()
            .unwrap()
            .join("nrlmsis2")
            .join("msis2.1_test_ref_dp.txt");
        let ref_text = fs::read_to_string(&testref).expect("read msis2.1_test_ref_dp.txt");

        let input_lines: Vec<&str> = input_text
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .collect();
        let ref_lines: Vec<&str> = ref_text
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .collect();

        assert_eq!(input_lines.len(), 200, "unexpected input row count");
        assert_eq!(ref_lines.len(), 200, "unexpected reference row count");

        let fields = [
            "He", "O", "N2", "O2", "Ar", "rho", "H", "N", "O*", "NO", "T",
        ];
        let mut max_rel = [0.0_f64; 11];
        let mut sum_rel = [0.0_f64; 11];
        let mut n_rel = [0_usize; 11];
        let mut worst_row = [0_usize; 11];
        let mut worst_in = [
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ];
        let mut worst_pred = [0.0_f64; 11];
        let mut worst_truth = [0.0_f64; 11];

        for (row_idx, (line_in, line_ref)) in input_lines.iter().zip(ref_lines.iter()).enumerate() {
            let in_cols: Vec<&str> = line_in.split_whitespace().collect();
            let ref_cols: Vec<&str> = line_ref.split_whitespace().collect();

            assert_eq!(
                in_cols.len(),
                9,
                "unexpected input column count in: {line_in}"
            );
            assert_eq!(
                ref_cols.len(),
                20,
                "unexpected reference column count in: {line_ref}"
            );

            let iyd: i32 = in_cols[0].parse().expect("parse iyd");
            let sec: f64 = in_cols[1].parse().expect("parse sec");
            let alt: f64 = in_cols[2].parse().expect("parse alt");
            let glat: f64 = in_cols[3].parse().expect("parse glat");
            let glong: f64 = in_cols[4].parse().expect("parse glong");
            let stl: f64 = in_cols[5].parse().expect("parse stl");
            let f107a: f64 = in_cols[6].parse().expect("parse f107a");
            let f107: f64 = in_cols[7].parse().expect("parse f107");
            let apd: f64 = in_cols[8].parse().expect("parse Ap");

            let ap = [apd; 7];
            let out = gtd8d_legacy(iyd, sec, alt, glat, glong, stl, f107a, f107, ap, 0)
                .unwrap_or_else(|error| panic!("gtd8d failed for row `{line_in}`: {error}"));

            for idx in 0..10 {
                let truth: f64 = ref_cols[9 + idx].parse().expect("parse density ref");
                let err = rel_err(out.d[idx], truth);
                if err > max_rel[idx] {
                    worst_row[idx] = row_idx + 1;
                    worst_in[idx] = (*line_in).to_string();
                    worst_pred[idx] = out.d[idx];
                    worst_truth[idx] = truth;
                }
                max_rel[idx] = max_rel[idx].max(err);
                sum_rel[idx] += err;
                n_rel[idx] += 1;
            }

            let truth_t: f64 = ref_cols[19].parse().expect("parse temperature ref");
            let err_t = rel_err(out.t[1], truth_t);
            if err_t > max_rel[10] {
                worst_row[10] = row_idx + 1;
                worst_in[10] = (*line_in).to_string();
                worst_pred[10] = out.t[1];
                worst_truth[10] = truth_t;
            }
            max_rel[10] = max_rel[10].max(err_t);
            sum_rel[10] += err_t;
            n_rel[10] += 1;
        }

        for idx in 0..11 {
            let mean_rel = if n_rel[idx] > 0 {
                sum_rel[idx] / n_rel[idx] as f64
            } else {
                0.0
            };
            eprintln!(
                "NRLMSIS2.1 regression {}: max_rel={:.6e}, mean_rel={:.6e}, n={}",
                fields[idx], max_rel[idx], mean_rel, n_rel[idx]
            );
            eprintln!(
                "  worst row {} in=`{}` pred={:.6e} ref={:.6e}",
                worst_row[idx], worst_in[idx], worst_pred[idx], worst_truth[idx]
            );
        }

        let max_allowed = [
            6.0e-4, // He
            6.0e-4, // O
            6.0e-4, // N2
            6.0e-4, // O2
            6.0e-4, // Ar
            6.0e-4, // rho
            6.0e-4, // H
            6.0e-4, // N
            6.0e-4, // O*
            6.0e-4, // NO
            5.0e-5, // T
        ];

        for idx in 0..11 {
            assert!(
                max_rel[idx].is_finite() && max_rel[idx] <= max_allowed[idx],
                "regression exceeded baseline tolerance for {}: max_rel={}, allowed={}",
                fields[idx],
                max_rel[idx],
                max_allowed[idx]
            );
        }
    }
}
