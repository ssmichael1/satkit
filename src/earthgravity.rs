use crate::utils::{datadir, download_if_not_exist};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use nalgebra as na;
type CoeffTable = na::DMatrix<f64>;

type DivisorTable = na::SMatrix<f64, 44, 44>;

use once_cell::sync::OnceCell;

///
/// Gravity model enumeration
///
/// For details of models, see:
/// <http://icgem.gfz-potsdam.de/tom_longtime>
///
#[derive(PartialEq, Eq, Hash)]
pub enum GravityModel {
    JGM3,
    JGM2,
    EGM96,
    ITUGrace16,
}

///
/// Singleton for JGM3 gravity model
///
pub fn jgm3() -> &'static Gravity {
    static INSTANCE: OnceCell<Gravity> = OnceCell::new();
    INSTANCE.get_or_init(|| Gravity::from_file("JGM3.gfc").unwrap())
}

///
/// Singleton for JGM2 gravity model
///
pub fn jgm2() -> &'static Gravity {
    static INSTANCE: OnceCell<Gravity> = OnceCell::new();
    INSTANCE.get_or_init(|| Gravity::from_file("JGM2.gfc").unwrap())
}

///
/// Singleton for EGM96 gravity model
///
pub fn egm96() -> &'static Gravity {
    static INSTANCE: OnceCell<Gravity> = OnceCell::new();
    INSTANCE.get_or_init(|| Gravity::from_file("EGM96.gfc").unwrap())
}

///
/// Singleton for ITU GRACE16 gravity model
///
pub fn itu_grace16() -> &'static Gravity {
    static INSTANCE: OnceCell<Gravity> = OnceCell::new();
    INSTANCE.get_or_init(|| Gravity::from_file("ITU_GRACE16.gfc").unwrap())
}

///
/// Gravity model hash
///
pub fn gravhash() -> &'static HashMap<GravityModel, &'static Gravity> {
    static INSTANCE: OnceCell<HashMap<GravityModel, &'static Gravity>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert(GravityModel::JGM3, jgm3());
        m.insert(GravityModel::JGM2, jgm2());
        m.insert(GravityModel::EGM96, egm96());
        m.insert(GravityModel::ITUGrace16, itu_grace16());
        m
    })
}

///
/// Return acceleration due to Earth gravity at the input position. The
/// acceleration does not include the centrifugal force, and is output
/// in m/s^2 in the International Terrestrial Reference Frame (ITRF)
///
/// # Arguments
///
/// * `pos` - nalgebra 3-vector representing ITRF position in meters
///
/// * `order` - The order of the gravity model to use.
///             Maximum is 16
///
/// * `model` - The gravity model to use, of type "GravityModel"
///
/// # References
///    
/// * For details of models, see: <http://icgem.gfz-potsdam.de/tom_longtime>
///
/// * For details of calculation, see Chapter 3.2 of:
///   "Satellite Orbits: Models, Methods, Applications",
///   O. Montenbruck and B. Gill, Springer, 2012.
///
pub fn accel(pos_itrf: &Vec3, order: usize, model: GravityModel) -> Vec3 {
    gravhash().get(&model).unwrap().accel(pos_itrf, order)
}

///
/// Return acceleration due to Earth gravity at the input position. , as
/// well as acceleratian partials with respect to ITRF position, e.e.
/// d a / dr
///
/// The acceleration does not include the centrifugal force, and is output
/// in m/s^2 in the International Terrestrial Reference Frame (ITRF)
///
/// # Inputs Arguments
///
/// * `pos` - nalgebra 3-vector representing ITRF position in meters
///
/// * `order` - The order of the gravity model to use.
///               Maximum is 16
///
/// * `model` - The gravity model to use, of type "GravityModel"
///
///  
/// # References
///
/// * For details of models, see: <http://icgem.gfz-potsdam.de/tom_longtime>
///
/// * For details of calculation, see Chapter 3.2 of:
///   "Satellite Orbits: Models, Methods, Applications",
///   O. Montenbruck and B. Gill, Springer, 2012.
///
pub fn accel_and_partials(pos_itrf: &Vec3, order: usize, model: GravityModel) -> (Vec3, Mat3) {
    gravhash()
        .get(&model)
        .unwrap()
        .accel_and_partials(pos_itrf, order)
}

pub fn accel_jgm3(pos_itrf: &Vec3, order: usize) -> Vec3 {
    jgm3().accel(pos_itrf, order)
}

#[derive(Debug, Clone)]
pub struct Gravity {
    pub name: String,
    pub gravity_constant: f64,
    pub radius: f64,
    pub max_degree: usize,
    pub coeffs: CoeffTable,
    pub divisor_table: DivisorTable,
    pub divisor_table2: DivisorTable,
}

type Legendre<const N: usize> = na::SMatrix<f64, N, N>;
type Vec3 = na::Vector3<f64>;
type Mat3 = na::Matrix3<f64>;

///
/// Return acceleration due to Earth gravity at the input position. The
/// acceleration does not include the centrifugal force, and is output
/// in m/s^2 in the International Terrestrial Reference Frame (ITRF)
///
/// # Inputs Arguments
///
/// * `pos` - Position as ITRF coordinate (satkit.itrfcoord) or numpy
///                3-vector representing ITRF position in meters
///
/// * `order` - Order of the gravity model, up to 40
///
/// # References
///
/// See Equation 3.33 of Montenbruck & Gill (referenced above) for
/// calculation details.
impl Gravity {
    pub fn accel(&self, pos: &Vec3, order: usize) -> Vec3 {
        // This is tedious, but using generics allows for vectors to be
        // allocated on the stack, which is faster
        if order == 1 {
            self.accel_t::<1, 5>(pos)
        } else if order == 2 {
            self.accel_t::<2, 6>(pos)
        } else if order == 3 {
            self.accel_t::<3, 7>(pos)
        } else if order == 4 {
            self.accel_t::<4, 8>(pos)
        } else if order == 5 {
            self.accel_t::<5, 9>(pos)
        } else if order == 6 {
            self.accel_t::<6, 10>(pos)
        } else if order == 7 {
            self.accel_t::<7, 11>(pos)
        } else if order == 8 {
            self.accel_t::<8, 12>(pos)
        } else if order == 9 {
            self.accel_t::<9, 13>(pos)
        } else if order == 10 {
            self.accel_t::<10, 14>(pos)
        } else if order == 11 {
            self.accel_t::<11, 15>(pos)
        } else if order == 12 {
            self.accel_t::<12, 16>(pos)
        } else if order == 13 {
            self.accel_t::<13, 17>(pos)
        } else if order == 14 {
            self.accel_t::<14, 18>(pos)
        } else if order == 15 {
            self.accel_t::<15, 19>(pos)
        } else if order == 16 {
            self.accel_t::<16, 20>(pos)
        } else if order == 17 {
            self.accel_t::<17, 21>(pos)
        } else if order == 18 {
            self.accel_t::<18, 22>(pos)
        } else if order == 19 {
            self.accel_t::<19, 23>(pos)
        } else if order == 20 {
            self.accel_t::<20, 24>(pos)
        } else if order == 21 {
            self.accel_t::<21, 25>(pos)
        } else if order == 22 {
            self.accel_t::<22, 26>(pos)
        } else if order == 23 {
            self.accel_t::<23, 27>(pos)
        } else if order == 24 {
            self.accel_t::<24, 28>(pos)
        } else if order == 25 {
            self.accel_t::<25, 29>(pos)
        } else if order == 26 {
            self.accel_t::<26, 30>(pos)
        } else if order == 27 {
            self.accel_t::<27, 31>(pos)
        } else if order == 28 {
            self.accel_t::<28, 32>(pos)
        } else if order == 29 {
            self.accel_t::<29, 33>(pos)
        } else if order == 30 {
            self.accel_t::<30, 34>(pos)
        } else if order == 31 {
            self.accel_t::<31, 35>(pos)
        } else if order == 32 {
            self.accel_t::<32, 36>(pos)
        } else if order == 33 {
            self.accel_t::<33, 37>(pos)
        } else if order == 34 {
            self.accel_t::<34, 38>(pos)
        } else if order == 35 {
            self.accel_t::<35, 39>(pos)
        } else if order == 36 {
            self.accel_t::<36, 40>(pos)
        } else if order == 37 {
            self.accel_t::<37, 41>(pos)
        } else if order == 38 {
            self.accel_t::<38, 42>(pos)
        } else if order == 39 {
            self.accel_t::<39, 43>(pos)
        } else {
            self.accel_t::<40, 44>(pos)
        }
    }

    pub fn accel_and_partials(&self, pos: &Vec3, order: usize) -> (Vec3, na::Matrix3<f64>) {
        // This is tedious, but using generics allows for vectors to be
        // allocated on the stack, which is faster
        if order == 1 {
            self.accel_and_partials_t::<1, 5>(pos)
        } else if order == 2 {
            self.accel_and_partials_t::<2, 6>(pos)
        } else if order == 3 {
            self.accel_and_partials_t::<3, 7>(pos)
        } else if order == 4 {
            self.accel_and_partials_t::<4, 8>(pos)
        } else if order == 5 {
            self.accel_and_partials_t::<5, 9>(pos)
        } else if order == 6 {
            self.accel_and_partials_t::<6, 10>(pos)
        } else if order == 7 {
            self.accel_and_partials_t::<7, 11>(pos)
        } else if order == 8 {
            self.accel_and_partials_t::<8, 12>(pos)
        } else if order == 9 {
            self.accel_and_partials_t::<9, 13>(pos)
        } else if order == 10 {
            self.accel_and_partials_t::<10, 14>(pos)
        } else if order == 11 {
            self.accel_and_partials_t::<11, 15>(pos)
        } else if order == 12 {
            self.accel_and_partials_t::<12, 16>(pos)
        } else if order == 13 {
            self.accel_and_partials_t::<13, 17>(pos)
        } else if order == 14 {
            self.accel_and_partials_t::<14, 18>(pos)
        } else if order == 15 {
            self.accel_and_partials_t::<15, 19>(pos)
        } else if order == 16 {
            self.accel_and_partials_t::<16, 20>(pos)
        } else if order == 17 {
            self.accel_and_partials_t::<17, 21>(pos)
        } else if order == 18 {
            self.accel_and_partials_t::<18, 22>(pos)
        } else if order == 19 {
            self.accel_and_partials_t::<19, 23>(pos)
        } else if order == 20 {
            self.accel_and_partials_t::<20, 24>(pos)
        } else if order == 21 {
            self.accel_and_partials_t::<21, 25>(pos)
        } else if order == 22 {
            self.accel_and_partials_t::<22, 26>(pos)
        } else if order == 23 {
            self.accel_and_partials_t::<23, 27>(pos)
        } else if order == 24 {
            self.accel_and_partials_t::<24, 28>(pos)
        } else if order == 25 {
            self.accel_and_partials_t::<25, 29>(pos)
        } else if order == 26 {
            self.accel_and_partials_t::<26, 30>(pos)
        } else if order == 27 {
            self.accel_and_partials_t::<27, 31>(pos)
        } else if order == 28 {
            self.accel_and_partials_t::<28, 32>(pos)
        } else if order == 29 {
            self.accel_and_partials_t::<29, 33>(pos)
        } else if order == 30 {
            self.accel_and_partials_t::<30, 34>(pos)
        } else if order == 31 {
            self.accel_and_partials_t::<31, 35>(pos)
        } else if order == 32 {
            self.accel_and_partials_t::<32, 36>(pos)
        } else if order == 33 {
            self.accel_and_partials_t::<33, 37>(pos)
        } else if order == 34 {
            self.accel_and_partials_t::<34, 38>(pos)
        } else if order == 35 {
            self.accel_and_partials_t::<35, 39>(pos)
        } else if order == 36 {
            self.accel_and_partials_t::<36, 40>(pos)
        } else if order == 37 {
            self.accel_and_partials_t::<37, 41>(pos)
        } else if order == 38 {
            self.accel_and_partials_t::<38, 42>(pos)
        } else if order == 39 {
            self.accel_and_partials_t::<39, 43>(pos)
        } else {
            self.accel_and_partials_t::<40, 44>(pos)
        }
    }

    fn accel_and_partials_t<const N: usize, const NP4: usize>(
        &self,
        pos: &Vec3,
    ) -> (Vec3, na::Matrix3<f64>) {
        let (v, w) = self.compute_legendre::<NP4>(pos);
        let accel = self.accel_from_legendre_t::<N, NP4>(&v, &w);
        let partials = self.partials_from_legendre_t::<N, NP4>(&v, &w);
        (accel, partials)
    }

    fn accel_t<const N: usize, const NP4: usize>(&self, pos: &Vec3) -> Vec3 {
        let (v, w) = self.compute_legendre::<NP4>(pos);

        self.accel_from_legendre_t::<N, NP4>(&v, &w)
    }

    // Equations 7.65 to 7.69 in Montenbruck & Gill
    fn partials_from_legendre_t<const N: usize, const NP4: usize>(
        &self,
        v: &Legendre<NP4>,
        w: &Legendre<NP4>,
    ) -> na::Matrix3<f64> {
        let mut daxdx = 0.0;
        let mut daxdy = 0.0;
        let mut daxdz = 0.0;
        let mut daydz = 0.0;
        let mut dazdz = 0.0;

        for n in 0..(N + 1) {
            let np2 = n + 2;
            // m = 0
            let cnm = self.coeffs[(n, 0)];
            let fnp1 = (n + 1) as f64;
            let fnp21 = ((n + 2) * (n + 1)) as f64;
            let vnp2m = v[(np2, 0)];
            daxdx += 0.5 * cnm * fnp21.mul_add(-vnp2m, v[(np2, 2)]);
            daxdy += 0.5 * cnm * w[(np2, 2)];
            daxdz += fnp1 * cnm * v[(np2, 1)];
            daydz += fnp1 * cnm * w[(np2, 1)];
            dazdz += fnp21 * cnm * vnp2m;
        }
        for m in 1..(N + 1) {
            let mm1 = m - 1;
            let mp1 = m + 1;
            let mp2 = m + 2;

            for n in m..(N + 1) {
                let np2 = n + 2;
                let cnm = self.coeffs[(n, m)];
                let snm = self.coeffs[(mm1, n)];
                let fnmmp1 = (n - m + 1) as f64;
                let fnmmp21 = (fnmmp1 + 1.) * fnmmp1;
                let fnmmp31 = (fnmmp1 + 2.) * fnmmp21;
                let vnp2mm1 = v[(np2, mm1)];
                let wnp2mm1 = w[(np2, mm1)];
                let vnp2m = v[(np2, m)];
                let wnp2m = w[(np2, m)];
                let wnp2mp1 = w[(np2, mp1)];
                let vnp2mp1 = v[(np2, mp1)];
                let vnp2mp2 = v[(np2, mp2)];
                let wnp2mp2 = w[(np2, mp2)];

                if m == 1 {
                    daxdx += 0.25
                        * fnmmp21.mul_add(
                            -(3.0 * cnm).mul_add(vnp2m, snm * wnp2m),
                            cnm.mul_add(vnp2mp2, snm * wnp2mp2),
                        );
                    daxdy += 0.25
                        * fnmmp21.mul_add(
                            -cnm.mul_add(wnp2m, snm * vnp2m),
                            cnm.mul_add(wnp2mp2, -(snm * vnp2mp2)),
                        );
                } else {
                    let mm2 = m - 2;
                    let fnmmp41 = (fnmmp1 + 3.0) * fnmmp31;
                    let vnp2mm2 = v[(np2, mm2)];
                    let wnp2mm2 = w[(np2, mm2)];
                    daxdx += 0.25
                        * fnmmp41.mul_add(
                            cnm.mul_add(vnp2mm2, snm * wnp2mm2),
                            (2.0 * fnmmp21).mul_add(
                                -cnm.mul_add(vnp2m, snm * wnp2m),
                                cnm.mul_add(vnp2mp2, snm * wnp2mp2),
                            ),
                        );
                    daxdy += 0.25
                        * fnmmp41.mul_add(
                            -cnm.mul_add(wnp2mm2, -(snm * vnp2mm2)),
                            cnm.mul_add(wnp2mp2, -(snm * vnp2mp2)),
                        );
                }
                daxdz += 0.5
                    * fnmmp1.mul_add(
                        cnm.mul_add(vnp2mp1, snm * wnp2mp1),
                        -(fnmmp31 * cnm.mul_add(vnp2mm1, snm * wnp2mm1)),
                    );
                daydz += 0.5
                    * fnmmp1.mul_add(
                        cnm.mul_add(wnp2mp1, -(snm * vnp2mp1)),
                        fnmmp31 * cnm.mul_add(wnp2mm1, -(snm * vnp2mm1)),
                    );
                dazdz += fnmmp21 * cnm.mul_add(vnp2m, snm * wnp2m);
            }
        }

        // From fact that laplacian is zero
        let daydy = -daxdx - dazdz;
        na::Matrix3::<f64>::new(
            daxdx, daxdy, daxdz, daxdy, daydy, daydz, daxdz, daydz, dazdz,
        ) * self.gravity_constant
            / self.radius.powi(3)
    }

    /// See Equation 3.33 in Montenbruck & Gill
    fn accel_from_legendre_t<const N: usize, const NP4: usize>(
        &self,
        v: &Legendre<NP4>,
        w: &Legendre<NP4>,
    ) -> Vec3 {
        let mut accel = Vec3::zeros();

        for n in 0..(N + 1) {
            for m in 0..(n + 1) {
                let cnm = self.coeffs[(n, m)];
                let mut snm = 0.0;
                if m > 0 {
                    snm = self.coeffs[(m - 1, n)];
                }
                if m == 0 {
                    accel[0] -= cnm * v[(n + 1, 1)];
                    accel[1] -= cnm * w[(n + 1, 1)];
                } else {
                    accel[0] += 0.5
                        * ((n - m + 2) as f64 * (n - m + 1) as f64).mul_add(
                            cnm.mul_add(v[(n + 1, m - 1)], snm * w[(n + 1, m - 1)]),
                            (-cnm).mul_add(v[(n + 1, m + 1)], -(snm * w[(n + 1, m + 1)])),
                        );

                    accel[1] += 0.5
                        * ((n - m + 2) as f64 * (n - m + 1) as f64).mul_add(
                            (-1.0 * cnm).mul_add(w[(n + 1, m - 1)], snm * v[(n + 1, m - 1)]),
                            (-cnm).mul_add(w[(n + 1, m + 1)], snm * v[(n + 1, m + 1)]),
                        );
                }
                accel[2] += (n - m + 1) as f64
                    * (-1.0 * cnm).mul_add(v[(n + 1, m)], -(snm * w[(n + 1, m)]));
            }
        }

        accel * self.gravity_constant / self.radius / self.radius
    }

    fn compute_legendre<const NP4: usize>(&self, pos: &Vec3) -> (Legendre<NP4>, Legendre<NP4>) {
        let rsq = pos.norm_squared();
        let scale = self.radius / rsq;
        let xfac = pos[0] * scale;
        let yfac = pos[1] * scale;
        let zfac = pos[2] * scale;
        let rfac = self.radius * scale;

        let mut v = Legendre::<NP4>::zeros();
        let mut w = Legendre::<NP4>::zeros();

        let mut vmm1mm1 = self.radius / rsq.sqrt();
        let mut wmm1mm1 = 0.0;
        v[(0, 0)] = vmm1mm1;
        w[(0, 0)] = wmm1mm1;

        for m in 0..NP4 {
            if m > 0 {
                let d = self.divisor_table[(m, m)];
                v[(m, m)] = d * xfac.mul_add(vmm1mm1, -(yfac * wmm1mm1));
                w[(m, m)] = d * xfac.mul_add(wmm1mm1, yfac * vmm1mm1);
            }

            vmm1mm1 = v[(m, m)];
            wmm1mm1 = w[(m, m)];
            let mut vnm2m = vmm1mm1;
            let mut wnm2m = wmm1mm1;

            let n = m + 1;
            if n >= NP4 {
                continue;
            }
            let d = self.divisor_table[(n, m)] * zfac;
            let mut vnm1m = d * vnm2m;
            let mut wnm1m = d * wnm2m;
            v[(n, m)] = vnm1m;
            w[(n, m)] = wnm1m;

            for n in (m + 2)..NP4 {
                let d = self.divisor_table[(n, m)] * zfac;
                let d2 = self.divisor_table2[(n, m)] * rfac;
                let vnm = d.mul_add(vnm1m, -(d2 * vnm2m));
                let wnm = d.mul_add(wnm1m, -(d2 * wnm2m));
                v[(n, m)] = vnm;
                w[(n, m)] = wnm;
                vnm2m = vnm1m;
                vnm1m = vnm;

                wnm2m = wnm1m;
                wnm1m = wnm;
            }
        }

        (v, w)
    }

    /// Load Gravity model coefficients from file
    /// Files are at:
    /// <http://icgem.gfz-potsdam.de/tom_longtime>
    pub fn from_file(filename: &str) -> Result<Self> {
        let path = datadir().unwrap_or(PathBuf::from(".")).join(filename);
        download_if_not_exist(&path, None)?;

        /*
        if !path.is_file() {
            return skerror!("File does not exist");
        }
        */

        let file = std::fs::File::open(&path).context("Failed to open gravity model file")?;

        let mut name = String::new();
        let mut gravity_constant: f64 = 0.0;
        let mut radius: f64 = 0.0;
        let mut max_degree: usize = 0;
        let mut header_cnt = 0;

        let lines: Vec<String> = io::BufReader::new(file)
            .lines()
            .map(|x| x.unwrap_or(String::from("")))
            .collect();

        // Read header lines
        for line in &lines {
            header_cnt += 1;

            let s: Vec<&str> = line.split_whitespace().collect();
            if s.len() < 2 {
                continue;
            }
            if s[0] == "modelname" {
                name = String::from(s[1]);
            } else if s[0] == "earth_gravity_constant" {
                gravity_constant = s[1].parse::<f64>()?;
            } else if s[0] == "radius" {
                radius = s[1].parse::<f64>()?;
            } else if s[0] == "max_degree" {
                max_degree = s[1].parse::<usize>()?;
                //cs = Some(na::DMatrix::<f64>::zeros(
                //    (max_degree + 1) as usize,
                //    (max_degree + 1) as usize,
                //));
            } else if s[0] == "end_of_head" {
                break;
            }
        }
        if max_degree == 0 {
            bail!("Invalid file; did not find max degree");
        }

        // Create matrix with lookup values
        let mut cs: CoeffTable = CoeffTable::zeros(max_degree + 1, max_degree + 1);

        for line in &lines[header_cnt..] {
            let s: Vec<&str> = line.split_whitespace().collect();
            if s.len() < 3 {
                bail!("Invalid line: {}", line);
            }

            let n: usize = s[1].parse()?;
            let m: usize = s[2].parse()?;
            let v1: f64 = s[3].parse()?;
            cs[(n, m)] = v1;
            if m > 0 {
                let v2: f64 = s[4].parse()?;
                cs[(m - 1, n)] = v2;
            }
        }

        // Convert from normalized coefficients to actual coefficients
        for n in 0..(max_degree + 1) {
            for m in 0..(n + 1) {
                let mut scale: f64 = 1.0;
                for k in (n - m + 1)..(n + m + 1) {
                    scale *= k as f64;
                }
                scale /= 2.0f64.mul_add(n as f64, 1.0);
                if m > 0 {
                    scale /= 2.0;
                }
                scale = 1.0 / f64::sqrt(scale);
                cs[(n, m)] *= scale;

                if m > 0 {
                    cs[(m - 1, n)] *= scale;
                }
            }
        }

        let mut d1 = DivisorTable::zeros();
        let mut d2 = DivisorTable::zeros();
        for m in 0..43 {
            if m > 0 {
                d1[(m, m)] = (2 * m - 1) as f64
            }
            let n = m + 1;
            d1[(n, m)] = (2 * n - 1) as f64 / (n - m) as f64;
            for n in (m + 2)..43 {
                d1[(n, m)] = (2 * n - 1) as f64 / (n - m) as f64;
                d2[(n, m)] = (n + m - 1) as f64 / (n - m) as f64;
            }
        }

        Ok(Self {
            name,
            gravity_constant,
            radius,
            max_degree,
            coeffs: cs,
            divisor_table: d1,
            divisor_table2: d2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::consts::OMEGA_EARTH;
    use crate::itrfcoord::ITRFCoord;
    use crate::types::Vec3;
    use approx::assert_relative_eq;

    #[test]
    fn test_gravity2() {
        // Lexington, ma
        let latitude: f64 = 42.4473;
        let longitude: f64 = -71.2272;
        let altitude: f64 = 0.0;
        let coord = ITRFCoord::from_geodetic_deg(latitude, longitude, altitude);
        let gaccel: Vec3 = jgm3().accel(&coord.into(), 6);
        let gaccel_truth = na::vector![-2.3360599811572618, 6.8730769266931615, -6.616497962860285];
        assert_relative_eq!(gaccel, gaccel_truth, max_relative = 1.0e-6);
    }

    #[test]
    fn test_gravity() {
        // Lexington, ma
        let latitude: f64 = 42.4473;
        let longitude: f64 = -71.2272;
        let altitude: f64 = 0.0;

        // reference gravity computations, using
        // JGM3 model, with 16 terms, found at:
        // http://icgem.gfz-potsdam.de/calcstat/
        // Outputs from above web page below:
        let reference_gravitation: f64 = 9.822206169031;
        // "gravity" includes centrifugal force, "gravitation" does not
        let reference_gravity: f64 = 9.803696372738;
        // Gravity deflections from normal along east-west and north-south
        // direction, in arcseconds
        let reference_ew_deflection_asec: f64 = -1.283542043355E+00;
        let reference_ns_deflection_asec: f64 = -1.311709802440E+00;

        let g = Gravity::from_file("JGM3.gfc").unwrap();
        let coord = ITRFCoord::from_geodetic_deg(latitude, longitude, altitude);
        let gravitation: Vec3 = g.accel(&coord.into(), 16);
        let centrifugal: Vec3 =
            Vec3::new(coord.itrf[0], coord.itrf[1], 0.0) * OMEGA_EARTH * OMEGA_EARTH;
        let gravity = gravitation + centrifugal;

        // Check gravitation matches the reference value
        // from http://icgem.gfz-potsdam.de/calcstat/
        assert!(f64::abs(gravitation.norm() / reference_gravitation - 1.0) < 1.0E-9);
        // Check that gravity matches reference value
        assert!(f64::abs(gravity.norm() / reference_gravity - 1.0) < 1.0E-9);

        // Rotate to ENU coordinate frame
        let g_enu: Vec3 = coord.q_enu2itrf().conjugate() * gravity;

        // Compute East/West and North/South deflections, in arcsec
        let ew_deflection: f64 = (-f64::atan2(g_enu[0], -g_enu[2])).to_degrees() * 3600.0;
        let ns_deflection: f64 = (-f64::atan2(g_enu[1], -g_enu[2])).to_degrees() * 3600.0;

        // Compare with reference values
        assert_relative_eq!(
            ew_deflection,
            reference_ew_deflection_asec,
            max_relative = 1.0e-5
        );
        assert_relative_eq!(
            ns_deflection,
            reference_ns_deflection_asec,
            max_relative = 1.0e-5
        );
    }

    #[test]
    fn test_partials() {
        use rand::random;
        let g = Gravity::from_file("JGM3.gfc").unwrap();

        for _idx in 0..100 {
            // Generate a random coordinate
            let latitude = random::<f64>() * 360.0;
            let longitude = random::<f64>().mul_add(180.0, -90.0);
            let altitude = random::<f64>().mul_add(100.0, 500.0);
            let coord = ITRFCoord::from_geodetic_deg(latitude, longitude, altitude);

            // generate a random shift
            let dpos = nalgebra::Vector3::<f64>::new(
                random::<f64>() * 100.0,
                random::<f64>() * 100.0,
                random::<f64>() * 100.0,
            );

            // get acceleration and partials at coordinate
            let (accel1, partials) = g.accel_and_partials(&coord.itrf, 6);

            // apply (small) random shift
            let v2 = coord.itrf + dpos;

            // get gravity accelaration at new coordinate
            let accel2 = g.accel(&v2, 6);

            // Get what would be expected from partial derivative
            let accel3 = accel1 + partials * dpos;

            // show that they are approximately equal
            assert_relative_eq!(accel2, accel3, max_relative = 1.0e-4);
        }
    }
}
