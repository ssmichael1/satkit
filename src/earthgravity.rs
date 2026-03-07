use crate::utils::{datadir, download_if_not_exist};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use crate::mathtypes::*;
type CoeffTable = DMatrix<f64>;

type DivisorTable = Matrix<44, 44>;

use once_cell::sync::OnceCell;

///
/// Gravity model enumeration
///
/// For details of models, see:
/// <http://icgem.gfz-potsdam.de/tom_longtime>
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GravityModel {
    JGM3,
    JGM2,
    EGM96,
    ITUGrace16,
}

impl std::fmt::Display for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GravityModel::JGM3 => write!(f, "JGM3"),
            GravityModel::JGM2 => write!(f, "JGM2"),
            GravityModel::EGM96 => write!(f, "EGM96"),
            GravityModel::ITUGrace16 => write!(f, "ITU_GRACE16"),
        }
    }
}

impl GravityModel {
    /// Get the singleton Gravity instance for this model
    pub fn get(&self) -> &'static Gravity {
        match self {
            GravityModel::JGM3 => jgm3(),
            GravityModel::JGM2 => jgm2(),
            GravityModel::EGM96 => egm96(),
            GravityModel::ITUGrace16 => itu_grace16(),
        }
    }
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
/// * `degree` - The maximum degree of the gravity model to use.
///   Maximum is 40
///
/// * `order` - The maximum order of the gravity model to use.
///   Must be ≤ `degree`.
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
pub fn accel(pos_itrf: &Vector3, degree: usize, order: usize, model: GravityModel) -> Vector3 {
    gravhash().get(&model).unwrap().accel(pos_itrf, degree, order)
}

///
/// Return acceleration due to Earth gravity at the input position, as
/// well as acceleration partials with respect to ITRF position, i.e.
/// d a / dr
///
/// The acceleration does not include the centrifugal force, and is output
/// in m/s^2 in the International Terrestrial Reference Frame (ITRF)
///
/// # Arguments
///
/// * `pos` - nalgebra 3-vector representing ITRF position in meters
///
/// * `degree` - The maximum degree of the gravity model to use.
///   Maximum is 40
///
/// * `order` - The maximum order of the gravity model to use.
///   Must be ≤ `degree`.
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
pub fn accel_and_partials(
    pos_itrf: &Vector3,
    degree: usize,
    order: usize,
    model: GravityModel,
) -> (Vector3, Matrix3) {
    gravhash()
        .get(&model)
        .unwrap()
        .accel_and_partials(pos_itrf, degree, order)
}

pub fn accel_jgm3(pos_itrf: &Vector3, degree: usize, order: usize) -> Vector3 {
    jgm3().accel(pos_itrf, degree, order)
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

type Legendre<const N: usize> = Matrix<N, N>;

/// Dispatch a runtime `degree` value to a const-generic method call.
/// NP4 is always degree + 4; the const expression `{ $d + 4 }` preserves
/// stack allocation for the Legendre matrices.
macro_rules! dispatch_degree {
    ($self:expr, $method:ident ($arg1:expr, $arg2:expr), $degree:expr,
     $($d:literal),+ $(,)?) => {
        match $degree {
            $($d => $self.$method::<$d, { $d + 4 }>($arg1, $arg2),)+
            _ => $self.$method::<40, 44>($arg1, $arg2),
        }
    };
}

///
/// Return acceleration due to Earth gravity at the input position. The
/// acceleration does not include the centrifugal force, and is output
/// in m/s^2 in the International Terrestrial Reference Frame (ITRF)
///
/// # Inputs Arguments
///
/// * `pos` - Position as ITRF coordinate (satkit.itrfcoord) or numpy
///   3-vector representing ITRF position in meters
///
/// * `order` - Order of the gravity model, up to 40
///
/// # References
///
/// See Equation 3.33 of Montenbruck & Gill (referenced above) for
/// calculation details.
impl Gravity {
    pub fn accel(&self, pos: &Vector3, degree: usize, order: usize) -> Vector3 {
        let max_order = order.min(degree);
        dispatch_degree!(self, accel_t(pos, max_order), degree,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39,
        )
    }

    pub fn accel_and_partials(&self, pos: &Vector3, degree: usize, order: usize) -> (Vector3, Matrix3) {
        let max_order = order.min(degree);
        dispatch_degree!(self, accel_and_partials_t(pos, max_order), degree,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39,
        )
    }

    fn accel_and_partials_t<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> (Vector3, Matrix3) {
        let (v, w) = self.compute_legendre::<NP4>(pos);
        let accel = self.accel_from_legendre_t::<N, NP4>(&v, &w, max_order);
        let partials = self.partials_from_legendre_t::<N, NP4>(&v, &w, max_order);
        (accel, partials)
    }

    fn accel_t<const N: usize, const NP4: usize>(&self, pos: &Vector3, max_order: usize) -> Vector3 {
        let (v, w) = self.compute_legendre::<NP4>(pos);

        self.accel_from_legendre_t::<N, NP4>(&v, &w, max_order)
    }

    // Equations 7.65 to 7.69 in Montenbruck & Gill
    fn partials_from_legendre_t<const N: usize, const NP4: usize>(
        &self,
        v: &Legendre<NP4>,
        w: &Legendre<NP4>,
        max_order: usize,
    ) -> Matrix3 {
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
        let max_m = (N + 1).min(max_order + 1);
        for m in 1..max_m {
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
        Matrix3::new(
            daxdx, daxdy, daxdz, daxdy, daydy, daydz, daxdz, daydz, dazdz,
        ) * self.gravity_constant
            / self.radius.powi(3)
    }

    /// See Equation 3.33 in Montenbruck & Gill
    fn accel_from_legendre_t<const N: usize, const NP4: usize>(
        &self,
        v: &Legendre<NP4>,
        w: &Legendre<NP4>,
        max_order: usize,
    ) -> Vector3 {
        let mut ax = 0.0;
        let mut ay = 0.0;
        let mut az = 0.0;

        // m = 0 terms
        for n in 0..(N + 1) {
            let cnm = self.coeffs[(n, 0)];
            ax -= cnm * v[(n + 1, 1)];
            ay -= cnm * w[(n + 1, 1)];
            az -= (n + 1) as f64 * cnm * v[(n + 1, 0)];
        }

        // m > 0 terms
        let max_m = (N + 1).min(max_order + 1);
        for m in 1..max_m {
            for n in m..(N + 1) {
                let cnm = self.coeffs[(n, m)];
                let snm = self.coeffs[(m - 1, n)];
                let fnmmp21 = (n - m + 2) as f64 * (n - m + 1) as f64;

                ax += 0.5
                    * fnmmp21.mul_add(
                        cnm.mul_add(v[(n + 1, m - 1)], snm * w[(n + 1, m - 1)]),
                        (-cnm).mul_add(v[(n + 1, m + 1)], -(snm * w[(n + 1, m + 1)])),
                    );

                ay += 0.5
                    * fnmmp21.mul_add(
                        (-cnm).mul_add(w[(n + 1, m - 1)], snm * v[(n + 1, m - 1)]),
                        (-cnm).mul_add(w[(n + 1, m + 1)], snm * v[(n + 1, m + 1)]),
                    );

                az -= (n - m + 1) as f64 * cnm.mul_add(v[(n + 1, m)], snm * w[(n + 1, m)]);
            }
        }

        Vector3::new(ax, ay, az) * self.gravity_constant / self.radius / self.radius
    }

    fn compute_legendre<const NP4: usize>(&self, pos: &Vector3) -> (Legendre<NP4>, Legendre<NP4>) {
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
    use crate::mathtypes::Vector3;
    use approx::assert_relative_eq;

    #[test]
    fn test_gravity_order_1() {
        // Order 1 = point mass: accel should be μ/r², radially inward
        let r = 7000.0e3; // 7000 km
        let pos = Vector3::new(r, 0.0, 0.0);
        let accel = jgm3().accel(&pos, 1, 1);
        let expected_mag = crate::consts::MU_EARTH / (r * r);
        assert_relative_eq!(accel.norm(), expected_mag, max_relative = 1.0e-6);
        // Should point radially inward (negative x)
        assert!(accel[0] < 0.0);
        assert!(accel[1].abs() < 1.0e-10);
        assert!(accel[2].abs() < 1.0e-10);
    }

    #[test]
    fn test_gravity_models_agree_order1() {
        // At order 1 (point mass), all models should agree closely
        let pos = Vector3::new(7000.0e3, 1000.0e3, 3000.0e3);
        let a_jgm3 = jgm3().accel(&pos, 1, 1);
        let a_jgm2 = jgm2().accel(&pos, 1, 1);
        let a_egm96 = egm96().accel(&pos, 1, 1);
        let a_grace = itu_grace16().accel(&pos, 1, 1);
        // All should be very close (small differences due to different GM values)
        assert_relative_eq!(a_jgm3, a_jgm2, max_relative = 1.0e-6);
        assert_relative_eq!(a_jgm3, a_egm96, max_relative = 1.0e-6);
        assert_relative_eq!(a_jgm3, a_grace, max_relative = 1.0e-6);
    }

    #[test]
    fn test_gravity_increases_with_order() {
        // Off-equator point: higher-order (J2 effect) should differ from order 1
        let coord = ITRFCoord::from_geodetic_deg(60.0, 30.0, 300.0e3);
        let a1 = jgm3().accel(&coord.itrf, 1, 1);
        let a16 = jgm3().accel(&coord.itrf, 16, 16);
        // They should differ (J2 effect is ~1e-3 relative)
        let diff = (a16 - a1).norm() / a1.norm();
        assert!(
            diff > 1.0e-4,
            "Order 16 vs 1 relative difference is {}, expected > 1e-4",
            diff
        );
    }

    #[test]
    fn test_gravity2() {
        // Lexington, ma
        let latitude: f64 = 42.4473;
        let longitude: f64 = -71.2272;
        let altitude: f64 = 0.0;
        let coord = ITRFCoord::from_geodetic_deg(latitude, longitude, altitude);
        let gaccel: Vector3 = jgm3().accel(&coord.itrf, 6, 6);
        let gaccel_truth =
            nalgebra::vector![-2.3360599811572618, 6.8730769266931615, -6.616497962860285];
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
        let gravitation: Vector3 = g.accel(&coord.itrf, 16, 16);
        let centrifugal: Vector3 =
            Vector3::new(coord.itrf[0], coord.itrf[1], 0.0) * OMEGA_EARTH * OMEGA_EARTH;
        let gravity = gravitation + centrifugal;

        // Check gravitation matches the reference value
        // from http://icgem.gfz-potsdam.de/calcstat/
        assert!(f64::abs(gravitation.norm() / reference_gravitation - 1.0) < 1.0E-9);
        // Check that gravity matches reference value
        assert!(f64::abs(gravity.norm() / reference_gravity - 1.0) < 1.0E-9);

        // Rotate to ENU coordinate frame
        let g_enu: Vector3 = coord.q_enu2itrf().conjugate() * gravity;

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
            let dpos = Vector3::new(
                random::<f64>() * 100.0,
                random::<f64>() * 100.0,
                random::<f64>() * 100.0,
            );

            // get acceleration and partials at coordinate
            let (accel1, partials) = g.accel_and_partials(&coord.itrf, 6, 6);

            // apply (small) random shift
            let v2 = coord.itrf + dpos;

            // get gravity accelaration at new coordinate
            let accel2 = g.accel(&v2, 6, 6);

            // Get what would be expected from partial derivative
            let accel3 = accel1 + partials * dpos;

            // show that they are approximately equal
            assert_relative_eq!(accel2, accel3, max_relative = 1.0e-4);
        }
    }

    #[test]
    fn test_zonal_only_differs_from_full() {
        // order=0 means zonal harmonics only (m=0 terms).
        // This should give a different result than order=degree for
        // a position that is off the polar axis.
        let coord = ITRFCoord::from_geodetic_deg(45.0, 30.0, 400.0e3);
        let a_full = jgm3().accel(&coord.itrf, 8, 8);
        let a_zonal = jgm3().accel(&coord.itrf, 8, 0);

        // They must differ (tesseral terms are non-zero off-axis)
        let diff = (a_full - a_zonal).norm();
        assert!(
            diff > 1.0e-6,
            "Zonal-only and full gravity should differ, got diff = {:e}",
            diff
        );

        // But both should be reasonable gravity magnitudes
        assert!(a_full.norm() > 5.0);
        assert!(a_zonal.norm() > 5.0);
    }

    #[test]
    fn test_order_less_than_degree() {
        // Verify that order < degree gives intermediate results
        let coord = ITRFCoord::from_geodetic_deg(45.0, 30.0, 400.0e3);
        let a_order0 = jgm3().accel(&coord.itrf, 8, 0);
        let a_order4 = jgm3().accel(&coord.itrf, 8, 4);
        let a_order8 = jgm3().accel(&coord.itrf, 8, 8);

        // All three should be distinct
        let diff_04 = (a_order4 - a_order0).norm();
        let diff_48 = (a_order8 - a_order4).norm();
        let diff_08 = (a_order8 - a_order0).norm();
        assert!(
            diff_04 > 1.0e-7,
            "order=4 and order=0 should differ, diff = {:e}",
            diff_04
        );
        assert!(
            diff_48 > 1.0e-7,
            "order=8 and order=4 should differ, diff = {:e}",
            diff_48
        );
        assert!(
            diff_08 > 1.0e-7,
            "order=8 and order=0 should differ, diff = {:e}",
            diff_08
        );
    }

    #[test]
    fn test_order_equals_degree_matches_legacy() {
        // When order == degree, results should be identical to the old behavior
        // (which implicitly set order = degree).
        // We verify this by comparing degree=6,order=6 against the known truth value.
        let latitude: f64 = 42.4473;
        let longitude: f64 = -71.2272;
        let coord = ITRFCoord::from_geodetic_deg(latitude, longitude, 0.0);
        let gaccel = jgm3().accel(&coord.itrf, 6, 6);
        let gaccel_truth =
            nalgebra::vector![-2.3360599811572618, 6.8730769266931615, -6.616497962860285];
        assert_relative_eq!(gaccel, gaccel_truth, max_relative = 1.0e-6);
    }

    #[test]
    fn test_partials_with_order_less_than_degree() {
        // Verify partials are consistent when order < degree
        let g = Gravity::from_file("JGM3.gfc").unwrap();
        let coord = ITRFCoord::from_geodetic_deg(45.0, 30.0, 400.0e3);
        let dpos = Vector3::new(50.0, -30.0, 80.0);

        // Use degree=6, order=2
        let (accel1, partials) = g.accel_and_partials(&coord.itrf, 6, 2);
        let v2 = coord.itrf + dpos;
        let accel2 = g.accel(&v2, 6, 2);
        let accel3 = accel1 + partials * dpos;

        assert_relative_eq!(accel2, accel3, max_relative = 1.0e-4);
    }
}
