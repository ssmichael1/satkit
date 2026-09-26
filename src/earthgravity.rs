use crate::utils::{datadir, download_if_not_exist};
use std::num::{ParseFloatError, ParseIntError};
use thiserror::Error;

/// Errors produced by the [`earthgravity`](crate::earthgravity) module.
#[derive(Debug, Error)]
pub enum Error {
    /// The header of the gravity model file did not declare a non-zero
    /// `max_degree`.
    #[error("Invalid file; did not find max degree")]
    MissingMaxDegree,

    /// A coefficient line had fewer than the required `n m C [S]` fields.
    #[error("Invalid line: {0}")]
    InvalidLine(String),

    /// Failed to open the gravity model file.
    #[error("Failed to open gravity model file: {0}")]
    OpenFailed(#[source] std::io::Error),

    /// No longer returned: [`Gravity::from_bytes`] decodes non-UTF-8 bytes
    /// (some ICGEM headers are Latin-1) lossily, since only the ASCII header
    /// keywords and coefficient rows are read. Retained so the variant set
    /// stays stable.
    #[error("gravity-model byte buffer is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// Returned by [`init_from_bytes`] / [`init_from_path`] when the gravity
    /// singleton for the requested [`GravityModel`] has already been
    /// initialized.
    #[error("gravity singleton for {0} is already initialized")]
    AlreadyInitialized(GravityModel),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ParseFloat(#[from] ParseFloatError),

    #[error(transparent)]
    ParseInt(#[from] ParseIntError),

    #[error(transparent)]
    Datadir(#[from] crate::utils::datadir::Error),

    #[error(transparent)]
    Download(#[from] crate::utils::download::Error),
}

/// Convenient type alias used throughout the `earthgravity` module.
pub type Result<T> = std::result::Result<T, Error>;

use crate::mathtypes::*;
type CoeffTable = DMatrix<f64>;

type DivisorTable = Matrix<MAX_COEFF_DIM, MAX_COEFF_DIM>;

/// Largest table dimension the evaluator can ever index. Acceleration and
/// partials are dispatched at degree ≤ 70 (see `dispatch_degree!`) and the
/// Cunningham recursion uses NP4 = degree + 4 ≤ 74, matching the 74×74
/// divisor tables. Storing coefficients beyond this is not just wasted
/// memory — for high-resolution models (EGM96 is degree 360, EGM2008 degree
/// 2190) it inflates the column-major stride of `coeffs`, scattering the
/// S-coefficient reads `coeffs[(m-1, n)]` across multi-KB strides and
/// thrashing the cache in the hot loops. Capping the stored table keeps the
/// working set small (a 74×74 table is 44 KB).
const MAX_COEFF_DIM: usize = MAX_GRAVITY_DEGREE as usize + 4;

/// Highest spherical-harmonic degree (and order) the evaluator supports.
///
/// The built-in coefficient tables are capped at [`MAX_COEFF_DIM`] rows and
/// the accelerator dispatches on degree ≤ 70; requests above this are
/// rejected by [`PropSettings::set_gravity`](crate::orbitprop::PropSettings::set_gravity)
/// and at [`propagate`](crate::orbitprop::propagate) entry rather than
/// silently clamped. The compiled-in models are truncated to exactly this
/// degree, so they give the same results as the full files. (EGM96 itself is
/// defined to degree 360 and EGM2008 to 2190; supporting that would need
/// heap-allocated Legendre tables — see the note on `MAX_COEFF_DIM`.)
pub const MAX_GRAVITY_DEGREE: u16 = 70;

use std::sync::OnceLock;

/// Parse a number that may use a Fortran `D` exponent (`1.0d0`,
/// `0.3986004415D+15`), as EGM2008 and the GGM05 files do.
fn parse_f64(tok: &str) -> std::result::Result<f64, ParseFloatError> {
    tok.parse::<f64>().or_else(|e| {
        if tok.contains(['d', 'D']) {
            tok.replace(['d', 'D'], "e").parse::<f64>()
        } else {
            Err(e)
        }
    })
}

///
/// Gravity model enumeration
///
/// For details of models, see:
/// <http://icgem.gfz-potsdam.de/tom_longtime>
///
/// EGM96, EGM2008, JGM2 and JGM3 are compiled into the library (truncated
/// to degree 70, the evaluator's cap) and need no data
/// directory or network. ITU_GRACE16 is fetched on first use through the
/// SHA-256-verified data manifest (its licence is CC BY 4.0, so it is not
/// redistributed inside the library); see [`ensure_loaded`].
///
/// Each model's tide system (see [`TideSystem`]) is recorded on load and the
/// propagator's solid-tide correction accounts for it, so any model can be
/// combined with any [`TideModel`](crate::orbitprop::TideModel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GravityModel {
    /// Joint Gravity Model 3 (Tapley et al. 1996). Zero-tide. Compiled in.
    JGM3,
    /// Joint Gravity Model 2 (Nerem et al. 1994). Tide-free. Compiled in.
    JGM2,
    /// Earth Gravitational Model 1996 (Lemoine et al. 1998). Tide-free.
    /// Compiled in. (The orbit propagator's default is [`EGM2008`](Self::EGM2008).)
    EGM96,
    /// ITU_GRACE16 (Akyilmaz et al. 2016), a GRACE-only satellite solution.
    /// Zero-tide. Downloaded on first use (CC BY 4.0).
    ITUGrace16,
    /// Earth Gravitational Model 2008 (Pavlis et al. 2012). Tide-free.
    /// Compiled in. The default for the orbit propagator.
    EGM2008,
}

impl std::fmt::Display for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::JGM3 => write!(f, "JGM3"),
            Self::JGM2 => write!(f, "JGM2"),
            Self::EGM96 => write!(f, "EGM96"),
            Self::ITUGrace16 => write!(f, "ITU_GRACE16"),
            Self::EGM2008 => write!(f, "EGM2008"),
        }
    }
}

impl GravityModel {
    /// Get the singleton Gravity instance for this model.
    ///
    /// # Panics
    ///
    /// Panics if the model's coefficient file cannot be loaded (for
    /// ITU_GRACE16: cannot be downloaded, e.g. offline). Call
    /// [`ensure_loaded`] first to get a typed error instead.
    pub fn get(&self) -> &'static Gravity {
        ensure_loaded(*self).unwrap_or_else(|e| {
            let fname = default_filename(*self);
            panic!(
                "Failed to load Earth gravity model {self:?} from \"{fname}\": {e}. \
                 Ensure the data files are present (set the SATKIT_DATA environment \
                 variable to your data directory, or run \
                 satkit::utils::update_datafiles to download them)."
            )
        })
    }
}

/// Permanent-tide convention of a gravity model's C̄20 coefficient
/// (IERS Conventions 2010, §6.2.2 and §1.1).
///
/// The Sun and Moon raise a *permanent* deformation of the Earth whose
/// contribution to C̄20 is A₀H₀k₂₀ ≈ −4.2×10⁻⁹. A **zero-tide** model keeps
/// that deformation in its coefficients (what a satellite actually senses);
/// a **tide-free** model has it removed with a conventional k₂₀. A
/// **mean-tide** model additionally keeps the direct tidal potential of
/// the bodies (A₀H₀ ≈ −1.4×10⁻⁸); it is a geoid convention and is not used
/// for orbit propagation, where the third-body force already supplies the
/// direct potential.
///
/// The IERS Step 1 solid-tide correction
/// ([`TideModel::SolidStep1`](crate::orbitprop::TideModel::SolidStep1))
/// includes the permanent tide, so it is only complete on a tide-free
/// model; for a zero-tide (or mean-tide) model the propagator removes the
/// permanent part from the correction
/// ([`tides::remove_permanent_tide`](crate::orbitprop::tides::remove_permanent_tide))
/// instead of double-counting it. The coefficients themselves are never
/// modified, so with tides off every model propagates exactly as published.
///
/// Read from the ICGEM `tide_system` header when present and recognised;
/// otherwise classified from the C̄20 value (see
/// [`TideSystem::classify_c20`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TideSystem {
    /// Permanent tide removed from C̄20 (ICGEM `tide_free`).
    TideFree,
    /// Permanent deformation retained in C̄20 (ICGEM `zero_tide`).
    ZeroTide,
    /// Permanent deformation and direct permanent potential retained
    /// (ICGEM `mean_tide`). Not appropriate for orbit propagation.
    MeanTide,
    /// No recognised header and a C̄20 that is not Earth-like (a custom or
    /// non-Earth model): treated as tide-free, i.e. the tide correction is
    /// applied as published.
    Unknown,
}

impl std::fmt::Display for TideSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::TideFree => "tide_free",
            Self::ZeroTide => "zero_tide",
            Self::MeanTide => "mean_tide",
            Self::Unknown => "unknown",
        })
    }
}

/// Fully-normalized C̄20 midway between the tide-free value of EGM96
/// (−4.84165372×10⁻⁴) and the zero-tide value of JGM3 / ITU_GRACE16
/// (−4.84169548×10⁻⁴): the two conventions differ by the permanent tide,
/// ≈4.2×10⁻⁹, so a headerless Earth model is classified by which side of
/// this it falls on.
const C20_TIDE_SYSTEM_THRESHOLD: f64 = -4.841675e-4;
/// Half-width of the C̄20 band accepted as "Earth-like" for classification.
/// Earth models agree to ~5×10⁻⁹; anything further off is a different body
/// or a synthetic test model.
const C20_EARTH_TOLERANCE: f64 = 1.0e-7;

impl TideSystem {
    /// Parse the value of an ICGEM `tide_system` header line. Accepts the
    /// documented `tide_free` / `zero_tide` / `mean_tide` and the
    /// space- or hyphen-separated spellings some files use (ITU_GRACE16
    /// writes `zero tide`). Anything else is [`Unknown`](Self::Unknown).
    pub fn from_header(value: &str) -> Self {
        let norm: String = value
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|c| {
                if c == '-' || c.is_whitespace() {
                    '_'
                } else {
                    c
                }
            })
            .collect();
        // Collapse runs of separators ("zero  tide" → "zero_tide").
        let norm = norm
            .split('_')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        match norm.as_str() {
            "tide_free" | "tidefree" => Self::TideFree,
            "zero_tide" | "zerotide" => Self::ZeroTide,
            "mean_tide" | "meantide" => Self::MeanTide,
            _ => Self::Unknown,
        }
    }

    /// Classify a model without a `tide_system` header from its
    /// fully-normalized C̄20. Returns [`Unknown`](Self::Unknown) when the
    /// value is not within 1×10⁻⁷ of Earth's.
    pub fn classify_c20(c20_normalized: f64) -> Self {
        if !c20_normalized.is_finite()
            || (c20_normalized - C20_TIDE_SYSTEM_THRESHOLD).abs() > C20_EARTH_TOLERANCE
        {
            Self::Unknown
        } else if c20_normalized < C20_TIDE_SYSTEM_THRESHOLD {
            Self::ZeroTide
        } else {
            Self::TideFree
        }
    }

    /// `true` if the model's C̄20 already contains the permanent tidal
    /// deformation, so a solid-tide correction that includes the permanent
    /// tide must have that part removed.
    pub const fn includes_permanent_tide(self) -> bool {
        matches!(self, Self::ZeroTide | Self::MeanTide)
    }
}

// Module-scope singletons. Moved out of their respective accessor
// functions so [`init_from_bytes`] / [`init_from_path`] can populate them
// before any caller triggers the lazy default load.
static JGM3_INSTANCE: OnceLock<Gravity> = OnceLock::new();
static JGM2_INSTANCE: OnceLock<Gravity> = OnceLock::new();
static EGM96_INSTANCE: OnceLock<Gravity> = OnceLock::new();
static ITU_GRACE16_INSTANCE: OnceLock<Gravity> = OnceLock::new();
static EGM2008_INSTANCE: OnceLock<Gravity> = OnceLock::new();

fn instance_for(model: GravityModel) -> &'static OnceLock<Gravity> {
    match model {
        GravityModel::JGM3 => &JGM3_INSTANCE,
        GravityModel::JGM2 => &JGM2_INSTANCE,
        GravityModel::EGM96 => &EGM96_INSTANCE,
        GravityModel::ITUGrace16 => &ITU_GRACE16_INSTANCE,
        GravityModel::EGM2008 => &EGM2008_INSTANCE,
    }
}

/// Default `.gfc` filename for the given model (used by the lazy
/// default-resolver to find the file under [`datadir`]).
const fn default_filename(model: GravityModel) -> &'static str {
    match model {
        GravityModel::JGM3 => "JGM3.gfc",
        GravityModel::JGM2 => "JGM2.gfc",
        GravityModel::EGM96 => "EGM96.gfc",
        GravityModel::ITUGrace16 => "ITU_GRACE16.gfc",
        GravityModel::EGM2008 => "EGM2008.gfc",
    }
}

/// Load the singleton for `model` if it is not loaded yet, returning a
/// typed error instead of panicking when its coefficient file is
/// unavailable.
///
/// The compiled-in models (EGM96, EGM2008, JGM2, JGM3) always succeed.
/// ITU_GRACE16 is fetched into the data directory on first use, so this
/// fails with a [`download`](crate::utils::download::Error) error when
/// offline (`SATKIT_OFFLINE=1`, a build without the `download` feature, or
/// no network) and no copy is present in a data search directory.
///
/// [`propagate`](crate::orbitprop::propagate) calls this on entry, and the
/// Python `gravity()` functions before evaluating; [`GravityModel::get`]
/// and the module-level [`accel`] panic with the same message instead.
pub fn ensure_loaded(model: GravityModel) -> Result<&'static Gravity> {
    let lock = instance_for(model);
    if let Some(g) = lock.get() {
        return Ok(g);
    }
    let gravity = Gravity::from_file(default_filename(model))?;
    // Two threads may both load; the first `set` wins and the other copy is
    // dropped — both parsed the same bytes.
    Ok(lock.get_or_init(|| gravity))
}

/// `true` if the singleton for `model` has been loaded (by any of
/// [`ensure_loaded`], [`GravityModel::get`], [`init_from_bytes`] or
/// [`init_from_path`]).
pub fn is_loaded(model: GravityModel) -> bool {
    instance_for(model).get().is_some()
}

///
/// Singleton for JGM3 gravity model
///
pub fn jgm3() -> &'static Gravity {
    GravityModel::JGM3.get()
}

///
/// Singleton for JGM2 gravity model
///
pub fn jgm2() -> &'static Gravity {
    GravityModel::JGM2.get()
}

///
/// Singleton for EGM96 gravity model
///
pub fn egm96() -> &'static Gravity {
    GravityModel::EGM96.get()
}

///
/// Singleton for ITU GRACE16 gravity model
///
/// Downloaded on first use; see [`ensure_loaded`] for the failure mode.
pub fn itu_grace16() -> &'static Gravity {
    GravityModel::ITUGrace16.get()
}

///
/// Singleton for EGM2008 gravity model
///
pub fn egm2008() -> &'static Gravity {
    GravityModel::EGM2008.get()
}

/// Initialize the gravity-model singleton for `model` from an in-memory
/// byte buffer.
///
/// The bytes must be a valid ICGEM `.gfc` text file (UTF-8).
///
/// Must be called *before* any acceleration / coefficient query for this
/// model, otherwise the lazy default-resolver init has already won and
/// this returns [`Error::AlreadyInitialized`].
pub fn init_from_bytes(model: GravityModel, bytes: &[u8]) -> Result<()> {
    let gravity = Gravity::from_bytes(bytes)?;
    instance_for(model)
        .set(gravity)
        .map_err(|_| Error::AlreadyInitialized(model))
}

/// Initialize the gravity-model singleton for `model` from a file at `path`.
///
/// Same semantics as [`init_from_bytes`]; see that function for details.
pub fn init_from_path(model: GravityModel, path: &std::path::Path) -> Result<()> {
    let gravity = Gravity::from_path(path)?;
    instance_for(model)
        .set(gravity)
        .map_err(|_| Error::AlreadyInitialized(model))
}

///
/// Return acceleration due to Earth gravity at the input position. The
/// acceleration does not include the centrifugal force, and is output
/// in m/s^2 in the International Terrestrial Reference Frame (ITRF)
///
/// # Arguments
///
/// * `pos` - 3-vector representing ITRF position in meters
///
/// * `degree` - The maximum degree of the gravity model to use.
///   Maximum is 70 ([`MAX_GRAVITY_DEGREE`]); a larger degree is evaluated
///   at 70 (the propagator rejects it instead).
///
/// * `order` - The maximum order of the gravity model to use.
///   Should be ≤ `degree`; a larger order is evaluated at `degree`.
///
/// * `model` - The gravity model to use, of type "GravityModel"
///
/// # References
///
/// * For details of models, see: <http://icgem.gfz-potsdam.de/tom_longtime>
///
/// * For details of calculation, see Chapter 3.2 of:
///   "Satellite Orbits: Models, Methods, Applications",
///   O. Montenbruck and E. Gill, Springer, 2000
///   (<https://doi.org/10.1007/978-3-642-58351-3>).
///
pub fn accel(pos_itrf: &Vector3, degree: usize, order: usize, model: GravityModel) -> Vector3 {
    model.get().accel(pos_itrf, degree, order)
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
/// * `pos` - 3-vector representing ITRF position in meters
///
/// * `degree` - The maximum degree of the gravity model to use.
///   Maximum is 70 ([`MAX_GRAVITY_DEGREE`]); a larger degree is evaluated
///   at 70 (the propagator rejects it instead).
///
/// * `order` - The maximum order of the gravity model to use.
///   Should be ≤ `degree`; a larger order is evaluated at `degree`.
///
/// * `model` - The gravity model to use, of type "GravityModel"
///
/// # References
///
/// * For details of models, see: <http://icgem.gfz-potsdam.de/tom_longtime>
///
/// * For details of calculation, see Chapter 3.2 of:
///   "Satellite Orbits: Models, Methods, Applications",
///   O. Montenbruck and E. Gill, Springer, 2000
///   (<https://doi.org/10.1007/978-3-642-58351-3>).
///
pub fn accel_and_partials(
    pos_itrf: &Vector3,
    degree: usize,
    order: usize,
    model: GravityModel,
) -> (Vector3, Matrix3) {
    model.get().accel_and_partials(pos_itrf, degree, order)
}

#[derive(Debug, Clone)]
pub struct Gravity {
    pub name: String,
    pub gravity_constant: f64,
    pub radius: f64,
    pub max_degree: usize,
    /// Permanent-tide convention of the coefficients, from the file's
    /// `tide_system` header or classified from C̄20 (see [`TideSystem`]).
    pub tide_system: TideSystem,
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
            _ => $self.$method::<{ MAX_GRAVITY_DEGREE as usize }, MAX_COEFF_DIM>($arg1, $arg2),
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
/// * `order` - Order of the gravity model, up to 70
///
/// # References
///
/// See Equation 3.33 of Montenbruck & Gill (referenced above) for
/// calculation details.
impl Gravity {
    pub fn accel(&self, pos: &Vector3, degree: usize, order: usize) -> Vector3 {
        // Clamp to the stored coefficient table: a custom low-degree model
        // combined with a larger requested degree would index past the table.
        // (A no-op for the built-in models, whose tables hold MAX_COEFF_DIM.)
        // The lower bound of 1 keeps degree 0 out of the `_ => 70` dispatch
        // arm; degree 1 is the point-mass field (the n=1 terms vanish for a
        // center-of-mass-origin model).
        let degree = degree.clamp(1, self.coeffs.nrows().saturating_sub(1));
        let max_order = order.min(degree);
        dispatch_degree!(
            self,
            accel_t(pos, max_order),
            degree,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
        )
    }

    pub fn accel_and_partials(
        &self,
        pos: &Vector3,
        degree: usize,
        order: usize,
    ) -> (Vector3, Matrix3) {
        // See the identical clamp in `accel` for rationale.
        let degree = degree.clamp(1, self.coeffs.nrows().saturating_sub(1));
        let max_order = order.min(degree);
        dispatch_degree!(
            self,
            accel_and_partials_t(pos, max_order),
            degree,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
        )
    }

    // On baseline x86-64, `f64::mul_add` is a call into the `fma` runtime
    // function per term, so the kernels are also compiled with the `fma`
    // feature and dispatched at runtime.

    fn accel_and_partials_t<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> (Vector3, Matrix3) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("fma") {
                // SAFETY: the `fma` feature was detected on this CPU.
                return unsafe { self.accel_and_partials_t_fma::<N, NP4>(pos, max_order) };
            }
        }
        self.accel_and_partials_t_inner::<N, NP4>(pos, max_order)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    unsafe fn accel_and_partials_t_fma<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> (Vector3, Matrix3) {
        self.accel_and_partials_t_inner::<N, NP4>(pos, max_order)
    }

    #[inline(always)]
    fn accel_and_partials_t_inner<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> (Vector3, Matrix3) {
        let (v, w) = self.compute_legendre::<NP4>(pos);
        let accel = self.accel_from_legendre_t::<N, NP4>(&v, &w, max_order);
        let partials = self.partials_from_legendre_t::<N, NP4>(&v, &w, max_order);
        (accel, partials)
    }

    fn accel_t<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> Vector3 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("fma") {
                // SAFETY: the `fma` feature was detected on this CPU.
                return unsafe { self.accel_t_fma::<N, NP4>(pos, max_order) };
            }
        }
        self.accel_t_inner::<N, NP4>(pos, max_order)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    unsafe fn accel_t_fma<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> Vector3 {
        self.accel_t_inner::<N, NP4>(pos, max_order)
    }

    #[inline(always)]
    fn accel_t_inner<const N: usize, const NP4: usize>(
        &self,
        pos: &Vector3,
        max_order: usize,
    ) -> Vector3 {
        let (v, w) = self.compute_legendre::<NP4>(pos);
        self.accel_from_legendre_t::<N, NP4>(&v, &w, max_order)
    }

    // Equations 7.65 to 7.69 in Montenbruck & Gill
    #[inline(always)]
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
        Matrix3::new([
            [daxdx, daxdy, daxdz],
            [daxdy, daydy, daydz],
            [daxdz, daydz, dazdz],
        ]) * self.gravity_constant
            / self.radius.powi(3)
    }

    /// See Equation 3.33 in Montenbruck & Gill
    #[inline(always)]
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

        numeris::vector![ax, ay, az] * self.gravity_constant / self.radius / self.radius
    }

    #[inline(always)]
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

    /// Load gravity-model coefficients from a file under [`datadir`] by
    /// basename. Auto-downloads via [`download_if_not_exist`] if missing.
    /// Files are at <http://icgem.gfz-potsdam.de/tom_longtime>.
    pub fn from_file(filename: &str) -> Result<Self> {
        // Precedence: a copy in the data directory wins (e.g. a full-degree
        // file installed by `update_datafiles`); otherwise the compiled-in
        // copy (degree 70, the evaluator's cap); a download
        // is only attempted for a name that is not embedded.
        if let Some(path) = crate::utils::find_data_file(filename) {
            return Self::from_path(&path);
        }
        if let Some(bytes) = crate::utils::embedded::get(filename) {
            return Self::from_bytes(&bytes);
        }
        let path = datadir()?.join(filename);
        download_if_not_exist(&path, None)?;
        Self::from_path(&path)
    }

    /// Load gravity-model coefficients from a file at `path`. No download
    /// is attempted — the file is expected to already exist.
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(Error::OpenFailed)?;
        Self::from_bytes(&bytes)
    }

    /// Load gravity-model coefficients from an in-memory byte buffer
    /// holding an ICGEM `.gfc` text file. Non-UTF-8 bytes (some model
    /// headers are Latin-1) are decoded lossily; only ASCII keywords and
    /// numbers are read.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(&String::from_utf8_lossy(bytes))
    }

    /// Parse gravity-model coefficients from an ICGEM `.gfc` text string.
    ///
    /// Reads the `modelname`, `earth_gravity_constant`, `radius`,
    /// `max_degree`, `norm` and `tide_system` header keywords, then the `gfc`
    /// coefficient rows (and `gfct` rows, the static part of an ICGEM 2.0
    /// time-variable model, taken at their reference epoch; the `trnd`,
    /// `asin`, `acos` and `dot` rows describing the time variation are
    /// skipped). Numbers may use Fortran `D` exponents (`1.0d0`). Any other
    /// row keyword after `end_of_head` is an [`Error::InvalidLine`].
    ///
    /// `norm` is `fully_normalized` (the default when absent) or
    /// `unnormalized`, whose coefficients are used as they are; any other
    /// value is an [`Error::InvalidLine`], as is a header without a positive
    /// `earth_gravity_constant` or `radius`. A `tide_system` value that is
    /// not recognised is treated like a missing one (classified from C̄20).
    pub fn parse(text: &str) -> Result<Self> {
        let mut name = String::new();
        let mut gravity_constant: f64 = 0.0;
        let mut radius: f64 = 0.0;
        let mut max_degree: usize = 0;
        let mut tide_system: Option<TideSystem> = None;
        let mut normalized = true;

        let mut lines = text.lines();

        // Read header lines
        for line in lines.by_ref() {
            let s: Vec<&str> = line.split_whitespace().collect();
            // Check for the header terminator before the two-token guard: the
            // ICGEM spec allows a bare "end_of_head" line (no ==== filler),
            // which would otherwise be skipped — silently treating the whole
            // file as header and yielding an all-zero model.
            if s.first() == Some(&"end_of_head") {
                break;
            }
            if s.len() < 2 {
                continue;
            }
            if s[0] == "modelname" {
                name = String::from(s[1]);
            } else if s[0] == "earth_gravity_constant" {
                gravity_constant = parse_f64(s[1])?;
            } else if s[0] == "radius" {
                radius = parse_f64(s[1])?;
            } else if s[0] == "max_degree" {
                max_degree = s[1].parse::<usize>()?;
            } else if s[0] == "tide_system" {
                tide_system = Some(TideSystem::from_header(&s[1..].join(" ")));
            } else if s[0] == "norm" {
                normalized = match s[1] {
                    "fully_normalized" => true,
                    "unnormalized" => false,
                    _ => return Err(Error::InvalidLine(line.to_string())),
                };
            } else if s[0] == "end_of_head" {
                break;
            }
        }
        if max_degree == 0 {
            return Err(Error::MissingMaxDegree);
        }
        // Without these the field is zero (GM) or NaN (radius) everywhere.
        for (value, keyword) in [
            (gravity_constant, "earth_gravity_constant"),
            (radius, "radius"),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidLine(format!(
                    "header {keyword} is missing or not a positive number ({value})"
                )));
            }
        }

        // Create matrix with lookup values. Cap the stored table at the
        // largest degree the evaluator can use (see `MAX_COEFF_DIM`); higher
        // coefficients in the file are unused and would only hurt cache
        // locality.
        let table_dim = (max_degree + 1).min(MAX_COEFF_DIM);
        let mut cs: CoeffTable = CoeffTable::zeros(table_dim, table_dim);

        for line in lines {
            let invalid = || Error::InvalidLine(line.to_string());
            // Need at least keyword, degree, order, and the C coefficient;
            // the S coefficient is required only when m > 0. The tokens are
            // read one at a time so the lines beyond the stored degree, most
            // of a full-resolution file, cost only two integer parses.
            let mut s = line.split_whitespace();
            match s.next() {
                // Blank line (e.g. a trailing newline).
                None => continue,
                // Static coefficients; `gfct` is the ICGEM 2.0 static part
                // of a time-variable coefficient, valid at its reference
                // epoch (the C and S columns are in the same positions).
                Some("gfc" | "gfct") => {}
                // ICGEM 2.0 time-variable terms: trend and periodic
                // components. Not modelled; skipping them yields the
                // field at the reference epoch.
                Some("trnd" | "asin" | "acos" | "dot") => continue,
                Some(_) => return Err(invalid()),
            }
            let n: usize = s.next().ok_or_else(invalid)?.parse()?;
            let m: usize = s.next().ok_or_else(invalid)?.parse()?;
            // The gfc format requires order <= degree; a violating line would
            // index outside the triangular layout below (panicking for large m,
            // silently aliasing another coefficient for moderate m).
            if m > n {
                return Err(invalid());
            }
            let c = s.next().ok_or_else(invalid)?;
            // Skip coefficients beyond the stored/evaluated degree.
            if n >= table_dim {
                continue;
            }
            cs[(n, m)] = parse_f64(c)?;
            if m > 0 {
                cs[(m - 1, n)] = parse_f64(s.next().ok_or_else(invalid)?)?;
            }
        }

        // Tide system: the header wins; a file without one (JGM2, JGM3), or
        // with a value that is not recognised, is classified from its
        // fully-normalized C̄20, read here before the denormalization below
        // (an unnormalized file's C20 is √5 C̄20). A header that contradicts
        // an Earth-like C̄20 is reported, since the propagator's tide
        // handling depends on it (silenced with SATKIT_QUIET=1).
        let c20 = match (table_dim > 2, normalized) {
            (false, _) => f64::NAN,
            (true, true) => cs[(2, 0)],
            (true, false) => cs[(2, 0)] / 5.0f64.sqrt(),
        };
        let by_value = TideSystem::classify_c20(c20);
        let tide_system = match tide_system {
            Some(declared) if declared != TideSystem::Unknown => {
                if by_value != TideSystem::Unknown
                    && declared != TideSystem::MeanTide
                    && declared != by_value
                    && std::env::var_os("SATKIT_QUIET").is_none()
                {
                    eprintln!(
                        "Warning: gravity model {name:?} declares tide_system {declared} \
                         but its C20 ({c20:e}) is a {by_value} value; using the header"
                    );
                }
                declared
            }
            _ => by_value,
        };

        // Convert from normalized coefficients to actual coefficients (an
        // unnormalized file already holds them)
        for n in (0..table_dim).filter(|_| normalized) {
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
        for m in 0..(MAX_COEFF_DIM - 1) {
            if m > 0 {
                d1[(m, m)] = (2 * m - 1) as f64
            }
            let n = m + 1;
            d1[(n, m)] = (2 * n - 1) as f64 / (n - m) as f64;
            for n in (m + 2)..(MAX_COEFF_DIM - 1) {
                d1[(n, m)] = (2 * n - 1) as f64 / (n - m) as f64;
                d2[(n, m)] = (n + m - 1) as f64 / (n - m) as f64;
            }
        }

        Ok(Self {
            name,
            gravity_constant,
            radius,
            max_degree,
            tide_system,
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

    const TINY_MODEL: &str = "\
modelname test
earth_gravity_constant 3.986004415e14
radius 6378136.3
max_degree 4
end_of_head
gfc 0 0 1.0 0.0
gfc 2 0 -4.841653e-4 0.0
gfc 2 2 2.439383e-6 -1.400273e-6
";

    #[test]
    fn tide_system_from_header_spellings() {
        for (s, want) in [
            ("tide_free", TideSystem::TideFree),
            ("zero_tide", TideSystem::ZeroTide),
            ("zero tide", TideSystem::ZeroTide), // ITU_GRACE16 spelling
            ("Zero-Tide", TideSystem::ZeroTide),
            ("mean_tide", TideSystem::MeanTide),
            ("unknown", TideSystem::Unknown),
            ("", TideSystem::Unknown),
        ] {
            assert_eq!(TideSystem::from_header(s), want, "{s:?}");
        }
        let with = |v: &str| {
            TINY_MODEL.replacen(
                "max_degree 4\n",
                &format!("max_degree 4\ntide_system {v}\n"),
                1,
            )
        };
        assert_eq!(
            Gravity::parse(&with("zero tide")).unwrap().tide_system,
            TideSystem::ZeroTide
        );
        assert_eq!(
            Gravity::parse(&with("tide_free")).unwrap().tide_system,
            TideSystem::TideFree
        );
        assert_eq!(
            Gravity::parse(&with("mean_tide")).unwrap().tide_system,
            TideSystem::MeanTide
        );
        assert!(TideSystem::MeanTide.includes_permanent_tide());
        assert!(TideSystem::ZeroTide.includes_permanent_tide());
        assert!(!TideSystem::TideFree.includes_permanent_tide());
        assert!(!TideSystem::Unknown.includes_permanent_tide());
    }

    #[test]
    fn tide_system_classified_from_c20_without_header() {
        // TINY_MODEL has no tide_system line and EGM96's C20 → tide-free.
        assert_eq!(
            Gravity::parse(TINY_MODEL).unwrap().tide_system,
            TideSystem::TideFree
        );
        // JGM3's C20 (4.2e-9 more negative) → zero-tide.
        let zt = TINY_MODEL.replace("-4.841653e-4", "-4.841695e-4");
        assert_eq!(
            Gravity::parse(&zt).unwrap().tide_system,
            TideSystem::ZeroTide
        );
        // A C20 that is not Earth's (a synthetic or non-Earth body) → unknown.
        let other = TINY_MODEL.replace("-4.841653e-4", "-2.0e-4");
        assert_eq!(
            Gravity::parse(&other).unwrap().tide_system,
            TideSystem::Unknown
        );
        // A model that stops below degree 2 has no C20 to classify.
        let deg1 = "modelname t\nearth_gravity_constant 3.986e14\nradius 6378136.3\nmax_degree 1\nend_of_head\ngfc 0 0 1.0 0.0\n";
        assert_eq!(
            Gravity::parse(deg1).unwrap().tide_system,
            TideSystem::Unknown
        );
        assert_eq!(
            TideSystem::classify_c20(-4.84165371736e-4),
            TideSystem::TideFree
        );
        assert_eq!(
            TideSystem::classify_c20(-4.84169548456e-4),
            TideSystem::ZeroTide
        );
    }

    /// A `tide_system` header value that is not recognised is treated like
    /// no header: the model is classified from C̄20 (it used to stay
    /// `Unknown`, skipping the permanent-tide handling of a zero-tide model).
    #[test]
    fn unrecognised_tide_header_falls_back_to_c20() {
        let with = |v: &str, c20: &str| {
            TINY_MODEL
                .replacen(
                    "max_degree 4\n",
                    &format!("max_degree 4\ntide_system {v}\n"),
                    1,
                )
                .replace("-4.841653e-4", c20)
        };
        for v in ["unknown", "conventional", "zero-ish"] {
            assert_eq!(
                Gravity::parse(&with(v, "-4.841653e-4"))
                    .unwrap()
                    .tide_system,
                TideSystem::TideFree,
                "{v}"
            );
            assert_eq!(
                Gravity::parse(&with(v, "-4.841695e-4"))
                    .unwrap()
                    .tide_system,
                TideSystem::ZeroTide,
                "{v}"
            );
            assert_eq!(
                Gravity::parse(&with(v, "-2.0e-4")).unwrap().tide_system,
                TideSystem::Unknown,
                "{v}"
            );
        }
    }

    /// An ICGEM `norm unnormalized` file holds the coefficients as used and
    /// is not de-normalized a second time (it used to be, scaling C22 by
    /// ~0.26); an unknown `norm` value is refused.
    #[test]
    fn unnormalized_header_is_honoured() {
        let g = Gravity::parse(TINY_MODEL).unwrap();
        let unnorm = format!(
            "modelname test\nearth_gravity_constant 3.986004415e14\nradius 6378136.3\n\
             max_degree 4\nnorm unnormalized\nend_of_head\n\
             gfc 0 0 {:e} 0.0\ngfc 2 0 {:e} 0.0\ngfc 2 2 {:e} {:e}\n",
            g.coeffs[(0, 0)],
            g.coeffs[(2, 0)],
            g.coeffs[(2, 2)],
            g.coeffs[(1, 2)],
        );
        let u = Gravity::parse(&unnorm).unwrap();
        for (n, m) in [(0, 0), (2, 0), (2, 2), (1, 2)] {
            assert_relative_eq!(u.coeffs[(n, m)], g.coeffs[(n, m)], max_relative = 1e-14);
        }
        // Classified from the normalized C̄20 it corresponds to.
        assert_eq!(u.tide_system, g.tide_system);
        let pos = numeris::vector![7.0e6, 1.0e6, 2.0e6];
        let (au, ag) = (u.accel(&pos, 4, 4), g.accel(&pos, 4, 4));
        assert!((au - ag).norm() / ag.norm() < 1e-13);

        let explicit =
            TINY_MODEL.replacen("max_degree 4\n", "max_degree 4\nnorm fully_normalized\n", 1);
        assert_eq!(Gravity::parse(&explicit).unwrap().coeffs, g.coeffs);
        let bogus = TINY_MODEL.replacen("max_degree 4\n", "max_degree 4\nnorm geodesy\n", 1);
        assert!(matches!(Gravity::parse(&bogus), Err(Error::InvalidLine(_))));
    }

    /// A header without a positive GM or radius is refused at parse time
    /// (a missing radius used to give NaN accelerations, a missing GM a
    /// zero field).
    #[test]
    fn missing_gm_or_radius_is_an_error() {
        for (from, to) in [
            ("earth_gravity_constant 3.986004415e14\n", ""),
            ("radius 6378136.3\n", ""),
            (
                "earth_gravity_constant 3.986004415e14",
                "earth_gravity_constant 0.0",
            ),
            ("radius 6378136.3", "radius -6378136.3"),
        ] {
            let txt = TINY_MODEL.replacen(from, to, 1);
            assert_ne!(txt, TINY_MODEL);
            assert!(
                matches!(Gravity::parse(&txt), Err(Error::InvalidLine(_))),
                "{from:?} -> {to:?}"
            );
        }
    }

    #[test]
    fn builtin_models_tide_systems() {
        // Header-declared: EGM96 / EGM2008 tide_free. Headerless, by value:
        // JGM2 tide-free, JGM3 zero-tide. (The forces guide once listed
        // JGM2 as zero-tide; its C20 says otherwise.)
        assert_eq!(egm96().tide_system, TideSystem::TideFree);
        assert_eq!(egm2008().tide_system, TideSystem::TideFree);
        assert_eq!(jgm2().tide_system, TideSystem::TideFree);
        assert_eq!(jgm3().tide_system, TideSystem::ZeroTide);
        // ITU_GRACE16 is downloaded on demand; only check it when available.
        if let Ok(g) = ensure_loaded(GravityModel::ITUGrace16) {
            assert_eq!(g.tide_system, TideSystem::ZeroTide);
            assert!(is_loaded(GravityModel::ITUGrace16));
        }
    }

    #[test]
    fn parse_fortran_d_exponents() {
        // EGM2008 writes its C00 row as `1.0d0`; GGM05 uses D in the header.
        let txt = TINY_MODEL
            .replace(
                "earth_gravity_constant 3.986004415e14",
                "earth_gravity_constant 0.3986004415D+15",
            )
            .replace("gfc 0 0 1.0 0.0", "gfc 0 0 1.0d0 0.0d0");
        let g = Gravity::parse(&txt).unwrap();
        assert_eq!(g.gravity_constant, 3.986004415e14);
        assert_eq!(g.coeffs[(0, 0)], 1.0);
        assert!(Gravity::parse(&TINY_MODEL.replace("gfc 0 0 1.0 0.0", "gfc 0 0 abc 0.0")).is_err());
    }

    #[test]
    fn parse_icgem2_time_variable_rows() {
        // `gfct` is the static part (read); trnd/asin/acos describe the
        // time variation (skipped); anything else is an error.
        let txt = format!(
            "{TINY_MODEL}gfct 3 0 9.5e-7 0.0 0.0 0.0 20050101.0000\n\
             trnd 3 0 1.0e-11 0.0 0.0 0.0\n\
             asin 3 0 2.0e-11 0.0 0.0 0.0 1.0\n\
             acos 3 0 3.0e-11 0.0 0.0 0.0 1.0\n\n"
        );
        let g = Gravity::parse(&txt).unwrap();
        let g0 = Gravity::parse(TINY_MODEL).unwrap();
        assert!(g.coeffs[(3, 0)] != 0.0, "gfct row read as a coefficient");
        assert_eq!(g.coeffs[(2, 0)], g0.coeffs[(2, 0)], "static rows untouched");
        let bad = format!("{TINY_MODEL}bogus 3 0 1.0 0.0\n");
        assert!(matches!(Gravity::parse(&bad), Err(Error::InvalidLine(_))));
    }

    #[test]
    fn from_bytes_tolerates_latin1_header() {
        // EIGEN-6C4's header carries a Latin-1 author name; only the ASCII
        // keywords matter.
        let mut bytes = b"modelname t\xe9st\n".to_vec();
        bytes.extend_from_slice(TINY_MODEL.as_bytes());
        assert!(Gravity::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn egm2008_agrees_with_egm96_at_low_degree() {
        // The two NGA models agree at the 1e-9 level in the low-degree
        // coefficients; at 400 km the degree-20 accelerations differ by
        // well under 1e-6 relative.
        let coord = ITRFCoord::from_geodetic_deg(35.0, -100.0, 400.0e3);
        let a96 = egm96().accel(&coord.itrf, 20, 20);
        let a08 = egm2008().accel(&coord.itrf, 20, 20);
        let rel = (a96 - a08).norm() / a96.norm();
        assert!(rel < 1.0e-6, "EGM96 vs EGM2008 relative difference {rel:e}");
        assert!(rel > 0.0, "the two models are not identical");
        assert_eq!(egm2008().name, "EGM2008");
        assert_eq!(GravityModel::EGM2008.to_string(), "EGM2008");
    }

    /// Textbook reference for the tests below: the potential
    /// V = μ/r Σₙ (R/r)ⁿ Σₘ Pₙₘ(sin φ)(Cₙₘ cos mλ + Sₙₘ sin mλ) with
    /// *unnormalised* Pₙₘ from the forward-column recursion and the stored
    /// (de-normalised) coefficients, differentiated by central differences.
    /// It shares nothing with the Cunningham V/W recursion the evaluator uses.
    fn reference_accel(
        g: &Gravity,
        pos: &Vector3,
        nmin: usize,
        nmax: usize,
        mmax: usize,
    ) -> Vector3 {
        let potential = |p: &Vector3| -> f64 {
            let r = p.norm();
            let t = p[2] / r;
            let u = (p[0] * p[0] + p[1] * p[1]).sqrt() / r;
            let lam = p[1].atan2(p[0]);
            let dim = nmax + 1;
            let mut pnm = vec![0.0_f64; dim * dim];
            pnm[0] = 1.0;
            for m in 1..=nmax {
                pnm[m * dim + m] = (2 * m - 1) as f64 * u * pnm[(m - 1) * dim + (m - 1)];
            }
            for m in 0..nmax {
                pnm[(m + 1) * dim + m] = (2 * m + 1) as f64 * t * pnm[m * dim + m];
                for n in (m + 2)..=nmax {
                    pnm[n * dim + m] = ((2 * n - 1) as f64 * t * pnm[(n - 1) * dim + m]
                        - (n + m - 1) as f64 * pnm[(n - 2) * dim + m])
                        / (n - m) as f64;
                }
            }
            let mut v = 0.0;
            for n in nmin..=nmax {
                let mut sum = 0.0;
                for m in 0..=n.min(mmax) {
                    let c = g.coeffs[(n, m)];
                    let s = if m > 0 { g.coeffs[(m - 1, n)] } else { 0.0 };
                    let ml = m as f64 * lam;
                    sum += pnm[n * dim + m] * (c * ml.cos() + s * ml.sin());
                }
                v += (g.radius / r).powi(n as i32) * sum;
            }
            g.gravity_constant / r * v
        };
        let h = 1.0;
        let mut a = Vector3::zeros();
        for i in 0..3 {
            let mut pp = *pos;
            pp[i] += h;
            let mut pm = *pos;
            pm[i] -= h;
            a[i] = (potential(&pp) - potential(&pm)) / (2.0 * h);
        }
        a
    }

    #[test]
    fn embedded_models_hold_degree_70() {
        // The compiled-in files hold every coefficient up to MAX_GRAVITY_DEGREE
        // (EGM96/EGM2008 truncated to it, JGM2/JGM3 natively degree 70).
        for g in [egm96(), egm2008(), jgm2(), jgm3()] {
            let n = MAX_GRAVITY_DEGREE as usize;
            assert!(
                g.coeffs.nrows() > n,
                "{}: {} rows",
                g.name,
                g.coeffs.nrows()
            );
            assert!(g.coeffs[(n, 0)] != 0.0, "{}: C(70,0) missing", g.name);
            assert!(g.coeffs[(n, n)] != 0.0, "{}: C(70,70) missing", g.name);
            assert!(g.coeffs[(n - 1, n)] != 0.0, "{}: S(70,70) missing", g.name);
        }
    }

    #[test]
    fn degree_70_matches_reference_potential() {
        // Degrees 41–70 were never exercised before the cap was raised (0.23);
        // check the full field, the new band alone and an order truncation
        // against the finite-difference reference at LEO, near the pole and
        // at GEO. The finite-difference floor is ~1e-9 relative.
        let pts = [
            ITRFCoord::from_geodetic_deg(35.0, -100.0, 400.0e3).itrf,
            ITRFCoord::from_geodetic_deg(89.0, 20.0, 400.0e3).itrf,
            ITRFCoord::from_geodetic_deg(0.0, 137.0, 400.0e3).itrf,
            ITRFCoord::from_geodetic_deg(-60.0, 45.0, 800.0e3).itrf,
        ];
        for g in [egm96(), egm2008(), jgm3()] {
            for pos in &pts {
                let a70 = g.accel(pos, 70, 70);
                let full = reference_accel(g, pos, 0, 70, 70);
                let rel = (a70 - full).norm() / full.norm();
                assert!(rel < 1.0e-7, "{} full field rel err {rel:e}", g.name);

                // Degrees 41–70 alone: a linear difference on both sides.
                let band = a70 - g.accel(pos, 40, 40);
                let band_ref = reference_accel(g, pos, 41, 70, 70);
                let rel = (band - band_ref).norm() / band_ref.norm();
                assert!(rel < 1.0e-6, "{} band 41–70 rel err {rel:e}", g.name);
                // ...and it is a real signal, not zeros from a short table.
                assert!(
                    band_ref.norm() > 1.0e-7 && band_ref.norm() < 1.0e-4,
                    "{} band 41–70 magnitude {:e}",
                    g.name,
                    band_ref.norm()
                );

                let a70o30 = g.accel(pos, 70, 30);
                let o30 = reference_accel(g, pos, 0, 70, 30);
                let rel = (a70o30 - o30).norm() / o30.norm();
                assert!(
                    rel < 1.0e-7,
                    "{} degree 70 / order 30 rel err {rel:e}",
                    g.name
                );

                let (a2, p2) = g.accel_and_partials(pos, 70, 70);
                assert_eq!(a2, a70);
                assert!(p2.as_slice().iter().all(|v| v.is_finite()));
            }
        }
        // GEO: the band is ~1e-40 m/s²; only the full field is meaningful.
        let geo = numeris::vector![42164.0e3 * 0.985, 42164.0e3 * 0.174, 0.0];
        let full = reference_accel(egm2008(), &geo, 0, 70, 70);
        let rel = (egm2008().accel(&geo, 70, 70) - full).norm() / full.norm();
        assert!(rel < 1.0e-7, "GEO rel err {rel:e}");
    }

    #[test]
    fn test_parse_rejects_order_above_degree() {
        // m > n would index outside the triangular coefficient layout
        // (panic for large m, silent aliasing for moderate m)
        let bad = format!("{}gfc 2 3 1.0 1.0\n", TINY_MODEL);
        assert!(matches!(Gravity::parse(&bad), Err(Error::InvalidLine(_))));
        // A large m used to panic on the flat matrix index
        let bad = format!("{}gfc 2 100 1.0 1.0\n", TINY_MODEL);
        assert!(Gravity::parse(&bad).is_err());
    }

    #[test]
    fn test_custom_model_degree_clamped() {
        // Requesting a degree beyond a custom low-degree model's table used
        // to index past the table (panic) or silently read S as C. It must
        // clamp to the stored degree instead.
        let g = Gravity::parse(TINY_MODEL).unwrap();
        let pos = numeris::vector![7000.0e3, 1000.0e3, 3000.0e3];
        let a_clamped = g.accel(&pos, 16, 16);
        let a_max = g.accel(&pos, 4, 4);
        assert_relative_eq!(a_clamped.norm(), a_max.norm(), max_relative = 1.0e-12);
        let (a2, p2) = g.accel_and_partials(&pos, 16, 16);
        assert!(a2.norm().is_finite());
        assert!(p2.as_slice().iter().all(|v| v.is_finite()));
        // Degree 0 no longer falls into the top dispatch arm
        let a0 = g.accel(&pos, 0, 0);
        assert!(a0.norm().is_finite());
    }

    #[test]
    fn test_gravity_order_1() {
        // Order 1 = point mass: accel should be μ/r², radially inward
        let r = 7000.0e3; // 7000 km
        let pos = numeris::vector![r, 0.0, 0.0];
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
        let pos = numeris::vector![7000.0e3, 1000.0e3, 3000.0e3];
        let a_jgm3 = jgm3().accel(&pos, 1, 1);
        let a_jgm2 = jgm2().accel(&pos, 1, 1);
        let a_egm96 = egm96().accel(&pos, 1, 1);
        let a_grace = itu_grace16().accel(&pos, 1, 1);
        // All should be very close (small differences due to different GM values)
        assert!((a_jgm3 - a_jgm2).norm() < 1.0e-6 * a_jgm3.norm().max(a_jgm2.norm()));
        assert!((a_jgm3 - a_egm96).norm() < 1.0e-6 * a_jgm3.norm().max(a_egm96.norm()));
        assert!((a_jgm3 - a_grace).norm() < 1.0e-6 * a_jgm3.norm().max(a_grace.norm()));
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
            numeris::vector![-2.3360599811572618, 6.8730769266931615, -6.616497962860285];
        assert!((gaccel - gaccel_truth).norm() < 1.0e-6 * gaccel.norm().max(gaccel_truth.norm()));
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
            numeris::vector![coord.itrf[0], coord.itrf[1], 0.0] * OMEGA_EARTH * OMEGA_EARTH;
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
            let dpos = numeris::vector![
                random::<f64>() * 100.0,
                random::<f64>() * 100.0,
                random::<f64>() * 100.0,
            ];

            // get acceleration and partials at coordinate
            let (accel1, partials) = g.accel_and_partials(&coord.itrf, 6, 6);

            // apply (small) random shift
            let v2 = coord.itrf + dpos;

            // get gravity accelaration at new coordinate
            let accel2 = g.accel(&v2, 6, 6);

            // Get what would be expected from partial derivative
            let accel3 = accel1 + partials * dpos;

            // show that they are approximately equal
            assert!((accel2 - accel3).norm() < 1.0e-4 * accel2.norm().max(accel3.norm()));
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
            numeris::vector![-2.3360599811572618, 6.8730769266931615, -6.616497962860285];
        assert!((gaccel - gaccel_truth).norm() < 1.0e-6 * gaccel.norm().max(gaccel_truth.norm()));
    }

    #[test]
    fn test_partials_with_order_less_than_degree() {
        // Verify partials are consistent when order < degree
        let g = Gravity::from_file("JGM3.gfc").unwrap();
        let coord = ITRFCoord::from_geodetic_deg(45.0, 30.0, 400.0e3);
        let dpos = numeris::vector![50.0, -30.0, 80.0];

        // Use degree=6, order=2
        let (accel1, partials) = g.accel_and_partials(&coord.itrf, 6, 2);
        let v2 = coord.itrf + dpos;
        let accel2 = g.accel(&v2, 6, 2);
        let accel3 = accel1 + partials * dpos;

        assert!((accel2 - accel3).norm() < 1.0e-4 * accel2.norm().max(accel3.norm()));
    }
}
