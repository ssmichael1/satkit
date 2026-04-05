//! Gauss-Jackson 8th-order fixed-step multistep integrator for 2nd-order ODEs.
//!
//! # Why this lives in satkit (not numeris)
//!
//! Unlike the general-purpose Runge-Kutta solvers in [`numeris::ode`], Gauss-
//! Jackson is an astrodynamics-specific algorithm. It was invented by Jackson
//! (1924) for celestial mechanics and refined through the 1960s-2000s
//! specifically for satellite orbit propagation. Every major space-surveillance
//! and astrodynamics code uses it — GMAT, STK, ODTK, SpecialK, the US Space
//! Surveillance Network — and essentially nobody outside astrodynamics does.
//! Other 2nd-order ODE domains have different preferences: molecular dynamics
//! uses velocity-Verlet, astrophysical N-body uses IAS15/symplectic methods,
//! and structural dynamics uses Newmark-β. Even SciPy doesn't ship it.
//!
//! Keeping it in satkit rather than numeris matches its actual scope: it's an
//! orbit-mechanics tool, the tuning advice is orbit-specific (LEO/GEO step
//! sizes), the tests and references are orbit-mechanics, and satkit users
//! expect to find orbit integrators here.
//!
//! # What it is
//!
//! A **multistep predictor-corrector method specialised for 2nd-order ODEs**
//! \\( \\ddot r = f(t, r, \\dot r) \\), combining:
//!
//! - **Gauss-Jackson** for position (a summed-form Störmer-Cowell method)
//! - **Summed-Adams** for velocity (a summed-form Adams-Bashforth-Moulton)
//!
//! Both run in lockstep sharing the same 9-point acceleration history, and
//! the force function is evaluated with the current \\((r, v)\\) — so
//! velocity-dependent forces such as atmospheric drag are handled naturally.
//!
//! # When to use Gauss-Jackson in satkit
//!
//! The efficiency advantage over high-order Runge-Kutta (e.g. [`RKV98`](numeris::ode::RKV98))
//! comes from reusing past function evaluations. For orbit propagation where
//! a single force evaluation can cost thousands of flops (high-degree gravity
//! fields, atmosphere models, third-body perturbations), Gauss-Jackson
//! typically uses **3-10× fewer force evaluations** than RKV98 for the same
//! accuracy.
//!
//! **Use Gauss-Jackson when:**
//! - Integrating over many orbital periods (days to months)
//! - Force evaluations are expensive (high-degree gravity, JB2008, SRP shadow)
//! - A fixed step is acceptable (well-conditioned orbits)
//!
//! **Prefer a satkit Runge-Kutta integrator ([`RKV98`](numeris::ode::RKV98),
//! [`RKV87`](numeris::ode::RKV87), etc.) when:**
//! - Force model has discontinuities (eclipse boundaries, impulsive maneuvers)
//! - Step size must adapt to local error (highly eccentric orbits near perigee)
//! - The integration interval is short (< ~20 steps — GJ8 needs ≥ 9 startup points)
//!
//! # Algorithm (Berry & Healy 2004)
//!
//! 1. **Startup.** Produce 9 equally-spaced points \\(r_{-4} \\dots r_{4}\\)
//!    around epoch using a single-step integrator (here: classical RK4).
//!    Then iterate "mid-corrector" formulas on the 8 non-epoch points until
//!    the accelerations converge — this refines each startup point to 8th-
//!    order accuracy.
//! 2. **PECE loop.** For each subsequent step:
//!    - **Predict** \\(r_{n+1}, v_{n+1}\\) from the 9 most recent accelerations
//!      using row \\(j=5\\) of the coefficient tables.
//!    - **Evaluate** \\(a_{n+1} = f(t_{n+1}, r_{n+1}, v_{n+1})\\).
//!    - **Correct** using row \\(j=4\\) with the newly-evaluated
//!      \\(a_{n+1}\\).
//!    - **Evaluate** once more for PECE (or iterate multiple corrector
//!      passes, configurable).
//!
//! The coefficient tables follow Berry & Healy's ordinate form and were
//! derived with exact-rational arithmetic from the paper's recurrences
//! (eqs 26, 31, 43, 47, 54, 59, 63, 67). Spot values were cross-checked
//! against paper Tables 3, 4, and 6.
//!
//! # Reference
//!
//! > Berry, M. M., and Healy, L. M., "Implementation of Gauss-Jackson
//! > Integration for Orbit Propagation", *J. Astronaut. Sci.*, Vol. 52,
//! > No. 3, July-September 2004, pp. 331-357.
//! > <https://drum.lib.umd.edu/handle/1903/2202>
//!
//! # Example (standalone use on a toy problem)
//!
//! ```
//! use satkit::orbitprop::ode::{GaussJackson8, GJSettings};
//! use numeris::Vector;
//!
//! // Harmonic oscillator: r'' = -r
//! let r0 = Vector::<f64, 1>::from_array([1.0]);
//! let v0 = Vector::<f64, 1>::from_array([0.0]);
//! let tau = 2.0 * std::f64::consts::PI;
//!
//! let settings = GJSettings {
//!     h: tau / 200.0,
//!     ..GJSettings::default()
//! };
//!
//! let sol = GaussJackson8::integrate(
//!     0.0, tau, &r0, &v0,
//!     |_t, r, _v| -*r, // acceleration
//!     &settings,
//! ).unwrap();
//!
//! assert!((sol.r[0] - 1.0).abs() < 1e-10);
//! assert!(sol.v[0].abs() < 1e-10);
//! ```
//!
//! For realistic orbit propagation, select this integrator via
//! [`PropSettings::integrator`](crate::orbitprop::PropSettings) with
//! [`Integrator::GaussJackson8`](crate::orbitprop::Integrator::GaussJackson8),
//! and set the fixed step size via
//! [`PropSettings::gj_step_seconds`](crate::orbitprop::PropSettings).

use numeris::{FloatScalar, Vector};
use numeris::ode::OdeError;

// ===== Coefficient tables =====
//
// The B_COEFFS (Summed-Adams) and A_COEFFS (Gauss-Jackson) arrays below were
// derived with exact-rational arithmetic from the recurrences in Berry &
// Healy (2004) §§ "Adams Method" through "Gauss-Jackson", and verified
// against paper Tables 3, 4, and 6. To regenerate, follow the derivation
// in the paper (eqs 26, 31, 43, 47, 54, 59, 63, 67) — see the module
// docstring for pointers.
//
// Rows are indexed by j in {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}:
//   j = -4..3  ↦ mid-corrector formulas for startup points (n = -4..3,
//                skipping n = 0 which is epoch and never corrected)
//   j =  4     ↦ corrector (used during normal stepping AND for the n = 4
//                startup point)
//   j =  5     ↦ predictor (used during normal stepping)
//
// Columns are indexed by k = -4..4 (array index 0..8), where k = +4
// corresponds to the most-recent acceleration in the 9-point stencil.

const B_COEFFS: [[f64; 9]; 10] = [
    [-2.869_754_464_285_714e-1,
     -5.890_197_861_552_028e-01,
     9.640_148_258_377_425e-01,
     -1.240_890_376_984_127e+00,
     1.140_564_373_897_707e+00,
     -7.209_438_381_834_215e-01,
     2.978_546_626_984_127e-01,
     -7.249_696_869_488_536e-02,
     7.892_554_012_345_68e-03], // j = -4
    [7.892_554_012_345_68e-03,
     -3.580_084_325_396_825_6e-01,
     -3.048_878_417_107_584e-01,
     3.010_402_888_007_055e-01,
     -2.464_285_714_285_714_4e-01,
     1.461_025_683_421_517e-01,
     -5.796_930_114_638_448e-02,
     1.372_271_825_396_825_4e-02,
     -1.463_982_583_774_250_5e-03], // j = -3
    [-1.463_982_583_774_250_5e-03,
     2.106_839_726_631_393_2e-02,
     -4.107_118_055_555_555_3e-01,
     -1.819_133_046_737_213_3e-01,
     1.165_784_832_451_499_1e-01,
     -6.196_676_587_301_587_5e-02,
     2.312_803_130_511_464e-2,
     -5.265_928_130_511_464e-03,
     5.468_75e-4], // j = -2
    [5.468_75e-4,
     -6.385_857_583_774_25e-3,
     4.075_589_726_631_393_5e-02,
     -4.566_493_055_555_555_5e-01,
     -1.130_070_546_737_213_4e-01,
     4.767_223_324_514_991e-02,
     -1.602_926_587_301_587_2e-02,
     3.440_531_305_114_638_6e-03,
     -3.440_531_305_114_638_7e-04], // j = -1
    [-3.440_531_305_114_638_7e-04,
     3.643_353_174_603_174_6e-03,
     -1.877_177_028_218_694_8e-02,
     6.965_636_022_927_689e-02,
     -5e-1,
     -6.965_636_022_927_689e-02,
     1.877_177_028_218_694_8e-02,
     -3.643_353_174_603_174_6e-03,
     3.440_531_305_114_638_7e-04], // j =  0
    [3.440_531_305_114_638_7e-04,
     -3.440_531_305_114_638_6e-03,
     1.602_926_587_301_587_2e-02,
     -4.767_223_324_514_991e-02,
     1.130_070_546_737_213_4e-01,
     -5.433_506_944_444_444e-1,
     -4.075_589_726_631_393_5e-02,
     6.385_857_583_774_25e-3,
     -5.468_75e-4], // j =  1
    [-5.468_75e-4,
     5.265_928_130_511_464e-03,
     -2.312_803_130_511_464e-2,
     6.196_676_587_301_587_5e-02,
     -1.165_784_832_451_499_1e-01,
     1.819_133_046_737_213_3e-01,
     -5.892_881_944_444_445e-1,
     -2.106_839_726_631_393_2e-02,
     1.463_982_583_774_250_5e-03], // j =  2
    [1.463_982_583_774_250_5e-03,
     -1.372_271_825_396_825_4e-02,
     5.796_930_114_638_448e-02,
     -1.461_025_683_421_517e-01,
     2.464_285_714_285_714_4e-01,
     -3.010_402_888_007_055e-01,
     3.048_878_417_107_584e-01,
     -6.419_915_674_603_175e-01,
     -7.892_554_012_345_68e-03], // j =  3
    [-7.892_554_012_345_68e-03,
     7.249_696_869_488_536e-02,
     -2.978_546_626_984_127e-01,
     7.209_438_381_834_215e-01,
     -1.140_564_373_897_707e+00,
     1.240_890_376_984_127e+00,
     -9.640_148_258_377_425e-01,
     5.890_197_861_552_028e-01,
     -7.130_245_535_714_286e-01], // j =  4
    [2.869_754_464_285_714e-1,
     -2.590_671_571_869_488_6e+00,
     1.040_361_304_012_345_6e+01,
     -2.440_379_216_269_841_2e+01,
     3.687_985_008_818_342_4e+01,
     -3.729_947_062_389_770_5e+01,
     2.534_682_787_698_412_5e+01,
     -1.129_513_089_726_631_3e+01,
     3.171_798_804_012_345_5e+00], // j =  5
];

const A_COEFFS: [[f64; 9]; 10] = [
    [6.107_264_986_171_236e-02,
     1.004_385_872_615_039_2e-01,
     -2.179_954_555_475_388_8e-01,
     3.026_027_386_964_887e-1,
     -2.871_720_052_709_636e-01,
     1.846_507_986_612_153_4e-01,
     -7.709_378_006_253_007e-2,
     1.889_758_197_049_863_6e-02,
     -2.067_782_237_053_070_5e-03], // j = -4
    [-2.067_782_237_053_070_5e-03,
     7.968_268_999_519_e-02,
     2.599_842_672_759_339_3e-02,
     -4.430_174_763_508_097e-02,
     4.206_217_682_780_183e-02,
     -2.663_144_340_227_673_4e-02,
     1.095_709_074_875_741_5e-02,
     -2.653_619_528_619_528_7e-03,
     2.875_418_370_210_037e-04], // j = -3
    [2.875_418_370_210_037e-04,
     -4.655_658_770_242_104e-03,
     9.003_419_612_794_612e-02,
     1.844_912_417_829_084_5e-03,
     -8.071_476_170_434_504e-03,
     5.831_905_363_155_364e-03,
     -2.477_929_092_512_426e-03,
     6.055_846_160_012_827e-04,
     -6.574_299_543_049_544e-05], // j = -2
    [-6.574_299_543_049_544e-05,
     8.792_287_958_954_625e-04,
     -7.022_406_605_739_94e-03,
     9.555_660_774_410_775e-02,
     -6.438_705_006_413_34e-03,
     2.121_412_538_079_205e-04,
     3.094_937_469_937_47e-04,
     -1.111_812_570_145_903_5e-04,
     1.389_765_712_682_379_3e-05], // j = -1
    [1.389_765_712_682_379_3e-05,
     -1.908_219_095_719_095_8e-04,
     1.379_544_452_461_119e-03,
     -8.189_809_804_393_138e-03,
     9.730_771_254_208_755e-02,
     -8.189_809_804_393_138e-03,
     1.379_544_452_461_119e-03,
     -1.908_219_095_719_095_8e-04,
     1.389_765_712_682_379_3e-05], // j =  0
    [1.389_765_712_682_379_3e-05,
     -1.111_812_570_145_903_5e-04,
     3.094_937_469_937_47e-04,
     2.121_412_538_079_205e-04,
     -6.438_705_006_413_34e-03,
     9.555_660_774_410_775e-02,
     -7.022_406_605_739_94e-03,
     8.792_287_958_954_625e-04,
     -6.574_299_543_049_544e-05], // j =  1
    [-6.574_299_543_049_544e-05,
     6.055_846_160_012_827e-04,
     -2.477_929_092_512_426e-03,
     5.831_905_363_155_364e-03,
     -8.071_476_170_434_504e-03,
     1.844_912_417_829_084_5e-03,
     9.003_419_612_794_612e-02,
     -4.655_658_770_242_104e-03,
     2.875_418_370_210_037e-04], // j =  2
    [2.875_418_370_210_037e-04,
     -2.653_619_528_619_528_7e-03,
     1.095_709_074_875_741_5e-02,
     -2.663_144_340_227_673_4e-02,
     4.206_217_682_780_183e-02,
     -4.430_174_763_508_097e-02,
     2.599_842_672_759_339_3e-02,
     7.968_268_999_519_e-02,
     -2.067_782_237_053_070_5e-03], // j =  3
    [-2.067_782_237_053_070_5e-03,
     1.889_758_197_049_863_6e-02,
     -7.709_378_006_253_007e-2,
     1.846_507_986_612_153_4e-01,
     -2.871_720_052_709_636e-01,
     3.026_027_386_964_887e-1,
     -2.179_954_555_475_388_8e-01,
     1.004_385_872_615_039_2e-01,
     6.107_264_986_171_236e-02], // j =  4
    [6.107_264_986_171_236e-02,
     -5.517_216_309_924_643e-01,
     2.217_512_976_992_144,
     -5.207_196_368_446_368e+00,
     7.879_804_681_236_973e+00,
     -7.982_325_887_846_721e+00,
     5.432_705_327_080_327e+00,
     -2.416_610_850_569_184e+00,
     6.500_924_360_169_151e-01], // j =  5
];

// ===== Public API =====

/// Settings for the Gauss-Jackson 8 integrator.
#[derive(Debug, Clone, Copy)]
pub struct GJSettings<T> {
    /// Step size. **Must be small enough that the truncation error at 8th
    /// order is acceptable** — typically 30-120 seconds for LEO, larger for
    /// higher orbits. No adaptive control is performed.
    pub h: T,
    /// Maximum PECE iterations per step after startup (default: 2). The
    /// classical PECE scheme uses exactly 1 correction (this parameter = 1);
    /// setting it higher allows PE(CE)^n iteration until the position and
    /// velocity converge.
    pub max_corrector_iters: usize,
    /// Relative tolerance for corrector convergence (default: 1e-13).
    pub corrector_tol: T,
    /// Maximum startup mid-corrector iterations (default: 12). Berry-Healy
    /// typically needs 3-6 iterations for LEO orbits.
    pub max_startup_iters: usize,
    /// Startup convergence tolerance on accelerations, relative (default: 1e-13).
    pub startup_tol: T,
    /// Maximum total steps before returning [`OdeError::MaxStepsExceeded`]
    /// (default: 1_000_000).
    pub max_steps: usize,
    /// Store per-step \\((t, r, v, a)\\) samples for later interpolation via
    /// [`GaussJackson8::interpolate`]. Adds ~100 bytes per step of memory.
    /// Default: `false`.
    pub dense_output: bool,
}

impl Default for GJSettings<f64> {
    fn default() -> Self {
        Self {
            h: 60.0,
            max_corrector_iters: 2,
            corrector_tol: 1.0e-13,
            max_startup_iters: 12,
            startup_tol: 1.0e-13,
            max_steps: 1_000_000,
            dense_output: false,
        }
    }
}

impl Default for GJSettings<f32> {
    fn default() -> Self {
        Self {
            h: 60.0,
            max_corrector_iters: 2,
            corrector_tol: 1.0e-5,
            max_startup_iters: 12,
            startup_tol: 1.0e-5,
            max_steps: 1_000_000,
            dense_output: false,
        }
    }
}

/// Stored per-step state for dense output / interpolation.
///
/// Populated when [`GJSettings::dense_output`] is `true`. Each array has one
/// entry per accepted step (including startup points that lie within the
/// requested integration interval). The `t`, `r`, `v`, `a` arrays are
/// parallel: index `i` gives time, position, velocity, and acceleration at
/// the `i`-th stored sample.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GJDenseOutput<T: FloatScalar, const D: usize> {
    /// Independent variable at each stored sample (ascending if forward
    /// integration, descending if backward).
    pub t: Vec<T>,
    /// Position at each stored sample.
    pub r: Vec<Vector<T, D>>,
    /// Velocity at each stored sample.
    pub v: Vec<Vector<T, D>>,
    /// Acceleration at each stored sample.
    pub a: Vec<Vector<T, D>>,
}

/// Result of a Gauss-Jackson 8 integration.
#[derive(Debug, Clone)]
pub struct GJSolution<T: FloatScalar, const D: usize> {
    /// Final independent variable value.
    pub t: T,
    /// Final position.
    pub r: Vector<T, D>,
    /// Final velocity.
    pub v: Vector<T, D>,
    /// Total acceleration evaluations.
    pub evals: usize,
    /// Total accepted steps (including the 8 startup points).
    pub steps: usize,
    /// Mid-corrector iterations used during startup.
    pub startup_iters: usize,
    /// Stored per-step samples for interpolation, if [`GJSettings::dense_output`]
    /// was set.
    pub dense: Option<GJDenseOutput<T, D>>,
}

/// 8th-order Gauss-Jackson predictor-corrector for 2nd-order ODEs.
///
/// See the [module documentation](self) for details.
pub struct GaussJackson8;

impl GaussJackson8 {
    /// Integrate a 2nd-order ODE \\( \\ddot r = f(t, r, \\dot r) \\) from
    /// `t0` to `tf` with initial state `(r0, v0)`.
    ///
    /// The force function `f` takes `(t, r, v)` and returns the
    /// acceleration. Velocity-dependent forces (drag, relativistic
    /// corrections) are fully supported.
    ///
    /// The step size is fixed (`settings.h`), chosen in the same direction
    /// as `tf - t0`. The actual last step is clipped so the solution is
    /// computed exactly at `tf`.
    pub fn integrate<T: FloatScalar, const D: usize>(
        t0: T,
        tf: T,
        r0: &Vector<T, D>,
        v0: &Vector<T, D>,
        mut f: impl FnMut(T, &Vector<T, D>, &Vector<T, D>) -> Vector<T, D>,
        settings: &GJSettings<T>,
    ) -> Result<GJSolution<T, D>, OdeError> {
        let zero = T::zero();
        let one = T::one();

        // Step direction: +1 or -1. We allow backward integration by flipping
        // the sign of h if tf < t0.
        let dir = if tf >= t0 { one } else { -one };
        let h = settings.h.abs() * dir;
        if h == zero {
            return Err(OdeError::StepNotFinite);
        }

        // If the total interval is smaller than 8 steps, Gauss-Jackson can't
        // start up. Caller should use an RK method for such short intervals.
        let total_span = tf - t0;
        if (total_span / h).to_f64().unwrap_or(0.0).abs() < 8.0 {
            return Err(OdeError::StepNotFinite); // Interval too short for GJ8
        }

        let mut nevals: usize = 0;

        // ---- Startup: compute 9 points r[0..9] at times t0..t0+8h ----
        //
        // The paper describes a symmetric ±4h startup around epoch, but since
        // we're starting at t0 and integrating forward, we instead use a
        // one-sided startup: r[0] = initial state at t0, r[1..8] computed by
        // RK4, then refined via mid-correctors. This uses the same coefficient
        // tables indexed by j shifted: startup point i (i = 0..8) uses
        // mid-corrector row j = i - 4 (so i=0 uses j=-4, ..., i=4 uses j=0
        // = skipped, ..., i=8 uses j=4 = corrector).
        //
        // j = 0 corresponds to "epoch in the paper's symmetric convention".
        // In our one-sided scheme we never correct r[4] either (it's the
        // "epoch" of the symmetric stencil).
        //
        // WAIT. That's wrong — in a one-sided startup, there is no "epoch in
        // the middle". We need a different approach.
        //
        // Actually, the paper's symmetric startup is an implementation detail
        // of f&g series initialization — f&g gives symmetric accuracy. For an
        // RK-based startup, the one-sided approach works IF we never "skip"
        // any point. So we use all 9 rows j = -4..4 to correct all 9 startup
        // points, including r[0] (which we DO correct, since our initial RK4
        // guesses are only 4th-order accurate).
        //
        // But we can't correct r[0] using its own row and not shift anything,
        // because the initial conditions at r[0] are exact. The refinement
        // should leave r[0] unchanged.
        //
        // Resolution: we shift the indexing so r[4] is epoch (exact initial
        // conditions). We use RK4 to integrate BACKWARD 4 steps from r[4] to
        // get r[0..4], and FORWARD 4 steps to get r[5..9]. Then we iterate
        // mid-correctors on all 8 non-epoch points. After convergence, we
        // discard r[0..4] (before-epoch) and proceed forward from r[4] = t0.
        let mut r: [Vector<T, D>; 9] = [Vector::<T, D>::zeros(); 9];
        let mut v: [Vector<T, D>; 9] = [Vector::<T, D>::zeros(); 9];
        let mut a: [Vector<T, D>; 9] = [Vector::<T, D>::zeros(); 9];

        // Epoch at index 4
        r[4] = *r0;
        v[4] = *v0;
        a[4] = f(t0, &r[4], &v[4]);
        nevals += 1;

        // Forward 4 steps from epoch using RK4 (indices 5, 6, 7, 8)
        for i in 5..9 {
            // Integrating from time index (i-1) to (i) — relative to epoch.
            let offset_prev = (i as f64) - 1.0 - 4.0;
            let tn = t0 + T::from(offset_prev).unwrap() * h;
            let (rn1, vn1, ne) = rk4_step(tn, &r[i - 1], &v[i - 1], h, &mut f);
            r[i] = rn1;
            v[i] = vn1;
            nevals += ne;
        }
        // Backward 4 steps from epoch using RK4 with step -h (indices 3, 2, 1, 0)
        for i in (0..4).rev() {
            // Integrating from time index (i+1) to (i) — relative to epoch.
            let offset_prev = (i as f64) + 1.0 - 4.0;
            let tn = t0 + T::from(offset_prev).unwrap() * h;
            let (rn1, vn1, ne) = rk4_step(tn, &r[i + 1], &v[i + 1], -h, &mut f);
            r[i] = rn1;
            v[i] = vn1;
            nevals += ne;
        }
        // Evaluate accelerations at all 9 startup points
        for i in 0..9 {
            if i == 4 {
                continue;
            }
            let offset = (i as f64) - 4.0;
            let ti = t0 + T::from(offset).unwrap() * h;
            a[i] = f(ti, &r[i], &v[i]);
            nevals += 1;
        }

        // Mid-corrector iteration (SECECE...CE in the paper).
        //
        // We iterate: for each non-epoch point i (i = 0..9, i != 4), apply
        // row j = i - 4 to correct (r[i], v[i]), then re-evaluate a[i].
        // Continue until all accelerations converge.
        let mut startup_iters = 0usize;
        for _iter in 0..settings.max_startup_iters {
            startup_iters += 1;
            let a_prev = a;

            // Compute s (first sum) and S (second sum) for all startup points
            // in terms of the current a array. Equations (75) and (86) with
            // integration constants C1', C2 determined so that the formulas
            // reproduce the epoch (r[4], v[4]) via row j=0.
            //
            // From the j=0 row applied at index i=4:
            //   v[4] = h * (s[4] + sum_k B_COEFFS[4][k+4] * a[4+k-4])
            //        = h * (s[4] + sum_k B_COEFFS[4][k+4] * a[k])  (k = -4..4)
            // Row index 4 in our j enumeration is j=0 (since rows 0..9 map
            // to j = -4..5). So we solve for s[4] = C1'/h, then reconstruct
            // other s values.
            //
            // Similarly:
            //   r[4] = h^2 * (S[4] + sum_k A_COEFFS[4][k+4] * a[k])
            // gives S[4].
            //
            // Then s[i] and S[i] for other i are computed by the recurrences
            // in eqs (75) and (86).
            let s4 = {
                // v[4] / h - sum_k B_COEFFS[4][k] * a[k]
                let mut acc = Vector::<T, D>::zeros();
                for k in 0..9 {
                    let bk = T::from(B_COEFFS[4][k]).unwrap();
                    if bk != zero {
                        acc += a[k] * bk;
                    }
                }
                v[4] / h - acc
            };
            let big_s4 = {
                // r[4] / h^2 - sum_k A_COEFFS[4][k] * a[k]
                let mut acc = Vector::<T, D>::zeros();
                for k in 0..9 {
                    let ak = T::from(A_COEFFS[4][k]).unwrap();
                    if ak != zero {
                        acc += a[k] * ak;
                    }
                }
                r[4] / (h * h) - acc
            };

            // Recurrences for the "new-convention" running sums:
            //   s_n = nabla^-1 r̈_n   (no -r̈_n/2 pull-out)
            //     forward:  s[i+1] = s[i] + a[i+1]
            //     backward: s[i]   = s[i+1] - a[i+1]
            //   S_n = nabla^-2 r̈_{n-1}
            //     forward:  S[i+1] = S[i] + s[i]
            //     backward: S[i]   = S[i+1] - s[i+1] + a[i+1]
            //
            // These are algebraically equivalent to Berry-Healy's eqs (75,
            // 86) but match our "normal-form" ordinate coefficients (where
            // the -r̈_n/2 contribution is kept in the sum rather than
            // pulled out).
            let mut s: [Vector<T, D>; 9] = [Vector::<T, D>::zeros(); 9];
            let mut big_s: [Vector<T, D>; 9] = [Vector::<T, D>::zeros(); 9];
            s[4] = s4;
            big_s[4] = big_s4;
            for i in 5..9 {
                big_s[i] = big_s[i - 1] + s[i - 1];
                s[i] = s[i - 1] + a[i];
            }
            for i in (0..4).rev() {
                s[i] = s[i + 1] - a[i + 1];
                big_s[i] = big_s[i + 1] - s[i + 1] + a[i + 1];
            }

            // Apply mid-correctors for each non-epoch point
            for i in 0..9 {
                if i == 4 {
                    continue;
                }
                // Row j = i - 4 (but j ranges -4..4, so row index = i)
                let row_b = &B_COEFFS[i];
                let row_a = &A_COEFFS[i];
                let mut sum_b = Vector::<T, D>::zeros();
                let mut sum_a = Vector::<T, D>::zeros();
                for k in 0..9 {
                    let bk = T::from(row_b[k]).unwrap();
                    let ak = T::from(row_a[k]).unwrap();
                    sum_b += a[k] * bk;
                    sum_a += a[k] * ak;
                }
                v[i] = (s[i] + sum_b) * h;
                r[i] = (big_s[i] + sum_a) * (h * h);
            }

            // Re-evaluate accelerations
            for i in 0..9 {
                if i == 4 {
                    continue;
                }
                let offset = (i as f64) - 4.0;
                let ti = t0 + T::from(offset).unwrap() * h;
                a[i] = f(ti, &r[i], &v[i]);
                nevals += 1;
            }

            // Convergence check: max absolute change in accelerations,
            // normalized.
            let mut max_delta = zero;
            let mut max_mag = T::from(1.0e-30).unwrap();
            for i in 0..9 {
                let delta = (a[i] - a_prev[i]).abs().scaled_norm();
                let mag = a[i].abs().scaled_norm();
                if delta > max_delta {
                    max_delta = delta;
                }
                if mag > max_mag {
                    max_mag = mag;
                }
            }
            if max_delta / max_mag < settings.startup_tol {
                break;
            }
        }

        // ---- Main PECE loop ----
        //
        // After startup convergence, the stencil r[0..9]/v[0..9]/a[0..9]
        // corresponds to times t0 + (i-4)*h for i = 0..9. The "current"
        // accepted point is r[8] at time t0 + 4h.
        //
        // Wait — that's wrong. The user asked for integration from t0 to tf
        // starting at r0 = r[4]. The startup gives us valid state at indices
        // 0..9, covering times t0 - 4h to t0 + 4h. We want to continue
        // forward from r[4] (= t0), using r[0..8] as the "past" for the
        // multistep formula... no, we already HAVE points beyond t0 (up to
        // t0 + 4h). So we should shift indexing: r[4] is our current time,
        // and r[5..9] are "future" points already computed by startup, each
        // accurate to 8th order.
        //
        // The cleanest way to proceed: treat r[8] (t0 + 4h) as the "current"
        // position after startup. We've already computed steps 1..4 (at
        // t0+h..t0+4h) as part of startup. Then we predict step 5 (t0+5h)
        // using the window r[0..9], slide, predict step 6, etc.
        //
        // Current window after startup: indices 0..9 correspond to times
        // t0 - 4h to t0 + 4h. "Latest" point = index 8 = time t0 + 4h.
        let mut t_current = t0 + T::from(4.0).unwrap() * h;

        // Running sums s and big_s at the epoch (index 4), derived from the
        // corrector-row identity (eqs 74, 87 at n = 0 in paper indexing).
        let mut s_cur: Vector<T, D> = {
            let mut acc = Vector::<T, D>::zeros();
            for k in 0..9 {
                let bk = T::from(B_COEFFS[4][k]).unwrap();
                if bk != zero {
                    acc += a[k] * bk;
                }
            }
            v[4] / h - acc
        };
        let mut big_s_cur: Vector<T, D> = {
            let mut acc = Vector::<T, D>::zeros();
            for k in 0..9 {
                let ak = T::from(A_COEFFS[4][k]).unwrap();
                if ak != zero {
                    acc += a[k] * ak;
                }
            }
            r[4] / (h * h) - acc
        };
        // Walk forward from index 4 to index 8. New-convention recurrences:
        //   big_s[i+1] = big_s[i] + s[i]
        //   s[i+1]     = s[i] + a[i+1]
        // Update big_s first since it uses the OLD s.
        for i in 4..8 {
            big_s_cur += s_cur;
            s_cur += a[i + 1];
        }

        // Initialise dense-output storage with the 5 forward startup samples
        // (epoch + 4 forward steps) if requested. Backward-integrated startup
        // points at t < t0 are outside the requested interval and are not
        // stored.
        let mut dense_store: Option<GJDenseOutput<T, D>> = if settings.dense_output {
            let mut d = GJDenseOutput {
                t: Vec::new(),
                r: Vec::new(),
                v: Vec::new(),
                a: Vec::new(),
            };
            for i in 4..9 {
                let offset = (i as f64) - 4.0;
                d.t.push(t0 + T::from(offset).unwrap() * h);
                d.r.push(r[i]);
                d.v.push(v[i]);
                d.a.push(a[i]);
            }
            Some(d)
        } else {
            None
        };

        // Number of completed steps (startup gave us 4 forward steps)
        let mut nsteps: usize = 4;

        while (tf - t_current) * dir > zero {
            // Guard against overrunning total step limit
            if nsteps >= settings.max_steps {
                return Err(OdeError::MaxStepsExceeded);
            }

            // Clamp the final step so we land on tf. We use a simple rule: if
            // the remaining distance is less than h (in magnitude), we fall
            // back to an RK4 substep for the final interval. This loses a
            // bit of accuracy on the last point but avoids implementing a
            // variable-step tail which is beyond the scope of v1.
            let remaining = tf - t_current;
            if (remaining.abs() - h.abs()).to_f64().unwrap_or(0.0) < -1.0e-12 * h.abs().to_f64().unwrap_or(1.0) {
                // Final short step via RK4
                let (rn, vn, ne) = rk4_step(t_current, &r[8], &v[8], remaining, &mut f);
                // Append the final sample if we're storing dense output.
                if let Some(ref mut d) = dense_store {
                    let a_final = f(tf, &rn, &vn);
                    d.t.push(tf);
                    d.r.push(rn);
                    d.v.push(vn);
                    d.a.push(a_final);
                    return Ok(GJSolution {
                        t: tf,
                        r: rn,
                        v: vn,
                        evals: nevals + ne + 1,
                        steps: nsteps + 1,
                        startup_iters,
                        dense: dense_store,
                    });
                }
                return Ok(GJSolution {
                    t: tf,
                    r: rn,
                    v: vn,
                    evals: nevals + ne,
                    steps: nsteps + 1,
                    startup_iters,
                    dense: None,
                });
            }

            // === Predict ===
            // In the new-convention form, the predictor formulas become:
            //   v_{n+1} = h (s_n + Σ B_COEFFS[9][k] · a[k])
            //   r_{n+1} = h² (S_{n+1} + Σ A_COEFFS[9][k] · a[k])
            //           = h² (S_n + s_n + Σ A_COEFFS[9][k] · a[k])
            // Array row 9 holds the j=5 predictor; array row 8 holds the
            // j=4 corrector. (Array index i ↦ j = i − 4.)
            let mut sum_b_pred = Vector::<T, D>::zeros();
            let mut sum_a_pred = Vector::<T, D>::zeros();
            for k in 0..9 {
                let bk = T::from(B_COEFFS[9][k]).unwrap();
                let ak_coef = T::from(A_COEFFS[9][k]).unwrap();
                sum_b_pred += a[k] * bk;
                sum_a_pred += a[k] * ak_coef;
            }
            let big_s_new = big_s_cur + s_cur;
            let mut r_new = (big_s_new + sum_a_pred) * (h * h);
            let mut v_new = (s_cur + sum_b_pred) * h;

            // === Evaluate ===
            let t_new = t_current + h;
            let mut a_new = f(t_new, &r_new, &v_new);
            nevals += 1;

            // === Correct (PECE, possibly multiple iterations) ===
            // The corrector uses row j=4. After sliding the window (dropping
            // a[0] and appending a_new as the 9th element), the reference
            // point for the corrector is the new point, so we apply row 4
            // with k = -4..4 mapping to [a[1], a[2], ..., a[8], a_new].
            //
            // For efficiency, we pre-compute the sum over a[1..9] (which
            // doesn't change during corrector iteration) and only vary the
            // a_new contribution.
            let mut sum_b_corr_fixed = Vector::<T, D>::zeros();
            let mut sum_a_corr_fixed = Vector::<T, D>::zeros();
            for k in 0..8 {
                // post-slide window index k corresponds to pre-slide index k+1
                let bk = T::from(B_COEFFS[8][k]).unwrap();
                let ak_coef = T::from(A_COEFFS[8][k]).unwrap();
                sum_b_corr_fixed += a[k + 1] * bk;
                sum_a_corr_fixed += a[k + 1] * ak_coef;
            }
            let b8 = T::from(B_COEFFS[8][8]).unwrap();
            let a8 = T::from(A_COEFFS[8][8]).unwrap();

            // The s_new and big_s_new values are updated once after we commit
            // the final corrected a_new. For the corrector formula (eqs 79,
            // 89), we need the "post-slide" s and big_s values. Since the
            // running sums are additive, post-slide s_new = s_cur + (a[8]+a_new)/2
            // and post-slide big_s_new = big_s_new (already computed above —
            // same quantity regardless of a_new because it only uses a[8]
            // from the old window).

            for _corr_iter in 0..settings.max_corrector_iters {
                // New-convention: s_{n+1} = s_n + a_{n+1}
                let s_new_post = s_cur + a_new;
                let r_cand = (big_s_new + sum_a_corr_fixed + a_new * a8) * (h * h);
                let v_cand = (s_new_post + sum_b_corr_fixed + a_new * b8) * h;

                let dr = (r_cand - r_new).abs().scaled_norm();
                let dv = (v_cand - v_new).abs().scaled_norm();
                let rmag = r_cand.abs().scaled_norm() + T::from(1.0e-30).unwrap();
                let vmag = v_cand.abs().scaled_norm() + T::from(1.0e-30).unwrap();

                r_new = r_cand;
                v_new = v_cand;

                if dr / rmag < settings.corrector_tol
                    && dv / vmag < settings.corrector_tol
                {
                    break;
                }

                // Re-evaluate for next iteration
                a_new = f(t_new, &r_new, &v_new);
                nevals += 1;
            }

            // Commit the step: slide the window
            for i in 0..8 {
                r[i] = r[i + 1];
                v[i] = v[i + 1];
                a[i] = a[i + 1];
            }
            r[8] = r_new;
            v[8] = v_new;
            a[8] = a_new;

            // Update s_cur and big_s_cur to correspond to the new latest
            // point. New-convention: s_new = s_old + a_new, big_s_new was
            // already computed as big_s_old + s_old. After slide, the new
            // latest a_new is at a[8]. (a[7] was the old a[8].)
            s_cur += a[8];
            big_s_cur = big_s_new;

            t_current = t_new;
            nsteps += 1;

            // Record this step in dense output if enabled
            if let Some(ref mut d) = dense_store {
                d.t.push(t_current);
                d.r.push(r[8]);
                d.v.push(v[8]);
                d.a.push(a[8]);
            }
        }

        Ok(GJSolution {
            t: t_current,
            r: r[8],
            v: v[8],
            evals: nevals,
            steps: nsteps,
            startup_iters,
            dense: dense_store,
        })
    }

    /// Interpolate the solution at time `t_interp` using quintic Hermite
    /// interpolation on the two bracketing stored samples.
    ///
    /// Requires that the integration was run with
    /// [`GJSettings::dense_output`] set to `true`. Returns `(r, v)` at the
    /// requested time. The interpolation is 5th-order accurate — lower than
    /// GJ8's native 8th order, but matches the approach described in
    /// Berry & Healy (2004) §"Effects of Step Size". For typical orbit-prop
    /// step sizes this is well below the integration error budget.
    pub fn interpolate<T: FloatScalar, const D: usize>(
        t_interp: T,
        sol: &GJSolution<T, D>,
    ) -> Result<(Vector<T, D>, Vector<T, D>), OdeError> {
        let dense = sol.dense.as_ref().ok_or(OdeError::NoDenseOutput)?;
        if dense.t.len() < 2 {
            return Err(OdeError::NoDenseOutput);
        }

        // Determine direction and bounds
        let forward = dense.t[dense.t.len() - 1] > dense.t[0];
        let t_first = dense.t[0];
        let t_last = dense.t[dense.t.len() - 1];
        let (lo, hi) = if forward {
            (t_first, t_last)
        } else {
            (t_last, t_first)
        };
        if t_interp < lo || t_interp > hi {
            return Err(OdeError::InterpOutOfBounds);
        }

        // Binary search for the bracketing interval.
        let idx = if forward {
            let i = dense.t.partition_point(|&x| x <= t_interp);
            i.saturating_sub(1)
        } else {
            let i = dense.t.partition_point(|&x| x >= t_interp);
            i.saturating_sub(1)
        };
        let idx = idx.min(dense.t.len() - 2);

        Ok(quintic_hermite(
            dense.t[idx], &dense.r[idx], &dense.v[idx], &dense.a[idx],
            dense.t[idx + 1], &dense.r[idx + 1], &dense.v[idx + 1], &dense.a[idx + 1],
            t_interp,
        ))
    }

    /// Interpolate at multiple points in one pass. `t_interps` must be sorted
    /// in the same direction as the integration.
    #[allow(clippy::type_complexity)] // `Vec<(r, v)>` return is self-explanatory
    pub fn interpolate_batch<T: FloatScalar, const D: usize>(
        t_interps: &[T],
        sol: &GJSolution<T, D>,
    ) -> Result<Vec<(Vector<T, D>, Vector<T, D>)>, OdeError> {
        let dense = sol.dense.as_ref().ok_or(OdeError::NoDenseOutput)?;
        if dense.t.len() < 2 {
            return Err(OdeError::NoDenseOutput);
        }

        let forward = dense.t[dense.t.len() - 1] > dense.t[0];
        let t_first = dense.t[0];
        let t_last = dense.t[dense.t.len() - 1];
        let (lo, hi) = if forward {
            (t_first, t_last)
        } else {
            (t_last, t_first)
        };

        let mut results = Vec::with_capacity(t_interps.len());
        let mut idx = 0usize;
        let last_idx = dense.t.len() - 2;

        for &t_interp in t_interps {
            if t_interp < lo || t_interp > hi {
                return Err(OdeError::InterpOutOfBounds);
            }
            // Walk idx forward while the next interval still brackets t_interp
            if forward {
                while idx < last_idx && dense.t[idx + 1] < t_interp {
                    idx += 1;
                }
            } else {
                while idx < last_idx && dense.t[idx + 1] > t_interp {
                    idx += 1;
                }
            }

            results.push(quintic_hermite(
                dense.t[idx], &dense.r[idx], &dense.v[idx], &dense.a[idx],
                dense.t[idx + 1], &dense.r[idx + 1], &dense.v[idx + 1], &dense.a[idx + 1],
                t_interp,
            ));
        }

        Ok(results)
    }
}

/// Quintic Hermite interpolation of a 2nd-order ODE solution.
///
/// Given two bracketing samples `(t_a, r_a, v_a, a_a)` and `(t_b, r_b, v_b, a_b)`
/// of a smooth trajectory, returns `(r(t), v(t))` at an arbitrary `t` in the
/// interval `[t_a, t_b]` (or reversed for backward integration).
///
/// Matches six constraints (position, velocity, and acceleration at each
/// endpoint), producing a unique degree-5 polynomial. The interpolation is
/// 5th-order accurate in the step size.
#[allow(clippy::too_many_arguments)] // (t, r, v, a) at both endpoints + t_query; a helper struct would only obscure this
fn quintic_hermite<T, const D: usize>(
    t_a: T, r_a: &Vector<T, D>, v_a: &Vector<T, D>, acc_a: &Vector<T, D>,
    t_b: T, r_b: &Vector<T, D>, v_b: &Vector<T, D>, acc_b: &Vector<T, D>,
    t: T,
) -> (Vector<T, D>, Vector<T, D>)
where
    T: FloatScalar,
{
    let h = t_b - t_a;
    let s = (t - t_a) / h;
    let half = T::from(0.5).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();
    let five = T::from(5.0).unwrap();
    let six = T::from(6.0).unwrap();
    let seven = T::from(7.0).unwrap();
    let ten = T::from(10.0).unwrap();
    let fifteen = T::from(15.0).unwrap();

    // Quintic p(s) = c0 + c1 s + c2 s² + c3 s³ + c4 s⁴ + c5 s⁵ with
    //   p(0)   = r_a          ⇒ c0 = r_a
    //   p'(0)  = h v_a        ⇒ c1 = h v_a
    //   p''(0) = h² a_a       ⇒ c2 = h² a_a / 2
    //   p(1), p'(1), p''(1) = r_b, h v_b, h² a_b   ⇒ solve for c3, c4, c5
    //
    // Let Δ1 = r_b - r_a - h v_a - h² a_a / 2
    //     Δ2 = h (v_b - v_a) - h² a_a
    //     Δ3 = h² (a_b - a_a)
    // Then  c3 = 10 Δ1 − 4 Δ2 + Δ3/2
    //       c4 = −15 Δ1 + 7 Δ2 − Δ3
    //       c5 =  6 Δ1 − 3 Δ2 + Δ3/2
    let hh = h * h;
    let d1 = *r_b - *r_a - *v_a * h - *acc_a * (hh * half);
    let d2 = (*v_b - *v_a) * h - *acc_a * hh;
    let d3 = (*acc_b - *acc_a) * hh;

    let c3 = d1 * ten - d2 * four + d3 * half;
    let c4 = d1 * (-fifteen) + d2 * seven - d3;
    let c5 = d1 * six - d2 * three + d3 * half;

    // Position
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let s5 = s4 * s;
    let r = *r_a + *v_a * (h * s) + *acc_a * (hh * half * s2) + c3 * s3 + c4 * s4 + c5 * s5;

    // Velocity (dp/dt = dp/ds / h)
    // dp/ds = h v_a + h² a_a s + 3 c3 s² + 4 c4 s³ + 5 c5 s⁴
    // v(t) = v_a + h a_a s + 3 c3 s² / h + 4 c4 s³ / h + 5 c5 s⁴ / h
    let v = *v_a
        + *acc_a * (h * s)
        + c3 * (three * s2 / h)
        + c4 * (four * s3 / h)
        + c5 * (five * s4 / h);

    (r, v)
}

/// Single RK4 step for a 2nd-order ODE r'' = f(t, r, v).
///
/// Returns `(r_new, v_new, num_evals)`. Used internally for startup
/// bootstrap only; the multistep algorithm takes over after.
fn rk4_step<T, const D: usize, F>(
    t: T,
    r: &Vector<T, D>,
    v: &Vector<T, D>,
    h: T,
    f: &mut F,
) -> (Vector<T, D>, Vector<T, D>, usize)
where
    T: FloatScalar,
    F: FnMut(T, &Vector<T, D>, &Vector<T, D>) -> Vector<T, D>,
{
    let half = T::from(0.5).unwrap();
    let sixth = T::from(1.0 / 6.0).unwrap();
    let two = T::from(2.0).unwrap();

    // k1 = v, l1 = f(t, r, v)
    let k1 = *v;
    let l1 = f(t, r, v);

    // k2 = v + h/2 * l1, l2 = f(t+h/2, r + h/2*k1, v + h/2*l1)
    let k2 = *v + l1 * (h * half);
    let l2 = f(t + h * half, &(*r + k1 * (h * half)), &k2);

    // k3 = v + h/2 * l2, l3 = f(t+h/2, r + h/2*k2, v + h/2*l2)
    let k3 = *v + l2 * (h * half);
    let l3 = f(t + h * half, &(*r + k2 * (h * half)), &k3);

    // k4 = v + h * l3, l4 = f(t+h, r + h*k3, v + h*l3)
    let k4 = *v + l3 * h;
    let l4 = f(t + h, &(*r + k3 * h), &k4);

    let r_new = *r + (k1 + k2 * two + k3 * two + k4) * (h * sixth);
    let v_new = *v + (l1 + l2 * two + l3 * two + l4) * (h * sixth);

    (r_new, v_new, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAU: f64 = 2.0 * std::f64::consts::PI;

    /// Harmonic oscillator r'' = -r, analytic solution r(t) = cos(t).
    #[test]
    fn gj8_harmonic_oscillator() {
        let r0 = Vector::<f64, 1>::from_array([1.0]);
        let v0 = Vector::<f64, 1>::from_array([0.0]);
        let settings = GJSettings {
            h: TAU / 500.0,
            ..GJSettings::default()
        };
        let sol = GaussJackson8::integrate(
            0.0, TAU, &r0, &v0,
            |_t, r, _v| -*r,
            &settings,
        ).unwrap();
        assert!(
            (sol.r[0] - 1.0).abs() < 1e-10,
            "harmonic r[0] = {:.3e}, err = {:.3e}", sol.r[0], (sol.r[0] - 1.0).abs()
        );
        assert!(sol.v[0].abs() < 1e-10, "harmonic v[0] = {:.3e}", sol.v[0]);
    }

    /// Damped harmonic oscillator with velocity-dependent force: r'' = -r - γv.
    /// Exercises the velocity argument of the force closure — this is the
    /// combined Gauss-Jackson + Summed-Adams formulation, not pure GJ.
    #[test]
    fn gj8_damped_oscillator() {
        let r0 = Vector::<f64, 1>::from_array([1.0]);
        let v0 = Vector::<f64, 1>::from_array([0.0]);
        let gamma: f64 = 0.01;
        let omega: f64 = 1.0;
        let t_final = 10.0_f64;

        let settings = GJSettings { h: 1.0e-2, ..GJSettings::default() };
        let sol = GaussJackson8::integrate(
            0.0, t_final, &r0, &v0,
            |_t, r, v| -*r * (omega * omega) - *v * gamma,
            &settings,
        ).unwrap();

        let wd = (omega * omega - 0.25 * gamma * gamma).sqrt();
        let env = (-0.5 * gamma * t_final).exp();
        let expected = env * (f64::cos(wd * t_final)
                              + 0.5 * gamma / wd * f64::sin(wd * t_final));
        assert!(
            (sol.r[0] - expected).abs() < 1e-8,
            "damped r = {:.6e}, expected {:.6e}, err = {:.3e}",
            sol.r[0], expected, (sol.r[0] - expected).abs()
        );
    }

    /// Kepler two-body: r'' = -r/|r|³. Circular orbit at r=1 has period 2π.
    /// Integrate 10 orbits and check energy + angular momentum conservation.
    #[test]
    fn gj8_kepler_circular_10_orbits() {
        let r0 = Vector::<f64, 3>::from_array([1.0, 0.0, 0.0]);
        let v0 = Vector::<f64, 3>::from_array([0.0, 1.0, 0.0]);

        let energy = |r: &Vector<f64, 3>, v: &Vector<f64, 3>| {
            let rmag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - 1.0 / rmag
        };
        let lz = |r: &Vector<f64, 3>, v: &Vector<f64, 3>| r[0] * v[1] - r[1] * v[0];

        let e0 = energy(&r0, &v0);
        let l0 = lz(&r0, &v0);

        let tf = 10.0 * TAU;
        let settings = GJSettings { h: TAU / 200.0, ..GJSettings::default() };
        let sol = GaussJackson8::integrate(
            0.0, tf, &r0, &v0,
            |_t, r, _v| {
                let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                *r * (-1.0 / (r2 * r2.sqrt()))
            },
            &settings,
        ).unwrap();

        let e_drift = ((energy(&sol.r, &sol.v) - e0) / e0.abs()).abs();
        let l_drift = (lz(&sol.r, &sol.v) - l0).abs() / l0.abs();
        let pos_err = ((sol.r[0] - 1.0).powi(2) + sol.r[1].powi(2) + sol.r[2].powi(2)).sqrt();

        assert!(e_drift < 1e-10, "energy drift = {:.3e}", e_drift);
        assert!(l_drift < 1e-10, "Lz drift = {:.3e}", l_drift);
        assert!(pos_err < 1e-6, "position drift after 10 orbits = {:.3e}", pos_err);
    }

    /// Quintic Hermite interpolation on stored dense output should recover
    /// the harmonic oscillator to near 5th-order accuracy in the step size.
    #[test]
    fn gj8_dense_interpolation_harmonic() {
        let r0 = Vector::<f64, 1>::from_array([1.0]);
        let v0 = Vector::<f64, 1>::from_array([0.0]);
        let tf = TAU;

        let settings = GJSettings {
            h: TAU / 100.0,
            dense_output: true,
            ..GJSettings::default()
        };
        let sol = GaussJackson8::integrate(
            0.0, tf, &r0, &v0,
            |_t, r, _v| -*r,
            &settings,
        ).unwrap();

        let dense = sol.dense.as_ref().expect("dense output should be present");
        assert!(dense.t.len() >= 100, "expected at least 100 stored samples");

        // Interpolate at times that are NOT exactly on step boundaries.
        // With h ≈ 0.063, errors should be O(h⁵) ≈ 1e-6.
        let test_points = [0.1, 1.0, 2.5, 4.0, 5.5];
        for &t in &test_points {
            let (r, v) = GaussJackson8::interpolate(t, &sol).unwrap();
            let expected_r = t.cos();
            let expected_v = -t.sin();
            assert!(
                (r[0] - expected_r).abs() < 1e-6,
                "interp r at t={}: got {}, expected {}, err = {:.3e}",
                t, r[0], expected_r, (r[0] - expected_r).abs()
            );
            assert!(
                (v[0] - expected_v).abs() < 1e-6,
                "interp v at t={}: got {}, expected {}, err = {:.3e}",
                t, v[0], expected_v, (v[0] - expected_v).abs()
            );
        }

        // Step boundaries should be exact (within fp round-off).
        for (i, &ti) in dense.t.iter().enumerate() {
            let (r, _) = GaussJackson8::interpolate(ti, &sol).unwrap();
            assert!(
                (r[0] - dense.r[i][0]).abs() < 1e-12,
                "at stored sample index {}, interp disagreed with sample", i
            );
        }
    }

    /// Batch interpolation matches single-point interpolation.
    #[test]
    fn gj8_dense_interpolation_batch() {
        let r0 = Vector::<f64, 1>::from_array([1.0]);
        let v0 = Vector::<f64, 1>::from_array([0.0]);
        let settings = GJSettings {
            h: TAU / 100.0,
            dense_output: true,
            ..GJSettings::default()
        };
        let sol = GaussJackson8::integrate(
            0.0, TAU, &r0, &v0,
            |_t, r, _v| -*r,
            &settings,
        ).unwrap();

        let times = [0.5, 1.3, 2.7, 4.2, 5.9];
        let batch = GaussJackson8::interpolate_batch(&times, &sol).unwrap();
        for (i, &t) in times.iter().enumerate() {
            let single = GaussJackson8::interpolate(t, &sol).unwrap();
            assert!((batch[i].0[0] - single.0[0]).abs() < 1e-15);
            assert!((batch[i].1[0] - single.1[0]).abs() < 1e-15);
        }
    }

    /// Sanity check: GJ8 should need fewer function evaluations than RKV98
    /// for comparable accuracy on a smooth Kepler problem.
    #[test]
    fn gj8_fewer_evals_than_rkv98() {
        use numeris::ode::{AdaptiveSettings, RKAdaptive, RKV98};

        let tf = 5.0 * TAU;
        let r0 = Vector::<f64, 3>::from_array([1.0, 0.0, 0.0]);
        let v0 = Vector::<f64, 3>::from_array([0.0, 1.0, 0.0]);

        let gj_sol = GaussJackson8::integrate(
            0.0, tf, &r0, &v0,
            |_t, r, _v| {
                let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                *r * (-1.0 / (r2 * r2.sqrt()))
            },
            &GJSettings { h: TAU / 100.0, ..GJSettings::default() },
        ).unwrap();

        let y0_6 = Vector::<f64, 6>::from_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let rk_sol = RKV98::integrate(
            0.0, tf, &y0_6,
            |_t, y| {
                let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
                let s = -1.0 / (r2 * r2.sqrt());
                Vector::from_array([y[3], y[4], y[5], y[0] * s, y[1] * s, y[2] * s])
            },
            &AdaptiveSettings::<f64> {
                abs_tol: 1e-11,
                rel_tol: 1e-11,
                ..Default::default()
            },
        ).unwrap();

        assert!(
            gj_sol.evals < rk_sol.evals,
            "GJ8 used {} evals, RKV98 used {} — expected GJ8 to be more efficient",
            gj_sol.evals, rk_sol.evals
        );
    }
}
