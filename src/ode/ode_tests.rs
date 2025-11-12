//! Test for ODE solvers
//!
//! Tests are for a simple 1D harmonic oscillator
//! which can be easily checked for correctness against the analytical solution.
//!
//! The tests are run for all adaptive solvers with and without interpolation.
//!

use super::ODEResult;
use super::RKAdaptive;
use super::RKAdaptiveSettings;

/// ODE state for 1-dimensional harmonic oscillator
/// y[0] = position
/// y[1] = velocity
type State = crate::mathtypes::Vector2;

/// Harmonic oscillator ODE y'' = -y
fn ydot(_t: f64, y: &State) -> ODEResult<State> {
    Ok([y[1], -y[0]].into())
}

/// Test harmonic oscillator
fn harmonic_oscillator<const N: usize, const NI: usize, F>(_integrator: F)
where
    F: RKAdaptive<N, NI>,
{
    use std::f64::consts::PI;

    let y0 = State::new(1.0, 0.0);

    let settings = RKAdaptiveSettings {
        dense_output: false,
        abserror: 1e-12,
        relerror: 1e-12,
        ..RKAdaptiveSettings::default()
    };

    let res = F::integrate(0.0, 2.0 * PI, &y0, ydot, &settings).unwrap();
    assert!((res.y[0] - 1.0).abs() < 1e-11);
    assert!((res.y[1]).abs() < 1e-11);
}

/// Test harmonic oscillator with interpolation
fn harmonic_oscillator_interp<const N: usize, const NI: usize, F>(_integrator: F)
where
    F: RKAdaptive<N, NI>,
{
    use std::f64::consts::PI;

    let y0 = State::new(1.0, 0.0);

    let settings = RKAdaptiveSettings {
        dense_output: true,
        abserror: 1e-12,
        relerror: 1e-12,
        ..RKAdaptiveSettings::default()
    };

    let res = F::integrate(0.0, PI, &y0, ydot, &settings).unwrap();

    let testcount = 100;
    (0..testcount).for_each(|idx| {
        let x = idx as f64 * PI / testcount as f64;
        let interp = F::interpolate(x, &res).unwrap();
        assert!((interp[0] - x.cos()).abs() < 1e-10);
        assert!((interp[1] + x.sin()).abs() < 1e-10);
    });
}

/// Test harmonic oscillator with all integrators
#[test]
fn test_harmonic_oscillator() {
    harmonic_oscillator(super::solvers::RKF45 {});
    harmonic_oscillator(super::solvers::RKTS54 {});
    harmonic_oscillator(super::solvers::RKV65 {});
    harmonic_oscillator(super::solvers::RKV87 {});
    harmonic_oscillator(super::solvers::RKV98 {});
    harmonic_oscillator(super::solvers::RKV98NoInterp {});
    harmonic_oscillator(super::solvers::RKV98Efficient {});
}

/// Test harmonic oscillator with all integrators with interpolation
#[test]
fn test_harmonic_oscillator_interp() {
    harmonic_oscillator_interp(super::solvers::RKTS54 {});
    harmonic_oscillator_interp(super::solvers::RKV65 {});
    harmonic_oscillator_interp(super::solvers::RKV87 {});
    harmonic_oscillator_interp(super::solvers::RKV98 {});
    harmonic_oscillator_interp(super::solvers::RKV98Efficient {});
}

/// Test FSAL optimization reduces function evaluations
#[test]
fn test_fsal_optimization() {
    use std::f64::consts::PI;

    let y0 = State::new(1.0, 0.0);

    let settings = RKAdaptiveSettings {
        dense_output: false,
        abserror: 1e-10,
        relerror: 1e-10,
        ..RKAdaptiveSettings::default()
    };

    // RKTS54 has FSAL enabled
    let res_fsal = super::solvers::RKTS54::integrate(0.0, 2.0 * PI, &y0, ydot, &settings).unwrap();

    // RKF45 does not have FSAL
    let res_no_fsal =
        super::solvers::RKF45::integrate(0.0, 2.0 * PI, &y0, ydot, &settings).unwrap();

    println!(
        "RKTS54 (FSAL=true): {} evals, {} accepted steps",
        res_fsal.nevals, res_fsal.naccept
    );
    println!(
        "RKF45 (FSAL=false): {} evals, {} accepted steps",
        res_no_fsal.nevals, res_no_fsal.naccept
    );

    // FSAL should save approximately 1 evaluation per accepted step
    // (exact savings depends on rejected steps)
    // For RKTS54 (7 stages), without FSAL we'd expect ~7*naccept evaluations
    // With FSAL, we expect ~(7*naccept - (naccept-1)) = 6*naccept + 1 evaluations
    let expected_savings = res_fsal.naccept.saturating_sub(1);
    let theoretical_no_fsal = res_fsal.naccept * 7;

    println!("Expected savings: ~{} evaluations", expected_savings);
    println!("Theoretical evals without FSAL: {}", theoretical_no_fsal);

    // Verify FSAL actually reduces evaluations
    assert!(
        res_fsal.nevals < theoretical_no_fsal,
        "FSAL should reduce function evaluations"
    );
}

/// Compare RKV98 Robust vs Efficient versions
#[test]
fn test_rkv98_comparison() {
    use std::f64::consts::PI;

    let y0 = State::new(1.0, 0.0);

    let settings = RKAdaptiveSettings {
        dense_output: false,
        abserror: 1e-12,
        relerror: 1e-12,
        ..RKAdaptiveSettings::default()
    };

    // RKV98 Robust (21 stages)
    let res_robust = super::solvers::RKV98::integrate(0.0, 2.0 * PI, &y0, ydot, &settings).unwrap();

    // RKV98 Efficient (26 stages)
    let res_efficient =
        super::solvers::RKV98Efficient::integrate(0.0, 2.0 * PI, &y0, ydot, &settings).unwrap();

    println!("\nRKV98 Comparison:");
    println!(
        "Robust (21 stages): {} evals, {} accepted, {} rejected steps",
        res_robust.nevals, res_robust.naccept, res_robust.nreject
    );
    println!(
        "Efficient (26 stages): {} evals, {} accepted, {} rejected steps",
        res_efficient.nevals, res_efficient.naccept, res_efficient.nreject
    );

    // Both should achieve the same accuracy
    assert!((res_robust.y[0] - 1.0).abs() < 1e-11);
    assert!((res_efficient.y[0] - 1.0).abs() < 1e-11);
}
