use super::ODEResult;
use super::RKAdaptive;
use super::RKAdaptiveSettings;

type State = nalgebra::Vector2<f64>;

fn ydot(_t: f64, y: &State) -> ODEResult<State> {
    Ok([y[1], -y[0]].into())
}

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
    (0..100).for_each(|idx| {
        let x = idx as f64 * PI / testcount as f64;
        let interp = F::interpolate(x, &res).unwrap();
        assert!((interp[0] - x.cos()).abs() < 1e-10);
        assert!((interp[1] + x.sin()).abs() < 1e-10);
    });
}

/// Test harmonic oscillator with all integrators
#[test]
fn test_harmonic_oscillator() {
    harmonic_oscillator(super::rkts54::RKTS54 {});
    harmonic_oscillator(super::rkv65::RKV65 {});
    harmonic_oscillator(super::rkv87::RKV87 {});
    harmonic_oscillator(super::rkv98::RKV98 {});
    harmonic_oscillator(super::rkv98_nointerp::RKV98NoInterp {});
}

/// Test harmonic oscillator with all integrators with interpolation
#[test]
fn test_harmonic_oscillator_interp() {
    harmonic_oscillator_interp(super::rkts54::RKTS54 {});
    harmonic_oscillator_interp(super::rkv65::RKV65 {});
    harmonic_oscillator_interp(super::rkv87::RKV87 {});
    harmonic_oscillator_interp(super::rkv98::RKV98 {});
}
