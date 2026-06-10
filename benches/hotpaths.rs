//! Criterion benchmarks for satkit hot paths
//!
//! Covers the per-call costs that dominate real workloads: high-precision
//! propagation (per-regime and per-integrator), SGP4, frame transforms,
//! spherical-harmonic gravity, JPL ephemeris lookup, and the NRLMSISE-00
//! density model.
//!
//! Data files (EOP, gravity models, JPL ephemeris, space weather) are
//! resolved through the normal `satkit::utils::datadir()` discovery, same
//! as the test suite. Run with `cargo bench`.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use satkit::consts;
use satkit::frametransform;
use satkit::mathtypes::{Matrix6, Matrix67, Vector3, Vector6};
use satkit::orbitprop::{propagate, Integrator, PropSettings, SatPropertiesSimple};
use satkit::{Duration, Instant};

fn epoch() -> Instant {
    Instant::from_rfc3339("2024-01-01T00:00:00Z").unwrap()
}

/// Circular orbit state at radius `r` (meters) with inclination `incl` (radians)
fn circular_state(r: f64, incl: f64) -> Vector6 {
    let v = (consts::MU_EARTH / r).sqrt();
    numeris::vector![r, 0.0, 0.0, 0.0, v * incl.cos(), v * incl.sin()]
}

const ISS_INCL_RAD: f64 = 51.6 * std::f64::consts::PI / 180.0;

fn bench_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagation");
    group
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(20));

    let t0 = epoch();
    let t1 = t0 + Duration::from_days(1.0);

    // LEO with drag + full default force model, adaptive RKV 9(8)
    let leo = circular_state(consts::WGS84_A + 500.0e3, ISS_INCL_RAD);
    let satprops = SatPropertiesSimple::new(0.02, 0.01);
    let settings = PropSettings::default();
    group.bench_function("leo_drag_1day_rkv98", |b| {
        b.iter(|| propagate(black_box(&leo), &t0, &t1, &settings, Some(&satprops)).unwrap())
    });

    // GEO, sun/moon gravity dominated, adaptive RKV 9(8)
    let geo = circular_state(consts::GEO_R, 0.0);
    group.bench_function("geo_1day_rkv98", |b| {
        b.iter(|| propagate(black_box(&geo), &t0, &t1, &settings, None).unwrap())
    });

    // GEO, fixed-step Gauss-Jackson 8
    let settings_gj8 = PropSettings {
        integrator: Integrator::GaussJackson8,
        gj_step_seconds: 300.0,
        ..PropSettings::default()
    };
    group.bench_function("geo_1day_gj8", |b| {
        b.iter(|| propagate(black_box(&geo), &t0, &t1, &settings_gj8, None).unwrap())
    });

    // LEO with 6x6 state transition matrix (C = 7 state columns)
    let mut leo_stm = Matrix67::zeros();
    leo_stm.set_block(0, 0, &leo);
    leo_stm.set_block(0, 1, &Matrix6::eye());
    group.bench_function("leo_stm_1day_rkv98", |b| {
        b.iter(|| propagate(black_box(&leo_stm), &t0, &t1, &settings, Some(&satprops)).unwrap())
    });

    group.finish();
}

fn bench_sgp4(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgp4");

    let line0 = "0 INTELSAT 902";
    let line1 = "1 26900U 01039A   06106.74503247  .00000045  00000-0  10000-3 0  8290";
    let line2 = "2 26900   0.0164 266.5378 0003319  86.1794 182.2590  1.00273847 16981";
    let tle = satkit::TLE::load_3line(line0, line1, line2).unwrap();

    // 1 day of output at 1-minute cadence
    let times: Vec<Instant> = (0..1441)
        .map(|i| tle.epoch + Duration::from_seconds(60.0 * i as f64))
        .collect();

    // Includes per-TLE SGP4 initialization (fresh TLE each iteration)
    group.bench_function("init_plus_1day_1min", |b| {
        b.iter(|| {
            let mut t = tle.clone();
            satkit::sgp4::sgp4(&mut t, black_box(times.as_slice())).unwrap()
        })
    });

    // Cached init state (same TLE reused), measures the propagation only
    let mut warm = tle.clone();
    let _ = satkit::sgp4::sgp4(&mut warm, &times[..1]).unwrap();
    group.bench_function("cached_1day_1min", |b| {
        b.iter(|| satkit::sgp4::sgp4(&mut warm, black_box(times.as_slice())).unwrap())
    });

    group.finish();
}

fn bench_frametransform(c: &mut Criterion) {
    let mut group = c.benchmark_group("frametransform");
    let tm = epoch();

    // Warm the EOP singleton so first-load cost isn't measured
    let _ = frametransform::qgcrf2itrf(&tm);

    group.bench_function("qgcrf2itrf", |b| {
        b.iter(|| frametransform::qgcrf2itrf(black_box(&tm)))
    });
    group.bench_function("qgcrf2itrf_approx", |b| {
        b.iter(|| frametransform::qgcrf2itrf_approx(black_box(&tm)))
    });
    group.bench_function("qteme2itrf", |b| {
        b.iter(|| frametransform::qteme2itrf(black_box(&tm)))
    });
    group.bench_function("gmst", |b| b.iter(|| frametransform::gmst(black_box(&tm))));

    group.finish();
}

fn bench_earthgravity(c: &mut Criterion) {
    let mut group = c.benchmark_group("earthgravity");

    // LEO position in ITRF (slightly off-axis to exercise all terms)
    let pos: Vector3 = numeris::vector![consts::WGS84_A + 400.0e3, 1000.0e3, 2000.0e3];
    let gravity = satkit::earthgravity::jgm3();

    group.bench_function("accel_deg4", |b| {
        b.iter(|| gravity.accel(black_box(&pos), 4, 4))
    });
    group.bench_function("accel_deg16", |b| {
        b.iter(|| gravity.accel(black_box(&pos), 16, 16))
    });
    group.bench_function("accel_and_partials_deg4", |b| {
        b.iter(|| gravity.accel_and_partials(black_box(&pos), 4, 4))
    });

    group.finish();
}

fn bench_jplephem(c: &mut Criterion) {
    let mut group = c.benchmark_group("jplephem");
    let tm = epoch();

    // Warm the ephemeris singleton
    let _ = satkit::jplephem::geocentric_pos(satkit::SolarSystem::Moon, &tm);

    group.bench_function("moon_geocentric_pos", |b| {
        b.iter(|| satkit::jplephem::geocentric_pos(satkit::SolarSystem::Moon, black_box(&tm)))
    });
    group.bench_function("sun_geocentric_state", |b| {
        b.iter(|| satkit::jplephem::geocentric_state(satkit::SolarSystem::Sun, black_box(&tm)))
    });

    group.finish();
}

fn bench_nrlmsise(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrlmsise");
    let tm = epoch();

    group.bench_function("density_400km", |b| {
        b.iter(|| {
            satkit::nrlmsise::nrlmsise(black_box(400.0), Some(0.5), Some(0.5), Some(&tm), true)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_propagation,
    bench_sgp4,
    bench_frametransform,
    bench_earthgravity,
    bench_jplephem,
    bench_nrlmsise
);
criterion_main!(benches);
