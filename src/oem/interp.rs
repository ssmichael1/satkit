//! Polynomial interpolation of OEM state vectors.

use super::{Interpolation, State};
use crate::mathtypes::Vector3;
use crate::Instant;

/// Interpolate `states` (non-empty, strictly increasing in time) at `time`.
///
/// `degree` (at least 1) is the polynomial degree: Lagrange uses
/// `degree + 1` points, Hermite `degree / 2 + 1`, fewer if the
/// table is shorter. The points are the ones centered on `time`.
pub(super) fn interp(
    states: &[State],
    time: &Instant,
    method: Interpolation,
    degree: usize,
) -> (Vector3, Vector3) {
    let want = match method {
        Interpolation::Lagrange => degree + 1,
        Interpolation::Hermite => degree / 2 + 1,
    };
    let n = want.clamp(1, states.len());
    // Index of the first state after `time`, then a window of `n` states
    // with as many before it as after
    let after = states.partition_point(|s| s.time <= *time);
    let first = after.saturating_sub(n.div_ceil(2)).min(states.len() - n);
    let window = &states[first..first + n];

    // Node times in seconds relative to `time`, so the polynomial is
    // evaluated at 0 and the arithmetic stays well scaled
    let t: Vec<f64> = window
        .iter()
        .map(|s| (s.time - *time).as_seconds())
        .collect();

    // A tabulated epoch is returned as it is, whatever the method
    if let Some(i) = t.iter().position(|&dt| dt == 0.0) {
        return (window[i].pos, window[i].vel);
    }

    match method {
        Interpolation::Lagrange => lagrange(window, &t),
        Interpolation::Hermite => hermite(window, &t),
    }
}

/// Lagrange interpolation at 0 of position and velocity, each on its own.
fn lagrange(window: &[State], t: &[f64]) -> (Vector3, Vector3) {
    let mut pos = [0.0; 3];
    let mut vel = [0.0; 3];
    for (i, s) in window.iter().enumerate() {
        // Basis polynomial i at 0
        let w: f64 = t
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &tj)| -tj / (t[i] - tj))
            .product();
        for k in 0..3 {
            pos[k] += w * s.pos[k];
            vel[k] += w * s.vel[k];
        }
    }
    (
        numeris::vector![pos[0], pos[1], pos[2]],
        numeris::vector![vel[0], vel[1], vel[2]],
    )
}

/// Hermite interpolation at 0: the polynomial matching position and its
/// derivative (velocity) at every node, and that polynomial's derivative.
fn hermite(window: &[State], t: &[f64]) -> (Vector3, Vector3) {
    let n = window.len();
    let m = 2 * n;
    // Every node twice
    let z: Vec<f64> = t.iter().flat_map(|&ti| [ti, ti]).collect();
    let mut pos = [0.0; 3];
    let mut vel = [0.0; 3];
    let mut c = vec![0.0; m];
    for k in 0..3 {
        // Newton divided differences, in place; a repeated node takes the
        // derivative as its first difference
        for i in 0..n {
            c[2 * i] = window[i].pos[k];
            c[2 * i + 1] = window[i].pos[k];
        }
        for j in 1..m {
            for i in (j..m).rev() {
                c[i] = if j == 1 && i % 2 == 1 {
                    window[i / 2].vel[k]
                } else {
                    (c[i] - c[i - 1]) / (z[i] - z[i - j])
                };
            }
        }
        // Horner evaluation of the Newton form and its derivative at 0
        let mut p = c[m - 1];
        let mut dp = 0.0;
        for i in (0..m - 1).rev() {
            dp = dp * -z[i] + p;
            p = p * -z[i] + c[i];
        }
        pos[k] = p;
        vel[k] = dp;
    }
    (
        numeris::vector![pos[0], pos[1], pos[2]],
        numeris::vector![vel[0], vel[1], vel[2]],
    )
}
