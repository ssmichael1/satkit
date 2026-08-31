# Keplerian Elements

This page describes the classical (Keplerian) orbital element set used by
[`satkit.kepler`](../api/kepler.md) (Rust: `satkit::kepler::Kepler`), the
conventions it follows, and what it does *not* do. For a worked notebook see
the [Keplerian Elements tutorial](../tutorials/Keplerian%20Elements.ipynb).

## The Element Set

A bound two-body orbit is described by six numbers. `satkit` stores them under
these names, in SI units and radians, together with the gravitational
parameter of the body they orbit:

| Field   | Symbol     | Meaning                                              |
|---------|------------|------------------------------------------------------|
| `a`     | $a$        | semi-major axis, **meters**, $a > 0$                 |
| `eccen` | $e$        | eccentricity, $0 \le e < 1$                          |
| `incl`  | $i$        | inclination, radians, $0 \le i \le \pi$              |
| `raan`  | $\Omega$   | right ascension of the ascending node, radians       |
| `argp`  | $\omega$   | argument of periapsis, radians                       |
| `nu`    | $\nu$      | true anomaly, radians                                |
| `mu`    | $\mu$      | gravitational parameter of the central body, m³ s⁻²; Earth's unless given |

The size and shape of the ellipse are $a$ and $e$; the orientation of the
orbital plane and of the ellipse within it are $i$, $\Omega$ and $\omega$; and
$\nu$ locates the satellite along the ellipse at the epoch of the elements.
The semiparameter (semi-latus rectum) $p = a(1 - e^2)$ is available as a
derived property, but the class is constructed from $a$, not $p$.

In Python the inclination property is spelled `inclination`; the constructor
argument and the Rust field are `incl`. The argument of periapsis was called
`w` before 0.22; in Python `w` still works as a constructor keyword and as a
property (kept indefinitely), with a `DeprecationWarning`. Angles returned by `from_pv` are
reduced to $[0, 2\pi)$; angles you set are stored as given.

### Validation

The Python constructor, `from_pv`, and every element setter reject an
element outside its domain with `ValueError`: a non-finite value,
$a \le 0$, $e \notin [0, 1)$, $i \notin [0, \pi]$ or $\mu \le 0$. The bounds
are strict — $e = 1$ is not a closed orbit and is refused rather than
producing NaN anomalies. (The `mean_anomaly` / `eccentric_anomaly` setters
are the exception: they accept any value and yield NaN for a non-finite one,
since they go through the capped Kepler's-equation solver.) In Rust, `Kepler::try_new`
performs the same checks (returning `kepler::Error::InvalidElement`, which
names the offending element) and `Kepler::validate` re-checks an element set
whose public fields were assigned directly; `Kepler::new` remains unchecked.

### Derived quantities

All are read-only properties in Python and methods in Rust:

| Property                | Symbol / formula                                   | Units |
|-------------------------|----------------------------------------------------|-------|
| `semiparameter`         | $p = a(1 - e^2)$                                   | m     |
| `periapsis`             | $r_p = a(1 - e)$                                   | m     |
| `apoapsis`              | $r_a = a(1 + e)$                                   | m     |
| `mean_motion`           | $n = \sqrt{\mu / a^3}$                             | rad/s |
| `period`                | $T = 2\pi / n$                                     | s     |
| `specific_energy`       | $\xi = -\mu / 2a$                                  | J/kg  |
| `angular_momentum`      | $h = \sqrt{\mu p}$                                 | m²/s  |
| `flight_path_angle`     | $\gamma = \operatorname{atan2}(e\sin\nu,\ 1 + e\cos\nu)$ — zero at periapsis and apoapsis, positive while climbing | rad |
| `argument_of_latitude`  | $u = \omega + \nu$, reduced to $[0, 2\pi)$ — defined for circular orbits | rad |
| `true_longitude`        | $\lambda = \Omega + \omega + \nu$, reduced to $[0, 2\pi)$ — defined for circular equatorial orbits | rad |

`satstate.from_kepler(time, k)` (Rust `SatState::from_kepler`) builds a
propagatable state from the two-body position and velocity of an element
set, taken as GCRF at `time`.

### Anomalies

Three angles can locate the satellite in its orbit
([Vallado 2013](references.md#vallado2013), §2.2):

- **True anomaly** $\nu$ — the angle at the focus (Earth's center) from
  perigee to the satellite. This is what is stored (`nu`).
- **Eccentric anomaly** $E$ — the angle at the *center* of the ellipse to the
  point on the auxiliary circle above the satellite. Related to $\nu$ by
  $\tan\frac{\nu}{2} = \sqrt{\frac{1+e}{1-e}}\tan\frac{E}{2}$.
- **Mean anomaly** $M$ — the angle that advances uniformly in time,
  $M = M_0 + n\,(t - t_0)$ with mean motion $n = \sqrt{\mu / a^3}$. Related to
  $E$ by Kepler's equation $M = E - e\sin E$.

Converting $M \to E$ requires solving Kepler's equation iteratively. `satkit`
uses Newton's method with the [Danby (1987)](references.md#danby1987) starting
value $E_0 = M + 0.85\,e\,\mathrm{sign}(\sin M)$ after reducing $M$ to
$[0, 2\pi)$, which converges in a handful of iterations for every $e < 1$
([Vallado 2013](references.md#vallado2013), Algorithm 2). The iteration is
capped, so a non-finite $M$ yields NaN rather than looping.

The class accepts any of the three anomalies when it is constructed, and
exposes all three as properties; `mean_anomaly` and `eccentric_anomaly` can
be assigned and are converted to `nu` on the spot.

## What the Elements Mean (and Don't)

**Osculating, not mean.** The elements are *osculating*: they describe the
two-body orbit that is tangent to the actual trajectory at the epoch of the
state. Under perturbations (oblateness, drag, third bodies, …) the osculating
elements vary continuously along the orbit — for a LEO satellite $a$ oscillates
by several kilometers within one revolution because of $J_2$ alone. They are
not the *mean* elements of an analytical theory. In particular, the elements
in a TLE are SGP4 mean elements and cannot be passed to `kepler` (or read
back from `from_pv`) without a significant, model-dependent error; use
[`satkit.TLE`](../api/tle.md) and the SGP4 propagator for those
(see [TLEs, SGP4 & OMMs](tle.md)).

**Frame.** `kepler` does no frame handling. `from_pv` interprets the position
and velocity you pass in whatever frame they are in, and `to_pv` returns the
state in that same frame. Elements are meaningful only in an inertial frame;
the rest of `satkit` assumes **GCRF**, so convert ITRF or TEME states with
[`frametransform`](../api/frametransform.md) first. Passing an Earth-fixed
state produces elements that are numerically valid but physically
meaningless.

**Central body.** Each element set carries its own $\mu$. By default it is
the Earth's, [`consts.MU_EARTH`](../api/consts.md) ($3.986004418 \times
10^{14}$ m³ s⁻²), and it is used everywhere: in the `from_pv` energy
equation, in `to_pv`, and in the mean motion, period and `propagate`. For a
lunar or heliocentric orbit pass `mu=` to the constructor or to `from_pv`
(Rust: `Kepler::with_mu`, `Kepler::from_pv_with_mu`); the six geometric
elements are unchanged, only the dynamics re-target the other body. Note
that `from_pv` interprets a state with whatever $\mu$ it is given — a lunar
state read with Earth's $\mu$ yields a valid-looking but wrong ellipse.

**Closed orbits only.** `from_pv` returns an error (Python: `ValueError`)
for parabolic or hyperbolic states ($e \ge 1$) and for rectilinear states
(zero angular momentum, where the orbital plane is undefined). The Python
constructor and `Kepler::try_new` reject $e \ge 1$ up front (see
[Validation](#validation)).

**Singular cases.** $\Omega$ is undefined for an equatorial orbit and $\omega$
for a circular one. `from_pv` follows the conventions of
[Vallado (2013)](references.md#vallado2013), Algorithm 9: for a circular
inclined orbit `argp` is 0 and `nu` holds the argument of latitude; for an
elliptical equatorial orbit `raan` is 0 and `argp` holds the true longitude of
perigee; for a circular equatorial orbit both are 0 and `nu` holds the true
longitude. In each case `to_pv` reproduces the input state. The
`argument_of_latitude` and `true_longitude` properties give the well-defined
combinations directly in every case.

## Conversions

- **Elements → state** follows [Vallado (2013)](references.md#vallado2013),
  Algorithm 10: the state is formed in the perifocal (PQW) frame and rotated
  by $R_z(\Omega)\,R_x(i)\,R_z(\omega)$.
- **State → elements** follows Algorithm 9, with every angle extracted by
  `atan2` rather than `acos` so that near-zero inclinations (down to
  $10^{-9}$ rad) and anomalies near perigee/apogee are recovered to full
  precision; the round trip state → elements → state is accurate to better
  than $10^{-6}$ relative for $e \le 0.999$.
- **`propagate(dt)`** is pure two-body motion: only the mean anomaly advances,
  by $n\,\Delta t$. No perturbation is applied. For anything beyond a quick
  look, use the numerical propagator ([Force Model](forces.md)).

## Examples

=== "Python"

    ```python
    import math
    import numpy as np
    import satkit as sk

    # Sun-synchronous-ish LEO, located by mean anomaly
    k = sk.kepler(
        a=7000.0e3,
        eccen=0.001,
        incl=math.radians(98.0),
        raan=math.radians(45.0),
        argp=0.0,
        mean_anomaly=math.radians(30.0),
    )
    print(f"period = {k.period / 60:.2f} min, nu = {math.degrees(k.nu):.3f} deg")

    # Elements -> GCRF state -> elements
    r, v = k.to_pv()
    k2 = sk.kepler.from_pv(r, v)
    assert abs(k2.a - k.a) < 1e-3

    # Two-body propagation by a quarter period
    k3 = k.propagate(k.period / 4)
    print(f"mean anomaly after T/4 = {math.degrees(k3.mean_anomaly):.3f} deg")

    # An osculating snapshot of a numerically propagated state
    t0 = sk.time(2024, 1, 1)
    state = sk.satstate(t0, r, v)
    state1 = state.propagate(t0 + sk.duration.from_hours(1))
    k_osc = sk.kepler.from_pv(state1.pos, state1.vel)
    print(f"osculating a after 1 h: {k_osc.a / 1e3:.3f} km")
    ```

=== "Rust"

    ```rust
    use satkit::kepler::{Anomaly, Kepler};
    use satkit::Duration;

    let k = Kepler::new(
        7000.0e3,                 // a, m
        0.001,                    // eccen
        98.0_f64.to_radians(),    // incl
        45.0_f64.to_radians(),    // raan
        0.0,                      // argp
        Anomaly::Mean(30.0_f64.to_radians()),
    );
    println!("period = {:.2} min, nu = {:.3} deg",
             k.period() / 60.0, k.nu.to_degrees());

    // Elements -> state -> elements
    let (r, v) = k.to_pv();
    let k2 = Kepler::from_pv(r, v)?;
    assert!((k2.a - k.a).abs() < 1e-3);

    // Two-body propagation by a quarter period
    let k3 = k.propagate(&Duration::from_seconds(k.period() / 4.0));
    println!("M after T/4 = {:.3} deg", k3.mean_anomaly().to_degrees());
    # Ok::<(), satkit::kepler::Error>(())
    ```

## References

- [Vallado, D. A. (2013)](references.md#vallado2013), *Fundamentals of Astrodynamics and Applications*, 4th ed., Microcosm Press. Algorithm 2 (Kepler's equation), Algorithm 9 (RV2COE), Algorithm 10 (COE2RV); §2.2–2.5.
- [Danby, J. M. A. (1987)](references.md#danby1987), "The solution of Kepler's equation, III," *Celestial Mechanics*, 40, 303–312. <https://doi.org/10.1007/BF01235847>
