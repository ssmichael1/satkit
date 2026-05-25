# Force Model

The `satkit` numerical propagator integrates the forces acting on a satellite to produce a change in velocity, then integrates the velocity to produce a change in position:

$$
\vec{v}(t_1)~=~\vec{v}(t_0) + \int_{t_0}^{t_1}~\vec{a}\left ( t,~\vec{p}_{t},~\vec{v}_{t} \right ) ~dt
$$

$$
\vec{p}(t_1)~=~\vec{p}(t_0) + \int_{t_0}^{t_1}~\vec{v}(t)~dt
$$

This page describes each force in $\vec{a}\left (t, \vec{p}, \vec{v}\right )$ that satkit models. The integration mechanics live in the [ODE Integrators](integrators.md) page; state vectors, the state transition matrix, and covariance propagation live in [State Vectors, STM & Covariance](satstate.md).

The force model follows the treatment in O. Montenbruck and E. Gill, [**"Satellite Orbits: Models, Methods, Applications"**](https://doi.org/10.1007/978-3-642-58351-3) — consult that book for more depth on any term.

## Summary

| Force | Default | Setting | Order of magnitude (LEO) |
|---|---|---|---|
| Earth gravity (spherical harmonics) | on | `gravity_degree`, `gravity_order`, `gravity_model` | $10^0$ m/s² |
| Sun third-body | on | `use_sun_gravity` | $10^{-6}$ m/s² |
| Moon third-body | on | `use_moon_gravity` | $10^{-6}$ m/s² |
| Atmospheric drag (NRLMSISE-00) | when alt < 700 km | `use_spaceweather`, [`satproperties.cd_a_over_m`](../api/satprop.md) | $10^{-7}$ to $10^{-3}$ m/s² |
| Solar radiation pressure | when `craoverm > 0` | [`satproperties.craoverm`](../api/satprop.md) | $10^{-8}$ to $10^{-7}$ m/s² |
| Solid Earth tides (IERS 2010 §6.2.1 Step 1) | on | `tide_model` | $10^{-7}$ m/s² |
| GR Schwarzschild (IERS 2010 §10.3) | on | `use_relativistic_correction` | $10^{-9}$ m/s² (LEO) |
| Continuous thrust | when configured | [`satproperties.thrusts`](../api/satprop.md) | user-specified |

The [Forces-vs-altitude plot](#forces-vs-altitude) at the bottom of this page shows how each contribution scales with orbital altitude.

## Earth Gravity

Earth's gravity dominates by many orders of magnitude. The Earth is not a point mass; its non-spherical mass distribution is captured by an expansion in spherical harmonics with coefficients $\bar{C}_{nm}, \bar{S}_{nm}$:

$$
V(r, \phi, \lambda) = \frac{GM_\oplus}{r}\sum_{n=0}^{N}\left(\frac{R_\oplus}{r}\right)^n \sum_{m=0}^{n} \bar{P}_{nm}(\sin\phi)\left[\bar{C}_{nm}\cos m\lambda + \bar{S}_{nm}\sin m\lambda\right]
$$

The $\bar{C}_{20}$ term (commonly called **J2**) captures Earth's equatorial bulge and is responsible for orbital precession.

Coefficient files come from [ICGEM](https://icgem.gfz-potsdam.de/home). satkit ships with four models, selectable via `gravity_model`:

| Model | Description |
|---|---|
| `egm96` | Earth Gravitational Model 1996 (default) |
| `jgm3` | Joint Gravity Model 3 |
| `jgm2` | Joint Gravity Model 2 |
| `itugrace16` | ITU GRACE 2016 |

The `gravity_degree` and `gravity_order` parameters cap the expansion (default 4×4). For high-precision work, degree-8 to degree-20 is typical; gains beyond ~degree-20 are small for satellites above ~500 km. Order may be set lower than degree to zero out the longitudinal (tesseral) terms.

## Third-Body Gravity (Sun, Moon)

The Sun and Moon each act as point-mass attractors. Their pull on the *Earth* must be subtracted to express acceleration in the geocentric frame:

$$
\vec{a}_\text{sun}~=~GM_\text{sun}\left[\frac{\vec{p}_\text{sun} - \vec{p}}{|\vec{p}_\text{sun}-\vec{p}|^3} - \frac{\vec{p}_\text{sun}}{|\vec{p}_\text{sun}|^3}\right]
$$

The Moon expression has the same form. Body positions come from the JPL DE-series ephemerides (default DE440). Disable either with `use_sun_gravity=False` / `use_moon_gravity=False`.

## Solid Earth Tides

The Sun and Moon deform the solid body of the Earth, which in turn perturbs the geopotential. satkit implements **IERS 2010 §6.2.1 Step 1** (frequency-independent Love-number response) — about 99% of the total solid-tide signal at ~5% per-step CPU overhead. The correction modifies the degree-2, degree-3, and a small degree-4 set of Stokes coefficients $\Delta\bar{C}_{nm}, \Delta\bar{S}_{nm}$ as a function of Sun and Moon ITRF positions and the IERS 2010 nominal Love numbers.

Frequency-dependent corrections (IERS 2010 §6.2.2 Step 2, 71 tidal constituents) are reserved for `TideModel::SolidFull` — currently a placeholder that falls back to Step 1. The Step 2 contribution is sub-mm class at LEO.

Set with `tide_model`:

| Value | Effect |
|---|---|
| `tidemodel.solid_step1` *(default)* | IERS §6.2.1 Step 1 — recommended |
| `tidemodel.solid_full` | Step 1 + Step 2 (Step 2 not yet implemented; behaves as Step 1) |
| `tidemodel.none` | Disable solid tides (use for reproducibility with pre-tide versions) |

## Atmospheric Drag

At altitudes below ~600 km the residual atmosphere imposes a drag force:

$$
\vec{a}_\text{drag}~=~-\frac{1}{2}\,C_d\frac{A}{m}\,\rho\,|\vec{v}_r|\vec{v}_r
$$

with $C_d$ the drag coefficient (typically 1.5–3), $A/m$ the area-to-mass ratio, $\rho$ the atmospheric density, and $\vec{v}_r$ the satellite velocity relative to the co-rotating atmosphere (assumed at rest in the Earth-fixed frame).

Density comes from the [NRLMSISE-00](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~00/) thermosphere model (pure Rust implementation), which reads space-weather indices (F10.7, Ap) automatically. Disable space-weather lookup with `use_spaceweather=False` to fall back to fixed nominal indices.

Drag is skipped above 700 km regardless of settings.

## Solar Radiation Pressure

Solar photons absorbed or scattered by the satellite transfer momentum, producing a force away from the Sun:

$$
\vec{a}_\text{SRP}~=~-P_\text{sun}\,C_R\frac{A}{m}\,\hat{p}_\text{sun} \cdot \nu(\vec{p}, \vec{p}_\text{sun})
$$

where $P_\text{sun} \approx 4.56 \times 10^{-6}$ N/m² is the radiation pressure at 1 AU, $C_R A/m$ is the satellite's radiation susceptibility (the user-supplied `satproperties.craoverm`), and $\nu(\vec{p}, \vec{p}_\text{sun}) \in [0, 1]$ is a shadow function that vanishes when the satellite is in Earth's umbra.

satkit uses a **cannonball model** — the satellite's surface is treated as if its normal points toward the Sun. For high-fidelity work (precise SRP modeling for GNSS or active satellite operations), a [box-wing model](https://link.springer.com/article/10.1007/s10569-014-9583-2) is needed; that is not currently provided.

## General-Relativistic Correction (Schwarzschild)

In strong gravitational fields, the static post-Newtonian Schwarzschild correction perturbs the orbit:

$$
\vec{a}_\text{GR}~=~\frac{GM_\oplus}{c^2 r^3}\left[\left(\frac{4GM_\oplus}{r} - v^2\right)\vec{r} + 4(\vec{r}\cdot\vec{v})\vec{v}\right]
$$

(IERS 2010 §10.3 Eq. 10.12 with PPN parameters $\beta = \gamma = 1$.) At GPS altitude this contributes ~1 m/day of position drift if omitted; ~3 m/day at GEO. Lense-Thirring and de Sitter terms are not yet implemented (sub-cm-class at MEO/GEO).

Toggle with `use_relativistic_correction` (default `True`).

## Continuous Thrust

Constant acceleration in a chosen frame (used to model low-thrust maneuvers or — in orbit-determination contexts — *empirical accelerations*: a fitted catch-all that absorbs un-modeled physics). See [`satkit.thrust`](../api/satprop.md) and the [GPS Example tutorial](../tutorials/GPS Example.ipynb) for usage.

## Future Propagation

When propagating into the future beyond the date range of downloaded data files:

- **Earth Orientation Parameters** ($\Delta UT1$, $x_p$, $y_p$): the last available values are held constant. This is much more accurate than defaulting to zero since EOPs change slowly.
- **Space Weather** (F10.7 solar flux, Ap geomagnetic index): if historical data isn't available, the [NOAA/SWPC solar cycle forecast](https://www.swpc.noaa.gov/products/solar-cycle-progression) supplies predicted F10.7 (out ~5 years). Ap defaults to 4. If neither source is available, F10.7 = 150 and Ap = 4 are used. Run `satkit.utils.update_datafiles()` to refresh both.

## Forces vs Altitude

The plot below, modeled on a similar figure in Montenbruck and Gill, shows each force's order of magnitude vs orbital altitude:

![Acceleration vs Altitude](../images/force_vs_altitude.svg)

## See Also

- **Tutorial**: [GPS Example](../tutorials/GPS Example.ipynb) — fits a GPS orbit against ESA SP3 truth and walks through the empirical-acceleration concept.
- **Theory**: [ODE Integrators](integrators.md) for the integration mechanics; [State Vectors, STM & Covariance](satstate.md) for state representation and covariance propagation.
- **API**: [`satkit.propagate`](../api/satprop.md), [`satkit.propsettings`](../api/satprop.md), [`satkit.satproperties`](../api/satprop.md).
