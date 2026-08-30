# Empirical SRP: ECOM

The [cannonball](forces.md#solar-radiation-pressure) is satkit's default solar-radiation-pressure model. For GNSS-class precise orbits the empirical **CODE Orbit Model (ECOM)** is available as an addition to it: a small set of constant and harmonic accelerations in a Sun-oriented frame whose coefficients are *estimated* in orbit determination rather than derived from a surface model.


!!! warning "Experimental"
    The ECOM interface (`EcomParams` / `ecomparams`, `satproperties(ecom=...)`, `SatProperties::srp_ecom`) is new and marked **experimental**: it may be reshaped in a minor release (for example into a general empirical-acceleration hook) as it finds users. The physics and conventions below are stable; the API surface is not yet covered by the usual compatibility promise.

## Model

ECOM is the empirical SRP parameterization used by CODE and most IGS analysis centres for GNSS precise orbit determination ([Beutler et al. 1994](references.md#beutler1994); [Springer et al. 1999](references.md#springer1999); [Arnold et al. 2015](references.md#arnold2015)). It expresses the non-gravitational acceleration of a nominally yaw-steering satellite in a Sun-oriented **DYB** frame:

$$
\hat{e}_D = \frac{\vec{r}_\odot - \vec{r}}{|\vec{r}_\odot - \vec{r}|}, \qquad
\hat{e}_Y = \frac{\hat{e}_D \times \hat{r}}{|\hat{e}_D \times \hat{r}|}, \qquad
\hat{e}_B = \hat{e}_D \times \hat{e}_Y
$$

($\hat{e}_D$ points from the satellite **to** the Sun; $\hat{e}_Y$ is the solar-panel rotation axis.) The acceleration is

$$
\vec{a}_\text{ECOM} = \nu \left[ D(\varphi)\,\hat{e}_D + Y(\varphi)\,\hat{e}_Y + B(\varphi)\,\hat{e}_B \right]
$$

$$
\begin{aligned}
D(\varphi) &= D_0 + D_c\cos\varphi + D_s\sin\varphi + D_{2c}\cos 2\varphi + D_{2s}\sin 2\varphi + D_{4c}\cos 4\varphi + D_{4s}\sin 4\varphi \\
Y(\varphi) &= Y_0 + Y_c\cos\varphi + Y_s\sin\varphi \\
B(\varphi) &= B_0 + B_c\cos\varphi + B_s\sin\varphi
\end{aligned}
$$

## Coefficients

| Coefficient | Axis | Term | In which model | Typical size (GPS, nm/s²) | Field name |
|---|---|---|---|---|---|
| $D_0$ | $\hat{e}_D$ (toward the Sun) | constant | all | $-80$ to $-110$ (≈ $-P_\odot C_R A/m$; negative because $\hat{e}_D$ points at the Sun) | `d0` |
| $Y_0$ | $\hat{e}_Y$ (solar-panel axis) | constant | all | $\sim 1$ (attitude/thermal "Y-bias") | `y0` |
| $B_0$ | $\hat{e}_B$ | constant | all | $\sim 1$–$5$, varies with $\beta$ | `b0` |
| $D_c, D_s$ | $\hat{e}_D$ | $\cos\varphi, \sin\varphi$ | ECOM1 only ($\varphi = u$) | $\lesssim 1$ | `dc`, `ds` |
| $Y_c, Y_s$ | $\hat{e}_Y$ | $\cos\varphi, \sin\varphi$ | ECOM1 only | $\lesssim 1$ | `yc`, `ys` |
| $B_c, B_s$ | $\hat{e}_B$ | $\cos\varphi, \sin\varphi$ | reduced ECOM and ECOM1 ($\varphi = u$); ECOM2 as $B_{1c}, B_{1s}$ ($\varphi = \Delta u$) | $\lesssim 2$ | `bc`, `bs` |
| $D_{2c}, D_{2s}$ | $\hat{e}_D$ | $\cos 2\Delta u, \sin 2\Delta u$ | ECOM2 | few, mostly in eclipse seasons | `d2c`, `d2s` |
| $D_{4c}, D_{4s}$ | $\hat{e}_D$ | $\cos 4\Delta u, \sin 4\Delta u$ | ECOM2 | few, mostly in eclipse seasons | `d4c`, `d4s` |

The constructors set the fields for you: `reduced(d0, y0, b0, bc, bs)`, `ecom1(d0, y0, b0, dc, ds, yc, ys, bc, bs)` and `ecom2(d0, y0, b0, b1c, b1s, d2c, d2s, d4c, d4s)` (ECOM2's $B_{1c}, B_{1s}$ are stored in `bc`, `bs`; `ecom2` sets `sun_relative=True`). Any coefficient left at zero costs nothing, so a 7-parameter ECOM2 is `ecom2(..., d4c=0, d4s=0)`. All values are accelerations in m/s².

## Conventions

where $\nu$ is the same shadow function as the cannonball term, applied to all three axes — the CODE/Bernese convention ("the acceleration due to the solar radiation pressure is switched off when the satellite is in the Earth's shadow", [Bernese GNSS Software v5.2](references.md#dach2015) §2.2.2.3), so coefficients taken from CODE products keep their meaning. The argument $\varphi$ is selected by `sun_relative`:

| `sun_relative` | $\varphi$ | Model family |
|---|---|---|
| `False` | argument of latitude $u$ from the ascending node (x-axis projection for an equatorial orbit) | ECOM1 ([Beutler 1994](references.md#beutler1994)), reduced ECOM ([Springer 1999](references.md#springer1999)) |
| `True` | $\Delta u = u - u_\odot$ measured from *orbit noon* — zero at the point closest to the Sun's projection into the orbit plane, $\pi$ at midnight; computed node-free and regular at all inclinations | ECOM2 ([Arnold 2015](references.md#arnold2015)) |

Because $\hat{e}_D$ points at the Sun, the physical $D_0$ is **negative** — about $-1\times10^{-7}$ m/s² for a GPS satellite ($C_R A/m \approx 0.02$ m²/kg), and 10–30 nm/s² when ECOM is applied as a residual on top of an a-priori model. $Y_0$ and the B terms are typically $\sim10^{-9}$ m/s². The coefficients are *estimated* in orbit determination; satkit propagates with the values you supply and adds the ECOM term to the cannonball, so use `craoverm=0` for a pure ECOM model:

## Usage

```python
import satkit as sk

# Reduced ECOM (D0, Y0, B0, Bc, Bs) in m/s^2 — e.g. from your own fit or a CODE product
ecom = sk.ecomparams.reduced(-1.06e-7, 1.0e-9, -3.2e-9, 1.2e-9, 0.3e-9)
props = sk.satproperties(craoverm=0.0, ecom=ecom)
res = sk.propagate(state, t0, t1, propsettings=settings, satproperties=props)

# ECOM2: D0, Y0, B0, B1c, B1s, D2c, D2s, D4c, D4s (argument Δu from orbit noon)
ecom2 = sk.ecomparams.ecom2(-1.06e-7, 1e-9, -3e-9, 0, 0, -2e-9, -3.6e-9, 1.9e-9, -0.7e-9)
```

Rust users can also implement `SatProperties::srp_ecom(&self, tm, state) -> Option<EcomParams>` to supply coefficients that change over the propagation (per-arc CODE values, an attitude-mode switch). Like the cannonball term, ECOM contributes no partials to the state transition matrix.

## What to expect

 Fitting an initial state plus the reduced 5-parameter ECOM to 3 days of IGS final GPS orbits (`python/examples/ecom_gps_validation.py`, 12×12 gravity, no a-priori box-wing) reproduces the orbit at the accuracy of the IGS product itself — 5 cm 3D RMS versus 3.8 m for the cannonball — and predicts the next 24 h to a median 6–7 cm 3D across the constellation (2-day fits, ten satellites), against ~5 cm for the IGS ultra-rapid predicted product and 8–10 cm reported by [Duan & Hugentobler (2021)](references.md#duan2021) from 3-day arcs with a full analysis-centre force model. Beyond a few days the error grows along-track roughly as $t^2$: ~0.8 m at 7 days, 10 m after ~12 days and ~100 m at 30 days from a 3-day fit (~15 days and ~60 m from a 7-day fit), because the true coefficients drift with the Sun elevation angle β over weeks — which is exactly why analysis centres re-estimate them every day. Constant ECOM coefficients are a short-arc (days) model, not a month-long one. The [ECOM Solar Radiation Pressure](../tutorials/ECOM Solar Radiation Pressure.ipynb) tutorial walks through the fit, the prediction and the constellation benchmark with plots.

## Practical notes

 SP3 epochs are in GPS time (`scale=timescale.GPS`), not UTC — reading them as UTC rotates the truth by 18 s of Earth rotation relative to the Sun/Moon geometry and quadruples the fit residual. And for arcs that cross Earth's shadow, use `integrator = gauss_jackson8`: the adaptive Runge–Kutta steppers can abort at a shadow boundary with *too many consecutive step rejections*, whereas the fixed-step multistep integrator is immune and fits an eclipsing satellite just as well (G08, 8% umbra: 4.7 cm fit, 5.6 cm at 24 h).

## See Also

- **Theory**: [Force Model](forces.md) — the cannonball term, shadow function and every other force this adds to.
- **Tutorial**: [ECOM Solar Radiation Pressure](../tutorials/ECOM%20Solar%20Radiation%20Pressure.ipynb) — fit, 30-day prediction and constellation benchmark against IGS orbits.
- **API**: [`satkit.ecomparams`](../api/satprop.md#satkit.ecomparams), [`satkit.satproperties`](../api/satprop.md).
- **References**: [Beutler et al. 1994](references.md#beutler1994), [Springer et al. 1999](references.md#springer1999), [Arnold et al. 2015](references.md#arnold2015), [Dach et al. 2015](references.md#dach2015).
