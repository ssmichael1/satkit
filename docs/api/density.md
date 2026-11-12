# Atmospheric Density (NRLMSISE-00)

Density and temperature from the NRLMSISE-00 model ([Picone et al. 2002](../guide/references.md#picone2002)), the model the propagator uses for drag. There are two entry points, which return the same `(density in kg/m³, temperature in K)` and differ only in their units:

| call | altitude | latitude, longitude |
|---|---|---|
| `satkit.nrlmsise00(alt_km, latitude_deg=..., longitude_deg=...)` | km | degrees |
| `satkit.density.nrlmsise(altitude_meters, latitude_rad, longitude_rad)`, or `nrlmsise(itrfcoord)` | m | radians |

Given a time, both read the solar flux and geomagnetic indices for it from the [space-weather table](spaceweather.md); with no time they use $F_{10.7} = F_{10.7A} = 150$ and $A_p = 4$. See the [Atmospheric Density](../tutorials/Atmospheric%20Density.ipynb) tutorial and the [Force Model](../guide/forces.md) guide.

::: satkit.density

::: satkit.nrlmsise00
