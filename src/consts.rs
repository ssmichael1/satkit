#[allow(non_upper_case_globals)]
/// WGS 84 semi-major axis, in meters.
///
/// Source: NGA, *World Geodetic System 1984*, defining parameters:
/// <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const WGS84_A: f64 = 6_378_137.0;

/// WGS 84 flattening, dimensionless (`1 / 298.257223563`).
///
/// Source: NGA, *World Geodetic System 1984*, defining parameters:
/// <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// WGS 84 equatorial Earth radius, in meters (alias of [`WGS84_A`]).
///
/// Source: NGA, *World Geodetic System 1984*, defining parameters:
/// <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const EARTH_RADIUS: f64 = WGS84_A;

/// WGS 84 geocentric gravitational parameter of Earth (atmosphere included), in m^3/s^2.
///
/// Source: NGA, *World Geodetic System 1984*, defining parameters:
/// <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const MU_EARTH: f64 = 3.986_004_418e14;

/// Gravitational parameter of the Moon, in m^3/s^2.
///
/// Source: JPL DE440, Park et al. (2021), as tabulated by JPL Solar System Dynamics:
/// <https://ssd.jpl.nasa.gov/astro_par.html>
pub const MU_MOON: f64 = 4.902_800_118e12;

/// Heliocentric gravitational parameter of the Sun, in m^3/s^2.
///
/// Source: JPL DE440, Park et al. (2021), as tabulated by JPL Solar System Dynamics:
/// <https://ssd.jpl.nasa.gov/astro_par.html>
pub const MU_SUN: f64 = 1.327_124_400_412_794_2e20;

/// Alternate name for the WGS 84 gravitational parameter of Earth.
///
/// Source: alias of [`MU_EARTH`]; NGA WGS 84 defining parameters:
/// <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const GM: f64 = MU_EARTH;

/// WGS 84 nominal mean angular velocity of Earth, in rad/s.
///
/// Source: NGA, *World Geodetic System 1984*, defining parameters:
/// <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const OMEGA_EARTH: f64 = 7.292_115e-5;

/// Speed of light in vacuum, in m/s (exact SI defining constant).
///
/// Source: BIPM, *The International System of Units (SI)*:
/// <https://www.bipm.org/en/measurement-units>
pub const C: f64 = 299_792_458.0;

/// Astronomical unit, in meters (exact).
///
/// Source: IAU 2012 Resolution B1:
/// <https://www.iau.org/static/resolutions/IAU2012_English.pdf>
pub const AU: f64 = 149_597_870_700.0;

/// IAU nominal solar photospheric radius, in meters (exact nominal conversion constant).
///
/// Source: IAU 2015 Resolution B3:
/// <https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf>
pub const SUN_RADIUS: f64 = 695_700_000.0;

/// IAU lunar reference-sphere radius, in meters.
///
/// Source: IAU WGCCRE value, documented by the USGS Lunar Data Interoperability Standard:
/// <https://psdi.astrogeology.usgs.gov/moon/standards/data_standards/>
pub const MOON_RADIUS: f64 = 1_737_400.0;

/// Earth-to-Moon mass ratio, dimensionless.
///
/// Source: ratio of the DE440 Earth and Moon gravitational parameters tabulated by JPL
/// (`398600.435507 / 4902.800118`): <https://ssd.jpl.nasa.gov/astro_par.html>
pub const EARTH_MOON_MASS_RATIO: f64 = 81.300_568_229_079_9;

/// Geosynchronous circular-orbit radius, in meters.
///
/// Derived as `(MU_EARTH / OMEGA_EARTH^2)^(1/3)` from the NGA WGS 84 defining
/// parameters: <https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84>
pub const GEO_R: f64 = 42_164_172.931_157_24;

/// JGM-3 gravitational parameter of Earth, in m^3/s^2.
///
/// Source: JGM-3 model, Tapley et al. (1996), *The Joint Gravity Model 3*:
/// <https://doi.org/10.1029/96JB01645>
pub const JGM3_MU: f64 = 3.986_004_415e14;

/// JGM-3 reference semi-major axis of Earth, in meters.
///
/// Source: JGM-3 model, Tapley et al. (1996), *The Joint Gravity Model 3*:
/// <https://doi.org/10.1029/96JB01645>
pub const JGM3_A: f64 = 6_378_136.3;

/// JGM-3 positive, unnormalized degree-two zonal coefficient `J2`, dimensionless.
///
/// The corresponding normalized Stokes coefficient is negative:
/// `J2 = -sqrt(5) * Cbar20`. Source: JGM-3 model, Tapley et al. (1996),
/// *The Joint Gravity Model 3*: <https://doi.org/10.1029/96JB01645>
pub const JGM3_J2: f64 = 0.001_082_636_022_982_994_5;
