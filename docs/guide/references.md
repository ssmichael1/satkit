# References

The models, algorithms and constants in `satkit` are taken from the sources
below. Pages on this site cite them as *(Author Year, §section)*, linking to
the entry here; each entry gives the section, equation or algorithm numbers the
code follows where that is useful.

## Books

<a id="vallado2013"></a>
- **Vallado, D. A. (2013).** *Fundamentals of Astrodynamics and Applications*,
  4th ed. Microcosm Press, Hawthorne, CA. ISBN 978-1881883180. Companion
  software and errata: <https://celestrak.org/software/vallado-sw.php>.
  Used for: SGP4 reference implementation (the C++ code satkit's port follows),
  GMST (Algorithm 15, Eq. 3-45), IAU-76/FK5 reduction (§3.7, Eqs. 3-88 to
  3-90), TEME (§3.7.3), RSW/NTW frames (§3.3, Eq. 3-31), Kepler's equation
  (Algorithm 2), Sun position (Algorithm 29, §5.1.1), sunrise/sunset
  (Algorithm 30, §5.3.1), Moon position (Algorithm 31, §5.2.3), Hohmann
  transfer (§6.3), Lambert background (Ch. 7), TDB−TT (Eq. 3-50).

<a id="montenbruck2000"></a>
- **Montenbruck, O., & Gill, E. (2000).** *Satellite Orbits: Models, Methods,
  Applications*. Springer, Berlin. <https://doi.org/10.1007/978-3-642-58351-3>.
  Used for: the force-model structure and the forces-vs-altitude figure
  (§3.1, Fig. 3.1), spherical-harmonic gravity (§3.2, Eqs. 3.28–3.33),
  third-body point-mass attraction (§3.3.1, Eq. 3.37), solar radiation pressure
  and the conical shadow model (§3.4, Eqs. 3.69–3.75, §3.4.2), atmospheric drag
  (§3.5), variational equations and the state transition matrix (§7.1–7.2,
  Eqs. 7.42, 7.75), covariance mapping (§8.1).

<a id="battin1999"></a>
- **Battin, R. H. (1999).** *An Introduction to the Mathematics and Methods of
  Astrodynamics*, revised ed. AIAA Education Series.
  <https://doi.org/10.2514/4.861543>. Used for: the hypergeometric-series
  form of the Lambert time-of-flight equation near the parabolic boundary.

<a id="hairer1996"></a>
- **Hairer, E., & Wanner, G. (1996).** *Solving Ordinary Differential Equations
  II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer.
  <https://doi.org/10.1007/978-3-642-05221-7>. Used for: the RODAS4 Rosenbrock
  integrator (§IV.7).

<a id="tapley2004"></a>
- **Tapley, B. D., Schutz, B. E., & Born, G. H. (2004).** *Statistical Orbit
  Determination*. Elsevier Academic Press.
  <https://doi.org/10.1016/B978-0-12-683630-1.X5019-X>. Used for: batch
  least-squares orbit determination and covariance propagation in the
  tutorials (§4.2–4.3).

<a id="markley2014"></a>
- **Markley, F. L., & Crassidis, J. L. (2014).** *Fundamentals of Spacecraft
  Attitude Determination and Control*. Springer.
  <https://doi.org/10.1007/978-1-4939-0802-8>. Used for: quaternion and
  local-vertical/local-horizontal frame conventions.

## Standards and conventions

<a id="petit2010"></a>
- **Petit, G., & Luzum, B. (eds.) (2010).** *IERS Conventions (2010)*. IERS
  Technical Note No. 36, Verlag des Bundesamts für Kartographie und Geodäsie,
  Frankfurt am Main. ISBN 3-89888-989-6.
  <https://iers-conventions.obspm.fr/content/tn36.pdf>. Used for: the
  ITRF↔GCRF reduction (Ch. 5: polar motion Eq. 5.3, Earth rotation angle
  Eq. 5.15, CIP X/Y and CIO locator series Tables 5.2a/5.2b/5.2d, frame bias
  §5.5.4 and Eq. 5.36), solid Earth tides (§6.2.1 Step 1, Eqs. 6.6–6.7,
  Table 6.3 Love numbers; §6.2.2 Step 2), time scales (Ch. 10, §10.1 TDB−TT),
  and the relativistic acceleration (§10.3, Eq. 10.12).

<a id="ccsds502"></a>
- **CCSDS (2023).** *Orbit Data Messages*. Recommended Standard CCSDS
  502.0-B-3 (Blue Book), Consultative Committee for Space Data Systems,
  April 2023. <https://ccsds.org/Pubs/502x0b3e1.pdf>. Used for: the Orbital
  Mean-Element Message (OMM) format and the RTN reference frame used for
  covariance and maneuver components.

<a id="nga2014"></a>
- **NGA (2014).** *Department of Defense World Geodetic System 1984: Its
  Definition and Relationships with Local Geodetic Systems*, Version 1.0.0,
  NGA.STND.0036_1.0.0_WGS84, National Geospatial-Intelligence Agency.
  <https://nsgreg.nga.mil/doc/view?i=4085>. Used for: the WGS 84 ellipsoid
  and defining parameters ($GM$, $\omega_\oplus$, $a$, $f$).

<a id="itu460"></a>
- **ITU-R (2002).** *Standard-frequency and time-signal emissions*,
  Recommendation ITU-R TF.460-6.
  <https://www.itu.int/rec/R-REC-TF.460-6-200202-I/en>. Used for: the
  definition of UTC and leap seconds.

<a id="bulletinc"></a>
- **IERS Bulletin C** — leap-second announcements (Earth Orientation Center,
  Observatoire de Paris). <https://hpiers.obspm.fr/iers/bul/bulc/bulletinc.dat>.
  Source of the TAI−UTC table compiled into `satkit`.

## Papers and reports

<a id="hoots1980"></a>
- **Hoots, F. R., & Roehrich, R. L. (1980).** *Models for Propagation of NORAD
  Element Sets*. Spacetrack Report No. 3, Aerospace Defense Command (reprinted
  by T. S. Kelso, 1988). <https://celestrak.org/NORAD/documentation/spacetrk.pdf>.
  The original SGP4/SDP4 description.

<a id="vallado2006"></a>
- **Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006).**
  "Revisiting Spacetrack Report #3." AIAA 2006-6753, AIAA/AAS Astrodynamics
  Specialist Conference, Keystone, CO. <https://doi.org/10.2514/6.2006-6753>.
  <https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf>.
  The modern SGP4 reference: the algorithm, the TEME frame, and the test
  vectors satkit's port is verified against.

<a id="vallado2008"></a>
- **Vallado, D. A., & Crawford, P. (2008).** "SGP4 Orbit Determination."
  AIAA 2008-6770, AIAA/AAS Astrodynamics Specialist Conference, Honolulu, HI.
  <https://doi.org/10.2514/6.2008-6770>. Used for: fitting TLEs to state
  vectors (`TLE.fit_from_states`).

<a id="picone2002"></a>
- **Picone, J. M., Hedin, A. E., Drob, D. P., & Aikin, A. C. (2002).**
  "NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons and
  scientific issues." *Journal of Geophysical Research: Space Physics*,
  107(A12), 1468. <https://doi.org/10.1029/2002JA009430>. satkit's density
  model is a Rust port of Dominik Brodowski's C implementation of NRLMSISE-00.

<a id="park2021"></a>
- **Park, R. S., Folkner, W. M., Williams, J. G., & Boggs, D. H. (2021).**
  "The JPL Planetary and Lunar Ephemerides DE440 and DE441." *The Astronomical
  Journal*, 161(3), 105. <https://doi.org/10.3847/1538-3881/abd414>.

<a id="folkner2009"></a>
- **Folkner, W. M., Williams, J. G., & Boggs, D. H. (2009).** "The Planetary
  and Lunar Ephemeris DE 421." IPN Progress Report 42-178, Jet Propulsion
  Laboratory. <https://ipnpr.jpl.nasa.gov/progress_report/42-178/178C.pdf>.

<a id="lemoine1998"></a>
- **Lemoine, F. G., et al. (1998).** *The Development of the Joint NASA GSFC
  and the National Imagery and Mapping Agency (NIMA) Geopotential Model
  EGM96*. NASA/TP-1998-206861. <https://ntrs.nasa.gov/citations/19980218814>.

<a id="tapley1996"></a>
- **Tapley, B. D., et al. (1996).** "The Joint Gravity Model 3." *Journal of
  Geophysical Research: Solid Earth*, 101(B12), 28029–28049.
  <https://doi.org/10.1029/96JB01645>.

<a id="nerem1994"></a>
- **Nerem, R. S., et al. (1994).** "Gravity model development for
  TOPEX/POSEIDON: Joint Gravity Models 1 and 2." *Journal of Geophysical
  Research: Oceans*, 99(C12), 24421–24447. <https://doi.org/10.1029/94JC01376>.

<a id="akyilmaz2016"></a>
- **Akyilmaz, O., et al. (2016).** *ITU_GRACE16: The global gravity field
  model including GRACE data up to degree and order 180 of ITU and other
  collaborating institutions*. GFZ Data Services.
  <https://doi.org/10.5880/icgem.2016.006>.

<a id="ince2019"></a>
- **Ince, E. S., et al. (2019).** "ICGEM – 15 years of successful collection
  and distribution of global gravitational models, associated services, and
  future plans." *Earth System Science Data*, 11, 647–674.
  <https://doi.org/10.5194/essd-11-647-2019>. The archive the gravity
  coefficient files are taken from: <https://icgem.gfz.de/>.

<a id="berry2004"></a>
- **Berry, M. M., & Healy, L. M. (2004).** "Implementation of Gauss-Jackson
  Integration for Orbit Propagation." *The Journal of the Astronautical
  Sciences*, 52(3), 331–357. <https://drum.lib.umd.edu/handle/1903/2202>.

<a id="verner2010"></a>
- **Verner, J. H. (2010).** "Numerically optimal Runge–Kutta pairs with
  interpolants." *Numerical Algorithms*, 53, 383–396.
  <https://doi.org/10.1007/s11075-009-9290-3>. Coefficient sets from
  <https://www.sfu.ca/~jverner/>: `RKV98.IIa.Efficient`, `RKV87.IIa.Robust`
  and `RKV65.IIIXb.Efficient` (via the `numeris` crate).

<a id="tsitouras2011"></a>
- **Tsitouras, Ch. (2011).** "Runge–Kutta pairs of order 5(4) satisfying only
  the first column simplifying assumption." *Computers & Mathematics with
  Applications*, 62(2), 770–775. <https://doi.org/10.1016/j.camwa.2011.06.002>.

<a id="izzo2015"></a>
- **Izzo, D. (2015).** "Revisiting Lambert's problem." *Celestial Mechanics
  and Dynamical Astronomy*, 121(1), 1–15.
  <https://doi.org/10.1007/s10569-014-9587-y>.

<a id="lancaster1969"></a>
- **Lancaster, E. R., & Blanchard, R. C. (1969).** *A Unified Form of
  Lambert's Theorem*. NASA Technical Note D-5368.
  <https://ntrs.nasa.gov/citations/19690027552>.

<a id="vincenty1975"></a>
- **Vincenty, T. (1975).** "Direct and Inverse Solutions of Geodesics on the
  Ellipsoid with Application of Nested Equations." *Survey Review*, 23(176),
  88–93. <https://doi.org/10.1179/sre.1975.23.176.88>.

<a id="bowring1976"></a>
- **Bowring, B. R. (1976).** "Transformation from Spatial to Geographical
  Coordinates." *Survey Review*, 23(181), 323–327.
  <https://doi.org/10.1179/sre.1976.23.181.323>. Used for: the
  Cartesian → geodetic conversion.

<a id="shoemake1985"></a>
- **Shoemake, K. (1985).** "Animating rotation with quaternion curves."
  *SIGGRAPH '85*, 245–254. <https://doi.org/10.1145/325334.325242>. Used for:
  spherical linear interpolation (SLERP).

<a id="clohessy1960"></a>
- **Clohessy, W. H., & Wiltshire, R. S. (1960).** "Terminal Guidance System
  for Satellite Rendezvous." *Journal of the Aerospace Sciences*, 27(9),
  653–658. <https://doi.org/10.2514/8.8704>. Origin of the RIC
  (radial / in-track / cross-track) relative-motion frame.

<a id="marquardt1963"></a>
- **Marquardt, D. W. (1963).** "An Algorithm for Least-Squares Estimation of
  Nonlinear Parameters." *Journal of the Society for Industrial and Applied
  Mathematics*, 11(2), 431–441. <https://doi.org/10.1137/0111030>.

<a id="nelder1965"></a>
- **Nelder, J. A., & Mead, R. (1965).** "A Simplex Method for Function
  Minimization." *The Computer Journal*, 7(4), 308–313.
  <https://doi.org/10.1093/comjnl/7.4.308>.

<a id="danby1987"></a>
- **Danby, J. M. A. (1987).** "The solution of Kepler's equation, III."
  *Celestial Mechanics*, 40, 303–312. <https://doi.org/10.1007/BF01235847>.
  Used for: the starting value of the Kepler-equation iteration.

<a id="wallace2006"></a>
- **Wallace, P. T., & Capitaine, N. (2006).** "Precession-nutation procedures
  consistent with IAU 2006 resolutions." *Astronomy & Astrophysics*, 459(3),
  981–985. <https://doi.org/10.1051/0004-6361:20065897>.

<a id="mathews2002"></a>
- **Mathews, P. M., Herring, T. A., & Buffett, B. A. (2002).** "Modeling of
  nutation and precession: New nutation series for nonrigid Earth and insights
  into the Earth's interior." *Journal of Geophysical Research: Solid Earth*,
  107(B4), 2068. <https://doi.org/10.1029/2001JB000390>. The IAU 2000A
  nutation model.

<a id="seidelmann1982"></a>
- **Seidelmann, P. K. (1982).** "1980 IAU Theory of Nutation: The final report
  of the IAU Working Group on Nutation." *Celestial Mechanics*, 27, 79–106.
  <https://doi.org/10.1007/BF01228952>. The nutation series used by the
  IAU-76/FK5 (`_approx`) reduction and by TEME.

<a id="charlot2020"></a>
- **Charlot, P., et al. (2020).** "The third realization of the International
  Celestial Reference Frame by very long baseline interferometry." *Astronomy
  & Astrophysics*, 644, A159. <https://doi.org/10.1051/0004-6361/202038368>.

<a id="altamimi2023"></a>
- **Altamimi, Z., Rebischung, P., Collilieux, X., Métivier, L., & Chanard, K.
  (2023).** "ITRF2020: an augmented reference frame refining the modeling of
  nonlinear station motions." *Journal of Geodesy*, 97, 47.
  <https://doi.org/10.1007/s00190-023-01738-w>.

<a id="standish"></a>
- **Standish, E. M., & Williams, J. G.** "Keplerian Elements for Approximate
  Positions of the Major Planets." JPL Solar System Dynamics.
  <https://ssd.jpl.nasa.gov/planets/approx_pos.html>. Used for: the
  low-precision planetary ephemerides (`satkit.planets`).

<a id="gmatspec"></a>
- **NASA Goddard Space Flight Center.** *General Mission Analysis Tool (GMAT)
  Mathematical Specifications*, distributed with GMAT (R2026A:
  `docs/GMATMathSpec.pdf`); an early draft is on NTRS at
  <https://ntrs.nasa.gov/citations/20080031744>. Used for: the GMAT
  configuration of the validation corpus (coordinate systems, §4.1.1 Table 4.1
  and §4.2.6 relativistic terms).

<a id="hughes2014"></a>
- **Hughes, S. P., Qureshi, R. H., Cooley, S. D., & Parker, J. J. (2014).**
  "Verification and Validation of the General Mission Analysis Tool (GMAT)."
  AIAA 2014-4151, AIAA/AAS Astrodynamics Specialist Conference.
  <https://doi.org/10.2514/6.2014-4151>.

<a id="rodriguez2012"></a>
- **Rodriguez-Solano, C. J., Hugentobler, U., & Steigenberger, P. (2012).**
  "Adjustable box-wing model for solar radiation pressure impacting GPS
  satellites." *Advances in Space Research*, 49(7), 1113–1128.
  <https://doi.org/10.1016/j.asr.2012.01.016>. The box-wing SRP model
  referred to (but not implemented) in the force-model guide.

<a id="beutler1994"></a>
- **Beutler, G., Brockmann, E., Gurtner, W., Hugentobler, U., Mervart, L.,
  Rothacher, M., & Verdun, A. (1994).** "Extended orbit modeling techniques at
  the CODE processing center of the International GPS Service for Geodynamics
  (IGS): theory and initial results." *Manuscripta Geodaetica*, 19, 367–386.
  The CODE empirical solar-radiation-pressure parameterization.

<a id="hilla2016"></a>
- **Hilla, S. (2016).** *The Extended Standard Product 3 Orbit Format (SP3-d)*.
  International GNSS Service. <https://files.igs.org/pub/data/format/sp3d.pdf>.
  The precise-orbit file format read in the GPS tutorials and tests.

## Data sources

<a id="celestrak-spacedata"></a>
- **CelesTrak Space Data** — <https://celestrak.org/SpaceData/>. Daily
  `EOP-All.csv` (Earth orientation parameters, from IERS Bulletin A /
  finals) and `SW-All.csv` (space weather: F10.7, Ap; from GFZ and NOAA).
- **NOAA/SWPC predicted solar cycle** —
  <https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json>,
  the F10.7 forecast used when propagating beyond the space-weather record.
- **NAIF generic kernels** —
  <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/>
  (`de440.bsp`, used by GMAT in the validation corpus). satkit itself reads
  JPL's binary DE files (`linux_p1550p2650.440`, `lnxp1900p2053.421`).
- **ICGEM** — <https://icgem.gfz.de/> (Ince et al. 2019): source of the
  `.gfc` gravity-coefficient files.
- **NRLMSISE-00 at CCMC** — <https://ccmc.gsfc.nasa.gov/models/NRLMSIS~00/>:
  reference implementation and documentation of the density model.

## Verification

The Rust and Python test suites check the implementation against:

- **SGP4** — the test vectors distributed with the reference C++ code of
  [Vallado et al. (2006)](#vallado2006).
- **JPL ephemerides** — JPL's `testpo` Chebyshev-interpolation test vectors
  for DE440/DE441 ([Park et al. 2021](#park2021)).
- **Frame transforms and Keplerian elements** — worked examples from
  [Vallado (2013)](#vallado2013).
- **Gravity** — reference values from [ICGEM](#ince2019).
- **Numerical propagation** — ESA/IGS precise GPS orbits in
  [SP3-d](#hilla2016) format, and the NASA GMAT reference corpus described in
  [Validation: GMAT Comparison](gmat_validation.md).
