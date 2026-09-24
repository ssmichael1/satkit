# Third-party data embedded in satkit

satkit compiles a few small datasets into the library so that frames and
gravity work with no data directory and no network (`data/embedded/*.gz`,
inflated on first use). They are **not** covered by the MIT / Apache-2.0
licence of the satkit source code; their own terms and attributions are
below. `data/embedded/SOURCES.json` records the SHA-256 of every original
file and of the embedded copy. Larger files — the JPL DE440/DE421
ephemerides and the Earth-orientation and space-weather tables — are
downloaded on demand, not embedded; their sources and licences are listed in
`data/README.md`. So is the optional ITU_GRACE16 gravity model (Akyilmaz et
al. 2016, GFZ Data Services, CC BY 4.0): it is fetched, unmodified and with
its attribution header, only when `gravmodel.itugrace16` is selected, and is
not part of the library or its packages. Everything compiled in is a US
Government work or an IERS table, freely redistributable.

## EGM2008 gravity model — US Government work

- Pavlis, N.K., Holmes, S.A., Kenyon, S.C., & Factor, J.K. (2012): *The
  development and evaluation of the Earth Gravitational Model 2008
  (EGM2008)*, J. Geophys. Res. 117, B04406. National Geospatial-Intelligence
  Agency (NGA).
- Distributed by ICGEM. A work of the United States Government, not subject
  to copyright (public domain).
- Modification: truncated from degree/order 2190 to degree/order 70.

## EGM96 gravity model — US Government work

- Lemoine, F.G., et al. (1998): *The Development of the Joint NASA GSFC and
  NIMA Geopotential Model EGM96*, NASA/TP-1998-206861. NASA Goddard Space
  Flight Center / National Imagery and Mapping Agency.
- Distributed by ICGEM. A work of the United States Government, not subject
  to copyright (public domain).
- Modification: truncated from degree/order 360 to degree/order 70.

## JGM-2 and JGM-3 gravity models — US Government work

- JGM-2: Nerem, R.S., et al. (1994), *Gravity model development for
  TOPEX/POSEIDON: Joint Gravity Models 1 and 2*, J. Geophys. Res. 99(C12).
- JGM-3: Tapley, B.D., et al. (1996), *The Joint Gravity Model 3*,
  J. Geophys. Res. 101(B12).
- NASA Goddard Space Flight Center / University of Texas Center for Space
  Research; distributed by ICGEM. US Government work (public domain).
- Embedded unmodified (both models are complete to degree 70).

## IERS Conventions (2010) precession-nutation tables

- Tables 5.2a, 5.2b and 5.2d (the X, Y and s + XY/2 series of the IAU
  2006/2000A precession-nutation model) from Petit, G. and Luzum, B. (eds.),
  *IERS Conventions (2010)*, IERS Technical Note No. 36, Frankfurt am Main:
  Verlag des Bundesamts für Kartographie und Geodäsie, 2010.
- International Earth Rotation and Reference Systems Service (IERS); freely
  redistributable. Embedded unmodified.
