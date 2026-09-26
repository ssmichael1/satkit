//! The public ephemeris API evaluates JPL's Chebyshev series at TDB.
//!
//! JPL ephemerides are tabulated in T_eph (TDB for practical purposes).
//! Evaluating them at TT instead is an easy regression to make and a hard
//! one to see: TDB − TT is at most ~1.7 ms, which moves the Moon by about
//! 1.7 m. The in-source `testvecs` test builds the ephemeris argument
//! directly and so never goes through [`Instant`]; this test does,
//! through the public `geocentric_pos`, against reference values taken
//! from JPL's own `testpo.440` (target 10 = Moon, centre 3 = Earth, which is
//! exactly what `geocentric_pos(SolarSystem::Moon, _)` returns).
//!
//! It needs only the default DE440 file (`linux_p1550p2650.440`, present
//! wherever the rest of the ephemeris tests run), not the test vectors, so
//! it runs in every CI job with a data directory.

use satkit::jplephem;
use satkit::{Instant, SolarSystem, TimeScale};

/// IAU 2012 astronomical unit, the value DE440 and its testpo use.
const AU_M: f64 = 149_597_870_700.0;

/// Rows of `testpo.440`: (JD in TDB, coordinate 1..=3, Moon − Earth in AU).
/// Chosen where |TDB − TT| is large (0.8 – 1.7 ms), spread over the file's
/// span, one axis each. The digits are copied verbatim from the file.
#[allow(clippy::excessive_precision)]
const MOON_ROWS: &[(f64, usize, f64)] = &[
    (2341697.5, 2, 0.00094906092695138126),  // 1699-04-01
    (2431334.5, 1, 0.00165357943672120311),  // 1944-09-01
    (2445731.5, 1, 0.00136846460924871334),  // 1984-02-01
    (2465179.5, 2, -0.00197978511614090892), // 2037-05-01
    (2468770.5, 2, 0.00100280754984753695),  // 2047-03-01
    (2543859.5, 3, 0.00045959765657421184),  // 2252-10-01
];

/// Through an `Instant` the TDB epoch is rounded to a microsecond, about a
/// millimetre of lunar motion; the TT/TDB mix-up is metres.
const TOL_M: f64 = 0.01;

#[test]
fn geocentric_moon_is_evaluated_at_tdb() {
    let mut discriminating = 0;
    for &(jd, coord, truth_au) in MOON_ROWS {
        let t = Instant::from_jd_with_scale(jd, TimeScale::TDB);
        let pos = jplephem::geocentric_pos(SolarSystem::Moon, &t).unwrap_or_else(|e| {
            panic!(
                "geocentric_pos(Moon) failed ({e}); this test needs the default DE440 \
                 file linux_p1550p2650.440 in the satkit data directory"
            )
        });
        let err_m = (pos[coord - 1] / AU_M - truth_au).abs() * AU_M;
        assert!(
            err_m < TOL_M,
            "JD {jd} (TDB) axis {coord}: geocentric Moon is {err_m:.4} m off JPL's testpo.440 \
             (tolerance {TOL_M} m); is the ephemeris being evaluated at TT instead of TDB?"
        );

        // The same label read as TT is 0.8 – 1.7 ms away; check that this
        // row would actually catch the mix-up.
        let t_tt = Instant::from_jd_with_scale(jd, TimeScale::TT);
        let pos_tt = jplephem::geocentric_pos(SolarSystem::Moon, &t_tt).unwrap();
        let err_tt_m = (pos_tt[coord - 1] / AU_M - truth_au).abs() * AU_M;
        eprintln!("JD {jd} axis {coord}: at TDB {err_m:.2e} m, at TT {err_tt_m:.3} m");
        if err_tt_m > 10.0 * TOL_M {
            discriminating += 1;
        }
    }
    assert_eq!(
        discriminating,
        MOON_ROWS.len(),
        "every reference row must separate TT from TDB by at least {} m",
        10.0 * TOL_M
    );
}
