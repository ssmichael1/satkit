//! Integration test for installing an EOP table from values
//! (`set_table`) and going back to the data-directory one (`reset`). A
//! separate test binary: both replace the process-wide table.

use satkit::earth_orientation_params::{self as eop, EopRecord, EopStatus};
use satkit::Instant;

#[test]
fn set_table_then_reset() {
    let r0 = EopRecord {
        mjd_utc: 60000.0,
        dut1: -0.0174,
        xp: 0.0563,
        yp: 0.3522,
        lod: 0.0007,
        dx: 0.2,
        dy: -0.1,
        observed: true,
    };
    // Rows need not be a day apart.
    let r1 = EopRecord {
        mjd_utc: 60002.0,
        dut1: -0.0188,
        xp: 0.0539,
        yp: 0.3516,
        lod: 0.0009,
        dx: 0.0,
        dy: 0.0,
        observed: true,
    };
    let r2 = EopRecord {
        mjd_utc: 60003.0,
        observed: false,
        ..r1
    };
    eop::set_table(&[r0, r1, r2]).expect("valid records");

    let v = eop::eop_from_mjd_utc(60000.5).expect("inside the table");
    let expect = [-0.01775, 0.0557, 0.35205, 0.00075, 0.15, -0.075];
    for (a, b) in v.iter().zip(expect) {
        assert!((a - b).abs() < 1e-12, "{v:?}");
    }
    let cov = eop::coverage().expect("coverage of the installed table");
    assert_eq!(cov.first, Instant::from_mjd_utc(60000.0));
    assert_eq!(cov.last_observed, Instant::from_mjd_utc(60002.0));
    assert_eq!(cov.last, Instant::from_mjd_utc(60003.0));
    assert_eq!(
        eop::status(&Instant::from_mjd_utc(59999.0)),
        EopStatus::BeforeTable
    );

    // An invalid table is refused and leaves the installed one in place.
    assert!(matches!(
        eop::set_table(&[r1, r0]),
        Err(eop::Error::InvalidTable(_))
    ));
    assert_eq!(eop::coverage(), Some(cov));

    // Frame transforms read the installed table.
    let t = Instant::from_mjd_utc(60000.5);
    let q_custom = satkit::frametransform::qitrf2gcrf(&t);

    match eop::reset() {
        Ok(()) => {
            let cov = eop::coverage().expect("default table after reset");
            assert!(cov.first < Instant::from_mjd_utc(42000.0), "{}", cov.first);
            assert_ne!(satkit::frametransform::qitrf2gcrf(&t), q_custom);
        }
        // No finals2000A.all and no network: the custom table is still gone.
        Err(e) => {
            eprintln!("reset could not load the default table ({e}); checking the cleared state");
            assert!(eop::coverage().is_none());
            assert_eq!(eop::status(&t), EopStatus::NotLoaded);
        }
    }
}
