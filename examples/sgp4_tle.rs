//! SGP4 with two-line element sets and OMMs: parsing, propagation, ground
//! track and look angles, and fitting a TLE to a set of states.
//!
//! SGP4 itself needs no data files. The TEME -> ITRF / GCRF rotations use
//! Earth orientation parameters (`finals2000A.all`, downloaded on first use);
//! without them satkit warns once and uses zeros.
//!
//! ```text
//! cargo run --example sgp4_tle
//! ```

use satkit::frametransform::{qteme2itrf, transform_state};
use satkit::omm::OMM;
use satkit::sgp4::{sgp4, SGP4Error};
use satkit::{Duration, Frame, ITRFCoord, Instant, Vector3, TLE};

/// A catalog snippet, as downloaded from CelesTrak or Space-Track
const CATALOG: &str = "\
ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009
2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299350
SHINSEI (MS-F2)
1  5485U 71080A   24324.43728894  .00000099  00000-0  13784-3 0  9992
2  5485  32.0564  70.0187 0639723 198.9447 158.6281 12.74214074476065
";

/// The same kind of element set as an Orbit Mean-Elements Message (JSON)
const OMM_JSON: &str = r#"[{
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "NORAD_CAT_ID": 25544,
    "EPOCH": "2024-01-01T12:00:00.000000",
    "MEAN_MOTION": 15.4891533,
    "ECCENTRICITY": 0.0007417,
    "INCLINATION": 51.6432,
    "RA_OF_ASC_NODE": 351.4697,
    "ARG_OF_PERICENTER": 130.5364,
    "MEAN_ANOMALY": 329.6482,
    "BSTAR": 0.0001027,
    "MEAN_MOTION_DOT": 0.00016717,
    "MEAN_MOTION_DDOT": 0.0
}]"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse every record; `check_checksums` also verifies column 69.
    // (`TLE::from_lines` does the same and stops at the first bad record.)
    let tles: Vec<TLE> = TLE::records(CATALOG.lines())
        .check_checksums(true)
        .collect::<Result<_, _>>()?;
    for tle in &tles {
        println!(
            "{:<16} #{:05}  epoch {}  incl {:.2} deg",
            tle.name, tle.sat_num, tle.epoch, tle.inclination
        );
    }
    let mut iss = tles[0].clone();

    // SGP4 takes a slice of times and returns TEME position (m) and
    // velocity (m/s) as 3xN matrices, plus an error code per time
    let times: Vec<Instant> = (0..=6)
        .map(|i| iss.epoch + Duration::from_minutes(15.0 * i as f64))
        .collect();
    let states = sgp4(&mut iss, &times)?;

    // Ground track: rotate TEME to ITRF and convert to geodetic
    println!("\ntime (UTC)                   lat (deg)  lon (deg)  alt (km)");
    for (i, t) in times.iter().enumerate() {
        if states.errcode[i] != SGP4Error::SGP4Success {
            println!("{t}  SGP4 error: {}", states.errcode[i]);
            continue;
        }
        let p_teme = Vector3::from_slice(states.pos.col_slice(i));
        let sat = ITRFCoord::from(qteme2itrf(t) * p_teme);
        println!(
            "{t}  {:9.3}  {:9.3}  {:8.1}",
            sat.latitude_deg(),
            sat.longitude_deg(),
            sat.hae() / 1e3
        );
    }

    // Passes over a ground station: elevation every 30 s for a day
    let station = ITRFCoord::from_geodetic_deg(42.3601, -71.0589, 20.0);
    let scan: Vec<Instant> = (0..2880)
        .map(|i| iss.epoch + Duration::from_seconds(30.0 * i as f64))
        .collect();
    let scan_states = sgp4(&mut iss, &scan)?;
    let elevation = |i: usize| {
        let p_teme = Vector3::from_slice(scan_states.pos.col_slice(i));
        let enu = ITRFCoord::from(qteme2itrf(&scan[i]) * p_teme).to_enu(&station);
        (enu[2] / enu.norm()).asin().to_degrees()
    };
    let elev: Vec<f64> = (0..scan.len()).map(elevation).collect();
    println!(
        "
passes above 10 deg elevation over the next day:"
    );
    let mut i = 0;
    while i < elev.len() {
        if elev[i] < 10.0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < elev.len() && elev[i] >= 10.0 {
            i += 1;
        }
        let (imax, emax) = (start..i)
            .map(|k| (k, elev[k]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        println!(
            "  {} to {}, max {emax:.1} deg at {}",
            scan[start].strftime("%Y-%m-%d %H:%M:%S")?,
            scan[i - 1].strftime("%H:%M:%S")?,
            scan[imax].strftime("%H:%M:%S")?
        );
    }

    // An OMM works wherever a TLE does (both implement `SGP4Source`)
    let mut omm = OMM::from_json_string(OMM_JSON)?.remove(0);
    let from_omm = sgp4(&mut omm, &times[..1])?;
    let from_tle = sgp4(&mut iss, &times[..1])?;
    let diff = Vector3::from_slice(from_omm.pos.col_slice(0))
        - Vector3::from_slice(from_tle.pos.col_slice(0));
    println!("\nOMM vs TLE at epoch: {:.3} m apart", diff.norm());

    // Fit a TLE to GCRF states. Here the states come from SGP4 itself, so
    // the fit should recover the original elements; in practice they come
    // from a precise propagation or from GNSS.
    let fit_times: Vec<Instant> = (0..=288)
        .map(|i| iss.epoch + Duration::from_minutes(5.0 * i as f64))
        .collect();
    let teme = sgp4(&mut iss, &fit_times)?;
    let states_gcrf = fit_times
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let p = Vector3::from_slice(teme.pos.col_slice(i));
            let v = Vector3::from_slice(teme.vel.col_slice(i));
            let (p, v) = transform_state(Frame::TEME, Frame::GCRF, t, &p, &v)?;
            Ok([p[0], p[1], p[2], v[0], v[1], v[2]])
        })
        .collect::<Result<Vec<[f64; 6]>, satkit::frametransform::Error>>()?;
    let (mut fitted, result) = TLE::fit_from_states(&states_gcrf, &fit_times, iss.epoch)?;
    // The fit returns the orbital elements only; copy the catalog fields
    fitted.name = iss.name.clone();
    fitted.sat_num = iss.sat_num;
    fitted.intl_desig = iss.intl_desig.clone();
    println!(
        "\nfit: {} after {} iterations, residual norm {:.3} m",
        result.status, result.n_iter, result.best_norm
    );
    println!(
        "  mean motion {:.8} rev/day (was {:.8}), bstar {:.4e} (was {:.4e})",
        fitted.mean_motion, iss.mean_motion, fitted.bstar, iss.bstar
    );
    let [line1, line2] = fitted.to_2line()?;
    println!("  {line1}\n  {line2}");

    Ok(())
}
