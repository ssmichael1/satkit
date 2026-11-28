use super::*;
use crate::Duration;

/// Two segments, the first with accelerations-free lines and a covariance,
/// modelled on the examples of CCSDS 502.0-B-3.
const KVN: &str = "\
CCSDS_OEM_VERS = 2.0
COMMENT  header comment
CREATION_DATE = 1996-11-04T17:22:31
ORIGINATOR = NASA/JPL

META_START
COMMENT  metadata comment
OBJECT_NAME = MARS GLOBAL SURVEYOR
OBJECT_ID = 1996-062A
CENTER_NAME = MARS BARYCENTER
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 1996-12-18T12:00:00.331
USEABLE_START_TIME = 1996-12-18T12:10:00.331
USEABLE_STOP_TIME = 1996-12-28T21:23:00.331
STOP_TIME = 1996-12-28T21:28:00.331
INTERPOLATION = HERMITE
INTERPOLATION_DEGREE = 7
META_STOP

COMMENT  data comment
1996-12-18T12:00:00.331  2789.619 -280.045 -1746.755  4.73372 -2.49586 -1.04195
1996-12-18T12:01:00.331  2783.419 -308.143 -1877.071  5.18604 -2.42124 -1.99608
1996-12-18T12:02:00.331  2776.033 -336.859 -2008.682  5.63678 -2.33951 -1.94687
1996-12-28T21:28:00.331 -3881.024  563.959 -682.773  -3.28827 -3.66735  1.63861

COVARIANCE_START
COMMENT  covariance comment
EPOCH = 1996-12-28T21:29:07.267
COV_REF_FRAME = RTN
 3.3313494e-04
 4.6189273e-04  6.7824216e-04
-3.0700078e-04 -4.2212341e-04  3.2319319e-04
-3.3493650e-07 -4.6860842e-07  2.4849495e-07  4.2960228e-10
-2.2118325e-07 -2.8641868e-07  1.7980986e-07  2.6088992e-10  1.7675147e-10
-3.0413460e-07 -4.9894969e-07  3.5403109e-07  1.8692631e-10  1.0088625e-10  6.2244443e-10
EPOCH = 1996-12-29T21:00:00
 3.4424505e-04
 4.5078162e-04  6.8935327e-04
-3.0600067e-04 -4.1101230e-04  3.3420420e-04
-3.2382549e-07 -4.5750731e-07  2.3738384e-07  4.3071339e-10
-2.1007214e-07 -2.7530757e-07  1.6870875e-07  2.5077881e-10  1.8786258e-10
-3.0302350e-07 -4.8783858e-07  3.4302008e-07  1.7581520e-10  1.0077514e-10  6.2244443e-10
COVARIANCE_STOP

META_START
OBJECT_NAME = MARS GLOBAL SURVEYOR
OBJECT_ID = 1996-062A
CENTER_NAME = MARS BARYCENTER
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 1996-12-28T21:29:07.267
STOP_TIME = 1996-12-30T01:28:02.267
META_STOP

1996-12-28T21:29:07.267 -2432.166 -063.042 1742.754 7.33702 -3.495867 -1.041945  1.234 -2.345 3.455
1996-12-28T22:08:02.5   -2445.234 -878.141 1873.073 1.86043 -3.421256 -0.996366 -1.234  2.345 0.001
1996-12-30T01:28:02.267  2164.375 1115.811 -688.131 -3.53328 -2.88452 0.88535    0.0    0.0   0.0
";

#[test]
fn kvn_header_and_metadata() {
    let oem = OEM::from_text(KVN).unwrap();
    assert_eq!(oem.version, "2.0");
    assert_eq!(oem.originator, "NASA/JPL");
    assert_eq!(oem.comments, ["header comment"]);
    assert_eq!(
        oem.creation_date,
        Instant::from_datetime(1996, 11, 4, 17, 22, 31.0).unwrap()
    );
    assert!(oem.classification.is_none() && oem.message_id.is_none());
    assert_eq!(oem.segments.len(), 2);

    let m = &oem.segments[0].metadata;
    assert_eq!(m.comments, ["metadata comment"]);
    assert_eq!(m.object_name, "MARS GLOBAL SURVEYOR");
    assert_eq!(m.object_id, "1996-062A");
    assert_eq!(m.center_name, "MARS BARYCENTER");
    assert_eq!(m.ref_frame, "EME2000");
    assert_eq!(m.frame(), Some(Frame::EME2000));
    assert_eq!(m.time_system, "UTC");
    assert_eq!(m.time_scale, TimeScale::UTC);
    assert_eq!(
        m.start_time,
        Instant::from_datetime(1996, 12, 18, 12, 0, 0.331).unwrap()
    );
    assert_eq!(
        m.useable_start_time,
        Some(Instant::from_datetime(1996, 12, 18, 12, 10, 0.331).unwrap())
    );
    assert_eq!(m.interpolation.as_deref(), Some("HERMITE"));
    assert_eq!(m.interpolation_degree, Some(7));

    let m2 = &oem.segments[1].metadata;
    assert!(m2.useable_start_time.is_none() && m2.interpolation.is_none());
}

#[test]
fn kvn_states_are_si() {
    let oem = OEM::from_kvn_string(KVN).unwrap();
    let seg = &oem.segments[0];
    assert_eq!(seg.comments, ["data comment"]);
    assert_eq!(seg.states.len(), 4);
    let s = &seg.states[0];
    assert_eq!(s.time, seg.metadata.start_time);
    assert!((s.pos[0] - 2_789_619.0).abs() < 1e-6);
    assert!((s.pos[2] + 1_746_755.0).abs() < 1e-6);
    assert!((s.vel[1] + 2_495.86).abs() < 1e-9);
    assert!(s.acc.is_none());

    // Second segment carries accelerations
    let s = &oem.segments[1].states[0];
    let a = s.acc.unwrap();
    assert!((a[0] - 1234.0).abs() < 1e-9 && (a[1] + 2345.0).abs() < 1e-9);
}

#[test]
fn kvn_covariance() {
    let oem = OEM::from_kvn_string(KVN).unwrap();
    let covs = &oem.segments[0].covariances;
    assert_eq!(covs.len(), 2);
    let c = &covs[0];
    assert_eq!(c.comments, ["covariance comment"]);
    assert_eq!(c.ref_frame.as_deref(), Some("RTN"));
    assert_eq!(
        c.time,
        Instant::from_datetime(1996, 12, 28, 21, 29, 7.267).unwrap()
    );
    // km² → m²
    assert!((c.matrix[(0, 0)] - 3.3313494e2).abs() < 1e-9);
    assert!((c.matrix[(1, 0)] - 4.6189273e2).abs() < 1e-9);
    assert!((c.matrix[(5, 5)] - 6.2244443e-4).abs() < 1e-15);
    // (vz, vx) element: row 5, column 3
    assert!((c.matrix[(5, 3)] - 1.8692631e-4).abs() < 1e-15);
    for i in 0..6 {
        for j in 0..6 {
            assert_eq!(c.matrix[(i, j)], c.matrix[(j, i)]);
        }
    }
    assert!(covs[1].ref_frame.is_none());
    assert!(oem.segments[1].covariances.is_empty());
}

/// A circular two-body orbit sampled every `step` seconds, as KVN.
fn circular_kvn(step: f64, n: usize, extra_meta: &str, time_system: &str) -> String {
    let mut s = format!(
        "CCSDS_OEM_VERS = 2.0\nCREATION_DATE = 2024-001T00:00:00\nORIGINATOR = TEST\n\
         META_START\nOBJECT_NAME = CIRC\nOBJECT_ID = 2024-001A\nCENTER_NAME = EARTH\n\
         REF_FRAME = GCRF\nTIME_SYSTEM = {time_system}\nSTART_TIME = 2024-01-01T00:00:00\n\
         STOP_TIME = 2024-01-02T00:00:00\n{extra_meta}META_STOP\n"
    );
    let t0 = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
    for i in 0..n {
        let t = i as f64 * step;
        let (p, v) = circular(t);
        let epoch = t0 + Duration::from_seconds(t);
        let (y, mo, d, h, mi, sec) = epoch.as_datetime();
        s += &format!(
            "{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{sec:09.6} {:.9} {:.9} {:.9} {:.12} {:.12} {:.12}\n",
            p[0] / 1e3,
            p[1] / 1e3,
            p[2] / 1e3,
            v[0] / 1e3,
            v[1] / 1e3,
            v[2] / 1e3
        );
    }
    s
}

/// Position (m) and velocity (m/s) on the circular orbit `t` seconds in.
fn circular(t: f64) -> ([f64; 3], [f64; 3]) {
    let a = 7.0e6;
    let n = (crate::consts::MU_EARTH / (a * a * a)).sqrt();
    let (sin, cos) = (n * t).sin_cos();
    // Inclined so that all three components vary
    let (si, ci) = 0.9_f64.sin_cos();
    (
        [a * cos, a * sin * ci, a * sin * si],
        [-a * n * sin, a * n * cos * ci, a * n * cos * si],
    )
}

fn max_err(got: &Vector3, want: &[f64; 3]) -> f64 {
    (0..3).map(|k| (got[k] - want[k]).abs()).fold(0.0, f64::max)
}

#[test]
fn interp_matches_orbit() {
    let t0 = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
    for meta in [
        "INTERPOLATION = LAGRANGE\nINTERPOLATION_DEGREE = 7\n",
        "INTERPOLATION = Hermite\nINTERPOLATION_DEGREE = 7\n",
        // No keyword: Hermite, degree 7
        "",
    ] {
        let oem = OEM::from_text(&circular_kvn(60.0, 40, meta, "UTC")).unwrap();
        // Start, middle and end of the table, off the nodes
        for t in [0.0, 17.3, 30.0, 1111.1, 2330.5, 2340.0] {
            let (pos, vel) = oem.interp(&(t0 + Duration::from_seconds(t))).unwrap();
            let (p, v) = circular(t);
            assert!(max_err(&pos, &p) < 1e-3, "{meta:?} t={t} pos");
            assert!(max_err(&vel, &v) < 1e-5, "{meta:?} t={t} vel");
        }
    }
}

#[test]
fn interp_method_and_degree() {
    let oem = OEM::from_text(&circular_kvn(60.0, 10, "", "UTC")).unwrap();
    let seg = &oem.segments[0];
    assert_eq!(seg.interpolation().unwrap(), (Interpolation::Hermite, 7));
    let t0 = seg.metadata.start_time;
    let t = t0 + Duration::from_seconds(270.0);
    let (p, _) = circular(270.0);

    // Linear interpolation over a 60 s step is off by the chord sag
    let (lin, _) = seg.interp_with(&t, Interpolation::Lagrange, 1).unwrap();
    let sag = max_err(&lin, &p);
    assert!(sag > 100.0 && sag < 5.0e3, "sag {sag}");
    // Cubic Hermite uses the same two points and the velocities
    let (herm, _) = seg.interp_with(&t, Interpolation::Hermite, 3).unwrap();
    assert!(max_err(&herm, &p) < 1.0);
    // A degree larger than the table uses every point
    let (all, _) = seg.interp_with(&t, Interpolation::Lagrange, 99).unwrap();
    assert!(max_err(&all, &p) < 1e-3);

    // A node is returned exactly
    let (pos, vel) = seg.interp(&seg.states[3].time).unwrap();
    assert_eq!(pos, seg.states[3].pos);
    assert_eq!(vel, seg.states[3].vel);

    assert!(matches!(
        seg.interp_with(&t, Interpolation::Lagrange, 0),
        Err(Error::InvalidDegree)
    ));
    let linear = OEM::from_text(&circular_kvn(60.0, 10, "INTERPOLATION = LINEAR\n", "UTC"));
    let (lin2, _) = linear.unwrap().interp(&t).unwrap();
    assert_eq!(lin2, lin);
}

#[test]
fn interp_range_and_segments() {
    let oem = OEM::from_kvn_string(KVN).unwrap();
    let seg = &oem.segments[0];
    // The usable span is narrower than the data
    let (start, stop) = seg.usable_span();
    assert_eq!(Some(start), seg.metadata.useable_start_time);
    assert_eq!(Some(stop), seg.metadata.useable_stop_time);
    let early = seg.metadata.start_time + Duration::from_seconds(30.0);
    assert!(!seg.contains(&early));
    assert!(matches!(seg.interp(&early), Err(Error::OutOfRange { .. })));
    assert!(matches!(oem.interp(&early), Err(Error::OutOfRange { .. })));

    // A time in the second segment is interpolated there
    let t = oem.segments[1].states[1].time;
    assert!(std::ptr::eq(oem.segment_at(&t).unwrap(), &oem.segments[1]));
    assert_eq!(oem.interp(&t).unwrap().0, oem.segments[1].states[1].pos);

    // Between the first segment's usable stop and the second segment
    let gap = stop + Duration::from_seconds(60.0);
    assert!(oem.segment_at(&gap).is_none());
    let after = oem.segments[1].metadata.stop_time + Duration::from_seconds(1.0);
    match oem.interp(&after) {
        Err(Error::OutOfRange {
            start: s, stop: e, ..
        }) => {
            assert_eq!(s, start);
            assert_eq!(e, oem.segments[1].metadata.stop_time);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn unsupported_interpolation_is_an_error_only_when_used() {
    let text = circular_kvn(60.0, 10, "INTERPOLATION = SPLINE\n", "UTC");
    let oem = OEM::from_text(&text).unwrap();
    let t = oem.segments[0].states[2].time + Duration::from_seconds(5.0);
    assert!(matches!(
        oem.interp(&t),
        Err(Error::UnsupportedInterpolation(s)) if s == "SPLINE"
    ));
    assert!(oem.interp_with(&t, Interpolation::Hermite, 5).is_ok());
}

#[test]
fn time_systems() {
    let utc = OEM::from_text(&circular_kvn(60.0, 3, "", "UTC")).unwrap();
    let gps = OEM::from_text(&circular_kvn(60.0, 3, "", "GPS")).unwrap();
    let tai = OEM::from_text(&circular_kvn(60.0, 3, "", "TAI")).unwrap();
    let tt = OEM::from_text(&circular_kvn(60.0, 3, "", "TT")).unwrap();
    let t = |o: &OEM| o.segments[0].states[0].time;
    assert_eq!(gps.segments[0].metadata.time_scale, TimeScale::GPS);
    // In 2024 UTC = TAI - 37 s = GPS - 18 s: the same label in GPS time is
    // 18 s earlier
    assert_eq!((t(&utc) - t(&gps)).as_seconds(), 18.0);
    assert_eq!((t(&utc) - t(&tai)).as_seconds(), 37.0);
    assert_eq!((t(&utc) - t(&tt)).as_seconds(), 37.0 + 32.184);
    assert_eq!(gps.segments[0].metadata.start_time, t(&gps));
    // The header's CREATION_DATE is UTC whatever the segments use
    assert_eq!(gps.creation_date, utc.creation_date);

    let met = OEM::from_text(&circular_kvn(60.0, 3, "", "MET"));
    match met {
        Err(Error::Kvn { line, message }) => {
            assert_eq!(line, 4);
            assert!(message.contains("MET"), "{message}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn day_of_year_epochs() {
    let text = "CCSDS_OEM_VERS = 1.0\nCREATION_DATE = 2024-060T12:00:00Z\nORIGINATOR = X\n\
                META_START\nOBJECT_NAME = A\nOBJECT_ID = B\nCENTER_NAME = EARTH\n\
                REF_FRAME = ITRF2000\nTIME_SYSTEM = UTC\nSTART_TIME = 2024-060T00:00:00\n\
                STOP_TIME = 2024-060T00:00:00\nMETA_STOP\n\
                2024-060T00:00:00.5 1 2 3 4 5 6\n";
    let oem = OEM::from_text(text).unwrap();
    // 2024 is a leap year: day 60 is 29 February
    assert_eq!(
        oem.segments[0].states[0].time,
        Instant::from_datetime(2024, 2, 29, 0, 0, 0.5).unwrap()
    );
    assert_eq!(oem.segments[0].metadata.frame(), Some(Frame::ITRF));
    // One state: only its own epoch can be asked for
    let t = oem.segments[0].states[0].time;
    assert_eq!(oem.interp(&t).unwrap().0[2], 3000.0);
}

#[test]
fn frames() {
    let frame = |name: &str| {
        let text = circular_kvn(60.0, 2, "", "UTC").replace("GCRF", name);
        OEM::from_text(&text).unwrap().segments[0].metadata.frame()
    };
    assert_eq!(frame("GCRF"), Some(Frame::GCRF));
    assert_eq!(frame("ICRF"), Some(Frame::ICRF));
    assert_eq!(frame("TEME"), Some(Frame::TEME));
    assert_eq!(frame("ITRF-93"), Some(Frame::ITRF));
    assert_eq!(frame("TOD"), None);
}

fn kvn_error(text: &str) -> (usize, String) {
    match OEM::from_text(text) {
        Err(Error::Kvn { line, message }) => (line, message),
        other => panic!("expected a KVN error, got {other:?}"),
    }
}

#[test]
fn kvn_errors_name_the_line() {
    let good = circular_kvn(60.0, 3, "", "UTC");

    let (line, msg) = kvn_error(&good.replace("ORIGINATOR = TEST", "ORIGINATR = TEST"));
    assert_eq!(line, 3);
    assert!(msg.contains("ORIGINATR"));

    let (line, msg) = kvn_error(&good.replace("CENTER_NAME = EARTH\n", ""));
    assert_eq!(line, 4, "the META_START line");
    assert!(msg.contains("CENTER_NAME"), "{msg}");

    let (line, msg) = kvn_error(&good.replace("STOP_TIME = 2024-01-02", "STOP_TIME = 2024-13-02"));
    assert_eq!(line, 4);
    assert!(msg.contains("STOP_TIME"), "{msg}");

    // A short ephemeris line
    let bad = format!("{good}2024-01-01T01:00:00 1 2 3 4 5\n");
    let (line, msg) = kvn_error(&bad);
    assert_eq!(line, bad.lines().count());
    assert!(msg.contains("found 6 fields"), "{msg}");

    // A value that is not a number
    let bad = format!("{good}2024-01-01T01:00:00 1 2 x 4 5 6\n");
    assert!(kvn_error(&bad).1.contains("\"x\""));

    // Epochs must increase
    let bad = format!("{good}2024-01-01T00:00:30 1 2 3 4 5 6\n");
    let (line, msg) = kvn_error(&bad);
    assert_eq!(line, bad.lines().count());
    assert!(msg.contains("strictly increasing"), "{msg}");

    // A covariance with too few values
    let bad =
        format!("{good}COVARIANCE_START\nEPOCH = 2024-01-01T00:00:00\n1 2 3\nCOVARIANCE_STOP\n");
    let (line, msg) = kvn_error(&bad);
    assert_eq!(line, bad.lines().count() - 2, "the EPOCH line");
    assert!(msg.contains("3 values"), "{msg}");

    let (_, msg) = kvn_error(&format!("{good}META_START\nOBJECT_NAME = A\n"));
    assert!(msg.contains("META_STOP"));
}

#[test]
fn other_errors() {
    assert!(matches!(
        OEM::from_text("hello"),
        Err(Error::UnrecognizedFormat)
    ));
    assert!(matches!(OEM::from_text(""), Err(Error::UnrecognizedFormat)));
    // Header only
    assert!(matches!(
        OEM::from_text(
            "CCSDS_OEM_VERS = 2.0\nCREATION_DATE = 2024-01-01T00:00:00\nORIGINATOR = X\n"
        ),
        Err(Error::Empty("segments"))
    ));
    // A segment with no states
    let empty = circular_kvn(60.0, 0, "", "UTC");
    assert!(kvn_error(&empty).1.contains("no state vectors"));
    // No CREATION_DATE
    let text = circular_kvn(60.0, 2, "", "UTC").replace("CREATION_DATE = 2024-001T00:00:00\n", "");
    assert!(matches!(
        OEM::from_text(&text),
        Err(Error::MissingField("CREATION_DATE"))
    ));
    assert!(matches!(
        OEM::from_file("/nonexistent/satkit.oem"),
        Err(Error::Io(_))
    ));
}

#[test]
fn tolerates_crlf_bom_and_blank_lines() {
    let text = format!("\u{feff}\r\n{}", KVN.replace('\n', "\r\n"));
    assert_eq!(OEM::from_text(&text).unwrap(), OEM::from_text(KVN).unwrap());
}

#[cfg(feature = "omm-xml")]
mod xml {
    use super::*;

    const XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<oem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:noNamespaceSchemaLocation="http://sanaregistry.org/r/ndmxml/ndmxml-1.0-master.xsd"
     id="CCSDS_OEM_VERS" version="2.0">
  <header>
    <COMMENT>header comment</COMMENT>
    <CREATION_DATE>1996-11-04T17:22:31</CREATION_DATE>
    <ORIGINATOR>NASA/JPL</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>MARS GLOBAL SURVEYOR</OBJECT_NAME>
        <OBJECT_ID>1996-062A</OBJECT_ID>
        <CENTER_NAME>MARS BARYCENTER</CENTER_NAME>
        <REF_FRAME>EME2000</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <START_TIME>1996-12-18T12:00:00.331</START_TIME>
        <USEABLE_START_TIME>1996-12-18T12:10:00.331</USEABLE_START_TIME>
        <USEABLE_STOP_TIME>1996-12-28T21:23:00.331</USEABLE_STOP_TIME>
        <STOP_TIME>1996-12-28T21:28:00.331</STOP_TIME>
        <INTERPOLATION>HERMITE</INTERPOLATION>
        <INTERPOLATION_DEGREE>7</INTERPOLATION_DEGREE>
      </metadata>
      <data>
        <COMMENT>data comment</COMMENT>
        <stateVector>
          <EPOCH>1996-12-18T12:00:00.331</EPOCH>
          <X units="km">2789.619</X>
          <Y units="km">-280.045</Y>
          <Z units="km">-1746.755</Z>
          <X_DOT units="km/s">4.73372</X_DOT>
          <Y_DOT units="km/s">-2.49586</Y_DOT>
          <Z_DOT units="km/s">-1.04195</Z_DOT>
        </stateVector>
        <stateVector>
          <EPOCH>1996-12-18T12:01:00.331</EPOCH>
          <X>2783.419</X>
          <Y>-308.143</Y>
          <Z>-1877.071</Z>
          <X_DOT>5.18604</X_DOT>
          <Y_DOT>-2.42124</Y_DOT>
          <Z_DOT>-1.99608</Z_DOT>
          <X_DDOT>0.001</X_DDOT>
          <Y_DDOT>0.002</Y_DDOT>
          <Z_DDOT>0.003</Z_DDOT>
        </stateVector>
        <covarianceMatrix>
          <COMMENT>covariance comment</COMMENT>
          <EPOCH>1996-12-28T21:29:07.267</EPOCH>
          <COV_REF_FRAME>RTN</COV_REF_FRAME>
          <CX_X units="km**2">3.3313494e-04</CX_X>
          <CY_X>4.6189273e-04</CY_X>
          <CY_Y>6.7824216e-04</CY_Y>
          <CZ_X>-3.0700078e-04</CZ_X>
          <CZ_Y>-4.2212341e-04</CZ_Y>
          <CZ_Z>3.2319319e-04</CZ_Z>
          <CX_DOT_X>-3.3493650e-07</CX_DOT_X>
          <CX_DOT_Y>-4.6860842e-07</CX_DOT_Y>
          <CX_DOT_Z>2.4849495e-07</CX_DOT_Z>
          <CX_DOT_X_DOT>4.2960228e-10</CX_DOT_X_DOT>
          <CY_DOT_X>-2.2118325e-07</CY_DOT_X>
          <CY_DOT_Y>-2.8641868e-07</CY_DOT_Y>
          <CY_DOT_Z>1.7980986e-07</CY_DOT_Z>
          <CY_DOT_X_DOT>2.6088992e-10</CY_DOT_X_DOT>
          <CY_DOT_Y_DOT>1.7675147e-10</CY_DOT_Y_DOT>
          <CZ_DOT_X>-3.0413460e-07</CZ_DOT_X>
          <CZ_DOT_Y>-4.9894969e-07</CZ_DOT_Y>
          <CZ_DOT_Z>3.5403109e-07</CZ_DOT_Z>
          <CZ_DOT_X_DOT>1.8692631e-10</CZ_DOT_X_DOT>
          <CZ_DOT_Y_DOT>1.0088625e-10</CZ_DOT_Y_DOT>
          <CZ_DOT_Z_DOT>6.2244443e-10</CZ_DOT_Z_DOT>
        </covarianceMatrix>
      </data>
    </segment>
  </body>
</oem>
"#;

    #[test]
    fn xml_matches_kvn() {
        let xml = OEM::from_text(XML).unwrap();
        let kvn = OEM::from_text(KVN).unwrap();
        assert_eq!(xml.version, "2.0");
        assert_eq!(xml.comments, kvn.comments);
        assert_eq!(xml.creation_date, kvn.creation_date);
        assert_eq!(xml.originator, kvn.originator);
        assert_eq!(xml.segments.len(), 1);
        let (x, k) = (&xml.segments[0], &kvn.segments[0]);
        assert_eq!(x.metadata.comments, Vec::<String>::new());
        assert_eq!(x.metadata.object_name, k.metadata.object_name);
        assert_eq!(x.metadata.start_time, k.metadata.start_time);
        assert_eq!(x.metadata.useable_stop_time, k.metadata.useable_stop_time);
        assert_eq!(x.metadata.interpolation_degree, Some(7));
        assert_eq!(x.comments, k.comments);
        assert_eq!(x.states.len(), 2);
        assert_eq!(x.states[0], k.states[0]);
        assert_eq!(x.states[1].pos, k.states[1].pos);
        let a = x.states[1].acc.unwrap();
        assert!((a[2] - 3.0).abs() < 1e-12);
        assert_eq!(x.covariances, vec![k.covariances[0].clone()]);
    }

    #[test]
    fn ndm_wrapper() {
        let body = XML.split_once("?>").unwrap().1;
        let one = format!("<ndm>{body}</ndm>");
        assert_eq!(OEM::from_text(&one).unwrap(), OEM::from_text(XML).unwrap());
        let two = format!("<ndm>{body}{body}</ndm>");
        assert!(matches!(
            OEM::from_text(&two),
            Err(Error::InvalidField { field: "ndm", .. })
        ));
        assert!(matches!(
            OEM::from_text("<ndm></ndm>"),
            Err(Error::Empty(_))
        ));
    }

    #[test]
    fn xml_errors() {
        let missing = XML.replace("<Y units=\"km\">-280.045</Y>", "");
        assert!(matches!(
            OEM::from_text(&missing),
            Err(Error::MissingField("Y"))
        ));
        let bad = XML.replace("<CY_Y>6.7824216e-04</CY_Y>", "");
        assert!(matches!(
            OEM::from_text(&bad),
            Err(Error::MissingField("CY_Y"))
        ));
        let bad = XML.replace("<X>2783.419</X>", "<X>abc</X>");
        assert!(matches!(
            OEM::from_text(&bad),
            Err(Error::InvalidField { field: "X", .. })
        ));
        assert!(matches!(OEM::from_text("<oem>"), Err(Error::Xml(_))));
    }
}

mod write {
    use super::*;
    use crate::orbitprop::{propagate, PropSettings, PropagationResult, SimpleState};

    fn assert_close(a: &OEM, b: &OEM) {
        assert_eq!(a.version, b.version);
        assert_eq!(a.comments, b.comments);
        assert_eq!(a.creation_date, b.creation_date);
        assert_eq!(a.originator, b.originator);
        assert_eq!(a.segments.len(), b.segments.len());
        for (x, y) in a.segments.iter().zip(&b.segments) {
            assert_eq!(x.metadata, y.metadata);
            assert_eq!(x.comments, y.comments);
            assert_eq!(x.states.len(), y.states.len());
            for (s, t) in x.states.iter().zip(&y.states) {
                assert_eq!(s.time, t.time);
                assert_eq!(s.acc.is_some(), t.acc.is_some());
                for k in 0..3 {
                    assert!((s.pos[k] - t.pos[k]).abs() < 1e-6);
                    assert!((s.vel[k] - t.vel[k]).abs() < 1e-9);
                    if let (Some(p), Some(q)) = (&s.acc, &t.acc) {
                        assert!((p[k] - q[k]).abs() < 1e-11);
                    }
                }
            }
            assert_eq!(x.covariances.len(), y.covariances.len());
            for (c, d) in x.covariances.iter().zip(&y.covariances) {
                assert_eq!(c.time, d.time);
                assert_eq!(c.ref_frame, d.ref_frame);
                assert_eq!(c.comments, d.comments);
                for i in 0..6 {
                    for j in 0..6 {
                        let (p, q) = (c.matrix[(i, j)], d.matrix[(i, j)]);
                        assert!((p - q).abs() <= 1e-13 * q.abs());
                    }
                }
            }
        }
    }

    #[test]
    fn kvn_round_trip() {
        let oem = OEM::from_kvn_string(KVN).unwrap();
        let text = oem.to_kvn_string().unwrap();
        let back = OEM::from_text(&text).unwrap();
        assert_close(&back, &oem);
        // Writing what was read back changes nothing
        assert_eq!(back.to_kvn_string().unwrap(), text);

        assert!(text.starts_with("CCSDS_OEM_VERS = 2.0\nCOMMENT header comment\n"));
        assert!(text.contains("CREATION_DATE = 1996-11-04T17:22:31.000000\n"));
        assert!(text.contains("USEABLE_START_TIME = 1996-12-18T12:10:00.331000\n"));
        assert!(text.contains("INTERPOLATION = HERMITE\nINTERPOLATION_DEGREE = 7\n"));
        // Kilometers, and nine values where there is an acceleration
        assert!(text.contains(
            "\n1996-12-18T12:00:00.331000 2789.619000000 -280.045000000 -1746.755000000 \
             4.733720000000 -2.495860000000 -1.041950000000\n"
        ));
        let acc_line = text
            .lines()
            .find(|l| l.starts_with("1996-12-28T21:29:07.267000"))
            .unwrap();
        assert_eq!(acc_line.split_whitespace().count(), 10);
        assert_eq!(text.matches("COVARIANCE_START").count(), 1);
        assert_eq!(text.matches("COV_REF_FRAME = RTN").count(), 1);
    }

    #[test]
    fn epochs_are_written_in_the_time_system() {
        let oem = OEM::from_kvn_string(&circular_kvn(60.0, 5, "", "GPS")).unwrap();
        let text = oem.to_kvn_string().unwrap();
        assert!(text.contains("TIME_SYSTEM = GPS\nSTART_TIME = 2024-01-01T00:00:00.000000\n"));
        assert!(text.contains("\n2024-01-01T00:01:00.000000 "));
        assert_close(&OEM::from_text(&text).unwrap(), &oem);
    }

    #[test]
    fn file_round_trip() {
        let oem = OEM::from_kvn_string(KVN).unwrap();
        let path = std::env::temp_dir().join(format!("satkit_oem_{}.oem", std::process::id()));
        oem.to_file(&path).unwrap();
        let back = OEM::from_file(&path);
        std::fs::remove_file(&path).unwrap();
        assert_close(&back.unwrap(), &oem);
    }

    fn states(n: usize) -> Vec<State> {
        let t0 = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
        (0..n)
            .map(|i| {
                let t = i as f64 * 60.0;
                let (p, v) = circular(t);
                State {
                    time: t0 + Duration::from_seconds(t),
                    pos: numeris::vector![p[0], p[1], p[2]],
                    vel: numeris::vector![v[0], v[1], v[2]],
                    acc: None,
                }
            })
            .collect()
    }

    #[test]
    fn built_from_states() {
        let seg = Segment::from_states("CIRC", "2024-001A", "TEME", states(4)).unwrap();
        let m = &seg.metadata;
        assert_eq!(m.center_name, "EARTH");
        assert_eq!(m.time_scale, TimeScale::UTC);
        assert_eq!(m.frame(), Some(Frame::TEME));
        assert_eq!(m.start_time, seg.states[0].time);
        assert_eq!(m.stop_time, seg.states[3].time);
        let oem = OEM::new("TEST", vec![seg]).unwrap();
        assert_eq!(oem.version, "2.0");
        let back = OEM::from_text(&oem.to_kvn_string().unwrap()).unwrap();
        // The creation date is written to the microsecond it is held to
        assert_close(&back, &oem);

        assert!(matches!(
            Segment::from_states("A", "B", "GCRF", Vec::new()),
            Err(Error::Empty(_))
        ));
        let mut reversed = states(3);
        reversed.reverse();
        assert!(matches!(
            Segment::from_states("A", "B", "GCRF", reversed),
            Err(Error::EpochOrder(_))
        ));
        assert!(matches!(OEM::new("TEST", Vec::new()), Err(Error::Empty(_))));
    }

    #[test]
    fn write_errors() {
        let good = OEM::new(
            "TEST",
            vec![Segment::from_states("CIRC", "2024-001A", "GCRF", states(3)).unwrap()],
        )
        .unwrap();
        assert!(good.to_kvn_string().is_ok());

        let mut oem = good.clone();
        oem.segments[0].metadata.object_name = "  ".to_string();
        assert!(matches!(
            oem.to_kvn_string(),
            Err(Error::MissingField("OBJECT_NAME"))
        ));

        // The label and the scale the epochs are written in must agree
        let mut oem = good.clone();
        oem.segments[0].metadata.time_system = "TAI".to_string();
        assert!(matches!(
            oem.to_kvn_string(),
            Err(Error::InvalidField {
                field: "TIME_SYSTEM",
                ..
            })
        ));
        oem.segments[0].metadata.time_scale = TimeScale::TAI;
        assert!(oem
            .to_kvn_string()
            .unwrap()
            .contains("START_TIME = 2024-01-01T00:00:37.000000\n"));

        let mut oem = good.clone();
        oem.comments.push("two\nlines".to_string());
        assert!(matches!(
            oem.to_kvn_string(),
            Err(Error::InvalidField {
                field: "COMMENT",
                ..
            })
        ));

        let mut oem = good.clone();
        oem.segments[0].states[1].vel[2] = f64::NAN;
        assert!(matches!(
            oem.to_kvn_string(),
            Err(Error::InvalidField { .. })
        ));

        let mut oem = good.clone();
        oem.segments[0].states.swap(0, 1);
        assert!(matches!(oem.to_kvn_string(), Err(Error::EpochOrder(_))));

        let mut oem = good;
        oem.segments.clear();
        assert!(matches!(oem.to_kvn_string(), Err(Error::Empty(_))));
    }

    fn propagated(minutes: f64, enable_interp: bool) -> PropagationResult<1> {
        let start = Instant::from_datetime(2024, 1, 1, 0, 0, 0.0).unwrap();
        let mut state = SimpleState::zeros();
        state[0] = 7.0e6;
        state[4] = (crate::consts::MU_EARTH / 7.0e6).sqrt() * 0.8;
        state[5] = (crate::consts::MU_EARTH / 7.0e6).sqrt() * 0.6;
        let settings = PropSettings {
            gravity_degree: 4,
            enable_interp,
            ..Default::default()
        };
        let stop = start + Duration::from_minutes(minutes);
        propagate(&state, &start, &stop, &settings, None).unwrap()
    }

    #[test]
    fn from_propagation() {
        let res = propagated(30.0, true);
        let step = Duration::from_seconds(60.0);
        let oem = OEM::from_propagation(&res, step, "SAT", "2024-001A", "TEST").unwrap();
        assert_eq!(oem.originator, "TEST");
        assert_eq!(oem.segments.len(), 1);
        let seg = &oem.segments[0];
        let m = &seg.metadata;
        assert_eq!(
            (m.object_name.as_str(), m.object_id.as_str()),
            ("SAT", "2024-001A")
        );
        assert_eq!(m.frame(), Some(Frame::GCRF));
        assert_eq!(m.time_scale, TimeScale::UTC);
        assert_eq!((m.start_time, m.stop_time), (res.time_begin, res.time_end));
        assert_eq!(seg.interpolation().unwrap(), (Interpolation::Hermite, 7));
        assert_eq!(seg.states.len(), 31);
        // The ends are the propagation's own states
        let last = &seg.states[30];
        for k in 0..3 {
            assert!((seg.states[0].pos[k] - res.state_begin[k]).abs() < 1e-6);
            assert!((last.pos[k] - res.state_end[k]).abs() < 1e-6);
            assert!((last.vel[k] - res.state_end[k + 3]).abs() < 1e-9);
        }

        // Read back from text, the table interpolates to the trajectory
        let back = OEM::from_text(&oem.to_kvn_string().unwrap()).unwrap();
        for secs in [17.0, 451.5, 1234.5, 1790.0] {
            let t = res.time_begin + Duration::from_seconds(secs);
            let want = res.interp(&t).unwrap();
            let (pos, vel) = back.interp(&t).unwrap();
            for k in 0..3 {
                assert!((pos[k] - want[k]).abs() < 1e-3, "pos {secs}");
                assert!((vel[k] - want[k + 3]).abs() < 1e-4, "vel {secs}");
            }
        }
    }

    #[test]
    fn from_propagation_sampling() {
        let res = propagated(30.0, true);
        let offsets = |oem: &OEM| -> Vec<f64> {
            oem.segments[0]
                .states
                .iter()
                .map(|s| (s.time - res.time_begin).as_seconds())
                .collect()
        };
        // A step that does not divide the span: the end is still there
        let step = Duration::from_minutes(7.0);
        let oem = OEM::from_propagation(&res, step, "S", "I", "O").unwrap();
        assert_eq!(offsets(&oem), [0.0, 420.0, 840.0, 1260.0, 1680.0, 1800.0]);
        // A step longer than the span: the two ends
        let oem = OEM::from_propagation(&res, Duration::from_days(1.0), "S", "I", "O").unwrap();
        assert_eq!(offsets(&oem), [0.0, 1800.0]);

        for step in [Duration::from_seconds(0.0), Duration::from_seconds(-60.0)] {
            assert!(matches!(
                OEM::from_propagation(&res, step, "S", "I", "O"),
                Err(Error::InvalidStep)
            ));
        }
        // No dense output, nothing to sample
        let step = Duration::from_seconds(60.0);
        assert!(matches!(
            OEM::from_propagation(&propagated(30.0, false), step, "S", "I", "O"),
            Err(Error::Propagation(_))
        ));
        // A zero-duration propagation is its one state
        let oem = OEM::from_propagation(&propagated(0.0, true), step, "S", "I", "O").unwrap();
        assert_eq!(oem.segments[0].states.len(), 1);
    }

    #[test]
    fn from_backward_propagation() {
        let res = propagated(-30.0, true);
        assert!(res.time_end < res.time_begin);
        let step = Duration::from_minutes(7.0);
        let oem = OEM::from_propagation(&res, step, "S", "I", "O").unwrap();
        let seg = &oem.segments[0];
        assert_eq!(seg.states.len(), 6);
        assert_eq!(seg.metadata.start_time, res.time_end);
        assert_eq!(seg.metadata.stop_time, res.time_begin);
        let (first, last) = (&seg.states[0], &seg.states[5]);
        for k in 0..3 {
            assert!((first.pos[k] - res.state_end[k]).abs() < 1e-6);
            assert!((last.pos[k] - res.state_begin[k]).abs() < 1e-6);
        }
    }
}
