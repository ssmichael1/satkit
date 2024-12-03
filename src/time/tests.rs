use super::Duration;
use super::Instant;

#[test]
fn test_j2000() {
    let g = Instant::J2000.as_datetime();
    assert!(g.0 == 2000);
    assert!(g.1 == 1);
    assert!(g.2 == 1);
    assert!(g.3 == 12);
    assert!(g.4 == 0);
    // J2000 is TT time, which is 32.184 seconds
    assert!((g.5 - 32.184).abs() < 1.0e-7);
}

#[test]
fn test_fromstring() {
    let time = Instant::from_string("March 4 2024").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 3);
    assert!(g.2 == 4);

    let time = Instant::from_string("2024-01-04 13:14:12.123000").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 1);
    assert!(g.2 == 4);
    assert!(g.3 == 13);
    assert!(g.4 == 14);
    assert!((g.5 - 12.123).abs() < 1.0e-7);
}

#[test]
fn test_unixtime() {
    let time = Instant::from_unixtime(1732939013.0);
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 30);
    assert!(g.3 == 3);
    assert!(g.4 == 56);
    assert!(g.5 == 53.0);

    let time = Instant::from_datetime(2016, 12, 31, 23, 59, 40.0);
    assert!(time.as_unixtime() == 1483228780.0);

    let g = time.as_datetime();
    assert!(g.0 == 2016);
    assert!(g.1 == 12);
    assert!(g.2 == 31);
    assert!(g.3 == 23);
    assert!(g.4 == 59);
    assert!(g.5 == 40.0);
}

#[test]
fn test_leapsecond() {
    // Beginning of leap second
    let mut t = Instant::new(1483228836000000);
    let g = t.as_datetime();
    assert!(g.0 == 2016);
    assert!(g.1 == 12);
    assert!(g.2 == 31);
    assert!(g.3 == 23);
    assert!(g.4 == 59);
    assert!(g.5 == 60.0);

    // Middle of a leap second
    let t2 = t + Duration::from_microseconds(100);
    let g = t2.as_datetime();
    assert!(g.0 == 2016);
    assert!(g.1 == 12);
    assert!(g.2 == 31);
    assert!(g.3 == 23);
    assert!(g.4 == 59);
    assert!((g.5 - 60.0001).abs() < 1.0e-7);

    // Just prior to leap second
    t -= Duration::from_seconds(1.0);
    let g = t.as_datetime();
    assert!(g.0 == 2016);
    assert!(g.1 == 12);
    assert!(g.2 == 31);
    assert!(g.3 == 23);
    assert!(g.4 == 59);
    assert!(g.5 == 59.0);

    // Just after leap second
    t += Duration::from_seconds(2.0);
    let g = t.as_datetime();
    assert!(g.0 == 2017);
    assert!(g.1 == 1);
    assert!(g.2 == 1);
    assert!(g.3 == 0);
    assert!(g.4 == 0);
    assert!(g.5 == 0.0);
}

#[test]
fn test_ops() {
    let t1 = Instant::from_datetime(2024, 11, 13, 8, 0, 3.0);
    let t2 = Instant::from_datetime(2024, 11, 13, 8, 0, 4.0);
    let dt = t2 - t1;
    assert!(dt.as_microseconds() == 1_000_000);
    let t2 = Instant::from_datetime(2024, 11, 13, 8, 0, 2.0);
    let dt = t2 - t1;
    assert!(dt.as_microseconds() == -1_000_000);
    let t2 = Instant::from_datetime(2024, 11, 13, 8, 1, 3.0);
    let dt = t2 - t1;
    assert!(dt.as_microseconds() == 60_000_000);

    let t3 = t2 + Duration::from_days(1.0);
    let g = t3.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 14);
    assert!(g.3 == 8);
    assert!(g.4 == 1);
    assert!(g.5 == 3.0);
}

#[test]
fn test_gps() {
    let g = Instant::GPS_EPOCH.as_datetime();
    assert!(g.0 == 1980);
    assert!(g.1 == 1);
    assert!(g.2 == 6);
    assert!(g.3 == 0);
    assert!(g.4 == 0);
    assert!(g.5 == 0.0);
}

#[test]
fn test_jd() {
    let time = Instant::from_datetime(2024, 11, 24, 12, 0, 0.0);
    assert!(time.as_jd() == 2_460_639.0);
    assert!(time.as_mjd() == 60_638.5);
}

#[test]
fn test_rfc3339() {
    let time = Instant::from_rfc3339("2024-11-24T12:03:45.123456Z").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 24);
    assert!(g.3 == 12);
    assert!(g.4 == 3);
    assert!(g.5 == 45.123456);

    let time = Instant::from_rfc3339("2024-11-24T12:03:45Z").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 24);
    assert!(g.3 == 12);
    assert!(g.4 == 3);
    assert!(g.5 == 45.0);

    // Test with milliseconds
    let time = Instant::from_rfc3339("2024-11-24T12:03:45.123Z").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 24);
    assert!(g.3 == 12);
    assert!(g.4 == 3);
    assert!(g.5 == 45.123);
}

#[test]
fn test_strptime() {
    let time = Instant::strptime("2024-11-24T12:03:45.123456", "%Y-%m-%dT%H:%M:%S.%f").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 24);
    assert!(g.3 == 12);
    assert!(g.4 == 3);
    assert!(g.5 == 45.123456);

    // Test with milliseconds
    let time = Instant::strptime("2024-11-24T12:03:45.123", "%Y-%m-%dT%H:%M:%S.%f").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 11);
    assert!(g.2 == 24);
    assert!(g.3 == 12);
    assert!(g.4 == 3);
    assert!(g.5 == 45.123);

    let time =
        Instant::strptime("February 13 2024 12:03:45.123456", "%B %d %Y %H:%M:%S.%f").unwrap();

    let g = time.as_datetime();
    assert!(g.0 == 2024);
    assert!(g.1 == 2);
    assert!(g.2 == 13);
    assert!(g.3 == 12);
    assert!(g.4 == 3);
    assert!(g.5 == 45.123456);

    let time = Instant::strptime("09-Jun-2023 22:27:19", "%d-%b-%Y %H:%M:%S").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2023);
    assert!(g.1 == 6);
    assert!(g.2 == 9);
    assert!(g.3 == 22);
    assert!(g.4 == 27);
    assert!(g.5 == 19.0);
}
