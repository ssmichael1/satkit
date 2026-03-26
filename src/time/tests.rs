use super::Duration;
use super::Instant;

#[test]
fn test_j2000() {
    let g = Instant::J2000.as_datetime();
    assert!(g.0 == 2000);
    assert!(g.1 == 1);
    assert!(g.2 == 1);
    assert!(g.3 == 11);
    assert!(g.4 == 58);
    // J2000 is 2000-01-01 12:00:00 TT = 11:58:55.816 UTC
    assert!((g.5 - 55.816).abs() < 1.0e-7);
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

    let time = Instant::from_datetime(2016, 12, 31, 23, 59, 40.0).unwrap();
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
fn test_day_of_year() {
    // Following examples from google
    let thedate = Instant::from_date(2025, 8, 16).unwrap();
    assert_eq!(thedate.day_of_year(), 228);

    let thedate = Instant::from_date(2024, 2, 29).unwrap();
    assert_eq!(thedate.day_of_year(), 60);

    let thedate = Instant::from_date(2023, 1, 1).unwrap();
    assert_eq!(thedate.day_of_year(), 1);

    // Include a time component
    let thetime = Instant::from_datetime(2024, 12, 31, 23, 59, 59.999999).unwrap();
    assert_eq!(thetime.day_of_year(), 366);

    // Leap year test
    let thedate = Instant::from_date(2024, 12, 31).unwrap();
    assert_eq!(thedate.day_of_year(), 366);

    // Check year modulo 100, but not 400 (Not a leap year!)
    let thedate = Instant::from_date(2100, 12, 31).unwrap();
    assert_eq!(thedate.day_of_year(), 365);

    // Check year modulo 400 (Leap year!)
    let thedate = Instant::from_date(2400, 12, 31).unwrap();
    assert_eq!(thedate.day_of_year(), 366);
}

#[test]
fn test_ops() {
    let t1 = Instant::from_datetime(2024, 11, 13, 8, 0, 3.0).unwrap();
    let t2 = Instant::from_datetime(2024, 11, 13, 8, 0, 4.0).unwrap();

    assert!(t1 == t1);
    assert!(t1 != t2);
    assert!(t1 < t2);
    assert!(t2 > t1);

    let dt = t2 - t1;
    assert!(dt.as_microseconds() == 1_000_000);
    let t2 = Instant::from_datetime(2024, 11, 13, 8, 0, 2.0).unwrap();
    let dt = t2 - t1;
    assert!(dt.as_microseconds() == -1_000_000);
    let t2 = Instant::from_datetime(2024, 11, 13, 8, 1, 3.0).unwrap();
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

    let d1 = Duration::from_seconds(4.0);
    let d2 = Duration::from_seconds(5.0);
    assert!(d1 < d2);
    assert!(d2 > d1);
    assert!(d1 <= d2);
    assert!(d2 >= d1);
    assert!(d1 == d1);
    assert!(d1 != d2);
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
    let time = Instant::from_datetime(2024, 11, 24, 12, 0, 0.0).unwrap();
    assert!(time.as_jd_utc() == 2_460_639.0);
    assert!(time.as_mjd_utc() == 60_638.5);
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
fn test_bounds() {
    let tm = Instant::from_date(2024, 13, 4);
    assert!(tm.is_err());

    let tm = Instant::from_date(2024, 2, 29);
    assert!(tm.is_ok());

    let tm = Instant::from_date(2024, 2, 30);
    assert!(tm.is_err());

    let tm = Instant::from_datetime(2024, 2, 29, 23, 59, 59.999999);
    assert!(tm.is_ok());

    // Should be error ... not in leap second
    let tm = Instant::from_datetime(2024, 2, 29, 23, 59, 60.5);
    assert!(tm.is_err());

    // Should be OK ... within a leap second
    let tm = Instant::from_datetime(2008, 12, 31, 23, 59, 60.5);
    assert!(tm.is_ok());
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

#[test]
fn test_from_gps_week_and_second() {
    // GPS epoch: January 6, 1980 00:00:00 UTC
    let gps_epoch = Instant::from_gps_week_and_second(0, 0.0);
    let g = gps_epoch.as_datetime();
    assert_eq!(g.0, 1980);
    assert_eq!(g.1, 1);
    assert_eq!(g.2, 6);
    assert_eq!(g.3, 0);
    assert_eq!(g.4, 0);
    assert!((g.5 - 0.0).abs() < 1.0e-6);

    // Week 1 should be 7 days later: January 13, 1980
    let week1 = Instant::from_gps_week_and_second(1, 0.0);
    let g = week1.as_datetime();
    assert_eq!(g.0, 1980);
    assert_eq!(g.1, 1);
    assert_eq!(g.2, 13);

    // Difference between week 0 and week 1 should be exactly 7 days
    let diff = week1 - gps_epoch;
    assert!((diff.as_seconds() - 604800.0).abs() < 1.0e-6);

    // Week 0, second 86400 should be January 7, 1980
    let day2 = Instant::from_gps_week_and_second(0, 86400.0);
    let g = day2.as_datetime();
    assert_eq!(g.0, 1980);
    assert_eq!(g.1, 1);
    assert_eq!(g.2, 7);

    // Verify consistency: from_gps_week_and_second(0, N) should equal
    // GPS epoch + N seconds
    let gps_epoch = Instant::from_gps_week_and_second(0, 0.0);
    let t_100k = Instant::from_gps_week_and_second(0, 100000.0);
    assert!((t_100k - gps_epoch).as_seconds() - 100000.0 < 1.0e-6);

    // Week 2, second 43200 = 14 days + 12 hours from GPS epoch
    let t_2w = Instant::from_gps_week_and_second(2, 43200.0);
    let expected_seconds = 2.0 * 604800.0 + 43200.0;
    assert!((t_2w - gps_epoch).as_seconds() - expected_seconds < 1.0e-6);

    // GPS MJD at GPS epoch should be 44244.0 (same as UTC MJD at that time)
    let gps_mjd = gps_epoch.as_mjd_with_scale(crate::TimeScale::GPS);
    assert!(
        (gps_mjd - 44244.0).abs() < 1.0e-6,
        "GPS MJD at GPS epoch: expected 44244.0, got {}",
        gps_mjd
    );

    // GPS MJD round-trip: from_mjd(GPS) -> as_mjd(GPS) should be identity
    let t = Instant::from_mjd_with_scale(60000.0, crate::TimeScale::GPS);
    let mjd_back = t.as_mjd_with_scale(crate::TimeScale::GPS);
    assert!(
        (mjd_back - 60000.0).abs() < 1.0e-6,
        "GPS MJD round-trip: expected 60000.0, got {}",
        mjd_back
    );

    // GPS MJD should differ from UTC MJD by accumulated leap seconds
    // At a modern time, TAI-UTC = 37s, so GPS-UTC = 37-19 = 18s
    let t_modern = Instant::from_datetime(2024, 6, 15, 12, 0, 0.0).unwrap();
    let utc_mjd = t_modern.as_mjd_with_scale(crate::TimeScale::UTC);
    let gps_mjd = t_modern.as_mjd_with_scale(crate::TimeScale::GPS);
    let diff_seconds = (gps_mjd - utc_mjd) * 86400.0;
    assert!(
        (diff_seconds - 18.0).abs() < 1.0e-3,
        "GPS-UTC offset: expected 18s, got {}s",
        diff_seconds
    );
}

#[test]
fn test_rfc3339_with_timezone_offset() {
    // UTC (Z suffix)
    let t_z = Instant::from_rfc3339("2024-01-01T12:00:00Z").unwrap();
    let g = t_z.as_datetime();
    assert_eq!(g.0, 2024);
    assert_eq!(g.3, 12);

    // +00:00 should be same as Z
    let t_plus0 = Instant::from_rfc3339("2024-01-01T12:00:00+00:00").unwrap();
    assert!((t_z - t_plus0).as_seconds().abs() < 1.0e-6);

    // -05:00 means local time is 5 hours behind UTC
    // So 00:00:00-05:00 = 05:00:00 UTC
    let t_minus5 = Instant::from_rfc3339("2024-01-01T00:00:00-05:00").unwrap();
    let g = t_minus5.as_datetime();
    assert_eq!(g.3, 5);
    assert_eq!(g.4, 0);

    // +05:30 means local time is 5.5 hours ahead of UTC
    // So 12:00:00+05:30 = 06:30:00 UTC
    let t_plus530 = Instant::from_rfc3339("2024-01-01T12:00:00+05:30").unwrap();
    let g = t_plus530.as_datetime();
    assert_eq!(g.3, 6);
    assert_eq!(g.4, 30);

    // With fractional seconds and offset
    let t_frac = Instant::from_rfc3339("2024-06-15T12:30:00.123456-03:00").unwrap();
    let g = t_frac.as_datetime();
    assert_eq!(g.3, 15);
    assert_eq!(g.4, 30);
}
