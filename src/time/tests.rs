use super::Duration;
use super::Instant;
use super::{TimeScale, Weekday};

#[test]
fn test_timescale_try_from() {
    assert_eq!(TimeScale::try_from(1), Ok(TimeScale::UTC));
    assert_eq!(TimeScale::try_from(4), Ok(TimeScale::TAI));
    assert_eq!(TimeScale::try_from(6), Ok(TimeScale::TDB));
    // Out-of-range values are rejected rather than silently mapped to Invalid.
    assert!(TimeScale::try_from(0).is_err());
    assert!(TimeScale::try_from(-1).is_err());
    assert!(TimeScale::try_from(7).is_err());
}

#[test]
fn test_weekday_try_from() {
    assert_eq!(Weekday::try_from(0), Ok(Weekday::Sunday));
    assert_eq!(Weekday::try_from(6), Ok(Weekday::Saturday));
    assert!(Weekday::try_from(7).is_err());
    assert!(Weekday::try_from(-1).is_err());
}

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

    // More than 6 fractional digits must truncate to microseconds, not panic
    // (regression: the error path parsed the whole fraction into i32 and
    // overflowed).
    let time =
        Instant::strptime("2023-01-01T00:00:00.12345678901", "%Y-%m-%dT%H:%M:%S.%f").unwrap();
    let g = time.as_datetime();
    assert!(g.0 == 2023 && g.1 == 1 && g.2 == 1);
    assert!((g.5 - 0.123456).abs() < 1.0e-9);

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

#[test]
fn test_extreme_dates_error_not_panic() {
    // Extreme years overflow the i64 microsecond count; must be an error,
    // not a debug-build overflow panic (or silent wrap in release)
    assert!(Instant::from_datetime(600_000_000, 1, 1, 0, 0, 0.0).is_err());
    assert!(Instant::from_datetime(-600_000_000, 1, 1, 0, 0, 0.0).is_err());
    // Saturates rather than overflowing
    let _ = Instant::from_gps_week_and_second(i32::MAX, 0.0);
    let _ = Instant::from_gps_week_and_second(i32::MIN, 0.0);
    // strftime must not panic for extreme instants, whose datetime
    // breakdown can yield out-of-range month/weekday values
    for inst in [
        Instant::INVALID,
        Instant::new(i64::MIN + 1),
        Instant::new(i64::MAX),
        Instant::from_mjd_with_scale(-1.0e9, TimeScale::UTC),
        Instant::from_mjd_with_scale(1.0e9, TimeScale::UTC),
    ] {
        assert!(inst.strftime("%Y-%m-%d %B %b %a %A").is_ok());
    }
    // ... while valid dates still format correctly
    let s = Instant::from_datetime(2024, 1, 15, 0, 0, 0.0)
        .unwrap()
        .strftime("%B")
        .unwrap();
    assert_eq!(s, "January");
}

#[test]
fn test_rfc3339_non_ascii_errors_not_panics() {
    // Regression: the timezone-offset scan byte-slices the last 6 bytes of
    // the input, which used to panic on non-char-boundary indices for
    // non-ASCII input. These must all return a clean error.
    assert!(Instant::from_rfc3339("ααα:00").is_err());
    assert!(Instant::from_rfc3339("X+aé:0").is_err());
    assert!(Instant::from_string("ααα:00").is_err());
    // from_string is lenient and may parse a valid prefix of this one; the
    // regression being tested is only that it must not panic
    let _ = Instant::from_string("2024-01-01T12:00:00+05:0é");
}

/// TDB − TT has a one-year period and ~1.66 ms amplitude (Vallado Eq. 3-50;
/// the series argument is in radians). Reference values from ERFA `dtdb`
/// at the geocenter; the one-term series is within ~50 µs of it.
#[test]
fn test_tdb_minus_tt() {
    let tdb_minus_tt = |mjd_tt: f64| {
        let t = Instant::from_mjd_with_scale(mjd_tt, TimeScale::TT);
        (t.as_mjd_with_scale(TimeScale::TDB) - t.as_mjd_with_scale(TimeScale::TT)) * 86400.0
    };
    // (TT MJD, ERFA dtdb in seconds)
    for (mjd, erfa) in [
        (51544.5, -9.930719894379447e-05),
        (60310.0, -1.1923612875657498e-04),
        (60400.0, 1.6359059807821287e-03),
        (57754.0, -4.952007962185753e-05),
    ] {
        let d = tdb_minus_tt(mjd);
        assert!((d - erfa).abs() < 60.0e-6, "MJD {mjd}: {d} vs ERFA {erfa}");
    }
    // Annual period: the extremes over any one year reach the amplitude,
    // and a year later the value repeats
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for day in 0..366 {
        let d = tdb_minus_tt(60310.0 + day as f64);
        lo = lo.min(d);
        hi = hi.max(d);
        let d_next_year = tdb_minus_tt(60310.0 + day as f64 + 365.25);
        assert!((d - d_next_year).abs() < 2.0e-6);
    }
    assert!(hi > 1.6e-3 && lo < -1.6e-3, "range [{lo}, {hi}]");

    // TDB -> Instant -> TDB round trip (to the microsecond resolution)
    for mjd in [51544.5, 57754.25, 60400.0, 60482.7] {
        let t = Instant::from_mjd_with_scale(mjd, TimeScale::TDB);
        let back = t.as_mjd_with_scale(TimeScale::TDB);
        assert!(((back - mjd) * 86400.0).abs() < 2.0e-6, "{mjd} -> {back}");
    }
}

/// Every table entry: 00:00:00 UTC on the day after a leap second is the
/// end, not the start, of the leap second, however it is constructed.
#[test]
fn test_midnight_after_leap_second() {
    let t = Instant::from_date(2017, 1, 1).unwrap();
    assert_eq!(t.raw, 1483228837000000);
    assert_eq!(t.to_string(), "2017-01-01T00:00:00.000000Z");
    let day = Instant::from_date(2016, 12, 31).unwrap();
    assert_eq!((t - day).as_microseconds(), 86_401_000_000);
    assert_eq!(Instant::from_mjd_utc(57754.0).raw, t.raw);
    assert_eq!(Instant::from_jd_utc(2457754.5).raw, t.raw);
    assert_eq!(Instant::from_unixtime(1483228800.0).raw, t.raw);
    assert_eq!(day.add_utc_days(1.0).raw, t.raw);
    assert_eq!(t.as_unixtime(), 1483228800.0);
    assert_eq!(t.as_mjd_utc(), 57754.0);

    // A sample of the (one-second) leap seconds in the table
    for (y, m) in [
        (1972, 7),
        (1973, 1),
        (1981, 7),
        (1990, 1),
        (1999, 1),
        (2006, 1),
        (2009, 1),
        (2012, 7),
        (2015, 7),
        (2017, 1),
    ] {
        let after = Instant::from_date(y, m, 1).unwrap();
        let (py, pm, pd) = if m == 1 { (y - 1, 12, 31) } else { (y, 6, 30) };
        let before = Instant::from_datetime(py, pm, pd, 23, 59, 59.0).unwrap();
        assert_eq!((after - before).as_microseconds(), 2_000_000, "{y}-{m}");
        let g = after.as_datetime();
        assert_eq!((g.0, g.1, g.2, g.3, g.4, g.5), (y, m, 1, 0, 0, 0.0));
        let leap = before + Duration::from_seconds(1.0);
        let g = leap.as_datetime();
        assert_eq!((g.0, g.1, g.2, g.3, g.4, g.5), (py, pm, pd, 23, 59, 60.0));
    }
}

/// A leap second can be entered by its own label and round-trips through
/// construction -> string -> construction.
#[test]
fn test_leap_second_label_roundtrip() {
    let t = Instant::from_datetime(2016, 12, 31, 23, 59, 60.5).unwrap();
    assert_eq!(t.raw, 1483228836500000);
    let s = t.to_string();
    assert_eq!(s, "2016-12-31T23:59:60.500000Z");
    assert_eq!(Instant::from_rfc3339(&s).unwrap().raw, t.raw);
    assert_eq!(t.as_rfc3339(), s);
    assert_eq!(Instant::from_rfc3339(&t.as_rfc3339()).unwrap().raw, t.raw);

    // Exactly :60 is the start of the leap second
    let t60 = Instant::from_datetime(2016, 12, 31, 23, 59, 60.0).unwrap();
    assert_eq!(t60.raw, 1483228836000000);
    assert_eq!(
        Instant::from_rfc3339("2016-12-31T23:59:60Z").unwrap().raw,
        t60.raw
    );
    // ... and a 30 June one
    let t = Instant::from_rfc3339("2015-06-30T23:59:60.25Z").unwrap();
    assert_eq!(t.to_string(), "2015-06-30T23:59:60.250000Z");
    assert_eq!(t.raw, 1435708835250000);

    // Not a leap-second day, or not the last minute of it
    assert!(Instant::from_datetime(2024, 2, 29, 23, 59, 60.0).is_err());
    assert!(Instant::from_datetime(2024, 12, 31, 23, 59, 60.5).is_err());
    assert!(Instant::from_datetime(2016, 12, 31, 23, 58, 60.0).is_err());
    assert!(Instant::from_datetime(2016, 12, 30, 23, 59, 60.0).is_err());
    assert!(Instant::from_rfc3339("2024-12-31T23:59:60Z").is_err());
    // Past the end of a one-second leap second
    assert!(Instant::from_datetime(2016, 12, 31, 23, 59, 61.0).is_err());
}

/// The +0.107758 s TAI − UTC step at 1972-01-01 00:00:00 UTC, from the
/// pre-1972 drifting offset (9.892242 s) to the 10 s of the leap-second table.
#[test]
fn test_1972_step() {
    let t = Instant::from_date(1972, 1, 1).unwrap();
    assert_eq!(t.raw, 63072010000000);
    let before = Instant::from_datetime(1971, 12, 31, 23, 59, 59.0).unwrap();
    // TAI − UTC one second before midnight: 9.892242 s less 1 s of drift
    // (0.002592 s/day), rounded to the microsecond
    assert_eq!(before.raw, 63071999000000 + 9_892_242);
    // The day is 86,400 UTC seconds, each 1 + 0.002592/86400 SI seconds,
    // plus the 0.107758 s step
    assert_eq!(
        (t - Instant::from_date(1971, 12, 31).unwrap()).as_microseconds(),
        86_400_000_000 + 2_592 + 107_758
    );

    // Every 10 ms across the step has a distinct, increasing label that
    // constructs back to the same instant
    let mut prev: Option<(i32, i32, i32, i32, i32, f64)> = None;
    for s in 6307200897..6307201002i64 {
        let inst = Instant::new(s * 10_000);
        let g = inst.as_datetime();
        let back = Instant::from_datetime(g.0, g.1, g.2, g.3, g.4, g.5).unwrap();
        assert!((back.raw - inst.raw).abs() <= 1, "{inst:?}");
        if let Some(p) = prev {
            assert!(
                (g.0, g.1, g.2, g.3, g.4) > (p.0, p.1, p.2, p.3, p.4)
                    || ((g.0, g.1, g.2, g.3, g.4) == (p.0, p.1, p.2, p.3, p.4) && g.5 > p.5),
                "{p:?} -> {g:?}"
            );
        }
        prev = Some(g);
    }
    assert_eq!(
        Instant::new(63072009892242).to_string(),
        "1971-12-31T23:59:60.000000Z"
    );
    assert_eq!(
        Instant::new(63072009999999).to_string(),
        "1971-12-31T23:59:60.107757Z"
    );
    assert_eq!(t.to_string(), "1972-01-01T00:00:00.000000Z");
    // Labels past the step's 0.107758 s are rejected
    assert!(Instant::from_datetime(1971, 12, 31, 23, 59, 60.107758).is_err());
    assert!(Instant::from_datetime(1971, 12, 31, 23, 59, 61.0).is_err());
    // A later one-second leap second does not accept 23:59:61
    assert!(Instant::from_datetime(1972, 6, 30, 23, 59, 61.0).is_err());
    // Nothing accepts 62 s or more
    assert!(Instant::from_datetime(1960, 12, 31, 23, 59, 62.0).is_err());
}

/// Times of day before 1970 (negative raw counts) break down with
/// non-negative fields.
#[test]
fn test_pre_1970_datetime() {
    let t = Instant::from_datetime(1960, 1, 1, 12, 0, 0.0).unwrap();
    let g = t.as_datetime();
    assert_eq!((g.0, g.1, g.2, g.3, g.4, g.5), (1960, 1, 1, 12, 0, 0.0));
    assert_eq!(t.to_string(), "1960-01-01T12:00:00.000000Z");
    assert_eq!(t.as_rfc3339(), "1960-01-01T12:00:00.000000Z");

    let t = Instant::from_datetime(1969, 12, 31, 23, 59, 59.5).unwrap();
    // TAI − UTC was 8.000082 s at 1970-01-01 (pre-1972 UTC)
    assert_eq!(t.raw, -500_000 + 8_000_082);
    assert_eq!(t.to_string(), "1969-12-31T23:59:59.500000Z");
    assert_eq!(Instant::UNIX_EPOCH, Instant::from_date(1970, 1, 1).unwrap());
    assert_eq!(Instant::UNIX_EPOCH.as_unixtime(), 0.0);
    assert_eq!(
        Instant::MJD_EPOCH,
        Instant::from_date(1858, 11, 17).unwrap()
    );

    // Whole-second sweep 1900..2030 (step not a multiple of a day): the
    // breakdown constructs back to the same instant (to 1 us in 1961-1971,
    // where about one TAI microsecond in 7e7 has no UTC label of its own)
    let mut raw = Instant::from_date(1900, 1, 1).unwrap().raw;
    let end = Instant::from_date(2030, 1, 1).unwrap().raw;
    while raw < end {
        let inst = Instant::new(raw);
        let g = inst.as_datetime();
        assert!((0..24).contains(&g.3) && (0..60).contains(&g.4) && g.5 >= 0.0);
        let back = Instant::from_datetime(g.0, g.1, g.2, g.3, g.4, g.5).unwrap();
        assert!((back.raw - raw).abs() <= 1, "{g:?}");
        raw += 7_919_377 * 1_000_000;
    }
}

/// UT1 − UTC is interpolated without the leap-second step, and UT1 is
/// continuous through the leap second in both directions.
#[test]
fn test_ut1_across_leap_second() {
    let dut1 = |t: &Instant| {
        (t.as_mjd_with_scale(TimeScale::UT1) - t.as_mjd_with_scale(TimeScale::UTC)) * 86400.0
    };
    let noon_before = Instant::from_datetime(2016, 12, 31, 12, 0, 0.0).unwrap();
    let d = dut1(&noon_before);
    // IERS: UT1 − UTC = −0.4075 s on 2016-12-31 and +0.5921 s on 2017-01-01
    assert!(
        (d - -0.408).abs() < 0.005,
        "UT1-UTC at 2016-12-31 12:00 = {d}"
    );

    // UT1 advances with elapsed (TAI) time through the leap second; UT1 − TAI
    // changes by ~1 ms/day, so over a few seconds it is constant to < 1 µs
    let start = Instant::from_datetime(2016, 12, 31, 23, 59, 58.0).unwrap();
    let ut1_0 = start.as_mjd_with_scale(TimeScale::UT1);
    for k in 0..20 {
        let dt = k as f64 * 0.25;
        let t = start + Duration::from_seconds(dt);
        let ut1 = t.as_mjd_with_scale(TimeScale::UT1);
        assert!(
            ((ut1 - ut1_0) * 86400.0 - dt).abs() < 5.0e-6,
            "{t}: UT1 advanced {} s over {dt} s",
            (ut1 - ut1_0) * 86400.0
        );
        // UT1 -> Instant round trip, including inside the leap second
        let back = Instant::from_mjd_with_scale(ut1, TimeScale::UT1);
        assert!(
            (back - t).as_seconds().abs() < 5.0e-6,
            "{t} -> UT1 -> {back}"
        );
    }
}

/// Across every leap second inside the EOP table, UT1 − TAI is continuous
/// (the EOP lookup interpolates UT1 − TAI), UT1 is monotonic, and UT1 ->
/// Instant inverts it. Before the table (or with none) UT1 − UTC is 0, so UT1
/// is the UTC MJD.
#[test]
fn test_ut1_across_every_leap_second() {
    let Some(cov) = crate::earth_orientation_params::coverage() else {
        return;
    };
    let (first, last) = (cov.first.as_mjd_utc(), cov.last_observed.as_mjd_utc());
    let mut checked = 0;
    for (y, m) in [
        (1972, 1),
        (1972, 7),
        (1973, 1),
        (1974, 1),
        (1981, 7),
        (1990, 1),
        (1999, 1),
        (2009, 1),
        (2017, 1),
    ] {
        let midnight = Instant::from_date(y, m, 1).unwrap();
        let mjd = midnight.as_mjd_utc();
        let times: Vec<Instant> = (-40..=40)
            .map(|k| midnight + Duration::from_microseconds(k * 50_000))
            .collect();
        if mjd - 1.0 <= first || mjd + 1.0 >= last {
            // Before the table: UT1 = UTC
            if mjd + 1.0 < first {
                for t in [midnight + Duration::from_seconds(0.5), times[0]] {
                    assert_eq!(
                        t.as_mjd_with_scale(TimeScale::UT1),
                        t.as_mjd_with_scale(TimeScale::UTC)
                    );
                }
            }
            continue;
        }
        let ut1: Vec<f64> = times
            .iter()
            .map(|t| t.as_mjd_with_scale(TimeScale::UT1))
            .collect();
        let d: Vec<f64> = times
            .iter()
            .zip(&ut1)
            .map(|(t, u)| (u - t.as_mjd_with_scale(TimeScale::TAI)) * 86_400.0)
            .collect();
        let (lo, hi) = d
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(hi - lo < 2.0e-6, "{y}-{m}: UT1 − TAI spans {} s", hi - lo);
        for (w, t) in ut1.windows(2).zip(&times) {
            assert!(w[1] > w[0], "UT1 not increasing at {t}");
        }
        for (t, u) in times.iter().zip(&ut1) {
            let back = Instant::from_mjd_with_scale(*u, TimeScale::UT1);
            assert!((back - *t).as_microseconds().abs() <= 2, "{t} -> {back}");
        }
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} leap seconds checked");
}
