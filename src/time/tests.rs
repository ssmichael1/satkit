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

// `as_datetime()` seconds are `µs as f64 * 1e-6`, so they equal a decimal
// literal exactly unless that product rounds differently (55.816 → 55.815999…);
// those cases compare the integer-µs `as_datetime_us()` instead.

#[test]
fn test_j2000() {
    // J2000 is 2000-01-01 12:00:00 TT = 11:58:55.816 UTC
    assert_eq!(
        Instant::J2000.as_datetime_us(),
        (2000, 1, 1, 11, 58, 55_816_000)
    );
}

#[test]
fn test_fromstring() {
    let time = Instant::from_string("March 4 2024").unwrap();
    assert_eq!(time.as_datetime(), (2024, 3, 4, 0, 0, 0.0));

    let time = Instant::from_string("2024-01-04 13:14:12.123000").unwrap();
    assert_eq!(time.as_datetime(), (2024, 1, 4, 13, 14, 12.123));
}

#[test]
fn test_unixtime() {
    let time = Instant::from_unixtime(1732939013.0);
    assert_eq!(time.as_datetime(), (2024, 11, 30, 3, 56, 53.0));

    let time = Instant::from_datetime(2016, 12, 31, 23, 59, 40.0).unwrap();
    assert_eq!(time.as_unixtime(), 1483228780.0);
    assert_eq!(time.as_datetime(), (2016, 12, 31, 23, 59, 40.0));
}

#[test]
fn test_leapsecond() {
    // Beginning of leap second
    let mut t = Instant::new(1483228836000000);
    assert_eq!(t.as_datetime(), (2016, 12, 31, 23, 59, 60.0));

    // Middle of a leap second
    let t2 = t + Duration::from_microseconds(100);
    assert_eq!(t2.as_datetime_us(), (2016, 12, 31, 23, 59, 60_000_100));

    // Just prior to leap second
    t -= Duration::from_seconds(1.0);
    assert_eq!(t.as_datetime(), (2016, 12, 31, 23, 59, 59.0));

    // Just after leap second
    t += Duration::from_seconds(2.0);
    assert_eq!(t.as_datetime(), (2017, 1, 1, 0, 0, 0.0));
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
    assert_eq!(t3.as_datetime(), (2024, 11, 14, 8, 1, 3.0));

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
    assert_eq!(Instant::GPS_EPOCH.as_datetime(), (1980, 1, 6, 0, 0, 0.0));
    assert_eq!(
        Instant::from_gps_week_and_second(0, 0.0),
        Instant::GPS_EPOCH
    );
    assert_eq!(
        Instant::GPS_EPOCH.as_mjd_with_scale(TimeScale::GPS),
        44244.0
    );
}

#[test]
fn test_jd() {
    let time = Instant::from_datetime(2024, 11, 24, 12, 0, 0.0).unwrap();
    assert!(time.as_jd_utc() == 2_460_639.0);
    assert!(time.as_mjd_utc() == 60_638.5);
}

#[test]
fn test_rfc3339() {
    for (s, sec) in [
        ("2024-11-24T12:03:45.123456Z", 45.123456),
        ("2024-11-24T12:03:45Z", 45.0),
        ("2024-11-24T12:03:45.123Z", 45.123), // milliseconds
    ] {
        let time = Instant::from_rfc3339(s).unwrap();
        assert_eq!(time.as_datetime(), (2024, 11, 24, 12, 3, sec), "{s}");
    }
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
    for (s, fmt, dt) in [
        (
            "2024-11-24T12:03:45.123456",
            "%Y-%m-%dT%H:%M:%S.%f",
            (2024, 11, 24, 12, 3, 45.123456),
        ),
        // Milliseconds
        (
            "2024-11-24T12:03:45.123",
            "%Y-%m-%dT%H:%M:%S.%f",
            (2024, 11, 24, 12, 3, 45.123),
        ),
        (
            "February 13 2024 12:03:45.123456",
            "%B %d %Y %H:%M:%S.%f",
            (2024, 2, 13, 12, 3, 45.123456),
        ),
        // More than 6 fractional digits round to the nearest microsecond,
        // and must not panic (regression: the error path parsed the whole
        // fraction into i32 and overflowed).
        (
            "2023-01-01T00:00:00.12345678901",
            "%Y-%m-%dT%H:%M:%S.%f",
            (2023, 1, 1, 0, 0, 0.123457),
        ),
        (
            "09-Jun-2023 22:27:19",
            "%d-%b-%Y %H:%M:%S",
            (2023, 6, 9, 22, 27, 19.0),
        ),
    ] {
        let time = Instant::strptime(s, fmt).unwrap();
        assert_eq!(time.as_datetime(), dt, "{s}");
    }
}

#[test]
fn test_rfc3339_with_timezone_offset() {
    // UTC (Z suffix)
    let t_z = Instant::from_rfc3339("2024-01-01T12:00:00Z").unwrap();
    assert_eq!(t_z.as_datetime(), (2024, 1, 1, 12, 0, 0.0));

    // +00:00 should be same as Z
    let t_plus0 = Instant::from_rfc3339("2024-01-01T12:00:00+00:00").unwrap();
    assert_eq!(t_z, t_plus0);

    for (s, dt) in [
        // -05:00: local time is 5 hours behind UTC
        ("2024-01-01T00:00:00-05:00", (2024, 1, 1, 5, 0, 0.0)),
        // +05:30: local time is 5.5 hours ahead of UTC
        ("2024-01-01T12:00:00+05:30", (2024, 1, 1, 6, 30, 0.0)),
        // With fractional seconds and offset
        (
            "2024-06-15T12:30:00.123456-03:00",
            (2024, 6, 15, 15, 30, 0.123456),
        ),
    ] {
        assert_eq!(Instant::from_rfc3339(s).unwrap().as_datetime(), dt, "{s}");
    }
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

/// TDB − TT against reference values from ERFA `dtdb` at the geocenter
/// (Vallado Eq. 3-50; the series argument is in radians); the one-term
/// series is within ~50 µs of it. The annual period and amplitude are
/// `tdb_minus_tt_bounded_and_annual`, and the TDB inverse `tdb_inverse`, in
/// `tests/properties.rs`.
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
        assert_eq!(after.as_datetime(), (y, m, 1, 0, 0, 0.0));
        let leap = before + Duration::from_seconds(1.0);
        assert_eq!(leap.as_datetime(), (py, pm, pd, 23, 59, 60.0));
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
            // Tuples compare lexicographically
            assert!(g > p, "{p:?} -> {g:?}");
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
    assert_eq!(t.as_datetime(), (1960, 1, 1, 12, 0, 0.0));
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

/// Float seconds, days and Unix times round to the nearest microsecond
/// rather than truncating (`0.000249 * 1e6` is `248.99999999999997`).
#[test]
fn test_float_to_microseconds_rounds() {
    let midnight = Instant::from_date(2024, 1, 1).unwrap();
    let us = |t: Instant| (t - midnight).as_microseconds();
    assert_eq!(
        us(Instant::from_datetime(2024, 1, 1, 0, 0, 0.000249).unwrap()),
        249
    );
    assert_eq!(
        us(Instant::from_datetime(2024, 1, 1, 0, 0, 1.0000004).unwrap()),
        1_000_000
    );
    assert_eq!(
        us(Instant::from_datetime(2024, 1, 1, 0, 0, 1.0000006).unwrap()),
        1_000_001
    );
    assert_eq!(
        us(Instant::from_datetime_with_scale(2024, 1, 1, 0, 0, 0.000249, TimeScale::UTC).unwrap()),
        249
    );

    // Uniform scales are built in integers: exact, not via a float MJD
    let tt =
        Instant::from_datetime_with_scale(2039, 2, 25, 21, 42, 35.990_07, TimeScale::TT).unwrap();
    let tt0 = Instant::from_datetime_with_scale(2039, 2, 25, 21, 42, 0.0, TimeScale::TT).unwrap();
    assert_eq!((tt - tt0).as_microseconds(), 35_990_070);

    for (d, us) in [
        (Duration::from_seconds(0.000249), 249),
        (Duration::from_seconds(-0.000249), -249),
        (Duration::from_milliseconds(0.249), 249),
        (Duration::from_minutes(0.000249 / 60.0), 249),
        (Duration::from_hours(1.0e-6 / 3600.0), 1),
        (Duration::from_days(1.0e-6 / 86400.0), 1),
        (Duration::from_seconds(0.4e-6), 0),
        (Duration::from_seconds(-0.6e-6), -1),
    ] {
        assert_eq!(d.as_microseconds(), us, "{d:?}");
    }

    let u = Instant::from_unixtime(1_700_000_000.000249);
    assert_eq!(u.as_unixtime_microseconds(), 1_700_000_000_000_249);
    assert_eq!(
        Instant::from_unixtime_microseconds(1_700_000_000_000_249),
        u
    );
    assert_eq!(
        Instant::from_gps_week_and_second(2300, 0.000249)
            - Instant::from_gps_week_and_second(2300, 0.0),
        Duration::from_microseconds(249)
    );
    let t = midnight + Duration::from_microseconds(990_070);
    assert_eq!(
        Instant::from_mjd_with_scale(t.as_mjd_with_scale(TimeScale::TAI), TimeScale::TAI),
        t
    );
    assert_eq!(
        midnight.add_utc_days(0.000249 / 86400.0),
        midnight + Duration::from_microseconds(249)
    );
}

/// A second that rounds up to the end of its minute carries into the next
/// minute — or into the leap second where one follows — and one that rounds
/// up to the end of a leap second is the next day's 00:00:00.
#[test]
fn test_rounding_up_at_end_of_minute() {
    // Ordinary minute
    assert_eq!(
        Instant::from_datetime(2024, 1, 1, 12, 0, 59.9999997).unwrap(),
        Instant::from_datetime(2024, 1, 1, 12, 1, 0.0).unwrap()
    );
    // End of a day without a leap second
    assert_eq!(
        Instant::from_datetime(2016, 6, 30, 23, 59, 59.9999997).unwrap(),
        Instant::from_date(2016, 7, 1).unwrap()
    );
    // Before a leap second: 23:59:60.000000
    let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.0).unwrap();
    assert_eq!(
        Instant::from_datetime(2016, 12, 31, 23, 59, 59.9999997).unwrap(),
        leap
    );
    // End of the leap second
    assert_eq!(
        Instant::from_datetime(2016, 12, 31, 23, 59, 60.9999997).unwrap(),
        Instant::from_date(2017, 1, 1).unwrap()
    );
    assert_eq!(
        Instant::from_datetime(2016, 12, 31, 23, 59, 60.9999997).unwrap() - leap,
        Duration::from_seconds(1.0)
    );
    // Leap-second offsets round too
    assert_eq!(
        Instant::from_datetime(2016, 12, 31, 23, 59, 60.000249).unwrap() - leap,
        Duration::from_microseconds(249)
    );
    // Still an error: no leap second at this minute
    assert!(Instant::from_datetime(2016, 6, 30, 23, 59, 60.0000004).is_err());
    assert!(Instant::from_datetime(2016, 12, 31, 23, 59, 61.0).is_err());
}

/// `as_datetime()` seconds feed back through `from_datetime` exactly, and
/// the formatted microseconds are the stored ones.
#[test]
fn test_as_datetime_roundtrip_exact() {
    let base = Instant::from_datetime(1972, 1, 1, 5, 30, 0.0).unwrap();
    for us in (0..60_000_000i64)
        .step_by(7_919)
        .chain([45_773_591, 59_999_999])
    {
        let t = base + Duration::from_microseconds(us);
        let (y, mo, d, h, mi, s) = t.as_datetime();
        assert_eq!(
            Instant::from_datetime(y, mo, d, h, mi, s).unwrap(),
            t,
            "{us}"
        );
        assert_eq!(Instant::from_rfc3339(&t.to_string()).unwrap(), t, "{us}");
    }
}

/// UTC offsets act on the calendar label (RFC 3339): exact across a leap
/// second, with the sign of `%z` meaning local minus UTC.
#[test]
fn test_utc_offset_applies_to_label() {
    let utc = |y, mo, d, h, mi, s| Instant::from_datetime(y, mo, d, h, mi, s).unwrap();
    assert_eq!(
        Instant::strptime("2024-01-01T12:00:00+0100", "%Y-%m-%dT%H:%M:%S%z").unwrap(),
        utc(2024, 1, 1, 11, 0, 0.0)
    );
    assert_eq!(
        Instant::strptime("2024-01-01T12:00:00-01:30", "%Y-%m-%dT%H:%M:%S%z").unwrap(),
        utc(2024, 1, 1, 13, 30, 0.0)
    );
    assert_eq!(
        Instant::from_rfc3339("2017-01-01T00:30:00+01:00").unwrap(),
        utc(2016, 12, 31, 23, 30, 0.0)
    );
    assert_eq!(
        Instant::from_rfc3339("2016-12-31T18:59:59.5-05:00").unwrap(),
        utc(2016, 12, 31, 23, 59, 59.5)
    );
    // A leap second written in local time
    assert_eq!(
        Instant::from_rfc3339("2017-01-01T00:59:60.25+01:00").unwrap(),
        utc(2016, 12, 31, 23, 59, 60.25)
    );
    assert_eq!(
        Instant::strptime("2016-12-31 18:59:60.5-0500", "%Y-%m-%d %H:%M:%S.%f%z").unwrap(),
        utc(2016, 12, 31, 23, 59, 60.5)
    );
    // ... but not where UTC has none
    assert!(Instant::from_rfc3339("2017-01-01T01:59:60+01:00").is_err());
    // Local calendar fields are still validated
    assert!(Instant::from_rfc3339("2024-02-30T12:00:00+01:00").is_err());
    // Before 1972 too (where a model of the drifting pre-1972 UTC − TAI
    // makes an elapsed-time offset wrong by up to milliseconds)
    assert_eq!(
        Instant::from_rfc3339("1965-06-01T02:30:00.25+14:00").unwrap(),
        utc(1965, 5, 31, 12, 30, 0.25)
    );
    assert_eq!(
        Instant::from_rfc3339("1965-05-31T00:30:00-12:00").unwrap(),
        utc(1965, 5, 31, 12, 30, 0.0)
    );
}

/// The split TDB date matches the single-f64 MJD and resolves far below a
/// microsecond.
#[test]
fn test_mjd_tdb_split() {
    let t = Instant::from_datetime(2024, 3, 1, 6, 0, 0.0).unwrap();
    let (day, frac) = t.mjd_tdb_split();
    let mjd = t.as_mjd_with_scale(TimeScale::TDB);
    assert_eq!(day as f64 + frac, mjd);
    assert!((0.0..1.0).contains(&frac));
    let (day2, frac2) = (t + Duration::from_microseconds(1)).mjd_tdb_split();
    assert_eq!(day2, day);
    let dt_us = (frac2 - frac) * 86_400.0e6;
    assert!((dt_us - 1.0).abs() < 1.0e-6, "{dt_us}");
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

/// UTC label → instant, for the parser regression tests below.
fn utc_label(y: i32, mo: i32, d: i32, h: i32, mi: i32, s: f64) -> Instant {
    Instant::from_datetime(y, mo, d, h, mi, s).unwrap()
}

/// `from_rfc3339` never drops a UTC offset or reads the local label as UTC:
/// once a zone is present it is applied or the parse fails, and leftover
/// input is an error. `±HH:MM` (RFC 3339), `±HHMM` and `±HH` (ISO 8601) are
/// all accepted.
#[test]
fn test_rfc3339_offset_never_dropped() {
    for (s, want) in [
        (
            "2024-01-01T12:00:00+0100",
            utc_label(2024, 1, 1, 11, 0, 0.0),
        ),
        (
            "2024-01-01T12:00:00+01:00",
            utc_label(2024, 1, 1, 11, 0, 0.0),
        ),
        ("2024-01-01T12:00:00+01", utc_label(2024, 1, 1, 11, 0, 0.0)),
        (
            "2024-01-01T12:00:00.123+0100",
            utc_label(2024, 1, 1, 11, 0, 0.123),
        ),
        ("2024-01-01T12:00:00-05", utc_label(2024, 1, 1, 17, 0, 0.0)),
        (
            "2024-01-01T12:00:00-0530",
            utc_label(2024, 1, 1, 17, 30, 0.0),
        ),
        ("2024-01-01t12:00:00z", utc_label(2024, 1, 1, 12, 0, 0.0)),
        (" 2024-01-01T12:00:00Z\n", utc_label(2024, 1, 1, 12, 0, 0.0)),
        // No zone: taken as UTC (lenient; RFC 3339 requires one)
        ("2024-01-01T12:00:00", utc_label(2024, 1, 1, 12, 0, 0.0)),
        ("2024-01-01T12:00:00.5", utc_label(2024, 1, 1, 12, 0, 0.5)),
    ] {
        assert_eq!(Instant::from_rfc3339(s).unwrap(), want, "{s:?}");
        assert_eq!(Instant::from_string(s).unwrap(), want, "{s:?}");
    }
    for s in [
        "2024-01-04T13:14:12+01:00x",
        "2024-01-01T12:00:00Z trailing",
        "2024-01-01T12:00:00ZZ",
        "2024-01-01T12:00:00+1",
        "2024-01-01T12:00:00+013",
        "2024-01-01T12:00:00+01:0",
        "2024-01-01T12:00:00+24:00",
        "2024-01-01T12:00:00+01:60",
        "2024-01-01T12:00:00.Z",
        "2024-01-01T12:00Z",
        "2024-1-01T12:00:00Z",
        // The offset path rejects these leap seconds (UTC has none at the
        // local label's UTC time); they must not then be read as UTC
        "2016-12-31T23:59:60+01:00",
        "2016-12-31T23:59:60.5-05:00",
    ] {
        assert!(Instant::from_rfc3339(s).is_err(), "{s:?} accepted");
    }
    assert!(matches!(
        Instant::from_rfc3339("2016-12-31T23:59:60+01:00"),
        Err(super::InstantError::InvalidLeapSecond)
    ));
}
/// The free-form parser reads a trailing `±HHMM` / `±HH:MM` / `±HH` as a
/// UTC offset (not as microseconds), keeps `HH:MM` without seconds, and
/// rejects numbers it cannot place.
#[test]
fn test_from_string_offsets_and_short_times() {
    for (s, want) in [
        (
            "2024-01-04 13:14:12 +0100",
            utc_label(2024, 1, 4, 12, 14, 12.0),
        ),
        (
            "2024-01-04 13:14:12-05:00",
            utc_label(2024, 1, 4, 18, 14, 12.0),
        ),
        (
            "2024-01-04 13:14:12.25 -05",
            utc_label(2024, 1, 4, 18, 14, 12.25),
        ),
        ("2024-01-04 13:14", utc_label(2024, 1, 4, 13, 14, 0.0)),
        (
            "2024-01-04 13:14 +01:00",
            utc_label(2024, 1, 4, 12, 14, 0.0),
        ),
        ("2024-01-04", utc_label(2024, 1, 4, 0, 0, 0.0)),
        ("March 4 2024 13:14:12", utc_label(2024, 3, 4, 13, 14, 12.0)),
        (
            "2023-03-05 11:03:45.453Z",
            utc_label(2023, 3, 5, 11, 3, 45.453),
        ),
        ("2024.01.04 13:14:12", utc_label(2024, 1, 4, 13, 14, 12.0)),
    ] {
        assert_eq!(Instant::from_string(s).unwrap(), want, "{s:?}");
    }
    for s in [
        "2024-01-04 13",
        "2024-01-04 13:14:12 123",
        "2024-01-04 13:14:12 +1",
        "2024-01-04 13:14:12 +99:00",
        "2024-01-04 13:14:12 +0100 7",
    ] {
        assert!(Instant::from_string(s).is_err(), "{s:?} accepted");
    }
}

/// `%z` takes `+`, `-` or `Z`/`z` first, exactly two digits per field, and
/// hours 00–23, minutes 00–59.
#[test]
fn test_strptime_z_rejects_malformed_offsets() {
    let fmt = "%Y-%m-%d %H:%M:%S%z";
    for z in [
        "5030", "+-1:30", "+99:99", "x0100", "+24:00", "+01:60", "+1", "+1:30", "+01:3", "",
    ] {
        let s = format!("2024-01-01 12:00:00{z}");
        assert!(Instant::strptime(&s, fmt).is_err(), "{s:?} accepted");
    }
    for (z, want) in [
        ("Z", utc_label(2024, 1, 1, 12, 0, 0.0)),
        ("z", utc_label(2024, 1, 1, 12, 0, 0.0)),
        ("+0100", utc_label(2024, 1, 1, 11, 0, 0.0)),
        ("+01:00", utc_label(2024, 1, 1, 11, 0, 0.0)),
        ("+01", utc_label(2024, 1, 1, 11, 0, 0.0)),
        ("-2359", utc_label(2024, 1, 2, 11, 59, 0.0)),
        ("-00:00", utc_label(2024, 1, 1, 12, 0, 0.0)),
    ] {
        let s = format!("2024-01-01 12:00:00{z}");
        assert_eq!(Instant::strptime(&s, fmt).unwrap(), want, "{s:?}");
    }
    // Leftover input and non-digits in numeric fields are errors
    assert!(Instant::strptime("2024-01-01 12:00:00 x", "%Y-%m-%d %H:%M:%S").is_err());
    assert!(Instant::strptime("2024-+1-01", "%Y-%m-%d").is_err());
    assert!(Instant::strptime("2024-01-01%", "%Y-%m-%d%%").is_ok());
}
/// The weekday comes from the integer UTC day, so it is right to the last
/// microsecond of the day (the f64 Julian Date resolves only ~40 µs), and
/// for dates before MJD 0.
#[test]
fn test_day_of_week_end_of_day() {
    let t = Instant::from_datetime(2024, 1, 1, 23, 59, 59.99998).unwrap();
    assert_eq!(t.day_of_week(), Weekday::Monday);
    assert_eq!(
        t.strftime("%a %A %w %Y-%m-%d").unwrap(),
        "Mon Monday 1 2024-01-01"
    );
    for (y, mo, d, wd) in [
        (2024, 1, 1, Weekday::Monday),
        (1858, 11, 17, Weekday::Wednesday), // MJD 0
        (1858, 11, 16, Weekday::Tuesday),
        (2000, 1, 1, Weekday::Saturday),
        (1970, 1, 1, Weekday::Thursday),
        (2016, 12, 31, Weekday::Saturday), // ends in a leap second
        (1, 1, 1, Weekday::Monday),        // proleptic Gregorian
        (-4713, 11, 24, Weekday::Monday),  // JD 0
    ] {
        let day = Instant::from_date(y, mo, d).unwrap();
        let next = day.add_utc_days(1.0);
        for us in [0, 1, 40, 999_999, 86_399_999_960, 86_399_999_999] {
            let t = day + Duration::from_microseconds(us);
            assert!(t < next);
            assert_eq!(t.day_of_week(), wd, "{t}");
        }
        assert_ne!(next.day_of_week(), wd, "{next}");
    }
    // Inside the leap second it is still Saturday 2016-12-31
    let leap = Instant::from_datetime(2016, 12, 31, 23, 59, 60.999999).unwrap();
    assert_eq!(leap.day_of_week(), Weekday::Saturday);
}

/// `utc_day_number` (and so `day_of_week`) saturates for extreme instants
/// instead of overflowing (a debug-build panic).
#[test]
fn test_utc_day_number_saturates() {
    for t in [
        Instant::from_mjd_utc(1.0e20),
        Instant::from_mjd_utc(-1.0e20),
        Instant::INVALID,
        Instant::new(i64::MAX),
        Instant::new(i64::MIN + 1),
    ] {
        let _ = t.utc_day_number();
        assert_ne!(t.day_of_week(), Weekday::Invalid);
    }
    assert_eq!(
        Instant::from_mjd_utc(1.0e20).utc_day_number(),
        i64::MAX.div_euclid(86_400_000_000)
    );
}
/// The calendar is exact (Euclidean division) over the whole i64 range,
/// including before −4712, where truncating division gave negative days.
#[test]
fn test_calendar_far_past() {
    for (y, mo, d) in [
        (-4716, 1, 1),
        (-4713, 11, 24),
        (-4800, 2, 29),
        (-10_000, 12, 31),
        (-200_000, 3, 1),
        (0, 2, 29),
        (-1, 12, 31),
        (200_000, 6, 15),
    ] {
        let t = Instant::from_date(y, mo, d).unwrap();
        assert_eq!(t.as_datetime(), (y, mo, d, 0, 0, 0.0), "{y}-{mo}-{d}");
    }
    assert_eq!(
        Instant::from_date(-4716, 1, 1).unwrap().to_string(),
        "-4716-01-01T00:00:00.000000Z"
    );
    // JD 0 is −4713-11-24 12:00 in the proleptic Gregorian calendar
    let jd0 = Instant::from_datetime(-4713, 11, 24, 12, 0, 0.0).unwrap();
    assert_eq!(jd0.as_jd_utc(), 0.0);
    // Consecutive days across the far past stay consecutive
    let mut t = Instant::from_date(-5000, 1, 1).unwrap();
    let mut prev = t.as_datetime();
    for _ in 0..800 {
        t = t.add_utc_days(1.0);
        let cur = t.as_datetime();
        assert!(
            (cur.0, cur.1, cur.2) > (prev.0, prev.1, prev.2),
            "{prev:?} -> {cur:?}"
        );
        assert!((1..=12).contains(&cur.1) && (1..=31).contains(&cur.2));
        prev = cur;
    }
    // Even INVALID (i64::MIN) has an in-range label
    let (_, mo, d, ..) = Instant::INVALID.as_datetime();
    assert!(
        (1..=12).contains(&mo) && (1..=31).contains(&d),
        "{}",
        Instant::INVALID
    );
}

/// `as_rfc3339` writes a 4-digit year (ISO 8601 expanded `±YYYY…` outside
/// 0000–9999) and `from_rfc3339` reads every one back.
#[test]
fn test_rfc3339_year_roundtrip() {
    for (y, s) in [
        (999, "0999-01-01T00:00:00.000000Z"),
        (1, "0001-01-01T00:00:00.000000Z"),
        (0, "0000-01-01T00:00:00.000000Z"),
        (-1, "-0001-01-01T00:00:00.000000Z"),
        (-4716, "-4716-01-01T00:00:00.000000Z"),
        (10_000, "+10000-01-01T00:00:00.000000Z"),
    ] {
        let t = Instant::from_date(y, 1, 1).unwrap();
        assert_eq!(t.as_rfc3339(), s);
        assert_eq!(t.to_string(), s);
        assert_eq!(Instant::from_rfc3339(s).unwrap(), t, "{s}");
        assert_eq!(
            Instant::strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").unwrap(),
            t,
            "{s}"
        );
    }
    // A sign needs at least 4 digits; unsigned years are exactly 4
    assert!(Instant::from_rfc3339("-001-01-01T00:00:00Z").is_err());
    assert!(Instant::from_rfc3339("999-01-01T00:00:00Z").is_err());
    assert!(Instant::from_rfc3339("10000-01-01T00:00:00Z").is_err());
}

/// Fractions beyond 6 digits round to the nearest microsecond and carry,
/// as `from_datetime` does: into the next minute, into a leap second where
/// one follows, and from the end of a leap second into the next day.
#[test]
fn test_parsed_fraction_rounds_and_carries() {
    for (s, want) in [
        (
            "2024-01-01T12:00:00.1234565Z",
            utc_label(2024, 1, 1, 12, 0, 0.123457),
        ),
        (
            "2024-01-01T12:00:00.12345649Z",
            utc_label(2024, 1, 1, 12, 0, 0.123456),
        ),
        (
            "2024-01-01T00:00:00.0002489999Z",
            utc_label(2024, 1, 1, 0, 0, 0.000249),
        ),
        (
            "2024-01-01T12:00:59.9999995Z",
            utc_label(2024, 1, 1, 12, 1, 0.0),
        ),
        (
            "2024-12-31T23:59:59.9999999Z",
            utc_label(2025, 1, 1, 0, 0, 0.0),
        ),
        (
            "2016-12-31T23:59:59.9999999Z",
            utc_label(2016, 12, 31, 23, 59, 60.0),
        ),
        (
            "2016-12-31T23:59:60.9999996Z",
            utc_label(2017, 1, 1, 0, 0, 0.0),
        ),
        // With an offset: into the UTC leap second, and past a minute
        (
            "2017-01-01T00:59:59.9999999+01:00",
            utc_label(2016, 12, 31, 23, 59, 60.0),
        ),
        (
            "2024-01-01T12:59:59.9999999+01:00",
            utc_label(2024, 1, 1, 12, 0, 0.0),
        ),
    ] {
        assert_eq!(Instant::from_rfc3339(s).unwrap(), want, "{s}");
        let spaced = s.replacen('T', " ", 1);
        assert_eq!(Instant::from_string(&spaced).unwrap(), want, "{spaced}");
    }
    assert_eq!(
        Instant::strptime("2016-12-31 23:59:60.99999951", "%Y-%m-%d %H:%M:%S.%f").unwrap(),
        utc_label(2017, 1, 1, 0, 0, 0.0)
    );
    // Rounding never makes a label valid that is not: no leap second here
    assert!(Instant::from_rfc3339("2024-12-31T23:59:60.0000001Z").is_err());
}
