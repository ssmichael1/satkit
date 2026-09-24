//! Assemble the three primary sources into the one table [`get`](super::get)
//! serves.
//!
//! Observed rows (GFZ) come first and win any overlap; the NOAA/SWPC 45-day
//! forecast continues at daily cadence from the day after the last observed
//! row; MSAFE takes over at monthly cadence after the last daily row. The
//! four 81-day F10.7 averages are then computed across the whole series,
//! because a centred average needs ±40 days and therefore reaches from the
//! observed record into the forecast — which is also what CelesTrak does.

use super::SpaceWeatherRecord;
use crate::{Duration, Instant};

/// Merge observed, daily-forecast and monthly-forecast rows into one
/// date-ordered table and fill the 81-day averages.
///
/// Any of the three may be empty. A monthly row whose month has already
/// started when the daily rows end is re-dated to the day after the last
/// daily row, so the table has no hole between the two cadences; a
/// 13-month-smoothed climatology is flat enough within a month for that to
/// be harmless, and the row's `data_type` still says what it is.
pub fn assemble(
    mut observed: Vec<SpaceWeatherRecord>,
    daily_forecast: Vec<SpaceWeatherRecord>,
    monthly_forecast: Vec<SpaceWeatherRecord>,
) -> Vec<SpaceWeatherRecord> {
    observed.sort_by(|a, b| a.date.partial_cmp(&b.date).unwrap());
    let mut table = observed;

    let after = |table: &[SpaceWeatherRecord]| table.last().map(|r| r.date.utc_day_number());

    // Daily forecast strictly after the observed record.
    let last_obs = after(&table);
    let mut daily: Vec<_> = daily_forecast
        .into_iter()
        .filter(|r| last_obs.is_none_or(|d| r.date.utc_day_number() > d))
        .collect();
    daily.sort_by(|a, b| a.date.partial_cmp(&b.date).unwrap());
    table.extend(daily);

    // Monthly forecast after the last daily row, with the straddling month
    // pulled forward so there is no gap.
    if let Some(last_daily) = after(&table) {
        let bridge_day = last_daily + 1;
        let mut monthly: Vec<_> = monthly_forecast
            .into_iter()
            .filter_map(|mut r| {
                let d = r.date.utc_day_number();
                if d > last_daily {
                    Some(r)
                } else if super::month_end_day(r.date) >= bridge_day {
                    // This month is still running when the daily rows stop.
                    r.date = r.date + Duration::from_days((bridge_day - d) as f64);
                    Some(r)
                } else {
                    None
                }
            })
            .collect();
        monthly.sort_by(|a, b| a.date.partial_cmp(&b.date).unwrap());
        table.extend(monthly);
    } else {
        let mut monthly = monthly_forecast;
        monthly.sort_by(|a, b| a.date.partial_cmp(&b.date).unwrap());
        table.extend(monthly);
    }

    fill_81day_averages(&mut table);
    table
}

/// Fill `f10p7_{obs,adj}_{c81,l81}` on every row.
///
/// `CENTER81` is the unweighted mean over `[t−40, t+40]` and `LAST81` over
/// `[t−80, t]`, both clipped to the table — the convention verified to
/// reproduce CelesTrak's published columns to the rounding digit. Between
/// monthly rows the daily series holds the most recent row's value, which is
/// also how [`get`](super::get) answers for those days.
pub fn fill_81day_averages(rows: &mut [SpaceWeatherRecord]) {
    let Some(first) = rows.first() else { return };
    let Some(last) = rows.last() else { return };
    let d0 = first.date.utc_day_number();
    let d1 = last.date.utc_day_number();
    let n = (d1 - d0 + 1) as usize;

    // Day-indexed step-held series, then prefix sums for O(1) window means.
    let mut obs = vec![0.0_f64; n];
    let mut adj = vec![0.0_f64; n];
    let mut ri = 0;
    let mut cur_obs = rows[0].f10p7_obs;
    let mut cur_adj = rows[0].f10p7_adj;
    for (i, day) in (d0..=d1).enumerate() {
        while ri + 1 < rows.len() && rows[ri + 1].date.utc_day_number() <= day {
            ri += 1;
            if rows[ri].f10p7_obs >= 0.0 {
                cur_obs = rows[ri].f10p7_obs;
            }
            if rows[ri].f10p7_adj >= 0.0 {
                cur_adj = rows[ri].f10p7_adj;
            }
        }
        obs[i] = cur_obs;
        adj[i] = cur_adj;
    }
    let prefix = |v: &[f64]| -> Vec<f64> {
        let mut p = Vec::with_capacity(v.len() + 1);
        p.push(0.0);
        for x in v {
            p.push(p.last().unwrap() + x);
        }
        p
    };
    let pobs = prefix(&obs);
    let padj = prefix(&adj);
    // Mean over day indices [a, b] inclusive, clipped to the table.
    let mean = |p: &[f64], a: i64, b: i64| -> f64 {
        let a = a.max(0) as usize;
        let b = (b.min(n as i64 - 1)) as usize;
        (p[b + 1] - p[a]) / (b + 1 - a) as f64
    };

    for r in rows.iter_mut() {
        let i = r.date.utc_day_number() - d0;
        r.f10p7_obs_c81 = mean(&pobs, i - 40, i + 40);
        r.f10p7_obs_l81 = mean(&pobs, i - 80, i);
        r.f10p7_adj_c81 = mean(&padj, i - 40, i + 40);
        r.f10p7_adj_l81 = mean(&padj, i - 80, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaceweather::SpaceWeatherDataType;

    fn row(y: i32, m: i32, d: i32, flux: f64, ty: SpaceWeatherDataType) -> SpaceWeatherRecord {
        SpaceWeatherRecord {
            date: Instant::from_date(y, m, d).unwrap(),
            bsrn: -1,
            nd: -1,
            data_type: ty,
            kp: [-1; 8],
            kp_sum: -1,
            ap: [10; 8],
            ap_avg: 10,
            cp: -1.0,
            c9: -1,
            isn: -1,
            f10p7_obs: flux,
            f10p7_adj: flux,
            f10p7_obs_c81: -1.0,
            f10p7_obs_l81: -1.0,
            f10p7_adj_c81: -1.0,
            f10p7_adj_l81: -1.0,
        }
    }

    #[test]
    fn test_ordering_and_overlap() {
        use SpaceWeatherDataType::*;
        let obs = vec![
            row(2026, 9, 1, 100.0, Observed),
            row(2026, 9, 2, 100.0, Observed),
        ];
        // forecast overlaps the observed record on 09-02: observed wins
        let daily = vec![
            row(2026, 9, 2, 999.0, PredictedDaily),
            row(2026, 9, 3, 110.0, PredictedDaily),
            row(2026, 9, 4, 110.0, PredictedDaily),
        ];
        let monthly = vec![
            row(2026, 9, 1, 120.0, PredictedMonthly), // straddles: pulled to 09-05
            row(2026, 10, 1, 130.0, PredictedMonthly),
        ];
        let t = assemble(obs, daily, monthly);
        let dates: Vec<_> = t
            .iter()
            .map(|r| (r.date.as_datetime().2, r.data_type))
            .collect();
        assert_eq!(
            dates,
            vec![
                (1, Observed),
                (2, Observed),
                (3, PredictedDaily),
                (4, PredictedDaily),
                (5, PredictedMonthly),
                (1, PredictedMonthly),
            ]
        );
        assert_eq!(
            t[1].f10p7_obs, 100.0,
            "observed row must not be overwritten"
        );
        assert_eq!(t[4].date, Instant::from_date(2026, 9, 5).unwrap());
        assert_eq!(t[4].f10p7_obs, 120.0);
    }

    #[test]
    fn test_81day_average_convention() {
        use SpaceWeatherDataType::Observed;
        // 200 days of a ramp: day i has flux i. Check the window formula.
        let mut rows: Vec<_> = (0..200)
            .map(|i| {
                let d = Instant::from_date(2020, 1, 1).unwrap() + Duration::from_days(i as f64);
                let mut r = row(2020, 1, 1, i as f64, Observed);
                r.date = d;
                r
            })
            .collect();
        fill_81day_averages(&mut rows);
        // interior day 100: centred mean of 60..=140 is 100, trailing of 20..=100 is 60
        assert!((rows[100].f10p7_obs_c81 - 100.0).abs() < 1e-9);
        assert!((rows[100].f10p7_obs_l81 - 60.0).abs() < 1e-9);
        // clipped at the start: day 0 centred is mean of 0..=40 = 20
        assert!((rows[0].f10p7_obs_c81 - 20.0).abs() < 1e-9);
        assert!((rows[0].f10p7_obs_l81 - 0.0).abs() < 1e-9);
        // clipped at the end: day 199 trailing is mean of 119..=199 = 159
        assert!((rows[199].f10p7_obs_l81 - 159.0).abs() < 1e-9);
    }

    #[test]
    fn test_averages_step_hold_across_monthly_rows() {
        use SpaceWeatherDataType::*;
        // one observed day then a monthly row 31 days later: the days in
        // between hold the observed value, then switch.
        let mut rows = vec![
            row(2026, 9, 1, 100.0, Observed),
            row(2026, 10, 2, 200.0, PredictedMonthly),
        ];
        fill_81day_averages(&mut rows);
        // trailing mean at the monthly row: 31 days of 100 + 1 day of 200
        let expect = (31.0 * 100.0 + 200.0) / 32.0;
        assert!((rows[1].f10p7_obs_l81 - expect).abs() < 1e-9);
    }
}
