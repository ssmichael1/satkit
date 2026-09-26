//! GFZ Potsdam `Kp_ap_Ap_SN_F107_since_1932.txt` reader — the observed record.
//!
//! The primary source for the measured geomagnetic indices: GFZ Helmholtz
//! Centre produces the IAGA-endorsed Kp/ap series, and redistributes the
//! DRAO / Natural Resources Canada F10.7 flux alongside it. Licensed
//! **CC BY 4.0** (Matzka et al. 2021), and daily from 1932.
//!
//! One column is deliberately **not** ingested: the sunspot number `SN` is
//! CC BY-**NC** 4.0 (WDC-SILSO, Royal Observatory of Belgium), a licence
//! satkit does not carry. NRLMSISE-00 does not use it, so [`isn`] is left at
//! the `-1` sentinel.
//!
//! The file publishes raw daily measurements only. Three fields CelesTrak
//! derives are computed here — `kp_sum`, `cp` and `c9` — and the four 81-day
//! averages are filled later by [`assemble`](super::assemble), because a
//! centred average needs ±40 days and so reaches into the forecast.
//!
//! [`isn`]: super::SpaceWeatherRecord::isn

use super::{Error, Result, SpaceWeatherDataType, SpaceWeatherRecord};
use crate::Instant;

/// Bartels planetary character figure, as a step table on `sum(ap1..ap8)`.
///
/// `(threshold, Cp, C9)`: the row with the largest threshold not exceeding
/// the daily ap sum applies. Transcribed from CelesTrak's published values
/// and verified to reproduce `CP`/`C9` on all 25195 observed rows of
/// `SW-All.csv`.
const CP_TABLE: [(i32, f64, i32); 24] = [
    (0, 0.0, 0),
    (23, 0.1, 0),
    (35, 0.2, 1),
    (45, 0.3, 1),
    (56, 0.4, 2),
    (67, 0.5, 2),
    (79, 0.6, 3),
    (91, 0.7, 3),
    (105, 0.8, 4),
    (121, 0.9, 4),
    (140, 1.0, 5),
    (165, 1.1, 5),
    (191, 1.2, 6),
    (229, 1.3, 6),
    (274, 1.4, 6),
    (321, 1.5, 7),
    (380, 1.6, 7),
    (454, 1.7, 7),
    (562, 1.8, 7),
    (730, 1.9, 8),
    (1135, 2.0, 9),
    (1425, 2.1, 9),
    (1887, 2.2, 9),
    (2166, 2.3, 9),
];

/// `(Cp, C9)` for a daily ap sum.
pub(super) fn cp_c9(ap_sum: i32) -> (f64, i32) {
    let mut out = CP_TABLE[0];
    for row in CP_TABLE {
        if ap_sum >= row.0 {
            out = row;
        } else {
            break;
        }
    }
    (out.1, out.2)
}

/// Kp in thirds of a unit (`3+` = 3⅓ → 10), from GFZ's decimal form.
///
/// GFZ writes Kp as a decimal (`3.333`); CelesTrak tabulates the same value
/// ×10 with the fractional third in the units digit (`33`). Both encodings
/// are exact on thirds, so the conversion goes through the integer count.
fn kp_thirds(kp: f64) -> Option<i32> {
    if kp < 0.0 {
        return None;
    }
    Some((kp * 3.0).round() as i32)
}

/// CelesTrak's ×10 tabulated Kp, from a count of thirds.
fn thirds_to_tabulated(thirds: i32) -> i32 {
    // `3+` = 10 thirds -> base 3, remainder 1 -> 33; `4-` = 11 -> 37.
    let base = thirds / 3;
    let frac = match thirds % 3 {
        0 => 0,
        1 => 3,
        _ => 7,
    };
    base * 10 + frac
}

/// Parse the GFZ `Kp_ap_Ap_SN_F107_since_1932.txt` table.
///
/// The 81-day averages are left at `-1`; [`assemble`](super::assemble) fills
/// them once the forecast rows are in place.
pub fn parse(text: &str) -> Result<Vec<SpaceWeatherRecord>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        // YYYY MM DD days days_m Bsr dB Kp1-8 ap1-8 Ap SN F10.7obs F10.7adj D
        if f.len() < 28 {
            return Err(Error::InvalidEntry);
        }
        let num = |s: &str, what: &'static str| -> Result<f64> {
            s.parse::<f64>().map_err(|_| Error::InvalidNumber(what))
        };
        let int = |s: &str, what: &'static str| -> Result<i32> {
            s.parse::<i32>().map_err(|_| Error::InvalidNumber(what))
        };

        let date = Instant::from_date(
            int(f[0], "year")?,
            int(f[1], "month")?,
            int(f[2], "day of month")?,
        )?;

        let mut kp = [-1_i32; 8];
        let mut kp_thirds_sum = 0_i32;
        let mut kp_complete = true;
        for i in 0..8 {
            match kp_thirds(num(f[7 + i], "kp")?) {
                Some(th) => {
                    kp[i] = thirds_to_tabulated(th);
                    kp_thirds_sum += th;
                }
                None => kp_complete = false,
            }
        }
        // CelesTrak's KP_SUM is the true sum ×10, not the sum of the ×10
        // values: summing the tabulated form loses the thirds.
        let kp_sum = if kp_complete {
            ((kp_thirds_sum as f64) * 10.0 / 3.0).round() as i32
        } else {
            -1
        };

        let mut ap = [-1_i32; 8];
        let mut ap_sum = 0_i32;
        let mut ap_complete = true;
        for i in 0..8 {
            let v = int(f[15 + i], "ap")?;
            if v < 0 {
                ap_complete = false;
            } else {
                ap[i] = v;
                ap_sum += v;
            }
        }
        let (cp, c9) = if ap_complete {
            cp_c9(ap_sum)
        } else {
            (-1.0, -1)
        };

        // `D`: 0 = Kp and SN preliminary, 1 = Kp definitive, 2 = both.
        let data_type = match int(f[27], "definitive flag")? {
            0 => SpaceWeatherDataType::ObservedPreliminary,
            _ => SpaceWeatherDataType::Observed,
        };

        out.push(SpaceWeatherRecord {
            date,
            bsrn: int(f[5], "bsrn")?,
            nd: int(f[6], "nd")?,
            data_type,
            kp,
            kp_sum,
            ap,
            ap_avg: int(f[23], "ap_avg")?,
            cp,
            c9,
            // SN is CC BY-NC 4.0 (SILSO) and deliberately not ingested.
            isn: -1,
            f10p7_obs: num(f[25], "f10.7 observed")?,
            f10p7_adj: num(f[26], "f10.7 adjusted")?,
            f10p7_obs_c81: -1.0,
            f10p7_obs_l81: -1.0,
            f10p7_adj_c81: -1.0,
            f10p7_adj_l81: -1.0,
        });
    }
    if out.is_empty() {
        return Err(Error::InvalidEntry);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kp_encoding_round_trip() {
        // GFZ decimal -> thirds -> CelesTrak tabulated ×10.
        for (decimal, tabulated) in [
            (0.0, 0),
            (0.667, 7),
            (1.333, 13),
            (1.667, 17),
            (2.0, 20),
            (2.333, 23),
            (3.333, 33),
            (3.667, 37),
            (9.0, 90),
        ] {
            let th = kp_thirds(decimal).unwrap();
            assert_eq!(thirds_to_tabulated(th), tabulated, "Kp {decimal}");
        }
        assert_eq!(kp_thirds(-1.0), None);
    }

    #[test]
    fn test_kp_sum_uses_thirds_not_tabulated_values() {
        // 2026-09-12: Kp = 1.333 2.333 2.333 2.000 1.667 1.333 1.000 0.667
        // CelesTrak publishes KP_SUM = 127. Summing the ×10 values gives 126,
        // which is the bug this arithmetic exists to avoid.
        let kps = [1.333, 2.333, 2.333, 2.0, 1.667, 1.333, 1.0, 0.667];
        let thirds: i32 = kps.iter().map(|k| kp_thirds(*k).unwrap()).sum();
        assert_eq!(((thirds as f64) * 10.0 / 3.0).round() as i32, 127);
        let naive: i32 = kps
            .iter()
            .map(|k| thirds_to_tabulated(kp_thirds(*k).unwrap()))
            .sum();
        assert_eq!(naive, 126);
    }

    #[test]
    fn test_cp_c9_table() {
        assert_eq!(cp_c9(0), (0.0, 0));
        assert_eq!(cp_c9(22), (0.0, 0));
        assert_eq!(cp_c9(23), (0.1, 0));
        assert_eq!(cp_c9(34), (0.1, 0));
        assert_eq!(cp_c9(35), (0.2, 1));
        assert_eq!(cp_c9(140), (1.0, 5));
        assert_eq!(cp_c9(2166), (2.3, 9));
        assert_eq!(cp_c9(99999), (2.3, 9));
    }

    #[test]
    fn test_parse_row() {
        let text = "\
# comment line
2026 09 12 34588 34588.5 2633 11  1.333  2.333  2.333  2.000  1.667  1.333  1.000  0.667    5    9    9    7    6    5    4    3     6  92  109.9  111.8 2
";
        let recs = parse(text).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.date, Instant::from_date(2026, 9, 12).unwrap());
        assert_eq!(r.bsrn, 2633);
        assert_eq!(r.nd, 11);
        assert_eq!(r.kp, [13, 23, 23, 20, 17, 13, 10, 7]);
        assert_eq!(r.kp_sum, 127);
        assert_eq!(r.ap, [5, 9, 9, 7, 6, 5, 4, 3]);
        assert_eq!(r.ap_avg, 6);
        assert_eq!(r.f10p7_obs, 109.9);
        assert_eq!(r.data_type, SpaceWeatherDataType::Observed);
        // SN present in the file but deliberately not ingested (CC BY-NC).
        assert_eq!(r.isn, -1);
        // averages are filled by `assemble`, not here
        assert_eq!(r.f10p7_obs_c81, -1.0);
    }

    #[test]
    fn test_preliminary_flag() {
        let row = |d: &str| {
            format!("2026 09 12 34588 34588.5 2633 11  1.333  2.333  2.333  2.000  1.667  1.333  1.000  0.667    5    9    9    7    6    5    4    3     6  92  109.9  111.8 {d}\n")
        };
        assert_eq!(
            parse(&row("0")).unwrap()[0].data_type,
            SpaceWeatherDataType::ObservedPreliminary
        );
        assert_eq!(
            parse(&row("2")).unwrap()[0].data_type,
            SpaceWeatherDataType::Observed
        );
    }
}
