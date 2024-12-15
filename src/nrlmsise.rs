use cty;

use crate::spaceweather;
use crate::Duration;
use crate::Instant;

/// Array containing the following magnetic values:
///   0 : daily AP
///   1 : 3 hr AP index for current time
///   2 : 3 hr AP index for 3 hrs before current time
///   3 : 3 hr AP index for 6 hrs before current time
///   4 : 3 hr AP index for 9 hrs before current time
///   5 : Average of eight 3 hr AP indicies from 12 to 33 hrs
///           prior to current time
///   6 : Average of eight 3 hr AP indicies from 36 to 57 hrs
///           prior to current time
#[allow(non_camel_case_types)]
#[repr(C)]
struct ap_array {
    a: [cty::c_double; 7],
}

///  
///   Switches: to turn on and off particular variations use these switches.
///   0 is off, 1 is on, and 2 is main effects off but cross terms on.
///
///   Standard values are 0 for switch 0 and 1 for switches 1 to 23. The
///   array "switches" needs to be set accordingly by the calling program.
//   The arrays sw and swc are set internally.
#[repr(C)]
#[allow(non_camel_case_types)]
struct nrlmsise_flags {
    switches: [cty::c_int; 24],
    sw: [cty::c_double; 24],
    swc: [cty::c_double; 24],
}

#[repr(C)]
#[allow(non_camel_case_types)]
struct nrlmsise_input {
    year: cty::c_int,     /* Year, currently ignored */
    day: cty::c_int,      /* day of year */
    sec: cty::c_double,   // seconds in day (UT)
    alt: cty::c_double,   // altitude in km
    g_lat: cty::c_double, // geodetic latitude (deg?)
    g_lon: cty::c_double, // geodetic longitude (deg?)
    lst: cty::c_double,   // local apparant solar time (hours)
    f107a: cty::c_double, // 81-day average of f10.7 flux
    f107: cty::c_double,  // daily F10.7 flux for previous day
    ap: cty::c_double,    // magnetic index (daily)
    ap_a: *const ap_array,
}

///   OUTPUT VARIABLES:
///      d[0] - HE NUMBER DENSITY(CM-3)
///      d[1] - O NUMBER DENSITY(CM-3)
///      d[2] - N2 NUMBER DENSITY(CM-3)
///      d[3] - O2 NUMBER DENSITY(CM-3)
///      d[4] - AR NUMBER DENSITY(CM-3)
///      d[5] - TOTAL MASS DENSITY(GM/CM3) [includes d[8] in td7d]
///      d[6] - H NUMBER DENSITY(CM-3)
///      d[7] - N NUMBER DENSITY(CM-3)
///      d[8] - Anomalous oxygen NUMBER DENSITY(CM-3)
///      t[0] - EXOSPHERIC TEMPERATURE
///      t[1] - TEMPERATURE AT ALT
#[repr(C)]
#[allow(non_camel_case_types)]
struct nrlmsise_output {
    d: [cty::c_double; 9],
    t: [cty::c_double; 2],
}

extern "C" {
    fn gtd7d(input: *mut nrlmsise_input, flags: *mut nrlmsise_flags, output: *mut nrlmsise_output);

    // Version below does not include atomic oxygen
    //fn gtd7(input: *mut nrlmsise_input, flags: *mut nrlmsise_flags, output: *mut nrlmsise_output);

}

///
/// NRL MSISE-00 model for atmosphere density
///
/// # Arguments
///
///   * `alt_km` -  Altitude in kilometers
///   * `lat_option` - Optional latitude in degrees (default: 0)
///   * `lon_option` - Optional longitude in detrees (default: 0)
///   * `time_option` -  Optional time, for when using space weather
///   * `use_spaceweather` -  Boolean to use space weather data
///
/// # Outputs
///
///   Tuple with two elements:      
///      * Atmosphere mass density in kg / m^3
///      * Temperature in Kelvin
///
pub fn nrlmsise(
    alt_km: f64,
    lat_option: Option<f64>,
    lon_option: Option<f64>,
    time_option: Option<Instant>,
    use_spaceweather: bool,
) -> (f64, f64) {
    let lat: f64 = lat_option.unwrap_or(0.0);
    let lon: f64 = lon_option.unwrap_or(0.0);
    let mut day_of_year: i32 = 1;
    let mut sec_of_day: f64 = 0.0;

    let mut f107a: f64 = 150.0;
    let mut f107: f64 = 150.0;
    let mut ap: f64 = 4.0;

    if time_option.is_some() {
        let time = time_option.unwrap();
        let (year, _mon, _day, dhour, dmin, dsec) = time.as_datetime();
        let fday: f64 = (time - Instant::from_date(year, 1, 1)).as_days() + 1.0;
        day_of_year = fday.floor() as i32;
        sec_of_day = (dhour as f64).mul_add(3600.0, dmin as f64 * 60.0) + dsec;

        if use_spaceweather {
            if let Ok(r) = spaceweather::get(time - Duration::from_days(1.0)) {
                f107a = r.f10p7_adj_c81;
                f107 = r.f10p7_adj;
                ap = r.ap_avg as f64;
            }
        }
    }

    let mut input: nrlmsise_input = nrlmsise_input {
        year: 2022,
        day: day_of_year,
        sec: sec_of_day,
        alt: alt_km,
        g_lat: lat,
        g_lon: lon,
        lst: sec_of_day / 3600.0 + lon / 15.0,
        f107a,
        f107,
        ap,
        ap_a: std::ptr::null(),
    };

    let mut flags: nrlmsise_flags = nrlmsise_flags {
        switches: [
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        sw: [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        swc: [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    };

    let mut output: nrlmsise_output = nrlmsise_output {
        d: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t: [0.0, 0.0],
    };
    unsafe {
        gtd7d(
            std::ptr::addr_of_mut!(input),
            std::ptr::addr_of_mut!(flags),
            std::ptr::addr_of_mut!(output),
        );
    }
    (output.d[5] * 1.0e3, output.t[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nrlmsise() {
        let tm: Instant =
            Instant::from_date(2010, 1, 1) + Duration::from_days(171.0 + 29000.0 / 86400.0);
        let (_density, _temperature) = nrlmsise(400.0, Some(60.0), Some(-70.0), Some(tm), true);
    }
}
