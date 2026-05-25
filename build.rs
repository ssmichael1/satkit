use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // NRLMSISE-00 is now pure Rust (src/nrlmsise.rs), no C compilation needed

    // Record git hash to compile-time environment variable. Sdist /
    // tarball builds (PyPI source distributions, conda-forge builds from
    // a github archive, etc.) have no .git directory — fall back to
    // "unknown" rather than blowing up the build.
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        git_output(&["rev-parse", "HEAD"])
    );
    println!(
        "cargo:rustc-env=GIT_TAG={}",
        git_output(&["describe", "--tags"])
    );
    println!("cargo:rustc-env=BUILD_DATE={}", build_date_iso8601());
}

fn git_output(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn build_date_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // Civil-from-days, Howard Hinnant: https://howardhinnant.github.io/date_algorithms.html
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y0 = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y0 + 1 } else { y0 };

    let h = tod / 3600;
    let mi = (tod % 3600) / 60;
    let s = tod % 60;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, h, mi, s)
}
