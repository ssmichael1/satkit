//! The build script (`build.rs`) embeds only the git hash and tag, and reads
//! git only when the crate is the top level of its own checkout.
//!
//! These tests compile `build.rs` on its own with `rustc` and run it with
//! `CARGO_MANIFEST_DIR` pointing at scratch directories: a copy of satkit
//! vendored inside another git repository must report `"unknown"` (it used
//! to embed the outer repository's hash), and the output must not change
//! from run to run (it used to embed the build time).

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=satkit",
            "-c",
            "user.email=satkit@example.com",
        ])
        .args(["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"])
        .args(args)
        .current_dir(dir)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .output()
        .expect("git runs");
    assert!(out.status.success(), "git {args:?} failed: {out:?}");
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

/// A fresh scratch directory; `None` when `git` or `rustc` is unavailable.
fn scratch(name: &str) -> Option<PathBuf> {
    for tool in ["git", "rustc"] {
        if Command::new(tool).arg("--version").output().is_err() {
            eprintln!("skipping: {tool} not found");
            return None;
        }
    }
    let dir = std::env::temp_dir().join(format!("satkit-build-meta-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Some(dir)
}

/// Compile `build.rs` into `dir` and return the binary.
fn compile_build_script(dir: &Path) -> PathBuf {
    let exe = dir.join("build-script");
    let status = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .args(["--edition", "2021", "-O", "-o"])
        .arg(&exe)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
        .status()
        .expect("rustc runs");
    assert!(status.success());
    exe
}

/// Run the build script as if `manifest_dir` were the crate's directory.
/// `ceiling` stops git's repository search from leaving the scratch tree.
fn run(exe: &Path, manifest_dir: &Path, ceiling: &Path) -> Vec<String> {
    let out = Command::new(exe)
        .current_dir(manifest_dir)
        .env("CARGO_MANIFEST_DIR", manifest_dir)
        .env("GIT_CEILING_DIRECTORIES", ceiling)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn env_lines(out: &[String]) -> Vec<&str> {
    out.iter()
        .filter(|l| l.starts_with("cargo:rustc-env="))
        .map(String::as_str)
        .collect()
}

#[test]
fn vendored_copy_reports_unknown() {
    let Some(root) = scratch("vendored") else {
        return;
    };
    let exe = compile_build_script(&root);
    let outer = root.join("outer");
    let crate_dir = outer.join("vendor").join("satkit");
    std::fs::create_dir_all(&crate_dir).unwrap();
    git(&outer, &["init", "-q"]);
    git(&outer, &["commit", "-q", "--allow-empty", "-m", "outer"]);
    git(&outer, &["tag", "outer-tag"]);

    let out = run(&exe, &crate_dir, &root);
    assert_eq!(
        env_lines(&out),
        [
            "cargo:rustc-env=GIT_HASH=unknown",
            "cargo:rustc-env=GIT_TAG=unknown"
        ]
    );
    // Nothing of the outer repository is watched.
    let watched: Vec<_> = out
        .iter()
        .filter(|l| l.starts_with("cargo:rerun-if-changed="))
        .collect();
    assert_eq!(watched, ["cargo:rerun-if-changed=build.rs"]);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn no_repository_reports_unknown() {
    let Some(root) = scratch("plain") else {
        return;
    };
    let exe = compile_build_script(&root);
    let crate_dir = root.join("satkit");
    std::fs::create_dir_all(&crate_dir).unwrap();

    let out = run(&exe, &crate_dir, &root);
    assert_eq!(
        env_lines(&out),
        [
            "cargo:rustc-env=GIT_HASH=unknown",
            "cargo:rustc-env=GIT_TAG=unknown"
        ]
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn own_checkout_reports_hash_and_is_reproducible() {
    let Some(root) = scratch("own") else {
        return;
    };
    let exe = compile_build_script(&root);
    let crate_dir = root.join("satkit");
    std::fs::create_dir_all(&crate_dir).unwrap();
    git(&crate_dir, &["init", "-q"]);
    git(&crate_dir, &["commit", "-q", "--allow-empty", "-m", "one"]);
    git(&crate_dir, &["tag", "v9.9.9"]);
    let head = git(&crate_dir, &["rev-parse", "HEAD"]);

    let out = run(&exe, &crate_dir, &root);
    assert_eq!(
        env_lines(&out),
        [
            format!("cargo:rustc-env=GIT_HASH={head}"),
            "cargo:rustc-env=GIT_TAG=v9.9.9".to_string(),
        ]
    );
    // HEAD and the branch it points to are watched, so a new commit reruns
    // the script.
    let branch = git(&crate_dir, &["symbolic-ref", "HEAD"]);
    for f in ["HEAD", branch.as_str()] {
        assert!(
            out.iter().any(|l| l.starts_with("cargo:rerun-if-changed=")
                && Path::new(&l["cargo:rerun-if-changed=".len()..])
                    .ends_with(Path::new(".git").join(f))),
            "{f} not watched: {out:?}"
        );
    }
    // No build time or other per-run value.
    assert_eq!(out, run(&exe, &crate_dir, &root));

    git(&crate_dir, &["commit", "-q", "--allow-empty", "-m", "two"]);
    let head2 = git(&crate_dir, &["rev-parse", "HEAD"]);
    let out2 = run(&exe, &crate_dir, &root);
    assert!(out2.contains(&format!("cargo:rustc-env=GIT_HASH={head2}")));
    let _ = std::fs::remove_dir_all(&root);
}

/// The embedded values are either real or `"unknown"`.
#[test]
fn githash_is_hex_or_unknown() {
    let h = satkit::utils::githash();
    assert!(
        h == "unknown" || (h.len() == 40 && h.chars().all(|c| c.is_ascii_hexdigit())),
        "{h}"
    );
    assert!(!satkit::utils::gittag().is_empty());
}
