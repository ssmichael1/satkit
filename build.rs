use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // NRLMSISE-00 is now pure Rust (src/nrlmsise.rs), no C compilation needed

    // Record the git hash and tag at compile time. Nothing else here depends
    // on the build machine or clock, so builds are reproducible.
    //
    // Git is read only when this crate is the top level of its own git
    // checkout. Sdist / crates.io / tarball builds have no .git, and a copy
    // of satkit vendored inside another repository would otherwise pick up
    // that repository's hash; both report "unknown".
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let in_own_repo = git_output(&manifest_dir, &["rev-parse", "--show-toplevel"])
        .and_then(|top| std::fs::canonicalize(top).ok())
        .is_some_and(|top| std::fs::canonicalize(&manifest_dir).is_ok_and(|m| m == top));

    let (hash, tag) = if in_own_repo {
        watch_git_refs(&manifest_dir);
        (
            git_output(&manifest_dir, &["rev-parse", "HEAD"]),
            git_output(&manifest_dir, &["describe", "--tags"]),
        )
    } else {
        (None, None)
    };
    let unknown = || "unknown".to_string();
    println!("cargo:rustc-env=GIT_HASH={}", hash.unwrap_or_else(unknown));
    println!("cargo:rustc-env=GIT_TAG={}", tag.unwrap_or_else(unknown));
}

/// Rerun this script when HEAD moves or tags change. `--git-path` resolves
/// each file for plain checkouts and worktrees alike. Only existing paths are
/// watched: cargo reruns every build for a missing one. The reftable ref
/// backend keeps refs in `reftable/`, which is watched as a whole.
fn watch_git_refs(dir: &Path) {
    let mut paths = vec![
        "HEAD".to_string(),
        "packed-refs".to_string(),
        "refs/tags".to_string(),
        "reftable".to_string(),
    ];
    if let Some(r) = git_output(dir, &["symbolic-ref", "-q", "HEAD"]) {
        paths.push(r);
    }
    for p in paths {
        if let Some(path) = git_output(dir, &["rev-parse", "--git-path", &p]) {
            let path = dir.join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

fn git_output(dir: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .current_dir(dir)
        // Find the repository from `dir` alone, not from an outer build's
        // environment.
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
