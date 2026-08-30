# conda-forge recipe

One recipe, submitted in
[conda-forge/staged-recipes#33466](https://github.com/conda-forge/staged-recipes/pull/33466)
(fork branch `ssmichael1/staged-recipes@add-satkit`). **This directory is the
source of truth**; the fork is a copy pushed from here.

`satkit/` builds the library (Rust extension via `setuptools-rust`; the PyPI
sdist is the source; Rust crate licences are bundled with
`cargo-bundle-licenses`). Since 0.21 it needs **no** data package: the IERS
tables and gravity models are compiled in and the JPL ephemeris downloads on
first use (SHA-256 verified) — see `docs/getting-started/datafiles.md`. The
recipe test runs offline with an empty data directory and no network.

There is deliberately no conda `satkit-data` package: satkit works without a
bundle, conda-forge's 100 MB cap would have forced a DE421-only bundle that
differed from the PyPI/GitHub content, and it would have needed a second
feedstock and a PyPI release before every bump. Offline conda users populate a
directory once with `satkit.utils.update_datafiles()` and set `SATKIT_DATA`,
or host a mirror and set `SATKIT_DATA_URL`.

## Release-time checklist

1. Release satkit `X.Y.Z` to PyPI; then in `satkit/recipe.yaml` set `context.version`
   and the sdist `sha256` (`pip download satkit==X.Y.Z --no-binary :all: --no-deps -d /tmp/s && shasum -a 256 /tmp/s/*.tar.gz`).
2. Lint locally: `conda-smithy recipe-lint --conda-forge recipes/satkit` (in a staged-recipes
   layout) and `rattler-build build --recipe recipes/satkit/recipe.yaml --render-only`.
3. Copy `recipes/satkit` to the fork, remove the fork's `recipes/satkit-data` if still present, commit,
   push to `ssmichael1/staged-recipes@add-satkit`, confirm CI green, and reply on the PR (the reviewer
   is `eunos-1128`; the PR description's checklist should tick the v1-format box).
4. After the feedstock exists, version bumps happen there (the bot opens PRs); keep this copy in sync.

Notes: `pip_check` is on (the only runtime dependency is numpy). `msis21.parm` is deliberately not
shipped anywhere (NRL non-commercial licence; satkit does not include NRLMSIS 2).
