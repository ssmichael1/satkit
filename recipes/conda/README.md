# conda-forge recipes

Two recipes, submitted together in
[conda-forge/staged-recipes#33466](https://github.com/conda-forge/staged-recipes/pull/33466)
(fork branch `ssmichael1/staged-recipes@add-satkit`). **This directory is the
source of truth**; the fork is a copy pushed from here.

| recipe | what it builds |
|---|---|
| `satkit/` | the library (Rust extension via `setuptools-rust`; PyPI sdist as source; bundles Rust crate licences with `cargo-bundle-licenses`). Since 0.21 it needs **no** data package: core tables are compiled in, the JPL ephemeris downloads on first use. The recipe test runs offline with an empty data directory. |
| `satkit-data/` | the optional offline bundle (`noarch: python`): DE421 + gravity models + IERS tables + leap seconds, laid out as `satkit_data/data/` so satkit finds it. Sources are the `data-v1` GitHub release assets, **generated** from `data/manifest.json` by `tools/conda_sources_from_manifest.py`. Ships DE421 (14 MB) not DE440 (102 MB) because of conda-forge's 100 MB cap. |

## Release-time checklist

1. Release satkit `X.Y.Z` to PyPI; then in `satkit/recipe.yaml` set `context.version`
   and the sdist `sha256` (`pip download satkit==X.Y.Z --no-binary :all: --no-deps -d /tmp/s && shasum -a 256 /tmp/s/*.tar.gz`).
2. If `data/manifest.json` changed (new `data-vN` release): `python tools/conda_sources_from_manifest.py --write`,
   bump `satkit-data` `context.version` (publish the PyPI `satkit-data` of that version first — the
   version must exist upstream), update `satkit-data/LICENSE` if files were added or removed.
   `python tools/conda_sources_from_manifest.py --check` must pass.
3. Lint locally: `conda-smithy recipe-lint --conda-forge recipes/satkit recipes/satkit-data`
   (in a staged-recipes layout) and `rattler-build build --recipe … --render-only`.
4. Copy both recipe directories to the fork (`recipes/satkit`, `recipes/satkit-data`), commit, push to
   `ssmichael1/staged-recipes@add-satkit`, confirm CI green, and reply on the PR (the reviewer is
   `eunos-1128`; the PR description's checklist should tick the v1-format box).
5. After the feedstocks exist, version bumps happen there (the bot opens PRs); keep this copy in sync.

Notes: `pip_check` is on for `satkit` (its only runtime dependency is numpy); off for `satkit-data`
because `build_data.py` lays files out directly with no dist-info. `msis21.parm` is deliberately not
shipped (NRL non-commercial licence; satkit does not include NRLMSIS 2).
