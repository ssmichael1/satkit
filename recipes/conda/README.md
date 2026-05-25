# conda-forge recipes

Draft conda-forge recipes for the satkit ecosystem. These are the
files that will be submitted to
[`conda-forge/staged-recipes`](https://github.com/conda-forge/staged-recipes)
to bootstrap the two feedstocks; iteration happens here in-tree so the
history travels with the source.

## Layout

| Path | Package | What it ships |
|---|---|---|
| [`satkit/meta.yaml`](satkit/meta.yaml) | `satkit` | The Python package — Rust extension built via PyO3 / setuptools-rust |
| [`satkit-data/meta.yaml`](satkit-data/meta.yaml) | `satkit-data` | Noarch Python data-only package: JPL DE421, gravity models, IERS tables, leap seconds, NRLMSIS parameters |

`satkit` depends on `satkit-data ≥ 0.10` at runtime, so installing
`satkit` from conda-forge will pull in `satkit-data` automatically.

## DE421 vs DE440 — different payloads, same name, same version

The two channel builds of `satkit-data` ship intentionally different
JPL ephemeris files:

| Channel | JPL file | Size | Span |
|---|---|---|---|
| PyPI | `linux_p1550p2650.440` (DE440 full) | ~98 MB | 1550–2650 |
| conda-forge | `lnxp1900p2053.421` (DE421) | ~13 MB | 1900–2053 |

The `satkit-data` *version* tracks the upstream PyPI version (currently
0.9.0); the JPL-file substitution is a **channel-specific build
choice, not a version bump**. Conda's run-dep solver works with this
because users on conda-forge get the conda-forge artifact and never
mix; users on PyPI get the PyPI wheel.

Why DE421 on conda-forge:

* conda-forge has a 100 MB soft cap per package; DE440 squeaks under
  it but reviewers grumble.
* For satellite-orbit work (Earth third-body, Sun/Moon ephemerides),
  the accuracy difference between DE421 and DE440 is unmeasurable —
  Sun/Moon positions agree at the sub-meter level at modern epochs.
* The 1900–2053 span covers virtually every active satellite use case.

Callers who need longer span or higher outer-planet precision can
install DE440 via `satkit.utils.update_datafiles()` post-install (it
lands in `datadir()` alongside DE421; the autodetect in
`jplephem::resolve_default_path` picks the highest DE-version
available).

## EOP / space-weather

`EOP-All.csv` and `SW-All.csv` are **not** bundled in the conda data
package. CelesTrak republishes them daily and shipping a stale
snapshot would mislead users. After install, run:

```python
import satkit
satkit.utils.update_datafiles()
```

…to download the current copies into `datadir()`.

## Local validation

Before submitting to staged-recipes, validate each recipe builds:

```bash
# conda-build (legacy meta.yaml format these recipes use)
conda build recipes/conda/satkit-data
conda build recipes/conda/satkit

# rattler-build (newer alternative)
rattler-build build --recipe recipes/conda/satkit-data/meta.yaml
rattler-build build --recipe recipes/conda/satkit/meta.yaml
```

The `satkit` recipe pulls the 0.18.0 sdist from PyPI directly. When
bumping for a future release, refresh the `sha256:` via
`pip hash satkit-<version>.tar.gz` against the new tarball.

## Submission

When ready, fork
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes),
copy each recipe into the fork's `recipes/<name>/` directory, and open
a single PR with both. Reviewers will then split out the two
feedstocks (`satkit-feedstock`, `satkit-data-feedstock`) on merge.
