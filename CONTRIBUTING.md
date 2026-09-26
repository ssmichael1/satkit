# Contributing to Satkit

Thank you for your interest in contributing to Satkit! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- **Rust**: Install the latest stable Rust toolchain from [rustup.rs](https://rustup.rs/)
- **Python**: Python 3.10 or later for Python bindings testing
- **Git**: For version control

### Setting Up Your Development Environment

1. **Fork and clone the repository**:
   - First, fork the repository on GitHub by clicking the "Fork" button at [github.com/ssmichael1/satkit](https://github.com/ssmichael1/satkit)
   - Then clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/satkit.git
   cd satkit
   ```
   - Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/ssmichael1/satkit.git
   ```

2. **Download the test data, build and run the tests** as described in
   [Running Tests](#running-tests).

## Development Workflow

### Making Changes

1. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Test your changes**:
   ```bash
   cargo test --features chrono
   cargo clippy --workspace --all-targets -- -D warnings
   cargo fmt --all -- --check
   ```

4. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub with a clear description of your changes

## Code Style and Standards

### Rust Code

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` to format your code (runs automatically in CI)
- Ensure `cargo clippy` passes without warnings
- Write documentation comments (`///`) for all public APIs
- Include examples in documentation where appropriate

### Python Bindings

- Follow [PEP 8](https://pep8.org/) style guidelines
- Provide type hints in `.pyi` stub files for IDE support. CI checks them
  against the compiled bindings with `python python/run_stubtest.py`
  (mypy's stubtest over every stub, native submodules included); the stub is
  wrong when they disagree, unless the entry in
  `python/stubtest_allowlist.txt` says why not. stubtest compares parameter
  names, kinds and defaults only
- Give every keyword a real PyO3 signature
  (`#[pyo3(signature = (pos, *, degree=6, order=None))]`), never a hand-parsed
  `**kwargs`: Python then rejects a misspelt keyword with `TypeError` and
  stubtest can check it. A value of the wrong type that needs its own message
  goes through `#[pyo3(from_py_with = ...)]` (see `arg_extractor!` in
  `python/src/pyutils.rs`)
- Include docstrings for all public functions and classes
- Conversions to another representation are methods named `to_X` (`to_mjd()`,
  `to_datetime()`, `to_rotation_matrix()`), paired with the `from_X` constructor
  that inverts them. Do not introduce `as_X` names in the Python API — `as_*`
  is the Rust core's convention, and the Python `as_*` aliases are deprecated
- Test Python bindings separately when making changes; see `python/test/` (`test_*.py`)
- Property vs method: **a property tells you something about the object; a
  method gives you something to use instead of it.** After reading a property
  (`#[getter]`) you are still working with the object and the result describes
  it: `x`, `angle`, `norm`, `period`, `day_of_year`, `weekday`, `converged`,
  `latitude_deg`, a position's `geodetic` form or its local `qenu2itrf` frame.
  A method's result replaces the object, in one of two ways: another instance
  of the same type (`inverse()`, `conj()`, `normalize()`), or a re-encoding for
  something else to consume (`to_rotation_matrix()` for numpy, `to_iso8601()`
  for a file, `to_omm()` for JSON). Re-encodings always return a non-satkit
  type and are always named `to_*` (the deprecated `as_*` aliases aside), so
  the call site reads as a conversion. Every zero-argument member is therefore a property unless it
  returns its own type or carries a conversion prefix; `test_api_shape.py`
  enforces this from the stub. Properties must be cheap and pure (no I/O, no
  allocation beyond the return value). There is no deprecation path between
  the two, so decide before the first release.

### Testing

- Write unit tests for all new functionality
- Add integration tests for complex features
- Ensure existing tests continue to pass
- Add test cases from published references when available (Vallado, JPL, etc.)

### Documentation

- Update relevant documentation for API changes
- Add examples for new features. Python examples are executed by
  `python/test/test_doc_examples.py`: every fenced ```` ```python ```` block
  in `docs/**/*.md` and `README.md` (a page's blocks run in order in one
  namespace), and the fenced or `>>>` code of every docstring, in the `.pyi`
  stubs and in the `///` comments of `python/src/*.rs` (each docstring runs
  on its own, with `satkit`, `sk` and `np` already imported; `>>>` output
  lines are not compared). Notebooks are executed by the docs build instead.
  Examples run with `SATKIT_OFFLINE=1`, in a scratch directory. Mark a block
  that cannot run there explicitly, on the line before its fence:
  - `<!-- skip-test: reason -->` — not run (it needs the network, a
    download, or a file the reader supplies)
  - `<!-- xfail-test: reason -->` — runs and must fail; for an example that
    exposes a known code bug (remove the marker with the fix)
  - `<!-- test-setup` … `-->` — hidden code run at that point of the page,
    for a fragment that uses names the prose defines elsewhere; the rendered
    page does not show it

  For docstrings the same markers live in `DOCSTRING_MARKS` and
  `DOCSTRING_SETUP` at the top of `test_doc_examples.py`, keyed by
  qualified name (`satkit.TLE.from_url`; `#n` selects one example of
  several). Nothing is skipped automatically. `SATKIT_OFFLINE=1` stops data
  file downloads but not `TLE.from_url` / `omm_from_url`, so mark every
  example that fetches anything
- Keep README.md up to date with new capabilities
- Document any breaking changes clearly

### Changelog

- Add **one user-facing line** per pull request under `## Unreleased` in
  `CHANGELOG.md`, in the matching `Added` / `Changed` / `Deprecated` /
  `Fixed` / `Docs` / `CI` / `Tests` section, ending with the PR link, e.g. `([#127](https://github.com/ssmichael1/satkit/pull/127))`.
- Keep it to two lines at most: what changed and, if useful, the one number
  that says why it matters. Implementation detail, measurements and rationale
  belong in the pull-request description — PRs are squash-merged, so that
  description becomes the commit message on `main` and is the permanent record.
- Mark breaking changes in bold (`**Breaking:**`).
- At release time, rename `Unreleased` to the version and date and prune the
  file so it keeps the last five releases. Older entries stay in git history
  (`git show vX.Y.Z:CHANGELOG.md`); there is no archive file.

### Releasing

1. Refresh dependencies: `cargo update`, then the full Rust and Python test
   suites. The committed `Cargo.lock` freezes the dependency set between
   releases, so this is the point where new crate versions get exercised;
   the refreshed lockfile ships with the release (it is in the sdist, and
   conda-forge builds with `--locked`).
2. Bump `version` in `Cargo.toml`, `python/Cargo.toml` and `pyproject.toml`
   (the release workflow rejects a tag whose version does not match all
   three), then `cargo build` so `Cargo.lock` records the new version.
3. Roll the changelog as described above.
4. Open the release PR and merge it; tag `vX.Y.Z` on `main` and push the
   tag. The release workflow publishes to crates.io and PyPI and creates the
   GitHub release. The conda-forge package is built from the PyPI sdist by
   [conda-forge/satkit-feedstock](https://github.com/conda-forge/satkit-feedstock):
   the autotick bot opens a version-bump PR there within a day of the PyPI
   release; check the dependencies it detected (it only updates the version
   and sha256) and merge it once CI is green.

## Types of Contributions

### Bug Reports

When you encounter a bug, please help us fix it by:

1. **Opening a GitHub Issue** at [github.com/ssmichael1/satkit/issues](https://github.com/ssmichael1/satkit/issues)
2. Include a minimal reproducible example
3. Specify your environment (OS, Rust version, Python version if applicable)
4. Describe expected vs actual behavior
5. Add relevant error messages or stack traces

### Feature Requests

If you have a suggestion for a feature:

1. **Open a GitHub Issue** at [github.com/ssmichael1/satkit/issues](https://github.com/ssmichael1/satkit/issues)
2. Use a clear, descriptive title
3. Explain the use case and potential benefits
4. Describe the proposed solution or API
5. Consider backward compatibility
6. Tag the issue with the `enhancement` label if possible

**Note**: Please open an issue to discuss significant new features before implementing them. This helps ensure alignment with project goals and avoids duplicated effort.

### Code Contributions

Bug fixes, correctness and performance improvements, test coverage and
documentation are all welcome; the [issue tracker](https://github.com/ssmichael1/satkit/issues)
lists open feature work.

## Running Tests

The tests need the data files and the reference test vectors:

```bash
pip install requests
python python/test/download_data.py astro-data
python python/test/download_testvecs.py satkit-testvecs
export SATKIT_DATA=astro-data SATKIT_TESTVEC_ROOT=satkit-testvecs
```

```bash
cargo test --features chrono         # the full Rust suite, as CI runs it
cargo test --test gmat_regression    # one integration-test file
cargo test <name> -- --nocapture     # tests whose name contains <name>, with output

# Python, in a virtual environment (builds the extension; needs the Rust toolchain)
pip install -e ".[test]"
pytest python/test/
python python/run_stubtest.py        # .pyi stubs against the compiled bindings
```

The GMAT regression tests need only the data files; their reference
trajectories are checked in.

## Continuous Integration

Every pull request runs the Build workflow (`.github/workflows/build.yml`):

- **rustfmt** and **clippy** (warnings are errors)
- **cargo deny**: RustSec advisories, dependency licences and sources (`deny.toml`)
- **sdist**: builds the source distribution and installs from it
- **Rust tests** on Linux, macOS and Windows, plus an offline run with an empty data directory, and `cargo doc`
- **Python tests** (Python 3.13): the full `pytest` suite, including the documentation and docstring examples, an offline smoke test, and stubtest

The docs site is rebuilt from `main`; a weekly schedule runs `cargo audit` and
a deep property-test pass; release builds smoke-test each wheel (3.10–3.14).
Ensure all CI checks pass before requesting review.

## Code Review Process

1. Maintainers will review your pull request
2. Address any feedback or requested changes
3. Once approved, maintainers will squash-merge your contribution (one commit per PR; the PR description becomes the commit message)
4. Your changes will be included in the next release

## Licensing

Satkit is dual-licensed under the MIT license ([LICENSE-MIT](LICENSE-MIT)) and the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)). Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Satkit by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Questions?

- **Open a GitHub Issue** at [github.com/ssmichael1/satkit/issues](https://github.com/ssmichael1/satkit/issues) for questions about contributing
- Email the maintainer: ssmichael@gmail.com
- Review existing issues and pull requests for examples
- Check the documentation at [satkit.dev](https://satkit.dev/)

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Documentation](https://pyo3.rs/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

Thank you for contributing to Satkit! 🛰️
