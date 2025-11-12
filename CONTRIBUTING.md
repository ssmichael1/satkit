# Contributing to Satkit

Thank you for your interest in contributing to Satkit! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- **Rust**: Install the latest stable Rust toolchain from [rustup.rs](https://rustup.rs/)
- **Python**: Python 3.8 or later for Python bindings testing
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

2. **Download test vectors and data files**:
   ```bash
   python -m pip install requests
   python ./python/test/download_testvecs.py
   ```

3. **Build the project**:
   ```bash
   cargo build
   ```

4. **Run the tests**:
   ```bash
   cargo test
   ```

## Development Workflow

### Making Changes

1. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Test your changes**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
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
- Provide type hints in `.pyi` stub files for IDE support
- Include docstrings for all public functions and classes
- Test Python bindings separately when making changes ; see `python/test/test.py`

### Testing

- Write unit tests for all new functionality
- Add integration tests for complex features
- Ensure existing tests continue to pass
- Add test cases from published references when available (Vallado, JPL, etc.)

### Documentation

- Update relevant documentation for API changes
- Add examples for new features
- Keep README.md up to date with new capabilities
- Document any breaking changes clearly

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

We welcome contributions in these areas:

#### High Priority
- Bug fixes and correctness improvements
- Performance optimizations
- Additional test coverage
- Documentation enhancements

#### New Features

- See issues in github page for current list

## Building Python Bindings

To build and test the Python package locally:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install build dependencies and test
pip install setuptools setuptools-rust setuptools-scm pytest

# Build and install in development mode
pip install -e .

# Run Python tests
python -m pytest python/test/test.py
```

When you're done developing, deactivate the virtual environment:

```bash
deactivate
```

## Running Tests

### Full Test Suite

```bash
cargo test
```

### Specific Test Module

```bash
cargo test ode::ode_tests
```

### With Output

```bash
cargo test -- --nocapture
```

## Continuous Integration

All pull requests are automatically tested via GitHub Actions:

- **Build**: Compiles on Linux, macOS, and Windows
- **Test**: Runs full test suite on all platforms
- **Lint**: Checks code style with clippy
- **Format**: Verifies formatting with rustfmt
- **Python**: Tests Python bindings on all supported versions (3.8-3.14)

Ensure all CI checks pass before requesting review.

## Code Review Process

1. Maintainers will review your pull request
2. Address any feedback or requested changes
3. Once approved, maintainers will merge your contribution
4. Your changes will be included in the next release

## Licensing

By contributing to Satkit, you agree that your contributions will be licensed under the MIT License, the same license as the project.

## Questions?

- **Open a GitHub Issue** at [github.com/ssmichael1/satkit/issues](https://github.com/ssmichael1/satkit/issues) for questions about contributing
- Email the maintainer: ssmichael@gmail.com
- Review existing issues and pull requests for examples
- Check the documentation at [satellite-toolkit.readthedocs.io](https://satellite-toolkit.readthedocs.io/)

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Documentation](https://pyo3.rs/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

Thank you for contributing to Satkit! 🛰️
