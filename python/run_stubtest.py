"""Run ``mypy.stubtest`` over every satkit stub file, native submodules included.

The compiled extension creates ``density``, ``frametransform``, ``jplephem``,
``moon``, ``planets``, ``spaceweather``, ``sun`` and ``utils`` as attributes of
``satkit.satkit`` and registers them in ``sys.modules`` as ``satkit.<name>``
when ``satkit`` is imported. This wrapper registers them again (harmless)
and lists every stub explicitly, so stubtest checks each
``python/satkit/<name>.pyi`` against its runtime module.

Usage (from the repository root, after ``pip install -e ".[test]"``)::

    python python/run_stubtest.py

Extra arguments are passed through to stubtest (e.g. ``--generate-allowlist``).
"""

import pathlib
import sys
import types

import satkit
import satkit.satkit
from mypy import stubtest

# The submodules the extension creates (sun, utils, ...), read from the
# extension itself so a new one is checked without editing this file.
NATIVE_SUBMODULES = tuple(
    sorted(n for n, m in vars(satkit.satkit).items() if isinstance(m, types.ModuleType))
)

ALLOWLIST = pathlib.Path(__file__).resolve().parent / "stubtest_allowlist.txt"


def main() -> int:
    for name in NATIVE_SUBMODULES:
        sys.modules[f"satkit.{name}"] = getattr(satkit.satkit, name)
    args = [
        "satkit",
        "--allowlist",
        str(ALLOWLIST),
        # Operator dunders are positional-only at runtime but not in the stubs.
        "--ignore-positional-only",
        *sys.argv[1:],
    ]
    return stubtest.test_stubs(stubtest.parse_options(args))


if __name__ == "__main__":
    sys.exit(main())
