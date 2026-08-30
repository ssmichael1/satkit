"""``import satkit`` must not break on any layout of the optional ``satkit_data`` bundle."""

import os
import types

import satkit


def test_namespace_package_without___file__(tmp_path):
    # conda's satkit-data (and any bare directory) imports as a namespace
    # package: __file__ is None, only __path__ is set.
    (tmp_path / "data").mkdir()
    mod = types.ModuleType("satkit_data")
    mod.__file__ = None
    mod.__path__ = [str(tmp_path)]
    assert satkit._optional_data_bundle_dirs(mod) == [str(tmp_path / "data")]


def test_regular_package_with___file__(tmp_path):
    (tmp_path / "data").mkdir()
    mod = types.ModuleType("satkit_data")
    mod.__file__ = str(tmp_path / "__init__.py")
    mod.__path__ = [str(tmp_path)]
    assert satkit._optional_data_bundle_dirs(mod) == [str(tmp_path / "data")]


def test_package_without_data_dir_registers_nothing(tmp_path):
    mod = types.ModuleType("satkit_data")
    mod.__file__ = None
    mod.__path__ = [str(tmp_path)]
    assert satkit._optional_data_bundle_dirs(mod) == []
