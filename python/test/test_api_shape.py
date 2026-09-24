"""Enforce the property-vs-method rule from CONTRIBUTING.md on the type stub.

A property tells you something about the object; a method gives you something
to use instead of it. Mechanically, for a zero-argument member of a class:

* it returns another instance of the same class  -> method (``inverse()``)
* its name starts with ``as_`` / ``to_``          -> method (a conversion)
* otherwise                                       -> property

``mypy.stubtest`` (run in CI) pins the runtime bindings to the stub, so
checking the stub is enough to check the bindings.
"""

import ast
import pathlib

import pytest

STUB = pathlib.Path(__file__).resolve().parents[1] / "satkit" / "satkit.pyi"
CONVERSION_PREFIXES = ("as_", "to_")


def _decorator_names(fn: ast.FunctionDef) -> set[str]:
    return {ast.unparse(d) for d in fn.decorator_list}


def _zero_arg_members():
    tree = ast.parse(STUB.read_text(), filename=str(STUB))
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decos = _decorator_names(node)
            if decos & {"staticmethod", "classmethod", "overload"}:
                continue
            if any(d.endswith(".setter") for d in decos):
                continue
            if node.name.startswith("__"):
                continue
            a = node.args
            if len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) != 1:
                continue
            if a.vararg or a.kwarg:
                continue
            ret = ast.unparse(node.returns) if node.returns else ""
            yield cls.name, node.name, "property" in decos, ret


MEMBERS = list(_zero_arg_members())


def test_stub_has_members():
    assert len(MEMBERS) > 100, "stub parse found suspiciously few members"


@pytest.mark.parametrize(
    "cls,name,is_property,ret", MEMBERS, ids=[f"{c}.{n}" for c, n, _, _ in MEMBERS]
)
def test_property_vs_method(cls, name, is_property, ret):
    returns_self = ret == cls
    is_conversion = name.startswith(CONVERSION_PREFIXES)
    should_be_method = returns_self or is_conversion
    if should_be_method:
        why = "returns its own type" if returns_self else "is a conversion"
        assert not is_property, f"{cls}.{name} {why}, so it must be a method"
    else:
        assert is_property, (
            f"{cls}.{name} describes the object, so it must be a @property; "
            "if it is a conversion, name it as_*/to_*"
        )
    if is_conversion:
        assert ret != cls, f"{cls}.{name} converts to its own type; rename it"
