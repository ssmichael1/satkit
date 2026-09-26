"""Compare the stub constructors with the runtime constructor signatures.

stubtest cannot check constructors of PyO3 classes: the runtime constructor is
``__new__`` (its signature is the class ``__text_signature__``) while the
stubs declare ``__init__``, so every constructor is on the stubtest allowlist
(python/stubtest_allowlist.txt). This test does that comparison instead,
for every class whose binding declares a signature: each keyword the stub
advertises must be accepted at runtime, positional/keyword kinds must agree,
defaults must be present on the same parameters, and every required runtime
parameter must be in the stub. Types are not compared.

Classes whose ``#[new]`` takes bare ``*args`` / ``**kwargs`` have no runtime
signature to compare with and are skipped.
"""

import ast
import inspect
import pathlib

import pytest

import satkit

STUB = pathlib.Path(__file__).resolve().parents[1] / "satkit" / "satkit.pyi"
P = inspect.Parameter


def _stub_params(fn: ast.FunctionDef) -> list[tuple[str, inspect._ParameterKind, bool]]:
    a = fn.args
    positional = [(x.arg, P.POSITIONAL_ONLY) for x in a.posonlyargs]
    positional += [(x.arg, P.POSITIONAL_OR_KEYWORD) for x in a.args]
    n_defaults = len(a.defaults)
    out = [
        (name, kind, i >= len(positional) - n_defaults)
        for i, (name, kind) in enumerate(positional)
    ]
    out += [
        (x.arg, P.KEYWORD_ONLY, d is not None) for x, d in zip(a.kwonlyargs, a.kw_defaults)
    ]
    return out[1:]  # drop self


def _stub_inits() -> dict[str, list[list[tuple[str, inspect._ParameterKind, bool]]]]:
    tree = ast.parse(STUB.read_text(), filename=str(STUB))
    out = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            inits = [
                _stub_params(f)
                for f in cls.body
                if isinstance(f, ast.FunctionDef) and f.name == "__init__"
            ]
            if inits:
                out[cls.name] = inits
    return out


def _runtime_signature(name: str) -> inspect.Signature | None:
    cls = getattr(satkit, name, None)
    if cls is None or not getattr(cls, "__text_signature__", None):
        return None
    sig = inspect.signature(cls)
    named = [p for p in sig.parameters.values() if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)]
    return sig if named or not sig.parameters else None


STUB_INITS = _stub_inits()
CASES = [name for name in sorted(STUB_INITS) if _runtime_signature(name) is not None]


def test_cases_found():
    assert len(CASES) >= 4, "expected kepler, satstate, ecomparams and time at least"


def _overload_problems(stub, runtime: inspect.Signature) -> list[str]:
    params = list(runtime.parameters.values())
    rt = {p.name: p for p in params if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)}
    rt_positional = [p for p in params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    var_pos = any(p.kind is P.VAR_POSITIONAL for p in params)
    var_kw = any(p.kind is P.VAR_KEYWORD for p in params)
    problems = []
    stub_names = set()
    for i, (name, kind, has_default) in enumerate(stub):
        stub_names.add(name)
        r = rt.get(name)
        if kind is P.POSITIONAL_ONLY:
            if not (var_pos or i < len(rt_positional)):
                problems.append(f"positional parameter {name!r} has no runtime slot")
            continue
        if r is None:
            if not var_kw:
                problems.append(f"{name!r} is not accepted as a keyword at runtime")
            continue
        if kind is P.POSITIONAL_OR_KEYWORD and r.kind is P.KEYWORD_ONLY:
            problems.append(f"{name!r} is keyword-only at runtime")
        if kind is P.KEYWORD_ONLY and r.kind is P.POSITIONAL_ONLY:
            problems.append(f"{name!r} is positional-only at runtime")
        if has_default != (r.default is not P.empty):
            problems.append(f"{name!r} default: stub {has_default}, runtime {r.default is not P.empty}")
    for name, r in rt.items():
        if r.default is P.empty and name not in stub_names:
            problems.append(f"required runtime parameter {name!r} is missing")
    return problems


@pytest.mark.parametrize("name", CASES)
def test_stub_init_matches_runtime(name):
    runtime = _runtime_signature(name)
    overloads = STUB_INITS[name]
    problems = []
    for n, stub in enumerate(overloads):
        problems += [f"overload {n}: {p}" for p in _overload_problems(stub, runtime)]
    advertised = {p for stub in overloads for p, _, _ in stub}
    problems += [
        f"runtime parameter {p!r} is in no stub overload"
        for p, rp in runtime.parameters.items()
        if rp.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD) and p not in advertised
    ]
    assert not problems, f"satkit.{name}{runtime} vs stub __init__:\n  " + "\n  ".join(problems)
