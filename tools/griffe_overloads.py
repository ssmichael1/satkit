"""
Griffe extension for the API docs: document functions that exist in the
stubs only as ``@overload`` signatures.

A stub (``.pyi``) function with overloads has no implementation, and griffe
attaches overloads only to an implementation, so every overload-only
function (``sgp4``, ``frametransform.gmst``, ``sun.pos_gcrf``,
``propresult.interp``, the ``time`` operators, ...) was left out of the
API pages, and a direct ``::: satkit.sgp4`` failed the build.

For each such name this adds a function that lists every overload, with
``overloads_only`` in ``mkdocs.yml`` showing just the overload signatures.
Its docstring is chosen from the overloads that have one:

1. the general form, if there is one: an overload whose docstring is at
   least twice as long as the first documented overload's (``sgp4``,
   ``time.__init__``, ``propresult.interp``);
2. otherwise every documented overload under its own signature, when the
   forms really differ: an operator, which dispatches on the type of its
   operand (``time + duration`` and ``time + float``, ``quaternion * quaternion``
   and ``quaternion * vector``), or forms that take different parameters
   (``density.nrlmsise``);
3. otherwise the first documented overload, which for the scalar / array
   pairs (``frametransform.gmst``, ``sun.pos_gcrf``, ...) is the scalar form.

Enabled in ``mkdocs.yml`` as ``tools/griffe_overloads.py:OverloadOnlyFunctions``.
"""

from __future__ import annotations

import ast
from typing import Any

import griffe


def _names(f: griffe.Function) -> tuple[str, ...]:
    return tuple(p.name for p in f.parameters if p.name not in ("self", "cls"))


def _signature(name: str, f: griffe.Function) -> str:
    params = []
    for p in f.parameters:
        if p.name in ("self", "cls"):
            continue
        text = p.name
        if p.annotation is not None:
            text += f": {p.annotation}"
        if p.default is not None:
            text += f" = {p.default}"
        params.append(text)
    ret = f" -> {f.returns}" if f.returns is not None else ""
    return f"{name}({', '.join(params)}){ret}"


def _combined(name: str, documented: list[griffe.Function], parent: griffe.Object) -> griffe.Docstring:
    """One docstring holding each overload's, under that overload's signature."""
    parts = []
    for f in documented:
        parts.append(f"```python\n{_signature(name, f)}\n```\n\n{f.docstring.value.strip()}")
    first = documented[0].docstring
    return griffe.Docstring(
        "\n\n".join(parts),
        lineno=first.lineno,
        endlineno=first.endlineno,
        parent=parent,
        parser=first.parser,
        parser_options=first.parser_options,
    )


class OverloadOnlyFunctions(griffe.Extension):
    def on_members(self, *, node: ast.AST | griffe.ObjectNode, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.kind not in (griffe.Kind.MODULE, griffe.Kind.CLASS):
            return
        for name, overloads in list(obj.overloads.items()):
            if not overloads or name in obj.members:
                continue
            documented = [f for f in overloads if f.docstring and f.docstring.value.strip()]
            main = documented[0] if documented else overloads[0]
            docstring = main.docstring
            returns = main.returns
            if len(documented) > 1:
                longest = max(documented, key=lambda f: len(f.docstring.value))
                operator = name.startswith("__") and name != "__init__"
                if len(longest.docstring.value) >= 2 * len(main.docstring.value):
                    main, docstring, returns = longest, longest.docstring, longest.returns
                elif operator or len({_names(f) for f in documented}) > 1:
                    # The return type differs between the forms; each block's
                    # signature carries it.
                    docstring, returns = None, None
            func = griffe.Function(
                name,
                lineno=main.lineno,
                endlineno=main.endlineno,
                parameters=main.parameters,
                returns=returns,
                decorators=[d for d in main.decorators if "overload" not in d.callable_path],
                docstring=docstring,
                parent=obj,
            )
            if docstring is None:
                func.docstring = _combined(name, documented, func)
            func.overloads = overloads
            obj.set_member(name, func)
            del obj.overloads[name]
