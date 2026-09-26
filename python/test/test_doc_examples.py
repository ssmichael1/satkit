"""Run the Python examples in the documentation and in the docstrings.

Sources (the notebooks under docs/tutorials/ are not covered: mkdocs-jupyter
already executes them during the docs build):

* every fenced ```python block in docs/**/*.md and README.md, including the
  ones inside MkDocs Material tabs (``=== "Python"``); other languages, Rust
  tabs included, are ignored. The blocks of a page run in order in one shared
  namespace, since later blocks build on earlier ones.
* the ``Example:`` code of every docstring, both the ones the API reference is
  rendered from (python/satkit/*.pyi) and the runtime ``__doc__`` compiled in
  from the ``///`` comments in python/src/*.rs. Fenced ```python blocks and
  ``>>>`` lines both count; expected-output lines after ``>>>`` are not
  compared. Each docstring runs in its own namespace, pre-seeded with
  ``satkit``, ``sk`` (the same module) and ``np``. A runtime docstring whose
  code is identical to a stub docstring's is run once.

Blocks are marked, never skipped by heuristics (see CONTRIBUTING.md):

* in Markdown, an HTML comment on the line(s) before the fence:
  ``<!-- skip-test: reason -->`` (not run; e.g. needs the network) or
  ``<!-- xfail-test: reason -->`` (runs and must fail; a known code bug).
  ``<!-- test-setup`` ... ``-->`` holds hidden code that runs at that point in
  the page, for a fragment that uses names the prose defines elsewhere.
* in docstrings, entries in ``DOCSTRING_MARKS`` / ``DOCSTRING_SETUP`` below,
  keyed by qualified name (``#n`` selects one of several examples).

Each page (and each stub file / runtime module of docstrings) runs in a
fresh subprocess, so examples that change global state (warnings disabled,
data directory changed) cannot leak into the other tests, with
``SATKIT_OFFLINE=1`` so a data file the directory lacks is an error rather
than a download. That does not stop ``TLE.from_url`` / ``omm_from_url``,
which is why every example that fetches anything must be marked.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUB_DIR = ROOT / "python" / "satkit"
NATIVE_SUBMODULES = (
    "density",
    "frametransform",
    "jplephem",
    "moon",
    "planets",
    "spaceweather",
    "sun",
    "utils",
)
PY_LANGS = {"python", "py", "python3", "pycon"}
FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*\{?\.?(?P<lang>[\w+-]*)")
MARK_RE = re.compile(r"<!--\s*(?P<kind>skip-test|xfail-test)\s*(?::\s*(?P<reason>.*?))?\s*-->")
SETUP_OPEN = "<!-- test-setup"
PRESEED = "import numpy as np\nimport satkit\nimport satkit as sk\n"
TIMEOUT_S = 900

NETWORK = "needs the network"

# Docstring examples that must not run or are known to fail, keyed by the
# qualified name of the documented object; applies to both the stub and the
# runtime docstring. Kinds: "skip" (not run), "xfail" (a code bug; must fail),
# "elsewhere" (a doc bug fixed on another open branch; may pass or fail).
DOCSTRING_MARKS: dict[str, tuple[str, str]] = {
    "satkit.TLE.from_url": ("skip", NETWORK + " (CelesTrak)"),
    "satkit.TLE.from_omm": ("skip", NETWORK + " (CelesTrak via omm_from_url)"),
    "satkit.omm_from_url": ("skip", NETWORK + " (CelesTrak)"),
    "satkit.omm_from_text": ("skip", NETWORK + " (CelesTrak via requests)"),
    "satkit.omm_from_file": ("skip", "needs a gp.xml saved from Space-Track or CelesTrak"),
    "satkit.utils.update_datafiles": ("skip", NETWORK + " (downloads the data files)"),
    "satkit.sgp4#2": ("skip", NETWORK + " (CelesTrak via omm_from_url)"),
}

_ISS = """tle = satkit.TLE.from_lines([
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357",
])
"""
_LEO = """t0 = satkit.time(2024, 1, 1)
t1 = t0 + satkit.duration.from_hours(2)
r = satkit.consts.earth_radius + 500e3
pos_gcrf = np.array([r, 0.0, 0.0])
vel_gcrf = np.array([0.0, np.sqrt(satkit.consts.mu_earth / r), 0.0])
sat = satkit.satstate(t0, pos_gcrf, vel_gcrf)
t_burn = t0 + satkit.duration.from_minutes(30)
"""

# Hidden setup for docstring examples that are fragments: the names a method's
# example takes for granted (an instance, a time). Runs in the example's
# namespace before it, like a Markdown `<!-- test-setup -->` block.
DOCSTRING_SETUP: dict[str, str] = {
    "satkit.TLE.from_file": _ISS
    + 'import pathlib\npathlib.Path("gps-ops.txt").write_text("\\n".join(2 * tle.to_3line()))\n',
    "satkit.TLE.to_2line": _ISS,
    "satkit.TLE.to_3line": _ISS,
    "satkit.TLE.fit_from_states": _ISS + "pos0, vel0 = satkit.sgp4(tle, satkit.time(2024, 1, 1))\n",
    "satkit.frame": _LEO,
    "satkit.frametransform.to_gcrf": _LEO + "v_ntw = np.array([0.0, 1.0, 0.0])\n",
    "satkit.frametransform.from_gcrf": _LEO + "v_gcrf = vel_gcrf\n",
    "satkit.kepler.to_pv": "k = satkit.kepler(7000e3, 0.001, 0.9, 0.0, 0.0, 0.0)\n",
    "satkit.kepler.propagate": "k = satkit.kepler(7000e3, 0.001, 0.9, 0.0, 0.0, 0.0)\n",
    "satkit.satstate.set_pos_uncertainty": _LEO,
    "satkit.satstate.add_maneuver": _LEO,
    "satkit.satstate.add_prograde": _LEO,
    "satkit.satstate.propagate": _LEO
    + "r, v = pos_gcrf, vel_gcrf\nt_end = t0 + satkit.duration.from_hours(3)\n",
    "satkit.ecomparams": _LEO + "state = np.concatenate([pos_gcrf, vel_gcrf])\nsettings = satkit.propsettings()\n",
}


@dataclasses.dataclass
class Block:
    group: str  # one subprocess per group
    ns: str  # blocks with the same ns share a namespace
    label: str  # test id
    filename: str  # for tracebacks
    lineno: int  # 1-based line of the first code line in `filename`
    code: str
    kind: str = "run"  # run | skip | xfail | elsewhere | setup
    reason: str = ""


def _strip_prompts(lines: list[str]) -> list[str]:
    """Doctest-style lines to plain code; output lines become blank lines so
    line numbers still match the source."""
    out = []
    for line in lines:
        s = line.lstrip()
        out.append(s[4:] if s.startswith((">>> ", "... ")) else "")
    return out


def _code(lines: list[str]) -> str:
    if any(line.lstrip().startswith(">>>") for line in lines):
        lines = _strip_prompts(lines)
    return textwrap.dedent("\n".join(lines))


def _scan_fence(lines: list[str], i: int) -> tuple[str, int, list[str]] | None:
    """If lines[i] opens a fence, return (lang, index after the close, body)."""
    m = FENCE_RE.match(lines[i])
    if not m:
        return None
    fence = m["fence"]
    j = i + 1
    while j < len(lines):
        s = lines[j].strip()
        if s.startswith(fence) and not s.strip(fence[0]):
            break
        j += 1
    return m["lang"].lower(), j + 1, lines[i + 1 : j]


# --- Markdown -----------------------------------------------------------------


def _markdown_blocks(path: pathlib.Path) -> tuple[list[Block], list[str]]:
    rel = path.relative_to(ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    problems: list[str] = []
    pending: tuple[str, str, int] | None = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(SETUP_OPEN):
            j = i + 1
            while j < len(lines) and lines[j].strip() != "-->":
                j += 1
            blocks.append(
                Block(rel, rel, f"{rel}:{i + 1}-setup", str(path), i + 2, _code(lines[i + 1 : j]), "setup")
            )
            i = j + 1
            continue
        m = MARK_RE.fullmatch(s)
        if m:
            kind = "skip" if m["kind"] == "skip-test" else "xfail"
            if not m["reason"]:
                problems.append(f"{rel}:{i + 1}: marker without a reason")
            pending = (kind, m["reason"] or "", i + 1)
            i += 1
            continue
        fence = _scan_fence(lines, i)
        if fence:
            lang, nxt, body = fence
            if lang in PY_LANGS:
                kind, reason = (pending[0], pending[1]) if pending else ("run", "")
                blocks.append(Block(rel, rel, f"{rel}:{i + 1}", str(path), i + 2, _code(body), kind, reason))
            elif pending:
                problems.append(f"{rel}:{pending[2]}: marker is followed by a {lang or 'plain'} block")
            pending = None
            i = nxt
            continue
        if s and pending:
            problems.append(f"{rel}:{pending[2]}: marker is not directly before a code block")
            pending = None
        i += 1
    if pending:
        problems.append(f"{rel}:{pending[2]}: marker at end of file")
    return blocks, problems


# --- Docstrings -----------------------------------------------------------------


def _docstring_code(doc: str) -> list[tuple[int, str]]:
    """(0-based line offset of the first code line, code) per example in doc."""
    lines = doc.splitlines()
    found = []
    prompt_start: int | None = None
    prompt_end = 0

    def flush():
        nonlocal prompt_start
        if prompt_start is not None:
            found.append((prompt_start, _code(lines[prompt_start:prompt_end])))
            prompt_start = None

    i = 0
    while i < len(lines):
        fence = _scan_fence(lines, i)
        if fence:
            flush()
            lang, nxt, body = fence
            if lang in PY_LANGS:
                found.append((i + 1, _code(body)))
            i = nxt
            continue
        if lines[i].lstrip().startswith(">>>"):
            if prompt_start is None:
                prompt_start = i
            prompt_end = i + 1
        i += 1
    flush()
    return found


def _docstring_blocks(
    group: str, qualname: str, doc: str, filename: str, first_line: int, prefix: str
) -> list[Block]:
    examples = _docstring_code(doc)
    ns = f"{prefix}:{qualname}"
    out = []
    if examples and qualname in DOCSTRING_SETUP:
        out.append(Block(group, ns, f"{ns}-setup", f"<setup {qualname}>", 1, DOCSTRING_SETUP[qualname], "setup"))
    for offset, code in examples:
        out.append(Block(group, ns, ns, filename, first_line + offset, code))
    return out


def _dedupe(blocks: list[Block]) -> list[Block]:
    """Drop repeated code within one namespace (overloads repeat their
    docstrings) and make the remaining labels unique."""
    seen: set[tuple[str, str]] = set()
    out = []
    for b in blocks:
        key = (b.ns, b.code if b.kind != "setup" else "<setup>")
        if key not in seen:
            seen.add(key)
            out.append(b)
    counts: dict[str, int] = {}
    for b in out:
        counts[b.label] = counts.get(b.label, 0) + 1
    index: dict[str, int] = {}
    for b in out:
        if counts[b.label] > 1:
            index[b.label] = index.get(b.label, 0) + 1
            b.label = f"{b.label}#{index[b.label]}"
    return out


def _apply_docstring_marks(blocks: list[Block]) -> set[str]:
    """Set kind/reason from DOCSTRING_MARKS ("qualname" or "qualname#n");
    return the keys that matched."""
    used = set()
    for b in blocks:
        if b.kind == "setup" or not b.ns.startswith(("pyi:", "rt:")):
            continue
        name = b.label.split(":", 1)[1]
        for key in (name, name.split("#")[0]):
            if key in DOCSTRING_MARKS:
                b.kind, b.reason = DOCSTRING_MARKS[key]
                used.add(key)
                break
    return used


def _stub_module_name(path: pathlib.Path) -> str:
    return "satkit" if path.stem in ("satkit", "__init__") else f"satkit.{path.stem}"


def _stub_blocks() -> list[Block]:
    blocks = []
    for path in sorted(STUB_DIR.glob("*.pyi")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def visit(body, prefix):
            for node in body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    qual = f"{prefix}.{node.name}"
                    first = node.body[0] if node.body else None
                    if (
                        isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)
                    ):
                        blocks.extend(
                            _docstring_blocks(rel, qual, first.value.value, str(path), first.lineno, "pyi")
                        )
                    if isinstance(node, ast.ClassDef):
                        visit(node.body, qual)

        visit(tree.body, _stub_module_name(path))
    return blocks


def _runtime_blocks(stub_code: set[str]) -> list[Block]:
    import satkit
    import satkit.satkit as ext

    modules = {"satkit": ext}
    modules.update({f"satkit.{n}": getattr(ext, n) for n in NATIVE_SUBMODULES})
    blocks = []
    for modname, module in modules.items():
        group = f"runtime:{modname}"
        objects = []
        for name, obj in vars(module).items():
            if name.startswith("_") or isinstance(obj, type(satkit)):
                continue
            objects.append((f"{modname}.{name}", obj))
            if isinstance(obj, type):
                for mname, member in vars(obj).items():
                    if not mname.startswith("_"):
                        objects.append((f"{modname}.{name}.{mname}", member))
        for qual, obj in objects:
            doc = getattr(obj, "__doc__", None)
            if not isinstance(doc, str) or (isinstance(obj, type) is False and doc == type(obj).__doc__):
                continue
            for b in _docstring_blocks(group, qual, doc, f"<docstring {qual}>", 1, "rt"):
                if _normalise(b.code) not in stub_code:
                    blocks.append(b)
    return blocks


def _normalise(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())


# --- Collection -------------------------------------------------------------------


def _collect() -> tuple[list[Block], list[str]]:
    blocks, problems = [], []
    pages = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
    for page in pages:
        b, p = _markdown_blocks(page)
        blocks += b
        problems += p
    stub = _stub_blocks()
    blocks += stub
    blocks += _runtime_blocks({_normalise(b.code) for b in stub if b.kind != "setup"})
    blocks = _dedupe(blocks)
    used = _apply_docstring_marks(blocks)
    known = {b.ns.split(":", 1)[1] for b in blocks if b.ns.startswith(("pyi:", "rt:"))}
    problems += [f"DOCSTRING_MARKS entry {q!r} matches no docstring example" for q in DOCSTRING_MARKS if q not in used]
    problems += [f"DOCSTRING_SETUP entry {q!r} matches no docstring example" for q in DOCSTRING_SETUP if q not in known]
    return blocks, problems


BLOCKS, PROBLEMS = _collect()
BY_GROUP: dict[str, list[Block]] = {}
for _b in BLOCKS:
    BY_GROUP.setdefault(_b.group, []).append(_b)


@functools.cache
def _run_group(group: str) -> dict[str, str | None]:
    """Run a group's blocks in one fresh interpreter; label -> traceback or None."""
    payload = [dataclasses.asdict(b) for b in BY_GROUP[group] if b.kind != "skip"]
    env = dict(os.environ, SATKIT_OFFLINE="1", MPLBACKEND="Agg")
    # The runner works in a scratch directory (examples may write files), so a
    # relative data directory, as CI uses, must be resolved here.
    if env.get("SATKIT_DATA"):
        env["SATKIT_DATA"] = os.path.abspath(env["SATKIT_DATA"])
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = pathlib.Path(tmp, "blocks.json"), pathlib.Path(tmp, "results.json")
        src.write_text(json.dumps(payload))
        proc = subprocess.run(
            [sys.executable, __file__, str(src), str(dst)],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if not dst.exists():
            raise RuntimeError(f"example runner crashed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
        return json.loads(dst.read_text())


def test_markers_and_collection():
    assert len(BLOCKS) > 100, "suspiciously few examples collected"
    assert not PROBLEMS, "\n".join(PROBLEMS)


def _param(b: Block):
    marks = []
    if b.kind == "skip":
        marks.append(pytest.mark.skip(reason=b.reason))
    elif b.kind == "xfail":
        marks.append(pytest.mark.xfail(reason=b.reason, strict=True))
    elif b.kind == "elsewhere":
        marks.append(pytest.mark.xfail(reason=b.reason, strict=False))
    return pytest.param(b, marks=marks, id=b.label)


@pytest.mark.parametrize("block", [_param(b) for b in BLOCKS])
def test_example(block: Block):
    error = _run_group(block.group)[block.label]
    if error is not None:
        pytest.fail(f"{block.label}\n{error}", pytrace=False)


# --- Runner (executed in the subprocess) -----------------------------------------


def _run(src: str, dst: str) -> None:
    import contextlib
    import io
    import traceback

    blocks = json.loads(pathlib.Path(src).read_text())
    namespaces: dict[str, dict] = {}
    results: dict[str, str | None] = {}
    for b in blocks:
        ns = namespaces.get(b["ns"])
        if ns is None:
            ns = namespaces[b["ns"]] = {"__name__": "__main__"}
            if b["ns"].startswith(("pyi:", "rt:")):
                exec(PRESEED, ns)
        code = "\n" * (b["lineno"] - 1) + b["code"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, b["filename"], "exec"), ns)
            results[b["label"]] = None
        except BaseException as e:  # noqa: BLE001 - SystemExit etc. from an example are failures too
            # Drop this runner's own frame from the traceback.
            tb = e.__traceback__.tb_next if e.__traceback__ else None
            results[b["label"]] = "".join(traceback.format_exception(type(e), e, tb))
    pathlib.Path(dst).write_text(json.dumps(results))


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2])
