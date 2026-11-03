"""AST import-layer DAG guard across non-test modules in stimside."""

from __future__ import annotations

import ast
from pathlib import Path

STIMSIDE_DIR = Path(__file__).parent

_ENGINE_NAMES = ("flip", "tableau", "coset", "magic")
_ALL_ENGINES = {f"engine:{n}" for n in _ENGINE_NAMES}

ALLOWED_LAYER_DEPS: dict[str, set[str]] = {
    "util": {"util"},
    "_native": {"_native"},
    "op_handlers": {"op_handlers", "util"},
    "dem": {"dem", "op_handlers", "util", "_native"},
    **{f"engine:{n}": {f"engine:{n}", "op_handlers", "util", "_native"} for n in _ENGINE_NAMES},
    "samplers": {"samplers", *_ALL_ENGINES, "dem", "op_handlers", "util"},
    "root": {"root", "samplers", *_ALL_ENGINES, "dem", "op_handlers", "util", "_native"},
}

ALLOWLIST: set[tuple[str, str]] = set()


def _layer_of(mod: str) -> str:
    leaf = mod.rsplit(".", 1)[-1]
    if leaf.endswith("_test") or leaf.startswith("test_"):
        return "tests"
    if mod == "stimside":
        return "root"
    sub = mod.removeprefix("stimside.")
    head = sub.split(".", 1)[0]
    if head in ("util", "_native", "op_handlers"):
        return head
    if head == "dem_generators":
        return "dem"
    for name in _ENGINE_NAMES:
        if head in (f"_{name}", f"simulator_{name}"):
            return f"engine:{name}"
    if head == "_sampling" or head.startswith("sampler_"):
        return "samplers"
    raise AssertionError(f"Unclassified stimside module: {mod!r}")


def _runtime_targets(tree: ast.AST, pkg_parts: list[str]) -> set[str]:
    targets: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            t = node.test
            is_tc = (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
            )
            if not is_tc:
                for stmt in node.body:
                    self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name.startswith("stimside"):
                    targets.add(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            mod = node.module or ""
            if node.level > 0:
                base = list(pkg_parts[: len(pkg_parts) - node.level + 1])
                if mod:
                    base.extend(mod.split("."))
                mod = ".".join(base)
            if mod.startswith("stimside"):
                for alias in node.names:
                    targets.add(f"{mod}.{alias.name}")

        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in ("import_module", "__import__") and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str) and arg0.value.startswith("stimside"):
                    targets.add(arg0.value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return targets


def test_import_layering() -> None:
    known_mods: set[str] = set()
    prod_files: list[tuple[str, list[str], Path]] = []
    for path in sorted(STIMSIDE_DIR.rglob("*.py")):
        parts = list(path.relative_to(STIMSIDE_DIR.parent).with_suffix("").parts)
        is_init = parts[-1] == "__init__"
        mod = ".".join(parts[:-1] if is_init else parts)
        known_mods.add(mod)
        if path.name.endswith("_test.py") or path.name.startswith("test_"):
            continue
        pkg_parts = parts[:-1]
        prod_files.append((mod, pkg_parts, path))

    violations: set[tuple[str, str]] = set()
    adj: dict[str, set[str]] = {src: set() for src, _, _ in prod_files}
    for src, pkg_parts, path in prod_files:
        src_layer = _layer_of(src)
        allowed = ALLOWED_LAYER_DEPS[src_layer]
        raw = _runtime_targets(ast.parse(path.read_text(encoding="utf-8")), pkg_parts)
        for cand in raw:
            dst = cand if cand in known_mods else cand.rsplit(".", 1)[0]
            if dst not in known_mods or dst == src:
                continue
            adj[src].add(dst)
            if _layer_of(dst) not in allowed:
                violations.add((src, dst))

    barrel_callers = sorted(src for src, deps in adj.items() if "stimside._magic.compile" in deps)
    assert not barrel_callers, f"Compat barrel stimside._magic.compile has production callers: {barrel_callers}"
    unexpected = sorted(violations - ALLOWLIST)
    stale = sorted(ALLOWLIST - violations)
    assert not unexpected, f"Disallowed layering imports: {unexpected}"
    assert not stale, f"Stale ALLOWLIST entries (remove them!): {stale}"

    state: dict[str, int] = {}
    stack: list[str] = []

    def _dfs(u: str) -> None:
        state[u] = 1
        stack.append(u)
        for v in sorted(adj.get(u, ())):
            st = state.get(v, 0)
            if st == 1:
                cyc = stack[stack.index(v) :] + [v]
                raise AssertionError(f"Import cycle detected: {' -> '.join(cyc)}")
            if st == 0:
                _dfs(v)
        stack.pop()
        state[u] = 2

    for mod_name in sorted(adj):
        if state.get(mod_name, 0) == 0:
            _dfs(mod_name)


def test_util_purity() -> None:
    util_dir = STIMSIDE_DIR / "util"
    for path in sorted(util_dir.glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name != "ctypes" for a in node.names), f"{path.name} imports ctypes"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "ctypes", f"{path.name} imports from ctypes"
            elif isinstance(node, ast.Name):
                assert not node.id.startswith("OMEGA"), f"{path.name} defines Magic constant {node.id}"
