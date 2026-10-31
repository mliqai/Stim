"""AST import-layer DAG guard across non-test modules in stimside."""

from __future__ import annotations

import ast
from pathlib import Path

STIMSIDE_DIR = Path(__file__).parent

ALLOWED_LAYER_DEPS: dict[str, set[str]] = {
    "util": {"util"},
    "_native": {"_native"},
    "op_handlers": {"op_handlers", "util"},
    "dem": {"dem", "op_handlers", "util", "_native"},
    "engines": {"engines", "op_handlers", "util", "_native"},
    "samplers": {"samplers", "engines", "dem", "op_handlers", "util"},
    "root": {"root", "samplers", "engines", "dem", "op_handlers", "util", "_native"},
}

# Pre-existing runtime layering violations; shrinks monotonically to empty by end of Phase 4.
ALLOWLIST: set[tuple[str, str]] = {
    ("stimside.op_handlers.leakage_handlers.leakage_uint8_coset", "stimside.simulator_coset"),
    ("stimside.op_handlers.leakage_handlers.leakage_uint8_flip", "stimside.simulator_flip"),
    ("stimside.op_handlers.leakage_handlers.leakage_uint8_magic", "stimside.simulator_magic"),
    ("stimside.op_handlers.leakage_handlers.leakage_uint8_tableau", "stimside.simulator_tableau"),
    ("stimside.util.magic_compile", "stimside.op_handlers.leakage_handlers.common_parsing"),
    ("stimside.util.magic_compile", "stimside.op_handlers.leakage_handlers.leakage_parameters"),
    ("stimside.util.magic_compile", "stimside.op_handlers.leakage_handlers.tag_registry"),
    ("stimside.util.tableside_kernels", "stimside.op_handlers.leakage_handlers.leakage_parameters"),
    ("stimside.util.tableside_kernels", "stimside.op_handlers.leakage_handlers.tag_registry"),
}


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
    if head in ("_flip", "_tableau", "_coset", "_magic") or head.startswith("simulator_"):
        return "engines"
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
    for src, pkg_parts, path in prod_files:
        src_layer = _layer_of(src)
        allowed = ALLOWED_LAYER_DEPS[src_layer]
        raw = _runtime_targets(ast.parse(path.read_text(encoding="utf-8")), pkg_parts)
        for cand in raw:
            dst = cand if cand in known_mods else cand.rsplit(".", 1)[0]
            if dst not in known_mods or dst == src:
                continue
            if _layer_of(dst) not in allowed:
                violations.add((src, dst))

    unexpected = sorted(violations - ALLOWLIST)
    stale = sorted(ALLOWLIST - violations)
    assert not unexpected, f"Disallowed layering imports: {unexpected}"
    assert not stale, f"Stale ALLOWLIST entries (remove them!): {stale}"
