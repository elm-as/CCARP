"""Automated AST & architectural compliance tests against ELMAS.md."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

BANNED_DIR_NAMES = {"utils", "helpers", "misc", "common"}
LAYER_ORDER = {
    "core": 0,
    "ingestion": 1,
    "features": 2,
    "modeling": 3,
    "evaluation": 4,
    "visualization": 5,
}
VALID_PREFIXES = ("is_", "has_", "can_")


def _max_ast_depth(node: ast.AST, depth: int = 0) -> int:
    """Compute maximum compound statement nesting inside a function body."""
    compound_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
    best = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        is_elif = (
            isinstance(node, ast.If)
            and isinstance(child, ast.If)
            and child in node.orelse
            and len(node.orelse) == 1
        )
        next_depth = depth if is_elif else (depth + 1 if isinstance(child, compound_types) else depth)
        best = max(best, _max_ast_depth(child, next_depth))
    return best


def test_no_banned_catchall_folders_or_files() -> None:
    for path in SRC_ROOT.rglob("*"):
        if path.name in BANNED_DIR_NAMES or path.stem in BANNED_DIR_NAMES:
            raise AssertionError(f"Banned ELMAS.md catch-all path: {path}")


def test_file_line_count_param_count_and_nesting_limits() -> None:
    py_files = list(SRC_ROOT.rglob("*.py")) + [PROJECT_ROOT / "main.py"]
    for py_path in py_files:
        text = py_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= 350, f"{py_path} has {len(lines)} lines (>350)"

        tree = ast.parse(text, filename=str(py_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            n_params = len(node.args.posonlyargs) + len(node.args.args) + len(node.args.kwonlyargs)
            assert n_params <= 3, f"{py_path}:{node.name} has {n_params} params (>3)"
            depth = _max_ast_depth(node)
            assert depth <= 3, f"{py_path}:{node.name} has nesting depth {depth} (>3)"


def test_unidirectional_domain_dependency_dag() -> None:
    for py_path in SRC_ROOT.rglob("*.py"):
        rel_parts = py_path.relative_to(SRC_ROOT).parts
        if len(rel_parts) < 2:
            continue
        src_layer = rel_parts[0]
        src_rank = LAYER_ORDER[src_layer]
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                target_layer = node.module.split(".")[1]
                assert LAYER_ORDER[target_layer] <= src_rank, (
                    f"Upward import violation: {py_path} imports {node.module}"
                )


def test_exported_functions_have_docstrings_and_verb_naming() -> None:
    for py_path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            assert ast.get_docstring(node), f"Missing docstring on {py_path}:{node.name}"
            has_verb_or_prefix = "_" in node.name or node.name.startswith(VALID_PREFIXES)
            assert has_verb_or_prefix, f"Non-descriptive function name {py_path}:{node.name}"
