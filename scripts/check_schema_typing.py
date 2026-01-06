#!/usr/bin/env -S uv run
"""Check for mutable and loose types in model annotations.

This script enforces:
1. IMMUTABLE TYPES: Use Sequence/Mapping/Set instead of list/dict/set
2. STRICT TYPING: No Any, Mapping[str, Any], Sequence[Any] etc.

The goal is "maximum strictness" - every field has an exact type.

Design Philosophy:
    - Error-only, no auto-fix: Forces conscious decision at each occurrence
    - Pure analysis: Checks exactly the files it's given (no internal filtering)
    - Escape hatches:
      - # noqa: mutable-type - suppress mutable type violations (list, dict, set)
      - # noqa: loose-typing - suppress loose typing violations (Any)

Usage:
    ./scripts/check_schema_typing.py <files...>
    ./scripts/check_schema_typing.py .          # All .py files in cwd

File selection is handled by the caller (pre-commit, shell globs, etc.).
This script is a pure analysis function - it checks what it's given.

Exit codes:
    0 - No violations found
    1 - Violations found
    2 - Usage error (no files specified)
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Configuration - What to check (not WHERE to check - that's the caller's job)
# =============================================================================

# Types that trigger mutable-type errors
MUTABLE_TYPES: frozenset[str] = frozenset(
    {
        # Lowercase built-ins (Python 3.9+) - concrete mutable types
        'list',
        'dict',
        'set',
        # Uppercase from typing module (legacy, deprecated in 3.9+)
        'List',
        'Dict',
        # Note: typing.Set is NOT here because it's an alias for collections.abc.Set
        # (the abstract interface), not the concrete set type
    }
)

# Types that trigger loose-typing errors
LOOSE_TYPES: frozenset[str] = frozenset(
    {
        'Any',  # Escape hatch that hides structure
    }
)

# Combined forbidden types
FORBIDDEN_TYPES: frozenset[str] = MUTABLE_TYPES | LOOSE_TYPES

# Types that are always allowed (don't recurse into these looking for violations)
ALLOWED_CONTAINERS: frozenset[str] = frozenset(
    {
        'Mapping',
        'Sequence',
        'Set',  # collections.abc.Set (abstract interface for set-like types)
        'tuple',
        'frozenset',
        'FrozenSet',  # typing module variant
    }
)

# Types that indicate "check nested contents" even though container is allowed
# e.g., Mapping[str, list[int]] - Mapping is allowed, but check the value type
TRANSPARENT_CONTAINERS: frozenset[str] = frozenset(
    {
        'Mapping',
        'Sequence',  # Check what's inside these
    }
)

# Base classes whose subclasses should be checked
MODEL_BASE_CLASSES: frozenset[str] = frozenset(
    {
        'StrictModel',
        'BaseModel',
        'PermissiveModel',
    }
)

# Classes that are exempt from checking (interface definitions)
EXEMPT_BASE_CLASSES: frozenset[str] = frozenset(
    {
        'TypedDict',
        'Protocol',
    }
)


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True)
class Violation:
    """A single type violation (mutable or loose)."""

    filepath: Path
    line: int
    column: int
    context: str  # 'field' or 'parameter'
    bad_type: str  # The forbidden type found
    violation_kind: str  # 'mutable' or 'loose'
    source_line: str
    suggestion: str


# =============================================================================
# AST Visitor
# =============================================================================


class MutableTypeChecker(ast.NodeVisitor):
    """AST visitor that checks for mutable types in model annotations."""

    SUGGESTIONS: dict[str, str] = {
        # Mutable type suggestions
        'list': 'Sequence[T] or tuple[T, ...]',
        'List': 'Sequence[T] or tuple[T, ...]',
        'dict': 'Mapping[K, V]',
        'Dict': 'Mapping[K, V]',
        'set': 'Set[T] (from collections.abc)',
        # Loose type suggestions
        'Any': 'a specific type (TypedDict, StrictModel, or concrete type)',
    }

    def __init__(self, filepath: Path, source_lines: Sequence[str]) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
        self.violations: list[Violation] = []  # Internal - OK to be mutable
        self._in_model_class = False
        self._function_depth = 0  # Track nesting depth in functions

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track when we enter a class that inherits from model bases."""
        # Check if this class should be exempt
        if self._is_exempt_class(node):
            # Don't check this class or its contents
            return

        # Check if this class inherits from a model base class
        is_model = self._is_model_class(node)

        # Save previous state (for nested classes)
        prev_in_model = self._in_model_class

        # Update state
        self._in_model_class = is_model

        # Visit children
        self.generic_visit(node)

        # Restore state
        self._in_model_class = prev_in_model

    def _is_model_class(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from a model base class."""
        for base in node.bases:
            base_name = self._get_type_name(base)
            if base_name in MODEL_BASE_CLASSES:
                return True
        return False

    def _is_exempt_class(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from an exempt base (TypedDict, Protocol)."""
        for base in node.bases:
            base_name = self._get_type_name(base)
            if base_name in EXEMPT_BASE_CLASSES:
                return True
        return False

    def _get_type_name(self, node: ast.expr) -> str:
        """Extract the name from a type expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_type_name(node.value)
        return ''

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check class field annotations (x: Type = value)."""
        # Only check if we're in a model class and NOT in a function body
        if self._in_model_class and self._function_depth == 0:
            self._check_annotation(node.annotation, node.lineno, 'field')
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check method parameter annotations."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async method parameter annotations."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Shared logic for visiting function definitions."""
        # Only check methods in model classes
        if self._in_model_class:
            self._check_function_parameters(node)

        # Track that we're in a function body
        self._function_depth += 1

        # Visit function body
        self.generic_visit(node)

        self._function_depth -= 1

    def _check_function_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check parameter annotations on a function/method."""
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs

        for arg in all_args:
            # Skip 'self' and 'cls' parameters
            if arg.arg in ('self', 'cls'):
                continue
            if arg.annotation:
                self._check_annotation(arg.annotation, arg.lineno, 'parameter')

    def _check_annotation(self, node: ast.expr, lineno: int, context: str) -> None:
        """Check if an annotation contains forbidden types."""
        bad_type = self._find_forbidden_type(node)
        if bad_type:
            # Determine violation kind
            kind = 'loose' if bad_type in LOOSE_TYPES else 'mutable'

            # Check for noqa comment with appropriate code
            if self._has_noqa(lineno, kind):
                return

            # Get the source line for context
            source_line = ''
            if 0 < lineno <= len(self.source_lines):
                source_line = self.source_lines[lineno - 1].strip()

            self.violations.append(
                Violation(
                    filepath=self.filepath,
                    line=lineno,
                    column=getattr(node, 'col_offset', 0),
                    context=context,
                    bad_type=bad_type,
                    violation_kind=kind,
                    source_line=source_line,
                    suggestion=self.SUGGESTIONS.get(bad_type, 'a more specific type'),
                )
            )

    def _find_forbidden_type(self, node: ast.expr) -> str | None:
        """Recursively find if annotation contains a forbidden type.

        Returns the forbidden type name if found, None otherwise.
        """
        if isinstance(node, ast.Name):
            # Simple name like 'list' or 'Sequence'
            if node.id in FORBIDDEN_TYPES:
                return node.id
            return None

        elif isinstance(node, ast.Subscript):
            # Generic like 'list[str]' or 'Mapping[str, Any]'
            container_name = self._get_type_name(node.value)

            # If the container itself is forbidden, report immediately
            if container_name in FORBIDDEN_TYPES:
                return container_name

            # If it's a transparent container, check nested contents
            if container_name in TRANSPARENT_CONTAINERS:
                return self._find_forbidden_in_slice(node.slice)

            # If it's an allowed container (tuple, frozenset), stop checking
            if container_name in ALLOWED_CONTAINERS:
                return None

            # For Union, Optional, etc. - check all components
            if container_name in ('Union', 'Optional'):
                return self._find_forbidden_in_slice(node.slice)

            # Unknown container - recurse into slice to be safe
            return self._find_forbidden_in_slice(node.slice)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type: X | Y
            left_violation = self._find_forbidden_type(node.left)
            if left_violation:
                return left_violation
            return self._find_forbidden_type(node.right)

        elif isinstance(node, ast.Attribute):
            # Qualified name like 'typing.List'
            if node.attr in FORBIDDEN_TYPES:
                return node.attr
            return None

        elif isinstance(node, ast.Constant):
            # String annotation (forward reference or PEP 563)
            if isinstance(node.value, str):
                return self._check_string_annotation(node.value)
            return None

        elif isinstance(node, ast.Tuple):
            # Multiple types (e.g., in Union slice)
            for element in node.elts:
                violation = self._find_forbidden_type(element)
                if violation:
                    return violation
            return None

        return None

    def _find_forbidden_in_slice(self, node: ast.expr) -> str | None:
        """Check for forbidden types in a subscript slice."""
        if isinstance(node, ast.Tuple):
            # Multiple type args like Mapping[str, int]
            for element in node.elts:
                violation = self._find_forbidden_type(element)
                if violation:
                    return violation
            return None
        else:
            # Single type arg like Sequence[str]
            return self._find_forbidden_type(node)

    def _check_string_annotation(self, annotation_str: str) -> str | None:
        """Parse and check a string annotation for forbidden types."""
        try:
            # Parse the string as a Python expression
            expr_node = ast.parse(annotation_str, mode='eval').body
            return self._find_forbidden_type(expr_node)
        except SyntaxError:
            # Invalid annotation - let other tools catch this
            return None

    def _has_noqa(self, lineno: int, violation_kind: str) -> bool:
        """Check if line has a noqa comment for this check.

        Args:
            lineno: Line number to check
            violation_kind: 'mutable' or 'loose' - determines which noqa code to check

        Returns:
            True if a matching noqa comment is found
        """
        if 0 < lineno <= len(self.source_lines):
            line = self.source_lines[lineno - 1]

            # Look for noqa comment
            if '# noqa' not in line.lower():
                return False

            # Find the noqa part
            noqa_idx = line.lower().find('# noqa')
            noqa_part = line[noqa_idx + 6 :].strip()

            # Bare # noqa ignores everything
            if not noqa_part or not noqa_part.startswith(':'):
                return True

            # Check for specific codes after the colon
            codes_part = noqa_part[1:].strip()
            codes = [c.strip().lower() for c in codes_part.split(',')]

            # Check for the appropriate noqa code based on violation kind
            if violation_kind == 'mutable' and 'mutable-type' in codes:
                return True
            if violation_kind == 'loose' and 'loose-typing' in codes:
                return True

        return False


# =============================================================================
# File Processing
# =============================================================================


def check_file(filepath: Path) -> list[Violation]:
    """Check a single file for mutable type violations."""
    try:
        source = filepath.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        print(f'Warning: Could not read {filepath}: {e}', file=sys.stderr)
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        print(f'Warning: Syntax error in {filepath}: {e}', file=sys.stderr)
        return []

    source_lines = source.splitlines()
    checker = MutableTypeChecker(filepath, source_lines)
    checker.visit(tree)

    return checker.violations


def find_python_files(root: Path, exclude_dirs: Set[str]) -> list[Path]:
    """Find all .py files recursively under root, skipping excluded directories."""
    return sorted(path for path in root.rglob('*.py') if not any(part in exclude_dirs for part in path.parts))


# =============================================================================
# Output Formatting
# =============================================================================


def format_violation(v: Violation) -> str:
    """Format a violation for display."""
    # Determine error type description and noqa code
    if v.violation_kind == 'loose':
        type_desc = 'Loose type'
        noqa_code = 'loose-typing'
    else:
        type_desc = 'Mutable type'
        noqa_code = 'mutable-type'

    return (
        f'{v.filepath}:{v.line}:{v.column}: error: '
        f"{type_desc} '{v.bad_type}' in {v.context} annotation\n"
        f'    {v.source_line}\n'
        f'    Suggestion: Use {v.suggestion}\n'
        f'    Silence with: # noqa: {noqa_code}'
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check for mutable and loose types in model annotations.',
        epilog='Example: %(prog)s . --exclude .venv .git',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files or directories to check (use . for current directory)',
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        metavar='DIR',
        help='Directories to exclude when searching recursively',
    )

    args = parser.parse_args()
    exclude_dirs = set(args.exclude)

    # Collect files to check
    files: list[Path] = []
    for arg in args.paths:
        path = Path(arg)
        if arg == '.' or path.is_dir():
            # Directory: find all .py files recursively
            files.extend(find_python_files(path, exclude_dirs))
        elif path.suffix == '.py' and path.is_file():
            files.append(path)
        # Silently skip non-.py files (allows pre-commit to pass mixed file lists)

    if not files:
        return 0

    # Check all files
    all_violations: list[Violation] = []
    for filepath in files:
        violations = check_file(filepath)
        all_violations.extend(violations)

    # Report violations
    if all_violations:
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line)):
            print(format_violation(v))
            print()

        file_count = len({v.filepath for v in all_violations})
        mutable_count = sum(1 for v in all_violations if v.violation_kind == 'mutable')
        loose_count = sum(1 for v in all_violations if v.violation_kind == 'loose')
        summary_parts = []
        if mutable_count:
            summary_parts.append(f'{mutable_count} mutable')
        if loose_count:
            summary_parts.append(f'{loose_count} loose')
        print(f'Found {len(all_violations)} violation(s) ({", ".join(summary_parts)}) in {file_count} file(s).')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
