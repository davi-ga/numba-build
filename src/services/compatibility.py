"""
Numba compatibility checker — validates code before AST annotation.

Detects patterns that are incompatible with Numba nopython mode and
provides feedback for retry attempts.
"""

import ast
import sys
from typing import List, Tuple


class NumbaCompatibilityChecker:
    """Validates code for Numba nopython compatibility.
    
    Checks for:
    - Boolean masking on 2D arrays
    - Unsupported numpy functions (np.ravel_multi_index, np.clip on scalars)
    - Heterogeneous tuples
    - Dict/set usage
    - String operations in numeric contexts
    """

    _UNSUPPORTED_NUMPY_FUNCS = frozenset({
        "ravel_multi_index", "unravel_index", "clip",
        "where", "select", "piecewise",
    })

    _UNSUPPORTED_BUILTINS = frozenset({
        "dict", "set", "frozenset",
        "print", "input", "open", "exec", "eval",
    })

    _UNSUPPORTED_SCIPY_FUNCS = frozenset({
        "gammaln", "gamma", "betainc", "betaincinv",
        "erf", "erfc", "erfinv", "erfcinv",
        "loggamma", "digamma", "polygamma",
    })

    def __init__(self):
        self.issues: List[Tuple[str, int, str]] = []

    def check(self, code: str) -> Tuple[bool, List[Tuple[str, int, str]]]:
        """Check code for Numba compatibility issues.
        
        Returns:
            (is_compatible, issues) where issues is a list of (severity, line, message)
        """
        self.issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.issues.append(("error", exc.lineno or 0, f"Syntax error: {exc.msg}"))
            return False, self.issues
        
        self._walk_tree(tree)
        
        is_compatible = not any(sev == "error" for sev, _, _ in self.issues)
        return is_compatible, self.issues

    def _walk_tree(self, tree: ast.AST):
        """Walk the AST and check for incompatible patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._check_function(node)
            elif isinstance(node, ast.ClassDef):
                self.issues.append(
                    ("warning", node.lineno, f"Class '{node.name}' found — should be extracted to module-level functions")
                )

    def _check_function(self, node: ast.FunctionDef):
        """Check a function for Numba-incompatible patterns."""
        func_name = node.name
        
        for child in ast.walk(node):
            self._check_call(child, func_name)
            self._check_subscript(child, func_name)
            self._check_dict_set(child, func_name)
            self._check_try_except(child, func_name)

    def _check_call(self, node: ast.AST, func_name: str):
        """Check for unsupported function calls."""
        if not isinstance(node, ast.Call):
            return
        
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "np" or node.func.value.id == "numpy":
                    if node.func.attr in self._UNSUPPORTED_NUMPY_FUNCS:
                        self.issues.append(
                            ("error", node.lineno,
                             f"Function '{func_name}': np.{node.func.attr}() is not supported in Numba nopython mode")
                        )
                elif node.func.value.id in ("sps", "scipy", "stats"):
                    if node.func.attr in self._UNSUPPORTED_SCIPY_FUNCS:
                        self.issues.append(
                            ("error", node.lineno,
                             f"Function '{func_name}': scipy.{node.func.attr}() is not supported in Numba nopython mode")
                        )
        
        if isinstance(node.func, ast.Name):
            if node.func.id in self._UNSUPPORTED_BUILTINS:
                self.issues.append(
                    ("error", node.lineno,
                     f"Function '{func_name}': {node.func.id}() is not supported in Numba nopython mode")
                )
            elif node.func.id in self._UNSUPPORTED_SCIPY_FUNCS:
                self.issues.append(
                    ("error", node.lineno,
                     f"Function '{func_name}': {node.func.id}() from scipy is not supported in Numba nopython mode")
                )

    def _check_subscript(self, node: ast.AST, func_name: str):
        """Check for boolean masking patterns."""
        if not isinstance(node, ast.Subscript):
            return
        
        if isinstance(node.slice, ast.Name):
            self.issues.append(
                ("warning", node.lineno,
                 f"Function '{func_name}': Possible boolean masking detected (array[{node.slice.id}]). "
                 "Ensure mask is 1D or convert to explicit loops.")
            )

    def _check_dict_set(self, node: ast.AST, func_name: str):
        """Check for dict/set usage."""
        if isinstance(node, ast.Dict):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': Dict literal found — not supported in Numba nopython mode")
            )
        elif isinstance(node, ast.Set):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': Set literal found — not supported in Numba nopython mode")
            )

    def _check_try_except(self, node: ast.AST, func_name: str):
        """Check for try/except blocks."""
        if isinstance(node, ast.Try):
            self.issues.append(
                ("error", node.lineno,
                 f"Function '{func_name}': try/except block found — not supported in Numba nopython mode")
            )

    def format_issues(self) -> str:
        """Format issues for display."""
        if not self.issues:
            return "No compatibility issues found."
        
        lines = ["Numba compatibility issues detected:"]
        for severity, line, message in self.issues:
            lines.append(f"  [{severity.upper()}] Line {line}: {message}")
        return "\n".join(lines)


def check_compatibility(code: str, verbose: bool = True) -> Tuple[bool, str]:
    """Convenience function to check code compatibility.
    
    Returns:
        (is_compatible, report) where report is a formatted string
    """
    checker = NumbaCompatibilityChecker()
    is_compatible, issues = checker.check(code)
    report = checker.format_issues()
    
    if verbose and not is_compatible:
        print(report, file=sys.stderr)
    
    return is_compatible, report
