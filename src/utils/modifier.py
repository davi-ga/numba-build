import ast
from typing import Union


class _EmptyListPatcher(ast.NodeTransformer):
    """Replaces bare ``x = []`` assignments with ``x = numba.typed.List()``.

    Only transforms the immediate body of the target function — nested
    functions, async functions, and classes are left untouched so that each
    scope is handled independently by ``Inserter``.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return node  # do not recurse into nested functions

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        return node  # do not recurse into nested async functions

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node  # do not recurse into nested classes

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
            # Replace `x = []` with `x = numba.typed.List()` so that numba
            # can infer the element type from subsequent append/index operations.
            node.value = ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="numba", ctx=ast.Load()),
                        attr="typed",
                        ctx=ast.Load(),
                    ),
                    attr="List",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
            ast.fix_missing_locations(node)
        return node


class Inserter(ast.NodeTransformer):

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Do not recurse into classes — numba.njit does not support methods.
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        # Async functions are not supported by numba.njit.
        return node

    # Builtins not supported in numba nopython mode.
    _UNSUPPORTED_BUILTINS = frozenset({"print", "input", "open", "exec", "eval","raise"})

    def _has_numba_decorator(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Attribute)
                and isinstance(dec.value, ast.Name)
                and dec.value.id == "numba"
            ):
                return True
            if isinstance(dec, ast.Name) and dec.id in ("njit", "jit"):
                return True
        return False

    def _has_unsupported_calls(self, node: ast.FunctionDef) -> bool:
        """Return True if the function body uses builtins unsupported by njit."""
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in self._UNSUPPORTED_BUILTINS
            ):
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if self._has_numba_decorator(node):
            return node

        if self._has_unsupported_calls(node):
            return node

        # Replace every bare `x = []` in the function body with
        # `x = numba.typed.List()` so numba can infer the element type.
        _EmptyListPatcher().generic_visit(node)

        decorator = ast.Attribute(
            value=ast.Name(id="numba", ctx=ast.Load()), attr="njit", ctx=ast.Load()
        )
        node.decorator_list.insert(0, decorator)

        return ast.fix_missing_locations(node)

    def importer(self, tree: ast.AST) -> None:
        if not any(
            isinstance(node, ast.Import) and node.names[0].name == "numba"
            for node in tree.body
        ):
            import_node = ast.Import(names=[ast.alias(name="numba")])
            tree.body.insert(0, import_node)
            ast.fix_missing_locations(import_node)
