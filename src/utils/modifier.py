import ast
from typing import Union


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
    _UNSUPPORTED_BUILTINS = frozenset({"print", "input", "open", "exec", "eval"})

    def _has_numba_decorator(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        for dec in node.decorator_list:
            # @numba.njit  /  @numba.jit
            if (
                isinstance(dec, ast.Attribute)
                and isinstance(dec.value, ast.Name)
                and dec.value.id == "numba"
            ):
                return True
            # bare @njit or @jit
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
        # Skip if already annotated to avoid duplicates.
        if self._has_numba_decorator(node):
            return node
        # Skip if the body uses builtins incompatible with nopython mode.
        if self._has_unsupported_calls(node):
            return node

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
