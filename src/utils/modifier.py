import ast
from typing import Union


class _EmptyListPatcher(ast.NodeTransformer):
    """Replaces bare ``x = []`` assignments with ``x = numba.typed.List()``."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
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


_DEEP_TO_LIST_SRC = """\
def _deep_to_list(obj):
    try:
        return [_deep_to_list(item) for item in obj]
    except TypeError:
        return obj
"""


class Inserter(ast.NodeTransformer):

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return node

    _UNSUPPORTED_BUILTINS = frozenset({"print", "input", "open", "exec", "eval", "raise"})

    def _has_numba_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
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
        for child in ast.walk(node):
            if isinstance(child, ast.Try):          # try/except not supported by njit
                return True
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in self._UNSUPPORTED_BUILTINS
            ):
                return True
        return False

    def _get_empty_list_names(self, func_node: ast.FunctionDef) -> set:
        """Return names of bare [] assignments in the function's direct body."""
        names = set()
        for stmt in func_node.body:
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.List)
                and len(stmt.value.elts) == 0
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                names.add(stmt.targets[0].id)
        return names

    def _returns_typed_list(self, func_node: ast.FunctionDef, list_names: set) -> bool:
        """Return True if any return statement returns a typed.List variable."""
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Return)
                and node.value is not None
                and isinstance(node.value, ast.Name)
                and node.value.id in list_names
            ):
                return True
        return False

    def _make_wrapper(self, orig_name: str, jit_name: str, args: ast.arguments) -> ast.FunctionDef:
        """Generate a plain Python wrapper that converts typed.List → list."""
        call_args = [
            ast.Name(id=a.arg, ctx=ast.Load())
            for a in args.posonlyargs + args.args
        ]
        call_kwargs = [
            ast.keyword(arg=a.arg, value=ast.Name(id=a.arg, ctx=ast.Load()))
            for a in args.kwonlyargs
        ]
        if args.vararg:
            call_args.append(
                ast.Starred(value=ast.Name(id=args.vararg.arg, ctx=ast.Load()), ctx=ast.Load())
            )
        if args.kwarg:
            call_kwargs.append(
                ast.keyword(arg=None, value=ast.Name(id=args.kwarg.arg, ctx=ast.Load()))
            )

        jit_call = ast.Call(
            func=ast.Name(id=jit_name, ctx=ast.Load()),
            args=call_args,
            keywords=call_kwargs,
        )
        deep_call = ast.Call(
            func=ast.Name(id="_deep_to_list", ctx=ast.Load()),
            args=[jit_call],
            keywords=[],
        )
        wrapper = ast.FunctionDef(
            name=orig_name,
            args=args,
            body=[ast.Return(value=deep_call)],
            decorator_list=[],
            returns=None,
            lineno=0,
            col_offset=0,
        )
        return ast.fix_missing_locations(wrapper)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Union[ast.FunctionDef, list]:
        if self._has_numba_decorator(node):
            return node
        if self._has_unsupported_calls(node):
            return node

        list_names = self._get_empty_list_names(node)
        needs_wrapper = bool(list_names) and self._returns_typed_list(node, list_names)

        _EmptyListPatcher().generic_visit(node)

        decorator = ast.Attribute(
            value=ast.Name(id="numba", ctx=ast.Load()), attr="njit", ctx=ast.Load()
        )
        node.decorator_list.insert(0, decorator)

        if needs_wrapper:
            orig_name = node.name
            jit_name = f"_{orig_name}_jit"
            node.name = jit_name
            wrapper = self._make_wrapper(orig_name, jit_name, node.args)

            return [ast.fix_missing_locations(node), ast.fix_missing_locations(wrapper)]

        return ast.fix_missing_locations(node)

    def importer(self, tree: ast.AST) -> None:
        if not any(
            isinstance(node, ast.Import) and node.names[0].name == "numba"
            for node in tree.body
        ):
            import_node = ast.Import(names=[ast.alias(name="numba")])
            tree.body.insert(0, import_node)
            ast.fix_missing_locations(import_node)

        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "_deep_to_list"
            for node in tree.body
        ):
            helper_node = ast.parse(_DEEP_TO_LIST_SRC).body[0]
            ast.fix_missing_locations(helper_node)
            insert_idx = 0
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_idx = i + 1
            tree.body.insert(insert_idx, helper_node)