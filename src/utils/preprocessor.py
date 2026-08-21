"""
Preprocessing transformers for forge.

These AST transformers run BEFORE the LLM to prepare code for Numba compilation:
- ClassExtractor: Extracts class methods to module-level functions
- IOStripper: Removes I/O operations (print, logging, file ops, try/except)
- BooleanMaskRewriter: Converts boolean masking to explicit loops
- VectorizeRewriter: Converts np.vectorize to explicit loops
- BuiltinRewriter: Converts hex(), bin(), oct() to Numba-compatible functions
- UniqueRewriter: Converts np.unique(return_counts=True) to manual implementation
"""

import ast
from typing import Union
from utils.vectorize_rewriter import VectorizeRewriter
from utils.builtin_rewriter import BuiltinRewriter
from utils.unique_rewriter import UniqueRewriter


class ClassExtractor(ast.NodeTransformer):
    """Extracts class methods to module-level functions.
    
    Transforms:
        class Foo:
            def method(self, x):
                return x + 1
    
    Into:
        def foo_method(self, x):
            return x + 1
    """

    def visit_ClassDef(self, node: ast.ClassDef) -> list:
        self.generic_visit(node)
        
        extracted_functions = []
        class_name = node.name.lower()
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    continue
                
                new_name = f"{class_name}_{item.name}"
                item.name = new_name
                
                if isinstance(item, ast.FunctionDef):
                    extracted_functions.append(item)
        
        return extracted_functions if extracted_functions else []


class IOStripper(ast.NodeTransformer):
    """Removes I/O operations that are incompatible with Numba nopython mode.
    
    Removes:
    - print() calls
    - logging calls
    - file operations (open, read, write)
    - try/except blocks (keeps the try body, removes except)
    - raise statements
    """

    _IO_FUNCTIONS = frozenset({
        "print", "input", "open", "exec", "eval",
        "logging", "logger", "log",
    })

    def visit_Expr(self, node: ast.Expr) -> Union[ast.Expr, None]:
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id in self._IO_FUNCTIONS:
                    return None
            elif isinstance(node.value.func, ast.Attribute):
                if isinstance(node.value.func.value, ast.Name):
                    if node.value.func.value.id in self._IO_FUNCTIONS:
                        return None
        return node

    def visit_Try(self, node: ast.Try) -> list:
        self.generic_visit(node)
        return node.body

    def visit_Raise(self, node: ast.Raise) -> None:
        return None


class BooleanMaskRewriter(ast.NodeTransformer):
    """Converts boolean masking operations to explicit loops.
    
    Transforms:
        result[bool_mask] = value
        result[result == 0.0] = 1.0
    
    Into:
        for _i in range(result.shape[0]):
            for _j in range(result.shape[1]):
                if bool_mask[_i, _j]:
                    result[_i, _j] = value
    """

    def _is_bool_mask_slice(self, slice_node: ast.AST) -> bool:
        """Check if a subscript slice is a boolean mask (Name or Compare)."""
        if isinstance(slice_node, ast.Name):
            return True
        if isinstance(slice_node, ast.Compare):
            return True
        return False

    def _make_indexed_condition(self, condition: ast.AST, array_name: str) -> ast.AST:
        """Replace array references in a Compare condition with indexed versions."""
        i_idx = ast.Name(id="_i", ctx=ast.Load())
        j_idx = ast.Name(id="_j", ctx=ast.Load())
        indexed = ast.Subscript(
            value=ast.Name(id=array_name, ctx=ast.Load()),
            slice=ast.Tuple(elts=[i_idx, j_idx], ctx=ast.Load()),
            ctx=ast.Load(),
        )

        class _ArrayReplacer(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == array_name:
                    return ast.copy_location(indexed, node)
                return node

        return _ArrayReplacer().visit(condition)

    def visit_Assign(self, node: ast.Assign) -> Union[ast.Assign, ast.For]:
        self.generic_visit(node)
        
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and self._is_bool_mask_slice(node.targets[0].slice)
        ):
            return node
        
        target_array = node.targets[0].value
        slice_node = node.targets[0].slice
        value = node.value
        
        if not isinstance(target_array, ast.Name):
            return node
        
        array_name = target_array.id
        
        i_idx = ast.Name(id="_i", ctx=ast.Load())
        j_idx = ast.Name(id="_j", ctx=ast.Load())
        
        inner_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=array_name, ctx=ast.Store()),
                    slice=ast.Tuple(elts=[i_idx, j_idx], ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=value,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
        
        if isinstance(slice_node, ast.Name):
            mask_name = slice_node.id
            if_test = ast.Subscript(
                value=ast.Name(id=mask_name, ctx=ast.Load()),
                slice=ast.Tuple(elts=[i_idx, j_idx], ctx=ast.Load()),
                ctx=ast.Load(),
            )
        elif isinstance(slice_node, ast.Compare):
            if_test = self._make_indexed_condition(slice_node, array_name)
        else:
            return node
        
        if_stmt = ast.If(
            test=if_test,
            body=[inner_assign],
            orelse=[],
        )
        
        j_loop = ast.For(
            target=ast.Name(id="_j", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=array_name, ctx=ast.Load()),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=1),
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
            body=[if_stmt],
            orelse=[],
        )
        
        i_loop = ast.For(
            target=ast.Name(id="_i", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=array_name, ctx=ast.Load()),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=0),
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
            body=[j_loop],
            orelse=[],
        )
        
        return ast.fix_missing_locations(i_loop)


def preprocess_code(code: str) -> str:
    """Apply all preprocessing transformers to the code.
    
    Returns the preprocessed code as a string.
    """
    tree = ast.parse(code)
    
    # Apply transformers in order
    tree = ClassExtractor().visit(tree)
    tree = IOStripper().visit(tree)
    tree = BooleanMaskRewriter().visit(tree)
    tree = VectorizeRewriter().visit(tree)
    tree = BuiltinRewriter().visit(tree)
    tree = UniqueRewriter().visit(tree)
    
    return ast.unparse(ast.fix_missing_locations(tree))
