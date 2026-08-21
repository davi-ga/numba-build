import ast

from utils.modifier import Inserter
from utils.numba_filter import NumbaDecoratorFilter


class AnnotatorService:
    def __init__(self):
        self.inserter = Inserter()

    def transform(self, modified_code: str) -> str:
        tree = ast.parse(modified_code)

        self.inserter.importer(tree)

        modified = self.inserter.visit(tree)

        # Remove @numba.njit de funções que usam bibliotecas externas
        modified = NumbaDecoratorFilter().visit(modified)

        return ast.unparse(modified)
