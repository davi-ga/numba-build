import ast

from utils.modifier import Inserter


class AnnotatorService:
    def __init__(self):
        self.inserter = Inserter()

    def transform(self, modified_code: str) -> str:
        tree = ast.parse(modified_code)

        self.inserter.importer(tree)

        modified = self.inserter.visit(tree)

        return ast.unparse(modified)
