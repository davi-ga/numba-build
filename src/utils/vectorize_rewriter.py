"""
VectorizeRewriter - Converte np.vectorize para loops explícitos

Transforma:
    result = np.vectorize(func)(array)
    
Em:
    result = np.empty_like(array)
    for i in range(len(array)):
        result[i] = func(array[i])
"""

import ast


class VectorizeRewriter(ast.NodeTransformer):
    """Converte np.vectorize(func)(array) para loops explícitos."""

    def visit_Call(self, node: ast.Call):
        """Detecta e reescreve chamadas de np.vectorize em qualquer contexto."""
        # Visita filhos primeiro
        self.generic_visit(node)
        
        # Verifica se é np.vectorize(func)(array)
        if not isinstance(node.func, ast.Call):
            return node
        
        # Verifica se a função interna é np.vectorize
        if not isinstance(node.func.func, ast.Attribute):
            return node
        
        if node.func.func.attr != 'vectorize':
            return node
        
        # Extrai a função e o array
        if len(node.func.args) != 1:
            return node
        
        func_arg = node.func.args[0]
        array_arg = node.args[0] if node.args else None
        
        if array_arg is None:
            return node
        
        # Gera uma expressão de lista comprehension: [func(x) for x in array]
        list_comp = ast.ListComp(
            elt=ast.Call(
                func=func_arg,
                args=[ast.Name(id='x', ctx=ast.Load())],
                keywords=[]
            ),
            generators=[
                ast.comprehension(
                    target=ast.Name(id='x', ctx=ast.Store()),
                    iter=array_arg,
                    ifs=[],
                    is_async=0
                )
            ]
        )
        
        return list_comp
