"""
UniqueRewriter - Converte np.unique com return_counts para implementação manual

Transforma:
    unique_vals, counts = np.unique(arr, return_counts=True)
    
Em:
    unique_vals, counts = _np_unique_with_counts(arr)
"""

import ast


class UniqueRewriter(ast.NodeTransformer):
    """Converte np.unique(arr, return_counts=True) para função auxiliar."""

    def visit_Assign(self, node: ast.Assign):
        """Detecta e reescreve assignments com np.unique(return_counts=True)."""
        # Verifica se é uma chamada de np.unique
        if not isinstance(node.value, ast.Call):
            return node
        
        if not isinstance(node.value.func, ast.Attribute):
            return node
        
        if node.value.func.attr != 'unique':
            return node
        
        # Verifica se tem return_counts=True
        has_return_counts = False
        for kw in node.value.keywords:
            if kw.arg == 'return_counts':
                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    has_return_counts = True
                    break
        
        if not has_return_counts:
            return node
        
        # Verifica se é uma atribuição dupla: unique_vals, counts = ...
        if len(node.targets) != 1:
            return node
        
        target = node.targets[0]
        if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
            return node
        
        # Substitui por chamada de função auxiliar
        array_arg = node.value.args[0] if node.value.args else None
        if array_arg is None:
            return node
        
        # Cria chamada: _np_unique_with_counts(arr)
        new_call = ast.Call(
            func=ast.Name(id='_np_unique_with_counts', ctx=ast.Load()),
            args=[array_arg],
            keywords=[]
        )
        
        node.value = new_call
        return node


# Função auxiliar Numba-compatível para np.unique com return_counts
NUMBA_UNIQUE_HELPER = '''
@numba.njit
def _np_unique_with_counts(arr):
    """Implementação Numba-compatível de np.unique(arr, return_counts=True)"""
    # Ordena o array
    sorted_arr = np.sort(arr)
    
    # Conta valores únicos
    n = len(sorted_arr)
    unique_count = 1
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i-1]:
            unique_count += 1
    
    # Aloca arrays de saída
    unique_vals = np.empty(unique_count, dtype=arr.dtype)
    counts = np.empty(unique_count, dtype=np.int64)
    
    # Preenche arrays
    unique_vals[0] = sorted_arr[0]
    counts[0] = 1
    idx = 0
    
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i-1]:
            idx += 1
            unique_vals[idx] = sorted_arr[i]
            counts[idx] = 1
        else:
            counts[idx] += 1
    
    return unique_vals, counts
'''
