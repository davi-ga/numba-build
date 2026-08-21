"""
BuiltinRewriter - Converte built-ins não suportados pelo Numba

Transforma:
    hex(integer)[2:]
    
Em:
    _int_to_hex(integer)  # função auxiliar Numba-compatível (já sem prefixo "0x")
"""

import ast


class BuiltinRewriter(ast.NodeTransformer):
    """Converte built-ins não suportados (hex, bin, oct) para funções auxiliares."""

    def visit_Subscript(self, node: ast.Subscript):
        """Detecta e reescreve hex()[2:], bin()[2:], oct()[2:]."""
        # Visita filhos primeiro
        self.generic_visit(node)
        
        # Verifica se é um subscript [2:]
        if not isinstance(node.slice, ast.Slice):
            return node
        
        # Verifica se lower é 2 (upper pode ser None para [2:])
        if node.slice.lower is None:
            return node
        
        if not isinstance(node.slice.lower, ast.Constant) or node.slice.lower.value != 2:
            return node
        
        # Verifica se o valor é uma chamada de hex(), bin(), oct() ou _int_to_hex(), etc
        if not isinstance(node.value, ast.Call):
            return node
        
        if not isinstance(node.value.func, ast.Name):
            return node
        
        # Se é hex()[2:], converte para _int_to_hex()
        if node.value.func.id == 'hex':
            node.value.func.id = '_int_to_hex'
            return node.value  # Remove o [2:]
        
        # Se é bin()[2:], converte para _int_to_bin()
        if node.value.func.id == 'bin':
            node.value.func.id = '_int_to_bin'
            return node.value  # Remove o [2:]
        
        # Se é oct()[2:], converte para _int_to_oct()
        if node.value.func.id == 'oct':
            node.value.func.id = '_int_to_oct'
            return node.value  # Remove o [2:]
        
        # Se já é _int_to_hex()[2:], remove o [2:]
        if node.value.func.id in ('_int_to_hex', '_int_to_bin', '_int_to_oct'):
            return node.value  # Remove o [2:]
        
        return node

    def visit_Call(self, node: ast.Call):
        """Detecta e reescreve chamadas de built-ins não suportados."""
        # Visita filhos primeiro
        self.generic_visit(node)
        
        # Verifica se é uma chamada de função
        if not isinstance(node.func, ast.Name):
            return node
        
        # Converte hex() para _int_to_hex()
        if node.func.id == 'hex':
            node.func.id = '_int_to_hex'
            return node
        
        # Converte bin() para _int_to_bin()
        if node.func.id == 'bin':
            node.func.id = '_int_to_bin'
            return node
        
        # Converte oct() para _int_to_oct()
        if node.func.id == 'oct':
            node.func.id = '_int_to_oct'
            return node
        
        return node


# Funções auxiliares Numba-compatíveis para built-ins
NUMBA_BUILTIN_HELPERS = '''
@numba.njit
def _int_to_hex(integer):
    """Converte inteiro para string hex (sem usar hex() built-in)"""
    hex_chars = "0123456789abcdef"
    if integer == 0:
        return "00"
    result = ""
    while integer > 0:
        result = hex_chars[integer % 16] + result
        integer = integer // 16
    if len(result) < 2:
        result = "0" + result
    return result

@numba.njit
def _int_to_bin(integer):
    """Converte inteiro para string binária (sem usar bin() built-in)"""
    if integer == 0:
        return "0"
    result = ""
    while integer > 0:
        result = str(integer % 2) + result
        integer = integer // 2
    return result

@numba.njit
def _int_to_oct(integer):
    """Converte inteiro para string octal (sem usar oct() built-in)"""
    if integer == 0:
        return "0"
    result = ""
    while integer > 0:
        result = str(integer % 8) + result
        integer = integer // 8
    return result
'''
