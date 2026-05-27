DEEP_TO_LIST_SRC = """
from numba.typed import List as NumbaList

def _sanitize_for_numba(obj):
    if isinstance(obj, list):
        typed_l = NumbaList()
        for item in obj:
            typed_l.append(_sanitize_for_numba(item))
        return typed_l
    elif isinstance(obj, tuple):
        return tuple(0.0 if isinstance(x, str) else _sanitize_for_numba(x) for x in obj)
    return obj

def _deep_to_list(obj):
    if isinstance(obj, (str, bytes)):
        return obj
    try:
        return [_deep_to_list(item) for item in obj]
    except TypeError:
        return obj
"""
