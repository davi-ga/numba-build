import numba

@numba.njit
def add(a, b):
    return a + b

@numba.njit
def subtract(a, b):
    return a - b

@numba.njit
def multiply(a, b):
    return a * b

@numba.njit
def divide(a, b):
    if b == 0:
        raise ValueError('Cannot divide by zero')
    return a / b