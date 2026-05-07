import numba
from calculator import add, divide

@numba.njit
def get_total(numbers):
    return sum(numbers)

@numba.njit
def compute_average(numbers):
    if not numbers:
        return 0
    return divide(get_total(numbers), len(numbers))

def run_stats():
    scores = [8, 6, 9, 7, 10]
    print(f'Average: {compute_average(scores)}')
if __name__ == '__main__':
    run_stats()