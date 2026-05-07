from calculator import add, divide

def compute_average(numbers):
    total = 0
    for n in numbers:
        total = add(total, n)
    return divide(total, len(numbers))

if __name__ == "__main__":
    scores = [8, 6, 9, 7, 10]
    avg = compute_average(scores)
    print(f"Average: {avg}")
