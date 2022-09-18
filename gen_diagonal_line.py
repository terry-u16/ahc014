n = 31
m = n * 3 - 2


print(f"{n} {m}")

for begin in [-1, 0, 1]:
    for y in range(n):
        x = begin + y
        if 0 <= x < n:
            print(f"{x} {y}")
