import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int)
args = parser.parse_args()
n = args.n

if n <= 0 or n % 2 != 1:
    print("invalid n.")
    exit()

score = np.zeros((n, n))
center = (n - 1) / 2

for x in range(n):
    for y in range(n):
        dx = x - center
        dy = y - center
        s = dx * dx + dy * dy + 1
        score[x, y] = s

score /= score.max()
sns.heatmap(score, cmap="coolwarm", vmin=0)
plt.show()