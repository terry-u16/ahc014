import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def read_input_params(dir: str) -> pd.DataFrame:
    index = []
    n_list = []
    m_list = []

    for seed in range(10000):
        index.append(seed)
        path = os.path.join(dir, f"{seed:04}.txt")
        with open(path) as f:
            n, m = map(int, f.readline().split())
            n_list.append(n)
            m_list.append(m)

    df = pd.DataFrame({"N": n_list, "M": m_list}, index=index)
    df["Density"] = df["M"] / (df["N"] * df["N"])
    return df


def read_results(path: str, input_params: pd.DataFrame) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["Results"]).set_index("Seed")
    return input_params.merge(df, left_index=True, right_index=True)


def draw_plot(df: pd.DataFrame, title: str, xlim: tuple[float, float], ylim: tuple[float, float]):
    n_list = range(31, 61 + 1, 2)
    fig, axes = plt.subplots(4, 4, figsize=(18, 10))
    fig.suptitle(title)

    for i in range(len(n_list)):
        n = n_list[i]
        data = df[df["N"] == n]
        ax = axes[i // 4][i % 4]
        x = data["Density"]
        y = data["Score"]
        p = np.polyfit(x, y, 1)
        (a, b) = p
        f = np.poly1d(p)
        ax.scatter(x, y, s=12, linewidths=0, alpha=0.3)
        ax.plot(xlim, f(xlim), color="darkorange")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("density")
        ax.set_ylabel("score")
        ax.set_title(f"N={n} : f(x)=({a:.2e})x+({b:.2e})")

    plt.tight_layout()

    fig.savefig(f"{title}.png")
    plt.show()


input_params = read_input_params("data/in")
results_x01 = read_results(
    "data/results/result_20220926_115006.json", input_params)
results_x02 = read_results(
    "data/results/result_20220926_130036.json", input_params)
results_x10 = read_results(
    "data/results/result_20220926_143858.json", input_params)

results_x02["Score"] = results_x02["Score"] - results_x01["Score"]
results_x10["Score"] = results_x10["Score"] - results_x01["Score"]

draw_plot(results_x01, "time_x1", (0.0, 0.1), (0, 3e6))
draw_plot(results_x02, "time_x2_diff", (0.0, 0.1), (-1e6, 1.5e6))
draw_plot(results_x10, "time_x10_diff", (0.0, 0.1), (-1e6, 1.5e6))
