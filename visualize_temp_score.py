import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_input_params(dir: str) -> pd.DataFrame:
    index = []
    n_list = []
    m_list = []

    for seed in range(50000):
        index.append(seed)
        path = os.path.join(dir, f"{seed:04}.txt")
        with open(path) as f:
            n, m = map(int, f.readline().split())
            n_list.append(n)
            m_list.append(m)

    df = pd.DataFrame({"N": n_list, "M": m_list}, index=index)
    df["Density"] = df["M"] / (df["N"] * df["N"])
    return df


df_score = pd.read_csv("./data/parameter_sample/result_20220928_020201.csv")
print(df_score.head())

df_param = read_input_params("./data/in/")
df_score = df_score.merge(df_param, left_index=True, right_index=True)
print(df_score.head())

df_score["temp0"] = np.log10(df_score["temp0"])
df_score["temp1"] = np.log10(df_score["temp1"])
#df_score = df_score[(df_score["Density"] <= 0.06)]
print(df_score.head())

print("-----")

for i in range(10):
    low = i * 0.1 + np.log10(5)
    high = (i + 1) * 0.1 + np.log10(5)
    f = np.power(10, low)
    t = np.power(10, high)
    s = df_score[(df_score["temp0"] > low) & (
        df_score["temp0"] < high)]["score"].mean()
    print(f"{f:.2f} {t:.2f} {s}")

print("-----")

for i in range(10):
    low = i * 0.1
    high = (i + 1) * 0.1
    f = np.power(10, low)
    t = np.power(10, high)
    s = df_score[(df_score["temp1"] > low) & (
        df_score["temp1"] < high)]["score"].mean()
    print(f"{f:.2f} {t:.2f} {s}")

#sns.pairplot(df_score[["score", "temp0", "temp1"]], plot_kws={'alpha': 0.5})
# plt.show()
