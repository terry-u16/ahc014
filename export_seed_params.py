import pandas as pd
import os

index = []
n_list = []
m_list = []

for seed in range(50000):
    index.append(seed)
    path = f"data/in/{seed:04}.txt"
    with open(path) as f:
        n, m = map(int, f.readline().split())
        n_list.append(n)
        m_list.append(m)

df = pd.DataFrame({"N": n_list, "M": m_list}, index=index)
df["Density"] = df["M"] / (df["N"] * df["N"])

df.to_csv("data/seed_params.csv")