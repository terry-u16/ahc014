import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parameters import TIMESTAMP


def load_input_params(dir: str) -> pd.DataFrame:
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

def load_df() -> pd.DataFrame:
    csv_path = f"./data/parameter_sample/result_{TIMESTAMP}.csv"
    df_score = pd.read_csv(csv_path)

    df_param = load_input_params("./data/in/")
    df_score = df_score.merge(df_param, left_on="seed", right_index=True, how="left")

    return df_score[["N", "Density", "temp0", "temp1", "score0", "score1"]]