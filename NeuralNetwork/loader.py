import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parameters import TIMESTAMP


def load_df() -> pd.DataFrame:
    csv_path = f"./data/parameter_sample/result_{TIMESTAMP}.csv"
    df_score = pd.read_csv(csv_path)

    df_param = pd.read_csv("./data/seed_params.csv")
    df_score = df_score.merge(df_param, left_on="seed",
                              right_index=True, how="left")

    return df_score[["N", "Density", "temp0", "temp1", "score"]]
