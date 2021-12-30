import numpy as np
import pandas as pd

import my_utils

path_to_project = my_utils.path_to_project


def load_heart_failure():
    data = pd.read_csv(path_to_project + "Datasets/heart_failure_clinical_records_dataset.csv")

    y = data.DEATH_EVENT.values
    x_data = data.drop(["DEATH_EVENT"], axis=1)
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values  # Normalize data

    return x, y, x.columns.values


if __name__ == "__main__":
    load_heart_failure()
