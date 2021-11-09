import numpy as np
import pandas as pd

import my_utils

path_to_project = my_utils.path_to_project


def load_fifa2018_stat():
    data = pd.read_csv(path_to_project + "Datasets/FIFA 2018 Statistics.csv")

    y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
    feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
    X = data[feature_names]

    return X, y, feature_names


if __name__ == "__main__":
    load_fifa2018_stat()
