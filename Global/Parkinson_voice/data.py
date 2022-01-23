import pandas as pd

import my_utils

path_to_project = my_utils.path_to_project


def load_parkinson_voice():
    df = pd.read_csv(path_to_project + "Datasets/Parkinsson disease.csv")  # import dataset

    df = df.drop(['name'], axis=1)
    df_corr = df.corr()

    # find highly correlated features and drop them
    higly_correlated_features = set()

    for feature_column in range(0, len(df_corr.columns)):
        if feature_column == 'status':
            continue
        feature_column_name = df_corr.columns[feature_column]
        for feature_row in range(0, len(df_corr.index)):
            feature_row_name = df_corr.index[feature_row]
            if feature_row_name == feature_column_name:
                continue
            corr_value = df_corr.iloc[feature_column][feature_row]
            if corr_value > 0.67:
                higly_correlated_features.add(feature_row_name)
    # print(higly_correlated_features)
    df = df.drop(higly_correlated_features, axis=1)

    X = df.drop(['status'], axis=1).values
    Y = df['status'].values
    target_names = set(Y)

    return X, Y, df.drop(['status'], axis=1).columns, target_names


if __name__ == "__main__":
    load_parkinson_voice()
