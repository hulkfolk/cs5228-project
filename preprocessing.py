import os
from typing import Tuple

import pandas as pd


def filter_raw_data() -> Tuple:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'raw_data', 'Xtrain.csv'))
    ids_to_remove = set()
    for index, row in df.iterrows():
        # remove nan value
        # remove 0 in NewExist
        # TODO: should we remove 0 and T in RevLineCr
        if row.isnull().any() or row['NewExist'] == 0:
            ids_to_remove.add(index)
    return tuple(ids_to_remove)


def save_cleaned_data(ids_to_remove: Tuple):
    x_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'raw_data', 'Xtrain.csv'))
    y_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'raw_data', 'Ytrain.csv'))
    cleaned_x_df = x_df.drop(list(ids_to_remove))
    cleaned_y_df = y_df.drop(list(ids_to_remove))

    cleaned_x_df.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'cleaned_data'), 'Xtrain.csv'))
    cleaned_y_df.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'cleaned_data'), 'Ytrain.csv'))


if __name__ == '__main__':
    save_cleaned_data(filter_raw_data())