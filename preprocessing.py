import os
import re
from typing import Tuple
from datetime import datetime

import pandas as pd


def filter_raw_data() -> Tuple:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'raw_data', 'Xtrain.csv'))
    ids_to_remove = set()
    for index, row in df.iterrows():
        # remove role with nan value, 0 in NewExist, RevLineCr not in [Y, N], LowDoc not in [Y, N]
        if row.isnull().any() or row['NewExist'] == 0 or row['RevLineCr'].strip() not in ['Y', 'N'] or row['LowDoc'].strip() not in ['Y', 'N']:
            ids_to_remove.add(index)
    return tuple(ids_to_remove)


def save_cleaned_data(ids_to_remove: Tuple):
    x_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'raw_data', 'Xtrain.csv'))
    y_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'raw_data', 'Ytrain.csv'))
    # FIXME: text columns is removed temporarily
    cleaned_x_df = x_df.drop(list(ids_to_remove)).drop(columns=['Id', 'Name', 'City', 'State', 'Zip', 'Bank', 'BankState', 'NAICS'])
    # convert date to float
    cleaned_x_df['ApprovalDate'] = cleaned_x_df['ApprovalDate'].apply(lambda x: datetime.strptime(x.strip(), '%d-%b-%y').timestamp())
    cleaned_x_df['ApprovalFY'] = cleaned_x_df['ApprovalFY'].apply(
        lambda x: datetime.strptime(x.replace('A', '').strip(), '%Y').timestamp() if re.match(r'^\d{4}A$', x) else datetime.strptime(x.strip(), '%Y').timestamp())
    cleaned_x_df['DisbursementDate'] = cleaned_x_df['DisbursementDate'].apply(lambda x: datetime.strptime(x.strip(), '%d-%b-%y').timestamp())
    # convert int to float
    cleaned_x_df['Term'] = cleaned_x_df['Term'].apply(lambda x: float(x))
    cleaned_x_df['NoEmp'] = cleaned_x_df['NoEmp'].apply(lambda x: float(x))
    cleaned_x_df['CreateJob'] = cleaned_x_df['CreateJob'].apply(lambda x: float(x))
    cleaned_x_df['RetainedJob'] = cleaned_x_df['RetainedJob'].apply(lambda x: float(x))
    cleaned_x_df['FranchiseCode'] = cleaned_x_df['FranchiseCode'].apply(lambda x: float(x))
    cleaned_x_df['UrbanRural'] = cleaned_x_df['UrbanRural'].apply(lambda x: float(x))
    # remove $ and , from currency value
    cleaned_x_df['DisbursementGross'] = cleaned_x_df['DisbursementGross'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    cleaned_x_df['BalanceGross'] = cleaned_x_df['BalanceGross'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    cleaned_x_df['GrAppv'] = cleaned_x_df['GrAppv'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    cleaned_x_df['SBA_Appv'] = cleaned_x_df['SBA_Appv'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    # convert categorical data into float
    cleaned_x_df['RevLineCr'] = cleaned_x_df['RevLineCr'].map({'N': 0.0, 'Y': 1.0})
    cleaned_x_df['LowDoc'] = cleaned_x_df['LowDoc'].map({'N': 0.0, 'Y': 1.0})

    cleaned_y_df = y_df.drop(list(ids_to_remove)).drop(columns=['Id'])

    cleaned_x_df.to_csv('Xtrain.csv', index=False)
    cleaned_y_df.to_csv('Ytrain.csv', index=False)


if __name__ == '__main__':
    save_cleaned_data(filter_raw_data())