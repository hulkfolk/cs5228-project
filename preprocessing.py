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


def process_x(ids: Tuple, source: str, destination: str):
    x_df = pd.read_csv(source)
    # FIXME: text columns is removed temporarily
    cleaned_x_df = x_df.drop(list(ids)).drop(columns=['Id', 'Name', 'City', 'State', 'Zip', 'BankState', 'NAICS'])
    # convert date to float
    cleaned_x_df['ApprovalDate'] = cleaned_x_df['ApprovalDate'].apply(lambda x: datetime.strptime(x.strip(), '%d-%b-%y').timestamp())
    cleaned_x_df['ApprovalFY'] = cleaned_x_df['ApprovalFY'].apply(
        lambda x: datetime.strptime(str(x).replace('A', '').strip(), '%Y').timestamp() if re.match(r'^\d{4}A$', str(x).strip()) else datetime.strptime(str(x).strip(), '%Y').timestamp())
    cleaned_x_df['DisbursementDate'] = cleaned_x_df['DisbursementDate'].apply(lambda x: datetime.strptime(x.strip(), '%d-%b-%y').timestamp() if not pd.isnull(x) else x)
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
    banks = cleaned_x_df['Bank'].unique()
    banks_index = {banks[i]: float(i) for i in range(len(banks))}
    cleaned_x_df['Bank'] = cleaned_x_df['Bank'].map(banks_index)

    cleaned_x_df.to_csv(destination, index=False)


def process_y(ids: Tuple, file_path: str):
    y_df = pd.read_csv(file_path)
    cleaned_y_df = y_df.drop(list(ids)).drop(columns=['Id'])
    cleaned_y_df.to_csv('Ytrain.csv', index=False)


if __name__ == '__main__':
    ids_to_remove = filter_raw_data()
    process_x(ids_to_remove, os.path.join(os.path.dirname(__file__), 'raw_data', 'Xtrain.csv'), 'Xtrain.csv')
    process_y(ids_to_remove, os.path.join(os.path.dirname(__file__), 'raw_data', 'Ytrain.csv'))