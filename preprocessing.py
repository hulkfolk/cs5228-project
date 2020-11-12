import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

CAT_FEATURES = ['City', 'State', 'Zip', 'Bank', 'BankState', 'NAICS', 'ApprovalDate', 'ApprovalFY',
                'NewExist', 'FranchiseCode', 'UrbanRural', 'RevLineCr', 'LowDoc', 'DisbursementDate', 'Recession']

NAICS_TO_INDUSTRY = {
    '11': 'AGS',
    '21': 'MINING',
    '22': 'UTILITIES',
    '23': 'CONSTRUCTION',
    '31': 'MANUFACTURING',
    '32': 'MANUFACTURING',
    '33': 'MANUFACTURING',
    '42': 'WHOLESALE_TRADE',
    '44': 'RETAIL_TRADE',
    '45': 'RETAIL_TRADE',
    '48': 'TRANSPORTATION_WAREHOUSING',
    '49': 'TRANSPORTATION_WAREHOUSING',
    '51': 'INFORMATION',
    '52': 'FINANCE_INSURANCE',
    '53': 'REAL_ESTATE',
    '54': 'SERVICES',
    '55': 'MANAGEMENT',
    '56': 'ADMIN',
    '61': 'EDU',
    '62': 'HEALTHCARE',
    '71': 'ENTERTAINMENT',
    '72': 'ACCOMMODATION_FOOD',
    '81': 'OTHER_SERVICES',
    '92': 'PUBLIC_ADMIN'
}

RECESSION_PERIODS = [[date(1990, 7, 1), date(1991, 3, 1)],
                     [date(2001, 3, 1), date(2001, 11, 1)],
                     [date(2007, 12, 1), date(2009, 6, 1)]]

def data_clean_up(records):
    """Clean up NaN values and remove useless columns"""
    new_records = copy.deepcopy(records)
    #Remove useless columns
    new_records = new_records.drop(columns=['Id', 'Name', 'BalanceGross'], axis=1)
    #Clean up NaN Values
    DEFAULT_MAPPING = { 'Bank': 'Unknown',
                        'BankState': 'Unknown',
                        'NewExist': 0, 
                        'RevLineCr': 'Undefined',
                        'LowDoc': 'Undefined',
                      }
    for col in new_records:
        if col in DEFAULT_MAPPING:
            new_records[col].fillna(DEFAULT_MAPPING.get(col), inplace=True)
    new_records["LowDoc"] = new_records.apply(lambda x: 'Undefined' if str(x.LowDoc.strip()) not in ('Y', 'N') else str(x.LowDoc.strip()), axis=1)
    new_records["RevLineCr"] = new_records.apply(lambda x: {'0': 'N', '0.0': 'N', 'T': 'Y'}.get(str(x.RevLineCr.strip()), 'N') if str(x.RevLineCr.strip()) not in ('Y', 'N', 'Undefined') else str(x.RevLineCr.strip()), axis=1)
    new_records["DisbursementDate"] = new_records.apply(lambda x: x.ApprovalDate if str(x.DisbursementDate)=='nan' else x.DisbursementDate, axis=1)
    new_records.reset_index(drop=True, inplace=True)
    return new_records

def with_in_recession(row):
    # load is labeled as 'Y' for Recession if the load is active for at least a month during the Recession time frame
    try:
        raw_disbursement_date = str(row['DisbursementDate'])
        if not pd.isnull(raw_disbursement_date):
            disbursement_date = datetime.strptime(raw_disbursement_date.strip(), '%d-%b-%y').date()
            term = int(row['Term']) if not pd.isnull(row['Term']) else 0
            recession = False
            for period in RECESSION_PERIODS:
                if (disbursement_date <= period[0] and
                    period[0] + timedelta(days=30) <= disbursement_date + timedelta(days=term*30)) or \
                        (period[0] <= disbursement_date <= period[1] and
                         disbursement_date + timedelta(days=30) <= period[1]):
                    recession = True
                    break
            return 'Y' if recession else 'N'
        else:
            return 'N'
    except:
        return 'N'

def data_transformation(records):
    """Clean up dirty values, reformating number columns and create new features"""
    # existing features
    records['Zip'] = records['Zip'].apply(lambda x: str(x))
    records['NAICS'] = records['NAICS'].apply(lambda x: NAICS_TO_INDUSTRY.get(str(x)[0:2], 'NONE') if len(str(x)) >= 2 else 'NONE')
    records['FranchiseCode'] = records['FranchiseCode'].apply(lambda x: 'N' if x in ['0', '1', 0, 1] else 'Y')
    records['NewExist'] = records['NewExist'].apply(lambda x: str(x))
    records['DisbursementGross'] = records['DisbursementGross'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    records['GrAppv'] = records['GrAppv'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    records['SBA_Appv'] = records['SBA_Appv'].apply(lambda x: float(x.replace('$', '').replace(',', '').strip()))
    records['ApprovalFY'] = records.ApprovalDate.apply(lambda x: str(datetime.strptime(x.strip(), '%d-%b-%y').year))

    # new features
    records['Recession'] = records.apply(with_in_recession, axis=1)
    # records['DisGross_GrAppv'] = records['DisbursementGross'] / records['GrAppv']
    # records['DisGross_SBAAppv'] = records['DisbursementGross'] / records['SBA_Appv']
    # records['GrAppv_SBAAppv'] = records['GrAppv'] / records['SBA_Appv']

    return records

def label_encoding(records):
    """Convert Object into Categorical and do Label Encoding"""
    new_records = copy.deepcopy(records)
    for col in new_records.columns:
        if new_records[col].dtype == object:
            new_records[col] = new_records[col].astype('category')
            new_records[col+'_Cat'] = new_records[col].cat.codes
            new_records.drop(columns=[col], axis=1, inplace=True)
    return new_records

def all_preprocess_without_label_encoding(records):
    records = data_clean_up(records)
    records = data_transformation(records)
    return records

def all_preprocess_with_label_encoding(records):
    records = data_clean_up(records)
    records = data_transformation(records)
    encoded_records = label_encoding(records)
    return encoded_records