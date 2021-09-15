import pandas as pd
import numpy as np
import os
import glob


def replace_nan(data):
    if data == 'nan%':
        data = '0%'

    return data


def model_eval(models):

    df = pd.read_csv(models)
    df = df.applymap(replace_nan)
    df.ix[:, 2:] = df.ix[:, 2:].applymap(lambda x: float(x[:-1]) / 100)
    result = pd.DataFrame({}, index=df.index, columns=df.columns)

    for k in range(0, len(df.index), 2):
        block = df.ix[k:k + 1, 7]
        delta = (block.ix[k + 1] - block.ix[k]) / abs(block.ix[k])

        if delta == np.inf:  # avoid zero division
            delta = block.ix[k + 1] - block.ix[k]

        if delta <= 0.01:
            result.ix[k] = df.ix[k]

    result = result.dropna()
    result.ix[:, 2:] = result.ix[:, 2:].applymap(lambda x: str(x * 100) + '%')
    return result

pred_months = []
pred_months.extend(['201502', '201503', '201504', '201505', '201506',
                    '201507', '201508', '201509', '201510', '201511', '201512'])
pred_months.extend(['201601', '201602', '201603', '201604', '201605', '201606',
                    '201607', '201608', '201609', '201610', '201611', '201612'])
pred_months.extend(['201701', '201702', '201703', '201704', '201705', '201706',
                    '201707', '201708', '201709', '201710', '201711', '201712'])
pred_months.extend(['201801', '201802'])

path0 = 'D:/rongshidata/experiment_data_1'
reportPaths = glob.glob(path0 + '/monthly/ReportXgb_V*_V*_Index*_MP*/')
for reportPath in reportPaths:
    for pred_month in pred_months:
        name = reportPath + 'for' + pred_month + '\\ReportTR' + pred_month + '.csv'
        path_to_write = reportPath + 'for' + pred_month + '\\Neo' + os.path.basename(name)
        modified_file = model_eval(name)
        modified_file.to_csv(path_to_write, index=False)
        print(pred_month, 'has been done.')
