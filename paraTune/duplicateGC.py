import glob
import os
import csv
import pandas as pd

path0 = 'D:/rongshidata/experiment_data_1'
feature_list = glob.glob(path0 + '/monthly/end/feature_v*')
lines = [['feature', 'quantile', 'parameter', 'duplicate rate']]

factors_3 = ['feature_v2', 'feature_v3', 'feature_v4', 'feature_v5', 'feature_v6',
             'feature_v7', 'feature_v8', 'feature_v9', 'feature_v10', 'feature_v11',
             'feature_v20', 'feature_v21', 'feature_v22']
factors_4 = ['feature_v12', 'feature_v13', 'feature_v14', 'feature_v15', 'feature_v16',
             'feature_v17']
factors_5 = ['feature_v18', 'feature_v19']

for feature in feature_list:
    range_list = glob.glob(feature + '/prob/*m_*_*')
    for data_range in range_list:
        param_list = glob.glob(data_range + '/deepforest/p*')
        for param in param_list:
            target_list = glob.glob(param + '/*prob.csv')
            for target in target_list:
                feature = os.path.basename(feature)
                data_range = os.path.basename(data_range)
                param = os.path.basename(param)
                file = os.path.basename(target)

                df = pd.read_csv(target, usecols=['yield_class'])
                duplicate_rate = 1 - len(df.drop_duplicates()) / len(df)

                if feature in factors_3:
                    featureClass = 'factor3'
                elif feature in factors_4:
                    featureClass = 'factor4'
                else:
                    featureClass = 'factor5'

                if data_range[0] == '6':
                    quanClass = '6m'
                elif data_range[0] == '1':
                    quanClass = '12m'
                elif data_range[0] == '3':
                    quanClass = '3m'
                else:
                    quanClass = '24m'

                line = [featureClass, quanClass, param, str(duplicate_rate)]
                lines.append(line)

        print(feature, data_range, 'has been done.')

with open(path0 + '/EvalReport_GC.csv', 'w', newline='') as record:
    cw = csv.writer(record)
    cw.writerows(lines)
