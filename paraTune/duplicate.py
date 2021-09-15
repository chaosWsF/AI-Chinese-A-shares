import glob
import os
import csv
import pandas as pd

main_directory = 'D:/rongshidata/experiment_data_1/monthly/end/'
# main_directory = 'D:/rongshidata/data_test/monthly/end/'
feature_list = glob.glob(main_directory + 'feature_v*')
# lines = [['feature number', 'fractile', 'NE, NL', 'date', 'duplicate rate']]
lines = [['feature number', 'fractile', 'NE', 'date', 'duplicate rate']]

for feature in feature_list:
    # range_list = glob.glob(feature + '/prob/*_*_*')
    range_list = glob.glob(feature + '/probXGB/*_*_*')
    for data_range in range_list:
        # param_list = glob.glob(data_range + '/PNE*_NL_*')
        param_list = glob.glob(data_range + '/PNE_*')
        for param in param_list:
            target_list = glob.glob(param + '/*prob.csv')
            for target in target_list:
                feature = os.path.basename(feature)
                data_range = os.path.basename(data_range)
                param = os.path.basename(param)
                file = os.path.basename(target)

                df = pd.read_csv(target, usecols=['yield_class'])
                # uniques = df.drop_duplicates()
                duplicate_rate = 1 - len(df.drop_duplicates()) / len(df)

                line = [feature, data_range, param[1:], file[:-8], str(duplicate_rate)]
                lines.append(line)

        print(feature, data_range, 'has been done.')

with open(main_directory + 'recordXGB.csv', 'w', newline='') as f:
    cw = csv.writer(f)
    cw.writerows(lines)
