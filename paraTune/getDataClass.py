import glob
import os
import csv

path0 = 'D:/rongshidata/experiment_data_1'
feature_list = glob.glob(path0 + '/monthly/end/feature_v*')
lines = [['feature', 'quantile']]

for feature in feature_list:
    range_list = glob.glob(feature + '/training/*m_*_*')
    for data_range in range_list:
        feature = os.path.basename(feature)
        data_range = os.path.basename(data_range)
        line = [feature, data_range]
        lines.append(line)


with open('parameter.csv', 'w', newline='') as record:
    cw = csv.writer(record)
    cw.writerows(lines)
