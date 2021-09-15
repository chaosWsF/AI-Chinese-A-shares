import pandas as pd
import csv
import glob


def advanced_eval(models, predmonth, stock_num=10):

    lines = []
    target = 'NeoReportTR' + predmonth + '.csv'
    df = pd.read_csv(models + target)
    df.ix[:, 2:] = df.ix[:, 2:].applymap(lambda x: float(x[:-1]) / 100)

    period1 = df.ix[:, [0, 3]]  # first 3 months
    period2 = df.ix[:, [0, 4]]  # second 4 months
    period3 = df.ix[:, [0, 5]]  # 5 months
    period1 = period1.sort_values(period1.columns[1], ascending=False)
    period2 = period2.sort_values(period2.columns[1], ascending=False)
    period3 = period3.sort_values(period3.columns[1], ascending=False)

    length1 = len(period1.index)
    length2 = len(period2.index)
    length3 = len(period3.index)

    header = ['PredMonth', 'StockNum'] + list(df.columns[8:])
    line = [predmonth, str(stock_num)]

    for key in range(8, len(df.columns)):
        key_month = df.ix[:, [0, key]]
        key_month = key_month.sort_values(key_month.columns[1], ascending=False)

        for i in range(1, 101):
            j1 = int((i / 100) * length1)
            j2 = int((i / 100) * length2)
            j3 = int((i / 100) * length3)

            if i <= 5:
                j = int((i / 100) * len(key_month.index))
            else:
                j = int(0.05 * len(key_month.index))

            high_rank = key_month.head(j)['folder']
            high_rank1 = period1.head(j1)['folder']
            high_rank2 = period2.head(j2)['folder']
            high_rank3 = period3.head(j3)['folder']

            intersection = set(high_rank) & set(high_rank1) & set(high_rank2) & set(high_rank3)

            if intersection != set():
                line.append(list(intersection)[0])
                break

    lines.append(line)

    result = models + 'SelectedModels' + predmonth + '.csv'
    with open(result, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(header)
        cw.writerows(lines)

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
        path = reportPath + 'for' + pred_month + '\\'
        advanced_eval(path, pred_month)
        print(pred_month, 'has been done.')
