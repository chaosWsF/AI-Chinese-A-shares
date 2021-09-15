import pandas as pd
import csv
import numpy as np
import glob


def gain_pools(directory, predmonth, stocknum):

    selected_model = directory + 'for' + predmonth + '\\SelectedModels' + predmonth + '.csv'
    df = pd.read_csv(selected_model)
    df = df.ix[:, 2:]
    stockpool = []

    for j in range(len(df.columns)):
        for i in range(len(df.index)):
            try:
                stockpool_name = '{0}\\SP_{1}_{1}_{2}.txt'.format(df.ix[i, j], predmonth, str(stocknum))
                with open(stockpool_name) as f:
                    next(f, None)
                    cr = csv.reader(f, delimiter='\t')
                    for row in cr:
                        stocks = row[1:]
                        stockpool.extend(stocks)
                        break
            except TypeError:
                pass

    stockpool_final = [stockpool[0]]
    for index_stock in range(1, len(stockpool)):
        if (not stockpool[index_stock] in stockpool_final) and (not stockpool[index_stock] in [' ', '']):
            stockpool_final.append(stockpool[index_stock])

    stockpool_final = np.array([stockpool_final]).T
    writing_path = directory + 'for' + predmonth + '\\' + 'StockPool_' + predmonth + '.txt'

    with open(writing_path, 'w', newline='') as fw:
        cw = csv.writer(fw, delimiter='\t')
        cw.writerows(stockpool_final)

pred_months = []
pred_months.extend(['201502', '201503', '201504', '201505', '201506',
                    '201507', '201508', '201509', '201510', '201511', '201512'])
pred_months.extend(['201601', '201602', '201603', '201604', '201605', '201606',
                    '201607', '201608', '201609', '201610', '201611', '201612'])
pred_months.extend(['201701', '201702', '201703', '201704', '201705', '201706',
                    '201707', '201708', '201709', '201710', '201711', '201712'])
pred_months.extend(['201801', '201802'])
stockNum = 500

path0 = 'D:/rongshidata/experiment_data_1'
# reportPaths = glob.glob(path0 + '/monthly/ReportXgb_V*_V*_Index*_MP*/')
reportPaths = glob.glob(path0 + '/monthly/ReportDF_V*_V*_Index*_MP*/')
for reportPath in reportPaths:
    for pred_month in pred_months:
        gain_pools(reportPath, pred_month, stocknum=stockNum)
        print(pred_month, 'has been done.')
