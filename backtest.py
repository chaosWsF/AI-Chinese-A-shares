import numpy as np
import pandas as pd
import os
import glob
import arrow
import csv


def get_stockpools(date, stocknum=(10, 20)):
    first = str(int(date[:4]) - 2) + date[4:]

    date0 = date[:4] + '-' + date[4:]
    date0 = arrow.get(date0)
    second = date0.shift(months=-1)
    second = second.format('YYYYMM')

    fix = 'SP_' + first + '_' + second + '_'
    stock_file_list = [fix + str(stocknum[a]) + '.txt' for a in range(len(stocknum))]

    return stock_file_list


def kill_blank(data):
    if data == ' ':
        data = np.nan
    return data


prefix = '2to19'
# prefix = '2to22'
usedIndex = 'Index1'
# usedIndex = 'Index2'
# modelParam = 'MP1'
modelParam = 'MP2'
# modelParam = 'MP3'
# modelParam = 'MP4'

predMonthList = []
predMonthList.extend(['201502', '201503', '201504', '201505', '201506',
                      '201507', '201508', '201509', '201510', '201511', '201512'])
predMonthList.extend(['201601', '201602', '201603', '201604', '201605', '201606',
                      '201607', '201608', '201609', '201610', '201611', '201612'])
predMonthList.extend(['201701', '201702', '201703', '201704', '201705', '201706',
                      '201707', '201708', '201709', '201710', '201711', '201712'])
predMonthList.extend(['201801', '201802'])

stockNum = (10, 20)

path0 = 'D:/rongshidata/experiment_data_1'
closePrice = pd.read_csv(path0 + '/close.txt', sep="\t", header=0, skiprows=1,
                         index_col=0, parse_dates=True,
                         date_parser=lambda dates: pd.datetime.strptime(dates, '%Y%m%d'))
closePrice = closePrice.dropna(axis=1, how='all')
df_index = list(closePrice.index)

for predMonth in predMonthList:
    stockPools = get_stockpools(predMonth, stocknum=stockNum)
    s = '{0}/monthly/ReportXgb_V{1}_{2}_{3}/for{4}/'
    res_dir = s.format(path0, prefix.replace('to', '_V'), usedIndex, modelParam, predMonth)
    config = res_dir + 'config' + predMonth + '.csv'
    configDates = []
    with open(config) as f:
        r = csv.reader(f)
        for row in r:
            configDates.append(tuple(row))

    line1 = ['folder', 'stock']
    lines = []
    for configDate in configDates:
        sd = ':'.join(configDate)
        line1.append(sd + '_TR')

    s1 = '{0}/monthly/combination{1}_{2}/comb{3}.txt'
    combFile = s1.format(path0, prefix, modelParam, predMonth)
    combData = pd.read_csv(combFile, sep='\t', header=None, names=['feature', 'quantile'])

    folder1 = glob.glob(path0 + '/monthly/end/' + 'feature_v*')
    for feature_v in folder1:
        folder2 = glob.glob(feature_v + '/prob/' + '*m_*_*')
        feature_v = os.path.basename(feature_v)
        for date_range in folder2:
            # folder3 = glob.glob(date_range + '/PNE*_NL_*')
            folder3 = glob.glob(date_range + '/PNE_*')
            date_range = os.path.basename(date_range)
            sentence = (combData['feature'] == feature_v) & (combData['quantile'] == date_range)
            if not sentence.any():
                continue

            for path in folder3:
                for stockPool in stockPools:
                    period_returns = [path, stockPool[17:-4]]
                    directory = path + '/' + stockPool
                    sp = pd.read_csv(directory, skiprows=1, header=None, sep='\t', prefix='S')

                    sp = sp.dropna(axis=1, how='all')
                    for probDate in range(1, len(sp.index), 2):
                        sp.drop(probDate, inplace=True)
                    sp = sp.applymap(kill_blank)
                    dateList = list(sp.S0)
                    sp.S0 = sp.S0.apply(lambda y: y[:4] + '-' + y[4:])

                    returnList = []
                    for effectDate in range(len(sp.index)):
                        spx = sp.iloc[effectDate]
                        spx = spx.dropna()
                        if len(spx) > 1:
                            datex = spx[0]
                            lsx = list(spx[1:])
                            tmp = closePrice.loc[datex, lsx]
                            i = df_index[df_index.index(tmp.index[0]) - 1]
                            j = tmp.index[-1]
                            tmp = closePrice.loc[i:j, lsx]
                            returns = np.mean((tmp.iloc[-1] - tmp.iloc[0]) / tmp.iloc[0])
                            returnList.append(returns)
                        else:
                            returnList.append(0)

                    for configDate in configDates:
                        try:
                            start = dateList.index(configDate[0])
                            end = dateList.index(configDate[1]) + 1
                            returnData = np.array(returnList[start:end]) + 1
                            se = end - start
                            result = ((np.prod(returnData) ** (12 / se)) - 1) * 100
                            period_returns.append(str(result) + '%')
                        except ValueError:
                            period_returns.append('')

                    lines.append(period_returns)

            print(predMonth, feature_v, date_range, 'Done')

    report = res_dir + 'ReportTR' + predMonth + '.csv'
    with open(report, 'w', newline='') as fw:
        w = csv.writer(fw)
        w.writerow(line1)
        w.writerows(lines)
