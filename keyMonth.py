import pandas as pd
import csv
import numpy as np
import arrow
import glob
import os


def next_month(date):
    """get the name of next month"""
    date0 = arrow.get(date)
    nextdate = date0.shift(months=+1)
    nextdate = nextdate.format('YYYY-MM')
    return nextdate


def get_config_date(date):
    date0 = arrow.get(date)

    date_b1 = date0.shift(months=-1)  # 1个月前
    date_b3 = date0.shift(months=-3)  # 3个月前
    date_b4 = date0.shift(months=-4)
    date_b7 = date0.shift(months=-7)
    date_b8 = date0.shift(months=-8)
    date_b12 = date0.shift(months=-12)
    date_b6 = date0.shift(months=-6)

    date_b1 = date_b1.format('YYYYMM')
    date_b3 = date_b3.format('YYYYMM')
    date_b4 = date_b4.format('YYYYMM')
    date_b7 = date_b7.format('YYYYMM')
    date_b8 = date_b8.format('YYYYMM')
    date_b12 = date_b12.format('YYYYMM')
    date_b6 = date_b6.format('YYYYMM')

    date1 = [date_b1, date_b1]
    date2 = [date_b3, date_b1]
    date3 = [date_b7, date_b4]
    date4 = [date_b12, date_b8]
    date5 = [date_b6, date_b1]
    date6 = [date_b12, date_b1]

    totaldate = [date1, date2, date3, date4,
                 date5, date6]

    return totaldate

predMonths = []
predMonths.extend(['2015-02', '2015-03', '2015-04', '2015-05', '2015-06',
                   '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12'])
predMonths.extend(['2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06',
                   '2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12'])
predMonths.extend(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
                   '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12'])
predMonths.extend(['2018-01', '2018-02'])
# prefix = 'Index1'
# prefix = 'Index2'
prefix = 'Index3'
# prefix = 'Index4'

path0 = 'D:/rongshidata/experiment_data_1'
date_range = 2      # 在2年内寻找
# 上证50 000016.SH; 中证500 000905.SH; 创业板 399006.SZ; 沪深300 000300.SH;
# 中小板 399005.SZ; 上证综指 000001.SH; 万德全A 881001.WI; 深成指 399001.SZ;
usedIndex = ['close', '000905.SH', '000016.SH', '399006.SZ', '000300.SH',
             '399005.SZ', '000001.SH', '881001.WI', '399001.SZ']
closeData = pd.read_csv(path0 + '/close.txt', memory_map=True, delimiter="\t",
                        skiprows=1, index_col=0, usecols=usedIndex, parse_dates=True,
                        date_parser=lambda dates: pd.datetime.strptime(dates, '%Y%m%d'))
closeIndex = list(closeData.index)

for pred_month in predMonths:

    start_month = str(int(pred_month[:4]) - date_range) + pred_month[4:]
    used_date = start_month
    lines = [['月份', '上证50', '中证500', '沪深300', '创业板', '上证50/中证500', '沪深300/创业板',
              '上证50/创业板', '沪深300/中小板', '上证综指/万德全A', '深成指/中证500']]

    while used_date != pred_month:
        if used_date == '2016-01':
            used_date = next_month(used_date)
            continue

        closeThisMonth = closeData.ix[used_date]
        i = closeIndex[closeIndex.index(closeThisMonth.index[0]) - 1]
        j = closeThisMonth.index[-1]
        closeThisMonth = closeData.ix[i:j]

        returns = closeThisMonth.ix[-1] / closeThisMonth.ix[0]
        ratio = returns['000016.SH'] / returns['000905.SH']
        ratio2 = returns['000300.SH'] / returns['399006.SZ']
        ratio3 = returns['000016.SH'] / returns['399006.SZ']
        ratio4 = returns['000300.SH'] / returns['399005.SZ']
        ratio5 = returns['000001.SH'] / returns['881001.WI']
        ratio6 = returns['399001.SZ'] / returns['000905.SH']

        lines.append([used_date.replace('-', ''), str(returns['000016.SH']), str(returns['000905.SH']),
                      str(returns['000300.SH']), str(returns['399006.SZ']), str(ratio), str(ratio2),
                      str(ratio3), str(ratio4), str(ratio5), str(ratio6)])

        used_date = next_month(used_date)

    market = path0 + '/monthly/index info/market' + pred_month.replace('-', '') + '.csv'

    with open(market, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerows(lines)

    colIndex = lines[0]
    monthCol = colIndex[0]
    indexData = pd.DataFrame(lines[1:], columns=colIndex)
    converter = {monthCol: str}
    for x in range(1, len(colIndex)):
        converter.update({colIndex[x]: np.float32})

    indexData = indexData.astype(converter)
    config = {}
    if prefix[-1] in ['1', '3']:
        indexLen = 7
    else:
        indexLen = len(colIndex)

    for x in range(1, indexLen):
        usedCol = colIndex[x]
        indexRank = indexData.sort_values(by=usedCol)
        minValue = indexRank.iloc[0, x]
        minDate = indexData[indexData[usedCol] == minValue]
        minDate = minDate[monthCol].values
        minDate = minDate[0]
        minstr = usedCol + '最低'
        maxValue = indexRank.iloc[-1, x]
        maxDate = indexData[indexData[usedCol] == maxValue]
        maxDate = maxDate[monthCol].values
        maxDate = maxDate[0]
        maxstr = usedCol + '最高'
        config.update({minDate: minstr, maxDate: maxstr})

        if prefix[-1] in ['3', '4']:
            minSecondValue = indexRank.iloc[1, x]
            minSecondDate = indexData[indexData[usedCol] == minSecondValue]
            minSecondDate = minSecondDate[monthCol].values
            minSecondDate = minSecondDate[0]
            minSecondstr = usedCol + '次低'
            maxSecondValue = indexRank.iloc[-2, x]
            maxSecondDate = indexData[indexData[usedCol] == maxSecondValue]
            maxSecondDate = maxSecondDate[monthCol].values
            maxSecondDate = maxSecondDate[0]
            maxSecondstr = usedCol + '次高'
            config.update({minSecondDate: minSecondstr, maxSecondDate: maxSecondstr})

        if '/' in usedCol:
            df = indexData.copy()
            df[usedCol] = abs(df[usedCol] - 1)
            dfRank = df.sort_values(by=usedCol)
            minRatio = dfRank.iloc[0, x]
            ratioDate = dfRank[dfRank[usedCol] == minRatio]
            ratioDate = ratioDate[monthCol].values
            ratioDate = ratioDate[0]
            ratiostr = usedCol + '最接近1'
            config.update({ratioDate: ratiostr})

    # print(config)     # key months
    configDate = get_config_date(pred_month)
    for dateKey in config.keys():
        configDate.append([dateKey, dateKey])

    reportDirPaths = glob.glob('{0}/monthly/Report*_V*_V*_{1}_MP*'.format(path0, prefix))
    for reportDirPath in reportDirPaths:
        configPath = reportDirPath + '/for' + pred_month.replace('-', '')
        if not os.path.exists(configPath):
            os.makedirs(configPath)

        configFile = configPath + '/config' + pred_month.replace('-', '') + '.csv'
        with open(configFile, 'w', newline='') as cf:
            cfw = csv.writer(cf)
            cfw.writerows(configDate)

    print(pred_month, prefix, 'Done')
