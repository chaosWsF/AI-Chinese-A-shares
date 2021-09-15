import glob
import os
import numpy as np
import csv

startMonth = '201701'
endMonth = '201802'

# strategyNum = 'Set1'
# strategyNum = 'Set2'
# strategyNum = 'Set3'
# strategyNum = 'Set4'
strategyNum = 'Set5'

if strategyNum[3:] == '2':
    dropNum = 5
elif strategyNum[3:] == '3':
    dropNum = 10
elif strategyNum[3:] == '4':
    dropNum = 20
elif strategyNum[3:] == '5':
    dropNum = 40

path0 = 'D:/rongshidata/experiment_data_1/monthly'
# reportList = glob.glob(path0 + '/ReportXgb_V*_V*_Index*_MP*')
reportList = glob.glob(path0 + '/ReportXgb_V2_V22_Index1_MP1')
for report in reportList:
    dateList = glob.glob(report + '/for*')
    dateNameList = [os.path.basename(z)[-6:] for z in dateList]
    i = dateNameList.index(startMonth)
    j = dateNameList.index(endMonth) + 1
    dateRange = startMonth + '-' + endMonth

    stockpools = []
    for x in range(i, j):
        if strategyNum == 'Set1':
            stockpoolPath = '{0}/StockPool_{1}.txt'.format(dateList[x], dateNameList[x])
        else:
            stockpoolPath = '{0}/StockPool_{1}_filter{2}.txt'.format(dateList[x], dateNameList[x], dropNum)

        stockpool = np.loadtxt(stockpoolPath, dtype=bytes).astype(str)
        stockpool = list(stockpool)
        stockpool.insert(0, dateNameList[x])
        stockpools.append(stockpool)

    stockpools.insert(0, [str(j - i), '100'])

    reportName = os.path.basename(report)
    filePath = '{0}/SP{1}_{2}_{3}.txt'.format(path0, reportName[9:], strategyNum, dateRange)
    with open(filePath, 'w', newline='') as f:
        fw = csv.writer(f, delimiter='\t')
        fw.writerows(stockpools)
