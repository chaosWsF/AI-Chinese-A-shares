import numpy as np
# import matplotlib.pyplot as plt
import csv
# import glob

rootPath = 'D:/rongshidata/experiment_data_1'
periodStr = 'monthly'
usedFeature = 'V2_V22'
indexCalcMethod = 'Index1'
modelParams = 'MP1'
usedAlgo = 'Xgb'

predMonthsList = []
predMonthsList.extend(['201502', '201503', '201504', '201505', '201506',
                       '201507', '201508', '201509', '201510', '201511', '201512'])
predMonthsList.extend(['201601', '201602', '201603', '201604', '201605', '201606',
                       '201607', '201608', '201609', '201610', '201611', '201612'])
predMonthsList.extend(['201701', '201702', '201703', '201704', '201705', '201706',
                       '201707', '201708', '201709', '201710', '201711', '201712'])
predMonthsList.extend(['201801', '201802'])

dropSPnum = 40

s = '{0}/{1}/ReportDF_{2}_{3}_{4}'
dfPath = s.format(rootPath, periodStr, usedFeature, indexCalcMethod, modelParams)
algoPath = dfPath.replace('DF', usedAlgo)

histData = []
for predMonth in predMonthsList:
    spDFPath = '{0}/for{1}/StockPool_{1}.txt'.format(dfPath, predMonth)
    spInitPath = '{0}/for{1}/StockPool_{1}.txt'.format(algoPath, predMonth)
    spFilterPath = spInitPath[:-4] + '_filter' + str(dropSPnum) + spInitPath[-4:]
    spDF = np.loadtxt(spDFPath, dtype=bytes).astype(str)
    spInit = np.loadtxt(spInitPath, dtype=bytes).astype(str)
    spIndexList = []
    for sp in spInit:
        index = np.where(spDF == sp)
        if not list(index[0]):
            i = len(spDF)
        else:
            i = index[0][0]

        spIndexList.append(i)

    histData.extend(spIndexList)

    spIndexArray = np.array(spIndexList)
    spIndex = np.argsort(spIndexArray)[::-1]
    spDroped = np.delete(spInit, spIndex[:dropSPnum])
    spIntegrated = np.append(spDroped, spDF[:dropSPnum])
    spIntegrated = spIntegrated.reshape((spIntegrated.shape[0], 1))
    with open(spFilterPath, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerows(spIntegrated)

# plt.hist(histData)
# plt.show()
