import os
import shutil
import arrow
import glob


def get_date_range(start, end):
    """get the date range of the used months"""
    start = start[:4] + '-' + start[4:]
    startdate = arrow.get(start)
    end = end[:4] + '-' + end[4:]
    enddate = arrow.get(end)
    return arrow.Arrow.range('month', startdate, enddate)


def get_train(date, quan_name):
    """get the file name of the training data"""
    date0 = date[:4] + '-' + date[4:]
    first = arrow.get(date0)
    quan = quan_name.split("m_")[0]
    m = -1 * int(quan)
    second = first.shift(months=-1)
    second = second.format("YYYYMM")
    first = first.shift(months=m)
    first = first.format('YYYYMM')
    ret = first + '-' + second + '_train.csv'
    return ret


def get_test(date):
    """get the file name of the test data"""
    ret = date + 'pred.csv'
    return ret

startDate = '201805'
endDate = '201805'

rootDir = 'D:/rongshidata'
# dataInfo = 'experiment_data_1'
dataInfo = 'experiment_data_2'
periodInfo = 'monthly'
usedQuantile = []
usedQuantile.extend(['6m_1_16', '6m_3_18'])
usedQuantile.extend(['12m_1_16', '12m_3_18'])
usedQuantile.extend(['3m_1_31', '3m_3_33'])
usedQuantile.extend(['24m_1_13', '24m_3_15'])
usedQuantile.extend(['36m_1_11', '36m_3_13'])

dir1st = 'D:/copy{0}_{1}'.format(startDate, endDate)
if not os.path.exists(dir1st):
    os.mkdir(dir1st)

closePriceFile = '{0}/{1}/close.txt'.format(rootDir, dataInfo)
shutil.copy(closePriceFile, dir1st)

featureDir = '{0}/{1}/{2}/end/feature_v*'.format(rootDir, dataInfo, periodInfo)
featureList = glob.glob(featureDir)
for feature in featureList:
    featureName = os.path.basename(feature)
    for Date in get_date_range(startDate, endDate):
        Date = Date.format('YYYYMM')
        testDataDir = '{0}/{1}/end/{2}/testing'.format(dir1st, periodInfo, featureName)
        if not os.path.exists(testDataDir):
            os.makedirs(testDataDir)

        testFile = feature + '/testing/' + get_test(Date)
        shutil.copy(testFile, testDataDir)

        trainDataList = glob.glob(feature + '/training/*m_*_*')
        for quantile in trainDataList:
            quantileName = os.path.basename(quantile)
            if quantileName not in usedQuantile:
                continue

            trainDataDir = '{0}/{1}/end/{2}/training/{3}'.format(dir1st, periodInfo, featureName, quantileName)
            if not os.path.exists(trainDataDir):
                os.makedirs(trainDataDir)

            trainFile = quantile + '/' + get_train(Date, quantileName)
            shutil.copy(trainFile, trainDataDir)
            print(quantile, 'DONE')
