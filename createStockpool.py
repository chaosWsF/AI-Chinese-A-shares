# coding: utf-8
import csv
from dateutil.relativedelta import relativedelta
from dateutil import parser
import arrow


def createStockPool(selectedStockNum, strtPeriodStr, endPeriodStr,
                    txtFilePrefix, probFileFolder):

    startDate = parser.parse(strtPeriodStr + '01').date()
    endDate = parser.parse(endPeriodStr + '01').date()

    # get date difference between start and end dates
    rd = relativedelta(startDate, endDate)

    # calculate period (month in this case, but can be changed in the future, such as week or bi-week) difference between start and end dates
    periodNumDiff = abs(rd.years) * 12 + abs(rd.months) + 1

    # wopen file for writing txt file
    writeTxtFile = open(probFileFolder + txtFilePrefix + strtPeriodStr + '_' + endPeriodStr + '_' + str(selectedStockNum) + '.txt', "w")

    # write txt file row \t column number as header
    writeTxtFile.write(str(periodNumDiff) + '\t' + str(selectedStockNum))

    writeTxtFile.write('\n')

    while startDate <= endDate:
        stockProbList = []
        periodName = str(startDate.year) + str(startDate.month).zfill(2)
        readPeriod = periodName + 'prob.csv'

        # csvfile = file(probFileFolder + readPeriod, 'rb')
        csvfile = open(probFileFolder + readPeriod, encoding='utf-8')
        reader = csv.reader(csvfile)
        startDate = startDate + relativedelta(months=1)

        for line in reader:
            # create an emtpy inner list each time
            oneProbList = []

            # set the stock id as the first item of the inner list
            oneProbList.append(line[0][0:9])

            # set the stock id as the second item of the inner list
            oneProbList.append(line[-1])

            # append the inner list into the stockProbList
            stockProbList.append(oneProbList)

        csvfile.close()

        stockProbList = stockProbList[1:]

        # sort the probability of stocks
        stockProbList.sort(key=lambda x: x[1], reverse=True)

        # write probability value of each above-corresponding stocks selected

        tempProbList = []
        for i in range(0, wholeSelectedStockNumber):
            # writeTxtFile.write(stockProbList[i][1] + '\t')
            tempProbList.append(stockProbList[i][1])

        indices = [i for i, x in enumerate(tempProbList) if x == stockProbList[selectedStockNum-1][1]]
        startIndex = indices[0]
        endIndex = indices[len(indices)-1]

        if endIndex == selectedStockNum-1:
            currentSelectedStockNum = endIndex + 1
        else:
            currentSelectedStockNum = startIndex

        # write one stock number as one cell in the txt file
        writeTxtFile.write(periodName + '\t')

        for i in range(0, currentSelectedStockNum):
            writeTxtFile.write(stockProbList[i][0] + '\t')
        for j in range(currentSelectedStockNum, wholeSelectedStockNumber):
            writeTxtFile.write(' ' + '\t')

        writeTxtFile.write('\n')

        writeTxtFile.write(periodName + '_p' + '\t')
        for i in range(0, currentSelectedStockNum):
            writeTxtFile.write(stockProbList[i][1] + '\t')
        for j in range(currentSelectedStockNum, wholeSelectedStockNumber):
            writeTxtFile.write(' ' + '\t')

        # write a new line as the next period
        writeTxtFile.write('\n')

    # close the output txt file and end the process
    writeTxtFile.close()

baseFolderStr = 'D:/rongshidata/experiment_data_1/'
periodTypeStr = 'monthly/'
samplePointTypeStr = 'end/'
probFileSubFolderStr = 'prob/'
txtFilePrefixStr = 'SP_'

predictMonthsList = []
predictMonthsList.extend(['201502', '201503', '201504', '201505', '201506',
                          '201507', '201508', '201509', '201510', '201511', '201512'])
predictMonthsList.extend(['201601', '201602', '201603', '201604', '201605', '201606',
                          '201607', '201608', '201609', '201610', '201611', '201612'])
predictMonthsList.extend(['201701', '201702', '201703', '201704', '201705', '201706',
                          '201707', '201708', '201709', '201710', '201711', '201712'])
predictMonthsList.extend(['201801', '201802'])

featureVerList = ['feature_v' + str(n) for n in range(2, 23)]

trainPeriodAndSampleTypeStrList = ['6m_1_16', '6m_3_18']
# trainPeriodAndSampleTypeStrList = ['6m_1_16', '6m_3_18', '12m_1_16', '12m_3_18',
#                                    '3m_1_31', '3m_3_33']

paramNumStrList = []
# paramNumStrList.extend(['NE_58', 'NE_59', 'NE_60'])
# paramNumStrList.extend(['NE_85', 'NE_86', 'NE_88'])
# paramNumStrList.extend(['NE_70', 'NE_72', 'NE_75'])
# paramNumStrList.extend(['NE_92', 'NE_93', 'NE_97'])
# paramNumStrList.extend(['NE_25', 'NE_26', 'NE_27', 'NE_28'])
# paramNumStrList.extend(['NE_40', 'NE_41', 'NE_44', 'NE_45'])
# paramNumStrList.extend(['NE55_NL_50', 'NE55_NL_90'])
# paramNumStrList.extend(['NE60_NL_50', 'NE60_NL_90'])
# paramNumStrList.extend(['NE65_NL_50', 'NE65_NL_90'])
# paramNumStrList.extend(['NE70_NL_50', 'NE70_NL_90'])
# paramNumStrList.extend(['NE75_NL_50', 'NE75_NL_90'])
# paramNumStrList.extend(['NE40_NL_50', 'NE40_NL_90'])
# paramNumStrList.extend(['NE43_NL_50', 'NE43_NL_90'])
# paramNumStrList.extend(['NE48_NL_50', 'NE48_NL_90'])
# paramNumStrList.extend(['NE53_NL_50', 'NE53_NL_90'])
paramNumStrList.extend(['0', '1', '2'])

wholeSelectedStockNumber = 1500
selectedStockNumberList = [500, 1000]
# wholeSelectedStockNumber = 32
# selectedStockNumberList = [10, 20]

startPeriodStrList = []
endPeriodStrList = []
for predictMonth in predictMonthsList:
    predictMonth0 = predictMonth[:4] + '-' + predictMonth[4:]
    predictMonth0 = arrow.get(predictMonth0)
    first = str(int(predictMonth[:4]) - 2) + predictMonth[4:]
    second = predictMonth0.shift(months=-1)
    second = second.format('YYYYMM')
    startPeriodStrList.extend([first, predictMonth])
    endPeriodStrList.extend([second, predictMonth])

periodNumber = len(startPeriodStrList)
for featureVer in featureVerList:
    for trainPeriodAndSampleTypeStr in trainPeriodAndSampleTypeStrList:
        for paramNumStr in paramNumStrList:
            featureDir = baseFolderStr + periodTypeStr + samplePointTypeStr + featureVer
            paramDir = trainPeriodAndSampleTypeStr + '/P' + paramNumStr
            probFileFolderStr = featureDir + '/' + probFileSubFolderStr + paramDir + '/'
            for selectedStockNumber in selectedStockNumberList:
                for periodIndex in range(periodNumber):
                    try:
                        createStockPool(selectedStockNumber, startPeriodStrList[periodIndex],
                                        endPeriodStrList[periodIndex], txtFilePrefixStr, probFileFolderStr)
                    except IOError:
                        print('Error occurred when reading or writing stock pool data between: ',
                              startPeriodStrList[periodIndex] + ' - ' + endPeriodStrList[periodIndex],
                              ' in', probFileFolderStr)

            print('Successfully wrote stock pool data files in : ' + probFileFolderStr)
