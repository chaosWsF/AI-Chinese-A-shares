from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   # Additional scklearn functions
import xlrd
import csv
import os
import sys


# all month is indicated in six numbers(four numbers of year and two number of day)
# like '201603','201711'
# get the name of last month
def last_month(date):
    date = str(date)
    year = date[:4]
    #print year;
    month = date[4:]
    if month=='10':
        month = '09'
    elif month == '01':
        year = str(int(year)-1)
        month = '12'
    else:
        month = int(month)-1
        if month<10:
            month = '0'+str(month)
        else:
            month = str(month)
    ret = year+month
    return ret


# get the name of next month
def next_month(date):
    date = str(date)
    year = date[:4]
    #print year;
    month = date[4:]
    #print month
    if month=='09':
        month = '10'
    elif month == '12':
        year = str(int(year)+1)
        month = '01'
    else:
        month = int(month)+1
        if month<10:
            month = '0'+str(month)
        else:
            month = str(month)
    ret = year+month
    return ret


# get the file name of the training data for 24 months
def get_train3(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    first = str(int(year)-2)+month
    second = last_month(date)
    ret = first+'-'+second+'_train.csv'
    return ret


# get the file name of the training data for 12 months
def get_train(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    first = str(int(year)-1)+month
    second = last_month(date)
    ret = first+'-'+second+'_train.csv'
    return ret


# get the file name of the training data for 3 months
def get_train4(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    a = int(month)-3
    if a<1:
        first = str(int(year)-1)
        if a+12>=10:
            first = first+str(a+12)
        else:
            first = first+'0'+str(a+12)
    else:
        first = year+'0'+str(a)
    second = last_month(date)
    ret = first+'-'+second+'_train.csv'
    return ret


# get the file name of the training data for 6 months
def get_train2(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    a = int(month)-6
    if a<1:
        first = str(int(year)-1)
        if a+12>=10:
            first = first+str(a+12)
        else:
            first = first+'0'+str(a+12)
    else:
        first = year+'0'+str(a)
    second = last_month(date)
    ret = first+'-'+second+'_train.csv'
    return ret


# get the file name of the test data
def get_test(date):
    date = str(date)
    ret = date+'pred.csv'
    return ret


# get the file name of the probability
def get_prob(date):
    date = str(date)
    ret = date+'prob.csv'
    return ret


# fit the data and write out
def modelfit(alg, ids,dtrain, dtest, predictors, writer, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):   
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    #print(dtrain)
    alg.fit(dtrain[predictors], dtrain['yield_class'],eval_metric='auc')
    
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['yield_class'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['yield_class'], dtrain_predprob))
    #predictors = ['mkt_freeshares_rank', 'mmt_rank', 'roa_growth_rank']
    #list1 = alg.predict(dtest[predictors])    
    #dtest['yield_class'] = alg.predict(dtest[predictors])
    #dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    #dtest.append('predprob')
    dtest['yield_class'] = alg.predict_proba(dtest[predictors])[:,1]
    #print (dtest)
   
    pp = [x for x in dtest.columns]
    pp.insert(0,'id')
    #print (pp)
    writer.writerow(pp)

    pp1 = [x for x in dtest.columns]
    for i in range(len(dtest_predprob)):
        tmp = []
        tmp.append(ids[i])
        for j in pp1:
            tmp.append(dtest[j][i])
        writer.writerow(tmp)
    print("\n")

factors_3 = ['feature_v2', 'feature_v3', 'feature_v4', 'feature_v5', 'feature_v6',
             'feature_v7', 'feature_v8', 'feature_v9', 'feature_v10', 'feature_v11',
             'feature_v20', 'feature_v21', 'feature_v22']
factors_4 = ['feature_v12', 'feature_v13', 'feature_v14', 'feature_v15', 'feature_v16',
             'feature_v17']
factors_5 = ['feature_v18', 'feature_v19']
start_date = '201802'
end_date = '201803'
path = 'D:/rongshidata/experiment_data_1/monthly/end'

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    training_path = os.path.join(file_path, 'training')
    prob_path = os.path.join(file_path, 'prob')
    # prob_path = os.path.join(file_path, 'probXGB')
    for train_file1 in os.listdir(training_path):
        # print(file, train_file1)
        if train_file1 == '6m_1_16':
            if file not in ['feature_v2', 'feature_v3', 'feature_v5', 'feature_v6', 'feature_v7', 'feature_v8', 'feature_v10', 'feature_v11',
                            'feature_v12', 'feature_v13', 'feature_v16', 'feature_v17', 'feature_v18', 'feature_v19']:
                continue
        elif train_file1 == '6m_3_18':
            if file not in ['feature_v2', 'feature_v3', 'feature_v5', 'feature_v6', 'feature_v8', 'feature_v9', 'feature_v11',
                            'feature_v16', 'feature_v17', 'feature_v18']:
                continue
        # elif train_file1 == '12m_1_16':
        #     if file not in ['feature_v2', 'feature_v4', 'feature_v15']:
        #         continue
        # elif train_file1 == '12m_3_18':
        #     if file not in ['feature_v2', 'feature_v4', 'feature_v13']:
        #         continue
        else:
            continue

        if (file in factors_4) or (file in factors_5):
            if train_file1[0] == '6':  # 6 months
                estimator_list = [85, 86, 88]
            elif train_file1[0] == '1':  # 12 months
                estimator_list = [58, 59, 60]
            else:
                estimator_list = [25, 26, 27, 28]
        else:  # 3-factors
            if train_file1[0] == '6':  # 6 months
                estimator_list = [92, 93, 97]
            elif train_file1[0] == '1':  # 12 months
                estimator_list = [70, 72, 75]
            else:
                estimator_list = [40, 41, 44, 45]

        tra_path = os.path.join(training_path, train_file1)
        prob_path2 = os.path.join(prob_path, train_file1)
        if not os.path.exists(prob_path2):
            os.makedirs(prob_path2)

        for xx in estimator_list:
            date = start_date
            while date != end_date:
                if train_file1[0] == '6':  # 6 months
                    train_file = os.path.join(tra_path, get_train2(date))
                elif train_file1[0] == '1':  # 12 months
                    train_file = os.path.join(tra_path, get_train(date))
                elif train_file1[0] == '2':  # 24 months
                    train_file = os.path.join(tra_path, get_train3(date))
                else:  # 3 months
                    train_file = os.path.join(tra_path, get_train4(date))
                test_path = os.path.join(file_path, 'testing')
                test_file = os.path.join(test_path, get_test(date))
                print(train_file)
                print(test_file)
                train = pd.read_csv(train_file)
                train = train._get_numeric_data()
                numeric_headers = list(train.columns.values)
                train = train.as_matrix()
                train = np.nan_to_num(train)
                train = pd.DataFrame(train)
                train.columns = numeric_headers

                test = pd.read_csv(test_file)
                IDs = test['id']
                # print(IDs)
                test = test._get_numeric_data()
                # print(test)
                numeric_headers1 = list(test.columns.values)
                test = test.as_matrix()
                test = np.nan_to_num(test)
                test = pd.DataFrame(test)
                test.columns = numeric_headers1

                target = 'yield_class'
                target2 = 'predprob'
                IDcol = 'id'
                predictors = [x for x in train.columns if x not in [target, target2, IDcol]]
                # predictors2 = [x for x in test.columns if x not in [target, target2, IDcol]]

                estimator = xx

                print("search for the local best")
                param_test1 = {'max_depth': range(3, 10, 1), 'min_child_weight': range(1, 6, 1)}
                param_test2 = {'gamma': [i / 10.0 for i in range(0, 5)]}
                param_test3 = {'subsample': [i / 10.0 for i in range(6, 10)], 'colsample_bytree': [i / 10.0 for i in range(6, 10)]}

                xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=6,
                                     min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                     objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

                clf = GridSearchCV(xgb1, param_test1, cv=5, scoring='precision_macro')
                # clf.fit(X_train, y_train)
                clf.fit(train[predictors], train[target])
                print(clf.best_params_)

                max_dep = clf.best_params_['max_depth']
                child = clf.best_params_['min_child_weight']

                print("end of search one!!")

                xgb2 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=max_dep,
                                     min_child_weight=child, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                     objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

                clf2 = GridSearchCV(xgb2, param_test2, cv=5, scoring='precision_macro')

                clf2.fit(train[predictors], train[target])
                print(clf2.best_params_)
                gam = clf2.best_params_['gamma']
                print("end of search two!!")

                xgb3 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=max_dep,
                                     min_child_weight=child, gamma=gam, subsample=0.8, colsample_bytree=0.8,
                                     objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

                clf3 = GridSearchCV(xgb3, param_test3, cv=5, scoring='precision_macro')
                clf3.fit(train[predictors], train[target])
                print(clf3.best_params_)
                print("end of search three!!")

                sub = clf3.best_params_['subsample']
                bytree = clf3.best_params_['colsample_bytree']
                xgb4 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=max_dep,
                                     min_child_weight=child, gamma=gam, subsample=sub, colsample_bytree=bytree,
                                     objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

                prob_path1 = prob_path2 + '/PNE_' + str(estimator)
                if not os.path.isdir(prob_path1):
                    os.makedirs(prob_path1)

                write_file = prob_path1 + '/' + get_prob(date)
                print(write_file)

                csvfile = open(write_file, "w", newline='')
                writer = csv.writer(csvfile)
                modelfit(xgb4, IDs, train, test, predictors, writer)
                csvfile.close()
                date = next_month(date)
