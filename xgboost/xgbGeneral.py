import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import csv
import os
import arrow


# all month is indicated in six numbers(four numbers of year and two number of day)
# like '201603','201711'
def last_month(date):
    """get the name of month"""
    date = str(date)
    date = date[:4] + '-' + date[4:]
    date0 = arrow.get(date)
    lastmon = date0.shift(months=-1)
    lastmon = lastmon.format('YYYYMM')
    return lastmon


def next_month(date):
    """get the name of next month"""
    date = str(date)
    date = date[:4] + '-' + date[4:]
    date0 = arrow.get(date)
    nextmon = date0.shift(months=1)
    nextmon = nextmon.format('YYYYMM')
    return nextmon


def get_train(date, quantile):
    """get the file name of the training data"""
    date = str(date)
    date0 = date[:4] + '-' + date[4:]

    first = arrow.get(date0)
    quan = quantile.split('m_')[0]
    m = -1 * int(quan)

    second = first.shift(months=-1)
    second = second.format("YYYYMM")

    first = first.shift(months=m)
    first = first.format('YYYYMM')

    ret = first + '-' + second + '_train.csv'

    return ret


def get_test(date):
    """get the file name of the test data"""
    date = str(date)
    ret = date + 'pred.csv'
    return ret


def get_prob(date):
    """get the file name of the probability"""
    date = str(date)
    ret = date + 'prob.csv'
    return ret


# fit the data and write out
def modelfit(alg, ids, dtrain, dtest, predictors, writer, useTrainCV=True,
             cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    # print(dtrain)
    alg.fit(dtrain[predictors], dtrain['yield_class'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['yield_class'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['yield_class'], dtrain_predprob))
    # predictors = ['mkt_freeshares_rank', 'mmt_rank', 'roa_growth_rank']
    # list1 = alg.predict(dtest[predictors])
    # dtest['yield_class'] = alg.predict(dtest[predictors])
    # dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]
    # dtest.append('predprob')
    dtest['yield_class'] = alg.predict_proba(dtest[predictors])[:, 1]
    # print (dtest)

    pp = [x for x in dtest.columns]
    pp.insert(0, 'id')
    # print (pp)
    writer.writerow(pp)

    pp1 = [x for x in dtest.columns]
    for i in range(len(dtest_predprob)):
        tmp = []
        tmp.append(ids[i])
        for j in pp1:
            tmp.append(dtest[j][i])
        writer.writerow(tmp)
    print("\n")

# TODO
start_date = '201804'
end_date = '201805'
# TODO
prefix = '2to19'
# prefix = '2to22'
mp = 'MP1'
# mp = 'MP2'
# mp = 'MP3'
# mp = 'MP4'

factors_3 = ['feature_v2', 'feature_v3', 'feature_v4', 'feature_v5', 'feature_v6',
             'feature_v7', 'feature_v8', 'feature_v9', 'feature_v10', 'feature_v11',
             'feature_v20', 'feature_v21', 'feature_v22']
factors_4 = ['feature_v12', 'feature_v13', 'feature_v14', 'feature_v15',
             'feature_v16', 'feature_v17']
factors_5 = ['feature_v18', 'feature_v19']

usedQuantile = ['6m_1_16', '6m_3_18', '3m_1_31', '3m_3_33', '12m_1_16', '12m_3_18']

# TODO
# path0 = 'D:/rongshidata/experiment_data_1'
path0 = 'D:/rongshidata/experiment_data_2'
combFile = '{0}/monthly/combination{1}_{2}/comb{3}.txt'.format(path0, prefix, mp, start_date)
# TODO
combData = pd.read_csv(combFile, sep='\t', header=None, names=['feature', 'quantile'])

path = path0 + '/monthly/end'
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    training_path = os.path.join(file_path, 'training')
    prob_path = os.path.join(file_path, 'prob')
    for train_file1 in os.listdir(training_path):
        # TODO
        sentence = (combData['feature'] == file) & (combData['quantile'] == train_file1)
        if not sentence.any():
            continue

        if train_file1 not in usedQuantile:
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

        for estimator in estimator_list:
            used_date = start_date
            while used_date != end_date:
                train_file = os.path.join(tra_path, get_train(used_date, quantile=train_file1))
                test_file = os.path.join(file_path, 'testing', get_test(used_date))
                prob_path1 = prob_path2 + '/PNE_' + str(estimator)
                if not os.path.isdir(prob_path1):
                    os.makedirs(prob_path1)

                write_file = prob_path1 + '/' + get_prob(used_date)
                if os.path.exists(write_file):
                    print('skip', write_file)
                    used_date = next_month(used_date)
                    continue

                print(train_file)
                print(test_file)
                print(write_file)

                train = pd.read_csv(train_file)
                train = train._get_numeric_data()
                numeric_headers = list(train.columns.values)
                train = train.as_matrix()
                train = np.nan_to_num(train)
                train = pd.DataFrame(train)
                train.columns = numeric_headers

                test = pd.read_csv(test_file)
                IDs = test['id']
                test = test._get_numeric_data()
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

                print("search for the local best")
                param_test1 = {'max_depth': range(3, 10, 1), 'min_child_weight': range(1, 6, 1)}
                param_test2 = {'gamma': [i / 10.0 for i in range(0, 5)]}
                param_test3 = {'subsample': [i / 10.0 for i in range(6, 10)],
                               'colsample_bytree': [i / 10.0 for i in range(6, 10)]}

                xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=estimator,
                                     max_depth=6, min_child_weight=2, gamma=0, subsample=0.8,
                                     colsample_bytree=0.8, objective='binary:logistic',
                                     nthread=4, scale_pos_weight=1, seed=27)

                clf = GridSearchCV(xgb1, param_test1, cv=5, scoring='precision_macro')
                # clf.fit(X_train, y_train)
                clf.fit(train[predictors], train[target])
                print(clf.best_params_)

                max_dep = clf.best_params_['max_depth']
                child = clf.best_params_['min_child_weight']
                print("end of search one!!")

                xgb2 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=max_dep,
                                     min_child_weight=child, gamma=0, subsample=0.8,
                                     colsample_bytree=0.8, objective='binary:logistic',
                                     nthread=4, scale_pos_weight=1, seed=27)

                clf2 = GridSearchCV(xgb2, param_test2, cv=5, scoring='precision_macro')

                clf2.fit(train[predictors], train[target])
                print(clf2.best_params_)
                gam = clf2.best_params_['gamma']
                print("end of search two!!")

                xgb3 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=max_dep,
                                     min_child_weight=child, gamma=gam, subsample=0.8,
                                     colsample_bytree=0.8, objective='binary:logistic',
                                     nthread=4, scale_pos_weight=1, seed=27)

                clf3 = GridSearchCV(xgb3, param_test3, cv=5, scoring='precision_macro')
                clf3.fit(train[predictors], train[target])
                print(clf3.best_params_)
                print("end of search three!!")

                sub = clf3.best_params_['subsample']
                bytree = clf3.best_params_['colsample_bytree']
                xgb4 = XGBClassifier(learning_rate=0.1, n_estimators=estimator, max_depth=max_dep,
                                     min_child_weight=child, gamma=gam, subsample=sub,
                                     colsample_bytree=bytree, objective='binary:logistic',
                                     nthread=4, scale_pos_weight=1, seed=27)

                csvfile = open(write_file, "w", newline='')
                writer = csv.writer(csvfile)
                modelfit(xgb4, IDs, train, test, predictors, writer)
                csvfile.close()
                used_date = next_month(used_date)
