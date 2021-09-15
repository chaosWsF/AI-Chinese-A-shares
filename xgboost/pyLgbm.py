import pandas as pd
import lightgbm as lgb
import os
import glob
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def modelfit(alg, dtrain, dtest, cv_folds=5, early_stopping_rounds=50):

    lgbm_param = alg.get_params()
    estr = lgbm_param.pop('n_estimators')
    lgbm_param.pop('silent')
    lgbmtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values)

    lgb.cv(lgbm_param, lgbmtrain, num_boost_round=estr, nfold=cv_folds, metrics='auc',
           early_stopping_rounds=early_stopping_rounds)
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print('\n')

    dtest[target] = alg.predict_proba(dtest[predictors])[:, 1]
    dtest.to_csv(write_file, index=False)


# all month is indicated in six numbers(four numbers of year and two number of day)
# like '201603','201711'
# get the name of last month
def last_month(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    if month == '10':
        month = '09'
    elif month == '01':
        year = str(int(year)-1)
        month = '12'
    else:
        month = int(month)-1
        if month < 10:
            month = '0' + str(month)
        else:
            month = str(month)
    ret = year + month
    return ret


# get the name of next month
def next_month(date):
    date = str(date)
    year = date[:4]
    month = date[4:]

    if month == '09':
        month = '10'
    elif month == '12':
        year = str(int(year)+1)
        month = '01'
    else:
        month = int(month) + 1

        if month < 10:
            month = '0' + str(month)
        else:
            month = str(month)

    ret = year + month

    return ret


# get the file name of the training data for 24 months
def get_train3(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    first = str(int(year)-2) + month
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


# get the file name of the training data for 12 months
def get_train(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    first = str(int(year)-1) + month
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


# get the file name of the training data for 3 months
def get_train4(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    a = int(month) - 3
    if a < 1:
        first = str(int(year)-1)
        if a + 12 >= 10:
            first = first + str(a + 12)
        else:
            first = first + '0' +str(a + 12)
    else:
        first = year + '0' + str(a)
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


# get the file name of the training data for 6 months
def get_train2(date):
    date = str(date)
    year = date[:4]
    month = date[4:]
    a = int(month) - 6
    if a < 1:
        first = str(int(year)-1)
        if a + 12 >= 10:
            first = first + str(a + 12)
        else:
            first = first + '0' + str(a + 12)
    else:
        first = year + '0' + str(a)
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


# get the file name of the test data
def get_test(date):
    date = str(date)
    ret = date + 'pred.csv'
    return ret


# get the file name of the probability
def get_prob(date):
    date = str(date)
    ret = date + 'prob.csv'
    return ret

# change following sentences to get the date you want

# start_date = '201501'
# end_date = '201710'

# Notes about parameters:
# for 4-factors-data,
#   12m 's range NE = [43, 48, 53]
#   3m 's range NE = [40, 43, 48]
#   6m 's range NE = [43, 48, 53]
# for 3-factors-data,
#   12m 's range NE = [55, 60, 65]
#   3m 's range NE = [65, 70, 75]
#   6m 's range NE = [65, 70, 75]
# for all, NL in range(50, 91), namely NL = [50, 90]

# estimator_start = 36
# estimator_end = 47
# nl_start = 50
# nl_end = 91

# stock_pool_loc = 'D:/rongshidata/rongshiFeatureSets1709_Ver2/'
# four_factors = [str(x) for x in range(12, 16)]

start_date = '201712'
end_date = '201801'
stock_pool_loc = 'D:/rongshidata/experiment_data_1/'       # change it to select the stock pool
feature_dirs = glob.glob(stock_pool_loc + 'monthly/end/feature_v*')
three_factors = ['v' + str(x) for x in range(2, 10)] + ['10', '11']
used_range = ['6m_1_16', '6m_3_18', '12m_1_16', '12m_3_18']

if __name__ == '__main__':
    for feature_dir in feature_dirs:
        # feature_v = os.path.basename(feature_dir)
        data_indexs = glob.glob(feature_dir + '/training/*_*_*')
        for data_index in data_indexs:
            data_index = os.path.basename(data_index)
            if not data_index in used_range:
                break

            if feature_dir[-2:] in three_factors:
                if data_index[0] in ['3', '6']:
                    estimator_list = [65, 70, 75]
                    nl_list = [50, 90]
                else:
                    estimator_list = [55, 60, 65]
                    nl_list = [50, 90]
            else:
                if data_index[0] in ['1', '6']:
                    estimator_list = [43, 48, 53]
                    nl_list = [50, 90]
                else:
                    estimator_list = [40, 43, 48]
                    nl_list = [50, 90]

            for estimator in estimator_list:
                for lf in nl_list:
                    date = start_date
                    while date != end_date:
                        if data_index[0] == '6':  # 6 months
                            train_file = os.path.join(feature_dir, 'training/' + data_index + '/' + get_train2(date))
                        elif data_index[0] == '1':  # 12 months
                            train_file = os.path.join(feature_dir, 'training/' + data_index + '/' + get_train(date))
                        elif data_index[0] == '2':  # 24 months
                            train_file = os.path.join(feature_dir, 'training/' + data_index + '/' + get_train3(date))
                        elif data_index[0] == '3':  # 3 months
                            train_file = os.path.join(feature_dir, 'training/' + data_index + '/' + get_train4(date))
                        else:
                            continue

                        if not os.path.exists(os.path.join(feature_dir, 'prob', data_index, 'PNE' + str(estimator) + '_NL_' + str(lf))):
                            os.makedirs(os.path.join(feature_dir, 'prob', data_index, 'PNE' + str(estimator) + '_NL_' + str(lf)))

                        test_file = os.path.join(feature_dir, 'testing/' + get_test(date))
                        write_file = os.path.join(feature_dir, 'prob/' + data_index + '/PNE' + str(estimator) + '_NL_' + str(lf) + '/' + get_prob(date))

                        print(train_file)
                        print(test_file)
                        print('loading data...')
                        train = pd.read_csv(train_file, na_filter=False, memory_map=True)
                        test = pd.read_csv(test_file, na_filter=False, memory_map=True)

                        target = 'yield_class'
                        predictors = [x for x in train.columns if x not in [target, 'id']]
                        x_train = train[predictors].values
                        y_train = train[target].values

                        print("search for the best params")
                        param_test1 = {'max_depth': range(5, 10), 'min_child_weight': range(1, 6)}
                        param_test2 = {'subsample': [i / 10.0 for i in range(6, 10)], 'colsample_bytree': [i / 10.0 for i in range(6, 10)]}

                        lgb_classifier1 = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=estimator, num_leaves=lf, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.7,
                                                             objective='binary', nthread=4, seed=27)

                        clf1 = GridSearchCV(lgb_classifier1, param_test1, refit=False, cv=5, scoring='precision_macro', n_jobs=2, pre_dispatch='2*n_jobs')
                        clf1.fit(x_train, y_train)
                        print(clf1.best_params_)

                        max_dep = clf1.best_params_['max_depth']
                        child = clf1.best_params_['min_child_weight']
                        print("end of search one!!")

                        if 2 ** max_dep <= lf:
                            n_leaf = 2 ** max_dep
                        else:
                            n_leaf = lf

                        lgb_classifier2 = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=estimator, num_leaves=n_leaf, max_depth=max_dep, min_child_weight=child, subsample=0.8,
                                                             colsample_bytree=0.7, objective='binary', nthread=4, seed=27)

                        clf2 = GridSearchCV(lgb_classifier2, param_test2, refit=False, cv=5, scoring='precision_macro', n_jobs=2, pre_dispatch='2*n_jobs')
                        clf2.fit(x_train, y_train)
                        print(clf2.best_params_)

                        sub = clf2.best_params_['subsample']
                        bytree = clf2.best_params_['colsample_bytree']
                        print("end of search two!!")

                        lgb_classifier3 = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=estimator, num_leaves=n_leaf, max_depth=max_dep, min_child_weight=child, subsample=sub,
                                                             colsample_bytree=bytree, objective='binary', nthread=4, seed=27)

                        modelfit(lgb_classifier3, train, test)
                        print(write_file)
                        print('\n')

                        date = next_month(date)
