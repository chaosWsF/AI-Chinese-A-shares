import pandas as pd
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


def last_month(date):
    """get the name of last month"""
    date = str(date)
    year = date[:4]
    month = date[4:]
    if month == '10':
        month = '09'
    elif month == '01':
        year = str(int(year) - 1)
        month = '12'
    else:
        month = int(month) - 1
        if month < 10:
            month = '0' + str(month)
        else:
            month = str(month)
    ret = year + month
    return ret


def next_month(date):
    """get the name of next month"""
    date = str(date)
    year = date[:4]
    month = date[4:]
    if month == '09':
        month = '10'
    elif month == '12':
        year = str(int(year) + 1)
        month = '01'
    else:
        month = int(month) + 1
        if month < 10:
            month = '0' + str(month)
        else:
            month = str(month)
    ret = year + month
    return ret


def get_train12(date):
    """get the file name of the training data for 12 months"""
    date = str(date)
    year = date[:4]
    month = date[4:]
    first = str(int(year) - 1) + month
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


def get_train6(date):
    """get the file name of the training data for 6 months"""
    date = str(date)
    year = date[:4]
    month = date[4:]
    a = int(month) - 6
    if a < 1:
        first = str(int(year) - 1)
        if a + 12 >= 10:
            first = first + str(a + 12)
        else:
            first = first + '0' + str(a + 12)
    else:
        first = year + '0' + str(a)
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


def get_train24(date):
    """get the file name of the training data for 24 months"""
    date = str(date)
    year = date[:4]
    month = date[4:]
    first = str(int(year) - 2) + month
    second = last_month(date)
    ret = first + '-' + second + '_train.csv'
    return ret


def get_train3(date):
    """get the file name of the training data for 3 months"""
    date = str(date)
    year = date[:4]
    month = date[4:]
    a = int(month) - 3
    if a < 1:
        first = str(int(year) - 1)
        if a + 12 >= 10:
            first = first + str(a + 12)
        else:
            first = first + '0' + str(a + 12)
    else:
        first = year + '0' + str(a)
    second = last_month(date)
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

start_date = '201709'
end_date = '201710'
path0 = 'D:/rongshidata/experiment_data_1/monthly/end'
paradata = pd.read_excel('parameter.xlsx')
pNum = 0

for factor in os.listdir(path0):
    factor_path = os.path.join(path0, factor)
    quantile_path = os.path.join(factor_path, 'training')
    prob_path = os.path.join(factor_path, 'prob')
    for quantile in os.listdir(quantile_path):
        basis1 = factor not in ['feature_v2', 'feature_v12', 'feature_v18']
        basis2 = quantile not in ['6m_1_16', '12m_1_16', '3m_1_31']
        if basis1 or basis2:
            continue

        # # 选择优势训练集
        # if quantile == '6m_1_16':
        #     if factor not in ['feature_v2', 'feature_v3', 'feature_v5', 'feature_v6',
        #                       'feature_v7', 'feature_v8', 'feature_v10', 'feature_v11',
        #                       'feature_v12', 'feature_v13', 'feature_v16', 'feature_v17',
        #                       'feature_v18', 'feature_v19']:
        #         continue
        # elif quantile == '6m_3_18':
        #     if factor not in ['feature_v2', 'feature_v3', 'feature_v5', 'feature_v6',
        #                       'feature_v8', 'feature_v9', 'feature_v11', 'feature_v16',
        #                       'feature_v17', 'feature_v18']:
        #         continue
        # # elif quantile == '12m_1_16':
        # #     if factor not in ['feature_v2', 'feature_v4', 'feature_v15']:
        # #         continue
        # # elif quantile == '12m_3_18':
        # #     if factor not in ['feature_v2', 'feature_v4', 'feature_v13']:
        # #         continue
        # else:
        #     continue

        train_path0 = os.path.join(quantile_path, quantile)
        prob_path0 = os.path.join(prob_path, quantile)
        if not os.path.exists(prob_path0):
            os.makedirs(prob_path0)

        # 选择参数
        params = paradata[(paradata['feature'] == factor) & (paradata['quantile'] == quantile)]
        params = params['parameters'].values
        params = params[0]
        params = params.split(',')
        params = [int(s) for s in params]
        for pnum in params:

            pNum = pnum
            config_path = 'parameters/p' + str(pNum) + '.json'
            config = load_json(config_path)
            gc = GCForest(config)  # should be a dict

            usedate = start_date
            while usedate != end_date:
                test_path = os.path.join(factor_path, 'testing')
                pred_path = os.path.join(test_path, get_test(usedate))

                pred_file = pd.read_csv(pred_path)
                target = 'yield_class'
                dataIndex = list(pred_file.columns)[1:-1]
                pred_file = pred_file.fillna(0)
                x_pred = pred_file[dataIndex].values

                if quantile[0] == '6':  # 6 months
                    train_path = os.path.join(train_path0, get_train6(usedate))
                elif quantile[0] == '1':  # 12 months
                    train_path = os.path.join(train_path0, get_train12(usedate))
                elif quantile[0] == '2':  # 24 months
                    train_path = os.path.join(train_path0, get_train24(usedate))
                else:  # 3 months
                    train_path = os.path.join(train_path0, get_train3(usedate))

                train_file = pd.read_csv(train_path)
                train_file = train_file.fillna(0)
                x_train = train_file[dataIndex].values
                y_train = train_file[target].values
                y_train = y_train.reshape((y_train.shape[0],))
                x_train_enc = gc.fit_transform(x_train, y_train)
                y_test = gc.predict(x_train)
                y_test_predproba = gc.predict_proba(x_train)[:, 1]
                acc = accuracy_score(y_train, y_test)
                auc = roc_auc_score(y_train, y_test_predproba)

                prob_path1 = prob_path0 + '/deepforest/p' + str(pNum)
                if not os.path.isdir(prob_path1):
                    os.makedirs(prob_path1)

                write_path = prob_path1 + '/' + get_prob(usedate)
                y_predproba = gc.predict_proba(x_pred)
                pred_file[target] = y_predproba[:, 1]
                pred_file.to_csv(write_path, index=False)

                print("Model Report")
                print(train_path)
                print(pred_path)
                print(write_path)
                print("Accuracy of GcForest = {:.2f} %".format(acc * 100))
                print("AUC Score of GcForest = {:.2f}".format(auc))

                usedate = next_month(usedate)
