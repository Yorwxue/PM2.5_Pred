# _*_ coding: utf-8 _*_

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import cPickle
import os
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from reader import read_data_sets, concatenate_time_steps
from missing_value_processer import missing_check
from feature_processor import data_coordinate_angle
from config import root

root_path = root()


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


pollution_site_map = {
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],  # 5
           '南投': ['南投', '竹山'],  # 2
           '彰化': ['二林', '彰化']},  # 2

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],  # 5
           '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],  # 8
           '基隆': ['基隆'],  # 1
           '桃園': ['大園', '平鎮', '桃園', '龍潭']}, # 4

    '宜蘭': {'宜蘭': ['冬山', '宜蘭']},  # 2

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],  # 3
           '苗栗': ['三義', '苗栗']},  # 2

    '花東': {'花蓮': ['花蓮'],  # 1
           '台東': ['臺東']},  # 1

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'], # 1
             '連江': ['馬祖'],  # 1
             '東吉嶼': [],
             '澎湖': ['馬公']},  # 1

    '雲嘉南': {'雲林': ['崙背', '斗六', '竹山'],  # 3
            '台南': ['善化', '安南', '新營', '臺南'],  # 4
            '嘉義': ['嘉義', '新港', '朴子']},  # 3

    '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],  # 8
           '屏東': ['屏東', '恆春', '潮州']}  # 3
}


high_alert = 53.5
low_alert = 35.5

local = '北部'
city = '台北'
site_list = pollution_site_map[local][city]  # ['中山', '古亭', '士林', '松山', '萬華']
target_site = '萬華'

training_year = ['2014', '2016']  # change format from   2014-2015   to   ['2014', '2015']
testing_year = ['2016', '2016']

training_duration = ['1/1', '10/31']
testing_duration = ['12/1', '12/31']
interval_hours = 24  # predict the label of average data of many hours later, default is 1
is_training = True

# local = os.sys.argv[1]
# city = os.sys.argv[2]
# site_list = pollution_site_map[local][city]
# target_site = os.sys.argv[3]
#
# training_year = [os.sys.argv[4][:os.sys.argv[4].index('-')], os.sys.argv[4][os.sys.argv[4].index('-')+1:]]  # change format from   2014-2015   to   ['2014', '2015']
# testing_year = [os.sys.argv[5][:os.sys.argv[5].index('-')], os.sys.argv[5][os.sys.argv[5].index('-')+1:]]
#
# training_duration = [os.sys.argv[6][:os.sys.argv[6].index('-')], os.sys.argv[6][os.sys.argv[6].index('-')+1:]]
# testing_duration = [os.sys.argv[7][:os.sys.argv[7].index('-')], os.sys.argv[7][os.sys.argv[7].index('-')+1:]]
# interval_hours = int(os.sys.argv[8])  # predict the label of average data of many hours later, default is 1
# is_training = True if (os.sys.argv[9] == 'True' or os.sys.argv[9] == 'true') else False  # True False

target_kind = 'PM2.5'

# clear redundancy work
if training_year[0] == training_year[1]:
    training_year.pop(1)
if testing_year[0] == testing_year[1]:
    testing_year.pop(1)
else:
    input('The range of testing year should not more than one year or crossing the bound of years.')

# checking years
rangeofYear = int(training_year[-1])-int(training_year[0])
for i in range(rangeofYear):
    if not(str(i+int(training_year[0])) in training_year):
        training_year.insert(i, str(i+int(training_year[0])))

# Training Parameters
# WIND_DIREC is a specific feature, that need to be processed, and it can only be element of input vector now.
pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NO2', 'WIND_SPEED', 'WIND_DIREC']  # , 'AMB_TEMP', 'RH'
data_update = False
# batch_size = 24 * 7
seed = 0

# Network Parameters
input_size = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))
time_steps = 12
# hidden_size = 20
output_size = 1

testing_month = testing_duration[0][:testing_duration[0].index('/')]
folder = root_path+"model/%s/%s/%sh/" % (local, city, interval_hours)
training_begining = training_duration[0][:training_duration[0].index('/')]
training_deadline = training_duration[-1][:training_duration[-1].index('/')]
print('site: %s' % target_site)
print('Training for %s/%s to %s/%s' % (training_year[0], training_duration[0], training_year[-1], training_duration[-1]))
print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
print('Target: %s' % target_kind)


# for interval
def ave(Y, interval_hours):
    reserve_hours = interval_hours - 1
    deadline = 0
    for i in range(len(Y)):
        # check the reserve data is enough or not
        if (len(Y) - i - 1) < reserve_hours:
            deadline = i
            break  # not enough
        for j in range(reserve_hours):
            Y[i] += Y[i + j + 1]
        Y[i] /= interval_hours
    if deadline:
        Y = Y[:deadline]
    return Y


# for interval
def higher(Y, interval_hours):
    reserve_hours = 1  # choose the first n number of biggest
    if interval_hours > reserve_hours:
        deadline = 0
        for i in range(len(Y)):
            # check the reserve data is enough or not
            if (len(Y) - i) < interval_hours:
                deadline = i
                break  # not enough
            higher_list = []
            for j in range(interval_hours):
                if len(higher_list) < reserve_hours:
                    higher_list.append(Y[i + j])
                elif Y[i + j] > higher_list[0]:
                    higher_list[0] = Y[i + j]
                higher_list = sorted(higher_list)
            Y[i] = np.array(higher_list).sum() / reserve_hours
        if deadline:
            Y = Y[:deadline]
    return Y


if is_training:
    # reading data
    print('Reading data .. ')
    start_time = time.time()
    initial_time = time.time()
    print('preparing training set ..')
    X_train = read_data_sets(sites=site_list+[target_site], date_range=np.atleast_1d(training_year),
                             beginning=training_duration[0], finish=training_duration[-1],
                             feature_selection=pollution_kind, update=data_update)
    X_train = missing_check(X_train)
    Y_train = np.array(X_train)[:, -len(pollution_kind):]
    Y_train = Y_train[:, pollution_kind.index(target_kind)]
    X_train = np.array(X_train)[:, :-len(pollution_kind)]

    print('preparing testing set ..')
    X_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                            beginning=testing_duration[0], finish=testing_duration[-1],
                            feature_selection=pollution_kind, update=data_update)
    Y_test = np.array(X_test)[:, -len(pollution_kind):]
    Y_test = Y_test[:, pollution_kind.index(target_kind)]
    X_test = missing_check(np.array(X_test)[:, :-len(pollution_kind)])

    final_time = time.time()
    print('Reading data .. ok, ', end='')
    time_spent_printer(start_time, final_time)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    if (len(X_train) < time_steps) or (len(X_test) < time_steps):
        input('time_steps(%d) too long.' % time_steps)

    # normalize
    print('Normalize ..')
    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    if 0 in std_X_train:
        input("Denominator can't be 0.")
    X_train = np.array([(x_train-mean_X_train)/std_X_train for x_train in X_train])
    X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])

    mean_y_train = np.mean(Y_train)
    std_y_train = np.std(Y_train)
    if not std_y_train:
        input("Denominator can't be 0.")
    Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]
    print('mean_y_train: %f  std_y_train: %f' % (mean_y_train, std_y_train))

    fw = open(folder + "%s_parameter_%s.pickle" % (target_site, target_kind), 'wb')
    cPickle.dump(str(mean_X_train) + ',' +
                 str(std_X_train) + ',' +
                 str(mean_y_train) + ',' +
                 str(std_y_train), fw)
    fw.close()

    # feature process
    if 'WIND_DIREC' in pollution_kind:
        index_of_kind = pollution_kind.index('WIND_DIREC')
        length_of_kind_list = len(pollution_kind)
        len_of_sites_list = len(site_list)
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        for i in range(len(X_train)):
            for j in range(len_of_sites_list):
                specific_index = index_of_kind + j * length_of_kind_list
                coordin = data_coordinate_angle((X_train[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
                X_train[i].insert(specific_index + j, coordin[1])
                X_train[i].insert(specific_index + j, coordin[0])
                if i < len(X_test):
                    coordin = data_coordinate_angle((X_test[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
                    X_test[i].insert(specific_index + j, coordin[1])
                    X_test[i].insert(specific_index + j, coordin[0])
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    print('Constructing time series data set ..')
    X_train = concatenate_time_steps(X_train[:-1], time_steps)
    Y_train = Y_train[time_steps:]

    X_test = concatenate_time_steps(X_test[:-1], time_steps)
    Y_test = Y_test[time_steps:]

    Y_real = np.copy(Y_test)

    Y_train = higher(Y_train, interval_hours)
    Y_test = higher(Y_test, interval_hours)
    # Y_train = Y_train[interval_hours - 1:]
    # Y_test = Y_test[interval_hours - 1:]
    Y_real = Y_real[interval_hours - 1:]

    train_seq_len = np.min([len(Y_train), len(X_train)])
    test_seq_len = np.min([len(Y_test), len(X_test)])

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    X_train = X_train[:train_seq_len]
    X_test = X_test[:test_seq_len]

    Y_train = Y_train[:train_seq_len]

    Y_test = Y_test[:test_seq_len]
    Y_real = Y_real[:test_seq_len]

    # delete data which have missing values
    i = 0
    while i < len(Y_test):
        if (not(Y_test[i] > -10000) or (not(Y_real[i] > -10000))):  # check missing or not, if Y_test[i] is missing, then this instruction will return True
            Y_test = np.delete(Y_test, i, 0)
            Y_real = np.delete(Y_real, i, 0)
            X_test = np.delete(X_test, i, 0)
            i = -1
        i += 1
    Y_test = np.array(Y_test, dtype=np.float)
    Y_real = np.array(Y_real, dtype=np.float)

    print('delete invalid testing data, remain ', len(X_test), 'test sequences')

    # --

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(Y_train)


else:  # is_training = false
    # mean and std
    fr = open(folder + "%s_parameter_%s.pickle" % (target_site, target_kind), 'rb')
    [mean_X_train, std_X_train, mean_y_train, std_y_train] = (cPickle.load(fr)).split(',')
    mean_X_train = mean_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
    while '' in mean_X_train:
        mean_X_train.pop(mean_X_train.index(''))
    mean_X_train = np.array(mean_X_train, dtype=np.float)
    std_X_train = std_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
    while '' in std_X_train:
        std_X_train.pop(std_X_train.index(''))
    std_X_train = np.array(std_X_train, dtype=np.float)
    mean_y_train = float(mean_y_train)
    std_y_train = float(std_y_train)
    fr.close()

    # reading data
    print('preparing testing set ..')
    X_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                            beginning=testing_duration[0], finish=testing_duration[-1],
                            feature_selection=pollution_kind, update=data_update)
    Y_test = np.array(X_test)[:, -len(pollution_kind):]
    Y_test = Y_test[:, pollution_kind.index(target_kind)]
    X_test = missing_check(np.array(X_test)[:, :-len(pollution_kind)])

    # normalize
    print('Normalize ..')
    if 0 in std_X_train:
        input("Denominator can't be 0.")
    X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])

    # feature process
    if 'WIND_DIREC' in pollution_kind:
        index_of_kind = pollution_kind.index('WIND_DIREC')
        length_of_kind_list = len(pollution_kind)
        len_of_sites_list = len(site_list)
        X_test = X_test.tolist()
        for i in range(len(X_test)):
            for j in range(len_of_sites_list):
                specific_index = index_of_kind + j * length_of_kind_list
                coordin = data_coordinate_angle(
                    (X_test[i].pop(specific_index + j)) * std_X_train[specific_index] + mean_X_train[
                        specific_index])
                X_test[i].insert(specific_index + j, coordin[1])
                X_test[i].insert(specific_index + j, coordin[0])
        X_test = np.array(X_test)
    Y_test = np.array(Y_test, dtype=np.float)

    # --

    print('Constructing time series data set ..')
    X_test = concatenate_time_steps(X_test[:-1], time_steps)
    Y_test = Y_test[time_steps:]

    Y_real = np.copy(Y_test)

    Y_test = higher(Y_test, interval_hours)
    # Y_test = Y_test[interval_hours - 1:]
    Y_real = Y_real[interval_hours - 1:]

    test_seq_len = np.min([len(Y_test), len(X_test)])

    print(len(X_test), 'test sequences')

    X_test = X_test[:test_seq_len]

    Y_test = Y_test[:test_seq_len]
    Y_real = Y_real[:test_seq_len]

    # delete data which have missing values
    i = 0
    while i < len(Y_test):
        if not (not(Y_test[i] > -10000) or (not(Y_real[i] > -10000))):  # check missing or not, if Y_test[i] is missing, then this command will return True
            Y_test = np.delete(Y_test, i, 0)
            Y_real = np.delete(Y_real, i, 0)
            X_test = np.delete(X_test, i, 0)
            i = -1
        i += 1
    Y_test = np.array(Y_test, dtype=np.float)
    Y_real = np.array(Y_real, dtype=np.float)

    print('delete invalid testing data, remain ', len(X_test), 'test sequences')

    # --

    X_test = np.array(X_test)

# -- xgboost --
print('- xgboost -')

filename = ("xgboost_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours, target_kind))
print(filename)

if is_training:
    xgb_model = xgb.XGBRegressor().fit(X_train, Y_train)

    fw = open(folder + filename, 'wb')
    cPickle.dump(xgb_model, fw)
    fw.close()
    print('model saved')
else:
    fr = open(folder + filename, 'rb')
    xgb_model = cPickle.load(fr)
    fr.close()

xgb_pred = xgb_model.predict(X_test)

print('rmse(xgboost): %.5f' % (np.mean((Y_test - (mean_y_train + std_y_train * xgb_pred))**2, 0)**0.5))


# -- random forest --
# print('- random forest -')
#
# filename = ("random_forest_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
#             % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours, target_kind))
# print(filename)
#
# if is_training:
#     rf_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                                      max_features='auto', max_leaf_nodes=None, min_samples_leaf=2,
#                                      min_samples_split=10, min_weight_fraction_leaf=0.0,
#                                      n_estimators=10, n_jobs=5, oob_score=False, random_state=None,
#                                      verbose=0, warm_start=False)
#     rf_model.fit(X_train, Y_train)
#
#     fw = open(folder + filename, 'wb')
#     cPickle.dump(rf_model, fw)
#     fw.close()
#     print('model saved')
# else:
#     fr = open(folder + filename, 'rb')
#     rf_model = cPickle.load(fr)
#     fr.close()
#
# rf_pred = rf_model.predict(X_test)
#
# print('rmse(random forest): %.5f' % (np.mean((Y_test - (mean_y_train + std_y_train * rf_pred))**2, 0)**0.5))

# --
filename = ("xgboost_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
            % (target_site, training_year[0], training_begining, training_year[-1], training_deadline, interval_hours, target_kind))

xg_pred = mean_y_train + std_y_train * xgb_pred
# rf_pred = mean_y_train + std_y_train * rf_pred

# plt_length = np.min([len(Y_real), len(Y_test), len(xg_pred), len(rf_pred)])
plt_length = np.min([len(Y_real), len(Y_test), len(xg_pred)])
plt.plot(np.arange(plt_length), Y_real[:plt_length], c='gray')
plt.plot(np.arange(plt_length), Y_test[:plt_length], c='mediumaquamarine')
plt.plot(np.arange(plt_length), xg_pred[:plt_length], color='palevioletred')
# plt.plot(np.arange(plt_length), rf_pred[:plt_length], color='pink')
plt.xticks(np.arange(0, plt_length, 24))
plt.yticks(np.arange(0, max(Y_test), 10))
plt.grid(True)
plt.rc('axes', labelsize=4)
plt.savefig(root_path + 'result/' + filename + '.png')
# plt.show()
