# _*_ coding: utf-8 _*_

from data_reader import data_reader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def param_code_to_param_code_no(param_code):
    param_code = int(param_code)
    if param_code == '風速':
        return 0
    elif param_code == '風向':
        return 1
    elif param_code == '溫度':
        return 2
    elif param_code == '露點':
        return 3
    elif param_code == '氣壓':
        return 4
    elif param_code == '降雨':
        return 5
    elif param_code == '相對濕度':
        return 6
    elif param_code == '太陽輻射':
        return 7
    elif param_code == '日照時數':
        return 8
    elif param_code == '日照分鐘數':
        return 9
    elif param_code == 'site pressure':
        return 10
    elif param_code == 'sea level pressure':
        return 11
    elif param_code == '降雨時數':
        return 12
    else:
        print("This param_code doesn't exist.")


def pollution_to_pollution_no(pollution):
    if pollution == 'SO2':return 0
    elif pollution == 'CO':return 1
    elif pollution == 'O3':return 2
    elif pollution == 'PM10':return 3
    elif pollution == 'PM2.5':return 4
    elif pollution == 'NOx':return 5
    elif pollution == 'NO':return 6
    elif pollution == 'NO2':return 7
    elif pollution == 'THC':return 8
    elif pollution == 'NMHC':return 9
    elif pollution == 'CH4':return 10
    elif pollution == 'UVB':return 11
    elif pollution == 'AMB_TEMP':return 12
    elif pollution == 'RAINFALL':return 13
    elif pollution == 'RH':return 14
    elif pollution == 'WIND_SPEED':return 15
    elif pollution == 'WIND_DIREC':return 16
    elif pollution == 'WS_HR':return 17
    elif pollution == 'WD_HR':return 18
    elif pollution == 'PH_RAIN':return 19
    elif pollution == 'RAIN_COND':return 20
    else:
        print("THis pollution(%s) hasn't been recorded." % pollution)


def observe(y_d_h_data, observe_year, observe_month, observe_city, observe_kind, observe_feature, target_feature):
    if observe_month == 1 or observe_month == 3 or observe_month == 5 or observe_month == 7 or observe_month == 8 or observe_month == 10 or observe_month == 12:
        feature_vector = np.zeros(31 * 24)
        target_vector = np.zeros(31 * 24)
    elif observe_month == 4 or observe_month == 6 or observe_month == 9 or observe_month == 11:
        feature_vector = np.zeros(30 * 24)
        target_vector = np.zeros(30 * 24)
    elif observe_month == 2:
        if '2/29' in y_d_h_data[str(observe_year)]:
            feature_vector = np.zeros(29 * 24)
            target_vector = np.zeros(29 * 24)
        else:
            feature_vector = np.zeros(28 * 24)
            target_vector = np.zeros(28 * 24)

    counter = 0
    for date in y_d_h_data[str(observe_year)].keys():
        month = int(date[:date.index('/')])
        day = int(date[date.index('/')+1:])
        if month == observe_month:
            for hours in range(24):
                if observe_kind == 'pollution':
                    data = y_d_h_data[str(observe_year)][date][observe_kind][observe_city][hours][int(pollution_to_pollution_no(observe_feature))]
                    target_data = y_d_h_data[str(observe_year)][date][observe_kind][observe_city][hours][int(pollution_to_pollution_no(target_feature))]

                elif observe_kind == 'weather':
                    data = y_d_h_data[str(observe_year)][date][observe_kind][observe_city][hours][int(param_code_to_param_code_no(observe_feature))]
                    target_data = y_d_h_data[str(observe_year)][date][observe_kind][observe_city][hours][int(param_code_to_param_code_no(target_feature))]
                else:
                    print('%s error' % observe_kind)
                    break

                try:
                    data = float(data)
                except:
                    # data = -2
                    if counter != 0:
                        data = np.sum(feature_vector)/counter
                    else:
                        data = -1

                try:
                    target_data = float(target_data)
                except:
                    # target_data = -2
                    if counter != 0:
                        target_data = np.sum(target_vector)/counter
                    else:
                        data = -1

                feature_vector[((day - 1) * 24) + hours] = data
                target_vector[((day - 1) * 24) + hours] = target_data
                counter += 1
    return target_vector, feature_vector


def correlation(x, y):
    std_x = np.std(x)
    std_y = np.std(y)

    expect_x = np.mean(x)
    expect_y = np.mean(y)

    corr_xy = (np.mean((x - expect_x) * (y - expect_y))) / (std_x * std_y)

    # corr_xy = np.corrcoef(x, y)

    return corr_xy


observe_year = 2017
observe_month_list = [1]
observe_city = '中山'  # 中山 古亭 左營 萬華
observe_kind = 'pollution'  # pollution / weather
observe_feature = 'SO2'  # 'PM2.5', 'O3', 'SO2', 'CO', 'NO2', 'WIND_SPEED', 'WIND_DIREC'
target_feature = 'PM2.5'  # os.sys.argv[1]  # 'O3'

y_d_h_data = data_reader(observe_year, observe_year)

target_vector = []
feature_vector = []
for observe_month in observe_month_list:
    target, feature = observe(y_d_h_data, observe_year, observe_month, observe_city, observe_kind, observe_feature, target_feature)
    target_vector += target.tolist()
    feature_vector += feature.tolist()

print('site: %s' % observe_city)
print('standard deviation: ', np.std(feature_vector))
print('correlation: ', correlation(target_vector, feature_vector))

plt.plot(np.arange(len(feature_vector)), feature_vector, c='gray')
plt.plot(np.arange(len(feature_vector)), target_vector, color='pink')

plt.xticks(np.arange(0, len(feature_vector), 24))
plt.yticks(np.arange(0, max(feature_vector), 10))
plt.grid(True)

plt.show()
# print()

# sys.stdout.write(str(correlation(target_vector, feature_vector)))
# sys.stdout.flush()
# sys.exit(0)
