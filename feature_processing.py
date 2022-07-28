import time

import matplotlib.pyplot as plt

from itertools import combinations
from configs.CONFIGS import *
from models.train_models import train_model
import pandas as pd


def gen_feature_comb():
    feature_comb_df = pd.DataFrame(
        columns=['feature_num', 'comb_num', 'feature_comb', 'r2', 'mae', 'mpea', 'timecost'])
    for feature_num in range(1, len(feature_list) + 1):
        comb_num = 1
        for feature_comb in combinations(feature_list, feature_num):
            feature_comb_list = target_list + list(feature_comb)
            __feature_dataframe = pd.DataFrame([[feature_num, comb_num, feature_comb_list]],
                                               columns=['feature_num', 'comb_num', 'feature_comb'])
            comb_num = comb_num + 1
            feature_comb_df = feature_comb_df.append(__feature_dataframe)
    feature_comb_df = feature_comb_df.reset_index().drop(columns='index')
    return feature_comb_df


def search_comb(raw_data, feature_comb_df):
    for i in range(len(feature_comb_df)):
        # print(feature_comb_df['feature_comb'][i])
        # print(i)
        starttime = time.time()
        input_size = feature_comb_df['feature_num'][i]
        data = raw_data[feature_comb_df['feature_comb'][i]]
        # time cut
        data = (data.loc[data['date'].dt.year >= start_year]).reset_index().drop(columns='index')
        data_len = int(len(data)*0.8)
        train_data = data[:data_len]
        test_data = data[data_len:]
        y_test, y_test_predict, eva_mae, eva_mpea, eva_r2 = train_model(input_size, train_data, test_data, epochs, learning_rate)

        endtime = time.time()
        timecost = endtime - starttime

        plt.figure(figsize=(10, 6))  # plotting
        plt.axvline(x=230, c='r', linestyle='--')
        plt.plot(y_test, label='Actuall Data')  # actual plot
        plt.plot(y_test_predict, label='Predicted Data')  # predicted plot
        plt.title('Time-Series Prediction')
        plt.legend()
        plt.savefig(
            './img/step1/feature_num{}-pic{}.png'.format(feature_comb_df['feature_num'][i], feature_comb_df['comb_num'][i]))
        plt.close()

        feature_comb_df['r2'][i] = round(eva_r2, 4)
        feature_comb_df['mae'][i] = round(eva_mae, 4)
        feature_comb_df['mpea'][i] = round(eva_mpea, 4)
        feature_comb_df['timecost'][i] = round(timecost, 4)
        print('feature_num{}-comb_num{} generation Success!({}/{}) {}'.format(feature_comb_df['feature_num'][i],
                                                                              feature_comb_df['comb_num'][i], i + 1,
                                                                              len(feature_comb_df), time.ctime()))
    feature_comb_df.to_csv(feature_res_output_io)
    print('feature_res.csv generation Success!')
    return feature_comb_df