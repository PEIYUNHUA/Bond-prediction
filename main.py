import time

import pre_processing as pre
import feature_processing as feature
import para_processing as para
import evaluation
from configs.CONFIGS import *
from models.train_models import *
from model_processing import *
from wechat_processing import *

from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import numpy as np

import pandas as pd
import re

from WindPy import w
from datetime import date
import os
import shutil

def run_main(n):
    # TYPE 1 MODEL GENERATION
    # TYPE 2 PREDICTIONS GENERATION
    # TYPE 3 COLLECT DATA
    # TYPE 4 CLEAR RES & IMG
    # TYPE 5 SEND WECHAT
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    TYPE = n
    # loop_monitor()

    if TYPE == 1:
        # MODEL GENERATION
        try:
            print('Model Generation Start!')
            # LOAD DATA
            try:
                raw_data = pre.get_raw_data(pre.data_io)
                print('Data loading Success!')
            except Exception as e:
                print('Data loading Failed!', e)

            # FEATURE COMB
            try:
                feature_comb_df = feature.gen_feature_comb()
                feature_comb_df.to_csv(feature_comb_output_io)
                print('feature_comb.csv generation Success!')
            except Exception as e:
                print('Feature comb generation Failed!', e)

            # COMB SELECT
            try:
                feature_comb_df = feature.search_comb(raw_data, feature_comb_df)
                best_comb_df = evaluation.evaluations_comb(feature_comb_df)
                best_comb_df.to_csv(best_feature_output_io)
                print('best_feature_res.csv generation Success!')
            except Exception as e:
                print('feature_res.csv generation Failed!', e)

            # PARAMETER SELECT
            try:
                par_res_df = para.search_para(raw_data, best_comb_df)
                best_comb_df = evaluation.evaluations_para(par_res_df)
                best_comb_df.to_csv(best_parameter_output_io)
                print('best_parameter_res.csv generation Success!')
            except Exception as e:
                print('best_parameter_res.csv generation Failed!', e)

            # SAVE BEST MODEL
            try:
                gen_models(raw_data, best_comb_df)
                print('best_models generation Success!')
            except Exception as e:
                print('best_models generation Failed!', e)

            # REFINE RESULT
            try:
                today_data = pre.get_raw_data(pre.data_io)
                len_data = len(today_data)
                for i in range(2, -1, -1):
                    today_data = pre.get_raw_data(pre.data_io)[0:len_data-i]
                    best_comb_df = pd.read_csv(best_parameter_output_io)
                    res_list, average_res = gen_prediction(today_data, best_comb_df)
                print('Data fitting Success!')
            except Exception as e:
                print('Data fitting Failed!', e)
            print('Model Generation Finished!')

        except Exception as e:
            print('Model Generation Failed!', e)

    elif TYPE == 2:
        # PREDICTIONS GENERATION
        try:
            today_data = pre.get_raw_data(pre.data_io)
            best_comb_df = pd.read_csv(best_parameter_output_io)
            res_list, average_res = gen_prediction(today_data, best_comb_df)
            print('Prediction Generation Success!')
        except Exception as e:
            print('Prediction Generation Failed!', e)

    elif TYPE == 3:
        # COLLECT DATA
        try:
            w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
            error, data = w.wsd(index_list, "CLOSE", data_start, options="", usedf=True)
            data.to_excel((today_str + '_' + data_io))
            # w.isconnected() # 判断WindPy是否已经登录成功

            # w.stop() # 当需要停止WindPy时，可以使用该命令
            #           # 注： w.start不重复启动，若需要改变参数，如超时时间，用户可以使用w.stop命令先停止后再启动。
            #           # 退出时，会自动执行w.stop()，一般用户并不需要执行w.stop
            # w.wsd（codes, fields, beginTime, endTime, options）
            print('Data Generation Success!')
        except Exception as e:
            print('Prediction Generation Failed!', e)

    elif TYPE == 4:
        # CLEAR ALL
        try:
            # COPY
            new_folder = backup_io + today_str
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            files = os.listdir(model_step1_io)
            for f in files:
                shutil.copy(model_step1_io + '/' + f, new_folder + '/' + f)
            shutil.copy(best_parameter_output_io, new_folder + '/best_parameter_res.csv')
            shutil.copy(log_io, new_folder + '/res.csv')
            print('COPY ALL Success!')
            # DELETE
            for i in filepaths:
                filepath = i
                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                else:
                    shutil.rmtree(filepath)
                    os.mkdir(filepath)
            print('DELETE ALL Success!')
        except Exception as e:
            print('CLEAR ALL Failed!', e)

    elif TYPE == 5:
        # SEND WECHAT
        try:
            wx_warning()
            print('SEND WECHAT Success!')
        except Exception as e:
            print('SEND WECHAT Failed!', e)
    else:
        print('TYPE error')


# if __name__ == '__main__':
    # # UPDATE_PREDICTION
    # run_main(3)
    # run_main(2)
    # run_main(5)
    #
    # # UPDATE_MODEL
    # run_main(4)
    # run_main(1)
    # run_main(2)
