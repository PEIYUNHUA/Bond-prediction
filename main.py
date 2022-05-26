import time

import matplotlib.pyplot as plt

import pre_processing as pre
import feature_processing as feature
import para_processing as para
import evaluation
from configs.CONFIGS import *
from models.train_models import *
from model_processing import *

from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import numpy as np

import pandas as pd
import re

if __name__ == '__main__':
    # TYPE 1 MODEL GENERATION
    # TYPE 2 PREDICTIONS GENERATION
    TYPE = 2

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
            print('Model Generation Finished!')

        except Exception as e:
            print('Model Generation Failed!', e)

    elif TYPE == 2:
        # PREDICTIONS GENERATION
        try:
            today_data = pre.get_raw_data(today_data_io)
            best_comb_df = pd.read_csv(best_parameter_output_io)
            res_list, average_res = gen_prediction(today_data, best_comb_df)
            print('Prediction Generation Success！result:{}, by{}'.format(average_res, res_list))
        except Exception as e:
            print('Prediction Generation Failed!', e)

    else:
        print('TYPE error')