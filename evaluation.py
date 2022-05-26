import pandas as pd
from configs.CONFIGS import *


def evaluations_comb(df):
    df['r2'] = 1 - df['r2']
    df.sort_values(by="r2", inplace=True, ascending=True)
    top10_r2 = df.head(top_num)
    df.sort_values(by="mae", inplace=True, ascending=True)
    top10_mae = df.head(top_num)
    df.sort_values(by="mpea", inplace=True, ascending=True)
    top10_mpea = df.head(top_num)
    # TODO case when none match
    __df = pd.merge(top10_r2, top10_mae, on=['feature_num', 'comb_num', 'r2', 'mae', 'mpea', 'timecost'])
    best_df = pd.merge(__df, top10_mpea, on=['feature_num', 'comb_num', 'r2', 'mae', 'mpea', 'timecost']).drop(
        columns=['feature_comb_x', 'feature_comb_y'])
    return best_df


def evaluations_para(df_input):
    df = df_input
    df = df.drop(columns=['y_test_predict'])
    df['r2'] = 1 - df['r2']
    df.sort_values(by="r2", inplace=True, ascending=True)
    top10_r2 = df.head(top_num)
    df.sort_values(by="mae", inplace=True, ascending=True)
    top10_mae = df.head(top_num)
    df.sort_values(by="mpea", inplace=True, ascending=True)
    top10_mpea = df.head(top_num)
    # TODO case when none match
    __df = pd.merge(top10_r2, top10_mae, on=['epochs', 'learning_rate', 'r2', 'mae', 'mpea', 'timecost'])
    best_df = pd.merge(__df, top10_mpea, on=['epochs', 'learning_rate', 'r2', 'mae', 'mpea', 'timecost']).drop(
        columns=['feature_comb_x', 'feature_comb_y'])
    return best_df