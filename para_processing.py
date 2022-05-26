import pandas as pd
import time
import matplotlib.pyplot as plt
from configs.CONFIGS import *
from models.train_models import train_model


def search_para(raw_data, best_comb_df):
    para_comb_df = best_comb_df[['feature_num', 'comb_num', 'feature_comb']]
    par_res_df = pd.DataFrame(columns=['feature_comb', 'r2', 'mae', 'mpea', 'timecost', 'epochs', 'learning_rate',  'y_test_predict'])
    count = 0
    for i in range(len(para_comb_df)):
        best_lists = para_comb_df['feature_comb'][i]
        for x in epochs_list:
            for y in learning_rate_list:
                epochs = x
                learning_rate = y
                input_size = len(list(set(target_list) ^ set(best_lists)))
                starttime = time.time()
                data = raw_data[best_lists]
                data = (data.loc[data['date'].dt.year >= start_year]).reset_index().drop(columns='index')
                # print(data.info())1820 460
                train_data = data[:1820]
                test_data = data[1820:]
                y_test, y_test_predict, eva_mae, eva_mpea, eva_r2 = train_model(input_size, train_data, test_data, epochs, learning_rate)
                endtime = time.time()
                timecost = endtime-starttime
                # print('r2:{} mae:{} mpea:{} timecost:{}'.format(round(eva_r2, 4), round(eva_mae, 4), round(eva_mpea, 4), round(timecost, 4)))

                plt.figure(figsize=(10, 6)) #plotting
                plt.axvline(x=230, c='r', linestyle='--')
                plt.plot(y_test, label='Actuall Data') #actual plot
                plt.plot(y_test_predict, label='Predicted Data') #predicted plot
                plt.title('Time-Series Prediction')
                plt.legend()
                plt.savefig('./img/step2/comb{}-epochs{}-learning_rate{}.png'.format(best_lists, x, y))
                plt.close()

                # par_res_df = pd.DataFrame(
                #     columns=['feature_comb', 'r2', 'mae', 'mpea', 'timecost', 'epochs', 'learning_rate',
                #              'y_test_predict'])
                par_res_df.loc[count, 'feature_comb'] = best_lists
                par_res_df.loc[count, 'r2'] = round(eva_r2, 4)
                par_res_df.loc[count, 'mae'] = round(eva_mae, 4)
                par_res_df.loc[count, 'mpea'] = round(eva_mpea, 4)
                par_res_df.loc[count, 'timecost'] = round(timecost, 4)
                par_res_df.loc[count, 'epochs'] = epochs
                par_res_df.loc[count, 'learning_rate'] = learning_rate
                par_res_df.loc[count, 'y_test_predict'] = y_test_predict

                count = count + 1
                print('comb{}-epochs{}-learning_rate{} saved! ({}/{}) {}'.format(best_lists, x, y, count, len(para_comb_df) * len(epochs_list) * len(learning_rate_list), time.ctime()))
    par_res_df.to_csv(parameter_res_output_io)
    return par_res_df