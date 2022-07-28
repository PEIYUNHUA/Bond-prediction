from models.train_models import *
from configs.CONFIGS import *
import csv


def gen_models(raw_data, best_model_df):
    for i in range(0, len(best_model_df)):
        best_lists = best_model_df['feature_comb'][i]
        epochs = best_model_df['epochs'][i]
        learning_rate = best_model_df['learning_rate'][i]

        input_size = len(list(set(target_list) ^ set(best_lists)))
        data = raw_data[best_lists]
        data = (data.loc[data['date'].dt.year >= start_year]).reset_index().drop(columns='index')
        data_len = int(len(data)*0.8)
        train_data = data[:data_len]
        test_data = data[data_len:]
        train_model_savings(i, input_size, train_data, test_data, epochs, learning_rate)


def gen_prediction(today_data, best_comb_df):
    feature_combs = np.array(best_comb_df['feature_comb']).tolist()
    res_list = []

    for i in range(len(best_comb_df)):
        re_list = eval(feature_combs[i])
        best_comb_df['feature_comb'][i] = re_list
    for i in range(0, len(best_comb_df)):
        best_lists = best_comb_df.loc[i]['feature_comb']
        input_size = len(list(set(target_list) ^ set(best_lists)))
        model = LSTM1(num_classes, input_size, hidden_size, num_layer, 1)
        model_dir = model_step1_output_io + '-{}'.format(i+1) + '.pt'
        model.load_state_dict(torch.load(model_dir))
        model.eval()

        data = today_data[best_lists]
        data = (data.loc[data['date'].dt.year >= start_year]).reset_index().drop(columns='index')

        x_feed = data.drop('date', axis=1)
        x_feed = x_feed.drop(target, axis=1)

        # means and fit
        sc = StandardScaler()
        # sc = MinMaxScaler()
        sc.fit(x_feed)
        x_feed_std = sc.transform(x_feed)
        x_feed_tensors = Variable(torch.Tensor(np.array(x_feed_std)))
        x_feed_tensors_final = torch.reshape(x_feed_tensors,   (x_feed_tensors.shape[0], 1, x_feed_tensors.shape[1]))
        output = model(x_feed_tensors_final)
        fin_pre = output.data.detach().cpu().numpy()
        fin_res = round((fin_pre[-1].astype('float').tolist())[0], 4)
        res_list.append(fin_res)
    latest_date = data.iloc[-1]['date']
    latest_real = data.iloc[-1][target]
    average_res = np.mean(res_list)
    average_res = round(average_res, 4)
    res_list.append('avg')
    res_list.append(average_res)
    res_list.append('predicting_date')
    res_list.append(latest_date)
    res_list.append('last_real')
    res_list.append(round(latest_real,4))
    # LOGS
    with open(log_io, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(res_list)
    return res_list, average_res