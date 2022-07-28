import requests
import json
import datetime
import pandas as pd
from configs.CONFIGS import *


def data_analysis():
	df = pd.read_csv(log_io)
	df = df.astype(str)
	real_list = []
	time_list = []
	new_df = pd.DataFrame(columns=['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10'])
	for x in range(0, df.shape[0]):
		avg_list = []
		for y in range(0, 10):
			each = df.loc[x][y]
			if each in ['avg', 'predicting_date', 'last_real', 'nan']:
				break
			else:
				avg_list.append(each)
		for y in range(0, 16):
			each = df.loc[x][y]
			if each == 'predicting_date':
				time_list.append(df.loc[x][y + 1])
			if each == 'last_real':
				real_list.append(df.loc[x][y + 1])
		avg_list += [None for i in range(10 - len(avg_list))]
		df_length = len(new_df)
		new_df.loc[df_length] = avg_list
	new_df['time'] = time_list
	res_today_time = time_list[-1]
	new_df['real'] = real_list
	for x in range(2, new_df.shape[0]):
		for y in range(0, 10):
			each = new_df.loc[x][y]
			if (each == new_df.loc[x - 2][y]) and (each == new_df.loc[x - 1][y]) and (each is not None):
				i = 0
				while new_df.loc[x + i][y] == each:
					new_df.loc[x - 2][y] = None
					new_df.loc[x - 1][y] = None
					new_df.loc[x + i][y] = None
					if x != df_length:
						i += 1
					if x + i == df_length:
						new_df.loc[x - 2][y] = None
						new_df.loc[x - 1][y] = None
						new_df.loc[x + i][y] = None
						break
	new_df['time'] = pd.to_datetime(new_df['time'], format='%Y/%m/%d %H:%M')
	new_df.set_index(["time"], inplace=True)
	new_df = new_df[~new_df.index.duplicated(keep='first')]
	new_df = new_df.astype(float)
	new_df.iloc[:, 0:10] = new_df.iloc[:, 0:10].where(
		new_df.iloc[:, 0:10].rank(axis=1, ascending=False, method='dense') > 1)
	new_df.iloc[:, 0:10] = new_df.iloc[:, 0:10].where(
		new_df.iloc[:, 0:10].rank(axis=1, ascending=True, method='dense') > 1)
	new_df['pre'] = new_df.iloc[:, 0:10].mean(axis=1)
	res_today_pre = round(new_df.tail(n=2).head(n=1)['pre'][0], 4)
	res_today_real = round(new_df.tail(n=1)['real'][0], 4)
	res_next_day_pre = round(new_df.tail(n=1)['pre'][0], 4)
	new_df['pre'] = new_df.iloc[:, 0:10].mean(axis=1).shift(1)
	last2_row = new_df.tail(n=2).head(n=1).iloc[:, 0:10]
	last1_row = new_df.tail(n=1).iloc[:, 0:10]
	res = 0
	for i in range(0, last2_row.shape[0]):
		for j in range(0, last2_row.shape[1]):
			diff = last1_row.iloc[i][j] - last2_row.iloc[i][j]
			if diff > 0:
				res += 1
			if diff < 0:
				res -= 1
	if res < 0:
		res_text = '预测下个交易日会跌'
	if res > 0:
		res_text = '预测下个交易日会涨'
	if res == 0:
		res_text = '预测下个交易日平稳'
	return res_today_time, res_today_pre, res_today_real, res_next_day_pre, new_df, res_text


def wx_warning():
	res_today_time, res_today_pre, res_today_real, res_next_day_pre, new_df, res_text = data_analysis()
	webhook = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=XXXXXXXXXXXXXXXXXXXXXXXXXX" # webhook地址
	header = {
		'Content-Type': "application/json"
	}
	body = {
		"msgtype": "markdown",
		"markdown": {
			"content":
				'本日时间:' + str(res_today_time) + ';   ' +
				'本日预测值:' + str(res_today_pre) + ';   ' +
				'本日实际值:' + str(res_today_real) + ';   ' +
				str(res_text) + ';   ' +
				'下一个交易日预测值（趋势结果优先）:' + str(res_next_day_pre)
		}
	}
	# resp = requests.post(webhook, headers=header, data=json.dumps(body), proxies=proxies)
	resp = requests.post(webhook, headers=header, data=json.dumps(body))



def data2dic():
	res_today_time, res_today_pre, res_today_real, res_next_day_pre, new_df, res_text = data_analysis()
	send_df = new_df.reset_index()
	send_df['pre'] = send_df['pre'].fillna(send_df.loc[0]['real'])
	send_df['date'] = send_df['time'].apply(lambda x: str(x)[0:10])
	send_df['diff'] = send_df['pre'] - send_df['real']
	send_df['diff'] = send_df['diff'].astype(float).apply(lambda x: '%.4f' % x)
	send_df['pre'] = send_df['pre'].astype(float).apply(lambda x: '%.4f' % x)
	send_df['real'] = send_df['real'].astype(float).apply(lambda x: '%.4f' % x)
	_date = send_df['date'].to_list()
	_pre = send_df['pre'].to_list()
	_real = send_df['real'].to_list()
	_diff = send_df['diff'].to_list()
	res_dic = {
		'date': _date,
		'real': _real,
		'pre': _pre,
		'diff': _diff
	}
	return res_dic