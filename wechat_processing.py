import requests
import json
import datetime
from threading import Timer
import pandas as pd
from configs.CONFIGS import *



def wx_warning():
	df = pd.read_csv(log_io)
	ifnull = df.iloc[-1].notnull()
	notnull = df.iloc[-1][ifnull]
	last_real = str(notnull[-1])
	predicting_date = notnull[-3]
	avg = notnull[-5]
	text = '今日预测值为:' + avg + ';   ' + '上一个交易日时间' + predicting_date + ';   ' + '上一个交易日真实值' + last_real
	webhook = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=51b43769-5e3b-46e5-812a-c2ad2775b779" # webhook地址
	header = {
		'Content-Type': "application/json"
	}
	body = {
		"msgtype": "markdown",
		"markdown": {
			"content": text
		}
	}
	resp = requests.post(webhook, headers=header, data=json.dumps(body))
	loop_monitor()

def loop_monitor():
	t = Timer(60, wx_warning)
	t.start()