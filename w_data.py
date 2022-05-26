from WindPy import w

w.start() # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒

test = w.wsd(["600010.SH", "000001.SZ"], "CLOSE", "2016-01-01", "2016-01-05", options="")

print(0)
# w.isconnected() # 判断WindPy是否已经登录成功

# w.stop() # 当需要停止WindPy时，可以使用该命令
#           # 注： w.start不重复启动，若需要改变参数，如超时时间，用户可以使用w.stop命令先停止后再启动。
#           # 退出时，会自动执行w.stop()，一般用户并不需要执行w.stop
# w.wsd（codes, fields, beginTime, endTime, options）
