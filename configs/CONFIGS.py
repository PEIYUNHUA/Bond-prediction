# DATA CONFIG
# INPUT IO
data_io = 'raw_data.xlsx'

# OUTPUT IO
feature_comb_output_io = './res/step1/feature_comb.csv'
feature_res_output_io = './res/step1/feature_res.csv'
best_feature_output_io = './res/step1/best_feature_res.csv'

parameter_res_output_io = './res/step2/parameter_res.csv'
best_parameter_output_io = './res/step2/best_parameter_res.csv'

model_step1_output_io = './output/models/best-model'
filepaths = ['./res/step1/', './res/step2/', './img/step1/', './img/step2/', './output/models']
log_io = './output/result/res.csv'


backup_io = './backups/'
model_step1_io = './output/models'
# ORIGINAL FEATURES
#     CHN1, CHN3, CHN5, CHN10,
#     USA1, USA3, USA5, USA10,
#     INDU, PPI, PMI, OECD,
#     SHERONG, JIANZHU, CRB, XINJIAN,
#     ERSHOU, R007, USDI, GOVBOND,
#     SHI, CPI, M1, GKBOND,
#     MID, GOLD, SPOT, IRBOND


# SELECTED FEATURES
target_list = [
                'date',
                'CHN10'
                ]
feature_list = [
                'USA10',
                # 'INDU',
                # 'PPI',
                # 'PMI',
                # 'OECD',
                'SHERONG',
                # 'JIANZHU'
                # 'CRB'
                # 'XINJIAN',
                # 'ERSHOU',
                'R007',
                'USDI',
                'GOVBOND',
                'SHI',
                'CPI',
                'M1',
                'GKBOND',
                'MID',
                'GOLD'
                # 'SPOT',
                # 'IRBOND',
                # 'CHN5ma5', 'CHN5ma10', 'CHN5ma20',
                # 'CHN10ma5', 'CHN10ma10', 'CHN10ma20',
                # 'CHN5UB', 'CHN5LB',
                # 'CHN10UB', 'CHN10LB',
                # 'USA5ma5', 'USA5ma10', 'USA5ma20',
                # 'USA10ma5', 'USA10ma10', 'USA10ma20',
                # 'USA5UB', 'USA5LB',
                # 'USA10UB', 'USA10LB'
                ]
index_list = [
                'S0059744',
                'S0059746',
                'S0059747',
                'S0059749',
                'G0000886',
                'G0000888',
                'G0000889',
                'G0000891',
                'M0000545',
                'M0001227',
                'M0017126',
                'G1000116',
                'M5525763',
                'S0178729',
                'S0031510',
                'S2707412',
                'S2707426',
                'M1001795',
                'M0000271',
                'M5639030',
                'M0020241',
                'M0000612',
                'M0001383',
                'M1004271',
                'M0000185',
                'S0031645',
                'M0067855',
                'M5635423',
            ]
columns_list = [
    'date',
    'CHN1', 'CHN3', 'CHN5', 'CHN10',
    'USA1', 'USA3', 'USA5', 'USA10',
    'INDU', 'PPI', 'PMI', 'OECD',
    'SHERONG', 'JIANZHU', 'CRB', 'XINJIAN',
    'ERSHOU', 'R007', 'USDI', 'GOVBOND',
    'SHI', 'CPI', 'M1', 'GKBOND',
    'MID', 'GOLD', 'SPOT', 'IRBOND'
    ]
ma_boll_list = ['CHN1', 'CHN3', 'CHN5', 'CHN10', 'USA1', 'USA3', 'USA5', 'USA10']

# start time
start_year = 2016
# data start
data_start = "2004-01-04"

# MODEL CONFIG
# LSTM
# epochs_list = [100, 1000]
epochs_list = [100, 1000, 10000]
learning_rate_list = [0.1, 0.01, 0.001]

epochs = 1000
learning_rate = 0.001
# input_size = feature_num
hidden_size = 2
num_layer = 1
num_classes = 1



# EVALUATION CONFIG
top_num = 10
# USE TOP_10 RESULT
# final_top_num = len(best_comb_df)
