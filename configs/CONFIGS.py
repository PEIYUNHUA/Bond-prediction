# DATA CONFIG
# INPUT IO
data_io = 'raw_data.xlsx'
today_data_io = 'today_data.xlsx'

# OUTPUT IO
feature_comb_output_io = './res/step1/feature_comb.csv'
feature_res_output_io = './res/step1/feature_res.csv'
best_feature_output_io = './res/step1/best_feature_res.csv'

parameter_res_output_io = './res/step2/parameter_res.csv'
best_parameter_output_io = './res/step2/best_parameter_res.csv'

model_step1_output_io = './models/LSTM1/res1/lstm1'
# model_step2_output_io = './models/LSTM1/res2/lstm1'

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
# feature_list = [
#                 'USA10',
#                 # 'INDU',
#                 # 'PPI',
#                 # 'PMI',
#                 # 'OECD',
#                 # 'SHERONG',
#                 # 'JIANZHU'
#                 # 'CRB'
#                 # 'XINJIAN',
#                 # 'ERSHOU',
#                 # 'R007',
#                 'USDI'
#                 # 'GOVBOND',
#                 # 'SHI',
#                 # 'CPI',
#                 # 'M1',
#                 # 'GKBOND',
#                 # 'MID',
#                 # 'GOLD',
#                 # 'SPOT',
#                 # 'IRBOND',
#                 # 'CHN5ma5', 'CHN5ma10', 'CHN5ma20',
#                 # 'CHN10ma5', 'CHN10ma10', 'CHN10ma20',
#                 # 'CHN5UB', 'CHN5LB',
#                 # 'CHN10UB', 'CHN10LB',
#                 # 'USA5ma5', 'USA5ma10', 'USA5ma20',
#                 # 'USA10ma5', 'USA10ma10', 'USA10ma20',
#                 # 'USA5UB', 'USA5LB',
#                 # 'USA10UB', 'USA10LB'
#                 ]
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
# start time
start_year = 2016

# MODEL CONFIG
# LSTM
# epochs_list = [100, 1000, 10000]
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
