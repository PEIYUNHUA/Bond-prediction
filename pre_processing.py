import pandas as pd

from configs.CONFIGS import *


def get_raw_data(data_io):
    raw_data = pd.read_excel(data_io)
    #   插值法处理na
    col_list = raw_data.columns.values[1:29]
    for i in col_list:
        raw_data[i] = raw_data[i].fillna(raw_data[i].interpolate())
    # add ma & boll
    bond_list = ['CHN1', 'CHN3', 'CHN5', 'CHN10', 'USA1', 'USA3', 'USA5', 'USA10']
    ma_boll_data = get_ma_boll(bond_list, raw_data)
    return raw_data


def get_ma_boll(bond_list, raw_data):
    # 增加三种ma
    for i in bond_list:
        raw_data[i + 'ma5'] = raw_data[i].rolling(5).mean()
        raw_data[i + 'ma10'] = raw_data[i].rolling(10).mean()
        raw_data[i + 'ma20'] = raw_data[i].rolling(20).mean()
    # 增加布林带(20d,2theta)
    for i in bond_list:
        raw_data[i + 'UB'] = raw_data[i].rolling(20).mean() + raw_data[i].rolling(20).std() * 2
        raw_data[i + 'LB'] = raw_data[i].rolling(20).mean() - raw_data[i].rolling(20).std() * 2
    return raw_data

    # ma_boll_data.to_excel('test.xlsx')
    # bond_dataset = pd.read_excel('test2.xlsx')
    # bond_dataset['MATURITY DATE'] = pd.to_datetime(bond_dataset['MATURITY DATE'], format='%d-%m-%Y %H:%M')
    # bond_dataset['year'] = bond_dataset['MATURITY DATE'].dt.year
    # bond_dataset['month'] = bond_dataset['MATURITY DATE'].dt.month
    # bond_dataset['day'] = bond_dataset['MATURITY DATE'].dt.day
    # del bond_dataset['MATURITY DATE']
    # X = bond_dataset.iloc[:, [1, 2, 3, 4, 5, 6, 8, 9, 10]].values
    # y = bond_dataset.iloc[:, 7].values
    #
    # le = LabelEncoder()
    # X[:, 0] = le.fit_transform(X[:, 0])
    #
    # # Splitting the dataset into the training set and test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # # Training the random forest regression model on the whole dataset
    # regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    # regressor.fit(X_train, y_train)
    #
    # # Predicting the test set results
    # y_pred = regressor.predict(X_test)
    # np.set_printoptions(precision=2)
    # print("hello")


#   相关性分析
# corr_pd = pd.corr(method='spearman')
# print(corr_pd)

# mm = MinMaxScaler()
# ss = StandardScaler()
# X_ss = ss.fit_transform(pd)
# y_mm = mm.fit_transform(pd)
# print("this")