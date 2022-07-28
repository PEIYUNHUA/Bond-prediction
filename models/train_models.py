import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from itertools import combinations, combinations_with_replacement

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mpea
from sklearn.metrics import r2_score as r2

import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from models.LSTM1 import LSTM1
import pre_processing as pre
import numpy as np
import pandas as pd

import datetime
import time

import seaborn as sns
import matplotlib.pyplot as plt

from configs.CONFIGS import *

# DEVICE CONFIG
# default device gpu as premium
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train_model(input_size, train_data, test_data, epochs, learning_rate):
    # x(input) y(output)
    x_train = train_data.drop('date', axis=1)
    x_train = x_train.drop(target, axis=1)
    y_train = train_data.iloc[:, 1:2]
    x_test = test_data.drop('date', axis=1)
    x_test = x_test.drop(target, axis=1)
    y_test = test_data.iloc[:, 1:2].reset_index().drop(columns='index')

    # means and fit
    sc = StandardScaler()
    # sc = MinMaxScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    x_train_tensors = Variable(torch.Tensor(np.array(x_train_std)))
    x_test_tensors = Variable(torch.Tensor(np.array(x_test_std)))
    y_train_tensors = Variable(torch.Tensor(np.array(y_train)))
    y_test_tensors=Variable(torch.Tensor(np.array(y_test)))

    x_train_tensors_final = torch.reshape(x_train_tensors,   (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
    x_test_tensors_final = torch.reshape(x_test_tensors,  (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))


    model = LSTM1(num_classes, input_size, hidden_size, num_layer, x_train_tensors_final.shape[1]).to(device)
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        outputs = model.forward(x_train_tensors_final.to(device))
        optimizer.zero_grad()

        loss = criterion(outputs, y_train_tensors.to(device))
        loss.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    predict = model(x_test_tensors_final.to(device))
    y_test_predict = predict.data.detach().cpu().numpy()
    eva_mae = mae(y_test, y_test_predict)
    eva_mpea = mpea(y_test, y_test_predict)
    eva_r2 = r2(y_test, y_test_predict)
    return y_test, y_test_predict, eva_mae, eva_mpea, eva_r2


def train_model_savings(i, input_size, train_data, test_data, epochs, learning_rate):
    # x(input) y(output)
    x_train = train_data.drop('date', axis=1)
    x_train = x_train.drop(target, axis=1)
    y_train = train_data.iloc[:, 1:2]
    x_test = test_data.drop('date', axis=1)
    x_test = x_test.drop(target, axis=1)
    y_test = test_data.iloc[:, 1:2].reset_index().drop(columns='index')

    # means and fit
    sc = StandardScaler()
    # sc = MinMaxScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    x_train_tensors = Variable(torch.Tensor(np.array(x_train_std)))
    x_test_tensors = Variable(torch.Tensor(np.array(x_test_std)))
    y_train_tensors = Variable(torch.Tensor(np.array(y_train)))
    y_test_tensors = Variable(torch.Tensor(np.array(y_test)))

    x_train_tensors_final = torch.reshape(x_train_tensors,   (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
    x_test_tensors_final = torch.reshape(x_test_tensors,  (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

    model = LSTM1(num_classes, input_size, hidden_size, num_layer, x_train_tensors_final.shape[1]).to(device)
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        outputs = model.forward(x_train_tensors_final.to(device))
        optimizer.zero_grad()

        loss = criterion(outputs, y_train_tensors.to(device))
        loss.backward()
        optimizer.step()


    # pt/pth/pkl
    # store the learned parameters
    torch.save(model.state_dict(), model_step1_output_io + '-' + str(i+1) + '.pt')
    # store the full model
    # torch.save(model, model_step1_output_io + '-' + str(i+1) + '.pt')
    print("best_model-{} saved".format(i+1))