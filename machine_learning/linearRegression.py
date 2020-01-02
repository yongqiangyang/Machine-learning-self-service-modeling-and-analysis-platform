# encoding: utf-8
import numpy as np
import torch
import xlrd
from torch import nn


# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self, features):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(features, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


def linear_regression(data_filepath, learning_rate, train_epochs, number, target_column, cols_num, n_clusters,
                      confirmed_databsae_col_name):
    content = ''
    workbook = xlrd.open_workbook(data_filepath)
    worksheet = workbook.sheet_by_index(0)
    ncols = worksheet.ncols
    x = []
    y = []
    for i in range(ncols):
        if i == target_column - 1:
            y = worksheet.col_values(i)
        else:
            x.append(worksheet.col_values(i))
    if confirmed_databsae_col_name:
        x = np.delete(x, 0, 1)
        y = y[1:]
    x = np.array(x, dtype=np.float32).T
    y = np.array(y, dtype=np.float32)
    y_train = y.reshape(y.shape[0], 1)
    x_train = torch.from_numpy(x)
    y_train = torch.from_numpy(y_train)
    # print(x_train.size())
    # print(y_train.size())
    # print(x_train)
    # print(y_train)

    model = linearRegression(ncols - 1)
    # 定义loss和优化函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(train_epochs):
        inputs = x_train
        target = y_train

        # forward
        out = model(inputs)
        loss = criterion(out, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            content = content + (f'Epoch[{epoch + 1}/{train_epochs}], loss: {loss.item():.6f}\n')
    torch.save(model.state_dict(), (f'./machine_learning/{number}.pth'))
    model_filepath = 'machine_learning/' + f'{number}.pth'
    return content, model_filepath
    # model.eval()
    # with torch.no_grad():
    #     predict = model(x_train)
    # predict = predict.data.numpy()

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    # plt.plot(x_train.numpy(), predict, label='Fitting Line')
    # # 显示图例
    # plt.legend()
    # plt.show()
    #
    # # 保存模型

#
# data_filepath = 'D:\Asoftware\ML-self-model\ex1data1.txt'
# learning_rate = 0.01
# train_epochs = 100
# print(linear_regression(data_filepath, learning_rate, train_epochs,2))

# model = linearRegression()
# model.load_state_dict(torch.load('./1.pth'))
# a = 6.101
# a = [a]
# x = torch.tensor(a)
# y = model(x)
# print(y)
