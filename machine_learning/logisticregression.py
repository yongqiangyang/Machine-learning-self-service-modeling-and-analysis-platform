import random
import time

import numpy as np
import torch
import xlrd
random.seed(30)
np.random.seed(30)
torch.manual_seed(30)
# 定义 Logistic Regression 模型
from torch import nn


class logsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logsticRegression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out

def logisticregression(data_filepath, learning_rate, train_epochs, number, target_column, cols_num,n_clusters,confirmed_databsae_col_name):
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
    len = x.shape[0]
    x_train = x[0:int(len * 0.8)]
    x_test = x[int(len * 0.8) + 1:len]
    y_train = y[0:int(len * 0.8)]
    y_test = y[int(len * 0.8) + 1:len]
    # y_train = y_train.reshape(y_train.shape[0], 1)
    # y_test = y_test.reshape(y_test.shape[0], 1)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(x_train.size())
    print(y_train.size())
    print(x_test.size())
    print(y_test.size())
    print(x_train)
    print(y_train)
    model = logsticRegression(cols_num-1, 2)  # 图片大小是28x28
    # 定义loss和optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(train_epochs):
        content = content + ('*' * 10) + '\n'
        content = content + (f'epoch {epoch + 1}\n')
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        out = model(x_train)
        # print(out)
        # print(label)
        loss = criterion(out, y_train)
        running_loss = loss.item()
        _, pred = torch.max(out, 1)
        running_acc = (pred == y_train).float().mean()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        content = content + (f'Finish {epoch + 1} epoch, Loss: {running_loss:.6f}, Acc: {running_acc:.6f}\n')
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        with torch.no_grad():
            out = model(x_test)
            loss = criterion(out, y_test)
        eval_loss = loss.item()
        _, pred = torch.max(out, 1)
        eval_acc = (pred == y_test).float().mean()
        content = content + (f'Test Loss: {eval_loss:.6f}, Acc: {eval_acc:.6f}\n')
        content = content + (f'Time:{(time.time() - since):.1f} s\n')
    torch.save(model.state_dict(), (f'./machine_learning/{number}.pth'))
    model_filepath = 'machine_learning/' + f'{number}.pth'
    return content, model_filepath
