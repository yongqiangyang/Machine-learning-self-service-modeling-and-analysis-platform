import numpy as np
import torch
import xlrd
from sklearn import svm
from sklearn.svm import SVC


def svm(data_filepath, learning_rate, train_epochs, number, target_column, cols_num,n_clusters,confirmed_databsae_col_name):
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
    y = np.array(y, dtype=np.int)
    y_train = y.reshape(y.shape[0], 1)
    # print(x.shape)a
    # print(y_train.shape)
    # print(x)
    # print(y_train)
    clf = SVC(probability=True, gamma='auto')
    clf.fit(x, y_train)
    return '训练成功', ''
    # print(clf.predict([[0.30769232,0.18627451,0.21686748 , 0.21686748,0.21686748,0.21686748,0.37354988, 0.26778483 ,0.25454545]]))


# data_filepath = 'D:\Asoftware\ML-self-model/file/winequality.xls'
# target_column = 10
# svm(data_filepath, 0, 0, 0, 10, 0)
