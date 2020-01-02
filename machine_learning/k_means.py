import numpy as np
import torch
import xlrd
from sklearn import svm


def k_means(data_filepath, learning_rate, train_epochs, number, target_column, cols_num,n_clusters,confirmed_databsae_col_name):
    workbook = xlrd.open_workbook(data_filepath)
    worksheet = workbook.sheet_by_index(0)
    ncols = worksheet.ncols
    x = []
    y = []
    for i in range(ncols):
        x.append(worksheet.col_values(i))
    if confirmed_databsae_col_name:
        x = np.delete(x, 0, 1)
    x = np.array(x, dtype=np.float32).T
    from sklearn.cluster import KMeans

    Kmean = KMeans(n_clusters=n_clusters)
    Kmean.fit(x)
    return '训练成功', ''
    # print(clf.predict([[0.30769232,0.18627451,0.21686748 , 0.21686748,0.21686748,0.21686748,0.37354988, 0.26778483 ,0.25454545]]))


# data_filepath = 'D:\Asoftware\ML-self-model/file/winequality.xls'
# target_column = 10
# svm(data_filepath, 0, 0, 0, 10, 0)
