from app.utils import check_file_type
from machine_learning.k_means import k_means
from machine_learning.logisticregression import logisticregression
from machine_learning.svm import svm
from machine_learning.linearRegression import linear_regression


def model_called(data_filepath, learning_rate, lossfunciton, model, optimizer_function, train_epochs, target_column,
                 cols_num, n_clusters, confirmed_databsae_col_name):
    argument = str(model) + str(lossfunciton) + str(optimizer_function)
    print(argument)
    switcher = {
        '000': linear_regression,
        '100': svm,
        '200': k_means,
        '300': logisticregression,
        '310': logisticregression
    }
    func = switcher.get(argument, lambda: "Invalid month")
    import os
    file_list = os.listdir('./machine_learning')
    number = [0]
    for file in file_list:
        if file.endswith('.pth'):
            number.append(int(file[:-4]))
    data_filepath = check_file_type(data_filepath)
    return func(data_filepath, learning_rate, train_epochs, max(number) + 1, target_column, cols_num, n_clusters,
                confirmed_databsae_col_name)

# import os
# file_list = os.listdir('.')
# number = []
# for file in file_list:
#     if file.endswith('.pth'):
#         print(file)
#         number.append(int(file[:-4]))
