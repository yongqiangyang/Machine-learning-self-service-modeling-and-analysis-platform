def check_file_type(filename):
    if filename.split('.')[1] == "xls":
        return filename
    else:
        return filename.split('.')[0] + '.xls'