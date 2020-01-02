import numpy as np
import torch
import xlrd
import xlwt
from flask import render_template, redirect, url_for, abort, flash, request, \
    current_app, jsonify, send_from_directory
from flask_login import login_required, current_user
from sklearn.svm import SVC
from werkzeug.utils import secure_filename

from app.utils import check_file_type
from machine_learning.linearRegression import linearRegression
from machine_learning.logisticregression import logsticRegression
from machine_learning.model_called import model_called
from . import main
from .forms import EditProfileForm, EditProfileAdminForm, TaskForm, ModelDeployForm, \
    process2_normalizedForm, \
    ModelTrainForm_2, ModelTrainForm_3, ModelTrainForm_4, ModelTrainForm_5, ModelTrainForm_6, ModelTrainForm_7, \
    ModelTrainForm_8, ModelTrainForm_9, ModelTrainForm_10, ModelForm_0, ModelForm_1, ModelForm_2, ModeltypeForm, \
    ModeltypeForm_1, ModeltypeForm_0, ModeltypeForm_2, ModelTrainForm_0, ModelForm_3, process2_normalizedForm2, \
    process2_normalizedForm3, confirmed_databsae_col_nameForm, databaseForm
from .. import db
from ..models import Permission, Role, User, Task
from ..decorators import admin_required

model_name = ['线性回归', '支持向量机', 'K-means算法', '逻辑回归']
lossFunction = ['最小二乘：MSEloss', '交叉熵损失：CrossEntropyLoss']
optimizerFunction = ['随机梯度下降: SGD']
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
process2_name = ['最大-最小归一化', 'Z-Score标准化', '对数变换']
algorithm = ['auto', 'full', 'elkan']
modeltype = ['回归模型', '分类模型', '聚类模型']


@main.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@main.route('/create_task', methods=['GET', 'POST'])
def create_task():
    form = TaskForm()
    if current_user.can(Permission.WRITE) and form.validate_on_submit():
        task = Task(name=form.name.data,
                    description=form.description.data,
                    author=current_user._get_current_object())
        db.session.add(task)
        db.session.commit()
        flash('创建机器学习任务成功！')
        return redirect(url_for('.create_task'))
    return render_template('create_task.html', form=form)


@main.route('/user/<username>')
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    tasks = user.task.order_by(Task.timestamp.desc()).all()
    return render_template('user.html', user=user, tasks=tasks)


@main.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.name = form.name.data
        current_user.location = form.location.data
        current_user.about_me = form.about_me.data
        db.session.add(current_user._get_current_object())
        db.session.commit()
        flash('你的信息已经更新.')
        return redirect(url_for('.user', username=current_user.username))
    form.name.data = current_user.name
    form.location.data = current_user.location
    form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', form=form)


@main.route('/edit-profile/<int:id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_profile_admin(id):
    user = User.query.get_or_404(id)
    form = EditProfileAdminForm(user=user)
    if form.validate_on_submit():
        user.email = form.email.data
        user.username = form.username.data
        user.confirmed = form.confirmed.data
        user.role = Role.query.get(form.role.data)
        user.name = form.name.data
        user.location = form.location.data
        user.about_me = form.about_me.data
        db.session.add(user)
        db.session.commit()
        flash('信息已经更新.')
        return redirect(url_for('.user', username=user.username))
    form.email.data = user.email
    form.username.data = user.username
    form.confirmed.data = user.confirmed
    form.role.data = user.role_id
    form.name.data = user.name
    form.location.data = user.location
    form.about_me.data = user.about_me
    return render_template('edit_profile.html', form=form, user=user)


@main.route('/delete_task/<int:id>')
def delete_task(id):
    task = Task.query.get_or_404(id)

    db.session.delete(task)
    db.session.commit()
    flash('机器学习任务已删除.')
    return render_template('index.html')


@main.route('/task_overview/<int:id>')
def task_overview(id):
    task = Task.query.get_or_404(id)

    return render_template('task_overview.html', task=task)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['xls', 'txt'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@main.route('/process1/<int:id>', methods=['GET', 'POST'])
def process1(id):
    task = Task.query.get_or_404(id)
    form = confirmed_databsae_col_nameForm()
    file_contents = []
    file_name = []
    if not task.confirmed_databsae:
        if form.validate_on_submit():
            task.confirmed_databsae_col_name = True
            task.confirmed_databsae = True
            db.session.add(task)
            db.session.commit()
            flash('创建数据库成功!')
            return redirect(url_for('.process1', id=task.id))
        else:
            cols_1_1 = request.form.get("1_1", '@', type=str)
            cols_2_1 = request.form.get("2_1", '@', type=str)
            cols_3_1 = request.form.get("3_1", '@', type=str)
            cols_4_1 = request.form.get("4_1", '@', type=str)
            cols_5_1 = request.form.get("5_1", '@', type=str)
            cols_6_1 = request.form.get("6_1", '@', type=str)
            cols_7_1 = request.form.get("7_1", '@', type=str)
            cols_8_1 = request.form.get("8_1", '@', type=str)
            cols_9_1 = request.form.get("9_1", '@', type=str)
            cols_10_1 = request.form.get("10_1", '@', type=str)
            if cols_1_1 != '@':
                task.cols_1 = cols_1_1
                task.cols_num = 1
            if cols_2_1 != '@':
                task.cols_2 = cols_2_1
                task.cols_num = 2
            if cols_3_1 != '@':
                task.cols_3 = cols_3_1
                task.cols_num = 3
            if cols_4_1 != '@':
                task.cols_4 = cols_4_1
                task.cols_num = 4
            if cols_5_1 != '@':
                task.cols_5 = cols_5_1
                task.cols_num = 5
            if cols_6_1 != '@':
                task.cols_6 = cols_6_1
                task.cols_num = 6
            if cols_7_1 != '@':
                task.cols_7 = cols_7_1
                task.cols_num = 7
            if cols_8_1 != '@':
                task.cols_8 = cols_8_1
                task.cols_num = 8
            if cols_9_1 != '@':
                task.cols_9 = cols_9_1
                task.cols_num = 9
            if cols_10_1 != '@':
                task.cols_10 = cols_10_1
                task.cols_num = 10
            if cols_1_1 != '@':
                task.confirmed_databsae_col_name = False
                task.confirmed_databsae = True
                db.session.add(task)
                db.session.commit()
                flash('创建数据库成功!')
                print('OKOKK')
                # return jsonify({"redirect": "/process1"})
                return redirect(url_for('.process1', id=task.id), code=302)
        return render_template('process1.html', task=task, file_contents=file_contents, file_name=file_name, result=0,
                               form=form)
    elif task.confirmed_databsae and not task.confirmed_1:
        form = databaseForm()
        if form.validate_on_submit():
            import MySQLdb
            task.ip = form.ip.data
            task.user_account = form.user_account.data
            task.user_password = form.user_password.data
            task.database = form.database.data
            task.table = form.table.data

            db1 = MySQLdb.connect(task.ip, task.user_account, task.user_password, task.database, charset='utf8')
            cursor = db1.cursor()
            sql1 = "SELECT column_name FROM information_schema.columns where table_schema = '{}' and table_name ='{}'".format(
                task.database, task.table)
            sql2 = "SELECT * FROM {}".format(task.table)
            app = current_app._get_current_object()
            data_filepath = app.config['UPLOAD_FOLDER'] + task.table + '.xls'
            task.data_filepath = data_filepath
            task.confirmed_1 = True
            book = xlwt.Workbook(encoding="utf-8", style_compression=0)
            sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
            try:
                cursor.execute(sql1)
                results = cursor.fetchall()
                cols_num = len(results)
                for i in range(cols_num):
                    if i == 0:
                        task.cols_1 = str(results[i][0])
                    elif i == 1:
                        task.cols_2 = str(results[i][0])
                    elif i == 2:
                        task.cols_3 = str(results[i][0])
                    elif i == 3:
                        task.cols_4 = str(results[i][0])
                    elif i == 4:
                        task.cols_5 = str(results[i][0])
                    elif i == 5:
                        task.cols_6 = str(results[i][0])
                    elif i == 6:
                        task.cols_7 = str(results[i][0])
                    elif i == 7:
                        task.cols_8 = str(results[i][0])
                    elif i == 8:
                        task.cols_9 = str(results[i][0])
                    elif i == 9:
                        task.cols_10 = str(results[i][0])
                cursor.execute(sql2)
                results = cursor.fetchall()
                row_num = 0
                for row in results:
                    for col in range(cols_num):
                        print("{} {} {}".format(row_num, col, row[col]))
                        sheet.write(row_num, col, float(row[col]))
                    row_num += 1
                book.save(task.data_filepath)
            except:
                print("Error: unable to fecth data")
            db1.close()
            db.session.add(task)
            db.session.commit()
            flash('连接数据库成功!')
            return redirect(url_for('.process1', id=task.id))
        if request.method == 'POST':
            import os
            app = current_app._get_current_object()
            f = request.files['file']
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                data_filepath = app.config['UPLOAD_FOLDER'] + f.filename
                task.data_filepath = data_filepath
                task.confirmed_1 = True
                if filename.split('.')[1] == "xls":
                    if task.confirmed_databsae_col_name:
                        import xlrd
                        workbook = xlrd.open_workbook(task.data_filepath)
                        worksheet = workbook.sheet_by_index(0)
                        nrows = worksheet.nrows
                        ncols = worksheet.ncols
                        task.cols_num = ncols
                        for i in range(ncols):
                            if i == 0:
                                task.cols_1 = worksheet.cell_value(0, i)
                            elif i == 1:
                                task.cols_2 = worksheet.cell_value(0, i)
                            elif i == 2:
                                task.cols_3 = worksheet.cell_value(0, i)
                            elif i == 3:
                                task.cols_4 = worksheet.cell_value(0, i)
                            elif i == 4:
                                task.cols_5 = worksheet.cell_value(0, i)
                            elif i == 5:
                                task.cols_6 = worksheet.cell_value(0, i)
                            elif i == 6:
                                task.cols_7 = worksheet.cell_value(0, i)
                            elif i == 7:
                                task.cols_8 = worksheet.cell_value(0, i)
                            elif i == 8:
                                task.cols_9 = worksheet.cell_value(0, i)
                            elif i == 9:
                                task.cols_10 = worksheet.cell_value(0, i)
                        book = xlwt.Workbook(encoding="utf-8", style_compression=0)
                        sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
                        for i in range(nrows):
                            if i == 0:
                                continue
                            for j in range(ncols):
                                sheet.write(i - 1, j, worksheet.cell_value(i, j))
                        book.save(task.data_filepath)
                else:
                    if task.confirmed_databsae_col_name:
                        f = open(task.data_filepath, 'r')
                        for line in open(task.data_filepath):
                            line = f.readline()
                            line = line.replace("\n", "").split(',')
                            for i in range(len(line)):
                                if i == 0:
                                    task.cols_1 = line[0]
                                    task.cols_num = 1
                                elif i == 1:
                                    task.cols_2 = line[1]
                                    task.cols_num = 2
                                elif i == 2:
                                    task.cols_3 = line[2]
                                    task.cols_num = 3
                                elif i == 3:
                                    task.cols_4 = line[3]
                                    task.cols_num = 4
                                elif i == 4:
                                    task.cols_5 = line[4]
                                    task.cols_num = 5
                                elif i == 5:
                                    task.cols_6 = line[5]
                                    task.cols_num = 6
                                elif i == 6:
                                    task.cols_7 = line[6]
                                    task.cols_num = 7
                                elif i == 7:
                                    task.cols_8 = line[7]
                                    task.cols_num = 8
                                elif i == 8:
                                    task.cols_9 = line[8]
                                    task.cols_num = 9
                                elif i == 9:
                                    task.cols_10 = line[9]
                                    task.cols_num = 10
                            break
                        f.close()
                        file_name = task.data_filepath.split('/')[-1]
                        f = open(task.data_filepath, 'r')
                        number = True
                        book = xlwt.Workbook(encoding="utf-8", style_compression=0)
                        sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
                        hang = 0
                        for line in open(task.data_filepath):
                            line = f.readline()
                            if number == True:
                                number = False
                                continue
                            line = line.replace("\n", "").split(',')
                            temp = list()
                            for i in range(len(line)):
                                temp.append(float(line[i]))
                                sheet.write(hang, i, float(line[i]))
                            hang += 1
                        book.save(task.data_filepath.split('.')[0] + '.xls')
                        f.close()
                    else:
                        f = open(task.data_filepath, 'r')
                        book = xlwt.Workbook(encoding="utf-8", style_compression=0)
                        sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
                        hang = 0
                        for line in open(task.data_filepath):
                            line = f.readline()
                            line = line.replace("\n", "").split(',')
                            temp = list()
                            for i in range(len(line)):
                                temp.append(float(line[i]))
                                sheet.write(hang, i, float(line[i]))
                            hang += 1
                        book.save(task.data_filepath.split('.')[0] + '.xls')
                        f.close()
                db.session.add(task)
                db.session.commit()
                return redirect(url_for('.process1', id=task.id))
            else:
                flash('上传文件格式错误，请重新上传！')
                return render_template('process1.html', task=task, file_contents=file_contents, file_name=file_name)
        return render_template('process1.html', task=task, file_contents=file_contents, file_name=file_name, form=form)
    else:
        import xlrd
        import json
        file_name = task.data_filepath.split('/')[-1]
        file_true_name = file_name.split('.')[0]
        data_filepath = check_file_type(task.data_filepath)
        workbook = xlrd.open_workbook(data_filepath)
        worksheet = workbook.sheet_by_index(0)
        nrows = worksheet.nrows
        ncols = worksheet.ncols
        dict = {}
        dict["data"] = []
        for i in range(nrows):
            dict["data"].append(worksheet.row_values(i))
        json_str = json.dumps(dict)
        app = current_app._get_current_object()
        f = open(app.config['JSON_UPLOAD_FOLDER'] + file_true_name + '.txt', 'w')
        f.write(json_str)
        f.close()
        return render_template('process1.html', task=task, ncols=ncols, file_name=file_name,
                               file_true_name=file_true_name)


@main.route('/process2/<int:id>', methods=['GET', 'POST'])
def process2(id):
    task = Task.query.get_or_404(id)
    if task.confirmed_1:
        tube_content = []
        form = process2_normalizedForm()
        form2 = process2_normalizedForm2()
        form3 = process2_normalizedForm3()
        if form.submit1.data and form.validate_on_submit():
            cols = form.cols.data
            data_filepath = check_file_type(task.data_filepath)
            workbook = xlrd.open_workbook(data_filepath)
            worksheet = workbook.sheet_by_index(0)
            nrows = worksheet.nrows
            ncols = worksheet.ncols
            book = xlwt.Workbook(encoding="utf-8", style_compression=0)
            sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
            for i in range(nrows):
                for j in range(ncols):
                    sheet.write(i, j, worksheet.cell_value(i, j))
            for i in range(len(cols)):
                temp = np.array(worksheet.col_values(int(cols[i]) - 1))
                min = np.amin(temp)
                max = np.amax(temp)
                nor = (temp - min) / (max - min)
                for j in range(0, len(nor)):
                    sheet.write(j, int(cols[i]) - 1, nor[j])
                if int(cols[i]) == 1:
                    task.process2_normalized_1 = True
                    task.process2_normalized_max_1 = max
                    task.process2_normalized_min_1 = min
                elif int(cols[i]) == 2:
                    task.process2_normalized_2 = True
                    task.process2_normalized_max_2 = max
                    task.process2_normalized_min_2 = min
                elif int(cols[i]) == 3:
                    task.process2_normalized_3 = True
                    task.process2_normalized_max_3 = max
                    task.process2_normalized_min_3 = min
                elif int(cols[i]) == 4:
                    task.process2_normalized_4 = True
                    task.process2_normalized_max_4 = max
                    task.process2_normalized_min_4 = min
                elif int(cols[i]) == 5:
                    task.process2_normalized_5 = True
                    task.process2_normalized_max_5 = max
                    task.process2_normalized_min_5 = min
                elif int(cols[i]) == 6:
                    task.process2_normalized_6 = True
                    task.process2_normalized_max_6 = max
                    task.process2_normalized_min_6 = min
                elif int(cols[i]) == 7:
                    task.process2_normalized_7 = True
                    task.process2_normalized_max_7 = max
                    task.process2_normalized_min_7 = min
                elif int(cols[i]) == 8:
                    task.process2_normalized_8 = True
                    task.process2_normalized_max_8 = max
                    task.process2_normalized_min_8 = min
                elif int(cols[i]) == 9:
                    task.process2_normalized_9 = True
                    task.process2_normalized_max_9 = max
                    task.process2_normalized_min_9 = min
            book.save(data_filepath)
            if task.process2 == None:
                task.process2 = '0'
            else:
                task.process2 = task.process2 + '0'
            db.session.add(task)
            db.session.commit()
            flash('最大-最小归一化成功！')
            return redirect(url_for('.task_overview', id=task.id))
        if form2.submit2.data and form2.validate_on_submit():
            cols = form.cols.data
            data_filepath = check_file_type(task.data_filepath)
            workbook = xlrd.open_workbook(data_filepath)
            worksheet = workbook.sheet_by_index(0)
            nrows = worksheet.nrows
            ncols = worksheet.ncols
            book = xlwt.Workbook(encoding="utf-8", style_compression=0)
            sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
            for i in range(nrows):
                for j in range(ncols):
                    sheet.write(i, j, worksheet.cell_value(i, j))
            for i in range(len(cols)):
                temp = np.array(worksheet.col_values(int(cols[i]) - 1))
                mean = np.mean(temp)
                std = np.std(temp)
                nor = (temp - mean) / std
                for j in range(0, len(nor)):
                    sheet.write(j, int(cols[i]) - 1, nor[j])
                if int(cols[i]) == 1:
                    task.process2_normalized_zscore_1 = True
                    task.process2_normalized_zscore_mean_1 = mean
                    task.process2_normalized_zscore_std_1 = std
                elif int(cols[i]) == 2:
                    task.process2_normalized_zscore_2 = True
                    task.process2_normalized_zscore_mean_2 = mean
                    task.process2_normalized_zscore_std_2 = std
                elif int(cols[i]) == 3:
                    task.process2_normalized_zscore_3 = True
                    task.process2_normalized_zscore_mean_3 = mean
                    task.process2_normalized_zscore_std_3 = std
                elif int(cols[i]) == 4:
                    task.process2_normalized_zscore_4 = True
                    task.process2_normalized_zscore_mean_4 = mean
                    task.process2_normalized_zscore_std_4 = std
                elif int(cols[i]) == 5:
                    task.process2_normalized_zscore_5 = True
                    task.process2_normalized_zscore_mean_5 = mean
                    task.process2_normalized_zscore_std_5 = std
                elif int(cols[i]) == 6:
                    task.process2_normalized_zscore_6 = True
                    task.process2_normalized_zscore_mean_6 = mean
                    task.process2_normalized_zscore_std_6 = std
                elif int(cols[i]) == 7:
                    task.process2_normalized_zscore_7 = True
                    task.process2_normalized_zscore_mean_7 = mean
                    task.process2_normalized_zscore_std_7 = std
                elif int(cols[i]) == 8:
                    task.process2_normalized_zscore_8 = True
                    task.process2_normalized_zscore_mean_8 = mean
                    task.process2_normalized_zscore_std_8 = std
                elif int(cols[i]) == 9:
                    task.process2_normalized_zscore_9 = True
                    task.process2_normalized_zscore_mean_9 = mean
                    task.process2_normalized_zscore_std_9 = std
            book.save(data_filepath)
            if task.process2 == None:
                task.process2 = '1'
            else:
                task.process2 = task.process2 + '1'
            db.session.add(task)
            db.session.commit()
            flash('Z-Score标准化成功！')
            return redirect(url_for('.task_overview', id=task.id))
        if form3.submit3.data and form3.validate_on_submit():
            cols = form.cols.data
            data_filepath = check_file_type(task.data_filepath)
            workbook = xlrd.open_workbook(data_filepath)
            worksheet = workbook.sheet_by_index(0)
            nrows = worksheet.nrows
            ncols = worksheet.ncols
            book = xlwt.Workbook(encoding="utf-8", style_compression=0)
            sheet = book.add_sheet('sheet0', cell_overwrite_ok=True)
            for i in range(nrows):
                for j in range(ncols):
                    sheet.write(i, j, worksheet.cell_value(i, j))
            for i in range(len(cols)):
                temp = np.array(worksheet.col_values(int(cols[i]) - 1))
                nor = np.log(temp + 1)
                for j in range(0, len(nor)):
                    sheet.write(j, int(cols[i]) - 1, nor[j])
                if int(cols[i]) == 1:
                    task.process2_normalized_log_1 = True
                elif int(cols[i]) == 2:
                    task.process2_normalized_log_2 = True
                elif int(cols[i]) == 3:
                    task.process2_normalized_log_3 = True
                elif int(cols[i]) == 4:
                    task.process2_normalized_log_4 = True
                elif int(cols[i]) == 5:
                    task.process2_normalized_log_5 = True
                elif int(cols[i]) == 6:
                    task.process2_normalized_log_6 = True
                elif int(cols[i]) == 7:
                    task.process2_normalized_log_7 = True
                elif int(cols[i]) == 8:
                    task.process2_normalized_log_8 = True
                elif int(cols[i]) == 9:
                    task.process2_normalized_log_9 = True
            book.save(data_filepath)
            if task.process2 == None:
                task.process2 = '2'
            else:
                task.process2 = task.process2 + '2'
            db.session.add(task)
            db.session.commit()
            flash('对数变换成功！')
            return redirect(url_for('.task_overview', id=task.id))
        if task.process2 != None:
            for i in range(len(task.process2)):
                tube_content.append(process2_name[int(task.process2[i])])
        return render_template('process2.html', task=task, form=form, form2=form2, form3=form3,
                               tube_content=tube_content)
    else:
        flash('您还没有完成数据接入任务！已自动跳转自数据接入任务！')
        return redirect(url_for('.process1', id=task.id))


@main.route('/process2_confirmed/<int:id>', methods=['GET', 'POST'])
def process2_confirmed(id):
    task = Task.query.get_or_404(id)
    task.confirmed_2 = True
    db.session.add(task)
    db.session.commit()
    return redirect(url_for('.task_overview', id=task.id))


@main.route('/process3/<int:id>', methods=['GET', 'POST'])
def process3(id):
    task = Task.query.get_or_404(id)
    if not task.confirmed_1:
        flash('您还没有完成数据接入任务！已自动跳转自数据接入任务！')
        return redirect(url_for('.process1', id=task.id))
    elif not task.confirmed_2:
        flash('您还没有完成数据预处理任务！已自动跳转自数据预处理任务！')
        return redirect(url_for('.process2', id=task.id))
    else:
        if not task.confirmed_model_type_3:
            form = ModeltypeForm()
            if form.validate_on_submit():
                task.modeltype = form.modeltype.data
                task.confirmed_model_type_3 = True
                db.session.add(task)
                db.session.commit()
                flash('选择模型类型成功！')
                return redirect(url_for('.process3', id=task.id))
            return render_template('process3.html', task=task, form=form, )
        if task.confirmed_model_type_3 and not task.confirmed_model_3:
            if task.modeltype == 0:
                form = ModeltypeForm_0()
            elif task.modeltype == 1:
                form = ModeltypeForm_1()
            elif task.modeltype == 2:
                form = ModeltypeForm_2()
            if form.validate_on_submit():
                task.model = form.model.data
                task.confirmed_model_3 = True
                db.session.add(task)
                db.session.commit()
                flash('选择模型成功！')
                return redirect(url_for('.process3', id=task.id))
            return render_template('process3.html', task=task, form=form, )
        if task.confirmed_model_type_3 and task.confirmed_model_3 and not task.confirmed_3:
            if task.model == 0:
                form = ModelForm_0()
                if form.validate_on_submit():
                    task.lossFunction = form.lossFunction.data
                    task.optimizerFunction = form.optimizerFunction.data
                    task.learningRate = form.learningRate.data
                    task.trainEpochs = form.trainEpochs.data
                    task.confirmed_3 = True
                    db.session.add(task)
                    db.session.commit()
                    flash('选择模型参数成功！')
                    return redirect(url_for('.process3', id=task.id))
            elif task.model == 1:
                form = ModelForm_1()
                if form.validate_on_submit():
                    task.Regular_coefficient = form.Regular_coefficient.data
                    task.kernel = form.kernel.data
                    task.trainEpochs = form.trainEpochs.data
                    task.Minimum_convergence_error = form.Minimum_convergence_error.data
                    task.lossFunction = 0
                    task.optimizerFunction = 0
                    task.learningRate = 0.00001
                    task.confirmed_3 = True
                    db.session.add(task)
                    db.session.commit()
                    flash('选择模型参数成功！')
                    return redirect(url_for('.process3', id=task.id))
            elif task.model == 2:
                form = ModelForm_2()
                if form.validate_on_submit():
                    task.algorithm = form.algorithm.data
                    task.n_clusters = form.n_clusters.data
                    task.n_ints = form.n_ints.data
                    task.Minimum_convergence_error = form.Minimum_convergence_error.data
                    task.Regular_coefficient = 10
                    task.kernel = 0
                    task.trainEpochs = 1000
                    task.lossFunction = 0
                    task.optimizerFunction = 0
                    task.learningRate = 0.00001
                    task.target_column = 11
                    task.confirmed_3 = True
                    db.session.add(task)
                    db.session.commit()
                    flash('选择模型参数成功！')
                    return redirect(url_for('.process3', id=task.id))
            else:
                form = ModelForm_3()
                if form.validate_on_submit():
                    task.algorithm = 0
                    task.n_clusters = 0
                    task.n_ints = 0
                    task.Minimum_convergence_error = form.Minimum_convergence_error.data
                    task.Regular_coefficient = 10
                    task.kernel = 0
                    task.trainEpochs = form.trainEpochs.data
                    task.lossFunction = form.lossFunction.data
                    task.optimizerFunction = form.optimizerFunction.data
                    task.learningRate = form.learningRate.data
                    task.confirmed_3 = True
                    db.session.add(task)
                    db.session.commit()
                    flash('选择模型参数成功！')
                    return redirect(url_for('.process3', id=task.id))
            return render_template('process3.html', task=task, form=form, )
        else:
            # return render_template('process3.html', task=task, form=form, model=model[task.model],
            #                        lossFunction=lossFunction[task.lossFunction],
            #                        optimizerFunction=optimizerFunction[task.optimizerFunction])
            return render_template('process3.html', task=task, model=model_name[task.model],
                                   lossFunction=lossFunction[task.lossFunction if task.lossFunction else 0],
                                   optimizerFunction=optimizerFunction[
                                       task.optimizerFunction if task.optimizerFunction else 0],
                                   kernel=kernel[task.kernel if task.kernel else 0],
                                   algorithm=algorithm[task.algorithm if task.algorithm else 0],
                                   modeltype=modeltype[task.modeltype if task.modeltype else 0])


@main.route('/process4/<int:id>', methods=['GET', 'POST'])
def process4(id):
    task = Task.query.get_or_404(id)
    if not task.confirmed_1:
        flash('您还没有完成数据接入任务！已自动跳转自数据接入任务！')
        return redirect(url_for('.process1', id=task.id))
    elif not task.confirmed_2:
        flash('您还没有完成数据预处理任务！已自动跳转自数据预处理任务！')
        return redirect(url_for('.process2', id=task.id))
    elif not task.confirmed_3:
        flash('您还没有完成模型选择任务！已自动跳转自模型选择任务！')
        return redirect(url_for('.process3', id=task.id))
    else:
        if task.modeltype == 2:
            form = ModelTrainForm_0()
        else:
            if task.cols_num == 2:
                form = ModelTrainForm_2()
            elif task.cols_num == 3:
                form = ModelTrainForm_3()
            elif task.cols_num == 4:
                form = ModelTrainForm_4()
            elif task.cols_num == 5:
                form = ModelTrainForm_5()
            elif task.cols_num == 6:
                form = ModelTrainForm_6()
            elif task.cols_num == 7:
                form = ModelTrainForm_7()
            elif task.cols_num == 8:
                form = ModelTrainForm_8()
            elif task.cols_num == 9:
                form = ModelTrainForm_9()
            elif task.cols_num == 10:
                form = ModelTrainForm_10()
        if form.validate_on_submit():
            if task.modeltype != 2:
                content, model_filepath = model_called(task.data_filepath, task.learningRate, task.lossFunction,
                                                       task.model,
                                                       task.optimizerFunction,
                                                       task.trainEpochs, int(form.target_column.data), task.cols_num,
                                                       task.n_clusters, task.confirmed_databsae_col_name)
            else:
                content, model_filepath = model_called(task.data_filepath, task.learningRate, task.lossFunction,
                                                       task.model,
                                                       task.optimizerFunction,
                                                       task.trainEpochs, 0, task.cols_num,
                                                       task.n_clusters, task.confirmed_databsae_col_name)
            task.confirmed_4 = True
            if task.modeltype != 2:
                task.target_column = int(form.target_column.data)
            task.train_content = content
            task.model_filepath = model_filepath
            db.session.add(task)
            db.session.commit()
            flash('训练模型成功')
            return redirect(url_for('.process4', id=task.id))
        return render_template('process4.html', task=task, form=form, content=task.train_content)


@main.route('/process5/<int:id>', methods=['GET', 'POST'])
def process5(id):
    task = Task.query.get_or_404(id)
    if not task.confirmed_1:
        flash('您还没有完成数据接入任务！已自动跳转自数据接入任务！')
        return redirect(url_for('.process1', id=task.id))
    elif not task.confirmed_2:
        flash('您还没有完成数据预处理任务！已自动跳转自数据预处理任务！')
        return redirect(url_for('.process2', id=task.id))
    elif not task.confirmed_3:
        flash('您还没有完成模型选择任务！已自动跳转自模型选择任务！')
        return redirect(url_for('.process3', id=task.id))
    elif not task.confirmed_4:
        flash('您还没有完成模型训练任务！已自动跳转自模型训练任务！')
        return redirect(url_for('.process4', id=task.id))
    else:
        form = ModelDeployForm()
        if form.validate_on_submit():
            import time
            time.sleep(5)
            flash('部署成功！请在下方链接中在线使用模型')
            task.confirmed_5 = True
            db.session.add(task)
            db.session.commit()
            return redirect(url_for('.process5', id=task.id))
        return render_template('process5.html', task=task, form=form)


@main.route('/cancel_process1/<int:id>', methods=['GET', 'POST'])
def cancel_process1(id):
    task = Task.query.get_or_404(id)
    task.confirmed_databsae = False
    task.confirmed_1 = False
    task.confirmed_2 = False
    task.confirmed_3 = False
    task.confirmed_4 = False
    task.confirmed_5 = False
    task.confirmed_model_3 = False
    task.process2 = None
    task.process2_normalized_1 = False
    task.confirmed_model_type_3 = False
    task.process2_normalized_2 = False
    task.process2_normalized_3 = False
    task.process2_normalized_4 = False
    task.process2_normalized_5 = False
    task.process2_normalized_6 = False
    task.process2_normalized_7 = False
    task.process2_normalized_8 = False
    task.process2_normalized_9 = False
    task.process2_normalized_zscore_1 = False
    task.process2_normalized_zscore_2 = False
    task.process2_normalized_zscore_3 = False
    task.process2_normalized_zscore_4 = False
    task.process2_normalized_zscore_5 = False
    task.process2_normalized_zscore_6 = False
    task.process2_normalized_zscore_7 = False
    task.process2_normalized_zscore_8 = False
    task.process2_normalized_zscore_9 = False
    db.session.add(task)
    db.session.commit()
    flash('撤销成功！')
    return redirect(url_for('.task_overview', id=task.id))


@main.route('/cancel_process2/<int:id>', methods=['GET', 'POST'])
def cancel_process2(id):
    task = Task.query.get_or_404(id)
    task.confirmed_2 = False
    task.process2 = None
    task.process2_normalized_1 = False
    task.process2_normalized_2 = False
    task.process2_normalized_3 = False
    task.process2_normalized_4 = False
    task.process2_normalized_5 = False
    task.process2_normalized_6 = False
    task.process2_normalized_7 = False
    task.process2_normalized_8 = False
    task.process2_normalized_9 = False
    task.process2_normalized_zscore_1 = False
    task.process2_normalized_zscore_2 = False
    task.process2_normalized_zscore_3 = False
    task.process2_normalized_zscore_4 = False
    task.process2_normalized_zscore_5 = False
    task.process2_normalized_zscore_6 = False
    task.process2_normalized_zscore_7 = False
    task.process2_normalized_zscore_8 = False
    task.process2_normalized_zscore_9 = False
    task.confirmed_3 = False
    task.confirmed_4 = False
    task.confirmed_5 = False
    task.confirmed_model_type_3 = False
    task.confirmed_model_3 = False
    db.session.add(task)
    db.session.commit()
    flash('撤销成功！')
    return redirect(url_for('.task_overview', id=task.id))


@main.route('/cancel_process3/<int:id>', methods=['GET', 'POST'])
def cancel_process3(id):
    task = Task.query.get_or_404(id)
    task.confirmed_3 = False
    task.confirmed_4 = False
    task.confirmed_5 = False
    task.confirmed_model_3 = False
    task.confirmed_model_type_3 = False
    db.session.add(task)
    db.session.commit()
    flash('撤销成功！')
    return redirect(url_for('.task_overview', id=task.id))


@main.route('/cancel_process4/<int:id>', methods=['GET', 'POST'])
def cancel_process4(id):
    task = Task.query.get_or_404(id)
    task.confirmed_4 = False
    task.confirmed_5 = False
    db.session.add(task)
    db.session.commit()
    flash('撤销成功！')
    return redirect(url_for('.task_overview', id=task.id))


@main.route('/cancel_process5/<int:id>', methods=['GET', 'POST'])
def cancel_process5(id):
    task = Task.query.get_or_404(id)
    task.confirmed_5 = False
    db.session.add(task)
    db.session.commit()
    flash('撤销成功！')
    return redirect(url_for('.task_overview', id=task.id))


@main.route("/predict", methods=["POST"])
def predict():
    model = request.form.get("model", 0, type=int)
    cols_num = request.form.get("cols_num", 0, type=int)
    target_column = request.form.get("target_column", 0, type=int)
    confirmed_databsae_col_name = request.form.get("confirmed_databsae_col_name", 0, type=int)
    process2_normalized_1 = request.form.get("process2_normalized_1", 0, type=int)
    process2_normalized_max_1 = request.form.get("process2_normalized_max_1", 0, type=float)
    process2_normalized_min_1 = request.form.get("process2_normalized_min_1", 0, type=float)
    process2_normalized_2 = request.form.get("process2_normalized_2", 0, type=int)
    process2_normalized_max_2 = request.form.get("process2_normalized_max_2", 0, type=float)
    process2_normalized_min_2 = request.form.get("process2_normalized_min_2", 0, type=float)
    process2_normalized_3 = request.form.get("process2_normalized_3", 0, type=int)
    process2_normalized_max_3 = request.form.get("process2_normalized_max_3", 0, type=float)
    process2_normalized_min_3 = request.form.get("process2_normalized_min_3", 0, type=float)
    process2_normalized_4 = request.form.get("process2_normalized_4", 0, type=int)
    process2_normalized_max_4 = request.form.get("process2_normalized_max_4", 0, type=float)
    process2_normalized_min_4 = request.form.get("process2_normalized_min_4", 0, type=float)
    process2_normalized_5 = request.form.get("process2_normalized_5", 0, type=int)
    process2_normalized_max_5 = request.form.get("process2_normalized_max_5", 0, type=float)
    process2_normalized_min_5 = request.form.get("process2_normalized_min_5", 0, type=float)
    process2_normalized_6 = request.form.get("process2_normalized_6", 0, type=int)
    process2_normalized_max_6 = request.form.get("process2_normalized_max_6", 0, type=float)
    process2_normalized_min_6 = request.form.get("process2_normalized_min_6", 0, type=float)
    process2_normalized_7 = request.form.get("process2_normalized_7", 0, type=int)
    process2_normalized_max_7 = request.form.get("process2_normalized_max_7", 0, type=float)
    process2_normalized_min_7 = request.form.get("process2_normalized_min_7", 0, type=float)
    process2_normalized_8 = request.form.get("process2_normalized_8", 0, type=int)
    process2_normalized_max_8 = request.form.get("process2_normalized_max_8", 0, type=float)
    process2_normalized_min_8 = request.form.get("process2_normalized_min_8", 0, type=float)
    process2_normalized_9 = request.form.get("process2_normalized_9", 0, type=int)
    process2_normalized_max_9 = request.form.get("process2_normalized_max_9", 0, type=float)
    process2_normalized_min_9 = request.form.get("process2_normalized_min_9", 0, type=float)

    process2_normalized_zscore_1 = request.form.get("process2_normalized_zscore_1", 0, type=int)
    process2_normalized_zscore_mean_1 = request.form.get("process2_normalized_zscore_mean_1", 0, type=float)
    process2_normalized_zscore_std_1 = request.form.get("process2_normalized_zscore_std_1", 0, type=float)

    process2_normalized_zscore_2 = request.form.get("process2_normalized_zscore_2", 0, type=int)
    process2_normalized_zscore_mean_2 = request.form.get("process2_normalized_zscore_mean_2", 0, type=float)
    process2_normalized_zscore_std_2 = request.form.get("process2_normalized_zscore_std_2", 0, type=float)

    process2_normalized_zscore_3 = request.form.get("process2_normalized_zscore_3", 0, type=int)
    process2_normalized_zscore_mean_3 = request.form.get("process2_normalized_zscore_mean_3", 0, type=float)
    process2_normalized_zscore_std_3 = request.form.get("process2_normalized_zscore_std_3", 0, type=float)

    process2_normalized_zscore_4 = request.form.get("process2_normalized_zscore_4", 0, type=int)
    process2_normalized_zscore_mean_4 = request.form.get("process2_normalized_zscore_mean_4", 0, type=float)
    process2_normalized_zscore_std_4 = request.form.get("process2_normalized_zscore_std_4", 0, type=float)

    process2_normalized_zscore_5 = request.form.get("process2_normalized_zscore_5", 0, type=int)
    process2_normalized_zscore_mean_5 = request.form.get("process2_normalized_zscore_mean_5", 0, type=float)
    process2_normalized_zscore_std_5 = request.form.get("process2_normalized_zscore_std_5", 0, type=float)

    process2_normalized_zscore_6 = request.form.get("process2_normalized_zscore_6", 0, type=int)
    process2_normalized_zscore_mean_6 = request.form.get("process2_normalized_zscore_mean_6", 0, type=float)
    process2_normalized_zscore_std_6 = request.form.get("process2_normalized_zscore_std_6", 0, type=float)

    process2_normalized_zscore_7 = request.form.get("process2_normalized_zscore_7", 0, type=int)
    process2_normalized_zscore_mean_7 = request.form.get("process2_normalized_zscore_mean_7", 0, type=float)
    process2_normalized_zscore_std_7 = request.form.get("process2_normalized_zscore_std_7", 0, type=float)

    process2_normalized_zscore_8 = request.form.get("process2_normalized_zscore_8", 0, type=int)
    process2_normalized_zscore_mean_8 = request.form.get("process2_normalized_zscore_mean_8", 0, type=float)
    process2_normalized_zscore_std_8 = request.form.get("process2_normalized_zscore_std_8", 0, type=float)

    process2_normalized_zscore_9 = request.form.get("process2_normalized_zscore_9", 0, type=int)
    process2_normalized_zscore_mean_9 = request.form.get("process2_normalized_zscore_mean_9", 0, type=float)
    process2_normalized_zscore_std_9 = request.form.get("process2_normalized_zscore_std_9", 0, type=float)

    process2_normalized_log_1 = request.form.get("process2_normalized_log_1", 0, type=int)
    process2_normalized_log_2 = request.form.get("process2_normalized_log_2", 0, type=int)
    process2_normalized_log_3 = request.form.get("process2_normalized_log_3", 0, type=int)
    process2_normalized_log_4 = request.form.get("process2_normalized_log_4", 0, type=int)
    process2_normalized_log_5 = request.form.get("process2_normalized_log_5", 0, type=int)
    process2_normalized_log_6 = request.form.get("process2_normalized_log_6", 0, type=int)
    process2_normalized_log_7 = request.form.get("process2_normalized_log_7", 0, type=int)
    process2_normalized_log_8 = request.form.get("process2_normalized_log_8", 0, type=int)
    process2_normalized_log_9 = request.form.get("process2_normalized_log_9", 0, type=int)

    process2 = request.form.get("process2", 0, type=str)
    data = []
    for i in range(cols_num):
        if i != target_column - 1:
            data.append(request.form.get(str(i + 1), 0, type=float))
    for i in range(len(process2)):
        if process2[i] == '0':
            if process2_normalized_1:
                data[0] = (data[0] - process2_normalized_min_1) / (
                        process2_normalized_max_1 - process2_normalized_min_1)
            if process2_normalized_2:
                data[1] = (data[1] - process2_normalized_min_2) / (
                        process2_normalized_max_2 - process2_normalized_min_2)
            if process2_normalized_3:
                data[2] = (data[2] - process2_normalized_min_3) / (
                        process2_normalized_max_3 - process2_normalized_min_3)
            if process2_normalized_4:
                data[3] = (data[3] - process2_normalized_min_4) / (
                        process2_normalized_max_4 - process2_normalized_min_4)
            if process2_normalized_5:
                data[4] = (data[4] - process2_normalized_min_5) / (
                        process2_normalized_max_5 - process2_normalized_min_5)
            if process2_normalized_6:
                data[5] = (data[5] - process2_normalized_min_6) / (
                        process2_normalized_max_6 - process2_normalized_min_6)
            if process2_normalized_7:
                data[6] = (data[6] - process2_normalized_min_7) / (
                        process2_normalized_max_7 - process2_normalized_min_7)
            if process2_normalized_8:
                data[7] = (data[7] - process2_normalized_min_8) / (
                        process2_normalized_max_8 - process2_normalized_min_8)
            if process2_normalized_9:
                data[8] = (data[8] - process2_normalized_min_9) / (
                        process2_normalized_max_9 - process2_normalized_min_9)
        if process2[i] == '1':
            if process2_normalized_zscore_1:
                data[0] = (data[0] - process2_normalized_zscore_mean_1) / process2_normalized_zscore_std_1
            if process2_normalized_zscore_2:
                data[1] = (data[1] - process2_normalized_zscore_mean_2) / process2_normalized_zscore_std_2
            if process2_normalized_zscore_3:
                data[2] = (data[2] - process2_normalized_zscore_mean_3) / process2_normalized_zscore_std_3
            if process2_normalized_zscore_4:
                data[3] = (data[3] - process2_normalized_zscore_mean_4) / process2_normalized_zscore_std_4
            if process2_normalized_zscore_5:
                data[4] = (data[4] - process2_normalized_zscore_mean_5) / process2_normalized_zscore_std_5
            if process2_normalized_zscore_6:
                data[5] = (data[5] - process2_normalized_zscore_mean_6) / process2_normalized_zscore_std_6
            if process2_normalized_zscore_7:
                data[6] = (data[6] - process2_normalized_zscore_mean_7) / process2_normalized_zscore_std_7
            if process2_normalized_zscore_8:
                data[7] = (data[7] - process2_normalized_zscore_mean_8) / process2_normalized_zscore_std_8
            if process2_normalized_zscore_9:
                data[8] = (data[8] - process2_normalized_zscore_mean_9) / process2_normalized_zscore_std_9
        if process2[i] == '2':
            if process2_normalized_log_1:
                data[0] = np.log(data[0] + 1)
            if process2_normalized_log_2:
                data[1] = np.log(data[1] + 1)
            if process2_normalized_log_3:
                data[2] = np.log(data[2] + 1)
            if process2_normalized_log_4:
                data[3] = np.log(data[3] + 1)
            if process2_normalized_log_5:
                data[4] = np.log(data[4] + 1)
            if process2_normalized_log_6:
                data[5] = np.log(data[5] + 1)
            if process2_normalized_log_7:
                data[6] = np.log(data[6] + 1)
            if process2_normalized_log_8:
                data[7] = np.log(data[7] + 1)
            if process2_normalized_log_9:
                data[8] = np.log(data[8] + 1)

    if model == 0:  # 线性回归
        b = request.form.get("b", 0, type=str)
        model = linearRegression(cols_num - 1)
        app = current_app._get_current_object()
        data_filepath = app.config['UPLOAD_FOLDER'] + '../' + str(b)
        model.load_state_dict(torch.load(data_filepath))
        x = torch.FloatTensor([[data]])
        print(x)
        y = model(x)
        return jsonify(result=y.item())
    elif model == 1:
        target_column = request.form.get("target_column", 0, type=int)
        data_filepath = request.form.get("data_filepath", 0, type=str)
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
        x = np.array(x, dtype=np.float32).T
        y = np.array(y, dtype=np.int)
        y_train = y.reshape(y.shape[0], 1)
        clf = SVC(probability=True, gamma='auto')
        clf.fit(x, y_train)
        return jsonify(result=int(clf.predict([data])))
    elif model == 2:
        target_column = 11
        data_filepath = request.form.get("data_filepath", 0, type=str)
        data_filepath = check_file_type(data_filepath)
        workbook = xlrd.open_workbook(data_filepath)
        worksheet = workbook.sheet_by_index(0)
        ncols = worksheet.ncols
        x = []
        for i in range(ncols):
            x.append(worksheet.col_values(i))
        if confirmed_databsae_col_name:
            x = np.delete(x, 0, 1)
        x = np.array(x, dtype=np.float32).T
        from sklearn.cluster import KMeans
        Kmean = KMeans(n_clusters=2)
        Kmean.fit(x)
        result = int(Kmean.predict([data])[0])
        print(result)
        return jsonify(result=result)
    else:
        b = request.form.get("b", 0, type=str)
        model = logsticRegression(cols_num - 1, 2)
        app = current_app._get_current_object()
        data_filepath = app.config['UPLOAD_FOLDER'] + '../' + str(b)
        model.load_state_dict(torch.load(data_filepath))
        x = torch.FloatTensor([data])
        # print(x)
        y = model(x)
        print(y)
        _, pred = torch.max(y, 1)
        print(pred)
        return jsonify(result=pred.item())


@main.route('/model/<int:id>', methods=['GET', 'POST'])
def model(id):
    task = Task.query.get_or_404(id)
    user = User.query.filter_by(id=task.author_id).first_or_404()
    return render_template('model.html', id=id, task=task, user=user, model_name=model_name[task.model])


@main.route('/downloads_file/<int:id>', methods=['GET', 'POST'])
@login_required
def downloads_file(id):
    task = Task.query.get_or_404(id)
    directory = current_app.config['UPLOAD_FOLDER']
    filename = task.data_filepath.split('/')[-1]
    return send_from_directory(directory, filename, as_attachment=True)
    # return render_template('process1.html', id=task.id)
