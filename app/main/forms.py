from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SelectField, \
    SubmitField, IntegerField, SelectMultipleField, PasswordField
from wtforms.validators import DataRequired, Length, Email, Regexp, Required
from wtforms import ValidationError
from flask_pagedown.fields import PageDownField
from ..models import Role, User


class NameForm(FlaskForm):
    name = StringField('你的名字?', validators=[DataRequired()])
    submit = SubmitField('提交')


class EditProfileForm(FlaskForm):
    name = StringField('真实姓名', validators=[Length(0, 64)])
    location = StringField('位置', validators=[Length(0, 64)])
    about_me = TextAreaField('关于自己')
    submit = SubmitField('提交')


class EditProfileAdminForm(FlaskForm):
    email = StringField('邮箱', validators=[DataRequired(), Length(1, 64),
                                          Email()])
    username = StringField('用户名', validators=[
        DataRequired(), Length(1, 64),
        Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
               'Usernames must have only letters, numbers, dots or '
               'underscores')])
    confirmed = BooleanField('认证')
    role = SelectField('角色', coerce=int)
    name = StringField('真实姓名', validators=[Length(0, 64)])
    location = StringField('位置', validators=[Length(0, 64)])
    about_me = TextAreaField('关于自己')
    submit = SubmitField('提交')

    def __init__(self, user, *args, **kwargs):
        super(EditProfileAdminForm, self).__init__(*args, **kwargs)
        self.role.choices = [(role.id, role.name)
                             for role in Role.query.order_by(Role.name).all()]
        self.user = user

    def validate_email(self, field):
        if field.data != self.user.email and \
                User.query.filter_by(email=field.data).first():
            raise ValidationError('邮箱已经被注册.')

    def validate_username(self, field):
        if field.data != self.user.username and \
                User.query.filter_by(username=field.data).first():
            raise ValidationError('用户名已经被使用.')


class databaseForm(FlaskForm):
    ip = StringField('IP地址', default="0.0.0.0", validators=[DataRequired()])
    user_account = StringField('用户账号', default="", validators=[DataRequired()])
    user_password = PasswordField('用户密码', default="", validators=[DataRequired()])
    database = StringField('数据库名', default="", validators=[DataRequired()])
    table = StringField('表格名', default="", validators=[DataRequired()])
    submit = SubmitField('提交')


class TaskForm(FlaskForm):
    name = StringField('任务名称', validators=[DataRequired()])
    description = TextAreaField("简单描述", validators=[DataRequired()])
    submit = SubmitField('提交')


class ModeltypeForm(FlaskForm):
    modeltype = SelectField('模型类型', validators=[Required()], choices=[('0', '回归模型'), ('1', '分类模型'), ('2', '聚类模型')])
    submit = SubmitField('提交')


class ModeltypeForm_0(FlaskForm):
    model = SelectField('模型\算法', validators=[Required()], choices=[('0', '线性回归')])
    submit = SubmitField('提交')


class ModeltypeForm_1(FlaskForm):
    model = SelectField('模型\算法', validators=[Required()], choices=[('1', '支持向量机(SVM)'), ('3', '逻辑回归')])
    submit = SubmitField('提交')


class ModeltypeForm_2(FlaskForm):
    model = SelectField('模型\算法', validators=[Required()], choices=[('2', 'k—means算法')])
    submit = SubmitField('提交')


class ModelForm_0(FlaskForm):
    lossFunction = SelectField('损失函数', validators=[Required()], choices=[('0', '最小二乘：MSEloss')])
    optimizerFunction = SelectField('优化函数', validators=[Required()], choices=[('0', '随机梯度下降: SGD')])
    learningRate = StringField('学习率', default="0.0001", validators=[DataRequired()])
    trainEpochs = IntegerField('训练迭代次数', default=1000, validators=[DataRequired()])
    regular_term = SelectField('正则项', validators=[Required()], choices=[('0', 'None'), ('1', 'L1'), ('2', 'L2')])
    Regular_coefficient = IntegerField('正则系数（若正则项为None，则无需填写）', default=10)
    Minimum_convergence_error = StringField('最小收敛误差', default="0.001", validators=[DataRequired()])
    submit = SubmitField('提交')


class ModelForm_1(FlaskForm):
    Regular_coefficient = StringField('正则系数', default=1.0)
    kernel = SelectField('内核类型', validators=[Required()],
                         choices=[('0', 'linear'), ('1', 'poly'), ('2', 'rbf'), ('3', 'sigmoid')])
    trainEpochs = IntegerField('训练迭代次数', default=1000, validators=[DataRequired()])
    Minimum_convergence_error = StringField('最小收敛误差', default="0.001", validators=[DataRequired()])
    submit = SubmitField('提交')


class ModelForm_2(FlaskForm):
    n_clusters = IntegerField('聚类数量', default=2, validators=[DataRequired()])
    Minimum_convergence_error = StringField('最小收敛误差', default="0.001", validators=[DataRequired()])
    algorithm = SelectField('算法类型', validators=[Required()],
                            choices=[('0', 'auto'), ('1', 'full'), ('2', 'elkan')])
    n_ints = IntegerField('训练重复次数', default=10, validators=[DataRequired()])
    submit = SubmitField('提交')


class ModelForm_3(FlaskForm):
    lossFunction = SelectField('损失函数', validators=[Required()],
                               choices=[('0', '最小二乘：MSEloss'), ('1', '交叉熵损失：CrossEntropyLoss')])
    trainEpochs = IntegerField('训练迭代次数', default=100, validators=[DataRequired()])
    optimizerFunction = SelectField('优化函数', validators=[Required()], choices=[('0', '随机梯度下降: SGD')])
    learningRate = StringField('学习率', default="0.001", validators=[DataRequired()])
    Minimum_convergence_error = StringField('最小收敛误差', default="0.001", validators=[DataRequired()])
    submit = SubmitField('提交')


class ModelTrainForm_0(FlaskForm):
    submit = SubmitField('开始训练')


class ModelTrainForm_2(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_3(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_4(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_5(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_6(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'),
                                         ],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_7(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'),
                                         ('7', '7')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_8(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'),
                                         ('7', '7'), ('8', '8')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_9(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'),
                                         ('7', '7'), ('8', '8'), ('9', '9')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_10(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'),
                                         ('7', '7'), ('8', '8'), ('9', '9'), ('10', '10')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class ModelTrainForm_10(FlaskForm):
    target_column = SelectField('目标列（其余列自动为训练特征列）',
                                choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'),
                                         ('7', '7'), ('8', '8'), ('9', '9'), ('10', '10')],
                                validators=[DataRequired()])
    submit = SubmitField('开始训练')


class confirmed_databsae_col_nameForm(FlaskForm):
    submit = SubmitField('从文件中读取列名')


class ModelDeployForm(FlaskForm):
    submit = SubmitField('将模型部署到生产环境中')


class process2_normalizedForm(FlaskForm):
    cols = SelectMultipleField('所选列',
                               choices=[('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8),
                                        ('9', 9)])
    submit1 = SubmitField('执行最大-最小归一化')


class process2_normalizedForm2(FlaskForm):
    cols = SelectMultipleField('所选列',
                               choices=[('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8),
                                        ('9', 9)])
    submit2 = SubmitField('执行Z-Score标准化')


class process2_normalizedForm3(FlaskForm):
    cols = SelectMultipleField('所选列',
                               choices=[('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8),
                                        ('9', 9)])
    submit3 = SubmitField('执行对数变换')
