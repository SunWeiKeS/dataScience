import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from sklearn.linear_model import Lasso


# 自定义的灰度预测函数
def GM11(x0):
    # 1-AGO序列, 累计求和
    x1 = np.cumsum(x0)
    # 紧邻均值（ＭＥＡＮ）生成序列
    z1 = (x1[:-1] + x1[1:]) / 2.0
    z1 = z1.reshape(len(z1), 1)
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Yn = x0[1:].reshape((len(x0) - 1, 1))
    # 矩阵计算，计算参数
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)
    # 还原值

    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))

    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) <
               0.6745 * x0.std()).sum() / len(x0)
    # 灰度预测函数、a、b、首项、方差比、小残差概率

    return f, a, b, x0[0], C, P

# 读取文件提取基本信息
def Orignal_Information():
    inputfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\data1.csv"
    data = pd.read_csv(inputfile)  # 读取数据

    """
    原始方法，替代方法可以使用describe()方法，然后进行筛选
    r = [data.min(), data.max(), data.mean(), data.std()]   # 依次计算最小值、最大值、均值、标准差
    r = pd.DataFrame(r, index=["Min", "Max", "Mean", "STD"]).T  # 计算相关系数矩阵
    """
    r = pd.DataFrame(data.describe()).T
    r_information = np.round(r, 2)  # 保留两位小数
    print(r_information)

    # 计算相关系数矩阵，并保留两位小数
    """
    相关系数可以用来描述定量和变量之间的关系，初步判断因变量与解释变量之间是否具有线性相关性
    """
    pearson_information = np.round(data.corr(method="pearson"), 2)
    print(pearson_information)

    # Adaptive-Lasso变量选择
    """
    原代码使用的是AdaptiveLasso，现更新为Lasso
    参数也由gamma变为tol（有待验证）
    """
    model = Lasso(tol=1)
    data_range = 13  # 数据有13维
    model.fit(data.iloc[:, 0:data_range], data["y"])
    # 各个特征的系数

    print(model.coef_)

# 地方财政收入灰色预测
def huise():
    """
    year： 开始年份
    feature_lst: 特征列
    roundnum： 四舍五入保留的位数
    """
    inputfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\data1.csv"
    outputfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\data1_GM11.xls"
    startyear = 1994
    feature_lst = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
    roundnum = 2

    data = pd.read_csv(inputfile)
    data.index = range(startyear, 2014)

    data.loc[2014] = None
    data.loc[2015] = None
    for i in feature_lst:
        f = GM11(data[i][list(range(startyear, 2014))].as_matrix())[0]
        # 2014年预测结果
        data[i][2014] = f(len(data) - 1)
        # 2015年预测结果
        data[i][2015] = f(len(data))
        data[i] = data[i].round(roundnum)

    print(data[feature_lst + ["y"]])
    data[feature_lst + ["y"]].to_excel(outputfile)

# 地方财政收入神经网络预测模型
def yuce():
    """
    feature_lst: 特征列
    input_dim、units: 表示训练模型层数和神经元个数
    roundnum: 四舍五入
    """

    # 灰色预测后保存的路径
    inputfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\data1_GM11.xls"
    # 神经网络预测后保存的结果
    outputfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\revenue.xls"
    # 模型保存路径
    modelfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\1-net.model"

    data = pd.read_excel(inputfile, index_col=0)
    feature = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']  # 特征所在列

    data_train = data.loc[range(1994, 2014)].copy()  # 取2014年前的数据建模
    data_mean = data_train.mean()
    data_std = data_train.std()
    data_train = (data_train - data_mean) / data_std  # 数据标准化
    x_train = data_train[feature].values  # 特征数据
    y_train = data_train['y'].values  # 标签数据

    model = Sequential()  # 建立模型
    model.add(Dense(input_dim=6, units=12))
    model.add(Activation("relu"))  # 用relu函数作为激活函数，能够大幅提供准确度
    model.add(Dense(input_dim=12, units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型
    model.fit(x_train, y_train, nb_epoch=10000, batch_size=16)  # 训练模型，学习一万次
    model.save_weights(modelfile)  # 保存模型参数

    # 预测，并还原结果。
    x = ((data[feature] - data_mean[feature]) / data_std[feature]).values
    data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
    data.to_excel(outputfile)

    data[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])
    plt.show()


#  政府性基金收入灰色预测
def huiseJiJin():
    x0 = np.array([3152063, 2213050, 4050122,
                   5265142, 5556619, 4772843, 9463330])
    f, a, b, x00, C, P = GM11(x0)
    print(a, b, x00, C, P)
    print(u'2014年、2015年的预测结果分别为：\n%0.2f万元和%0.2f万元' % (f(8), f(9)))
    print(u'后验差比值为：%0.4f' % C)
    p = pd.DataFrame(x0, columns=["y"], index=range(2007, 2014))
    p.loc[2014] = None
    p.loc[2015] = None
    p["y_pred"] = [f(i) for i in range(1, 10)]
    p["y_pred"] = p["y_pred"].round(2)
    p.index = pd.to_datetime(p.index, format="%Y")

    p.plot(style=["b-o", "r-*"], xticks=p.index)
    plt.show()
if __name__ == '__main__':
    # Orignal_Information()
    # huise()
    # yuce()
    huiseJiJin()
    pass
