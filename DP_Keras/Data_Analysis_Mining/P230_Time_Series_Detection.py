import pandas as pd
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA


# 平稳性检验
def stationarityTest():
    """
    为了确定原始数据序列中没有随机趋势或确定趋势，
    需要对数据进行平稳性检验，否则将会产生“伪回归”现象。
    本案例采用单位根检验（ADF）的方法或者时序图的方法进行平稳性检验。
    :return:
    """
    # 参数初始化
    discfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\discdata_processed.xls"
    data = pd.read_excel(discfile)
    # 去除最后5个数据
    predictnum = 5
    data = data.iloc[:len(data) - predictnum]

    # 平稳性检测
    diff = 0
    adf = ADF(data["CWXT_DB:184:D:\\"])
    while adf[1] > 0.05:  # adf[1]为p值，p小于0.05认为是平稳的
        diff = diff + 1
        adf = ADF(data["CWXT_DB:184:D:\\"].diff(diff).dropna())

    print(u"原始序列经过%s阶差分后归于平稳，p值为%s" % (diff, adf[1]))


# 白噪声检验
def whitenoiseTest():
    """
    为了验证序列中有用的信息是否已被提取完毕，需要对噪声进行白噪声检验。
    如果序列检验为白噪声序列，就说明序列中的有用信息已被提取完毕，剩下的
    全是随机扰动，无法进行预测和使用。本案例使用LB统计量的方法进行白噪声检验
    :return:
    """
    # 参数初始化
    discfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\discdata_processed.xls"

    data = pd.read_excel(discfile)
    data = data.iloc[:len(data) - 5]  # 不适用最后五个数据

    # 白噪声检测
    [[lb], [p]] = acorr_ljungbox(data["CWXT_DB:184:D:\\"], lags=1)
    if p < 0.05:
        print(u"原始序列为非白噪声序列，对应的p值为：%s" % p)
    else:
        print(u"原始序列为白噪声序列，对应的p值为：%s" % p)

    [[lb], [p]] = acorr_ljungbox(
        data["CWXT_DB:184:D:\\"].diff().dropna(), lags=1)

    if p < 0.05:
        print(u"一阶差分序列为非白噪声序列，对应的p值为：%s" % p)
    else:
        print(u"一阶差分序列为白噪声序列，对应的p值为：%s" % p)
    print(lb)


# 模型识别  得到模型参数
def findOptimalpq():
    """
    采用极大似然比方法进行模型的参数估计，估计各个参数的值。
    然后针对各个不同模型，采用BIC信息准则对模型进行定阶，确
    定p，q参数，从而选择最优模型。
    :return:
    """
    discfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\discdata_processed.xls"

    data = pd.read_excel(discfile, index_col="COLLECTTIME")

    data = data.iloc[:len(data) - 5]  # 不使用最后五个数据
    xdata = data["CWXT_DB:184:D:\\"]

    # 定阶
    pmax = int(len(xdata) / 10)  # 一般阶数不超过length/10
    qmax = int(len(xdata) / 10)
    # 定义bic矩阵
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:  # 存在部分报错，所以用try来跳过报错
                tmp.append(ARIMA(xdata, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    print(bic_matrix)
    # 找出最小值
    p, q = bic_matrix.stack().idxmin()
    print(u"BIC最小的p值和q值为：%s、%s" % (p, q))


# 模型检验 与    预测
def arimaModelCheck():
    """
    模型确定后，检验其残差序列是否为白噪声。如果不是白噪声，
    说明残差中还存在有用的信息，需要修改模型或者进一步提取。
    本案例所确定的ARIMA（0,1,1）模型成功通过检验。
    :return:
    """
    discfile = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\discdata_processed.xls"
    # 残差延迟个数
    lagnum = 12

    data = pd.read_excel(discfile, index_col="COLLECTTIME")
    xtest_value = data['CWXT_DB:184:D:\\'][-5:]  # 测试集
    data = data.iloc[:len(data) - 5]  # 不使用最后五个
    xdata = data["CWXT_DB:184:D:\\"]

    # 训练模型并预测，计算残差
    arima = ARIMA(xdata, (0, 1, 1)).fit()  # 建立并训练模型
    xdata_pred = arima.predict(typ="levels")  # 预测
    pred_error = (xdata_pred - xdata).dropna()  # 计算残差

    lb, p = acorr_ljungbox(pred_error, lags=lagnum)
    h = (p < 0.05).sum()  # p值小于0.05，认为是白噪声
    if h > 0:
        print(u"模型ARIMA（0,1,1)不符合白噪声检验")
    else:
        print(u"模型ARIMA（0,1,1)符合白噪声检验")
    print(u"lb值为：%s" % lb)

    # 模型预测
    forecast_values, forecasts_standard_error, forecast_confidence_interval = arima.forecast(5)
    print(u"未来五天预测结果：%s" % forecast_values)
    predictdata = pd.DataFrame(xtest_value)
    predictdata.insert(0, 'CWXT_DB:184:D:\\_predict', forecast_values)
    # predictdata.rename(columns={'CWXT_DB:184:D:\\': u'实际值', 'CWXT_DB:184:D:\_predict': u'预测值'}, inplace=True)
    predictdata.rename(columns={'CWXT_DB:184:D:\_predict': u'预测值', 'CWXT_DB:184:D:\\': u'实际值'}, inplace=True)
    result_d = predictdata.applymap(lambda x: '%.2f' % x)  # 将表格中各个浮点值都格式化
    result_d.to_excel('G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\predictdata.xls')


# 误差计算
def calErrors():
    """
    为了评价时序预测模型效果的好坏，本案例采用3个衡量模型预测精度的统计量指标：
    平均绝对误差、均方根误差和平均绝对百分误差。结合业务分析，误差阈值设定为1.5。
    :return:
    """
    file = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\predictdata.xls"
    data = pd.read_excel(file,index_col='COLLECTTIME')
    data=data.applymap(lambda x: x/(10**6)) # 单位换算成GB
    print(data)

    # 计算误差
    abs_ = (data[u"预测值"] - data[u"实际值"]).abs()
    mae_ = abs_.mean()
    rmse_ = ((abs_ ** 2).mean()) ** 0.5
    mape_ = (abs_ / data[u"实际值"]).mean()


    errors = 1.5
    print('误差阈值为%s' % errors)

    if (mae_ < errors) & (rmse_ < errors) & (mape_ < errors):
        print(u'平均绝对误差为：%.4f, \n均方根误差为：%.4f, \n平均绝对百分误差为：%.4f' % (mae_, rmse_, mape_))
        print('误差检验通过！')

    else:
        print('误差检验不通过！')

def draw_view():
    """
    数据可视化
    :return:
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    inputfile = 'G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\discdata_processed.xls'

    data = pd.read_excel(inputfile)  # 原始数据
    train_data = data.iloc[:len(data) - 5]  # 训练数据

    inputfile2 = "G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\predictdata.xls"
    result = pd.read_excel(inputfile2)  # 预测数据本文包含了预测值与实际值

    plt.rc('figure', figsize=(9, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig = plt.figure()
    fig.set(alpha=0.2)  # 设置图标透明度

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.grid(axis='y', linestyle='--')  # 设置背景分布
    ax1.grid(axis='x', linestyle='-.')

    ax1.set_title(u"D盘空间时序预测图")
    ax1.set(xlabel=u'日期', ylabel=u'磁盘使用大小')

    # 图上时间间隔显示为7天
    ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 32), interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # 设置显示格式
    plt.subplots_adjust(bottom=0.13, top=0.95)  #

    # 添加显示训练数据 对应train_data['COLLECTTIME'] x轴 ,train_data['CWXT_DB:184:D:\\'] y轴
    ax1.plot(train_data['COLLECTTIME'], train_data['CWXT_DB:184:D:\\'], 'co--', label='CWXT_DB:184:D:\\')
    # 添加显示实际数据 具体同上
    ax1.plot(result['COLLECTTIME'], result[u'实际值'], 'm+--', label='实际值')
    ax1.plot(result['COLLECTTIME'], result[u'预测值'], 'y*-', label='预测值')

    ax1.legend()  # 显示图中label标签
    fig.autofmt_xdate()  # 自动根据标签长度进行旋转

    plt.show()


if __name__ == '__main__':
    # stationarityTest()
    # whitenoiseTest()
    # findOptimalpq()
    # arimaModelCheck()
    # calErrors()
    draw_view()
    pass
