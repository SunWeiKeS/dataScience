from scipy.interpolate import lagrange
import pandas as pd

inputfile = 'G:\\# Project\\数据集\\UsingDataSet\\Python数据分析与挖掘\\missing_data.xls'
outputfile = 'missing_data_processed.csv'

data = pd.read_excel(inputfile, header=None)  # 读入数据


# 自定义列向量插值函数
# s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
    y = y[y.notnull()]  # 剔除空值
    return lagrange(y.index, list(y))(n)  # 插值并返回插值结果


# 逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j] = ployinterp_column(data[i], j)

data.to_csv(outputfile, header=None, index=False)
