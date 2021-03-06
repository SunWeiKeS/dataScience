import pandas as pd
import numpy as np
'''
数据的基本统计分析

函数
===================================================================
适用于Series和DataFrame类型

.sum()                       计算数据的总和，按0轴计算，下同
.count()                     非NaN值的数量
.mean()     .median()        计算数据的算术平均值、算术中位数
.var()      .std()           计算数据的方差、标准差
.min()      .max()           计算数据的最小值、最大值

—————————————————————————————————————————————————————————————————
.describe()                  针对0轴（各列）的统计汇              |
count  mean   std    min    25%    50%    75%    max            |
—————————————————————————————————————————————————————————————————

===================================================================
适用于Series类型

.argmin() .argmax()          计算数据最大值、最小值所在位置的索引位置（自动索引）
.idxmin() .idxmax()          计算数据最大值、最小值所在位置的索引（自定义索引）

===================================================================
'''
a=pd.Series([9,8,7,6],index=['a','b','c','d'])
print(a)

print(a.describe())
print(a.describe()['count'])
print(a.describe()['25%'])
print('===========================================================')

b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])
print(b)
print(b.describe())
print(b.describe().loc['max'])
print(b.describe()[2])











