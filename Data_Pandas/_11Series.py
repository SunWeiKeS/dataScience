import pandas as pd
import numpy as np
'''
Series类型由一组数据及与之相关的数据索引组成
index_0   data_a
index_1   data_d
index_2   data_c
index_3   data_b
索引      数据
'''
#自动索引
a= pd.Series([9,8,7,6])
print(a)

# 自定义索引方式 作为第二个参数，可以省略index=
b= pd.Series([9,8,7,6],index=['a','b','c','d'])
print(b)




