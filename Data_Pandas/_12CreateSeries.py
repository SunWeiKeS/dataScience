# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

'''
Series类型可以由如下类型创建：
• Python列表，        index与列表元素个数一致
• 标量值，            index表达Series类型的尺寸
• Python字典，        键值对中的“键”是索引，index从字典中进行选择操作
• ndarray，          索引和数据都可以通过ndarray类型创建
• 其他函数，          range()函数等
'''
#从标量值创建
s=pd.Series(25,index=['a','b','c'])  #不能省略 index
print(s)

#从字典类型创建
d=pd.Series({'a':9,'b':8,'c':7})
print(d)
e=pd.Series({'a':9,'b':8,'c':7,'e':5},index=['c','a','b','d'])
print(e)


#从ndarray类型创建
n=pd.Series(np.arange(5))
print(n)

m=pd.Series(np.arange(5),index=np.arange(9,4,-1)) # 可以数据少，但是索引不能少
print(m)
