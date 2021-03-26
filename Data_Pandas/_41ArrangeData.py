import pandas as pd
import numpy as np
'''
Pandas库的数据排序
3.1413   3.1404                           基本统计（含排序）
3.1401   3.1376  摘要                     分布/累计统计
3.1398   3.1349                           数据特征
               (数据形成有损特征的过程)     相关性、周期性等
一组数据                                   数据挖掘（形成知识）
表达一个或多个含义
'''

'''
=========================================================================
.sort_index()方法在指定轴上根据索引进行排序，默认升序
.sort_index(axis=0, ascending=True) Ture 升序 False逆序
=========================================================================
'''

b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])

print(b,'\n')
print(b.sort_index(),'\n') #升序
print(b.sort_index(ascending=False),'\n') #逆序

c=b.sort_index(axis=1,ascending=False)#指定轴逆序
print(c,'\n')
print(c.sort_index(),'\n')

'''
=========================================================================
.sort_values()方法在指定轴上根据数值进行排序，默认升序
Series.sort_values(axis=0, ascending=True)
DataFrame.sort_values(by, axis=0, ascending=True)
by : axis轴上的某个索引或索引列表
=========================================================================
'''
print("===============================================================")
d=b.sort_values(2,ascending=False)
print(d,'\n') #升序
#升序 spyder报错 pycharm没有报错。。很尴尬
print(d.sort_values('a',axis=1,ascending=False),'\n')

print("================================================================")

# NaN会统一放到排序末尾

a=pd.DataFrame(np.arange(12).reshape(3,4),index=['a','b','c'])
b=pd.DataFrame(np.arange(20).reshape(4,5),index=['c','a','d','b'])
c=a+b
print(c)
print(c.sort_values(2,ascending=False))
print(c.sort_values(2,ascending=True))























