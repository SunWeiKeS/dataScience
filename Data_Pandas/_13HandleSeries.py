import pandas as pd
import numpy as np
'''
Series是一维带“标签”数组
index_0 data_a
Series基本操作类似ndarray和字典，根据索引对齐

1 Series类型包括index和values两部分
2 Series类型的操作类似ndarray类型
3 Series类型的操作类似Python字典类型
'''

b= pd.Series([9,8,7,6],index=['a','b','c','d'])
print(b)

print(b.index) #.index 获得索引

print(b.values)# .values 获得数据


#自动索引和自定义索引并存
print(b['b'])
print(b[1])
#两套索引并存，但不能混用
print(b[['c','d',0]])

##============================================================================
'''
Series类型的操作类似ndarray类型：
• 索引方法相同，采用[]
• NumPy中运算和操作可用于Series类型
• 可以通过自定义索引的列表进行切片
• 可以通过自动索引进行切片，如果存在自定义索引，则一同被切片
'''
b=pd.Series([9,8,7,6],['a','b','c','d'])
print(b)


print(b[3])
print(b[:3])
print(b[b>b.median()])
print(np.exp(b))


##============================================================================
'''
Series类型的操作类似Python字典类型：
• 通过自定义索引访问
• 保留字in操作
• 使用.get()方法
'''
b=pd.Series([9,8,7,6,105],['a','b','c','d','f'])

print(b['b'])
print('c' in b)#判断的是索引
print(9 in b)
print(b.get('g',15))#如果存在f，返回f对应的值，如果不存在，返回100

##Series的对齐操作=============================================================
'''
Series类型在运算中会自动对齐不同索引的数据
'''
a=pd.Series([1,2,3],['c','d','e'])
b=pd.Series([9,8,7,6],['a','b','c','d'])
print(a+b)

##============================================================================
'''
Series对象和索引都可以有一个名字，存储在属性.name中
Series对象可以随时修改并即刻生效
'''
b=pd.Series([9,8,7,6],['a','b','c','d'])
b.name='Series对象'
b.index.name = '索引列'
b['a']=100
b['b','c']=66
print(b)







