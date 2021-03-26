import pandas as pd

dl={'城市':['北京','上海','广州','深圳','沈阳'],
    '环比':[101.5,101.2,101.3,102.0,100.1],
    '同比':[120.7,127.3,119.4,140.9,101.4],
    '定基':[121.4,127.8,120.0,145.5,101.6]}
d=pd.DataFrame(dl,index=['c1','c2','c3','c4','c5'])
d=d.reindex(index=['c5','c4','c3','c2','c1'])
d=d.reindex(columns=['城市','同比','环比','定基'])
print(d)

'''
索引类型
Series和DataFrame的索引是Index类型
Index对象是不可修改类型

==============================================================================
常用方法：
方法                                        说明
.append(idx)                  连接另一个Index对象，产生新的Index对象
.diff(idx)                    计算差集，产生新的Index对象
.intersection(idx)            计算交集
.union(idx)                   计算并集
.delete(loc)                  删除loc位置处的元素
.insert(loc,e)                在loc位置增加一个元素e
'''

nc=d.columns.delete(2)
ni=d.index.insert(5,'c0')
#nd=d.reindex(index=ni,columns=nc,method='ffill')
#print(nd)

#==============================================================================
'''
.drop()能够删除Series和DataFrame指定行或列索引
'''
a=pd.Series([9,8,7,6],index=['a','b','c','d'])
print(a)
a=a.drop(['b','c'])
print(a)
print(d)
print('====================================')
temp=d.drop('c5')
print(temp)
print('====================================')
temp=d.drop('同比',axis=1) #默认axis为0
print(temp)














