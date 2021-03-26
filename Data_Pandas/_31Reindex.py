import pandas as pd
'''
如何改变Series和DataFrame对象？
增加或重排：重新索引
            .reindex()能够改变或重排Series和DataFrame索引
删除：drop
'''
dl={'城市':['北京','上海','广州','深圳','沈阳'],
    '环比':[101.5,101.2,101.3,102.0,100.1],
    '同比':[120.7,127.3,119.4,140.9,101.4],
    '定基':[121.4,127.8,120.0,145.5,101.6]}
d=pd.DataFrame(dl,index=['c1','c2','c3','c4','c5'])
print(d)
print('-----------------------------------------------------')
print(d.index)
print(d.columns)
print(d.values)
print(d['同比'])
print(d.ix['c2'])
print(d['同比']['c2'])
print('-----------------------------------------------------')

'''
重新索引
'''
d=d.reindex(index=['c5','c4','c3','c2','c1'])
print(d)

d=d.reindex(columns=['城市','同比','环比','定基'])

print(d)


'''
.reindex(index=None, columns=None, …)的参数
参数                      说明
index, columns          新的行列自定义索引
fill_value              重新索引中，用于填充缺失位置的值
method                  填充方法, ffill当前值向前填充，bfill向后填充
limit                   最大填充量
copy                    默认True，生成新的对象，False时，新旧相等不复制
'''
newc=d.columns.insert(4,'新增')
newd=d.reindex(columns= newc,fill_value=200)
print(newc)
print(newd)
















