# -*- coding: utf-8 -*-
"""
统计函数
sum(a, axis=None) 根据给定轴axis计算数组a相关元素之和，axis整数或元组
mean(a, axis=None) 根据给定轴axis计算数组a相关元素的期望，axis整数或元组
average(a,axis=None,weights=None) 根据给定轴axis计算数组a相关元素的加权平均值
std(a, axis=None) 根据给定轴axis计算数组a相关元素的标准差
var(a, axis=None) 根据给定轴axis计算数组a相关元素的方差

axis=0 表示列  axis=1表示行
"""
import numpy as np
a=np.arange(15).reshape(3,5)
print(a)
print(np.sum(a))
print(np.mean(a,axis=0))
print(np.mean(a,axis=1))
print(np.average(a,axis=0,weights=[10,5,1]))
print(np.std(a))
print(np.var(a))

'''
min(a) max(a) 计算数组a中元素的最小值、最大值
argmin(a) argmax(a) 计算数组a中元素最小值、最大值的降一维后下标
unravel_index(index, shape) 根据shape将一维下标index转换成多维下标
ptp(a) 计算数组a中元素最大值与最小值的差
median(a) 计算数组a中元素的中位数（中值）
'''


'''
np.gradient(f) 计算数组f中元素的梯度，当f为多维时，返回每个维度梯度
梯度：连续值之间的变化率，即斜率
XY坐标轴连续三个X坐标对应的Y轴值：a, b, c，其中，b的梯度是： (c‐a)/2
'''
a=np.random.randint(0,20,(5))
print(a)
print(np.gradient(a))














