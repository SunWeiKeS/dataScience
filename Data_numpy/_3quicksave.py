'''
该方法需要读取时知道存入文件时数组的维度和元素类型
a.tofile()和np.fromfile()需要配合使用
可以通过元数据文件来存储额外信息
'''

'''
np.save(fname, array) 或np.savez(fname, array)
• fname : 文件名，以.npy为扩展名，压缩扩展名为.npz
• array : 数组变量

np.load(fname)
• fname : 文件名，以.npy为扩展名，压缩扩展名为.npz
'''
import numpy as np

a=np.arange(100).reshape(5,10,2)
np.save('a.npy',a)
print(a)
b=np.load('a.npy')
print(b)