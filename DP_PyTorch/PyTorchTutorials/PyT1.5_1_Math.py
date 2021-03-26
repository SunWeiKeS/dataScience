# In[1]:
"""
PyTorch深度学习：60分钟入门(Translation) PyTorch
"""

import torch

x = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
x # 此处在notebook中输出x的值来查看具体的x内容
x.size()

#NOTE: torch.Size 事实上是一个tuple, 所以其支持相关的操作*
y = torch.rand(5, 3)

#此处 将两个同形矩阵相加有两种语法结构
x + y # 语法一
torch.add(x, y) # 语法二

# 另外输出tensor也有两种写法
result = torch.Tensor(5, 3) # 语法一
torch.add(x, y, out=result) # 语法二
y.add_(x) # 将y与x相加

# 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
# 例如：x.copy_(y), x.t_(), 这俩都会改变x的值。

#另外python中的切片操作也是支持的

# In[2]:
"""
- 100+ Tensor的操作，包括换位、索引、切片、数学运算、线性算法和随机数等等
- [torch - PyTorch 0.1.9 documentation](https://pytorch.org/docs/stable/torch.html)
- 注意Torch的Tensor和numpy的array会共享他们的存储空间，修改一个会导致另外的一个也被修改
"""
import torch
# 此处演示tensor和numpy数据结构的相互转换
a = torch.ones(5)
b = a.numpy()

# 此处演示当修改numpy数组之后,与之相关联的tensor也会相应的被修改
a.add_(1)
print(a)
print(b)

# 将numpy的Array转换为torch的Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 另外除了CharTensor之外，所有的tensor都可以在CPU运算和GPU预算之间相互转换
# 使用CUDA函数来将Tensor移动到GPU上
# 当CUDA可用时会进行GPU的运算
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()

