# In[1]: dataloader怎么实现
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):  # 继承自Dataset
    def __init__(self):
        pass

    def __getitem__(self, index):  # 魔法方法会被自动调用
        """
        通过索引可以把某条数据拿出来
        dataset[index]
        """
        pass

    def __len__(self):
        """
        可以使用的时候返回数据集的数据条数
        :return:
        """
        pass


# 加载数据
dataset = DiabetesDataset()
"""
initialize loader with batch-size shuffle, process number
num_workers 多少并行的线程读取数据
"""
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
# In[2]: diabetes.csv dataloader怎么实现

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy_data = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = xy_data.shape[0]
        self.x_data = torch.from_numpy(xy_data[:, :-1])  # 包左不包右
        self.y_data = torch.from_numpy(xy_data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset("diabetes.csv")
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model=Model()
criterion = torch.nn.BCELoss(size_average=False)  # 可以继承NN构造自己的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 可以继承NN构造自己的优化器

for epoch in range(100):
    # enumerate 的目的是为了获得知道这是当前第几次迭代
    for i,data in enumerate(train_loader,0):
        # 1 prepare data    data=x、y元组 每次是x[i] y[i] 会自动转换成tensor
        inputs,labels =data
        # 2 forward
        y_pred = model(inputs)

        # 前向传播
        loss = criterion(y_pred, labels)
        print(epoch,i, loss.item())

        # 2 初始化梯度
        optimizer.zero_grad()

        # 3 反向传播
        loss.backward()
        # 4 计算损失并更新权重
        optimizer.step()