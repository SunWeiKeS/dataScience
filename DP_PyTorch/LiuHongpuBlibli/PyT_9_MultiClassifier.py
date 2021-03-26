# In[1]: 实现一般形式交叉熵 NLLLoss Negative Log LikeliHood Loss
import numpy as np

y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])
y_pred = np.exp(z) / np.exp(z).sum()
loss = (-y * np.log(y_pred)).sum()
print(loss)

# In[2]: 内置交叉熵（含有softmax层的损失函数，不需要再神经网络最后一层做非线性变换了）
import torch

y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
print(loss)

# In[3]: 实例

import torch

criterion = torch.nn.CrossEntropyLoss()

Y = torch.LongTensor([2, 0, 1])
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])
Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])
l1 = criterion(Y_pred1, Y)
l2 = criterion(Y_pred2, Y)
print("batch loss1= ", l1.data, "\nbatch loss2= ", l2.data)

# In[3]: 手写数字多分类
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
# 1 载入数据
transform = transforms.Compose([
    transforms.ToTensor(),  # 把图片转换成pytorch可识别的张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化 0.1307均值  0.3081标准差
])
train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# 2 构建网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()
# 3 构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
"""
加完动量后loss下降非常快，可以将动量理解为惯性作用，所以每次更新的幅度都更大。
"""
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4 定义训练器和测试器
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()  # 优化器在优化前要记得梯度清零 在反向传播之前

        # forward +backward +update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()  # 反向传播
        optimizer.step()  # update更新权重

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试代码不需要计算梯度,这里面的就不会计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 拿出每一行对应的最大值的下标和最大值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuray on test set: %d %%' % (100 * correct / total))


# 5 训练和测试
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            test()  # 每十轮测试一次
