# In[1]:
import torch

in_channels, out_channels = 5, 10  # 输入输出
width, height = 100, 100  # 图片大小
kernel_size = 3  # 卷积核大小 一般奇数
batch_size = 1  # 小批量输入数据

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)
output = conv_layer(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)  # 卷积层对宽度高度没要求，但是in_channels必须相一致，不然会出错

# In[2]:
import torch

input = [3, 4, 5, 6, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
"""
view(1,1,5,5)   # 相当于numpy reshape 
Batch批量维度、Channel维度、width宽度、height高度
"""
input = torch.Tensor(input).view(1, 1, 5, 5)

# 输入通道1 输出通道1 卷积核大小为3 padding为1在外圈填充一层 bias即偏置量 这里不需要
# conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
"""
步长为2的版本  输入通道1 输出通道1 卷积核大小为3 不进行填充，步长设置为1
"""
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data  # 将卷积核权重赋值给卷积层，用于初始化
output = conv_layer(input)
print(output)

# In[2]:
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 64
# 1 载入数据
transform = transforms.Compose([
    transforms.ToTensor(),  # 把图片转换成pytorch可识别的张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化 0.1307均值  0.3081标准差
])
train_dataset = datasets.MNIST(root='../dataset/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# 构建模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n,1,28,28) to (n,784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x


model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=False)  # 可以继承NN构造自己的损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 可以继承NN构造自己的优化器


# 4 定义训练器和测试器
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = input.to(device), target.to(device)  # 将输入和目标都放到GPU上
        # 初始化梯度
        optimizer.zero_grad()  # 优化器在优化前要记得梯度清零 在反向传播之前
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, target)
        # 反向传播
        loss.backward()  # 反向传播

        # 计算损失更新权重
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
            inputs, target = data
            inputs, target = input.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)  # 拿出每一行对应的最大值的下标和最大值
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuray on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


# In[3]: Inception Module
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.brach1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.brach5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.brach5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.brach3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.brach3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.brach3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.brach_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        brach1x1 = self.brach1x1(x)

        brach5x5 = self.brach5x5_1(x)
        brach5x5 = self.brach5x5_2(brach5x5)

        brach3x3 = self.brach3x3_1(x)
        brach3x3 = self.brach3x3_2(brach3x3)
        brach3x3 = self.brach3x3_3(brach3x3)

        brach_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        brach_pool = self.brach_pool(brach_pool)

        # concatenate
        outputs = [brach1x1, brach5x5, brach3x3, brach_pool]
        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)  # 88=24x3+16

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))  # 10
        x = self.incep1(x)  # 88
        x = F.relu(self.mp(self.conv2(x)))  # 20
        x = self.incep2(x)  # 88
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


# In[4]: Residual Block
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
