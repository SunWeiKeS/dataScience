"""
#%% md

- 使用 torch.nn 包可以进行神经网络的构建。
- 现在你对autograd有了初步的了解，而nn建立在autograd的基础上来进行模型的定义和微分。
- nn.Module中包含着神经网络的层，同时forward(input)方法能够将output进行返回。
- 这是一个简单的前馈神经网络。 从前面获取到输入的结果，从一层传递到另一层，最后输出最后结果。
- 一个典型的神经网络的训练过程是这样的:
    1. 定义待可学习参数的网络结构;
    2. 数据集输入;
    3. 对输入进行处理,主要体现在网络的前向传播;
    4. 计算loss function;
    5. 反向传播求梯度;
    6. 根据梯度改变参数值,最简单的实现方式为:
       - 通常使用简单的更新规则: weight = weight + learning_rate * gradient
"""
# In[1]:定义网络
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        """
        an affine operation: y = Wx + b
        nn.Linear（）是用于设置网络中的全连接层的
        包含：
            ·in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
            ·out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，
                         当然，它也代表了该全连接层的神经元个数。
        """
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2 = nn.Linear(120, 84) # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3 = nn.Linear(84, 10) # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self,x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 输入x经过卷积conv1之后，经过激活函数ReLU（原来这个词是激活函数的意思），使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x)) # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x)) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) #输入x经过全连接3，然后更新x
        return x

     #使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self,x):
        size= x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features=1
        for s in size:
            num_features *=s
        return  num_features
# In[2]:
net =Net()
print(net)

#一个模块可学习的参数都在net.parameters中
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

from torch.autograd import Variable
# forward函数的输入与输出都是autograd.Variable类型的.
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

# 将所有参数的梯度缓存清零，然后进行随机梯度的的反向传播：
net.zero_grad()
out.backward(torch.randn(1,10))

"""
### note
- torch.nn包仅支持对批量数据的处理,而不能对单个样本进行处理.
- 例如,nn.Conv2d只接受4维的张量:
         nSamples * nChannels * Height * Width
- 如果只有单个样本,那么使用input.unsqueeze(0)来增加假的batch维度.

### 回顾
- torch.Tensor：一个用过自动调用 backward()实现支持自动梯度计算的多维数组，并且保存关于这个向量的梯度 w.r.t.
- nn.Module：神经网络模块。封装参数、移动到GPU上运行、导出、加载等。
- nn.Parameter：一种变量，当把它赋值给一个Module时，被自动地注册为一个参数。
- autograd.Function：实现一个自动求导操作的前向和反向定义，每个变量操作至少创建一个函数节点，每一个Tensor的操作都会创建一个接到创建Tensor和 编码其历史 的函数的Function节点。

### 重点如下
>- 定义一个网络
- 处理输入，调用backword

### 还剩
>- 计算损失
- 更新网络权重

### 损失函数 Loss Function
- 一个损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少。
- nn包下提供了几种不同的损失函数
- 最简单的是nn.MSELoss,它计算输出和目标间的均方误差
"""
# In[3]:
output=net(input)
target = torch.randn(10) # 随机值作为样例
target=target.view(1,-1) # 使target和output的shape相同

criterion=nn.MSELoss()
loss =criterion(output,target)
print(loss)
"""
- 沿着loss的反向传播方向，依次用.grad_fn属性，就可以得到如下所示的计算图．

      -> input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
- 所以，当我们调用 loss.backward()时,整张计算图都会 根据loss进行微分，
 而且图中所有设置为requires_grad=True的张量 所有Variable的.grad属性会被累加．

 以下几条语句对反向求梯度做了解释：
"""
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU
# In[4]: 反向传播
"""
### Backprop
- 反向传播的过程只需要调用loss.backgrad()函数即可．
- 但是由于变量的梯度是累加的，所以在求backward之前应该先对现有的梯度清零．
以下调用了loss.backgrad()函数，并分别查看conv1.bais在反向传播前后的梯度。
"""
net.zero_grad()     # 清除梯度
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# In[5]:更新权重
"""
### 更新权重
- 在实践中最简单的权重更新规则是随机梯度下降（SGD）
   - weight = weight - learning_rate * gradient
   - 我们可以使用简单的Python代码实现这个规则：
    ```text
      learning_rate = 0.01
      for f in net.parameters():
          f.data.sub_(f.grad.data * learning_rate)
    ```

- 为了满足不同的更新规则，
比如 SGD, Nesterov-SGD, Adam, RMSProp等
pytorch提供了一个很小的包：torch.optim
"""
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
"""
通常用在每个mini-batch之中，
而scheduler.step()通常用在epoch里面,但是不绝对，可以根据具体的需求来做。
只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。
"""

















