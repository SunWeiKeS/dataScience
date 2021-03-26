#### 4 pytorch反向传播
```python
w=torch.Tensor([1.0])
w.requires_grad =True

print("Predict (before training)", 4, forward(4).item)

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l=loss(x,y)
        l.backward()
        print("\t grad: ",x,y,w.grad.item())
        
        w.grad.data.zero_() # 手动清零
    print("Progress:", epoch,  l.item())
        
print("Predict (after training)", 4, forward(4).item())
```

#### 5 pytorch 实现线性回归、逻辑回归
Pytorch等框架的核心都是
- **构造计算图**

四个步骤
1. 准备数据集
2. 设计模型使用class from nn.Module
3. 构造损失函数和优化器 使用PyTorch API
4. 训练周期 前向传播、反向传播，更新权重

不同的优化器： 
- torch.optim.Adagrad
- torch.optim.Adam
- torch.optim.Adamax
- torch.optim.ASGD
- torch.optim.LBFGS
- torch.optim.RMSprop
- torch.optim.Rprop
- torch.optim.SGD

不同的激活函数
- Unit step (Heaviside)
    - exmaple   Perceptron variant
- Sign
    - exmaple   Perceptron variant
- Linear
    - exmaple   Adaline,linear regression
- Piece-wise Linear
    - exmaple   support vector machine
- Logistic (sigmoid)
    - exmaple   linear regression、Multi-layer NN
- Hyperbolic tangent
    - exmaple   Multi-layer Neural Networks
- Rectifier,ReLU (Rectitied Linear Unit)
    - exmaple   Multi-layer Neural Networks
- Rectifier,softplus
    - exmaple   Multi-layer Neural Networks

#### 8 加载数据集 Dataset and DataLoader
- Dataset用于构造数据集
- DataLoader拿出mini-batch

三个概念
- Epoch
    - Define: one forward pass and one backward pass 
    of all the training examples
- Batch-Size
    - the number of training examples in one forward backward pass
- Iterations
    - number of passes,each pass using [batch size] number of examples
1w个样本，每次minibatch拿1k个训练，iterations为10

```python
# training cycle

# 每个周期训练一次
for epoch in range(training_epochs):
    # Loop over all batchs 每个epoch内嵌入一个mini-Batch
    for i in range(total_batch):
```

DataLoader：batch_size=2,shuffle=True
代码实现见py文件 PyT_8_minbatch.py

**错误处理**
- 在window 和linux下面处理进程的方式不同，分别对应spawn、fork
```python
##############这样在win下面会报错####################
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
for epoch in range(100):
    for i,data in enumerate(train_loader,0):
```
处理方式：
- wrap the code with an if-clause to protect code
from executing multiple times
```python
##############这样在win下面会报错####################
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
if __name__ == '__main__':
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
```
- enumerate(sequence, [start=0])
    - sequence -- 一个序列、迭代器或其他支持迭代对象。
    - start -- 下标起始位置。


使用pytorch自带的MNIST
```python
train_Dataset = torchvision.datasets.MNIST( # torchvision可以实现数据集的训练集和测试集的下载
    root="./data", # 下载数据，并且存放在data文件夹中
    train=True, # train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分。
    transform=transforms.ToTensor(), # 数据的标准化等操作都在transforms中，此处是转换
    download=True
)

test_Dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = DataLoader(dataset=train_Dataset,
                          batch_size=32,
                          shuffle=True)
test_loader = DataLoader(dataset=test_Dataset,
                          batch_size=32,
                          shuffle=False)# 没必要打乱
for epoch in range(training_epochs):
    for bach_idx,(inputs,target) in enumerate(train_loader):
        ...
```
#### 9 多分类问题
交叉熵损失和NLL之间的区别
- CrossEntropyLoss()=log_softmax() + NLLLoss() 
- NLLLoss 的 输入 是一个对数概率向量和一个目标标签.
 它不会为我们计算对数概率. 
 适合网络的最后一层是log_softmax. 
 损失函数 nn.CrossEntropyLoss() 与 NLLLoss() 相同,
 唯一的不同是它为我们去做 softmax.

特征提取
- 傅里叶变换
- 小波变换
- CNN自动特征提取

#### 10 CNN
CNN处理MNIST数据集分类问题
- 1 Input 1x28x28
    - Convolution 5x5 
- 2 C1 Feature maps 4x24x24
    - Subsampling下采样 2x2 pooling层 大小减一半（没有权重）
- 3 S1 Feature maps 4x12x12
    - Convolution 5x5 
- 4 C2 Feature maps 8x8x8
    - Subsampling下采样 2x2 pooling层 大小减一半（没有权重）
- 5 S2 Feature maps 8x4x4
    - Fully Connected
- 6 n1
    - Fully Connected //softmax
- 7 n2 Output

2~5 特征提取层

6~7 分类层 


Fully Connected Neural NetWork
- Input Layer  
    - (batch,1,28,28)
- Conv2d Layer
    - Cin =1 Cout=10 kernel =5
    - (batch,10,24,24)
    - self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
- ReLU Layer   
    - (batch,10,24,24)
- Pooling Layer
    - kernel = 2x2
    - (batch,10,12,12)
    - self.pooling=torch.nn.MaxPool2d(2)
- Conv2d Layer
    - Cin =10 Cout=20 kernel =5
    - (batch,20,8,8)
    - self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
- ReLU Layer
    - (batch,20,8,8)
- Pooling Layer
    - kernel = 2x2
    - (batch,20,4,4)-->(batch,320)
    - self.pooling=torch.nn.MaxPool2d(2)
- Linear Layer
    - fin=320,fout=10
    - (batch,10)
    - self.fc=torch.nn.Linear(320,10)
- Output Layer

#### 11 计算迁移到gpu

device =torch


#### 12 Advanced CNN
减少代码的冗余
- 函数/类减少冗余，代替相似的代码

- average pooling：
```python
init:
    self.brach_pool=torch.nn.Conv2d(in_channels,24,kernel_size=1)

forward:
    brach_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
    brach_pool=self.brach_pool(brach_pool)
```
- 1x1Conv
```python
self.brach1x1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
brach1x1=self.brach1x1(x)
```
- 1x1Conv+5x5Conv
```python
self.brach5x5_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
self.brach5x5_2=torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

brach5x5 = self.brach5x5_1(x)
brach5x5 = self.brach5x5_2(brach5x5)
```
- 1x1Conv+3x3Conv+3x3Conv
```python
self.brach3x3_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
self.brach3x3_2=torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
self.brach3x3_3=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

brach3x3 = self.brach3x3_1(x)
brach3x3 = self.brach3x3_2(brach3x3)
brach3x3 = self.brach3x3_3(brach3x3)
```
- 拼接到一起的运算就是 concatenate
```python
outputs=[brach1x1,brach5x5,brach3x3,brach_pool]
return torch.cat(outputs,dim=1)

```

Can we stack layers to go deeper?
- 并不一定层数越多越好，可能会出现梯度消失
- Residual net 要保证输入和输出的张量维度是一样的
```python
# Residual 网络的实现
class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.channels=channels
        self.conv1 =torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 =torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)
```
#### 12 Basic RNN
- 以天气预测为例，x1,x2,x3每个含有3个特征
  - 拼成含有9个维度的长项量
  - 四天一组，预测第四天

Xt --> RNN Cell --> ht
- RNN Cell 本质是一个线性层
- 线性层RNN Cell 所有的都是同一个线性层
    - RNN中使用激活函数tanh //分布在-1到1 更多一些
    - cell =torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)
    - hidden  =cell(input,hidden) 
        - input of shape(batch,input_size)
        - (input) hidden of shape(batch,hidden_size)
        - (output) hidden of shape(batch,hidden_size)

- 怎么使用
    - 假设我们的序列满足以下：
        - batchSize=1
        - seqLen=3
        - inputSize=4
        - hiddenSize=2
    - RNNCell输入和输出的shape：
        - input.shape=(batchSize,inputSize)
        - output.shape=(batchSize,hiddenSize)
    - 序列可以被压缩到一个如下形状的张量tensor
        - dataset.shape=(seqLen,batchSize,inputSize)

直接使用RNN
- num_layers = 2 # 层数目
    - cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    - inputs=[] 
    - out,hidden =cell(inputs,hidden) # 这里实际上inputs应该是输入整个时序序列，hidden即h0
         - (input) input of shape(seqSize,batch,input_size) 
         - (input) hidden of shape(numLayers,batch,hidden_size) # numLayers 层数
         - (output) output of shape(seqSize,batch,hidden_size)
         - (output) hidden of shape(numLayers,batch,hidden_size)
- 怎么使用
    - 假设我们的序列满足以下：
        - batchSize=1
        - seqLen=3
        - inputSize=4、hiddenSize=2
        - numLayers=1
    - RNN中 input 和 h_0 的shape：
        - input.shape = (seqlen,batchSize,inputSize)
        - h_0.shape= (numLayers,batch,hidden_size)
    - RNN中 output 和 h_n 的shape：
        - output.shape=(seqSize,batch,hidden_size)
        - h_n.shape= (numLayers,batch,hidden_size)

    - 如果设置 batch_first =True，则输入和输出张量变为
    (batchSize,seqLen,input_size)
        - cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        - inputs=torch.randn(batch_size,seq_len,hidden_size)

##### 小例子
- the inputs of RNN Cell should be 数字构成的向量
    - 1 单词拆分成字符数组
    - 2 构造字典 Dictionary
    - 3 取出下标Indices
    - 4 One-Hot Vectors （seqLen是长度，inputsize是宽度，即特征数）

独热编码缺点
- 维度太高
- 坐标稀疏
- 是硬编码数据
##### 小例子 embedding +linear
我们希望得到的是低维、稠密、可从数据中学习的
- 流行的方法 Embedding vectors
- embedding 层
    - 参数num_embeddings 字典大小、embedding_dim 每个尺寸 构成矩阵高度和宽度
    - 输入 长整型的值
    - 输出从(seq*batch)加上embedding，变成三维
- Linear
    - input (N,*,in_features)
    - output (N,*,out_features)
    
#### 13 LSTM
- 输入 h0 c0
- 输出 hn cn
折中办法 GRU


#### 14 RNN Classifier
使用的模型
- x --> Embedding layer
- Embedding layer --> GRU Layer
    - h0 -->  GRU Layer
- GRU Layer -->Hn
- hn --> Linear Layer
- Linear Layer --> o

GRU 层输入输出
input of GRU Layer with shape
    input: (seqLen,batchSize,hiddenSize)
    hidden: (nLayers*nDirections,batchSize,hiddenSize)

outputs of GRU Layer with shape
    output: (seqLen,batchSize,hiddenSize*nDirections)
    hidden: (nLayers*nDirections,batchSize,hiddenSize)

名字转换层tensor的过程,得到batch*seq的矩阵
- 字符串转-->一个个字符-->ASCII
- ASCII进行填充
- --> 转秩 --> 排序









