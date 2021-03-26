# In[1]: 怎么使用RNNCell
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

# 构造rnncell
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq,batch,features) 相比之前多了序列的维度
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('Input size:', input.shape)

    hidden = cell(input, hidden)  # 隐层等于这一次的输入加上上一次的隐层

    print('output size: ', hidden.shape)
    print(hidden)

# In[2]: 怎么使用RNN RNN运算非常耗时间

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# (seq_Len,batch_Size,input_Size)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print("Output size: ", out.shape)
print("Output: ", out)
print("Hidden size: ", hidden.shape)
print("Hidden: ", hidden)

# In[3]: RNNCell
import torch

batch_size = 1
input_size = 4
hidden_size = 4

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot_lookup = [[1, 0, 0, 0],  # 0
                  [0, 1, 0, 0],  # 1
                  [0, 0, 1, 0],  # 2
                  [0, 0, 0, 1]]  # 3

"""
先看 for x in x_data
意思是x_data里面的数据 
比如说
    1 ，对应one_hot_lookup[1] 即 [0, 1, 0, 0]
    2 ，对应one_hot_lookup[2] 即 [0, 0, 1, 0]
"""
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 维度为 seq*inputsize

# Reshape the inputs to (seqLen,batchSize,inputSize)
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  # -1 表示不确定的数字
# Reshape the labels to (seqLen,1)
labels = torch.LongTensor(y_data).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size  # 只有在构造h_0的时候用得到
        self.input_size = input_size
        self.hidden_size = hidden_size
        """
        input of shape(batch,input_size)
        hidden of shape(batch,hidden_size)
        """
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)

# 构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)  # 可以继承NN构造自己的优化器

# train
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string:', end='')
    for input, label in zip(inputs, labels):  # inputs(seqlen batchsize inputsize) input(batchsize inputsize)
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()  # 反向传播
    optimizer.step()  # update更新权重
    print(',Epoch[%d/15] loss=%.4f' % (epoch + 1, loss.item()))

# In[4]: RNN
import torch

input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot_lookup = [[1, 0, 0, 0],  # 0
                  [0, 1, 0, 0],  # 1
                  [0, 0, 1, 0],  # 2
                  [0, 0, 0, 1]]  # 3
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 维度为 seq*inputsize
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)  # -1 表示不确定的数字
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size  # 只有在构造h_0的时候用得到
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        out, _ = self.rnn(inputs, hidden)
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)

# 构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.5)  # 可以继承NN构造自己的优化器

# train
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted string:', ''.join([idx2char[x] for x in idx]), end='')
    print(',Epoch[%d/15] loss=%.3f' % (epoch + 1, loss.item()))

# In[4]: embedding and linear layer
import torch
# 参数
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]  # hello 维度(batch,seq_len)
y_data = [3, 1, 2, 3, 2]  # ohlol   维度(batch*seq_len)

inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

        """
        input of fc layer
            batchsize,seqlen,hiddensize
        output of fc layer
            batchsize，seqlen，numclass
        """
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        print(x.shape)
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # batch seqLen embeddingSize
        print(x.shape)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)  # shape为了使用交叉熵 batchsize*seqlen,numClass

net = Model()

# 构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.5)  # 可以继承NN构造自己的优化器

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted string:', ''.join([idx2char[x] for x in idx]), end='')
    print(',Epoch[%d/15] loss=%.3f' % (epoch + 1, loss.item()))

