import torch.nn as nn
import torch.nn.functional as F
import torch

"""
seq2seq属于encoder-decoder结构的一种，基本思想就是利用两个RNN，一个RNN作为encoder，另一个RNN作为decoder。

编码器encoder负责将输入序列压缩成指定长度的向量，seq2seq网络的编码器是RNN，它为输入句子中的每个单词输出一些值。 
对于每个输入字，编码器输出一个向量和一个隐藏状态，并将隐藏状态用于下一个输入字。
解码器decoder则负责根据语义向量生成指定的序列，是另一个RNN，它采用编码器输出矢量并输出一系列单词来创建翻译。
"""

"""
这里涉及到embedding等是在自然语言处理用到的方法，具体在timeseries里面要进行适当修改
"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        """
        输入
            input of shape (seq_len, batch, input_size)
            h_0 of shape (num_layers * num_directions, batch, hidden_size)
        输出
            output of shape (seq_len, batch, num_directions * hidden_size)
            h_n of shape (num_layers * num_directions, batch, hidden_size)
        """

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)# 可以把gru看作激活函数

        return output, hidden

    def initHidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.output[0])  # 等于hn
        return output, hidden

    def initHidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)


MAX_LENGTH = 100


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)
