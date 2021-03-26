import csv
import gzip

import torch
import time
import math

# 准备数据
from torch.nn.utils.rnn import pack_padded_sequence # 补0操作保证长度一致
from torch.utils.data import Dataset, DataLoader

# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128
USE_GPU = True


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm%ds' % (m, s)


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = r'G:\# Project\数据集\UsingDataSet\Other_CSV\names_train.csv.gz' if is_train_set else r'G:\# Project\数据集\UsingDataSet\Other_CSV\names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num =len(self.country_list)


    def __getitem__(self, index):
        """
        拿到的名字是字符串，国家是索引
        """

        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def id2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum()  # 我们模型的输出大小


def name2list(name): # ord读取ASCII值
    arr = [ord(c) for c in name]
    return arr, len(arr)

def create_tensor(tensor):
    if USE_GPU:
        device=torch.device("cuda:0")
        tensor =tensor.to(device)
    return tensor

def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        """
        并不是补充0，而是先构造一个全0的张量，然后一个个贴过去
        """
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True) # 降序
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1  # 双向GRU

        """
        输入 seqLen,batchSize
        输出 seqLen,batchSize,hiddenSize
        """
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        """
        输入 hidden_size
        输出 hidden_size
        
        input of GRU Layer with shape
            input: (seqLen,batchSize,hiddenSize)
            hidden: (nLayers*nDirections,batchSize,hiddenSize)

        outputs of GRU Layer with shape
            output: (seqLen,batchSize,hiddenSize*nDirections)
            hidden: (nLayers*nDirections,batchSize,hiddenSize)
        """
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)  # n_layers * n_directions,batch_size,hidden_size

    def forward(self, input, seq_lengths):
        # input shape: B x S --> S x B
        input = input.t()  # 转秩
        batch_size = input.size(1)  # 保存为了构造初始隐层

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)
        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # shape seqlLen,batchSize,hiddenSize
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            # 如果使用双向GRU 需要凭借正向和反向的隐藏层的拼接
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)
        return fc_output

# 训练
def trainModel():
    total_loss=0
    for i,(names,countries) in enumerate(trainloader,1):

        inputs,seq_lengths,target =make_tensors(names,countries)
        output=classifier(inputs,seq_lengths)
        loss =criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss +=loss.item()
        if i %10 ==0:
            print(f'[{time_since(start)}] Epoch {epoch}',end='')
            print(f'[{i*len(inputs)}]/{len(trainset)}',end='')
            print(f'loss={total_loss/(i*len(inputs))}')
    return total_loss

def testModel():
    correct=0
    total =len(testset)
    print("evaluating trained model ...")

    with torch.no_grad():
        for i,(names,countries) in enumerate(testloader,1):
            inputs,seq_lengths,target =make_tensors(names,countries)
            output=classifier(inputs,seq_lengths)
            pred=output.max(dim=1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent='%.2f'%(100*correct/total)
        print(f'test set: Accuary {correct}/{total} {percent}%')
    return correct/total

if __name__ == '__main__':
    """
    N_CHARS 英文字母 字符数量
    HIDDEN_SIZE 隐层维度
    N_COUNTRY 分类的数量
    N_LAYER GRU层数
    """
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc)
