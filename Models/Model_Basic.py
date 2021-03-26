#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import json
import time
import random
import matplotlib.pyplot as plt
# from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

# assign device on gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# In[ ]:


with open('./whole_data.json', 'r') as f:
    data_dict = json.load(f)

list_2, list_3, list_4, list_5, list_6 = data_dict['2'], data_dict['3'], data_dict['4'], data_dict['5'], data_dict['6']
x_2, x_3, x_4, x_5, x_6 = np.array(list_2), np.array(list_3), np.array(list_4), np.array(list_5), np.array(list_6)
y_2, y_3, y_4, y_5, y_6 = np.zeros(len(x_2)), np.ones(len(x_3)), np.ones(len(x_4))*2, np.ones(len(x_5))*3, np.ones(len(x_6))*4

x = np.concatenate((x_2, x_3, x_4, x_5, x_6), axis=0)
y = np.concatenate((y_2, y_3, y_4, y_5, y_6), axis=0)
print(x.shape)
print(y.shape)
print(x[602])
print(y[602])


# In[ ]:


# shuffle dataset with same order
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)


# # Hyper Params

# In[ ]:


# hyper paramaters
max_epochs = 100
batch_size = 32
input_dim = 4
hidden_dim = 128
output_dim = 5
dropout_rate = 0.5
learning_rate = 0.0003
weight_decay = 0.0
grad_clip = 6.0
seed = 1


# In[ ]:


def standardization(data_x):
    new_x = data_x.copy()
    new_x[:, 0] = (data_x[:, 0] - np.mean(data_x[:, 0], axis=0)) / np.std(data_x[:, 0], axis=0)
    new_x[:, 1] = (data_x[:, 1] - np.mean(data_x[:, 1], axis=0)) / np.std(data_x[:, 1], axis=0)
    new_x[:, 2] = (data_x[:, 2] - np.mean(data_x[:, 2], axis=0)) / np.std(data_x[:, 2], axis=0)
    new_x[:, 3] = (data_x[:, 3] - np.mean(data_x[:, 3], axis=0)) / np.std(data_x[:, 3], axis=0)
    return new_x

x_standarded = standardization(x)

# print(x[0])
# print(y[0])
# print(x_standarded[0])


# In[ ]:


# split dataset
def split_data(x, y):
    train_x = x[: 2400]
    dev_x = x[2400: 2700]
    test_x = x[2700:]
    train_y = y[: 2400]
    dev_y = y[2400: 2700]
    test_y = y[2700:]
    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)

(train_x, train_y), (dev_x, dev_y), (test_x, test_y) = split_data(x_standarded, y)
print(train_x.shape)
print(dev_x.shape)
print(test_x.shape)

def get_data_loader(x, y, batch_size: int, shuffle: bool):
    x_tensor = torch.tensor(x, dtype=torch.float, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    dataset = Data.TensorDataset(x_tensor, y_tensor)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

train_dataLoader = get_data_loader(x = train_x, y = train_y, batch_size=batch_size, shuffle=False)
dev_dataLoader = get_data_loader(x = dev_x, y = dev_y, batch_size=batch_size, shuffle=False)


# In[ ]:


# build model
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, in_tensor):
        res = self.dropout(torch.relu(self.fc_1(in_tensor)))
        res = torch.relu(self.fc_2(res))
        out_tensor = torch.log_softmax(self.fc_3(res), dim=1)
        
        return out_tensor

def weights_init(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0.0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
    
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


model = MyModel(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.NLLLoss()

# train and evaluate
def train(data_loader):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        train_output = model(batch_x)
        loss = criterion(train_output, batch_y.reshape(-1))
        loss.backward()
        epoch_loss += loss.item()
#         nn.utils.clip_grad_norm_(model.parameters())
        optimizer.step()
    return epoch_loss / len(data_loader)

def validate(data_loader):
    model.eval()
    epoch_loss = 0
    for batch_x, batch_y in data_loader:
        dev_output = model(batch_x)
        loss = criterion(dev_output, batch_y.reshape(-1))
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

loss_train, loss_dev = [], []
start_time = time.time()

for epoch in range(max_epochs):
    train_loss = train(train_dataLoader)
    dev_loss = validate(dev_dataLoader)
    loss_train.append(train_loss)
    loss_dev.append(dev_loss)

print(f'duration: {time.time() - start_time}')

plt.figure()
plt.plot(loss_train)
plt.plot(loss_dev)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()


# In[ ]:


def evaluate(infer_x):
    model.eval()
    with torch.no_grad():
        infer_x = torch.tensor(infer_x, dtype=torch.float, device=device)
        infer_y = model(infer_x.unsqueeze(0))
        infer_y = infer_y.max(1)[1]
    return infer_y

infer_test = []
for idx_data in test_x:
    infer_output = evaluate(idx_data)
    infer_test.append(infer_output.item())

num = 0
for tgt, gen in zip(test_y, infer_test):
    if tgt == gen:
        num += 1

accuracy = (num / len(test_y)) * 100
print(f'accuracy: {accuracy:.2f}')


# In[ ]:




