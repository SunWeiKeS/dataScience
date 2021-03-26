# In[1]: 加载数据
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd

import Data_Science.DP_PyTorch.Dive_into_DL_PyTorch.d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)  # 设置默认tensor类型

path = r"G:\# Project\数据集\UsingDataSet\Kaggle_housePrice_data\\"
train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")

# In[2]: 查看数据分布
print(train_data.shape)  # 输出 (1460, 81)  训练数据集包括1460个样本、80个特征和1个标签。
print(test_data.shape)  # 输出 (1459, 80) 测试数据集包括1459个样本和80个特征。我们需要将测试数据集中每个样本的标签预测出来。
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])  # 让我们来查看前4个样本的前4个特征、后2个特征(-3,-2)和标签（SalePrice）(最后一个)

"""
可以看到第一个特征是Id，
它能帮助模型记住每个训练样本，
但难以推广到测试样本，
所以我们不使用它来训练。
我们将所有的训练数据和测试数据的79个特征按样本连结。
"""
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 前闭后开集合 即不含-1

# In[3]:  预处理数据
"""
我们对连续数值的特征做标准化（standardization）：
设该特征在整个数据集上的均值为μ，标准差为σ。
那么，我们可以将该特征的每个值先减去μ再除以σ得到标准化后的每个特征值。
对于缺失的特征值，我们将其替换成该特征的均值。
"""
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 数值特征
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

# 标准化后，每个特征的均值变为0，所以可以直接⽤0来替换缺失值
all_features = all_features.fillna(0)

"""
接下来将离散数值转成指示特征。
举个例⼦，假设特征MSZoning⾥⾯有两个不同的离散值RL和RM，
那么这⼀步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
如果⼀个样本原来在MSZoning⾥的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
"""
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)  # (2919, 354)

"""
可以看到这⼀步转换将特征数从79增加到了354。
最后，通过 values 属性得到NumPy格式的数据，并转成 NDArray ⽅便后⾯的训练。
"""
n_train = train_data.shape[0]  # 训练集长度
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data["SalePrice"].values, dtype=torch.float).view(-1, 1)  # -1在这里的意思是让电脑帮我们计算

# In[4]: 训练模型
"""
我们使用一个基本的线性回归模型和平方损失函数来训练模型
"""
loss = torch.nn.MSELoss()


def get_net(feature_num: int):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


# 对数均⽅根误差的实现如下
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))  # 按维度返回最大值
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))  # 以自然数e为底的对数函数
    return rmse.item()


"""
使用了Adam优化算法。相对之前使用的小批量随机梯度下降，它对学习率相对不那么敏感。
"""


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    # 使用Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls


# In[5]: k折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


"""
在k折交叉验证中我们训练k次并返回训练和验证的平均误差。
有时候你会发现⼀组参数的训练误差可以达到很低，
但是在k折交叉验证上的误差可能反⽽较⾼。
这种现象很可能是由过拟合造成的。
因此，当训练误差降低时，我们要观察k折交叉验证上的误差是否也相应降低
"""


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d,train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


# In[6]: 模型选择
"""
我们使用一组未经调优的超参数并计算交叉验证误差。
可以改动这些超参数来尽可能减小平均测试误差。
"""
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

"""
有时候你会发现一组参数的训练误差可以达到很低，
但是在KK折交叉验证上的误差可能反而较高。
这种现象很可能是由过拟合造成的。
因此，当训练误差降低时，我们要观察KK折交叉验证上的误差是否也相应降低。
"""
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))


# In[7]:  预测并在Kaggle提交结果
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds=net(test_features).detach().numpy()

    """
    preds 是2维数据  ndim查看维度，shape查看形状
    形状是(1459, 1)-->[[x],[x],[x],[x],[x],[x],[x]]  
            reshape(1,-1) 后
            转秩成 (1, 1459)，这样行数变为1，列为1459-->[[x,x,x,x,x,x,x]]
    这样取[0]其实是把里面的数据全部都取出来了
    """
    test_data['SalePrice']=pd.Series(preds.reshape(1,-1)[0])
    submission=pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    # submission.to_csv("./submission.csv",index=False)

"""
设计好模型并调好超参数之后，
下⼀步就是对测试数据集上的房屋样本做价格预测。
如果我们得到与交叉验证时差不多的训练误差，
那么这个结果很可能是理想的，
可以在Kaggle上提交结果。
"""

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr,
               weight_decay, batch_size)