---
title: 深度学习实验
typora-root-url: DeepLearning1
date: 2024-07-20 16:49:09
summary: 介绍一些pytorch的一些操作，然后实现线性回归和softmax回归
categories: 深度学习
---



使用初始化一个1×3的矩阵M和一个2×1的矩阵N，对两矩阵进行减法操作（要求实现三种不同的形式），给出结果并分析三种方式的不同（如果出现报错，分析报错的原因），同时需要指出在计算过程中发生了什么

```python
import torch

a=torch.randn(1,3)
b=torch.randn(2,1)
print(a)
print(b)
print(a-b)
print(torch.sub(a,b))
print(a.sub(b))
```



  利用创建两个大小分别 3×2 和 4×2的随机数矩阵P和Q，要求服从均值为0，标准差0.01为的正态分布；② 对第二步得到的矩阵Q进行形状变换得到的Q的转置QT; ③对上述得到的矩阵P和矩阵QT求矩阵相乘

```python
P = torch.normal(0, 0.01, (3, 2))
Q= torch.normal(0, 0.01, (4, 2))
QT = torch.t(Q)
result = torch.mm(P, QT)
print(P)
print(Q)
print(QT)
print(result)
```



给定公式 y3=y1+y2=x^2+x^3，且 x=1。利用学习所得到的Tensor的相关知识，求y3对的x的梯度。要求在计算过程中，在计算x^3时中断梯度的追踪，观察结果并进行原因分析提示, 可使用withtorch.no_grad()， 举例:

withtorch.no_grad():

y = x *5

```python
x=torch.ones(1,1,requires_grad=True)
y1=x**2
with torch.no_grad():
    y2=x**3
y3=y1+y2
y3.mean().backward()
print(x)
print(x.grad)
```



要求动手从0实现 logistic 回归（只借助Tensor和Numpy相关的库）在人工构造的数据集上进行训练和测试（可借助nn.BCELoss或nn.BCEWithLogitsLoss作为损失函数，从零实现二元交叉熵为选作）

```python
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
def createdata():
    n_data = torch.ones(50, 2) # 数据的基本形态
    x1 = torch.normal(2 * n_data, 1) # shape=(50, 2)
    y1 = torch.zeros(50) # 类型0 shape=(50, 1)
    x2 = torch.normal(-2 * n_data, 1) # shape=(50, 2)
    y2 = torch.ones(50) # 类型1 shape=(50, 1)
    # 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
    x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
    y = torch.cat((y1, y2), 0).type(torch.FloatTensor)
    return x, y

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的     
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch         
        yield  features.index_select(0, j), labels.index_select(0, j)

def myLogistic(x, w, b):
    return 1/(1 + torch.exp(-1 * torch.mm(x,w) + b))

# 定义二分类交叉熵损失函数
def binary_cross_entropy(y_pred, y_true):
    # 防止梯度爆炸
    epsilon = 1e-7
    # 计算损失
    loss = -torch.mean(y_true * torch.log(y_pred + epsilon) + (1 - y_true) * torch.log(1 - y_pred + epsilon))
    return loss

def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
features,labels=createdata()
plt.scatter(features.data.numpy()[:, 0], features.data.numpy()[:, 1], c=labels.data.numpy(), s=100, lw=0,  cmap='RdYlGn') 
plt.show()

num_inputs = 2
lr = 0.01
num_epochs = 20
batch_size = 10
net = myLogistic
loss = binary_cross_entropy
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
train_all_loss = []
for epoch in range(num_epochs):       # 训练模型一共需要num_epochs个迭代周期     # 在每一个迭代周期中，会使用训练数据集中所有样本一次
    for X, y in data_iter(batch_size, features, labels):       # x和y分别是小批量样本的特征和标签
        l = loss(net(X, w, b), y).sum()     # l是有关小批量X和y的损失         
        l.backward()     # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)     # 使用小批量随机梯度下降迭代模型参数         
        w.grad.data.zero_()    # 梯度清零         
        b.grad.data.zero_()
    labels_pred = net(features, w, b)
    train_l = loss(labels_pred, labels.view(-1, 1))
    train_all_loss.append(train_l.item())
    labels_pred = torch.tensor(np.where(labels_pred>0.5, 1, 0),dtype=torch.float32)
    acc = (labels_pred.squeeze() == labels.squeeze()).sum().item() / 100
    print('epoch: %d loss:%.5f acc: %.3f'%(epoch+1,train_l.item(), acc))
test_data,test_labels= createdata()
plt.scatter(test_data.data.numpy()[:, 0], test_data.data.numpy()[:, 1], c=test_labels.data.numpy(), s=100, lw=0,  cmap='RdYlGn') 
plt.show()
with torch.no_grad():
    labels_pred_test = net(test_data,w,b)
    test_l =  binary_cross_entropy(labels_pred_test, test_labels.view(-1,1))
    labels_pred_test = torch.tensor(np.where(labels_pred_test>0.5, 1, 0),dtype=torch.float32)
    acc_test = (labels_pred_test.squeeze() == test_labels.squeeze()).sum().item() / 100
    print('Test_loss: %.5f Test_acc: %.3f'%(test_l, acc_test))
```



 要求动手从0实现 softmax 回归（只借助Tensor和Numpy相关的库）在Fashion-MNIST数据集上进行训练和测试，并从loss、训练集以及测试集上的准确率等多个角度对结果进行分析（要求从零实现交叉熵损失函数）

```python
import torch
from IPython import display
from d2l import torch as d2l
from torchvision import transforms
import torchvision
from torch.utils import data

batch_size = 256

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集,然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, '
            f'train loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, '
              f'test acc {test_acc:.3f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

