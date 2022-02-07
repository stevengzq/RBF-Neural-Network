import scipy.io
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# 加载数据集
data = torch.from_numpy(scipy.io.loadmat('data_train.mat')['data_train']).to(torch.float32)
label = torch.from_numpy(scipy.io.loadmat('label_train.mat')['label_train']).to(torch.float32)

# 按照 6：2：2分割数据集
data_train = data[0:int(data.shape[0] * 0.6), :]
label_train = label[0:int(data.shape[0] * 0.6), :]

data_cross_validation = data[int(data.shape[0] * 0.6):int(data.shape[0] * 0.8), :]
label_cross_validation = label[int(data.shape[0] * 0.6):int(data.shape[0] * 0.8), :]

data_test = data[int(data.shape[0] * 0.8):, :]
label_test = label[int(data.shape[0] * 0.8):, :]

# data_label[data_label == -1] = 0  # 修改数据标签为-1的数据为0，方便后续计算cross entropy loss
# data_test = torch.from_numpy(scipy.io.loadmat('data_test.mat')['data_test']).to(torch.float32)

# 计算hidden layer参数
n_neurons = 4
centers = KMeans(n_clusters=n_neurons).fit(data_train).cluster_centers_  # hidden layer neurons centers found by K-means
sigma = n_neurons * np.nanmax(squareform(pdist(centers))) / n_neurons  # sigma value in hidden layer neurons

# hidden layer 输出计算
hidden_output_train = np.zeros((data_train.shape[0], n_neurons))
hidden_output_cross_validation = np.zeros((data_cross_validation.shape[0], n_neurons))
hidden_output_test = np.zeros((data_test.shape[0], n_neurons))

for i in range(data_train.shape[0]):
    d = np.sum(np.power(np.tile(data_train[i], (n_neurons, 1)) - centers, 2), axis=1)
    o = np.exp(-1 * d / (2 * sigma ** 2))  # 单个数据的hidden layer输出
    hidden_output_train[i] = o

for j in range(data_cross_validation.shape[0]):
    d = np.sum(np.power(np.tile(data_cross_validation[j], (n_neurons, 1)) - centers, 2), axis=1)
    o = np.exp(-1 * d / (2 * sigma ** 2))
    hidden_output_cross_validation[j] = o

for k in range(data_test.shape[0]):
    d = np.sum(np.power(np.tile(data_test[k], (n_neurons, 1)) - centers, 2), axis=1)
    o = np.exp(-1 * d / (2 * sigma ** 2))
    hidden_output_test[k] = o

# 将ndarray矩阵转换为tensor张量
hidden_output_train = torch.from_numpy(hidden_output_train).to(torch.float32)
hidden_output_cross_validation = torch.from_numpy(hidden_output_cross_validation).to(torch.float32)
hidden_output_test = torch.from_numpy(hidden_output_test).to(torch.float32)


# 按照线性回归模型定义output layer
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(n_neurons, 1, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


# 定义sigmoid函数用于后续分类器
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# 定义accuracy函数用于计算分类器准确度
def accuracy(x, y):
    output = output_layer(x)
    prediction = sigmoid(output)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = -1
    accuracy = sum(prediction == y) / y.shape[0] * 100
    return accuracy


# training
output_layer = LinearRegression()
optimizer = torch.optim.SGD(output_layer.parameters(), lr=0.0001)
loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()
EPOCH = 2000

loss_history = []
accuracy_history = []

for epoch in range(EPOCH):
    for i in range(data_train.shape[0]):
        output = output_layer(hidden_output_train[i])  # shape:[2]
        # output = torch.unsqueeze(output, dim=0)  # 针对逻辑回归二分类问题需要给output在指定维度（dim=0）增加一维使其变成一个向量（shape：[1,2]）
        loss = loss_function(output, label_train[i])

        optimizer.zero_grad()  # clear gradient for this training step
        loss.backward()  # 利用back propagation算法计算模型中各个参数的gradient
        optimizer.step()  # update parameters

    # calculating classification accuracy
    cross_validation_accuracy = accuracy(hidden_output_cross_validation, label_cross_validation)
    train_accuracy = accuracy(hidden_output_train, label_train)

    accuracy_history.append(cross_validation_accuracy)
    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print('Epoch', epoch, '| train loss:%.4f' % loss.item(), '| train accuracy:%2.2f' % train_accuracy, '%',
              '| validation accuracy:%2.2f' % cross_validation_accuracy, '%')

# plot the model training process
fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
ax1.plot(range(EPOCH), loss_history, 'b')
ax1.set_ylabel('MSE Loss')
ax1.set_xlabel('Training Epoch')
ax1.set_title('Training performance with %d hidden neurons' % n_neurons)

ax2 = ax1.twinx()
ax2.plot(range(EPOCH), accuracy_history, 'r')
ax2.set_ylabel('Validation Accuracy (%)')

plt.show()

# accuracy summary
train_accuracy = accuracy(hidden_output_train, label_train)
test_accuracy = accuracy(hidden_output_test, label_test)
print('\nModel Performance |', n_neurons, 'hidden neurons |', 'train accuracy:%2.2f' % train_accuracy, '%',
      '| test accuracy:%2.2f' % test_accuracy, '%')
