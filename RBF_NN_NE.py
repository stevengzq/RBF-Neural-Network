import scipy.io
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.cluster import KMeans

# 加载数据集
data = scipy.io.loadmat('data_train.mat')['data_train']
label = scipy.io.loadmat('label_train.mat')['label_train']

# 按照 8：2分割数据集
data_train = data[0:int(data.shape[0] * 0.8), :]
label_train = label[0:int(data.shape[0] * 0.8), :]

data_test = data[int(data.shape[0] * 0.8):, :]
label_test = label[int(data.shape[0] * 0.8):, :]

# 计算hidden layer参数
n_neurons = 4
centers = KMeans(n_clusters=n_neurons).fit(data_train).cluster_centers_  # hidden layer neurons centers found by K-means
sigma = n_neurons * np.nanmax(squareform(pdist(centers))) / np.sqrt(2*n_neurons)  # sigma value in hidden layer neurons


class RBFNN(object):
    def __init__(self, data_train, label_train, neurons, center, sigma):
        self.data_train = data_train
        self.label_train = label_train
        self.n_neurons = neurons
        self.centers = center
        self.sigma = sigma

    def hidden_layer(self, input_data):
        hidden_output = np.zeros((input_data.shape[0], self.n_neurons))
        for i in range(input_data.shape[0]):
            d = np.sum(np.power(np.tile(input_data[i], (self.n_neurons, 1)) - self.centers, 2), axis=1)
            o = np.exp(-1 * d / (2 * self.sigma ** 2))  # 单个数据的hidden layer输出
            hidden_output[i] = o
        fai = np.column_stack((np.ones((input_data.shape[0], 1)), hidden_output))  # 增加bias unit
        return fai

    def output_layer(self, fai):
        w = self.train()
        return np.dot(fai, w)

    def train(self):
        fai = self.hidden_layer(self.data_train)
        w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(fai), fai)), np.transpose(fai)),
                   self.label_train)  # use normal equation to compute the weight
        return w


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def classifier(x):
    prediction = sigmoid(x)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = -1
    return prediction


def accuracy(data, label):
    acc = sum(data == label) / label.shape[0] * 100
    return acc


model = RBFNN(data_train, label_train, n_neurons, centers, sigma)
w = model.train()
test_fai = model.hidden_layer(data_test)
test_output = model.output_layer(test_fai)
test_prediction = classifier(test_output)
test_accuracy = accuracy(test_prediction, label_test)

train_fai = model.hidden_layer(data_train)
train_output = model.output_layer(train_fai)
train_prediction = classifier(train_output)
train_accuracy = accuracy(train_prediction, label_train)

print('test accuracy:', test_accuracy[0], '%', '| train accuracy:', train_accuracy[0], '%')


test = scipy.io.loadmat('data_test.mat')['data_test']
fai = model.hidden_layer(test)
output = classifier(model.output_layer(fai))