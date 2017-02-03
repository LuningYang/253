import numpy as np
from sklearn import preprocessing
from mnist import MNIST
from matplotlib import pyplot as plt
import random
import time
import math

class NeuralNet:
    def __init__(self, layers, activation, lr):
        """
        initialize the class
        :param layers: the number of nodes for each layer
        :param activation: activation function of hidden layer
        :param lr: learning rate
        """
        self.depth = len(layers)
        self.activation = activation
        # hidden activation
        self.weights = []
        self.outputs = [0] * self.depth
        self.moment = [0] * (self.depth-1)
        self.a = [0] * self.depth
        self.lr = lr
        self.delta = []
        for i in range(self.depth - 1):
            # avoid all-positive parameters
            #self.weights.append(np.random.rand(layers[i] + 1, layers[i + 1]) - 0.5) #this initialization is for Problem 3
            self.weights.append(np.random.normal(0, 1.0/math.sqrt(layers[i]+1),(layers[i]+1)*layers[i+1]).reshape(layers[i] + 1, layers[i + 1]))
            self.delta.append(np.zeros((layers[i] + 1, layers[i+1])))

    def feedforward(self, data):
        """
        this function performs feedforward process
        :param data: 
        :return: the fitted probabilities of the output layer
        """
        self.outputs[0] = data
        for i in range(self.depth - 1):
            # add bias item
            ones = np.ones((len(self.outputs[i][:]), 1))
            self.outputs[i] = np.hstack((self.outputs[i][:], ones))
            self.a[i + 1] = np.dot(self.outputs[i][:], self.weights[i])
            # activated for next layer
            if i < self.depth - 2:
                if self.activation == 'logistic':
                    self.outputs[i+1] = sigmoid(self.a[i+1])
                elif self.activation == 'tanh':
                    self.outputs[i+1] = np.tanh(self.a[i+1])
            else:
                self.outputs[i + 1] = softmax(self.a[i + 1])
        return self.outputs[-1][:]

    def backprop_moment(self, data, target, lamb):
        """
        backpropagation using momentum
        :param data: feature matrix
        :param target: true label matrix
        :param lamb: learning rate for momentum term
        :return: updated moment to be used for next iteration
        """
        self.feedforward(data)
        pre_delta = 0
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                delta = target - self.outputs[i + 1]  # n*k
                self.delta[i] = delta[:]
                pre_delta = np.sum(delta[:], axis=0) / len(data)
                pre_delta = np.hstack((pre_delta, [1]))
                moment = lamb * self.moment[i] + self.lr * np.dot(self.outputs[i][:].T, self.delta[i]) / len(data)
                self.moment[i] = moment
            else:
                if self.activation == 'logistic':
                    delta = (self.outputs[i+1]*np.subtract(1.0, self.outputs[i+1])) * \
                            np.dot(self.weights[i+1], pre_delta.T[:-1])
                elif self.activation == 'tanh':
                    delta = (np.divide(np.sinh(self.outputs[i+1]), np.cosh(self.outputs[i+1]))) * \
                            np.dot(self.weights[i + 1], pre_delta.T[:-1])
                # output of this layer
                self.delta[i] = delta
                pre_delta = np.sum(delta, axis=0) / len(data)
                moment = lamb * self.moment[i] + self.lr * np.dot(self.outputs[i][:].T, self.delta[i][:, :-1]) / len(data)
                self.moment[i] = moment
        for i in range(self.depth-1):
            self.weights[i] += self.moment[i]
        return moment

    def backprop(self, data, target):
        """
        backpropagation using gradient descent
        :param data: feature matrix
        :param target: true label matrix
        :return: 
        """
        self.feedforward(data)
        #compute the deltas for each layer
        for i in reversed(range(self.depth - 1)):
            if i == self.depth - 2:
                delta = target-self.outputs[i+1][:]
                # n*k matrix, delta matrix for output layer
                self.delta[i] = delta
                pre_delta = np.sum(delta, axis=0)/len(data)
                # 1*k, sum of delta for n data
                pre_delta = np.hstack((pre_delta, [1]))
            else:
                if self.activation == 'logistic':
                    delta = (self.outputs[i+1]*np.subtract(1.0, self.outputs[i+1])) * \
                            np.dot(self.weights[i+1], pre_delta.T[:-1])
                elif self.activation == 'tanh':
                    delta = (np.divide(np.sinh(self.outputs[i+1]), np.cosh(self.outputs[i+1]))) * \
                            np.dot(self.weights[i + 1], pre_delta.T[:-1])
                # output of this layer
                self.delta[i] = delta
                pre_delta = np.sum(delta, axis=0)/len(data)
        #update weights        
        for i in range(self.depth-1):
            if i < self.depth-2:
                self.weights[i] += self.lr * np.dot(self.outputs[i][:].T, self.delta[i][:, :-1]) / len(data)
                continue
            self.weights[i] += self.lr * np.dot(self.outputs[i][:].T, self.delta[i]) / len(data)

    def predict(self, data, label):
        """
        :param data: feature matrix of data to be predicted
        :param label: the true label matrix
        :return: prediction accuracy
        """
        prob = self.feedforward(data)
        # predict data using the updated weights
        prob = np.argmax(prob, axis=1)#get the outputs from softmax probabilities
        return np.sum(label.flatten() == prob) * 1.0 / len(label)
        def compute_loss(self, outputs, targets):
        return -np.sum(np.sum(targets * np.log(outputs), axis=0)) / len(targets)
    def compute_loss(self, outputs, targets):
        """
        compute the cross entropy loss
        :param outputs: the probabilities for the ouput layer
        :param targets: true label matrix
        :return: cross entropy loss
        """
        return -np.sum(np.sum(targets * np.log(outputs), axis=0)) / len(targets)
    
    def test(self, data, targets):
        """
        check gradient descent using numerical approximation
        :param data: feature matrix
        :param targets: true label matrix
        :return: the matrix of gradient of each weight obtained by numerical approximation
        """
        approximation = []
        for i in range(self.depth-1):
            approximation.append(np.copy(self.weights[i]))
        epsilon = 0.01
        for layer in range(self.depth-1):
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    self.weights[layer][i,j] += epsilon
                    self.feedforward(data)
                    loss1 = self.compute_loss(self.outputs[-1], targets)
                    self.weights[layer][i,j] -= 2*epsilon
                    self.feedforward(data)
                    loss2 = self.compute_loss(self.outputs[-1], targets)
                    d = (loss1-loss2)/(2*epsilon)
                    self.weights[layer][i,j] = approximation[layer][i,j]
                    approximation[layer][i,j] = d
        return approximation

    def train(self, train_data, train_label, val_data, val_label, test_data, test_label,
              n_iter, b_size, lamb, momentum=False, test_weight=False):
        """

        :param train_data: feature matrix of training set with dimension of n*784
        :param train_label: vector of length n, indicating labels for training data
        :param val_data: feature matrix of hold-out set
        :param val_label: label vector for hold-out set
        :param test_data:feature matrix of test set
        :param test_label: label vector for test set
        :param n_iter: times of iteration
        :param b_size: batch size, if using full batch, then this equals to 1
        :param lamb: momentum learning rate if using momentum
        :param momentum: if true, use momentum
        :param test_weight: if true, check gradient descent using numerical approximation method
        :return: train accuracy, validation accuracy(hold out set), test accuracy
        """
        train_acc = []
        valid_acc = []
        test_acc = []
        pre_weights = []#for numerical comparison
        for i in range(self.depth-1):
            pre_weights.append(np.copy(self.weights[i]))
        for i in range(n_iter):
            t = multi_label(train_label)#change label vector to label matrix with n*10 dimension
            n_batch = len(train_data) / b_size
            for j in range(n_batch):#use backpropagation to update weights using all batches
                b_data = train_data[j * b_size:(j + 1) * b_size, :]
                b_t = t[j * b_size:(j + 1) * b_size, :]
                approximation = self.test(b_data, b_t)#for comparing numerical approximation and gradient
                if momentum:
                    self.backprop_moment(b_data, b_t, lamb)
                else:
                    self.backprop(b_data, b_t)
            train_acc.append(self.predict(train_data, train_label))
            valid_acc.append(self.predict(val_data, val_label))
            test_acc.append(self.predict(test_data, test_label))
            if (test_weight):#for numerical comparison
                for l in range(self.depth - 1):
                    print(np.max(np.abs(np.subtract(approximation[l], np.subtract(pre_weights[l], self.weights[l]) / self.lr)), axis=0))
            if n_iter == 20:#reduce learning rate to avoid vabrivation and shuffle data
                self.lr /= 2.0
                idx = range(len(train_label))
                random.shuffle(idx)
                train_data = train_data[idx[:], :]
                train_label = train_label[idx[:]]
            print(train_acc[-1], valid_acc[-1], test_acc[-1])
        print 'train accuracy: {}\t validation accuracy: {}\t test accuracy:{}'.format(train_acc[-1], valid_acc[-1], test_acc[-1])
        plt.figure(2)
        plt.plot(range(len(valid_acc)), valid_acc)
        plt.figure(1)
        plt.plot(range(len(train_acc)), train_acc)
        plt.show()
        return train_acc, valid_acc, test_acc


def multi_label(label):
    enc = preprocessing.OneHotEncoder()
    enc.fit([[i] for i in range(10)])
    label_vector = enc.transform(label.reshape(-1, 1)).toarray()
    return label_vector


def softmax(data):
    res = np.exp(data)
    res = res / np.sum(res, axis=1).reshape(-1, 1)
    return res


def sigmoid(data):
    data = np.exp(-data)
    return np.divide(1.0, np.add(data, 1.0))


def main():
    mndata = MNIST('../data')
    data = mndata.load_training()
    test = mndata.load_testing()
    train_feature = preprocessing.scale(1.0*np.array(data[0])/255, axis=1, with_mean=True, with_std=False)
    test_feature = preprocessing.scale(1.0*np.array(test[0])/255, axis=1, with_mean=True, with_std=False)
    train_label = np.array(data[1])
    test_label = np.array(test[1])
    # print(test_label)
    val_feature = train_feature[50000:60000, :]
    val_label = train_label[50000:60000]
    train_feature = train_feature[:50000, :]
    train_label = train_label[:50000]
    net = NeuralNet([784, 800, 10], 'tanh', 0.01)
    train_acc, val_acc, test_acc = net.train(train_feature, train_label, val_feature, val_label, test_feature, test_label,
                                             n_iter=200, b_size=128, lamb=0.01, momentum=True)


if __name__ == '__main__':
    main()
