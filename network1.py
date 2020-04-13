import numpy as np
import random
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """返回神经网络的输出如果a是输入"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs):
            """我想迭代多少次，每一次都需要随机重排列"""
            random.shuffle(training_data)
            """将training_date划分成小片，一片片的"""
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range[0, n, mini_batch_size]]
            """依次更新每个小切片"""
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            """看看是否有测试数据"""
            if test_data:
                print("测试数据了")
            else:
                print("Epoch %d complete", j)

    def update_mini_batch(self, mini_batch, eta):
        """初始化和偏置矩阵和权重矩阵一样的矩阵，待更新"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            """利用BP算法算出梯度"""
            delta_nabla_b, delta_nabla_w = self.backpro(x, y)
            """拿到梯度"""
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            """更新"""
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]

    def backpro(self, x, y):
        """返回一个tuple(nabla_b, nabla_w) 表示代价函数C_x的梯度。
        nabla_b和nabla_w是numpy数组的逐层列表，
        类似于self.baises 和self.weights"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 正向传播
        # 激活元
        activation = x  # 输入元
        activations = [x]  # 逐层存储所有的激活元
        zs = []  # 存储z=wx+b 向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activations)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].tranpose())

        # l=1是倒数第一层，l=2倒数第二层
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l + 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        #np.argmax(self.feedforward(x))找出最大值,即预测结果的index，和y放在一起
        test_results = [(np.argmax(self.feedforward(x)), y)for (x, y) in test_data]
        #比较result(预测值，标签)是否相等
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))
