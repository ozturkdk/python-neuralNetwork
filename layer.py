import numpy as np

class Layer():
    def __init__(self):
        self.input = None

class DenseLayer(Layer):
    def __init__(self, neurons_in, neurons_out):
        Layer.__init__(self)
        self.weights = np.random.rand(neurons_out, neurons_in) - 0.5
        self.bias = np.zeros([neurons_out, 1])

    def feedForward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def getGradient(self, input):
        if input is self.weights:
            return self.input
        if input is self.input:
            return self.weights
        if input is self.bias:
            return np.ones((len(self.bias), len(self.bias)))
            
class ActivationLayer(Layer):
    def __init__(self, activation):
        Layer.__init__(self)
        self.activation = activation
        
    def feedForward(self, input):
        self.input = input
        if self.activation == 'sigmoid':
            return self.sigmoid(self.input)

    def getGradient(self, input):
        if self.activation == 'sigmoid':
            return self.sigmoidDerivative(input)

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoidDerivative(self, input):
        return input * (1 - input)

class LossLayer(Layer):
    def __init__(self, loss, target):
        Layer.__init__(self)
        self.loss = loss
        self.target = target

    def feedForward(self, input):
        self.input = input
        if self.loss == "mse":
            return self.mse(self.target, input)

    def getGradient(self, input):
        if self.loss == "mse":
            return self.mseDerivative(self.target, input)

    def mse(self, target, input):
        return 1/2 * (target - input) ** 2

    def mseDerivative(self, target, input):
        return target - input