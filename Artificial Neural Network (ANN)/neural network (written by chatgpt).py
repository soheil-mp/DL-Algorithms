import numpy as np

class NeuralNetwork:
  def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
    # initialize weights and biases for each layer
    self.weights = []
    self.biases = []
    for i, hidden_size in enumerate(hidden_sizes):
      if i == 0:
        self.weights.append(np.random.randn(input_size, hidden_size))
      else:
        self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_size))
      self.biases.append(np.zeros(hidden_size))
    self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
    self.biases.append(np.zeros(output_size))
    self.learning_rate = learning_rate
    
  def sigmoid(self, x):
    # apply sigmoid activation function
    return 1 / (1 + np.exp(-x))
  
  def forward(self, x):
    # forward pass through the network
    activations = []
    for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
      if i == 0:
        z = np.dot(x, weight) + bias
      else:
        z = np.dot(activations[-1], weight) + bias
      a = self.sigmoid(z)
      activations.append(a)
    return activations
  
  def backward(self, x, y, activations):
    # backward pass through the network to calculate gradients
    grad_weights = []
    grad_biases = []
    error = y - activations[-1]
    delta = error * self.sigmoid(activations[-1], deriv=True)
    grad_weights.append(np.dot(activations[-2].T, delta))
    grad_biases.append(delta)
    for i in range(2, len(self.weights)+1):
      delta = np.dot(delta, self.weights[-i+1].T) * self.sigmoid(activations[-i], deriv=True)
      grad_weights.append(np.dot(activations[-i-1].T, delta))
      grad_biases.append(delta)
    return grad_weights[::-1], grad_biases[::-1]
  
  def train(self, x, y):
    # train the network on a single example
    activations = self.forward(x)
    grad_weights, grad_biases = self.backward(x, y, activations)
    # update weights and biases using gradient descent
    self.weights = [weight - self.learning_rate * grad_weight
                    for weight, grad_weight in zip(self.weights, grad_weights)]
    self.biases = [bias - self.learning_rate * grad_bias
                   for bias, grad_bias in zip(self.biases, grad_bi
