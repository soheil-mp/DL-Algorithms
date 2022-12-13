
# Import the libraries
import numpy as np
from testCases_v4a import *

########################
#    NEURAL NETWORK    #
########################

# Class for the Neural Network
class NeuralNetwork():
    
    #####################
    #    CONSTRACTOR    #
    #####################
    
    # Constructor function
    def __init__(self):
        
        pass
    
    ##################################
    #    PARAMETER INITIALIZATION    #
    ##################################
    
    # Function for initializing the parameters of the model with L layers
    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        # Set the seed for the random generator
        np.random.seed(3)
        
        # Initialize the parameters of the model
        parameters = {}
        
        # Number of layers in the network
        L = len(layer_dims)     
        
        # Loop over the layers
        for l in range(1, L):
            
            # Initialize the parameters of the model
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
            
            # Check if the parameters have the correct dimensions
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1)) 
            
        return parameters  
    
    #####################
    #    FEEDFORWARD    #
    #####################
    
    # Function for implementing the forward propagation   
    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation. 
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1) 
        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        # Implement the linear forward propagation
        Z = np.dot(W,A) + b
        
        # Check if the parameters have the correct dimensions
        assert(Z.shape == (W.shape[0], A.shape[1]))
        
        # Store the parameters in a dictionary
        cache = (A, W, b)
        
        return Z, cache  
    
    # Function for implementing the linear forward propagation 
    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer 
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" 
        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        
        # If the activation is sigmoid
        if activation == "sigmoid":
            
            # Linear forward propagation
            Z, linear_cache = self.linear_forward(A_prev, W, b) 
            
            # Sigmoid activation function     
            A, activation_cache = self.sigmoid(Z) 
            
        # If the activation is ReLU
        elif activation == "relu":
            
            # Linear forward propagation
            Z, linear_cache = self.linear_forward(A_prev, W, b) 
            
            # ReLU activation function
            A, activation_cache = self.relu(Z) 
            
        # Check if the parameters have the correct dimensions
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        
        # Store the parameters in a dictionary
        cache = (linear_cache, activation_cache) 
        
        return A, cache
    
    # Function for L layers forward propagation
    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """ 
        
        # List to store the caches
        caches = []
        
        # Activation output (of input)
        A = X
        
        # Number of layers
        L = len(parameters) // 2                  
        
        # Loop over L-1 layers
        for l in range(1, L):
            
            # Previous activation output
            A_prev = A 
            
            # Linear forward propagation of hidden layers
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            
            # Store the cache
            caches.append(cache)
            
        # Linear forward propagation of output
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        
        # Store the cache
        caches.append(cache)
        
        # Check if the parameters have the correct dimensions
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches 
    
    #######################
    #    COST FUNCTION    #
    #######################
    
    # Function for computing the cost  
    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7). 
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples) 
        Returns:
        cost -- cross-entropy cost
        """
        
        # Number of examples
        m = Y.shape[1] 
        
        # Cross-entropy cost
        cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
        
        # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        cost = np.squeeze(cost)     
        
        # Check if the parameters have the correct dimensions 
        assert(cost.shape == ())
        
        return cost  
    
    #########################
    #    BACKPROPAGATION    #
    #########################
    
    # Function for linear backward propagation
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l) 
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer 
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        
        # Retrieve the parameters from the cache
        A_prev, W, b = cache
        
        # Number of examples
        m = A_prev.shape[1] 
        
        # Compute the gradients
        dW = (1/m) * np.dot(dZ, A_prev.T)                 
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        
        # Check if the parameters have the correct dimensions
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db  
    
    # Function for linear activation backward propagation
    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        
        # Retrieve the parameters from the cache
        linear_cache, activation_cache = cache
        
        # If the activation is relu
        if activation == "relu":
            
            # Compute the gradients
            dZ = self.relu_backward(dA, activation_cache)
            
        # If the activation is sigmoid
        elif activation == "sigmoid":
            
            # Compute the gradients
            dZ = self.sigmoid_backward(dA, activation_cache)
            
        # Compute the gradients
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db  
    
    # Function for backward propagation for L layers
    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        
        # Initialize the gradients
        grads = {}
        
        # Number of layers
        L = len(caches) 
        
        # Number of examples
        m = AL.shape[1]
        
        # Reshape Y to the same shape as AL
        Y = Y.reshape(AL.shape) 
        
        # Compute the gradient of the cost with respect to AL
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Get the cache of the last layer
        current_cache = caches[L-1] 
        
        # Backward propagation for the last layer
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        
        # Loop over layers (reversed)
        for l in reversed(range(L-1)):
            
            # Get the cache of the current layer
            current_cache = caches[l]
            
            # Apply the linear activation backward function
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            
            # Set the gradients for the current layer
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        return grads 
    
    # Function for updating parameters
    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """
        
        # Number of layers
        L = len(parameters) // 2  
        
        # Loop over layers
        for l in range(L):
            
            # Update the parameters
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
        return parameters  
    
    #######################
    #    ACTIVATION FUNCTION    #
    #######################
    
    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy
        
        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(self, Z):
        """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache


    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ

    
    
# Initialize the neural network
neural_network = NeuralNetwork()
    
# Initialize the parameters    
parameters = neural_network.initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))      
    
# Feedforward 1
A, W, b = linear_forward_test_case() 
Z, linear_cache = neural_network.linear_forward(A, W, b)
print("Z = " + str(Z))     
    
# Feedforward 2
A_prev, W, b = linear_activation_forward_test_case() 
A, linear_activation_cache = neural_network.linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A)) 
A, linear_activation_cache = neural_network.linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))      
    
# Feedforward 3
X, parameters = L_model_forward_test_case_2hidden()
AL, caches = neural_network.L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))      
    
# Cost function
Y, AL = compute_cost_test_case() 
print("cost = " + str(neural_network.compute_cost(AL, Y)))      
    
# Backpropagation 1
dZ, linear_cache = linear_backward_test_case() 
dA_prev, dW, db = neural_network.linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))     
    
# Backpropagation 2
dAL, linear_activation_cache = linear_activation_backward_test_case() 
dA_prev, dW, db = neural_network.linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n") 
dA_prev, dW, db = neural_network.linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))      
    
# Backpropagation 3
AL, Y_assess, caches = L_model_backward_test_case()
grads = neural_network.L_model_backward(AL, Y_assess, caches)
print_grads(grads)  
    
# Update parameters
parameters, grads = update_parameters_test_case()
parameters = neural_network.update_parameters(parameters, grads, 0.1) 
print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # Constructor function
    # def __init__(self):
        
    #     # Seed the random number generator
    #     np.random.seed(1)
        
    #     # Weights for the 3x2 matrix
    #     self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        
    # # Sigmoid function
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))
    
    # # Sigmoid derivative
    # def sigmoid_derivative(self, x):
    #     return x * (1 - x)
    
    # # Train the neural network
    # def train(self, training_inputs, training_outputs, training_iterations):
    #     for iteration in range(training_iterations):
    #         # Pass the training set through the network
    #         output = self.think(training_inputs)
            
    #         # Calculate the error
    #         error = training_outputs - output
            
    #         # Multiply the error by the input and again by the gradient of the sigmoid curve
    #         adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            
    #         # Adjust the weights
    #         self.synaptic_weights += adjustments
    
    # # The neural network thinks
    # def think(self, inputs):
    #     # Pass inputs through the neural network
    #     inputs = inputs.astype(float)
    #     output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        
    #     return output