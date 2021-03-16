import numpy as np

class Network:
    """
    Constructs a neural network according to a list detailing the network structure.

    Args:
        network_structure: List of integers, number of nodes in a layer. The entry network_structure[0]
        equals the number of input features and the last entry is 1 for binary classification.
        network_structure[i] equals the number of hidden units in layer i.
    """

    def __init__(self, network_structure):
        self.num_layers = len(network_structure) - 1
        # state dicts, use integer layer id as keys
        # the range is 0,...,num_layers for x and
        # 1,...,num_layers for all other dicts
        self.w = dict() # weights
        self.b = dict() # biases
        self.z = dict() # outputs of linear layers
        self.x = dict() # outputs of activation layers
        self.dw = dict() # error derivatives w.r.t. w
        self.db = dict() # error derivatives w.r.t. b

        self.init_wb(network_structure)


    def init_wb(self, network_structure):
        """ Initialize all parameters w[i] and b[i] for i = 1,..., num_layers of the neural network.
        Tip: If the current weight of the neuron layer is a matrix of n_i x n_(i-1), i is the 
        network layer number, and n is the number of nodes in the network layer represented by the 
        subscript i.
        For example, [3,2,4,1] is a 3-layer structure: the input size is 3, the
        number of neurons in the hidden layer 1 is 2, the number of neurons in the hidden layer 2 is 4
        and the output size of layer 3 is 1.
        Initialize weight matrices randomly from a normal distribution with variance 1 / n_(i-1).
        Initialize biases to 0.
        """
        np.random.seed(1)
        i = 0
        while (i < len(network_structure) - 1):
            layer_i = network_structure[i]
            layer_i_puls_one = network_structure[i + 1]
            self.w[i] = np.random.normal(0, np.sqrt(1/layer_i), (layer_i_puls_one, layer_i))
            self.b[i] = 0

            i = i + 1

    def sigmoid(self, z):
        """ Sigmoid function.

        Args:
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the sigmoid function
        """
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_backward(self, dx_out, z):
        """ Backpropagation for the sigmoid function.

        Args:
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the sigmoid function in layer i
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            dz_in: (n_i, b) numpy array, partial derivatives dE/dz_i of the error with respect to
                the output of the i-th linear layer
        """
        return dx_out * self.sigmoid(z) * (1 - self.sigmoid(z))


    def relu(self, z):
        """ ReLU function.

        Args:
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the ReLU function
        """
        return np.maximum(0, z)

    def relu_backward(self, dx_out, z):
        """ Backpropagation for the ReLU function.

        Args:
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the ReLU function in layer i
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            dz_in: (n_i, b) numpy array, partial derivatives dE/dz_i of the error with respect to
                the output of the i-th linear layer
        """
        dx_out[z <= 0] = 0
        return dx_out


    def activation_func(self, func, z):
        """ Select and perform forward pass through activation function.

        Args:
            func: string, either "sigmoid" or "relu"
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the ReLU function
        """
        if func == "sigmoid":
            return self.sigmoid(z)
        elif func == "relu":
            return self.relu(z)


    def activation_func_backward(self, func, dx_out, z):
        """ Select and perform backward pass through activation function.

        Args:
            func: string, either "sigmoid" or "relu"
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the activation function in layer i
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            dz_in: (n_i, b) numpy array, partial derivatives dE/dz_i of the error with respect to
                the output of the i-th linear layer
        """
        if func == "sigmoid":
            return self.sigmoid_backward(dx_out, z)
        elif func == "relu":
            return self.relu_backward(dx_out, z)


    def layer_forward(self, x_in, func, i):
        """ Forward propagation through the i-th network layer.
        Uses the states of w[i] and b[i].
        Updates the states of z[i] and x[i].

        Args:
            x_in: (n_(i-1), b) numpy array, input of the i-th linear layer
            func: string, either "sigmoid" or "relu" determining the activation
                of the i-th layer
            i: int, layer id

        Returns:
            x_out: (n_i, b) numpy array, output of the i-th linear layer
        """
        self.z[i] = (self.w[i] @ x_in) + self.b[i]
        x_out = self.activation_func(func, self.z[i])

        return x_out


    def forward(self, x):
        """ Neural network forward propagation. Use ReLU activations in all but the last layer.
        Use the sigmoid function to output class probabilities in the last layer.
        Calls layer_forward in order to update the states of z[i] and x[i] for all layers i.
        Updates the state of x[0].
    
        Args:
            x: (n_0, b) numpy array, input for the forward pass

        Returns:
            predictions: (1, b) numpy array, the network's predictions.
        """
        self.x[0] = x
        for i in range(self.num_layers - 1):
            self.x[i + 1] = self.layer_forward(self.x[i], "relu", i)
        self.x[self.num_layers] = self.layer_forward(self.x[self.num_layers - 1], "sigmoid", self.num_layers - 1)

        return self.x[self.num_layers]


    def layer_backward(self, dx_out, func, i):
        """ Backward propagation through the i-th network layer.
        Uses the states of z[i] and x[i-1], as well as w[i] and b[i].
        Updates the states of dw[i] and db[i].

        Args:
            dx_out: (n_i, b) numpy array, partial derivatives dE/dx_i of error with respect to 
                    the output of the activation function in layer i
            func: string, either "sigmoid" or "relu" determining the activation
                of the i-th layer
            i: int, layer id

        Returns:
            dx_in: (n_(i-1), b) numpy array, partial derivatives dE/dx_(i-1) of error
                with respect to the input of layer i
        """
        dz = self.activation_func_backward(func, dx_out, self.z[i - 1])
        self.dw[i - 1] = (1 / self.x[i-1].shape[1]) * (dz @ self.x[i - 1].T)
        self.db[i - 1] = (1 / self.x[i-1].shape[1]) * np.sum(dz, axis=1, keepdims=True)

        return np.dot(self.w[i - 1].T, dz)


    def back_propagation(self, y):
        """ Neural network backward propagation. Use ReLU activations in all but the last layer.
        Use the sigmoid function in the last layer.
        First, 
        Calls layer_backward in order to update the states of dw[i] and db[i] for all layers i.
    
        Args:
            y: (1, b) numpy array, labels needed to back propagate the error

        Returns:
            dx_in: (n_0, b) numpy array, partial derivatives dE/dx_0 of error
                with respect to the network's input
        """
        batch_size = y.shape[1]
        # get predictions from the state dict
        predictions = self.x[self.num_layers]
        # compute the derivative of the mean error regarding the network's output
        d_predictions = - (np.divide(y, predictions) - np.divide(1 - y, 1 - predictions)) / batch_size
        # backward pass through the output layer, updates states of dw and db for the last layer
        dx_in = self.layer_backward(d_predictions, "sigmoid", self.num_layers)
        # iteratively perform backward propagation through the network layers,
        # update states of dw and db for the i-th layer
        for i in reversed(range(1, self.num_layers)):
            dx_in =  self.layer_backward(dx_in, "relu", i)

        return dx_in


    def update_wb(self, lr):
        """ Update the states of w[i] and b[i] for all layers i based on gradient information
        stored in dw[i] and db[i] and the learning rate.

        Args:
            lr: learning rate
        """
        for i in range(self.num_layers - 1, -1, -1):
            self.w[i] = self.w[i] - lr * self.dw[i]
            self.b[i] = self.b[i] - lr * self.db[i]


    def shuffle_data(self, X, Y):
        """ Shuffles the data arrays X and Y randomly. You can use
        np.random.permutation for this method. Make sure that the label
        belonging X_shuffled[:,i] is shuffled to Y_shuffled[:,i].

        Args:
            X: (n_0, B) numpy array, B feature vectors with N_0-dimensional features
            Y: (1, B) numpy array, labels

        Returns:
            X_shuffled: (n_0, B) numpy array, shuffled version of X
            Y_shuffled: (1, B) numpy array, shuffled version of Y
        """
        perm = np.random.permutation(X.shape[1])
        x = np.zeros(X.shape)
        y = np.zeros(Y.shape)
        for idx, p in enumerate(perm):
            x[:, p] = X[:, idx]
            y[:, p] = Y[:, idx]

        return x, y


    def train(self, X, Y, lr, batch_size, num_epochs):
        """ Trains the neural network with stochastic gradient descent by calling
        shuffle_data once per epoch and forward, back_propagation and update_wb
        per iteration. Start a new epoch if the number of remaining data points
        not yet used in the epoch is smaller than the mini batch size.

        Args: 
            X: (n_0, B) numpy array, B feature vectors with N_0-dimensional features
            Y: (1, B) numpy array, labels
            lr: learning rate
            batch_size: mini batch size for SGD
            num_epochs: number of training epochs
        """
        num_examples = X.shape[1]
        it_per_epoch = num_examples // batch_size
        for _ in range(num_epochs):
            X, Y = self.shuffle_data(X, Y)
            for i in range(it_per_epoch):
                # extract mini batches
                x = X[:, i * batch_size : (i+1) * batch_size]
                y = Y[:, i * batch_size : (i+1) * batch_size]
                # perform a forward pass, update states of x and z
                _ = self.forward(x)
                # update states of dw and db by performing a backward pass
                _ = self.back_propagation(y)
                # update states of w and b by a SGD step
                self.update_wb(lr)

