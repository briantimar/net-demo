import numpy as np

def dkl(ybatch, y, onehot=True):
    """ the kl divergence between a batch and classifier network's output.
        Each should have the shape nbatch, m
        where m is the number of classes"""
    if onehot:
        return -np.mean( np.sum(ybatch * np.log(y), axis=1))
    else:
        return np.mean( np.sum( ybatch * ( np.log(ybatch) - np.log(y)) , axis=1 ))

def accuracy(ybatch, y):
    nbatch,k=ybatch.shape
    maxvals = np.repeat(np.max(y, axis=1).reshape((nbatch,1)), k, axis=1)
    return np.mean(np.sum(ybatch * (y==maxvals), axis=1))


class FCNetwork:
    """fully connected network"""

    nl_functions = {'relu': lambda x: np.maximum(0, x),
                    'sigmoid': lambda x : 1.0 / (1.0 + np.exp(-x)),
                    'softmax': lambda x: np.exp(x)/ np.sum(np.exp(x))}
    nl_gradients = {'relu': lambda x, y: np.array(x>0, dtype=np.float32),
                    'sigmoid': lambda x, y: y * (1 - y)}

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.weights=[None] * (num_layers+1)
        self.biases = [None] * (num_layers+1)
        self.trainable = dict(weights=self.weights,
                                biases=self.biases)

        #stores the size of the layers
        self.sizes= [None] * (num_layers + 1)
        #stores the values of the local gradients
        self.local_gradients = dict()
        #stores the values of the full gradients
        self.gradients = dict([(i+1, dict()) for i in range(num_layers)])
        #stores the values of the neuron outputs
        self.y = dict()
        #stores the nonlinearity types
        self.nonlinearities=dict()
        self.cost_function = None

    def set_cost_function(self, f):
        """f = a string specifying the cost function to be used.
            Allowed values:  'dkl' """
        self.cost_function = f

    def set_layer_sizes(self, sizes):
        """ sizes = a list of integers specifiying the size of each layer.
                    length should be equal to num_layers +1
                    the zeroth element is the size of the input layer"""
        self.sizes = sizes

    def set_activation_functions(self, fn_list):
        for i in range(self.num_layers):
            self.nonlinearities[i+1] = fn_list[i]

    def initialize_weights(self, i, how='xavier'):
        """ initializes the weight matrix for layer i"""
        nprev, ncur = self.sizes[i-1], self.sizes[i]
        if how == 'xavier':
            weights = np.random.normal(size=(nprev, ncur)) / np.sqrt(nprev)
        else:
            raise NotImplementedError
        self.weights[i] = weights

    def initialize_biases(self, i, how='zero'):
        n = self.sizes[i]
        if how=='zero':
            b = np.zeros(n)
        else:
            raise NotImplementedError
        self.biases[i] = b

    def initialize_all(self, weight_init='xavier', bias_init='zero'):
        for i in range(1,self.num_layers+1):
            self.initialize_weights(i, how=weight_init)
            self.initialize_biases(i, how=bias_init)

    def set_weights(self,  w,i):
        """ sets the ith layer of weights to w.
            dimension of the weight matrix:
                <n i-1, n i>
                where n i-1, ni are the numbers of units in the previous and current layer resp."""
        if i < 1:
            raise ValueError("the first layer is reserved for data")
        self.weights[i] = w

    def set_bias(self, b,i):
        if i < 1:
            raise ValueError("the first layer is reseved for data")
        self.biases[i] = b

    def get_weights(self, i):
        return self.weights[i]

    def get_bias(self, i):
        return self.biases[i]

    def get_weights_grad(self, i):
        return self.gradients[i]['weights']
    def get_bias_grad(self, i):
        return self.gradients[i]['bias']


    def get_nonlinearity(self, i):
        s = self.nonlinearities[i]
        return self.nl_functions[s]

    def get_activation(self, inputs, i):
        """ given inputs <nsamp, n> from the previous layer i-1, compute the activation seen by layer i """
        W, b = self.get_weights(i), self.get_bias(i)
        nsamp = inputs.shape[0]
        b = np.repeat( np.reshape(b, (1, b.shape[0])), nsamp, axis=0)
        return np.dot(inputs, W) + b

    def apply_nonlinearity(self, activation, i):
        f = self.get_nonlinearity(i)
        return f(activation)

    def get_local_gradient(self, activation, y, i):
        """ Returns the derivative of the nonlinearity of layer i, evaluated on activation.
                y is the output of layer i"""
        nl = self.nonlinearities[i]
        grad = self.nl_gradients[nl]
        return grad(activation, y)


    def forward_pass(self, batch):
        """ does a single forward pass thru the network with the data batch provided.
            shape of batch: (Nbatch, N0)
            where Nbatch is the number of samples, and N0 the size of the input layer.

            output of the network is stored in the output ('y') of the final layer
            """
        Nbatch, n0 = batch.shape
        if n0 != self.sizes[0]:
            raise ValueError("data provided is wrong shape")
        inputs = batch
        self.y[0] = batch
        for i in range(1, self.num_layers+1):
            #get the acivation for the ith layer
            activation =self.get_activation(inputs, i)
            #print(i, " activation ", activation)
            #apply nonlinearity to it
            y = self.apply_nonlinearity(activation, i)
            #print(i, "y", y)
            #store the local gradient value for use in backprop
            #not needed for the last layer, its local gradient is packaged into the 'terminal gradient' computation
            if i < self.num_layers:
                self.local_gradients[i] = self.get_local_gradient(activation, y, i)
            self.y[i]=y
            inputs = y

    def backward_pass(self, ybatch):
        """ Runs a single backward pass of the network, using the training y-values of ybatch.
            """
        #gradients of the cost function WRT the inputs of the final layers
        terminal_grads = self.get_terminal_grads_by_x(ybatch)
        #holds the gradients of the cost function WRT the inputs of a particular layer of neurons
        grads_by_x = terminal_grads
        for i in range(self.num_layers, 0, -1):
            #store the gradients WRT the weights and biases of this layer
            self.update_weights_grad(grads_by_x, self.y[i-1], i)
            self.update_bias_grad(grads_by_x, i)
            #pass the gradient on to the next layer
            if i>1:
                grads_by_x = self.propagate_x_gradient(grads_by_x, i)

    def propagate_x_gradient(self, grads_by_x, i):
        """ Returns the gradient df / dx [i-1], given its value on the ith layer
            grads_by_x will have shape nsamp, ni, where nsamp is the number of batch samples and ni the number
            of neurons in layer i"""
        local_deriv = self.local_gradients[i-1]  #shape nsamp, ni-1
        W = self.get_weights(i)                  #shape n-1, ni

        dX= local_deriv * np.dot(grads_by_x, np.transpose(W))
        return dX

    def update_weights_grad(self, grads_by_x, yprev,i):
        """Updates the weight gradient for layer i.
                    grads_by_x = grad WRT activation x at layer i
                    yprev = the vector of output values from the previous layer i-1
                    """
        dW = self.compute_weights_grad(grads_by_x, yprev)
        self.gradients[i]['weights'] = dW

    def _get_terminal_softmax_grads_by_x(self, y):
        """Assumes that the y-vectors are one-hot labels with m classes.
            Shape of y: nbatch, m
            Assumes a KL ( or crossentropy) loss function"""
        if self.cost_function != 'dkl':
            raise ValueError("cost function {0} not implemented for softmax".format(self.cost_function))
        yself = self.y[self.num_layers]
        if y.shape != yself.shape:
            raise ValueError
        return (yself - y)

    def get_terminal_grads_by_x(self, ybatch):
        """ Returns the gradients d C / dx for the cost function defined on the
            current minibatch."""
        f = self.nonlinearities[self.num_layers]
        batch_gradient_fns = { 'softmax': self._get_terminal_softmax_grads_by_x}
        return batch_gradient_fns[f](ybatch)

    def get_cost_function(self, ybatch):
        f = self.nonlinearities[self.num_layers]
        y = self.y[self.num_layers]
        batch_cost_functions = {'softmax': lambda ybatch, y: dkl(ybatch, y)}
        return batch_cost_functions[f](ybatch, y)

    def update_bias_grad(self, grads_by_x, i):
        db = self.compute_bias_grad(grads_by_x)
        self.gradients[i]['bias'] = db

    def compute_weights_grad(self, grads_by_x, yprev):
        """ computes the gradient WRT weight matrix W of layer i
                grads_by_x: the gradients with respect to the activation ('x') of a particular layer.
                                shape: nsamp, Ni
                yprev:   the output values of the previous laeyer
                                shape: nsamp, Ni-1"""
        return np.einsum('ij,ik->ijk',yprev, grads_by_x)

    def compute_bias_grad(self, grads_by_x):
        """ computes the gradient WRT the bias vector of layer i"""
        return grads_by_x

    def compute_cost_function(self, ybatch):
        """evaluates the cost function, averaged over a batch. compares the current
            network output with the values in ybatch. """
        return self.get_cost_function(ybatch)

    def compute_accuracy(self, ybatch):
        return accuracy(ybatch, self.y[self.num_layers])

    def update_weights_SGD(self, lr):
        """ performs a gradient-descent update of the weights with learning rate lr"""
        for i in range(1, self.num_layers+1):
            dW, db = self.get_weights_grad(i), self.get_bias_grad(i)
            W, b = self.get_weights(i), self.get_bias(i)
            W -= lr * np.mean(dW, axis=0)
            b -= lr * np.mean(db, axis=0)
            self.set_weights(W, i)
            self.set_bias(b, i)


    def train_SGD(self, batch_gen, epochs, lr=0.1):
        costs=[]
        accuracy=[]
        for ep in range(epochs):
            batches_x, batches_y = batch_gen()
            for x, y in zip(batches_x, batches_y):
                #set the input and push forward
                self.forward_pass(x)
                #go backwards and get the gradients
                self.backward_pass(y)
                #compute the batch-cost
                cost = self.compute_cost_function(y)
                acc = self.compute_accuracy(y)
                #update the weights
                self.update_weights_SGD(lr)
                costs.append(cost)
                accuracy.append(acc)
            print("epoch {0}, cost function {1},accuracy {2}".format(ep, cost, acc))
        return costs, accuracy


class BatchGen:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.mx, self.my = x.shape[1], y.shape[1]
        self.nsamp= x.shape[0]
        self.all = np.concatenate([x, y], axis=1)
        self.batch_size = batch_size

    def get_batches(self):
        np.random.shuffle(self.all)
        xb, yb = self.all[:, :self.mx], self.all[:,self.mx:]
        batches_x = [xb[i* self.batch_size: (i+1) * self.batch_size] for i in range(self.nsamp//self.batch_size)]
        batches_y = [yb[i* self.batch_size: (i+1) * self.batch_size] for i in range(self.nsamp//self.batch_size)]
        return batches_x, batches_y
