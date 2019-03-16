import numpy as np
import time
import random


"""
neuralnet.py
~~~~~~~~~~

The functions in this module will allow for the creation of a
feed-forward neural network with standard sigmoidal units.
The weights for each node will be optimized with a stochastic gradient descent
function (the derivatives will be calculated with backpropagation)
and the network will be trained with cross-validation on a training set.
The inputs will be the network architechure (number of input nodes, hidden layer nodes,
and output nodes), learning rate, and training data.
Note: This code is based off of the nerual network code provided at:
http://neuralnetworksanddeeplearning.com/chap1.html and https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
"""
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return np.exp(-z)/((1+np.exp(-z))**2)

class Network:

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in each layer
        of the network. Since I am creating a three layer network, the list
        should be three layers specifying the size of the input, hidden, and output
        layer, respectively.
        The biases and weights for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1 for the biases and mean 0, and
        variance 1/(# inputs)**0.5 for the weights. Note that the first
        layer is assumed to be an input layer.
        """
        #self.input      = x
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = [] # will record the cost for all inputs in training data
        self.cost_overall = []
        #self.y          = y
        #self.output     = np.zeros(y.shape)

    def default_weight_initializer(self):
        #Initialize the bias for all the hidden layers and output layer
        #as a y x 1 array where y = # neurons in the layer
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        #Initialize the weights for all the hidden layers and output layer
        #as a y x n array where y = # neurons in the layer n = # input nodes
        self.weights = [np.random.randn(y, n)/np.sqrt(n)
                        for y, n in zip(self.sizes[1:], self.sizes[:-1])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input by feeding it
        through each layer.
        """
        for b, w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def cost_fn(self, a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        #return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        return np.sum(np.nan_to_num(y*np.log(a)+(1-y)*np.log(1-a)))

    def cost_derivative(self, a, y):

        return np.nan_to_num((a-y)/(a*(1-a)))

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        #Intialize the gradient values as zero
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x #input
        activations = [] # list to store all the activations, layer by layer
        activations.append(activation)
        zs = [] # list to store all the z vectors (weighted inputs), layer by layer
        for i in range(len(self.biases)):
            #print(activation)
            #Go through each layer and store the activations and weighted inputs (z)
            z = np.dot(self.weights[i], activation) + np.array(self.biases[i]) #Calculate weighted input
            zs.append(z)
            activation = sigmoid(z) #Calculate output
            activations.append(activation)
        # backward pass
        self.cost.append(self.cost_fn(self.feedforward(activations[0]), y)) #-1
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta, train_len):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)`` and ``eta``
        is the learning rate,
        """
        #Initialize update matrices
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #cost_mini_holder = []
        for pair in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(pair[0], pair[1])
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #cost_mini_holder.append(cost)

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        #return np.mean(cost_mini_holder)
    def SGD(self, training_data, epochs, mini_batch_size, eta): #test_data=None
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. Epochs designates how many times you will pass through
        the training set. Mini_batch_size designates the size of the sample
        of the training data that the model will iterate through before
        updating the weights and biases. Eta is the learning rate
        If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """

        start = time.time()

        n = len(training_data)
        #cost_avg = []
        for j in range(epochs): #for each epoch
            self.cost = []
            random.shuffle(training_data) #shuffle the training data

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #partition data into mini-batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, n) #update weights
            self.cost_overall.append(-1/n * np.sum(self.cost))
            if j > 2 and abs(self.cost_overall[-1] - self.cost_overall[-2]) < 0.00001:
                end = time.time()
                print("Epochs complete in:", (end-start)/60, " min")
                return
        end = time.time()
        print("Epochs complete in:", (end-start)/60, " min")
    #This was written specifically to use for the Rap1 training data
    def cross_val(self, pos_data, neg_data, epochs, mini_batch_size, eta, k):
        self.accuracy = []
        random.shuffle(pos_data)
        pos_batch_size = len(pos_data) // k
        pos_chunks = [pos_data[j:j+pos_batch_size] for j in range(0, len(pos_data), k)]
        random.shuffle(neg_data)
        neg_batch_size = len(neg_data) // k
        neg_chunks = [neg_data[j:j+neg_batch_size] for j in range(0, len(neg_data), k)]
        for i in range(k):
            pos_data_holder = []
            neg_data_holder = []
            self.default_weight_initializer()
            pos_test = pos_chunks[i]
            neg_test = neg_chunks[i]
            for x in list(set(list(range(k))) - set([i])):
                pos_data_holder = pos_data_holder + pos_chunks[x]
                neg_data_holder = neg_data_holder + neg_chunks[x]
            train_data = pos_data_holder + neg_data_holder
            self.SGD(train_data, epochs, mini_batch_size, eta)
            self.accuracy.append(self.evaluate(pos_test+neg_test))

    def evaluate(self, test_data):
        """Return the % of test inputs for which the neural
        network outputs the correct result. """
        test_results = [([round(i[0]) for i in self.feedforward(x)][0], y) for (x, y) in test_data]
        accuracy = [(x,y) for (x,y) in test_results if x == y]

        return len(accuracy)/len(test_data)
    def evaluate_8_3_8(self, test_data):
        """Return the otuput for the 8x3x8 encoder"""
        test_results = [([round(i[0]) for i in self.feedforward(x)], y) for (x, y) in test_data]


        return test_results

    def output(self, test_data):
        """
        Return the original sequence and the probability of it being a Rap1 sequence
        """
        vec_2_base = {tuple([1,0,0,0]):"A",tuple([0,1,0,0]):"T", tuple([0,0,1,0]):"C", tuple([0,0,0,1]):"G"}
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        sequences = []
        output = []
        for pair in test_data:
            sequence = ''
            for i in range(0,68,4):
                index = pair[0][i:i+4]
                test = []
                for j in index:
                    test = test + list(j)
                sequence = sequence + vec_2_base[tuple(test)]
            sequences.append(sequence)
        for k in range(len(test_data)):
            output.append([sequences[k],test_results[k][0][0][0]])
        return output
