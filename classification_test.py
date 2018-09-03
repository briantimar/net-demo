import numpy as np
from NNbase import FCNetwork, BatchGen

import matplotlib.pyplot as plt
plt.ion()

max_samples=1000

data_test = np.genfromtxt("mnist_test.csv",delimiter=',', max_rows=max_samples)
data_tr = np.genfromtxt("mnist_test.csv", delimiter=',', max_rows=max_samples)

ntest, ntr = data_test.shape[0], data_tr.shape[0]

labels_test = data_test[:, 0]
images_test = data_test[:, 1:].reshape((ntest, 28,28))
labels_tr = data_tr[:, 0]
images_tr = data_tr[:, 1:].reshape((ntr, 28,28))

#normalize the images
def normalize_mu_sigma(im_array):
    n,x,y = im_array.shape
    mu, sigma = np.mean(im_array, axis=(1,2)), np.std(im_array, axis=(1,2))
    mu = np.repeat( np.repeat( mu.reshape(n, 1, 1), x, axis=1), y, axis=2)
    sigma = np.repeat( np.repeat( sigma.reshape(n, 1, 1), x, axis=1), y, axis=2)
    return (im_array - mu) / sigma

def maxnormalize(im_array):
    n,x,y = im_array.shape
    mu = np.mean(im_array, axis=(1,2))
    mu = np.repeat( np.repeat( mu.reshape(n, 1, 1), x, axis=1), y, axis=2)
    im_array -= mu
    max = np.max(np.abs(im_array), axis=(1,2))
    max = np.repeat( np.repeat( max.reshape(n, 1, 1), x, axis=1), y, axis=2)
    im_array /= max
    return im_array

def digit_onehot_embedding(y):
    y_onehot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        y_onehot[i,int(y[i])] = 1
    return y_onehot

#images_test = normalize_mu_sigma(images_test)
#images_tr = normalize_mu_sigma(images_tr)
images_test= maxnormalize(images_test)
images_tr = maxnormalize(images_tr)


Nsamp=100
x,y = images_tr[:Nsamp, :, :], labels_tr[:Nsamp ]
x = np.reshape(x, (Nsamp, 28 * 28))
y = digit_onehot_embedding(y)

num_layers = 2
layer_sizes = [784, 256, 10]
activations = [ 'relu', 'softmax']
fcnet = FCNetwork(num_layers)
fcnet.set_layer_sizes(layer_sizes)
fcnet.set_cost_function('dkl')
fcnet.set_activation_functions(activations)
fcnet.initialize_all()
batch_size =32
epochs=1000
batchgen = BatchGen(x, y, batch_size).get_batches
x,y=batchgen()
xb,yb = x[0], y[0]

# plt.hist(fcnet.weights[2].flatten())
costs, acc=fcnet.train_SGD(batchgen, epochs, lr=1e-2)
plt.plot(costs)
plt.plot(acc)
