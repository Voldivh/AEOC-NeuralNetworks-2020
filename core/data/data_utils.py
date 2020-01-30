import numpy as np
import matplotlib.pyplot as plt

def generate_data_for_regression(plot=True, num_points=1000, seed=123, function=None):
    np.random.seed(seed)
    x = np.random.random(num_points)*(1 - (-1)) + (-1) # max=1, min=-1
    if(function is None):
        y = np.square(x) + np.exp(x/2)
    else:
        y = function(x)
    y = y + np.random.normal(scale=0.07, size=num_points)
    data = np.vstack((x, y)).T
    if(plot):
        plt.figure(figsize=(7,7))
        plt.plot(x, y, 'o', markersize = 2)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y^*$')
        plt.show()
    return data

def generate_training_batches(data, batch_size=32):
    # data should be an array of shape (N, input_dim + output_dim)
    N = data.shape[0]
    data_new = data.copy()
    np.random.shuffle(data_new)
    num_batches = N/batch_size
    num_batches_int = int(num_batches)
    full_id = batch_size * num_batches_int
    residue = num_batches - num_batches_int
    ids = np.arange(full_id) % num_batches_int
    batches = []
    for i in range(num_batches_int):
        batches.append(data_new[np.where(ids == i)])
    if residue > 0:
        batches.append(data_new[full_id:])
    return batches

def plot_regression_results(data, X, Y, xlim=None, ylim=None):
    x, y = data[:, 0], data[:, 1]
    data_overplot = np.vstack((X.reshape(-1), Y.reshape(-1))).T
    plt.figure(figsize=(7,7))
    plt.plot(x, y, 'o', markersize = 2)
    x_op = data_overplot[:, 0].reshape(-1)
    y_op = data_overplot[:, 1].reshape(-1)
    plt.plot(x_op, y_op, color='tab:red', linewidth=3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y^*$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()
