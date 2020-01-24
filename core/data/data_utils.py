import numpy as np
import matplotlib.pyplot as plt

def f_reg(N=1000, max=1, min=-1, seed=123):
    assert max > min, "Error in f_reg(N, max, min, seed): The max argument must be greater than the min argument."
    np.random.seed(seed)
    x = np.random.random(N)*(max - min) + min
    y = np.square(x) + np.exp(x/2) + np.random.normal(scale=0.07, size=N)
    return x, y

def f_class(N=1000, max=1, min=0, seed=123):
    assert max > min, "Error in f_clas(N, max, min, seed): The max argument must be greater than the min argument."
    np.random.seed(seed)
    x1 = np.random.random(N)*(max - min) + min
    x2 = np.random.random(N)*(max - min) + min
    lim = fun_class(x1)
    y = (x2 > lim).astype(float)
    return x1, x2, y

def fun_class(x):
    return 1 - np.exp(-x) - 0.5*x**3 + 0.25

def generate_data_for_regression(plot=True, seed=123, data_overplot=None):
    x, y = f_reg(seed=seed)
    data = np.vstack((x, y)).T
    if(plot):
        plt.figure()
        plt.plot(x, y, 'o', markersize = 2)
        if(data_overplot is not None):
            x_op = data_overplot[:, 0].reshape(-1)
            y_op = data_overplot[:, 1].reshape(-1)
            plt.plot(x_op, y_op, color='tab:red')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y^*$')
        plt.xlim([-1,  1])
        plt.ylim([0.5, 3])
        plt.show()
    return data

def generate_data_for_classifier(plot=True, plot_line = False, seed=123):
    x1, x2, y = f_class(seed=seed)
    data = np.vstack((x1, x2, y)).T
    if(plot):
        c11 = x1[np.where(y==0)]
        c12 = x2[np.where(y==0)]
        c21 = x1[np.where(y==1)]
        c22 = x2[np.where(y==1)]

        plt.figure()
        plt.plot(c11, c12, 'o', markersize=4, color='tab:blue')
        plt.plot(c21, c22, 'o', markersize=4, color='tab:red')
        if(plot_line):
            x = np.linspace(0, 1, 100)
            plt.plot(x, fun_class(x), color='k', linewidth=2)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.xlim([0,  1])
        plt.ylim([0,  1])
        plt.show()
    return data

def generate_training_batches(data, batch_size=32):
    # data should be an array of shape (N, d+1)
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
