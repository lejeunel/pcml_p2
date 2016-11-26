import numpy as np
import matplotlib.pyplot as plt
import helpers as hp
import sys
import pdb
import implementations as imp

def load_data_higgs(path_dataset):
    """Load data and convert it to the metrics system."""
    col_pred = 1
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)

    #id_ = data[:,0]

    data = np.delete(data,col_pred,axis=1)
    data = np.delete(data,0,axis=1)

    #Read character of class 's' or 'b'
    y = np.genfromtxt(
        path_dataset, delimiter=",",dtype="str", skip_header=1,usecols=col_pred)
    y_out = np.zeros(y.shape)
    s_ind = np.where(y == 's')
    b_ind = np.where(y == 'b')
    y_out[s_ind] = 1
    #y_out[b_ind] = -1
    y_out[b_ind] = 0


    return data,y_out #,id_

def write_submission_higgs(y,id_,path):

    y.shape = (y.shape[0],1)
    id_.shape = (id_.shape[0],1)
    header = "Id,Prediction"
    np.savetxt(path,np.concatenate((id_,y),axis=1),fmt='%d',delimiter=',',header=header,comments='')

    return True

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mse(y_true,y_estim):
    """
    Computes the mean squared error between two outputs
    INPUTS
    @y_true (Nx1): Output vector (True values)
    @y_estim (Nx1): Output vector (Estimated values)
    Where N is the number of samples
    OUTPUT
    @ mse: MSE value
    """

    N = y_true.shape[0] #Number of samples
    y_true = y_true.reshape(N,1)
    y_estim = y_estim.reshape(N,1)
    e = y_true - y_estim
    e_squared = e**2
    mse = (1/float(2*N))*np.sum(e_squared)

    return mse

def rmse(y_true, y_estim):
    rmse = np.sqrt(2*mse(y_true, y_estim))
    return rmse

def compute_loss(y, tx, we):
    """
    this function computes the loss of the estimation as the mse
    INPUTS
    @y (Nx1): Output vector (True values)
    @tx (NxD): Input vector
    @w (Dx1): weights
    Where N is the number of samples
    OUTPUT
    @loss: mse
    """
    #pdb.set_trace()
    loss = mse(y,np.dot(tx,we))
    return loss

def compute_gradient(y, tx, we):
    """
    INPUTS
    @y (Nx1): Output vector (True values)
    @tx (NxD): Input vector
    @w (Dx1): weights
    Where N is the number of samples
    OUTPUTS
    @grad (Dx1): gradient for each dimension
    """
    N = y.shape[0]
    y = y.reshape(N,1)
    e = y - np.dot(tx,we)
    grad =  -np.dot(tx.transpose(),e)/float()
    print('MSE = ' + str(mse(y, np.dot(tx,we))) + ' pred = ' + str(sum(np.dot(tx,we))) + ' grad = ' + str(grad) + ' N = ' + str(N))
    return grad

def comp_ls_gradient(N,x,e):
    grad = (-1./N)*np.dot(x.transpose(),e)
    print('grad = ' + str(grad))
    return grad

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.
    INPUTS
    @x (NxD) : vector of input data
    @degree : degree of the polynomial basis
    OUTPUTS
    @phi_tilda (N x degree): matrix
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    N = x.shape[0]
    phi_tilda = np.zeros((N,degree))
    for j in range(N):
        sample = x[j,:]
        for i in range(degree):
            phi_x = np.sum(np.power(sample, i))
            phi_tilda[j,i] = phi_x
    return phi_tilda


def sigmoid(t):
    """
    apply sigmoid function on t.
    """
    sigma_ = np.exp(t) / (1 + np.exp(t))
    return sigma_

def compute_logistic_loss(y, tx, w):
    """
    compute the cost by negative log likelihood. (lecture 5b)
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    @w (Dx1) : weights vector
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @loss: logistic loss
    """
    N = y.shape[0]
    loss = 0
    for n in range(N):
        # sample shape is (D,)
        sample = tx[n,:]
        addend = np.log(1+np.exp(np.dot(sample.transpose(), w))) - y[n]*np.dot(sample.transpose(), w)
        loss += addend

    return loss

def compute_logistic_gradient(y, tx, w):
    """
    compute the gradient of logistic loss. (lecture 5b)
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    @w (Dx1) : weights vector
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @grad: logistic gradient
    """
    y = y.reshape(y.shape[0],1)
    prediction = sigmoid(np.dot(tx,w))
    print('logistic regression prediction probability = ' + str(prediction))
    right_term = prediction - y
    grad = np.dot(tx.transpose(), right_term)
    return grad

def impute_lr(data):
    #find columns that have no -999
    clear_cols = [i for i in range(data.shape[1]) if -999 not in data[:,i]]
    #find rows that have no -999
    clear_rows = [i for i in range(data.shape[0]) if -999 not in data[i,:]]
    #pdb.set_trace()
    dirty_cols = [i for i in range(data.shape[1]) if i not in clear_cols]
    dirty_rows = [i for i in range(data.shape[0]) if i not in clear_rows]

    clear_samples = np.copy(data[clear_rows, :])
    #clear_samples, mean_x, std_x = hp.standardize(clear_samples)
    w_lr = list()
    mse= list()
    #pdb.set_trace()
    for feature in dirty_cols:
        wf = imp.least_squares(clear_samples[:, feature], clear_samples[:, clear_cols])
        w_lr.append(wf[0])
        #pdb.set_trace()
        #mse.append(compute_loss(clear_samples[:, feature], clear_samples[:, clear_cols] ,wf[0]))
        for sample in dirty_rows:
            if data[sample,feature] == -999:
                replacement = np.dot(data[sample, clear_cols].transpose(), wf[0])
                data[sample, feature] = replacement

    return data
