import numpy as np
import sys
import pdb

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Estimate parameters of linear regression using gradient descent
    INPUTS
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    @initial_w (Dx1): Inintial values of the weights
    @max_iters: the number of epochs
    @gamma: learning rate of the gradient descent algorithm
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @w: Optimal weights, array (Dx1)
    @mse: MSE
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    we = initial_w
    for n_iter in range(max_iters):
        # calculate gradient and losses
        grad = compute_gradient(y,tx,we)
        loss = compute_loss(y,tx,we)
        # update weights
        we = we - gamma*grad
        # store w and loss
        ws.append(np.copy(we))
        losses.append(loss)
    # return the last weight array and losses
    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Estimate parameters of linear regression using stochastic gradient descent
    INPUTS
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    @initial_w (Dx1): Inintial values of the weights
    @max_iters: the number of epochs (== the number of samples to see)
    @gamma: learning rate of the gradient descent algorithm
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @w: Optimal weights, array (Dx1)
    @mse: MSE
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    we = initial_w
    n_iter = 0
    for mini_y, mini_x in batch_iter(y, tx, 1, num_batches=max_iters, shuffle=True):
        # calculate gradient and losses
        grad = compute_gradient(mini_y, mini_x,we)
        loss = compute_loss(mini_y, mini_x,we)
        # update weights
        we = we - gamma*grad
        # store w and loss
        ws.append(np.copy(we))
        losses.append(loss)
        n_iter += 1
    # return the last weight array and losses
    return ws[-1], losses[-1]

def least_squares(y,tx):
    """
    Estimate parameters of linear system using normal equations for least squares regression
    INPUTS
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @w: Optimal weights, array (Dx1)
    @mserror: MSE

    Ref: https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Derivation_of_the_normal_equations
    """

    #(tx*x)^(-1)*tx
    factor = np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)),tx.transpose())
    we = np.dot(factor,y)
    predictions = np.dot(tx,we)
    mserror = mse(y, predictions)
    return we, mserror

def ridge_regression(y, tx, lambda_):
    """
    Estimate parameters of linear system using normal equations for ridge regression
    INPUTS
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @w: Optimal weights, array (Mx1)
    @mse: MSE
    """
    N = y.shape[0]
    phi_tilda = tx
    M = phi_tilda.shape[1]
    factor1 = np.linalg.inv(np.dot(phi_tilda.transpose(), phi_tilda) + 2*N*lambda_*np.identity(M))
    factor2 = np.dot(factor1, phi_tilda.transpose())
    we = np.dot(factor2,y)
    predictions = np.dot(tx,we)
    mse = mse(y, predictions)
    return we, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Estimate parameters of linear system using logistic regression
    INPUTS
    @tx ((DxN): input matrix
    @y (Nx1): Output vector
    @gamma: learning rate
    @max_iters: number of epochs
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @w: Optimal weights, array (Dx1)
    @mse: MSE
    """
    ws = [initial_w]
    losses = []
    we = initial_w
    for n_iter in range(max_iters):
        # calculate gradient and losses
        grad = compute_logistic_gradient(y,tx,we)
        loss = compute_logistic_loss(y,tx,we)
        # update weights
        we = we - gamma*grad
        # store w and loss
        ws.append(np.copy(we))
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): logistic loss={l}, w={w}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w=we))a
    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Estimate parameters of linear system using regularized logistic regression
    INPUTS
    @tx ((DxN): input matrix
    @y (Nx1): Output vector
    @lambda_: regularization parameter
    @gamma: learning rate
    @max_iters: number of epochs
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @w: Optimal weights, array (Dx1)
    @mse: MSE
    """
    ws = [initial_w]
    losses = []
    we = initial_w
    for n_iter in range(max_iters):
        # calculate gradient and losses
        grad = compute_logistic_gradient(y,tx,we)
        loss = compute_logistic_loss(y,tx,we)
        # calculate the regularization factor (L2 norm regularization)
        penalty = lambda_ * sum(we**2)
        we = we - gamma*grad - penalty
        # store w and loss
        ws.append(np.copy(we))
        losses.append(loss)

    return ws[-1], losses[-1]

"""---------------------------------HELPERS------------------------------------------------"""
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
    """
    Standardize the original data set.
    """
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
    grad =  -np.dot(tx.transpose(),e)/float(N)
    return grad

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.
    INPUTS
    @x (NxD) : vector of input data
    @degree : degree of the polynomial basis
    OUTPUTS
    @phi_tilda (N x D): polynomial matrix
    """
    D = x.shape[1]
    phi_tilda = np.zeros(x.shape)
    for j in range(D):
        feature = x[:,j]
        phi_x = 0
        for i in range(degree):
            phi_x += np.power(feature,degree)

    return phi_tilda


def sigmoid(t):
    """
    apply sigmoid function on t.
    """
    sigma_ = np.exp(t) / (1 + np.exp(t))
    return sigma_

def compute_logistic_loss(y, tx, ww):
    """
    compute the cost by negative log likelihood. (lecture 5b)
    @y (Nx1): Output vector, y = 1 for signal and 0 for background
    @tx (NxD): Input matrix
    @w w(Dx1) : weights vector
    Where N and D are respectively the number of samples and dimension of input vectors
    OUTPUTS
    @loss: logistic loss
    """
    N = y.shape[0]
    loss = 0
    for n in range(N):
        # sample shape is (D,)
        sample = tx[n,:]
        addend = np.log(1+np.exp(np.dot(sample.transpose(), ww))) - y[n]*np.dot(sample.transpose(), ww)
        loss += addend

    return loss

def compute_logistic_gradient(y, tx, we):
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
    prediction = sigmoid(np.dot(tx,we))
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
    #clear_samples, mean_x, std_x = standardize(clear_samples)
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

def cross_validation(x,y,k, mode, gamma=None, lambda_=None, max_iters=None, initial_w=None):
    """
    INPUT:
    @x : input data, dimensions (NxD)
    @y : target labels, (Nx1) array
    @k : number of folds
    @gamma: learning rate in case of gradient descent models
    @lambda_: regularized term in case of regularized versions of algorithms
    @max_iters: number of epochs in case of iterative models
    @initial_w (Dx1): initial weigths for iterative models
    OUTPUT:
    @acc: (10x1) the accuracy of prediction on the validation dataset of every fold
    @losses: mse
    @weights: the calculated weigths of every fold
    """
    # data dimensions
    D = x.shape[1]
    # split the data into k groups
    x_split = np.array_split(x, k, axis=0)
    y_split = np.array_split(y, k, axis=0)
    #initialize weights and metrics
    weights = list()
    acc = list()
    losses = list()

    #loop over folds
    for fold in range(k):
        # divide the data into the training set of (k-1) groups, and the validation set of 1 group
        x_train = [x_split[i] for i in range(k) if i!=fold]
        y_train = [y_split[i] for i in range(k) if i!=fold]
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = x_split[fold]
        y_val = y_split[fold]
        # choose the classification method
        if mode == 'linear_regression_eq':
            update, loss = least_squares(y_train, x_train)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'ridge_regression_eq':
            update, loss = ridge_regression(y_train, x_train, lambda_)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'linear_regression_GD':
            update, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'linear_regression_SGD':
            update, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'logistic_regression':
            update, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            predicted_prob = sigmoid(predictions)
            pr_bool = predicted_prob>0.5
        elif mode == 'reg_logistic_regression':
            update, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            predicted_prob = sigmoid(predictions)
            pr_bool = predicted_prob>0.5
        else:
            raise ValueError(mode + ' mode of classification is not defined')
        weights.append(update)
        losses.append(loss)
        # transform the targets into boolean arrays
        y_bool = y_val==1
        y_bool = y_bool.reshape(y_bool.shape[0],1)
        correct = pr_bool == y_bool
        # calculate the accuracy as the ratio of correctly classified samples over the total number of samples
        acc.append(sum(correct)/float(len(y_val)))
    return acc, losses, weights
