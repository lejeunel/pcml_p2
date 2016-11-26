import numpy as np
import matplotlib.pyplot as plt
import helpers as hp
import helpers_final as H
import sys
import pdb
import implementations as imp

def pca(x,nb_dims):

    cov_mat = np.cov(x.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    for ev in eig_vec_cov:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    #print('Covariance Matrix:\n', cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    nb_reduced_dim = 5
    w_list = list()

    for i in range(nb_reduced_dim):
        w_list.append(eig_pairs[i][1])

    mat_w = np.asarray(w_list)

    x_proj = np.dot(mat_w,x.T).T

    return x_proj, eig_pairs

def mse(y_true,y_estim):
    """
    Computes the mean squared error between two outputs
        y_true (Nx1): Output vector (True values)
        y_estim (Nx1): Output vector (Estimated values)
    Where N is the number of samples
    Out: MSE value
    """

    N = x.shape[0] #Number of samples
    e = y_true - y_estim
    e_squared = e**2
    mse = (1./(2*N))*np.sum(e_squared)

    return mse

def mse_lin(y,x,w):
    """
    Computes the mean squared error of a linear system.
    In: x (DxN): Input matrix
        y (Nx1): Output vector
        w (Dx1): Weight vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: MSE value
    """

    N = x.shape[0] #Number of samples
    e = y - np.dot(x.transpose(),w)
    e_squared = e**2
    mse = (1./(2*N))*np.sum(e_squared)

    return mse

def least_squares_SGD(y,x,gamma,max_iters,B=1,init_guess = None):
    """
    Estimate parameters of linear system using stochastic least squares gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        init_guess (Dx1): Initial guess
        gamma: step_size
        B: batch size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess is None):
        init_guess = np.zeros((x.shape[1],1))


    N = x.shape[0]
    w = list()
    w = init_guess

    for minibatch_y, minibatch_x in hp.batch_iter(y, x, B, num_batches=max_iters, shuffle=True):
        w = w - gamma*comp_ls_gradient(N,minibatch_x,minibatch_y-np.dot(minibatch_x,w))

    return w

def least_squares_inv_ridge(y,phi_tilda,lambda_):
    """
    Estimate parameters of regularized (ridge/L2-norm) system using matrix inversion
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        lambda_: regularization parameter
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    shape_phi = phi_tilda.shape
    N = shape_phi[1]
    lambda_p = lambda_*2*N
    left_term = np.linalg.inv(np.dot(phi_tilda.transpose(),phi_tilda) + lambda_p*np.identity(shape_phi[1]))
    left_term = np.dot(left_term,phi_tilda.transpose())
    w = np.dot(left_term,y)

    return w

def logit(beta,x):
    """
    Computes the probability values 1/(1+exp(-(beta[0] + beta[1:]*x)))
    Note: This function is safe for high values of tx*beta, values are replaced.
    In: x (NxD+1): Input matrix
        beta (D+1 x 1): Parameter vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Probability values (Nx1)
    """

    tbeta = beta.transpose()
    tx = x.transpose()
    the_exp = np.exp(-np.dot(tbeta,tx))
    ind_inf = np.isinf(the_exp)
    res = 1/(1 + np.exp(-np.dot(tbeta,tx)))
    res[ind_inf] = 0.0

    return np.ravel(res)

def comp_grad_logit(beta,x,y, lambda_):
    """
    Computes the gradient of the logistic regression cost function
    In: x (NxD+1): Input matrix
        beta (D+1 x 1): Parameter vector
        y (N x 1): Output vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Gradient (Dx1)
    """

    tx = x.transpose()
    N = tx.shape[1]

    probs = logit(beta,x)
    y.shape = (y.shape[0],1)
    probs.shape = (probs.shape[0],1)
    y_minus_p = y - probs

    grad1 = np.dot(tx,y_minus_p)
    grad = np.zeros((tx.shape[0],1))
    grad[0] = np.dot(tx[0,:], y_minus_p)
    grad[1:] = np.dot(tx[1:,:], y_minus_p) + lambda_/N*beta[1:]
    #pdb.set_trace()
    #print(grad1 - grad)
    return grad

def logit_GD(y,x,gamma,max_iters,init_guess = None):
    """
    Estimate parameters of logistic regression using gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        init_guess (Dx1): Initial guess
        gamma: step_size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess is None):
        init_guess = np.zeros((x.shape[1],1))
    else:
        init_guess.shape = (init_guess.shape[0],1)

    N = x.shape[0]
    w = list()
    w = init_guess

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w = w - gamma*comp_grad_logit(w,x,y)

    return w

def logit_GD_ridge(y,x,gamma,lambda_,max_iters,init_guess = None):
    """
    Estimate parameters of logistic regression using gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        lambda_: regularization parameter
        init_guess (Dx1): Initial guess
        gamma: step_size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess == None):
        init_guess = np.zeros((x.shape[1],1))
    else:
        init_guess.shape = (init_guess.shape[0],1)

    w = init_guess

    N = x.shape[0]

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w = w - gamma*(comp_grad_logit(w,x,y)+ 2*lambda_*w)

    return w

def thr_probs(probs,thr):
    """
    Thresholds the values of probs s.t. probs>=thr gives 1 and probs<thr gives 0
    In: probs (Nx1): Input matrix
        thr: Threshold value
    Out: Classes
    """

    classes = np.zeros(probs.shape)
    ind_sup = np.where(probs >= thr)
    classes[ind_sup] = 1

    return classes

def binary_tpr_fpr(y_true,y_pred):
    """
    Computes the true/false positive rates. y_true must be either +1 or -1
    In: y_true (Nx1): Training values
        y_pred (Nx1): Predicted values
    Out: True/False positives rates
    """

    positives = np.where(y_true == 1)
    negatives = np.where(y_true == -1)
    true_pos = np.where(y_pred[positives[0]] == 1)[0].shape[0]
    false_pos = np.where(y_pred[negatives[0]] == 1)[0].shape[0]
    true_neg = np.where(y_pred[negatives[0]] == -1)[0].shape[0]
    false_neg = np.where(y_pred[positives[0]] == -1)[0].shape[0]

    tpr = float(true_pos)/(float(true_pos) + float(false_neg))
    fpr = float(false_pos)/(float(false_pos) + float(true_neg))

    return tpr,fpr

def missclass_error_rate(y_true,y_pred):
    """
    Computes the missclassification error rate. y_true must be either +1 or -1
    In: y_true (Nx1): Training values
        y_pred (Nx1): Predicted values
    Out: Missclassification error rate
    """

    missclass = np.ones((y_true.shape[0],1))
    missclass[np.where(np.ravel(y_true) == np.ravel(y_pred))[0]] = 0

    miss_rate = np.sum(missclass)/y_true.shape[0]

    return miss_rate

def knn_impute(A,M,K=10,nb_rand_ratio = 0.1):
    """
    Imputes missing values of arrays B and C using k-nearest-neighbors. Arrays are obtained using isolate_missing
    In: A (NxD): Training data (No missing values. Will be used for kNN)
        M (N'xD): Training data with missing values (nan)
        K : Number of nearest neighbors
        nb_rand_ratio: Ratio of clean samples to choose randomly from.
    Out: Matrix of size (N+N'xD) without missing values
    """

    print("Computing k-NN for K=", K, " taking", np.round(nb_rand_ratio*A.shape[0]), " samples")
    for i in range(M.shape[0]):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("%f percent" % (float(i+1)/float(M.shape[0])*100))
        sys.stdout.flush()
        ok_cols = ~np.isnan(M[i,:])
        nan_cols = np.isnan(M[i,:])
        norms = np.array([])
        rand_ind = np.random.randint(0,A.shape[0],np.round(A.shape[0]*nb_rand_ratio))
        for j in range(len(rand_ind)):
            this_norm = np.linalg.norm(A[rand_ind[j],ok_cols]-M[i,ok_cols])
            norms = np.append(norms,this_norm)
        sorted_ind = np.argsort(norms)
        sorted_norms = norms[sorted_ind[0:K]]
        sorted_norms = (sorted_norms/np.sum(sorted_norms)).reshape((K,1))
        this_kNN = A[sorted_ind[0:K],:]
        #Change to weighted mean!!
        mean_kNN = np.mean(this_kNN*sorted_norms,axis=0)
        M[i,nan_cols] = mean_kNN[nan_cols]

    return np.concatenate((A,M),axis=0)


def findOffending(data,offending):
    """
    Looks for values corresponding to offending and outputs matrix of same size as data
    with 1->offending and 0 otherwise
    """
    out = np.zeros(data.shape)
    out[np.where(data == offending)] = 1
    return out

def isolate_missing(x,offend):
    """
    Divide the training data matrix into 3 parts (see below)
    Input:
        x (NxD) input matrix
        offend: offending value to look for
    Output:
        A: All features of all samples are not offending (OK). No features are removed.
        B: Matrix of reduced feature dimension. Has no missing values (but lower "column" dimension)
        C: Most features of most samples are offending.
        a_cols: Column indices of A
        b_cols: Column indices of B
        c_cols: Column indices of C
    """

    offending_rows = np.array([])
    offending_cols = np.array([])
    ok_rows = np.array([])
    ok_cols = np.array([])

    offend_mat = findOffending(x,offend)

    #Replace offending values by NaN
    i_offend, j_offend = np.where(offend_mat)
    x[i_offend,j_offend] = np.nan

    for i in range(x.shape[0]):
        if(np.where(offend_mat[i,:])[0].size > 0): #Found offending values in this row
            offending_rows = np.append(offending_rows,i)
        else:
            ok_rows = np.append(ok_rows,i)

    for i in range(x.shape[1]):
        if(np.where(offend_mat[:,i])[0].size > 0): #Found offending values in this column
            offending_cols = np.append(offending_cols,i)
        else:
            ok_cols = np.append(ok_cols,i)

    a_grid = np.ix_(ok_rows.astype(int),np.arange(0,x.shape[1]))
    b_grid = np.ix_(offending_rows.astype(int),ok_cols.astype(int))
    c_grid = np.ix_(offending_rows.astype(int),offending_cols.astype(int))
    A = x[a_grid]
    B = x[b_grid]
    C = x[c_grid]

    new_rows = np.concatenate((ok_rows.astype(int),offending_rows.astype(int)))
    new_cols = np.concatenate((ok_cols.astype(int),offending_cols.astype(int)))

    return (A,B,C,new_cols,new_rows)

def imputer(x,offend,mode):
    """
    Deal with offending values using following modes:
    'del_row': Deletes rows
    'mean': Replace with mean value of column
    'median': Replace with median value of column
    ."""

    offend_mat = findOffending(x,offend)

    if(mode == 'del_row'):
        ok_rows = np.where(np.sum(offend_mat,axis=1) == 0)
        ok_rows = ok_rows[0]
        clean_x = np.squeeze(x[ok_rows,:])
        return clean_x

    for i in range(x.shape[1]):
        not_ok_rows = np.where(offend_mat[:,i] == 1)
        if(mode == 'mean'):
            this_val = np.mean(x[offend_mat[:,i] == 0,i])
        elif(mode == 'median'):
            this_val = np.median(x[offend_mat[:,i] == 0,i])

        x[not_ok_rows,i] = this_val

    return x

def logistic_regression(y,tx,gamma,max_iters,init_guess = None):
    """
    Estimate parameters of logistic regression using gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        init_guess (Dx1): Initial guess
        gamma: step_size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """
    x=tx.transpose()
    if(init_guess == None):
        init_guess = np.zeros((tx.shape[0],1))
    else:
        init_guess.shape = (init_guess.shape[0],1)

    N = tx.shape[1]
    w = list()
    w.append(init_guess)

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w.append(w[-1] - gamma*comp_grad_logit(w[-1],x,y, 0))

    return w[-1]

#from sklearn.linear_model import SGDClassifier
def cross_validation(x,y,k, mode, gamma=None, lambda_=None, max_iters=None, initial_w=None):
    """
    INPUT:
    @x : input data, dimensions (NxD)
    @y : target labels, (Nx1) array
    @k : number of folds
    OUTPUT:
    """
    D = x.shape[1]
    #randomly permute data maybe?
    x_split = np.array_split(x, k, axis=0)
    y_split = np.array_split(y, k, axis=0)
    #initialize weights and metrics
    weights = list()
    acc = list()
    tpr = list()
    fpr = list()
    losses = list()

    #loop over folds
    for fold in range(k):
        #create model
        #train_ind = [i for i in range(k) if i!=fold]
        #val_ind = [i for i in range(k) if i==fold]
        #pdb.set_trace()
        x_train = [x_split[i] for i in range(k) if i!=fold]
        y_train = [y_split[i] for i in range(k) if i!=fold]
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = x_split[fold]
        y_val = y_split[fold]
        #model = Proj1_Model(x_train, y_train, mode)
        #train model for fold
        #weights[k] = model.train()
        """here the choice of method"""
        if mode == 'linear_regression_eq':
            update, loss = imp.least_squares(y_train, x_train)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'ridge_regression_eq':
            update, loss = imp.ridge_regression(y_train, x_train, lambda_)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'linear_regression_GD':
            update, loss = imp.least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'linear_regression_SGD':
            update, loss = imp.least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            pr_bool = predictions>=np.mean(predictions)
        elif mode == 'logistic_regression':
            update, loss = imp.logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            predicted_prob = H.sigmoid(predictions)
            #pdb.set_trace()
            pr_bool = predicted_prob>0.5
        elif mode == 'reg_logistic_regression':
            update, loss = imp.reg_logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
            predictions = np.dot(x_val, update)
            predicted_prob = H.sigmoid(predictions)
            #pdb.set_trace()
            pr_bool = predicted_prob>0.5
        weights.append(update)
        losses.append(loss)
        pr_bool = predictions>=np.mean(predictions)
        y_bool = y_val==1
        correct = pr_bool == y_bool
        tp = np.logical_and(correct,y_bool)
        fp = np.logical_and(np.logical_not(correct), pr_bool)
        #tp = [i for i in range(len(pr_bool)) if (pr_bool[i] == True and y_bool[i] == True)]
        #all_p = [i for i in range(len(pr_bool)) if y_bool == True]
        #fp = [i for i in range(len(pr_bool)) if (pr_bool == True and y_bool == False)]
        #all_n = [i for i in range(len(pr_bool)) if y_bool == False]
        #print('True signal samples:' + str(sum(y_val)) + ' - Predicted signal samples:' + str(sum(pr_bool)))
        acc.append(sum(correct)/float(len(y_val)))
        tpr.append(sum(tp)/float(sum(y_bool)))
        fpr.append(sum(fp)/float(sum(np.logical_not(y_bool))))
        #acc[k] = model.acc()
        #tpr[k] = model.tpr()
        #fpr[k] = model.fpr()
    return acc, tpr, fpr, losses
