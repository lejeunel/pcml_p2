# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def write_submission_higgs(y,id_,path):

    y.shape = (y.shape[0],1)
    id_.shape = (id_.shape[0],1)
    header = "Id,Prediction"
    np.savetxt(path,np.concatenate((id_,y),axis=1),fmt='%d',delimiter=',',header=header,comments='')

    return True

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
        B: All features of all samples are not offending. Some features are removed
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
    a_cols = np.arange(0,x.shape[1])
    b_cols = np.arange(0,B.shape[1])
    c_cols = np.arange(B.shape[1]+1,x.shape[1])

    return (A,B,C,a_cols,b_cols,c_cols,ok_rows.astype(int))

def imputer(x,offend,mode):
    """Deal with offending values using following modes:
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

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    std_x = np.std(x)
    x = (x - mean_x) / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


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
