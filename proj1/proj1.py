import pdb
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp

#initialize parameters
gamma = 0.000001 #0.000001
lambda_ = np.arange(0,100, 10)
#lambda_ = [0]
max_iters = 100
modes = ['linear_regression_eq', 'ridge_regression_eq', 'linear_regression_GD', 'linear_regression_SGD', 'logistic_regression', 'reg_logistic_regression']
mode = modes[5]

#data_train = hp.load_data('../train.csv')[:,2:]
#y = hp.load_data('../train.csv')[:,1:2]
data,y = imp.load_data_higgs('../data/train.csv')
clear_cols = [i for i in range(data.shape[1]) if -999 not in data[:,i]]
clear_rows = [i for i in range(data.shape[0]) if -999 not in data[i,:]]
#x = data[clear_rows,:]
#x = data[:, clear_cols]
#x = imp.impute_lr(data)
#x = imp.build_poly(data, 30)
x, mean_x, std_x = imp.standardize(data)
#A,B,C,new_cols,new_rows = tb.isolate_missing(data,-999)
#x_knn = knn_impute(A,M,K=10,nb_rand_ratio = 0.1)
x_aug = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
initial_w = np.zeros((x.shape[1],1))

signal = [i for i in range(y.shape[0]) if y[i] == 1]
bkg = [i for i in range(y.shape[0]) if y[i] == 0]
"""
for i in range(data.shape[1]):
    #THERE IS NO SINGLE DISCRIMINATIVE FEATURE
    plt.scatter(data[signal,i], y[signal], c='r', marker='x')
    plt.hold(True)
    plt.scatter(data[bkg,i], y[bkg], c='b', marker='o')
    plt.show()
    pdb.set_trace()
"""
pdb.set_trace()  ######### Break Point ###########
"""
we,lo = imp.logistic_regression(y, x, initial_w, max_iters, gamma)
predictions = np.dot(x, we)
predicted_prob = imp.sigmoid(predictions)
pr_bool = predicted_prob>0.5
y_bool = y==1
y_bool = y_bool.reshape(y_bool.shape[0],1)
correct = pr_bool == y_bool
print(sum(correct)/float(len(y_bool)))
pdb.set_trace()  ######### Break Point ###########
"""
mean_acc = list()
#mean_tpr = list()
#mean_fpr = list()
mean_losses = list()
for l in lambda_:
    acc, losses, weights = imp.cross_validation(x,y,10, mode=mode, gamma=gamma, lambda_=l, max_iters = max_iters, initial_w=initial_w)
    mean_acc.append(np.mean(acc))
    #mean_tpr.append(np.mean(tpr))
    #mean_fpr.append(np.mean(fpr))
    losses = np.array(losses)
    mean_losses.append(np.mean(losses, axis=0))
pdb.set_trace()  ######### Break Point ###########
# save variables
accFile = 'mean_acc_' + mode
#tprFile = 'mean_tpr_' + mode
#fprFile = 'mean_fpr_' + mode
lossesFile = 'mean_losses_' + mode
np.save(accFile, mean_acc)
#np.save(tprFile, mean_tpr)
#np.save(fprFile, mean_fpr)
np.save(lossesFile, mean_losses)
