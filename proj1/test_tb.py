import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers_final as hp
import pdb

#initialize parameters
gamma = 0.001 #0.000001
#lambda_ = np.arange(0,100)
lambda_ = [0]
l = lambda_[0]
max_iters = 1000
modes = ['linear_regression_eq', 'ridge_regression_eq', 'linear_regression_GD', 'linear_regression_SGD', 'logistic_regression', 'reg_logistic_regression']
mode = modes[4]


nb_samples = 100
sig_noise = 0.5
x = np.linspace(0,1,nb_samples)
x.shape = (1,nb_samples)
x = np.append(np.ones((1,100)),x,axis=0)
x = x.transpose()
initial_w = np.zeros((x.shape[1],1))
w = np.array([3,3])
w.shape = (2,1)
y_true = np.dot(x,w)
noise = np.random.normal(0,sig_noise,nb_samples)
noise.shape = (nb_samples,1)
y = y_true + noise
y.shape = (nb_samples,1)

#mse = tb.mse_lin(y,x,w)
pdb.set_trace()
acc,tpr,fpr, losses = tb.cross_validation(x,y,10, mode=mode, gamma=gamma, lambda_=l, max_iters = max_iters, initial_w=initial_w)
pdb.set_trace()
print("Gradient descent: True vs. estimate")
print(w)
print(w_estimate)

w_estimate = tb.least_squares_SGD(y,x.transpose(),2,30000,B=nb_samples/4)
print("Stochastic gradient descent: True vs. estimate")
print(w)
print(w_estimate)

w_estimate = tb.least_squares_inv(y,x.transpose())
print("Closed-form (normal) equations: True vs. estimate")
print(w)
print(w_estimate)

w_estimate = tb.least_squares_inv_ridge(y,x.transpose(),lambda_)
print("Closed-form (normal) equations regularized: True vs. estimate")
print(w)
print(w_estimate)

#Test logistic regression (two gaussian distrib. with different mean and variance)

mean_0 = 1.80
std_0 = np.sqrt(0.20)

mean_1 = 1.0
std_1 = np.sqrt(0.15)

N = 100
x0 = np.random.normal(loc=mean_0, scale=std_0, size=N)
x1 = np.random.normal(loc=mean_1, scale=std_1, size=N)
#plt.hist(x0,10); plt.show()
#plt.hist(x1,10); plt.show()

x = np.concatenate((x0,x1),axis=0)
x,mean_x,std_x = hp.standardize(x)
x.shape = (2*N,1)
x_aug = np.concatenate((np.ones((2*N,1)),x),axis=1)
y = np.concatenate((np.zeros((N,1)),np.ones((N,1))))

beta_init = np.ones((x_aug.shape[1]))

beta_estim = tb.logit_GD(y,x_aug,0.01,max_iters=1000,init_guess = beta_init)

#Compute probabilities using estimated beta
probs = tb.comp_p_x_beta_logit(beta_estim,x_aug)
plt.stem(probs);
plt.xlabel('Samples')
plt.ylabel('probabilities')
plt.show()

#Estimated classification
cla = tb.thr_probs(probs,0.5)
