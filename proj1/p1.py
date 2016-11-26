import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp
import boosting as bst

#data_train = hp.load_data('../train.csv')[:,2:]
data_train,y_train,id_train = hp.load_data_higgs('../train.csv')
data_test,_,id_test = hp.load_data_higgs('../test.csv')

N = data.shape[0]

data_knn = np.load('train_kNN.npz')
x_knn = data_knn['x_knn']
y_A = data_knn['y_A']

x_mean_imp = tb.imputer(data,-999,'mean')
x_median_imp = tb.imputer(data,-999,'median')

x_A,x_B,x_C,new_cols,new_rows = tb.isolate_missing(data_train,-999)
x_A_test,x_B_test,x_C_test,_,_ = tb.isolate_missing(data_test,-999)

#Rearrange columns to align with features of training
x_test = data_test[:,new_cols]

x_knn =  tb.knn_impute(x_A,np.concatenate((x_B,x_C),axis=1),K=10,nb_rand_ratio=0.01)
x_knn_test =  tb.knn_impute(x_A_test,np.concatenate((x_B_test,x_C_test),axis=1),K=10,nb_rand_ratio=0.01)

x_good_cols = np.concatenate((x_A[:,0:x_B.shape[1]],x_B),axis=0)

print("Study of conditional probabilities on missing values")
x_bad_rows = np.concatenate((x_B,x_C),axis=1)
y_bad_rows = y[new_rows[x_A.shape[0]:]]
x_good_rows = x_A
y_good_rows = y[new_rows[0:x_A.shape[0]]]

p_miss_1 = y_bad_rows.shape[0]/y.shape[0]
p_miss_0 = y_good_rows.shape[0]/y.shape[0]

p_y1_miss_1 = (np.where(y_bad_rows == 1)[0].shape[0]/y.shape[0])/(p_miss_1)
p_y0_miss_1 = (np.where(y_bad_rows == -1)[0].shape[0]/y.shape[0])/(p_miss_1)
p_y0_miss_0 = (np.where(y_good_rows == -1)[0].shape[0]/y.shape[0])/(p_miss_0)
p_y1_miss_0 = (np.where(y_good_rows == 1)[0].shape[0]/y.shape[0])/(p_miss_0)

print("P(Y=1|X has missing attr) = ", p_y1_miss_1)
print("P(Y=-1|X has missing attr) = ", p_y0_miss_1)
print("P(Y=1|X has no missing attr) = ", p_y1_miss_0)
print("P(Y=-1|X has no missing attr) = ", p_y0_miss_0)

y_knn = y[new_rows] #Rearrange y
y_A = y_A[0:x_A.shape[0]]


x_miss = np.concatenate((np.zeros((y_good_rows.shape[0],1)),np.ones((y_bad_rows.shape[0],1))),axis=0)
x_knn_aug = np.concatenate((x_miss, x_knn),axis=1)
x_good_cols_aug = np.concatenate((x_miss,x_good_cols),axis=1)

x_pca,eig_pairs = pca(x_A,nb_dims)

w_estim = tb.least_squares_inv_ridge(y_train,x_proj,1)
w_estim = tb.least_squares_SGD(y_train,x_proj,0.00001,500,B=100,init_guess=w_estim)

z = np.dot(x_proj,w_estim)
y_tilda = np.sign(z)

tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

nb_iters = 120

#PCA
F = bst.train_adaboost(y_A,x_proj,nb_iters)
y_tilda =  bst.predict(F,x_proj)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
error_rate = tb.missclass_error_rate(y_A,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

#Missing values replaced with mean
F = bst.train_adaboost(y,x_median_imp,nb_iters)
y_tilda =  bst.predict(F,x_median_imp)
tpr,fpr = tb.binary_tpr_fpr(y,y_tilda)
error_rate = tb.missclass_error_rate(y,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

#Missing values replaced with mean
F = bst.train_adaboost(y,x_mean_imp,nb_iters)
y_tilda =  bst.predict(F,x_mean_imp)
tpr,fpr = tb.binary_tpr_fpr(y,y_tilda)
error_rate = tb.missclass_error_rate(y,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

#Missing values replaced with K-Nearest-Neighbors
F = bst.train_adaboost(y_knn,x_knn,nb_iters)
y_tilda =  bst.predict(F,x_knn)
tpr,fpr = tb.binary_tpr_fpr(y_knn,y_tilda)
error_rate = tb.missclass_error_rate(y_knn,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)
y_pred_test = bst.predict(F,x_test)
hp.write_submission_higgs(y_pred_test,id_test,"../submission.csv")

#Missing values replaced with K-Nearest-Neighbors, x_miss attribute added
F = bst.train_adaboost(y_knn,x_knn_aug,nb_iters)
f = [F[i]['stump']['feat'] for i in range(len(F))]
y_tilda =  bst.predict(F,x_knn_aug)
tpr,fpr = tb.binary_tpr_fpr(y_knn,y_tilda)
error_rate = tb.missclass_error_rate(y_knn,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)


#Only "good" samples (without missing values)
y_clean = y_A[0:x_A.shape[0]]
F = bst.train_adaboost(y_clean,x_A,nb_iters)
y_tilda =  bst.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_clean,y_tilda)
error_rate = tb.missclass_error_rate(y_clean,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

y_pred_test = bst.predict(F,x_test)
hp.write_submission_higgs(y_pred_test,id_test,"../submission.csv")

#Only "good" columns (without missing values)
F = bst.train_adaboost(y_knn,x_good_cols,nb_iters)
y_tilda =  bst.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
error_rate = tb.missclass_error_rate(y_A,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

#Only "good" columns (without missing values), x_miss attribute added
nb_iters = 120
F = bst.train_adaboost(y_knn,x_good_cols_aug,nb_iters)
f = [F[i]['stump']['feat'] for i in range(len(F))]
y_tilda =  bst.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
error_rate = tb.missclass_error_rate(y_A,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)
