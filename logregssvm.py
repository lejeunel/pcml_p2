import numpy as np
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM
from sklearn.base import BaseEstimator
from sklearn import (linear_model,svm,preprocessing,metrics)

class LogregSSVM(BaseEstimator):
    """An example of classifier"""

    def __init__(self, C_logreg=1., C_ssvm=1., inference='qpbo',inference_cache=50,tol=1.,max_iter=200,n_jobs=1):
        """
        Called when initializing the classifier
        """
        self.C_logreg = C_logreg
        self.C_ssvm = C_ssvm
        self.inference = inference
        self.inference_cache = inference_cache
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs

        self.logreg = linear_model.LogisticRegression(C=C_logreg)
        self.crf = EdgeFeatureGraphCRF(inference_method=inference)
        self.ssvm = OneSlackSSVM(self.crf, inference_cache=inference_cache, C=C_ssvm, tol=tol, max_iter=max_iter, n_jobs=n_jobs)

    def fit(self, X, Y=None):
        """
        Each training sample for X is a tuple (node_features, edges, edge_features), where node_features is a numpy array of node-features (of shape (n_nodes, n_node_features)), edges is a array of edges between nodes, of shape (n_edges, 2) as in GraphCRF, and edge_features is a feature for each edge, given as a numpy array of shape (n_edges, n_edge_features).
        """

        #Reshape features and ground truths for logreg
        X_feats_lin =  np.asarray([X[i][0][j] for i in range(len(X)) for j in range(len(X[i][0]))])
        Y_lin =  np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])

        self.logreg.fit(X_feats_lin, Y_lin)

        #Reshape features and ground truths for ssvm
        X_crf = [(self.logreg.predict_proba(X[i][0]), np.asarray(X[i][1]), np.asarray(X[i][2]).reshape(-1,1)) for i in range(len(X))]

        self.ssvm.fit(X_crf, Y)

        return self

    def predict(self, X):
        """
        Each training sample for X is a tuple (node_features, edges, edge_features), where node_features is a numpy array of node-features (of shape (n_nodes, n_node_features)), edges is a array of edges between nodes, of shape (n_edges, 2) as in GraphCRF, and edge_features is a feature for each edge, given as a numpy array of shape (n_edges, n_edge_features).
        """

        X_crf = [(self.logreg.predict_proba(X[i][0]), np.asarray(X[i][1]), np.asarray(X[i][2]).reshape(-1,1)) for i in range(len(X))]

        Z_crf = self.ssvm.predict(X_crf)

        return Z_crf

    def score(self, X, Y):
        """
        Returns the F1-score
        """

        Z = self.predict(X)
        Z_lin =  np.asarray([Z[i][j] for i in range(len(Z)) for j in range(len(Z[i]))])
        Y_lin = np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])
        #print(metrics.f1_score(Y_lin, Z_lin, average='weighted'))
        return metrics.f1_score(Y_lin, Z_lin, average='weighted')
