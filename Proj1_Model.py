class Proj1_model:

    def _init_(self, input_data, targets, method):
        # initialize model attributes
        self.method = method
        self.input = input_data
        self.targets = targets

    def calc_params(self):
        if self.method == 'Linear regression':
            fun = 'tb.least_squares_inv(y,x)'
        elif self.method == "Ridge regression":
            params = 2
        elif self.method == 'Logistic regression':
            params = 3
        elif self.method == 'Regularized logistic regression':
            params = 4
        else:
            # raise error if the classification method is not implemented
            raise ValueError('The specified method: "' + self.method + '" is not \
            implemented for Project 1. Please choose between: Option 1\n Option 2\n \
            Option 3\n' )
        return params

    def train(self):
        return weights

    def predict(self, x_test):
        prediction = np.zeros(shape(self.y))
        return predictions

    def acc(self, x):
        check = self.y==self.predict()
        acc = sum(check) / len(check)
        acc = 2
        return acc

    def tpr(self):
        tp = [i for i in self.y == self.predict() if self.y==True]
        fn = [i for i in self.y != self.predict() if self.predict()==False]
        tpr = len(tp) / (len(tp)+len(fn))
        tpr = 2
        return tpr

    def fpr(self):
        fp = [i for i in self.y!=self.predict() if self.predict()==True]
        tn = [i for i in self.y==self.predict() if self.y==False]
        fpr = len(fp) / (len(fp)+len(tn))
        fpr = 2
        return fpr
